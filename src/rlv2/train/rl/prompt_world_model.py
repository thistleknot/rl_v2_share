"""src/rlv2/train/rl/prompt_world_model.py — ensemble world model of the prompt->judge environment.

Notes: docs/dispositions.md "WORLD-MODEL REWARD ROUTING, rl_k" (2026-09-23); concepts: Model-Free vs
Model-Based and Nonlinear Learned Dynamics (docs/INSPIRATIONS.md).
Task: playbook.md "WORLD MODEL" W1; "SURPASS 11" S11-B1/B2. Source frame: Dyna-Q with an ensemble world model, mapped onto this rig: our environment is ONE-STEP
(prompt -> judge verdict), so the WorldModel is reward-only, and the Dyna-Q
EnsembleWorldModel becomes K bootstrap-resampled linear probes over frozen
nomic embeddings (labels number in the hundreds; token-level nets would be
data-starved — playbook-uncertainty-menu.md E9 addendum).

Ensemble variance is MODEL DISAGREEMENT, not a calibrated failure probability —
part 6's own caveat, kept verbatim. Predictions are SELECTION-ONLY: they steer
which prompts get real rollouts (informed PER priors, mgpo.py W2); no gradient
ever flows from this model, which is what keeps the reward-hacking caveat that
parked E3 inapplicable here.

Leakage tripwire: the probe must NEVER train on the 16 frozen eval questions —
that would launder eval difficulty into curriculum selection. build_labels
FAILS CLOSED on any overlap.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys

import numpy as np

EVAL_FILE = "data/rl/r3_eval.jsonl"
LABELS_FILE = "data/prompt_labels.jsonl"
MODEL_FILE = "data/prompt_world_model.npz"
K_ENSEMBLE = 5
HIDDEN = 64
SEED = 7


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def build_labels(ledger_path: str | None = None,
                 labels_path: str = LABELS_FILE,
                 eval_path: str = EVAL_FILE) -> list[dict]:
    """Harvest (prompt, w, l) rows from the visit ledger and the labeling sweep.

    Require: every row carries a non-empty prompt and w+l >= 1.
    Guarantee: ZERO overlap with the frozen eval's questions — raises otherwise.
    """
    rows: dict[str, dict] = {}
    if labels_path and os.path.exists(labels_path):
        for line in io.open(labels_path, encoding="utf-8"):
            r = json.loads(line)
            rows[_key(r["prompt"])] = {"prompt": r["prompt"],
                                       "w": int(r["w"]), "l": int(r["l"])}
    if ledger_path and os.path.exists(ledger_path):
        for k, e in json.load(open(ledger_path, encoding="utf-8")).items():
            if not isinstance(e, dict) or not e.get("prompt"):
                continue                       # legacy bare-count entry: no label value
            cur = rows.setdefault(_key(e["prompt"]),
                                  {"prompt": e["prompt"], "w": 0, "l": 0})
            cur["w"] += int(e.get("w", 0))
            cur["l"] += int(e.get("l", 0))
    out = [r for r in rows.values() if (r["w"] + r["l"]) >= 1 and r["prompt"].strip()]

    evq = {json.loads(l)["question"].strip().lower()
           for l in io.open(eval_path, encoding="utf-8")}
    leaked = [r["prompt"][:60] for r in out if r["prompt"].strip().lower() in evq]
    if leaked:
        raise AssertionError(
            "EVAL LEAKAGE: %d frozen-eval question(s) in the world-model training "
            "set — this launders eval difficulty into curriculum selection. "
            "First: %r" % (len(leaked), leaked[0]))
    return out


def _forward(params, X):
    h = np.maximum(0.0, X @ params["W1"] + params["b1"])
    z = h @ params["W2"] + params["b2"]
    return 1.0 / (1.0 + np.exp(-z))


def _train_head(X, y, seed, epochs=300, lr=0.05):
    """One 768->HIDDEN->1 head, full-batch gradient descent on log loss. numpy only:
    the whole point is CPU-seconds and zero VRAM (E9 addendum)."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    p = {"W1": rng.normal(0, 0.05, (d, HIDDEN)), "b1": np.zeros(HIDDEN),
         "W2": rng.normal(0, 0.05, (HIDDEN, 1)), "b2": np.zeros(1)}
    for _ in range(epochs):
        h = np.maximum(0.0, X @ p["W1"] + p["b1"])
        z = (h @ p["W2"] + p["b2"]).ravel()
        pred = 1.0 / (1.0 + np.exp(-z))
        g = (pred - y)[:, None] / len(y)
        p["W2"] -= lr * (h.T @ g)
        p["b2"] -= lr * g.sum(0)
        gh = (g @ p["W2"].T) * (h > 0)
        p["W1"] -= lr * (X.T @ gh)
        p["b1"] -= lr * gh.sum(0)
    return p


def train_ensemble(labels: list[dict], embed_fn, k: int = K_ENSEMBLE,
                   out_path: str = MODEL_FILE, seed: int = SEED):
    """K heads, each on a bootstrap resample (the Dyna-Q EnsembleWorldModel).
    Target: empirical p(correct) = w / (w + l) per prompt."""
    X = embed_fn([r["prompt"] for r in labels])
    y = np.asarray([r["w"] / (r["w"] + r["l"]) for r in labels], dtype=np.float64)
    rng = np.random.default_rng(seed)
    heads = []
    for i in range(k):
        idx = rng.integers(0, len(labels), len(labels))
        heads.append(_train_head(X[idx], y[idx], seed=seed + i))
    payload = {}
    for i, p in enumerate(heads):
        for name, arr in p.items():
            payload[f"h{i}_{name}"] = arr
    payload["k"] = np.asarray([k])
    np.savez(out_path, **payload)
    return heads


def load_ensemble(path: str = MODEL_FILE):
    z = np.load(path)
    k = int(z["k"][0])
    return [{name: z[f"h{i}_{name}"] for name in ("W1", "b1", "W2", "b2")}
            for i in range(k)]


def predict(heads, X) -> tuple[np.ndarray, np.ndarray]:
    """(p_mean, p_var) per row. p_var is MODEL DISAGREEMENT — part 6: 'best read as
    model disagreement, not as a calibrated probability'."""
    preds = np.stack([_forward(p, X).ravel() for p in heads])
    return preds.mean(0), preds.var(0)


def validate(heads, labels: list[dict], embed_fn, visits=None) -> dict:
    """The two gates that must PASS before any selection use (part 6: model loss is
    not planning reliability):
      auc        — held-out-ish discrimination on the label set (in-sample floor;
                   the honest split happens when labels allow it)
      var_slope  — disagreement must DECREASE with visit count (part 6 Q2); a flat
                   or rising slope means the ensemble is noise. Only computable when
                   visit counts are supplied.
    """
    X = embed_fn([r["prompt"] for r in labels])
    y = np.asarray([1.0 if r["w"] > r["l"] else 0.0 for r in labels])
    p_mean, p_var = predict(heads, X)
    order = np.argsort(p_mean)
    ranks = np.empty(len(y)); ranks[order] = np.arange(len(y))
    pos, neg = ranks[y == 1], ranks[y == 0]
    auc = ((pos[:, None] > neg[None, :]).mean()
           if len(pos) and len(neg) else float("nan"))
    out = {"auc": float(auc), "n": len(labels),
           "p_var_mean": float(p_var.mean())}
    if visits is not None:
        v = np.asarray(visits, dtype=np.float64)
        if v.std() > 0 and p_var.std() > 0:
            out["var_visit_corr"] = float(np.corrcoef(v, p_var)[0, 1])
    return out


# ---------------------------------------------------------------------------
# QA-verifier mode (SURPASS 11 Arm B): the same K-head ensemble, but over
# (question + answer) embeddings with the per-sample JUDGE SCORE as target —
# a learned action-value model that picks among sampled answers, i.e. the
# "separate second model" of DQN (Target Networks), used at decision time only.
# ---------------------------------------------------------------------------

QA_MODEL_FILE = "data/qa_verifier.npz"


def _qa_text(question: str, answer: str) -> str:
    return question.strip() + "\n" + answer.strip()


def build_qa_labels(harvest_stems: list[str], pool_path: str = "data/rl/r3_rl.jsonl",
                    eval_path: str = EVAL_FILE) -> list[dict]:
    """Join each harvest's .gen.jsonl (answer texts) with its .scored.jsonl
    (judge scores) BY ID, and the question text from the pool by id.

    Require: gen/scored ids align; every score list matches its completions list.
    Guarantee: zero frozen-eval questions in the label set — raises otherwise.
    """
    qtext = {json.loads(l)["id"]: json.loads(l)["question"]
             for l in io.open(pool_path, encoding="utf-8")}
    rows: list[dict] = []
    for stem in harvest_stems:
        gens = {r["id"]: r["completions"] for r in
                (json.loads(l) for l in io.open(stem + ".gen.jsonl", encoding="utf-8"))}
        for l in io.open(stem + ".scored.jsonl", encoding="utf-8"):
            r = json.loads(l)
            comps = gens.get(r["id"])
            if comps is None or r["id"] not in qtext:
                continue
            for ans, sc in zip(comps, r["scores"]):
                if not (ans or "").strip():
                    continue
                rows.append({"question": qtext[r["id"]], "answer": ans,
                             "score": min(1.0, max(0.0, float(sc)))})
    evq = {json.loads(l)["question"].strip().lower()
           for l in io.open(eval_path, encoding="utf-8")}
    leaked = [r for r in rows if r["question"].strip().lower() in evq]
    if leaked:
        raise AssertionError(
            "EVAL LEAKAGE: %d frozen-eval question(s) in the QA-verifier training "
            "set. First: %r" % (len(leaked), leaked[0]["question"][:60]))
    return rows


def train_qa_ensemble(labels: list[dict], embed_fn, k: int = K_ENSEMBLE,
                      out_path: str = QA_MODEL_FILE, seed: int = SEED):
    """Same bootstrap-resampled heads as train_ensemble; target is the judge's
    own per-sample score (soft label in [0,1] under the sigmoid/log-loss head)."""
    X = embed_fn([_qa_text(r["question"], r["answer"]) for r in labels])
    y = np.asarray([r["score"] for r in labels], dtype=np.float64)
    rng = np.random.default_rng(seed)
    heads = []
    for i in range(k):
        idx = rng.integers(0, len(labels), len(labels))
        heads.append(_train_head(X[idx], y[idx], seed=seed + i))
    payload = {}
    for i, p in enumerate(heads):
        for name, arr in p.items():
            payload[f"h{i}_{name}"] = arr
    payload["k"] = np.asarray([k])
    np.savez(out_path, **payload)
    return heads


def qa_auc(heads, labels: list[dict], embed_fn, floor: float = 0.7) -> float:
    """Held-out discrimination: does the verifier rank judge-correct answers
    (score >= floor) above judge-wrong ones?"""
    X = embed_fn([_qa_text(r["question"], r["answer"]) for r in labels])
    y = np.asarray([1.0 if r["score"] >= floor else 0.0 for r in labels])
    p_mean, _ = predict(heads, X)
    order = np.argsort(p_mean)
    ranks = np.empty(len(y)); ranks[order] = np.arange(len(y))
    pos, neg = ranks[y == 1], ranks[y == 0]
    return float((pos[:, None] > neg[None, :]).mean()) if len(pos) and len(neg) \
        else float("nan")


def qa_rerank(receipt_path: str, arm: str, heads, embed_fn,
              eval_path: str = "data/rl/r3_eval_canonical.jsonl",
              floor: float = 0.7, is_degenerate=None) -> dict:
    """Offline pick: for each frozen-eval question, the verifier chooses among the
    arm's stored k samples; the pick is settled against the ALREADY-RECORDED judge
    score for that (question, rep) in the same receipt. Zero new judge calls.

    An unscored pick (the judge dropped that rep) counts as WRONG — floors law.
    """
    if is_degenerate is None:
        from rlv2.train.rl.grpo import _is_degenerate as is_degenerate
    receipt = json.load(open(receipt_path, encoding="utf-8"))
    gens = json.load(open(receipt_path + ".generations.json", encoding="utf-8"))[arm]
    evrows = [json.loads(l) for l in io.open(eval_path, encoding="utf-8")]
    qids = [r["item_id"] for r in evrows]
    scores = {}                                   # (qid, rep) -> judge score
    for rec in receipt["records"]:
        s = rec["scores"].get(arm)
        if s and s.get("score") is not None:
            scores[(rec["qid"], rec["rep"])] = min(1.0, max(0.0, float(s["score"])))
    picks, correct = [], 0
    for qi, qid in enumerate(qids):
        cands = []
        for rep in range(len(gens)):
            text = gens[rep][qi]
            if (text or "").strip() and not is_degenerate(text):
                cands.append((rep, text))
        if not cands:
            picks.append({"qid": qid, "rep": None, "judge": None, "correct": False})
            continue
        X = embed_fn([_qa_text(evrows[qi]["question"], t) for _, t in cands])
        p_mean, _ = predict(heads, X)
        rep, _text = cands[int(np.argmax(p_mean))]
        judged = scores.get((qid, rep))
        ok = judged is not None and judged >= floor
        correct += int(ok)
        picks.append({"qid": qid, "rep": rep, "verifier": float(p_mean.max()),
                      "judge": judged, "correct": ok})
    return {"arm": arm, "n": len(qids), "correct": correct, "picks": picks}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--labels", default=LABELS_FILE)
    ap.add_argument("--out", default=MODEL_FILE)
    ap.add_argument("--qa-train", nargs="*", default=None,
                    help="harvest stems (without .gen/.scored suffix) to train the "
                         "QA verifier on")
    ap.add_argument("--qa-holdout", default=None,
                    help="one harvest stem held out for the AUC gate")
    ap.add_argument("--qa-out", default=QA_MODEL_FILE)
    ap.add_argument("--qa-rerank", default=None,
                    help="k8 receipt path: offline verifier pick over its stored "
                         "generations")
    ap.add_argument("--qa-arm", default="r1a_ipo")
    a = ap.parse_args()
    from rlv2.data.pools import embed
    if a.qa_train:
        labels = build_qa_labels(a.qa_train)
        print(f"[qa] {len(labels)} judged (q,a,score) triples "
              f"from {len(a.qa_train)} harvests (leakage tripwire passed)")
        heads = train_qa_ensemble(labels, embed, out_path=a.qa_out)
        if a.qa_holdout:
            held = build_qa_labels([a.qa_holdout])
            print(f"[qa] held-out {a.qa_holdout}: n={len(held)} "
                  f"AUC={qa_auc(heads, held, embed):.3f} (gate 0.70)")
        return
    if a.qa_rerank:
        heads = load_ensemble(a.qa_out)
        rep = qa_rerank(a.qa_rerank, a.qa_arm, heads, embed)
        print(f"[qa-rerank] arm={rep['arm']} verifier-pick correct: "
              f"{rep['correct']}/{rep['n']}")
        for p in rep["picks"]:
            print("  %-10s rep=%s verifier=%s judge=%s %s" % (
                p["qid"], p["rep"],
                ("%.2f" % p["verifier"]) if "verifier" in p else "-",
                ("%.2f" % p["judge"]) if p["judge"] is not None else "UNSCORED",
                "OK" if p["correct"] else "wrong"))
        return
    labels = build_labels(ledger_path=a.ledger, labels_path=a.labels)
    print(f"[wm] {len(labels)} labeled prompts (leakage tripwire passed)")
    heads = train_ensemble(labels, embed, out_path=a.out)
    rep = validate(heads, labels, embed)
    print(f"[wm] auc={rep['auc']:.3f} n={rep['n']} "
          f"p_var_mean={rep['p_var_mean']:.4f} -> {a.out}")


if __name__ == "__main__":
    main()
