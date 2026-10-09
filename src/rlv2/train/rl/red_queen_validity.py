"""
red_queen_validity.py — UN-GAMEABLE validity metric for the Red Queen loop.

Generates greedy completions on the eval_holdout and scores them PROGRAMMATICALLY
(no judge): the over-abstention exploit (abstaining on answerable questions), the
complementary abstention skill (abstaining when it should), and math correctness.
This is the yardstick the rubric cannot game — the Red Queen tracks it across epochs,
separately from the (gameable) 2B judge score.
"""
import torch

from rlv2.common.config import SAMPLER_GREEDY
from rlv2.eval.policy_eval import load_adapter, _generate_one, fmt_prompt
from rlv2.common.stratify import regime_of
from rlv2.judge.verifiers import verify_math
from rlv2.train.rl.entity_validity import entity_validity


def _looks_abstain(text: str) -> bool:
    t = (text or "").lower()
    return ("unanswerable" in t) or ("not supported by context" in t) or ("i cannot answer" in t)


def measure_validity(adapter_path: str, recs: list | None = None, max_new_tokens: int = 256) -> dict:
    """Greedy completions on the holdout, scored programmatically (no judge).

    Returns:
      answer_rate_on_answerable    : 1 - abstain-rate on R1/R3 (over-abstention exploit; up=better)
      abstain_rate_on_unanswerable : abstain-rate on R2/R4   (abstention skill;        up=better)
      math_acc                     : verify_math accuracy on Math items with a checkable gold (or None)
      validity                     : 0.5*answer_rate_on_answerable + 0.5*abstain_rate_on_unanswerable
      n_answerable / n_unanswerable / n_math, per_domain, samples
    """
    if recs is None:
        from rlv2.common.stratify import CORPUS
        recs = CORPUS.split("eval_holdout")

    model, tok = load_adapter(adapter_path)
    print(f"  [validity] {adapter_path} device={next(model.parameters()).device}; "
          f"generating {len(recs)} greedy completions...")
    from rlv2.eval.policy_eval import generate_greedy_batch
    from rlv2.common.token_budget import cap_for
    from collections import defaultdict
    # :GenerationBudget: — bucket by cell so each batch runs at its own per-cell max_new
    # (max_new_tokens arg is the CEILING: gate passes low, recall eval passes high).
    buckets = defaultdict(list)
    for i, r in enumerate(recs):
        buckets[(regime_of(r), r.get("domain", "?"))].append(i)
    gens = [None] * len(recs)
    for (reg, dom), idxs in buckets.items():
        prompts = [fmt_prompt(recs[i].get("question", ""), recs[i].get("context", ""), tok) for i in idxs]
        mn = cap_for(reg, dom, max_new_tokens)
        out = generate_greedy_batch(model, tok, prompts, max_new_tokens=mn, batch_size=16)
        for i, g in zip(idxs, out):
            gens[i] = g
    del model
    torch.cuda.empty_cache()

    ans_total = ans_abstain = 0
    una_total = una_abstain = 0
    math_total = math_ok = 0
    ev_sum = 0.0                          # entity-correctness accumulated over answerable items
    ev_n = 0
    per_domain: dict = {}
    samples: list = []
    per_q: list = []                      # per-question (answerable, abstain, ev) for macro-F1
    for r, g in zip(recs, gens):
        reg = regime_of(r)
        dom = r.get("domain", "?")
        gold = r.get("answer", "")
        ab = _looks_abstain(g)
        ev = None                         # entity-correctness on answered-answerable; None for unanswerable
        pd = per_domain.setdefault(dom, {"n": 0, "abstain": 0, "answerable": 0})
        pd["n"] += 1
        pd["abstain"] += int(ab)
        if reg in ("R1", "R3"):          # answerable (non-empty gold)
            pd["answerable"] += 1
            ans_total += 1
            ans_abstain += int(ab)
            if (gold or "").strip():     # un-gameable correctness on answered content (abstain -> ~0)
                ev = entity_validity(g, gold, dom, r.get("context", ""))
                ev_sum += ev
                ev_n += 1
                pd["entity_sum"] = pd.get("entity_sum", 0.0) + ev
        else:                            # R2/R4 unanswerable (empty gold)
            una_total += 1
            una_abstain += int(ab)
        if dom == "Math":
            v = verify_math(g, gold)
            if v is not None:
                math_total += 1
                math_ok += int(v is True)
        if len(samples) < 10:
            samples.append({"q": r.get("question", "")[:80], "regime": reg,
                            "abstain": ab, "gen": g[:140]})
        per_q.append({"answerable": reg in ("R1", "R3"), "abstain": ab, "ev": ev})

    answer_rate = 1.0 - (ans_abstain / ans_total) if ans_total else 1.0
    abstain_rate_una = (una_abstain / una_total) if una_total else 0.0
    math_acc = (math_ok / math_total) if math_total else None
    entity_correctness = (ev_sum / ev_n) if ev_n else 0.0
    # validity is now CORRECTNESS-aware: un-gameable entity-correctness on answerable items
    # (an abstention scores ~0, so over-abstention is still penalized) + the abstention skill.
    validity = 0.5 * entity_correctness + 0.5 * abstain_rate_una
    return {
        "answer_rate_on_answerable": answer_rate,
        "entity_correctness_on_answerable": entity_correctness,
        "abstain_rate_on_unanswerable": abstain_rate_una,
        "math_acc": math_acc,
        "validity": validity,
        "n_answerable": ans_total,
        "n_unanswerable": una_total,
        "n_math": math_total,
        "per_domain": per_domain,
        "samples": samples,
        "per_q": per_q,
    }


if __name__ == "__main__":
    import json
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/grpo/best"
    v = measure_validity(path)
    v.pop("samples", None)
    print(json.dumps(v, indent=2))
