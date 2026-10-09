"""src/rlv2/eval/score_r3_categories.py — the R3 objective readout, per CATEGORY, at k samples.

WHY A SEPARATE READER. `rl_calibration_eval.py` reports a regime x DOMAIN grid (R1-R4 x
Math/Code/STEM/Instruction/Agentic/ReadingComp). That grid was built for the VibeThinker
mix. Under the §3.5.1 scope cut every item is R3 and every item's domain is "domain", so
the grid collapses to a single cell and tells you nothing.

The axis that matters now is CATEGORY -- the four dataset regimes mimicked onto our domain
(owner, 2026-08-26: "we are only interested in r3 f-1 across 4 dataset regimes ... applied
to our domain"). `f1_abstain` is retained ONLY for backwards compatibility with existing
calibration.json entries; with no abstention class it is structurally 0.0 and carries no
information here.

WHY k SAMPLES. §6.10: 20 items at k=1 have too few outcomes to separate a real move from
decode variance. k=8 turns 20 questions into 160 observations at seconds of cost. The seed
is fixed so base and checkpoint are compared on the SAME draws (§1.5).

Run:
  PYTHONPATH=src python -u src/rlv2/eval/score_r3_categories.py --model skills_model --tag base_k8
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st

EVAL = "data/rl/r3_eval_canonical.jsonl"
CATEGORY_ORDER = ["Math", "Code", "STEM", "Instruction"]
# Legacy fallback threshold, used ONLY for records with no verdict (see is_correct).
# Raised 0.5 -> CORRECT_FLOOR 2026-08-31: `judge.combine` maps an `incorrect` verdict to
# strictly BELOW 0.70 ((0.70-0.01)*quality), so at 0.5 a fluent wrong answer with
# quality >= 0.72 counted as correct -- the exact leak the docstring below describes.
# At the canonical floor the fallback and the verdict can no longer disagree.
from rlv2.judge.judge import CORRECT_FLOOR
TAU = CORRECT_FLOOR


def is_correct(rec: dict) -> bool:
    """Did this sample answer the question? The VERDICT decides when there is one.

    THE SCORE THRESHOLD LEAKS. `judge.combine` maps an `incorrect` verdict to
    CORRECT_FLOOR * quality, i.e. up to 0.70 -- so at TAU=0.5 any WRONG answer written with
    quality >= 0.72 was counted CORRECT. That is precisely the fluent-but-wrong failure mode
    the three-way verdict exists to catch, and it inflated every k=8 reading taken before
    2026-08-28.

    `half_epoch_eval.evaluate` already puts `verdict` on every record and its own comment
    says the verdict -- not a threshold on a score -- is what the gate counts. This makes the
    aggregation agree with that.

    The TAU fallback is kept so older records that carry only a reward still summarise, but
    it is a floor, not a measurement.
    """
    v = rec.get("verdict")
    if v is not None:
        return v == "correct"
    return float(rec["reward"]) >= TAU


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def summarise(records: list[dict], k: int) -> dict:
    """records: one row per (item, sample) with keys item_id, category, reward."""
    by_item: dict[str, list[float]] = collections.defaultdict(list)
    ok_of: dict[str, list[bool]] = collections.defaultdict(list)
    cat_of: dict[str, str] = {}
    for r in records:
        by_item[r["item_id"]].append(float(r["reward"]))
        ok_of[r["item_id"]].append(is_correct(r))
        cat_of[r["item_id"]] = r.get("category", "?")

    per_cat: dict[str, dict] = {}
    for cat in CATEGORY_ORDER:
        items = [i for i, c in cat_of.items() if c == cat]
        if not items:
            per_cat[cat] = {"n_items": 0, "n_obs": 0, "correct": 0,
                            "rate": 0.0, "reward_mean": 0.0}
            continue
        obs = [v for i in items for v in by_item[i]]
        correct = sum(1 for i in items for ok in ok_of[i] if ok)
        per_cat[cat] = {
            "n_items": len(items),
            "n_obs": len(obs),
            "correct": correct,
            "rate": correct / max(1, len(obs)),
            "reward_mean": sum(obs) / max(1, len(obs)),
        }

    all_obs = [v for vs in by_item.values() for v in vs]
    all_ok = [ok for oks in ok_of.values() for ok in oks]
    n_ok = sum(1 for ok in all_ok if ok)
    return {
        "k": k,
        "n_items": len(by_item),
        "n_obs": len(all_obs),
        "correct": n_ok,
        "rate": n_ok / max(1, len(all_obs)),
        "reward_mean": sum(all_obs) / max(1, len(all_obs)),
        "per_category": per_cat,
        "per_item": {i: {"category": cat_of[i],
                         "correct": sum(1 for ok in ok_of[i] if ok),
                         "k": len(vs),
                         "reward_mean": sum(vs) / max(1, len(vs)),
                         "reward_sd": st.pstdev(vs) if len(vs) > 1 else 0.0}
                     for i, vs in by_item.items()},
    }


def render(summary: dict, tag: str, baseline: dict | None = None) -> str:
    """The §6.10 grid, on the axis that carries information."""
    out = []
    out.append("=" * 74)
    out.append("R3 OBJECTIVE — %s   (%d items x k=%d = %d observations)"
               % (tag, summary["n_items"], summary["k"], summary["n_obs"]))
    out.append("=" * 74)
    out.append("%-14s %7s %9s %8s %9s %8s" %
               ("category", "items", "correct", "rate", "reward", "Δ rate"))
    out.append("-" * 74)
    for cat in CATEGORY_ORDER:
        c = summary["per_category"][cat]
        delta = ""
        if baseline:
            b = baseline["per_category"].get(cat, {}).get("rate")
            if b is not None:
                delta = "%+.3f" % (c["rate"] - b)
        out.append("%-14s %7d %5d/%-3d %8.3f %9.3f %8s" %
                   (cat, c["n_items"], c["correct"], c["n_obs"],
                    c["rate"], c["reward_mean"], delta))
    out.append("-" * 74)
    d = "%+.3f" % (summary["rate"] - baseline["rate"]) if baseline else ""
    out.append("%-14s %7d %5d/%-3d %8.3f %9.3f %8s" %
               ("TOTAL", summary["n_items"], summary["correct"], summary["n_obs"],
                summary["rate"], summary["reward_mean"], d))
    out.append("")
    out.append("f1_abstain: 0.000 (structural — R3 has no abstention class; retained "
               "for backwards compatibility only)")
    out.append("")
    out.append("PER-ITEM (a mean over 20 items hides everything — §6.10):")
    for i, v in sorted(summary["per_item"].items()):
        flag = "" if v["correct"] else "   <- never correct"
        out.append("  %-10s %-12s %d/%d  reward %.3f ±%.3f%s"
                   % (i, v["category"], v["correct"], v["k"],
                      v["reward_mean"], v["reward_sd"], flag))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True,
                    help="jsonl of {item_id, category, reward} — one row per sample")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--baseline", default=None,
                    help="a previous summary json, to print Δ rate per category")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    summary = summarise(load(a.records), a.k)
    base = json.load(open(a.baseline, encoding="utf-8")) if a.baseline else None
    print(render(summary, a.tag, base))

    out = a.out or os.path.join("data", "rl", "r3_objective_%s.json" % a.tag)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
