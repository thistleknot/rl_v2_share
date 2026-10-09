"""src/rlv2/common/token_budget.py — :GenerationBudget: per-(regime x domain) cell max_new_tokens.

Answer length is a function of the CELL, not the question (H2, confirmed: within-cell
corr(answer,question) weak everywhere except R3|Code; cross-cell p50 spans 2->186 tokens).
So fit answer length ALONE, per cell: gold-answer token counts -> Box-Cox -> normal PI in
transformed space -> invert -> 97.5% upper = that cell's max_new_tokens, clamped to
[FLOOR, CEILING]. Abstention cells (R2/R4) have empty gold -> pooled abstention-string cap.

MAX ONLY. No min_new_tokens anywhere: a min suppresses EOS, which changes the measured output
distribution AND forces tokens past the abstention string measure_validity keys on.

Cache: data/cell_max_new.json  (regenerate with `python token_budget.py`).
"""
import json
from pathlib import Path

CACHE = Path("data/cell_max_new.json")
FLOOR, CEILING, ABSTAIN_CAP, PI = 16, 1024, 32, 0.975


def compute(ceiling: int = CEILING) -> dict:
    """Fit the per-cell cap from CORPUS gold answers. Returns {"R#|Domain": max_new}."""
    import numpy as np
    from scipy import stats
    from scipy.special import inv_boxcox
    from collections import defaultdict
    from rlv2.common.stratify import CORPUS, cell_key, is_abstain
    from transformers import AutoTokenizer
    from rlv2.common.config import MODEL_NAME

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    lens: dict = defaultdict(list)
    abstain_cells = set()
    for r in CORPUS.records():
        c = cell_key(r)
        if is_abstain(r):
            abstain_cells.add(c)
            continue
        n = len(tok(r.get("answer", "") or "", add_special_tokens=False)["input_ids"])
        if n > 0:
            lens[c].append(n)

    out: dict = {}
    for c, vals in lens.items():
        v = np.array(vals)
        if len(v) >= 8:
            bc, lam = stats.boxcox(v + 1)
            hi = float(inv_boxcox(bc.mean() + stats.norm.ppf(PI) * bc.std(), lam) - 1)
        else:                                   # small-n: empirical p95 (parametric PI unstable)
            hi = float(np.percentile(v, 95))
        out[f"{c[0]}|{c[1]}"] = int(min(max(hi, FLOOR), ceiling))
    for c in abstain_cells:                     # empty gold -> pooled abstention-string cap
        out[f"{c[0]}|{c[1]}"] = ABSTAIN_CAP
    return out


def load() -> dict:
    """Cached per-cell caps; compute+cache on first use."""
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    table = compute()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(table, indent=2), encoding="utf-8")
    return table


def cap_for(regime: str, domain: str, ceiling: int) -> int:
    """Per-cell cap, clamped by the caller's ceiling (gate=low, recall-eval=high)."""
    base = regime.replace("[inferred]", "")
    return min(load().get(f"{base}|{domain}", ceiling), ceiling)


def filter_truncating(records: list, ceiling: int = CEILING) -> tuple:
    """:TruncationFreeSampling: (data half) — drop any record whose GOLD-answer token count
    exceeds its cell's :GenerationBudget: cap, so a completion generated under that cap can
    never truncate the answer. Returns (kept, dropped). Loads the tokenizer once."""
    from transformers import AutoTokenizer
    from rlv2.common.config import MODEL_NAME
    from rlv2.common.stratify import cell_key
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    caps = load()
    kept, dropped = [], []
    for r in records:
        c = cell_key(r)
        cap = min(caps.get(f"{c[0]}|{c[1]}", ceiling), ceiling)
        n = len(tok(r.get("answer", "") or "", add_special_tokens=False)["input_ids"])
        (dropped if n > cap else kept).append(r)
    return kept, dropped


def batch_completion_cap(records: list, ceiling: int = CEILING) -> int:
    """:TruncationFreeSampling: (generation half) — the max_completion_length a batch must use so
    NO record in it truncates: the largest per-cell :GenerationBudget: cap present in the batch.
    Short cells still EOS early under it; long cells are allowed to finish."""
    from rlv2.common.stratify import cell_key
    caps = load()
    return max((min(caps.get(f"{cell_key(r)[0]}|{cell_key(r)[1]}", ceiling), ceiling)
                for r in records), default=ceiling)


if __name__ == "__main__":
    table = compute()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"Wrote {CACHE} ({len(table)} cells):")
    for k in sorted(table):
        print(f"  {k:22} {table[k]}")
