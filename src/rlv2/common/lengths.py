"""src/rlv2/common/lengths.py — log-normal median+2MAD filter, then 4 quantile partitions, to stop paying
for padding.

THE WASTE, measured on the 400-row SFT subset (2026-08-05):

    token lengths   min 81 · p25 94 · p50 206 · p75 353 · max 448
    pad-to-448      179,200 token-slots
    4-quantile      109,877 token-slots      -> 39% FEWER

Every row was being padded to a single global MAX_LEN, so a 94-token row paid for 448. The median
supervised answer in that set is SIX tokens; the length spread is entirely prompt-side (ReadingComp
carries long contexts, Math/Code do not). Batching length-alike rows together and padding to the
BLOCK max instead of the global max recovers that 39% directly as wall-clock.

WHY LOG-NORMAL, NOT RAW. Token lengths are right-skewed and heavy-tailed -- a handful of long
contexts drag the mean and inflate a raw sigma, so a mean+2sd band on raw lengths keeps outliers
that a log-space band would catch. `log -> median -> MAD` is the robust pair: the median resists the
tail and MAD resists it twice over. `1.4826 * MAD` is the consistency constant that makes MAD a
sigma-equivalent under normality, so `median +- 2*1.4826*MAD` in log space is the ~95% band.

TWO SEPARATE JOBS, deliberately not fused:

    filter_outliers()   DROPS rows outside the band. Changes the DATA. Report the count -- on the
                        current subset it drops ZERO (band [24,1756] vs max 448), which is the
                        honest answer, not a reason to skip the check.
    quantile_blocks()   GROUPS the survivors into 4 length-alike blocks. Changes the BATCHING only,
                        never the data. This is where the 39% lives.

A filter that silently drops training rows is a data decision and must be visible; a batching
strategy is not. Fusing them hides the first behind the second.

RELATION TO `token_budget.py` (the incumbent, do not duplicate it): that module fits per-(regime x
domain) GENERATION caps -- how many tokens to let the model EMIT -- via Box-Cox PI on gold-answer
length. This module governs how many tokens we PAY FOR on the input side. Different axis, same
statistical instinct. Use both: token_budget for `max_new_tokens`, this for batching.

    from rlv2.common.lengths import filter_outliers, quantile_blocks
    kept, dropped = filter_outliers(lengths)
    for block in quantile_blocks(kept_indices, lengths):  # 4 blocks, length-alike
        ...
"""
import math

MAD_K = 1.4826          # MAD -> sigma consistency constant under normality
BAND_SIGMAS = 2.0
N_BLOCKS = 4


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        raise ValueError("median of empty sequence")
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


def log_mad_band(lengths, sigmas=BAND_SIGMAS):
    """Return (lo, hi) token-length band from median +- sigmas*MAD in LOG space.

    Require : all lengths > 0
    Guarantee: lo <= hi, both in raw token units (log space inverted on the way out)
    """
    assert lengths, "no lengths given"
    assert all(n > 0 for n in lengths), "token lengths must be positive to take a log"
    lg = [math.log(n) for n in lengths]
    med = _median(lg)
    mad = _median([abs(x - med) for x in lg])
    if mad == 0.0:
        # Degenerate: over half the rows share one length. A zero-width band would drop everything
        # outside that mode, which is a data massacre dressed as a filter. Fall back to no filtering.
        return min(lengths), max(lengths)
    spread = sigmas * MAD_K * mad
    return math.exp(med - spread), math.exp(med + spread)


def filter_outliers(lengths, sigmas=BAND_SIGMAS):
    """Split indices into (kept, dropped) by the log-MAD band. DROPS DATA -- report the count.

    Guarantee: kept + dropped partition range(len(lengths)) exactly, no index in both.
    """
    lo, hi = log_mad_band(lengths, sigmas)
    kept, dropped = [], []
    for i, n in enumerate(lengths):
        (kept if lo <= n <= hi else dropped).append(i)
    assert len(kept) + len(dropped) == len(lengths)
    return kept, dropped


def quantile_blocks(indices, lengths, n_blocks=N_BLOCKS):
    """Group indices into n_blocks length-alike blocks by quantile. BATCHING ONLY -- drops nothing.

    Returns a list of (block_max_len, [indices]) sorted short -> long. Pad each block to its OWN max
    instead of the global max; that is the entire saving.

    Guarantee: every input index appears in exactly one block.
    """
    idx = sorted(indices, key=lambda i: lengths[i])
    if not idx:
        return []
    n, out = len(idx), []
    for b in range(n_blocks):
        s, e = (b * n) // n_blocks, ((b + 1) * n) // n_blocks
        chunk = idx[s:e]
        if chunk:
            out.append((max(lengths[i] for i in chunk), chunk))
    assert sum(len(c) for _, c in out) == n, "blocks lost rows"
    return out


def report(lengths, global_max, n_blocks=N_BLOCKS, sigmas=BAND_SIGMAS, label=""):
    """Print the measured saving. Call before training so the number is on the record, not assumed."""
    lo, hi = log_mad_band(lengths, sigmas)
    kept, dropped = filter_outliers(lengths, sigmas)
    blocks = quantile_blocks(kept, lengths, n_blocks)
    flat = global_max * len(lengths)
    part = sum(bm * len(ix) for bm, ix in blocks)
    s = sorted(lengths)
    print(f"[length{'/' + label if label else ''}] n={len(lengths)} "
          f"min {s[0]} p50 {s[len(s)//2]} max {s[-1]}", flush=True)
    print(f"[length] log-MAD band [{lo:.0f},{hi:.0f}] -> dropped {len(dropped)} "
          f"({100*len(dropped)/len(lengths):.1f}%)", flush=True)
    print(f"[length] pad-to-{global_max}: {flat:,} slots | {len(blocks)}-block: {part:,} slots "
          f"-> {100*(1-part/flat):.0f}% fewer", flush=True)
    return {"band": [lo, hi], "dropped": len(dropped), "blocks": [bm for bm, _ in blocks],
            "slots_flat": flat, "slots_blocked": part,
            "saving_frac": 1 - part / flat if flat else 0.0}


if __name__ == "__main__":
    import io
    import json
    import sys
    sys.path.insert(0, ".")
    from transformers import AutoTokenizer
    from rlv2.common.config import MODEL_NAME
    # policy_eval.py lives in sprawl/ after the 1e515fa restructure; src/ modules that
    # import it unqualified die at run time (ModuleNotFoundError). Same defect fixed in
    # src/rlv2/train/sft/train.py -- swept across every src/ importer rather than one at a time.
    import os as _os, sys as _sys
    from rlv2.eval.policy_eval import fmt_prompt
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    rows = [json.loads(l) for l in io.open("data/rl/vibe_sft_subset.jsonl", encoding="utf-8")]
    lens = []
    for r in rows:
        g = (r.get("gold") or "").strip() or (
            "not supported by context" if (r.get("context") or "").strip() else "unanswerable")
        lens.append(len(tok(fmt_prompt(r.get("prompt", ""), r.get("context", "") or "",
                                       tok))["input_ids"])
                    + len(tok(g, add_special_tokens=False)["input_ids"]) + 1)
    report([min(n, 448) for n in lens], 448, label="vibe_sft_subset")
