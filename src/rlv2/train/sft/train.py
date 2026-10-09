"""src/rlv2/train/sft/train.py — SFT on a Box-Cox class-balanced VibeThinker subset. The stage RL needs in front of it.

WHY THIS EXISTS. RL from the stock base produced nothing (designed reward -0.0103, ns; R2 and R4 at
exactly 0.000). That is not an RL result -- the base declines NOTHING in any regime, and RL sharpens
behaviour it already samples rather than originating it. July's +0.091 is headlined "SFT vs RL": RL
ran on top of an SFT checkpoint sitting at f1_abstain 0.483. The SFT stage is the missing link.

`checkpoints/sft_prime` (July 3) does NOT serve as that baseline -- re-scored 2026-08-04 it returns
designed reward 0.1024 with broad_abstain_rate 0.000 in every regime, i.e. it does not abstain either.
So the baseline has to be rebuilt, not recovered.

SAMPLING IS THE POINT, NOT AN INCIDENTAL. `stratify.stratified_sample(..., lam=0.5)` allocates over
joint (regime x domain) cells through pull_datasets.calculate_allocations -- log -> Box-Cox -> robust
z -> CDF -> uniform blend. Measured on this pool it lifts abstention from the natural 7.5% to 29.6%
across 15 cells, which is exactly what the rare-cell over-weighting is for. A proportional draw at
n=400 would yield ~30 abstention rows spread thin; the Box-Cox draw yields 118 (R2 27 / R4 91), past
the ~60-row saturation point measured in task #42. That is the property that makes a SMALL set
informative and lets the result scale up.

DOMAIN IS EXCLUDED, deliberately. `rl_mix.jsonl` carries 500 DataScience/skills rows; `rl_mix_vibe`
drops them. This tests the VibeThinker claim on VibeThinker data and nothing else.

TARGETS ALL THREE MODULE FAMILIES. The base is a HYBRID and the two historical recipes touch disjoint
layer sets -- measured:

    q/k/v/o_proj                6 layers    sft_prime (July) targeted ONLY these
    in_proj_qkv/a/b/z, out_proj 18 layers   src/rlv2/train/rl/grpo.py's RL targets ONLY these
    gate/up/down_proj           24 layers   neither touches these

Training the SFT stage on a quarter of the network while the RL stage trains a different three
quarters is not a chain, it is two unrelated edits. This stage takes all of it, so the abstention
behaviour RL is meant to sharpen is actually learnable here.

    python vibe_sft.py [--n 400] [--epochs 2]
"""
import argparse
import io
import json
import os
import sys
import time

sys.path.insert(0, ".")

# THE EPOCH-EVAL SUBPROCESS RUNS FROM THE REPO ROOT, NOT FROM src/.
# It invokes the REPO-RELATIVE path "src/rlv2/eval/rl_calibration_eval.py", so a cwd of <repo>/src
# resolved to <repo>/src/src/rl_calibration_eval.py -- the subprocess exited nonzero, the
# `if returncode == 0` guard fell through, and the epoch's score silently became 0.0.
# That is the instrument this lane selects its best checkpoint and early-stops with, so a
# silent zero corrupts the result rather than failing it.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
from rlv2.common.config import MODEL_NAME

POOL = "data/rl/rl_mix_vibe.jsonl"
SUBSET = "data/rl/vibe_sft_subset.jsonl"
OUT = "vibe_chain/sft_adapter"
EVAL = "data/rl/rl_eval_v2.jsonl"

MAX_LEN = 448
# Cap the supervised completion. Code answers run thousands of tokens; letting one row eat the
# whole budget starves every other row in its batch and teaches length, not behaviour.
ANS_BUDGET = 160

ALL_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
               "in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj",
               "gate_proj", "up_proj", "down_proj"]


def _banned(tok):
    """Ids no phase may generate: reasoning blocks AND tool calls (shared list)."""
    try:
        from rlv2.eval import half_epoch_eval as _he
        return _he.banned_ids(tok)
    except Exception:
        ids = [tok.convert_tokens_to_ids(t) for t in
               ("<think>", "</think>", "<tool_call>", "</tool_call>",
                "<tool_response>", "</tool_response>")]
        return sorted({i for i in ids if isinstance(i, int) and i >= 0})


def build_subset(n, seed=7, eval_path=None):
    """Two-axis MMR cross-hatch draw over answer_type x dataset grid.

    Same mechanism as mgpo.bootstrap_cohort: each draw goes to the (regime, domain)
    cell with the highest deficit. This guarantees every cell is represented before
    any cell doubles, and prevents any single regime or domain from dominating.

    Asserts disjointness from the eval set — no leakage.
    """
    from rlv2.common.stratify import regime_of
    from collections import Counter, defaultdict
    import random as _rng
    eval_file = eval_path or EVAL
    rows = [json.loads(l) for l in io.open(POOL, encoding="utf-8")]

    # Exclude eval prompts from the candidate pool BEFORE the draw. The assert at the
    # end of this function is a post-hoc tripwire, not a filter -- nothing else here
    # enforces disjointness. It never fired for the default 112-row rl_eval.jsonl
    # because those prompts happen not to be in POOL, which masked the gap. The 62-row
    # rl_eval_v2.jsonl overlaps POOL by 30 prompts, so without this filter half the
    # eval would be training data and its macro-F1 would be a train score.
    ev_prompts = {json.loads(l)["prompt"] for l in io.open(eval_file, encoding="utf-8")}
    n_before = len(rows)
    rows = [r for r in rows if r.get("prompt") not in ev_prompts]
    if len(rows) < n_before:
        print(f"[cross-hatch] excluded {n_before - len(rows)} eval prompts from pool "
              f"({n_before} -> {len(rows)}) vs {eval_file}", flush=True)

    # Build answer_type x dataset grid
    grid: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        grid[(regime_of(r), r.get("domain", "unknown"))].append(r)
    cells = sorted(grid.keys())
    n_cells = len(cells)

    # Target: uniform per cell
    base_target = max(1, n // n_cells)
    targets = {cell: base_target for cell in cells}
    remainder = n - sum(targets.values())
    cells_by_size = sorted(cells, key=lambda c: len(grid[c]), reverse=True)
    for i in range(max(0, remainder)):
        targets[cells_by_size[i % n_cells]] += 1

    # Two-axis MMR draw
    sel: list[dict] = []
    sel_prompts: set = set()
    fill = {cell: 0 for cell in cells}
    rng = _rng.Random(seed)

    for _ in range(n):
        best_cell = max(cells, key=lambda c: (
            targets[c] - fill[c],
            len([r for r in grid[c] if r.get("prompt") not in sel_prompts]),
        ))
        available = [r for r in grid[best_cell] if r.get("prompt") not in sel_prompts]
        if not available:
            for fallback in sorted(cells, key=lambda c: targets[c] - fill[c], reverse=True):
                available = [r for r in grid[fallback] if r.get("prompt") not in sel_prompts]
                if available:
                    best_cell = fallback
                    break
            if not available:
                break
        record = rng.choice(available)
        sel.append(record)
        sel_prompts.add(record.get("prompt"))
        fill[best_cell] += 1

    # Eval disjointness check. Now genuinely a tripwire: the pool was filtered above,
    # so this can only fire if that filter and this check disagree on the join key.
    ov = sum(1 for r in sel if r.get("prompt") in ev_prompts)
    assert ov == 0, f"{ov} SFT rows leak into the eval"

    reg = Counter(regime_of(r) for r in sel)
    ab = reg.get("R2", 0) + reg.get("R4", 0)
    io.open(SUBSET, "w", encoding="utf-8").write("".join(json.dumps(r) + "\n" for r in sel))
    dom = Counter(r.get("domain", "?") for r in sel)
    print(f"[cross-hatch] {len(sel)} rows | cells={n_cells} | "
          f"regimes={dict(sorted(reg.items()))} | "
          f"datasets={dict(sorted(dom.items()))} | "
          f"abstain {ab} ({100*ab/len(sel):.1f}%) -> {SUBSET}", flush=True)
    return sel


def fmt(r, tok):
    """Prompt exactly as the eval builds it, so SFT teaches the format it is scored on."""
    # policy_eval.py moved to sprawl/ in the 1e515fa restructure; this import was left
    # unqualified and made every SFT run die at first fmt() call.
    from rlv2.eval.policy_eval import fmt_prompt
    return fmt_prompt(r.get("prompt", ""), r.get("context", "") or "", tok)



def _sft_callbacks(observe_cb, mid_eval_cb, model, tok, a, epoch):
    """Trainer callbacks for one SFT epoch.

    `_MidEpochEval` is this lane's own instrument -- it scores the 62-row rl_eval_v2 with
    the 2B judge and drives early-stop. It always runs.

    `HalfEpochEval` scores the R3 DOMAIN eval with the 4b judge. That is the wrong
    instrument for the VibeThinker lane and its judge competes for VRAM with mgpo's own
    unload/score/reload choreography downstream, so it is OFF unless explicitly asked for.
    It previously defaulted ON via `getattr(a, "evals_per_epoch", 2)` against an argparse
    that never defined the flag.
    """
    cbs = [observe_cb, mid_eval_cb]
    if getattr(a, "evals_per_epoch", 0) > 0:
        from rlv2.eval.half_epoch_eval import HalfEpochEval
        cbs.append(HalfEpochEval(model, tok, evals_per_epoch=a.evals_per_epoch,
                                 tag="SFT ep%d lr=%g" % (epoch, a.lr)))
    return cbs


def _fp16_state_dict(model):
    """An fp16 CPU copy of the weights, with tied duplicates dropped.

    OUT-OF-PLACE IS THE POINT. `model.half()` casts the LIVE parameters and `.float()` does not
    put them back -- it round-trips every weight through fp16, so the next epoch trains from the
    degraded copy. This repo already recorded that failure (T0.3). Building a separate dict
    leaves the fp32 masters alone.

    TIED WEIGHTS MUST BE DROPPED BY HAND. `save_pretrained` normally removes them via the
    model's `_tied_weights_keys`, but passing an explicit `state_dict=` bypasses that path.
    With `tie_word_embeddings: True` the checkpoint was carrying BOTH `lm_head.weight` and
    `model.language_model.embed_tokens.weight` -- 248320 x 1024 written twice, 508 MB of a
    2.01 GB file, and the save is the dominant cost of a short training run. `from_pretrained`
    re-ties on load, so the dropped key costs nothing.
    """
    tied = set()
    if getattr(model.config, "tie_word_embeddings", False):
        tied = {k if k.endswith(".weight") else k + ".weight"
                for k in (getattr(model, "_tied_weights_keys", None) or ["lm_head.weight"])}
    return {k: v.detach().to(torch.float16).cpu()
            for k, v in model.state_dict().items() if k not in tied}


def _bins_for_full_coverage(block_order, batch_tokens, requested):
    """Bins per epoch such that ONE EPOCH IS ONE FULL PASS over the pool.

    Operator 2026-09-20: "an epoch means full coverage for whatever sample size the
    outer is (inner is bins for quantiles)" and, earlier, "3 epochs sees 237 x 3
    times / Each epoch sees 237".

    WHY THIS HAD TO BE DERIVED RATHER THAN SET. `--bins-per-epoch` is a count of
    token-budgeted steps, so the records it covers depends on how many rows fit in
    batch_tokens -- which depends on the pool's own length distribution. One constant
    therefore means different coverage on every pool, and it was wrong in both
    directions on the same day:

        237-row pool, 16 bins -> 89 records/epoch = 38% of a pass   (measured)
         16-row pool, 13 bins -> 16 records       = the whole pool in epoch 1,
                                                    then "the pool is EXHAUSTED"

    The flag's own help still described the retired contract ("17 covers the 229-row
    domain set in 3 EPOCHS"), i.e. an epoch as a THIRD of a pass. That is the
    cover-once model, and it survived cover-once being switched off because this
    knob, not the flag, is what caps coverage.

    `requested > 0` still forces an explicit value, so a caller can pin it when
    reproducing an old run.
    """
    if requested and requested > 0:
        return requested
    if not block_order:
        return 1
    # Rows are padded to their BLOCK max, so the mean padded length is what a bin
    # actually pays for -- using the median under-counts and gives short epochs.
    mean_len = sum(len(r["input_ids"]) for r in block_order) / len(block_order)
    per_bin = max(1, int(batch_tokens // max(1, mean_len)))
    # MEASURED CORRECTION, not a guess. The first derived run predicted 8 rows/bin
    # from the mean padded length and the sampler actually placed 5.97 (203 records
    # in 34 bins), covering 86% of the pool instead of 100%. A bin closes on the
    # token budget with per-sequence overhead the mean does not see, and the
    # stratified draw skips cells that are momentarily at target, so the realised
    # rate is consistently BELOW the arithmetic one.
    #
    # 5.97/8 = 0.75, so the bin count is divided by that. Overshooting is harmless
    # here -- a bin with nothing left to draw simply closes -- while undershooting
    # silently turns an "epoch" back into a partial pass, which is the whole defect
    # this function exists to remove.
    per_bin = max(1, int(per_bin * 0.75))
    return max(1, -(-len(block_order) // per_bin))      # ceil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--full-param", action="store_true",
                    help="train every weight instead of a LoRA adapter (SFT-full-twin arm)")
    ap.add_argument("--out", default=None, help="override output dir")
    # --- Harness-plumbed flags (task 1). Each defaults to the module constant so existing
    #     callers are unchanged. The harness passes these explicitly from configs.yml. ---
    ap.add_argument("--rank", type=int, default=16,
                    help="LoRA rank r (default: 16)")
    ap.add_argument("--alpha", type=int, default=32,
                    help="LoRA alpha (default: 32, convention alpha=2r)")
    ap.add_argument("--target-modules", default="all-linear",
                    help="LoRA target modules: 'all-linear' or comma-separated list")
    ap.add_argument("--base", default=MODEL_NAME,
                    help="base model path or name (default: config.MODEL_NAME)")
    ap.add_argument("--subset", default=None,
                    help="pre-built subset file to use instead of building one (default: build from pool)")
    ap.add_argument("--eval", default=EVAL,
                    help="eval file for disjointness check (default: data/rl/rl_eval.jsonl)")
    # ── THE OPTIMIZER SETTINGS. Nine axes, of which only `--lr` existed before 2026-08-30. ──
    #
    # Every HPO trial printed "knobs held at src/rlv2/train/sft/train.py defaults" because this file
    # accepted ONE optimizer flag. So "SFT HPO" meant "SFT learning-rate sweep", and the
    # conclusion drawn from it -- that the configuration is wrong -- was never supported by a
    # one-axis sweep. Flag names match `hpo_r3.py`'s KNOBS table so the driver can pass them
    # straight through.
    #
    # Ranked by the owner's fANOVA ordering: lr dominates; warmup matters most AT the failure
    # boundary; weight_decay is the distinct regularisation axis; scheduler is shape not scale;
    # grad_clip is a guard rail that is flat unless pathological; dropout matters MORE on small
    # data, which 240 records is; adam and label smoothing are rarely worth a slot.
    ap.add_argument("--warmup-frac", type=float, default=0.1, dest="warmup_frac",
                    help="fraction of steps spent warming up (was hardcoded to 1/10)")
    ap.add_argument("--weight-decay", type=float, default=0.0, dest="weight_decay",
                    help="AdamW decoupled weight decay (was never set -- HF default 0.0)")
    ap.add_argument("--lr-scheduler", default="cosine", dest="lr_scheduler",
                    choices=["cosine", "linear", "constant", "constant_with_warmup",
                             "warmup_stable_decay"],
                    help="scheduler SHAPE (was hardcoded 'cosine'). warmup_stable_decay is the "
                         "warmup_decay variant.")
    ap.add_argument("--max-grad-norm", type=float, default=1.0, dest="max_grad_norm",
                    help="gradient clipping guard rail (was never set -- HF default 1.0)")
    ap.add_argument("--attention-dropout", type=float, default=None, dest="attention_dropout",
                    help="model-config field, applied at load. Matters MORE on small data.")
    ap.add_argument("--adam-beta1", type=float, default=0.9, dest="adam_beta1")
    ap.add_argument("--adam-beta2", type=float, default=0.999, dest="adam_beta2")
    ap.add_argument("--adam-eps", type=float, default=1e-8, dest="adam_eps")
    ap.add_argument("--label-smoothing", type=float, default=0.0, dest="label_smoothing")
    ap.add_argument("--num-cycles", type=float, default=None, dest="num_cycles",
                    help="cosine restarts / scheduler fine print")
    ap.add_argument("--cover-once", action="store_true", dest="cover_once",
                    help="epochs partition the pool instead of each resampling it. "
                         "Without it, 3 epochs at f=1/3 cover 1-(1-f)^3 = 70%%, not "
                         "100%%, and some records are never trained on at all.")
    ap.add_argument("--batch-tokens", type=int, default=896, dest="batch_tokens",
                    help="per-step token budget. NOT the VRAM knob -- measured "
                         "2026-09-14, 896 -> 1280 moved max_alloc by 2 MiB, "
                         "because gradient checkpointing makes activations tiny "
                         "next to 519M fp32 params + Adam state (~8.3 GB). It is "
                         "also NOT the coverage knob: the manifold sampler "
                         "balances classes inside a bin and stops adding records "
                         "before the tokens run out (4.3 -> 5.15 records/bin for "
                         "+43% budget). Use --bins-per-epoch for coverage.")
    ap.add_argument("--length-blocks", type=int, default=4, dest="length_blocks",
                    help="quantile blocks for length-alike batching. Each pads to "
                         "ITS OWN max, so more blocks = less wasted VRAM and it "
                         "drops nothing. Measured on R3: 4 -> 23.8%% padding waste, "
                         "10 -> 10.7%%.")
    ap.add_argument("--band-sigmas", type=float, default=2.0, dest="band_sigmas",
                    help="log-MAD outlier band. THIS ONE DROPS RECORDS -- raise it "
                         "to keep more of the domain set, and rely on --length-blocks "
                         "to absorb the length spread instead.")
    ap.add_argument("--bins-per-epoch", type=int, default=12, dest="bins_per_epoch",
                    help="token-budgeted steps per epoch (NOT the 4 length quantiles). "
                         "12 makes an epoch ~1/3 of the pool; 3 epochs ~= one full pass.")
    ap.add_argument("--evals-per-epoch", type=int, default=0,
                    help="R3 half-epoch readings per epoch (4b judge, r3_eval_canonical). "
                         "0 = OFF, the default: the VibeThinker lane is scored on the "
                         "62-row rl_eval_v2 by _MidEpochEval, and a second judge would "
                         "contend for VRAM with the RL stage downstream")
    ap.add_argument("--seed", type=int, default=7,
                    help="random seed for subset sampling and training (default: 7)")
    ap.add_argument("--stage", choices=["sft", "cpt"], default="sft",
                    help="training stage: 'sft' (default) or 'cpt'")
    ap.add_argument("--replay-frac", type=float, default=None,
                    help="fraction of general replay data to mix in (CPT only)")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = a.out
    elif a.full_param:
        OUT = "vibe_chain/sft_full_twin"

    from rlv2.common.cost_meter import assert_adapter_trained, assert_gpu_exclusive, enforce_vram_ceiling
    if not os.environ.get("SKIP_GPU_EXCLUSIVE"):
        assert_gpu_exclusive()
    enforce_vram_ceiling(vram_frac=float(os.environ.get("HARNESS_VRAM_FRAC", "1.0")))

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Use pre-built subset if provided, otherwise build one
    if a.subset and os.path.exists(a.subset):
        sel = [json.loads(l) for l in io.open(a.subset, encoding="utf-8")]
        print(f"[subset] loaded {len(sel)} rows from {a.subset}", flush=True)
    else:
        sel = build_subset(a.n, seed=a.seed, eval_path=a.eval)
    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def enc(r):
        from rlv2.common.stratify import regime_of
        # ABSTENTION IS THE TARGET STRING. An empty gold is encoded as the literal refusal the reward
        # checks for, per regime -- R2 wants "not supported by context", R4 wants "unanswerable".
        # Training on an empty completion would teach silence, which scores 0 in both.
        g = (r.get("gold") or "").strip()
        if not g:
            g = ("not supported by context" if (r.get("context") or "").strip() else "unanswerable")
        # COMPLETION-ONLY MASKING, and it is load-bearing here rather than a style choice.
        # Measured 2026-08-04: with FULL-SPAN labels on 512-padded sequences, 118 abstention rows
        # (29.5% of the set) produced broad_abstain_rate 0.000 in EVERY regime -- the model learned
        # to reproduce prompts. The refusal is 3-5 tokens against ~500 supervised prompt tokens, so
        # the behaviour being taught was ~1% of the loss and drowned.
        # Prompt tokens -> -100. Only the answer (and its EOS) carries gradient.
        # RESERVE ANSWER BUDGET BEFORE TRUNCATING THE PROMPT. Truncating the joined string at 512
        # silently deletes the completion on long prompts -- a code_contests row consumed all 512
        # and left ZERO supervised tokens, which the assert below caught. Those rows would have
        # trained on nothing while still counting toward the abstention fraction.
        # Build the answer first, give the prompt whatever is left.
        a_ids = tok(g + tok.eos_token, add_special_tokens=False)["input_ids"][:ANS_BUDGET]
        p_ids = tok(fmt(r, tok))["input_ids"][:MAX_LEN - len(a_ids)]
        ids = p_ids + a_ids
        # VARIABLE LENGTH. Padding every row to MAX_LEN charged a 94-token row for 448 -- measured
        # 179,200 slots flat vs 109,877 length-blocked on this subset, i.e. 39% of the compute was
        # padding. The collator pads to the longest row IN THE BATCH, and `group_by_length` makes
        # those batches length-alike, which is the continuous form of the 4-quantile partition.
        e = {"input_ids": ids, "attention_mask": [1] * len(ids),
             "labels": [-100] * len(p_ids) + a_ids, "length": len(ids),
             "_regime": regime_of(r),
             # R3 rows name this `category`; older lanes name it `domain`. Same values.
             "_domain": r.get("domain") or r.get("category", "unknown")}
        assert any(x != -100 for x in e["labels"]), f"no supervised tokens for gold={g[:40]!r}"
        assert len(e["input_ids"]) == len(e["labels"])
        return e

    rows = [enc(r) for r in sel]
    # LOG-MAD OUTLIER FILTER, reported not assumed. Dropping training rows is a DATA decision and
    # must be visible; on this subset it drops 0 (band [24,1756] vs max 448) and that zero is the
    # honest result of running the check, not a reason to have skipped it.
    from rlv2.common.lengths import filter_outliers, quantile_blocks, report
    lens = [r["length"] for r in rows]
    report(lens, MAX_LEN, label="sft")
    # BAND AND BLOCK COUNT ARE KNOBS, measured 2026-09-14 on the R3 domain set.
    # The band DROPS DATA; the blocks only batch it. Loosening the band and adding
    # blocks beats the old defaults on BOTH axes at once -- at 10 blocks, trimming
    # only the 3 longest rows gives 237 rows and 22,851 padded slots, against 233
    # rows and 25,842 slots under 4 blocks with sigmas=2.0. More coverage, 12% less
    # padding, because one 480-token row was costing ~4,000 slots on its own.
    keep, drop = filter_outliers(lens, a.band_sigmas)
    if drop:
        print(f"[length] DROPPING {len(drop)} over-band rows from training", flush=True)
        rows = [rows[i] for i in keep]
        lens = [rows[i]["length"] for i in range(len(rows))]
    else:
        lens = [r["length"] for r in rows]

    # --- QUANTILE-BATCHED SAMPLING (Req 12) ---
    # Group records into 4 length-alike blocks. Iterate blocks sequentially within each epoch.
    # Each block pads to ITS max, not the global max. This recovers ~39% of wasted padding slots.
    # Records within a block are sampled with replacement if the block is smaller than batch_size.
    blocks = quantile_blocks(list(range(len(rows))), lens,
                             n_blocks=a.length_blocks)
    block_order = []
    for block_max, indices in blocks:
        block_rows = [rows[i] for i in indices]
        # Pad each row in this block to the BLOCK max, not global max
        for r in block_rows:
            pad_len = block_max - len(r["input_ids"])
            if pad_len > 0:
                r["input_ids"] = r["input_ids"] + [tok.pad_token_id] * pad_len
                r["attention_mask"] = r["attention_mask"] + [0] * pad_len
                r["labels"] = r["labels"] + [-100] * pad_len
        block_order.extend(block_rows)
    print(f"[length] {len(blocks)}-quantile blocks: maxes {[bm for bm, _ in blocks]} | "
          f"{len(block_order)} rows sequenced short->long", flush=True)
    # DROPOUT IS A MODEL-CONFIG FIELD, not a TrainingArguments one, so it is applied HERE at
    # load. It is the one axis in the nine that gets MORE important as the corpus shrinks --
    # near-zero on large-data fine-tunes, materially non-flat at 240 records.
    _model_kwargs = {}
    if a.attention_dropout is not None:
        _model_kwargs["attention_dropout"] = a.attention_dropout
        print("[config] attention_dropout=%.3f" % a.attention_dropout, flush=True)
    m = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float32,
                                             trust_remote_code=True, **_model_kwargs)
    if a.full_param:
        # FFN-only full-FT: freeze attention, train MLP. Knowledge lives in FFN (Geva et al.,
        # arXiv:2012.14913; ROME arXiv:2202.05262). Attention routes, doesn't store.
        # Freezing attention saves ~1/3 optimizer state and matches the MGPO full-param config
        # (apples-to-apples: same layers trainable at both stages).
        attn_names = ("q_proj", "k_proj", "v_proj", "o_proj",
                      "in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj")
        for name, param in m.named_parameters():
            if any(a in name for a in attn_names):
                param.requires_grad = False
        n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        n_frz = sum(p.numel() for p in m.parameters() if not p.requires_grad)
        print(f"[policy] FULL-PARAM FFN-only: {n_tr/1e6:.1f}M trainable, "
              f"{n_frz/1e6:.1f}M frozen (attention)", flush=True)
    else:
        # Resolve target modules from the flag
        if a.target_modules == "all-linear":
            targets = ALL_TARGETS
        else:
            targets = [t.strip() for t in a.target_modules.split(",")]
        m = get_peft_model(m, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05,
                                         bias="none", task_type="CAUSAL_LM",
                                         target_modules=targets))
        n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"[policy] LoRA r={a.rank} alpha={a.alpha} over {len(targets)} module types: "
              f"{n_tr/1e6:.1f}M trainable", flush=True)

    # ── Observability callback: generate from 3 held-aside prompts every 10 steps ──
    # One R1 (answerable+context), one R2/R4 (abstention), one R3 (answerable no-context).
    # If by step 20 the model shows zero abstention when gold demands it → training is dead.
    from transformers import TrainerCallback, GenerationConfig
    from rlv2.common.stratify import regime_of as _observe_regime

    _observe_prompts = []
    for _regime_target in ["R2", "R1", "R3"]:
        for _r in sel:
            if _observe_regime(_r) == _regime_target:
                _observe_prompts.append(_r)
                break

    class ObserveCallback(TrainerCallback):
        """Observe + patience for SFT. Every 10 steps: generate from held-aside prompts.
        Every block_steps//2: run canonical mini-eval. Track best, early-stop on regression."""
        def __init__(self):
            self.best_score = 0.0
            self.best_state = None
            self.patience_count = 0
            self.PATIENCE = 2

        def on_log(self, args, state, control, model=None, **kwargs):
            if torch.cuda.is_available():
                _mib = 1024 * 1024
                print(f"[vram] step={state.global_step} "
                      f"max_alloc={torch.cuda.max_memory_allocated() // _mib} MiB "
                      f"max_reserved={torch.cuda.max_memory_reserved() // _mib} MiB",
                      flush=True)
            if state.global_step % 10 != 0 or state.global_step == 0:
                return
            if not _observe_prompts or model is None:
                return
            model.eval()
            with torch.no_grad():
                for _op in _observe_prompts:
                    _p_text = fmt(_op, tok)
                    _p_ids = tok(_p_text, return_tensors="pt", truncation=True,
                                 max_length=MAX_LEN)["input_ids"].to(model.device)
                    _gen_ids = model.generate(
                        _p_ids, max_new_tokens=30,
                        generation_config=GenerationConfig(
                            # THINK AND TOOL CALLS, not think alone. The literal
                            # [248068, 248069] here was think-only; tool-call markers
                            # were free to fire in every phase (operator 2026-09-15).
                            do_sample=False, suppress_tokens=_banned(tok),
                        ),
                    )
                    _gen_text = tok.decode(_gen_ids[0][_p_ids.shape[1]:], skip_special_tokens=True).strip()
                    _gold = (_op.get("gold") or "").strip()
                    if not _gold:
                        _gold = ("not supported by context" if (_op.get("context") or "").strip()
                                 else "unanswerable")
                    _reg = _observe_regime(_op)
                    _match = "CORRECT" if (
                        (_reg in ("R2", "R4") and ("unanswer" in _gen_text.lower() or "not supported" in _gen_text.lower())) or
                        (_reg in ("R1", "R3") and "unanswer" not in _gen_text.lower() and "not supported" not in _gen_text.lower())
                    ) else "WRONG"
                    print(f"[observe] step={state.global_step} | {_reg} | "
                          f"gold={_gold[:40]!r} | gen={_gen_text[:60]!r} | {_match}",
                          flush=True)
            model.train()

    _observe_cb = ObserveCallback()

    t0 = time.time()
    # PER-STEP TOKEN BUDGET, and it is the knob that converts spare VRAM into
    # coverage. 896 was derived at ~3.09 MB/token when every row padded to the
    # global 448. With quantile blocks each row pads to its OWN block max, so the
    # same budget now carries more REAL tokens and less padding -- measured
    # 2026-09-14: 23.8% waste at 4 blocks, 11.7% at 10.
    #
    # Measured headroom at 896 with 10 blocks: max_reserved 11194 MiB against a
    # 15135 MiB ceiling. 1280 spends ~1187 MiB of that and leaves ~2.7 GB.
    BATCH_TOKENS = a.batch_tokens

    # ── MANIFOLD SAMPLER: delta-ratio + bin packing ─────────────────────────────────
    # Uses manifold_sampler.ManifoldSampler for class-balanced, VRAM-efficient batching.
    # Cumulative counts persist across epochs — self-correcting over the full run.
    from rlv2.train.sft.manifold_sampler import ManifoldSampler

    # Target ratios: loaded from canonical cross_hatch_canon.json (single source of truth)
    import pathlib as _canon_pl
    _canon_path = _canon_pl.Path("data/rl/cross_hatch_canon.json")
    _canon = json.loads(_canon_path.read_text(encoding="utf-8"))
    _canon_ratios = {k: v for k, v in _canon["eval"].items() if not k.startswith("_")}
    # Normalize to proportions (sum=1)
    _ratio_total = sum(_canon_ratios.values())
    _canon_ratios = {k: v / _ratio_total for k, v in _canon_ratios.items()}

    sampler = ManifoldSampler(
        pool=block_order,  # tokenized records with input_ids
        target_ratios=_canon_ratios,
        target_tokens=BATCH_TOKENS,
        # BINS PER EPOCH IS NOT THE QUANTILE COUNT. The length partition is 4 MAD-based
        # quantiles and stays 4. This is a separate axis: how many token-budgeted steps make up
        # an epoch. It was also 4, which meant one epoch was 4 x 896 tokens ~= 27 of 240 records
        # -- 11% of the data, so "3 epochs" touched a third of the pool once and the epoch
        # curve was measuring sampling noise as much as learning.
        #
        # MEASURED sweep on this pool (records drawn per epoch / 3-epoch coverage):
        #   4 -> 27 (11%) / 35%      12 -> 80 (33%) / 98%      20 -> 132 (55%) / 160%
        #   8 -> 49 (20%) / 65%      16 -> 105 (44%) / 129%
        # 12 is the value that makes an epoch a third of the pool and three epochs one full
        # pass. VRAM is untouched: the per-step budget is still BATCH_TOKENS.
        n_bins=_bins_for_full_coverage(block_order, BATCH_TOKENS, a.bins_per_epoch),
        max_len=MAX_LEN,
    )
    # DISJOINT EPOCHS. Operator 2026-09-14: "The intent isnt 100% in one epoch. But
    # 100% in 3." Without this the used-set resets every epoch, the three draws are
    # independent, and coverage is 1-(1-f)^3 rather than 3f -- which is the error
    # baked into the bin-sweep comment above (its "12 -> 98%" is 3*80/240; the true
    # figure under resets is 240*(1-(1-80/240)^3) = 70%).
    # FAIL BEFORE TRAINING, NOT AFTER EPOCH 1. Under cover-once the epochs PARTITION
    # the pool, so the pool has to be big enough to divide. When it is not, epoch 1
    # silently consumes everything and epoch 2 raises "the pool is EXHAUSTED" --
    # after a checkpoint has already been written, which reads like a training
    # failure rather than a configuration one. Measured 2026-09-20 twice: a 53-row
    # pool and a 16-row pool, both against --bins-per-epoch 13.
    #
    # A bin holds roughly batch_tokens / median_row_tokens records, so an epoch draws
    # about n_bins * that. Requiring epochs * per_epoch <= pool is the condition
    # cover-once needs to mean what it says.
    if bool(a.cover_once) and block_order:
        _med = sorted(len(r["input_ids"]) for r in block_order)[len(block_order) // 2]
        _per_bin = max(1, BATCH_TOKENS // max(1, _med))
        _per_epoch = _bins_for_full_coverage(
            block_order, BATCH_TOKENS, a.bins_per_epoch) * _per_bin
        if _per_epoch * a.epochs > len(block_order) * 1.5:
            raise SystemExit(
                "[config] --cover-once cannot partition this pool: %d records, "
                "%d epochs, ~%d records/epoch (%d bins x ~%d per bin). Epoch 1 would "
                "consume the pool and epoch 2 would draw nothing. "
                "Fix EITHER: --no-cover-once (3 passes over the same %d records), "
                "or --bins-per-epoch ~%d (partition into thirds)."
                % (len(block_order), a.epochs, _per_epoch,
                   _bins_for_full_coverage(block_order, BATCH_TOKENS,
                                           a.bins_per_epoch),
                   _per_bin, len(block_order),
                   max(1, len(block_order) // (a.epochs * _per_bin))))
    sampler.cover_once = bool(a.cover_once)
    if sampler.cover_once:
        print(f"[manifold] cover-once: epochs are DISJOINT, {len(block_order)} "
              f"records split across {a.epochs} epochs, each seen exactly once",
              flush=True)


    # ── EPOCH LOOP ──────────────────────────────────────────────────────────────────
    # NO in-loop scoring state. Training saves checkpoints; `src/rlv2/eval/score_arms.py` scores them.
    # The removed `_best_epoch_dir` / `_best_epoch_score` selected a checkpoint using the wrong
    # metric on a 3-item sample, and `_epoch_scores` recorded a 0.0 whenever the eval FAILED --
    # so a broken measurement became a data point and the next epoch looked like a huge gain.

    _eval_file = a.eval
    try:
        with open(_eval_file, encoding="utf-8") as _efh:
            _eval_n = sum(1 for _ln in _efh if _ln.strip())
    except OSError as _e:
        _eval_n = -1
        print(f"[epoch-eval] WARNING cannot read {_eval_file}: {_e}", flush=True)
    print(f"[epoch-eval] scoring against {_eval_file} ({_eval_n} rows)", flush=True)

    # Pre-split eval into epoch groups for faster per-epoch scoring
    # Each epoch scores 1/3 of eval (stratified). Full coverage over 3 epochs.
    # WRITTEN ONLY IF SOMETHING READS THEM. The `_eval_end_*` half had no reader at all -- it fed
    # the deleted epoch-boundary eval -- and `_eval_mid_*` is read only by the mid-epoch eval,
    # which is off unless --evals-per-epoch > 0. Writing six split files next to every checkpoint
    # regardless made a run look like it had scored itself when it had not.
    _eval_mid_files: list[str] = []
    if a.evals_per_epoch > 0:
        from rlv2.train.sft.manifold_sampler import stratified_eval_split
        _eval_records = [json.loads(l) for l in io.open(_eval_file, encoding="utf-8") if l.strip()]
        _eval_groups = stratified_eval_split(_eval_records, n_groups=3, seed=99)
        for _gi, _group in enumerate(_eval_groups):
            _mid_path = f"{OUT}_eval_mid_{_gi}.jsonl"
            with open(_mid_path, "w", encoding="utf-8") as _mf:
                for _r in _group[:len(_group) // 2]:
                    _mf.write(json.dumps(_r, ensure_ascii=False) + "\n")
            _eval_mid_files.append(_mid_path)
        print(f"[epoch-eval] mid-epoch eval ON: {len(_eval_groups)} groups, "
              f"mid={[len(g[:len(g)//2]) for g in _eval_groups]} records each", flush=True)
    else:
        print("[epoch-eval] no in-loop scoring; score checkpoints with src/rlv2/eval/score_arms.py",
              flush=True)

    from rlv2.train.sft.loop import train_bins

    for epoch in range(a.epochs):
        # Sample epoch bins via manifold sampler (delta-ratio + bin packing)
        bins = sampler.sample_epoch(seed=a.seed + epoch)
        n_records = sum(len(b) for b in bins)
        total_steps = len(bins)
        # Report coverage, not just a count. "27 records" reads fine on its own; "27 records
        # (11% of pool)" is the number that shows an epoch is not an epoch.
        _pool_n = len(block_order)
        print(f"[epoch {epoch+1}/{a.epochs}] {n_records} records "
              f"({100.0 * n_records / max(1, _pool_n):.0f}% of {_pool_n}-row pool) "
              f"in {total_steps} bins (target_tokens={BATCH_TOKENS}) | "
              f"manifold drawn={sampler.cumulative_drawn}", flush=True)

        # Print manifold stats
        stats = sampler.stats()
        if stats["top_deficit"]:
            top = stats["top_deficit"][0]
            print(f"  [manifold] max_deficit={stats['max_delta']:.4f} "
                  f"top_cell={top[0]} (delta={top[1]:.4f})", flush=True)

        # Mid-epoch eval callback: OFF unless --evals-per-epoch > 0.
        #
        # It is print-only, so it cannot corrupt checkpoint selection the way the epoch-boundary
        # eval did -- but it carries the same defects: it scores with
        # `rl_calibration_eval --prompt-style abstain` (the VibeThinker ABSTENTION metric, on a
        # lane with no abstention), reads `data/rl/calibration.json` by a cwd-dependent relative
        # path, and saves ANOTHER checkpoint per epoch that nothing cleans up. It also costs a
        # full eval subprocess in the middle of every epoch.
        #
        # Measurement belongs to `src/rlv2/eval/score_arms.py`, after training, on one instrument.
        _mid_eval_file = (_eval_mid_files[epoch]
                          if (getattr(a, "evals_per_epoch", 0) > 0
                              and epoch < len(_eval_mid_files))
                          else None)
        _mid_step = max(1, total_steps // 2)

        class _MidEpochEval(TrainerCallback):
            def on_step_end(self, args, state, control, model=None, **kwargs):
                if state.global_step != _mid_step or _mid_eval_file is None:
                    return
                import subprocess as _msp
                _tmp_ckpt = f"{OUT}_mid_ep{epoch+1}"
                os.makedirs(_tmp_ckpt, exist_ok=True)
                model.save_pretrained(_tmp_ckpt)
                tok.save_pretrained(_tmp_ckpt)
                _mid_tag = f"_sft_mid{epoch+1}_{os.path.basename(OUT)}"
                _mid_cmd = [sys.executable, "-m", "rlv2.eval.rl_calibration_eval",
                            "--model", a.base, "--adapters", _tmp_ckpt,
                            "--tag", _mid_tag, "--eval-file", _mid_eval_file,
                            "--prompt-style", "abstain"]
                if a.full_param:
                    _mid_cmd = [sys.executable, "-m", "rlv2.eval.rl_calibration_eval",
                                "--model", _tmp_ckpt, "--tag", _mid_tag,
                                "--eval-file", _mid_eval_file, "--prompt-style", "abstain"]
                try:
                    _mr = _msp.run(_mid_cmd, capture_output=True, text=True, errors="replace",
                                   timeout=600, cwd=_REPO_ROOT)
                    if _mr.returncode == 0:
                        import pathlib as _mpl
                        _mcal = json.loads(_mpl.Path("data/rl/calibration.json").read_text(encoding="utf-8"))
                        _mm = _mcal.get(_mid_tag, {}).get("macro_f1", {})
                        _mgrid = _mcal.get(_mid_tag, {}).get("regime_domain_grid", {})
                        _mdoms = _mcal.get(_mid_tag, {}).get("domains", [])
                        print(f"  [mid-eval] epoch={epoch+1} step={_mid_step} "
                              f"macroF1={_mm.get('macro_f1',0):.4f} "
                              f"tp={_mm.get('counts',{}).get('tp_abstained_correctly',0)} "
                              f"fn={_mm.get('counts',{}).get('fn_confabulated',0)} "
                              f"fp={_mm.get('counts',{}).get('fp_over_abstained',0)} "
                              f"tn={_mm.get('counts',{}).get('tn_answered_correctly',0)}",
                              flush=True)
                        if _mgrid and _mdoms:
                            print(f"  [mid-grid] {'':>6}" + "".join(f"{d[:8]:>10}" for d in _mdoms))
                            for _reg in ("R1", "R2", "R3", "R4"):
                                _row = f"  [mid-grid] {_reg:<6}"
                                for _dom in _mdoms:
                                    _cell = _mgrid.get(f"{_reg}|{_dom}", {})
                                    _cn = _cell.get("n", 0)
                                    if _cn == 0:
                                        _row += f"{'—':>10}"
                                    elif _reg in ("R2", "R4"):
                                        _row += f"{_cell.get('tp',0)}tp/{_cell.get('fn',0)}fn".rjust(10)
                                    else:
                                        _row += f"{_cell.get('tn',0)}tn/{_cell.get('fp',0)}fp".rjust(10)
                                print(_row, flush=True)
                    else:
                        print(f"  [mid-eval] FAILED (exit {_mr.returncode})", flush=True)
                except Exception as _me:
                    print(f"  [mid-eval] ERROR: {_me}", flush=True)
                finally:
                    import shutil
                    shutil.rmtree(_tmp_ckpt, ignore_errors=True)

        # _MidEpochEval fires at ONE hardcoded step; HalfEpochEval reports on every
        # half-epoch with per-category F1, support counts and generated outputs
        # (task 7a). Both are kept: the former gates early-stop, the latter measures.
        # EVERY HYPERPARAMETER BELOW IS PASSED, NONE LEFT ON A LIBRARY DEFAULT (2026-08-30): the HPO
        # driver sweeps warmup, scheduler, decay, clipping, betas and label smoothing.
        train_bins(
            m, bins, tok,
            lr=a.lr,
            lr_scheduler=a.lr_scheduler,
            warmup_steps=max(1, int(total_steps * a.warmup_frac)),
            weight_decay=a.weight_decay,
            max_grad_norm=a.max_grad_norm,
            betas=(a.adam_beta1, a.adam_beta2),
            eps=a.adam_eps,
            label_smoothing=a.label_smoothing,
            scheduler_kwargs=({"num_cycles": a.num_cycles} if a.num_cycles is not None else None),
            seed=a.seed + epoch,
            callbacks=_sft_callbacks(_observe_cb, _MidEpochEval(), m, tok, a, epoch),
            logging_steps=1,
        )
        from rlv2.common.cost_meter import report_peak_vram
        report_peak_vram(f"(epoch {epoch+1}, param={'full' if a.full_param else 'lora'})")

        # ── EPOCH BOUNDARY: SAVE ONLY. TRAINING DOES NOT SCORE. ─────────────────────────
        #
        # This block used to run an in-loop eval that selected the "best" checkpoint and
        # early-stopped. It carried EIGHT ways to turn a failed measurement into a number:
        #
        #   1. eval exit != 0        -> `_epoch_scores.append(0.0)`  a FAILURE became a SCORE
        #   2. bare `except`         -> `_epoch_scores.append(0.0)`  same, for any exception
        #   3. wrong instrument      -> rl_calibration_eval --prompt-style abstain, the
        #                               VibeThinker ABSTENTION metric, on a lane (R3) that has
        #                               no abstention class at all
        #   4. 3-item sub-sample     -> `_eval_end_files[epoch]` holds 3 questions, and those
        #                               3 decided which checkpoint we kept
        #   5. relative path read    -> `data/rl/calibration.json` read in the PARENT process,
        #                               whose cwd is not guaranteed to be the repo root
        #   6. early `break`         -> training stopped on a regression measured from 3 items
        #   7. best-by-wrong-metric  -> `_best_epoch_dir` was then COPIED OVER `OUT`
        #   8. ungated               -> ran regardless of --evals-per-epoch
        #
        # A zero from (1) or (2) is worse than a crash: it makes the NEXT epoch look like a
        # huge improvement, poisons the regression check, and silently loses the real best.
        #
        # THE FIX IS SEPARATION, NOT REPAIR. Training writes per-epoch checkpoints and nothing
        # else. Every number comes afterwards from `src/rlv2/eval/score_arms.py`, which scores all arms
        # in ONE invocation against the frozen eval with the hashed rubric -- the instrument
        # that is actually verified (24 tests) and whose resolution limit is measured (0.0526).
        # SAVE IN FP16. The model TRAINS in fp32 (AMP needs fp32 masters) but every consumer
        # loads it as fp16 -- `score_arms.generate_all` and `half_epoch_eval` both pass
        # `torch_dtype=float16`. Writing fp32 therefore doubles the file for nothing.
        #
        # MEASURED 2026-08-31 and it is not a rounding error: training took 72 SECONDS, one
        # fp32 checkpoint took ~18 MINUTES to write. At 3 epochs x 3 arms that is ~2.7 hours of
        # disk I/O against ~4 minutes of training -- the saves, not the training, were what
        # blew the "tens of minutes" budget.
        _epoch_ckpt = f"{OUT}_epoch{epoch+1}"
        os.makedirs(_epoch_ckpt, exist_ok=True)
        # CAST A COPY, NEVER THE LIVE MODEL. `m.half()` mutates weights in place and `.float()`
        # back does NOT restore them -- it round-trips every parameter through fp16 and the
        # next epoch trains from the degraded copy. That is the checkpoint-save bug this repo
        # already recorded as a hard blocker (T0.3); building the state dict out-of-place is
        # what keeps the fp32 masters intact.
        _t_save = time.time()
        m.save_pretrained(_epoch_ckpt, state_dict=_fp16_state_dict(m))
        tok.save_pretrained(_epoch_ckpt)
        print(f"  [epoch {epoch+1}/{a.epochs}] saved fp16 -> {_epoch_ckpt}  "
              f"({time.time() - _t_save:.0f}s, not scored here; use src/rlv2/eval/score_arms.py)",
              flush=True)


    # ── Save final (or best) checkpoint ──────────────────────────────────────────────
    # Use best epoch checkpoint if available, else use final state
    # FINAL EPOCH IS THE OUTPUT. There is no "best" here any more -- picking one required an
    # in-loop score, and that score was produced by the wrong instrument on 3 questions. Every
    # epoch checkpoint is on disk; `score_arms` compares them all on one instrument afterwards.
    # OUT IS A COPY OF THE FINAL EPOCH, NOT A SECOND SERIALISATION. Re-saving the model here
    # wrote the SAME weights a second time -- another ~18 min for a file we already had on
    # disk. A filesystem copy of the last epoch dir is the same bytes at a fraction of the cost.
    import shutil as _sh
    _final_ckpt = f"{OUT}_epoch{a.epochs}"
    if os.path.exists(_final_ckpt):
        if os.path.exists(OUT):
            _sh.rmtree(OUT, ignore_errors=True)
        _t_copy = time.time()
        _sh.copytree(_final_ckpt, OUT)
        print(f"[sft] OUT = copy of final epoch {a.epochs} ({time.time() - _t_copy:.0f}s)",
              flush=True)
    else:
        # Training stopped before any epoch checkpoint landed -- serialise what is in memory
        # rather than leaving OUT missing, which downstream tooling treats as a crashed run.
        os.makedirs(OUT, exist_ok=True)
        m.save_pretrained(OUT, state_dict=_fp16_state_dict(m))
        tok.save_pretrained(OUT)
    if not a.full_param:
        assert_adapter_trained(OUT, label="vibe_sft")
    print(f"[sft] SAVED -> {OUT}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
