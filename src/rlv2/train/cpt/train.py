"""src/rlv2/train/cpt/train.py — Continued Pretraining with gradual annealing and Box-Cox stratified corpus sampling.

Purpose: Deposit domain knowledge into the base model weights via NTP (next-token prediction)
on a stratified corpus of book chapters, EntiGraph rephrases, and skill text. Gradual annealing
mixes general replay data with domain text to prevent catastrophic forgetting.

Preconditions:
  - Corpus sources exist (book extracts, cpt_corpus_full.txt, ml_cpt.bible.txt)
  - Base model (skills_model or specified via --base) loadable
  - 16GB VRAM available (full-param or QLoRA, depending on --full-param)

Failure modes:
  - Empty corpus after filtering → raises with source list
  - VRAM OOM → use adamw_8bit + gradient_checkpointing (auto-configured for full-param)
  - Post-trainability regression → eval shows macroF1 dropped vs base

References:
  - Cameron Wolfe midtraining notes: gradual anneal 80/20 → 40/60 general/domain
  - Raschka Ch5: NTP cross-entropy, learning rate warmup + cosine decay
  - Full pipeline spec: evaluate by POST-TRAINABILITY (downstream SFT/RL performance)
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

# MUST precede the first CUDA allocation. The dominant tensor in this run is the
# cross-entropy logits (248,320 vocab x 512 tokens), and at 16 GB the allocator fails on
# FRAGMENTATION rather than true exhaustion -- the OOM that killed the 6-epoch trial had
# 907 MB reserved-but-unallocated while asking for 1.89 GB. expandable_segments lets the
# allocator grow a segment instead of demanding one contiguous block.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, ".")

# ── Corpus sources ────────────────────────────────────────────────────────────────
# Each source is a (label, path, type) tuple. type = "text" (raw NTP) or "jsonl" (has "text" field).
# Box-Cox allocation is over chunk counts per source.
CORPUS_SOURCES = [
    ("entigraph_skills", "data/domain/cpt_corpus_full.txt", "text"),
    ("ml_bible", "data/domain/ml_cpt.bible.txt", "text"),
    ("book_ch5_pretraining", "references/book_extracts/llm_from_scratch__ch5_pretraining.txt", "text"),
    ("book_ch7_sft", "references/book_extracts/llm_from_scratch__ch7_instruction_finetuning.txt", "text"),
    ("book_appd_training", "references/book_extracts/llm_from_scratch__appendix_d_training_loop.txt", "text"),
    ("book_appe_lora", "references/book_extracts/llm_from_scratch__appendix_e_lora.txt", "text"),
    ("book_ch5_refinement", "references/book_extracts/reasoning_model__ch5_self_refinement.txt", "text"),
    ("book_ch6_grpo", "references/book_extracts/reasoning_model__ch6_grpo_from_scratch.txt", "text"),
    ("book_ch7_improving_grpo", "references/book_extracts/reasoning_model__ch7_improving_grpo.txt", "text"),
    ("book_ch8_distillation", "references/book_extracts/reasoning_model__ch8_distillation.txt", "text"),
]

# General replay: a portion of the training data that preserves general capability.
# Uses the SFT training data as "general" (it's the closest thing to diverse QA we have).
GENERAL_REPLAY_PATH = "data/rl/rl_mix.jsonl"

# Defaults
BASE_MODEL = "skills_model"
CHUNK_SIZE = 512       # tokens per training chunk (context window during CPT)
LR_PEAK = 1e-4        # CPT LoRA lr (Biderman IFT-general band; full-param uses 1e-5)
LR_FULL = 1e-5        # Full-param CPT lr (10x lower than LoRA per the established rule)
WARMUP_FRAC = 0.1     # 10% of steps for linear warmup
MIN_LR_FRAC = 0.1     # cosine decays to 10% of peak
EPOCHS = 3
LORA_R = 64           # CPT uses larger rank than RL (more knowledge to deposit; config.py anchor)
LORA_ALPHA = 32       # alpha/r = 0.5 — conservative update scale for knowledge deposit


def load_text_corpus(path: str) -> str:
    """Load a text file as a single string for chunking."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_jsonl_texts(path: str, field: str = "text") -> list[str]:
    """Load texts from a JSONL file (one JSON object per line with a text field)."""
    texts = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get(field, "")
            if t:
                texts.append(t)
    return texts


def chunk_text(text: str, tokenizer, chunk_size: int = 512) -> list[list[int]]:
    """Tokenize text and split into non-overlapping chunks of chunk_size tokens.

    Require: text non-empty, chunk_size > 0.
    Guarantee: each returned chunk has exactly chunk_size tokens (last partial chunk dropped).
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(token_ids) - chunk_size + 1, chunk_size):
        chunks.append(token_ids[i:i + chunk_size])
    return chunks


def pack_atoms(texts: list[str], tokenizer, budget: int = 512,
               merge: bool = True) -> tuple[list[list[int]], dict]:
    """Pack whole records into sequences of at most `budget` tokens, never splitting one.

    Each JSONL record is an ATOM. The skill chunks arriving here were already segmented on
    paragraph boundaries by `semantic_chunk.py` (81 records, median ~5 paragraphs, each
    carrying its own `tokens` count). The previous path joined them all with "\n\n" and
    re-sliced every 512 tokens, which cut across record boundaries and mid-sentence --
    throwing away the segmentation and mixing unrelated skills inside one training
    sequence. Greedy accumulation to the budget keeps each record intact.

    An atom LONGER than the budget is emitted alone and truncated, and counted in the
    returned stats: splitting it mid-paragraph to satisfy the budget would destroy the
    exact boundary this function exists to preserve, and silently dropping it would hide
    real corpus content (53.8% of the RedPajama sidecar is over-window).

    Require: budget > 0.
    Guarantee: every atom appears in exactly one output sequence, in input order; every
               output sequence is <= budget tokens.
    """
    if budget <= 0:
        raise ValueError("budget must be positive, got %d" % budget)

    atoms = [tokenizer.encode(t, add_special_tokens=False) for t in texts if t and t.strip()]
    stats = {"atoms": len(atoms), "over_budget": 0, "truncated_tokens": 0}

    out: list[list[int]] = []
    cur: list[int] = []
    placed = 0
    for a in atoms:
        if len(a) > budget:
            # FAIL CLOSED. The length gate in build_cpt_mix_r3.py caps every record at 512
            # BEFORE it reaches training -- skills get chunked, everything else dropped, and
            # the receipt prints what happened. An over-budget record arriving here means
            # that gate was bypassed, and quietly truncating it would restore exactly the
            # silent mid-sentence cut the gate exists to prevent.
            raise ValueError(
                "record of %d tokens exceeds the %d-token budget. The length gate should "
                "have capped or dropped it upstream -- rebuild the corpus with "
                "`python -m rlv2 data cpt-mix` rather than truncating here." % (len(a), budget))
        if not merge:
            out.append(a)                        # one record, one sample
            placed += 1
            continue
        if len(cur) + len(a) > budget:
            out.append(cur)
            cur = []
        cur.extend(a)
        placed += 1
    if cur:
        out.append(cur)

    # Conservation: an atom silently dropped by a packing bug is invisible in the loss
    # curve and shows up only as a model that never learned a skill it was shown.
    assert placed == len(atoms), (
        "atom conservation failed: %d atoms in, %d placed" % (len(atoms), placed))

    lens = sorted(len(x) for x in out)
    if lens:
        stats.update(
            records=len(out),
            p50=lens[len(lens) // 2],
            p95=lens[int(0.95 * (len(lens) - 1))],
            max=lens[-1],
            fill=sum(lens) / (len(lens) * budget),
        )
    return out, stats


def derive_token_budget(vocab_size: int, n_params: int = 0, full_param: bool = False,
                        safety: float = 0.12) -> tuple[int, dict]:
    """Tokens per batch that the cross-entropy logits can afford, from MEASURED VRAM.

    A TOKEN budget rather than a record count, because that is what keeps tokens-per-step
    roughly constant across length blocks. Only `n_records` then differs between blocks,
    so the gradient-noise scale does not co-vary with document length.

    `safety=0.12` is BRACKETED BY TWO MEASUREMENTS on this card, not chosen:

        batch 2 x 512 = 1,024 tok/step   ran clean past step 10
        batch 5 x 512 = 2,918 tok/step   OOM, asked 2.37 GiB with 745 MiB free

    The analytic terms below (weights + grads + optimizer state) account for ~10.1 GB, but
    10.85 GiB was actually resident at the failure -- a gap of activations, gradient-
    checkpoint buffers and allocator slack that no closed form here predicts, so the
    fraction is fitted to the observed bracket rather than derived. 0.12 of an 8.28 GB
    headroom is ~1,000 tokens, the side of the bracket that works.

    RE-DERIVE IT if the model, the card, or gradient checkpointing changes -- a fitted
    constant carried to different hardware is exactly the magic number this file avoids
    everywhere else.

    Require: vocab_size > 0.
    Guarantee: a positive token budget plus every term behind it.
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size=%r" % vocab_size)

    per_token = vocab_size * 4                  # fp32 logits, one token
    info = {"per_token_bytes": per_token, "measured": False}
    if not torch.cuda.is_available():
        info["why"] = "no CUDA"
        return 2048, info

    total = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated(0)
    # Gradients (n x 4 B) and adamw_8bit state (n x 2 B) do not exist until the first
    # backward pass, so they are added analytically from the REAL parameter count.
    pending = int(n_params) * (4 + 2) if (full_param and n_params) else 0
    cap = int(total * 0.92)                     # driver + CUDA context reserve
    headroom = max(0, cap - allocated - pending)
    budget = max(256, int((headroom * safety) // per_token))
    info.update(measured=True, total=total, allocated=allocated, pending=pending,
                headroom=headroom, budget=budget, safety=safety)
    return budget, info


def derive_batch_size(max_len: int, vocab_size: int, requested: int,
                      n_params: int = 0, full_param: bool = False,
                      safety: float = 0.75) -> tuple[int, dict]:
    """Largest per-device batch whose LOGITS still fit in measured free VRAM.

    MEASURED, not guessed. Call this AFTER the model and optimizer are on the card, so
    `memory_allocated` reflects the real static footprint rather than an estimate.

    The dominant transient in this run is the cross-entropy logits tensor, and it is big
    because the vocabulary is: 248,320 entries. At fp32 (the loss upcasts) one sequence of
    512 tokens costs 512 x 248,320 x 4 B = 509 MB, so batch 4 asks for 2.03 GB in a single
    allocation. The static model + adamw_8bit state measured 12.80 GiB of a 14.78 GiB
    ceiling, leaving under 2 GB -- which is why batch 4 OOM'd at step 4 while batch 2 ran.

    `safety` holds back a quarter of the headroom for activations and allocator slack; the
    logits are the largest single tensor but they are not the only one.

    Require: max_len > 0, vocab_size > 0, requested >= 1.
    Guarantee: returns a batch size in [1, requested] plus the numbers behind it.
    """
    if max_len <= 0 or vocab_size <= 0 or requested < 1:
        raise ValueError("max_len=%r vocab=%r requested=%r" % (max_len, vocab_size, requested))

    per_seq = max_len * vocab_size * 4          # fp32 logits for one sequence
    info = {"per_seq_bytes": per_seq, "requested": requested, "measured": False}

    if not torch.cuda.is_available():
        info["why"] = "no CUDA — leaving the requested batch alone"
        return requested, info

    total = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated(0)

    # ALLOCATED IS NOT THE WHOLE STATIC FOOTPRINT, and reading it alone is how the first
    # version of this returned "fits 23" for a batch that OOMs at 4. At the point a batch
    # must be chosen, gradients and optimizer state DO NOT EXIST YET -- they are allocated
    # lazily on the first backward pass. So they are added analytically, from the real
    # parameter count of the real model, not guessed:
    #
    #   gradients        n_params x 4 B   (fp32 masters -> fp32 grads)
    #   adamw_8bit state n_params x 2 B   (m and v, one byte each)
    #
    # For 1006.7M params full-param that is ~6.0 GB on top of the ~4.0 GB of weights,
    # which reconciles with the 12.80 GiB observed at the step-4 OOM.
    pending = int(n_params) * (4 + 2) if (full_param and n_params) else 0

    cap = int(total * 0.92)                     # 0.92: driver + CUDA context reserve
    headroom = max(0, cap - allocated - pending)
    fits = int((headroom * safety) // per_seq)
    chosen = max(1, min(requested, fits))
    info.update(measured=True, total=total, allocated=allocated, pending=pending,
                headroom=headroom, fits=fits, chosen=chosen)
    return chosen, info


def build_corpus(tokenizer, chunk_size: int = 512, sources: list | None = None,
                 pack_records: bool = False) -> dict[str, list[list[int]]]:
    """Build the CPT corpus: per-source chunks for Box-Cox allocation.

    Returns: {source_label: [chunk_token_ids, ...]}
    """
    if sources is None:
        sources = CORPUS_SOURCES

    corpus: dict[str, list[list[int]]] = {}
    for label, path, src_type in sources:
        if not os.path.exists(path):
            print(f"  [WARN] source missing: {path} — skipping {label}", flush=True)
            continue

        if src_type == "text":
            text = load_text_corpus(path)
            chunks = chunk_text(text, tokenizer, chunk_size)
        elif src_type == "jsonl":
            # RECORD-AWARE. Do not concatenate-and-reslice: these records are already
            # semantic atoms, and joining them makes one training sequence span several
            # unrelated skills.
            texts = load_jsonl_texts(path)
            if pack_records:
                chunks, pack_stats = pack_atoms(texts, tokenizer, chunk_size)
            else:
                # ONE RECORD = ONE SAMPLE. Packing merges short records until they reach
                # the budget, which drives every sample back to ~512 tokens -- and that
                # makes the quantile blocks downstream identical (all max=512, batch=1),
                # turning the length binning into a no-op. The length gate already
                # guarantees every record is <= the cap, so there is nothing to pack FOR:
                # the padding it would save is recovered instead by padding to each
                # block's own max. Measured: packed p50 414 / all blocks batch 1, versus
                # unpacked p50 194 / blocks batch 6, 5, 2, 1.
                chunks, pack_stats = pack_atoms(texts, tokenizer, chunk_size, merge=False)
            print("  [%s] packed %d atoms -> %d records | p50=%d p95=%d max=%d | fill=%.0f%%"
                  % (label, pack_stats["atoms"], pack_stats.get("records", 0),
                     pack_stats.get("p50", 0), pack_stats.get("p95", 0),
                     pack_stats.get("max", 0), 100 * pack_stats.get("fill", 0)),
                  flush=True)
            if pack_stats["over_budget"]:
                print("  [%s] %d atom(s) exceeded the %d-token budget and were truncated "
                      "(%d tokens lost). Reported, not swallowed."
                      % (label, pack_stats["over_budget"], chunk_size,
                         pack_stats["truncated_tokens"]), flush=True)
        else:
            raise ValueError(f"Unknown source type {src_type!r} for {label}")

        if chunks:
            corpus[label] = chunks
            print(f"  [{label}] {path} → {len(chunks)} chunks ({len(chunks)*chunk_size} tokens)",
                  flush=True)
        else:
            print(f"  [WARN] {label}: no chunks produced (file too short?)", flush=True)

    if not corpus:
        raise ValueError(
            f"CPT corpus is empty after loading {len(sources)} sources. "
            f"Check paths: {[s[1] for s in sources]}"
        )
    return corpus


def allocate_chunks(corpus: dict[str, list[list[int]]], n_total: int, lam: float = 0.5) -> dict[str, int]:
    """Box-Cox stratified allocation across sources.

    Smaller sources get boosted representation (lam=0.5 blend of uniform and CDF-proportional).
    """
    from rlv2.data.pull_datasets import calculate_allocations

    labels = sorted(corpus.keys())
    sizes = [len(corpus[lbl]) for lbl in labels]
    allocs = calculate_allocations(sizes, n_total, lam=lam)
    result = {lbl: alloc for lbl, alloc in zip(labels, allocs)}

    print(f"  [allocation] n={n_total} lam={lam}:", flush=True)
    for lbl in labels:
        raw_pct = 100 * len(corpus[lbl]) / sum(sizes)
        draw_pct = 100 * result[lbl] / max(n_total, 1)
        print(f"    {lbl:<30} raw={len(corpus[lbl]):>5} ({raw_pct:>5.1f}%) → "
              f"draw={result[lbl]:>5} ({draw_pct:>5.1f}%)", flush=True)
    return result


def sample_epoch(
    corpus: dict[str, list[list[int]]],
    allocation: dict[str, int],
    general_chunks: list[list[int]],
    anneal_frac: float,
    rng: random.Random,
) -> list[list[int]]:
    """Sample one epoch's training data with annealing between general and domain.

    anneal_frac: fraction of domain data (e.g. 0.2 at start → 0.6 at end).
    The rest is general replay.

    Returns: list of token_id chunks in shuffled order.
    """
    # Domain chunks: sample per allocation
    domain_chunks: list[list[int]] = []
    for label, n in allocation.items():
        pool = corpus[label]
        if n <= len(pool):
            domain_chunks.extend(rng.sample(pool, n))
        else:
            # Oversample with replacement for rare sources
            domain_chunks.extend(pool)
            domain_chunks.extend(rng.choices(pool, k=n - len(pool)))

    # The REAL split is computed below from `total_chunks`. A dead first attempt lived
    # here until 2026-08-28: it computed `len(domain) * anneal / anneal`, which is just
    # `len(domain)` for any non-zero anneal, and its result was discarded. It cost a
    # killed run at 00:15 because it reads as 'annealing does nothing'.
    # Actually: total = domain_sample + general_sample
    # domain contributes anneal_frac of total, general contributes (1 - anneal_frac)
    total_chunks = len(domain_chunks)
    n_domain_keep = int(total_chunks * anneal_frac)
    n_general_add = total_chunks - n_domain_keep

    domain_sample = rng.sample(domain_chunks, min(n_domain_keep, len(domain_chunks))) if n_domain_keep <= len(domain_chunks) else domain_chunks[:n_domain_keep]

    if n_general_add > 0 and general_chunks:
        general_sample = rng.choices(general_chunks, k=n_general_add) if n_general_add > len(general_chunks) else rng.sample(general_chunks, n_general_add)
    else:
        general_sample = []

    epoch_data = domain_sample + general_sample
    rng.shuffle(epoch_data)
    return epoch_data


def main():
    ap = argparse.ArgumentParser(description="Continued Pretraining with gradual annealing")
    ap.add_argument("--warmup-frac", type=float, default=WARMUP_FRAC,
                    help="fraction of steps spent in linear warmup (default 0.1). "
                         "Converted to warmup_steps -- warmup_ratio is deprecated (§7.3/12)")
    ap.add_argument("--lr-scheduler", default="cosine",
                    choices=["cosine", "linear", "constant_with_warmup",
                             "cosine_with_restarts", "cosine_with_min_lr",
                             "warmup_stable_decay", "inverse_sqrt", "polynomial",
                             "constant"],
                    help="decay SHAPE after warmup (default cosine, the proven setting)")
    ap.add_argument("--weight-decay", type=float, default=0.1,
                    help="AdamW weight decay (default 0.1)")
    ap.add_argument("--evals-per-epoch", type=int, default=2,
                    help="F1 readings per epoch on the frozen eval (default 2 = every half "
                         "epoch). A short HPO trial can raise this to keep enough points "
                         "to read a shape from")
    ap.add_argument("--pack-records", action="store_true", default=False,
                    help="merge short records together up to the token budget. OFF by "
                         "default: packing drives every sample to ~512 tokens, which "
                         "collapses the quantile length blocks into one and makes dynamic "
                         "batching a no-op")
    ap.add_argument("--auto-batch", action="store_true", default=True,
                    help="derive per-device batch from MEASURED free VRAM and the longest "
                         "record, holding batch x grad_accum constant. On by default: a "
                         "guessed constant batch is what OOM'd this run three times")
    ap.add_argument("--no-auto-batch", dest="auto_batch", action="store_false",
                    help="use --batch-size verbatim, e.g. to measure an OOM ceiling")
    ap.add_argument("--max-grad-norm", type=float, default=1.0,
                    help="gradient clipping (default 1.0, the HF default). 0.0 DISABLES "
                         "clipping -- that was an unexamined override removed under T0.7 "
                         "and must not come back via a sweep grid: without it one spiked "
                         "batch destroys an fp16 + adamw_8bit run, and clipping is what "
                         "emits grad_norm into the log at all")
    ap.add_argument("--attention-dropout", type=float, default=None,
                    help="attention dropout, applied to the model config at load. This "
                         "architecture (qwen3_5_text) has NO hidden_dropout field -- "
                         "attention is the only dropout knob that exists here. None "
                         "leaves the config untouched")
    ap.add_argument("--adam-beta1", type=float, default=0.9,
                    help="AdamW beta1 (HF default 0.9)")
    ap.add_argument("--adam-beta2", type=float, default=0.999,
                    help="AdamW beta2 (HF default 0.999)")
    ap.add_argument("--adam-eps", type=float, default=1e-8,
                    help="AdamW epsilon (HF default 1e-8)")
    ap.add_argument("--num-cycles", type=float, default=None,
                    help="scheduler shape: cosine cycles. Only for cosine / "
                         "cosine_with_restarts; forwarded only when set")
    ap.add_argument("--min-lr-rate", type=float, default=None,
                    help="scheduler shape: floor as a FRACTION of peak lr. Only for the "
                         "cosine_with_min_lr family; forwarded only when set")
    ap.add_argument("--corpus", default=None,
                    help="domain corpus jsonl (records with a 'text' field) that REPLACES "
                         "CORPUS_SOURCES. CORPUS_SOURCES is the pre-scope-cut draw -- book "
                         "chapters and a 300-skill EntiGraph, none of it trees-ensembles. "
                         "Pass data/rl/cpt_mix_r3.jsonl for the R3 experiment.")
    ap.add_argument("--general-replay", default=None,
                    help="override GENERAL_REPLAY_PATH; annealing mixes this against the "
                         "domain corpus, so the domain corpus must NOT contain replay too")
    ap.add_argument("--base", default=BASE_MODEL,
                    help="base model to continue pretraining (default: skills_model)")
    ap.add_argument("--out", default="harness_runs/cpt",
                    help="output directory for CPT checkpoint")
    ap.add_argument("--epochs", type=int, default=EPOCHS, choices=[1, 2, 3],
                    help="training epochs (default: 3)")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                    help="tokens per training chunk (default: 512)")
    ap.add_argument("--lr", type=float, default=None,
                    help="peak learning rate (default: 1e-4 for LoRA, 1e-5 for full-param)")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="micro batch size (default: 4)")
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="gradient accumulation steps (default: 4, effective batch=16)")
    ap.add_argument("--full-param", action="store_true",
                    help="full-parameter CPT (requires adamw_8bit); default is LoRA r=64")
    ap.add_argument("--rank", type=int, default=LORA_R,
                    help="LoRA rank for CPT (default: 64, larger than RL to deposit more knowledge)")
    ap.add_argument("--alpha", type=int, default=LORA_ALPHA,
                    help="LoRA alpha (default: 32; update_scale = alpha/r = 0.5)")
    ap.add_argument("--anneal-start", type=float, default=0.2,
                    help="domain fraction at epoch 1 (default: 0.2 = 80%% general + 20%% domain)")
    ap.add_argument("--anneal-end", type=float, default=0.6,
                    help="domain fraction at final epoch (default: 0.6 = 40%% general + 60%% domain)")
    ap.add_argument("--lam", type=float, default=0.5,
                    help="Box-Cox lambda for corpus allocation (default: 0.5)")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override total steps (for smoke tests)")
    args = ap.parse_args()

    # Resolve learning rate
    if args.lr is None:
        args.lr = LR_FULL if args.full_param else LR_PEAK
    # Update scale reporting
    if not args.full_param:
        update_scale = args.lr * args.alpha / args.rank
        print(f"[config] LoRA r={args.rank} alpha={args.alpha} lr={args.lr} "
              f"→ update_scale={update_scale:.2e}", flush=True)
    else:
        print(f"[config] full-param lr={args.lr}", flush=True)

    print(f"[config] epochs={args.epochs} chunk_size={args.chunk_size} "
          f"anneal={args.anneal_start:.0%}→{args.anneal_end:.0%} domain fraction",
          flush=True)

    # Load tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Build corpus
    print("\n[corpus] loading sources...", flush=True)
    # --corpus REPLACES CORPUS_SOURCES. The default list is the pre-scope-cut draw (book
    # chapters + a 300-skill EntiGraph); none of it is trees-ensembles, so training on it
    # would deposit the wrong domain entirely.
    _sources = [("r3_mix", args.corpus, "jsonl")] if args.corpus else None
    if args.general_replay:
        global GENERAL_REPLAY_PATH
        GENERAL_REPLAY_PATH = args.general_replay
    corpus = build_corpus(tok, chunk_size=args.chunk_size, sources=_sources,
                          pack_records=args.pack_records)
    total_domain_chunks = sum(len(v) for v in corpus.values())
    total_domain_tokens = total_domain_chunks * args.chunk_size
    print(f"[corpus] {len(corpus)} sources, {total_domain_chunks} chunks, "
          f"{total_domain_tokens:,} tokens total\n", flush=True)

    # General replay chunks
    general_chunks: list[list[int]] = []
    if os.path.exists(GENERAL_REPLAY_PATH):
        replay_texts = []
        for line in io.open(GENERAL_REPLAY_PATH, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # TWO SHAPES. QA-style replay carries prompt/gold; a general-text corpus
            # such as SlimPajama carries a bare `text`. Reading only the QA shape made
            # every general-text replay file load as ZERO chunks -- the loader then
            # warned and trained 100% domain, so every run tonight silently had NO
            # replay while believing it was annealing. `text` is checked FIRST because
            # a general corpus is what replay is actually for.
            body = obj.get("text") or ""
            if not body:
                prompt = obj.get("prompt", "")
                gold = obj.get("gold", obj.get("answer", ""))
                body = (prompt + "\n" + gold) if prompt else ""
            if body.strip():
                replay_texts.append(body)
        if replay_texts:
            all_replay = "\n\n".join(replay_texts)
            general_chunks = chunk_text(all_replay, tok, args.chunk_size)
            print(f"[general-replay] {GENERAL_REPLAY_PATH} → {len(general_chunks)} chunks", flush=True)
    if not general_chunks:
        # FAIL CLOSED WHEN REPLAY WAS EXPLICITLY REQUESTED. Silently dropping to 100%
        # domain turns a replay ARM into a duplicate of the no-replay arm: the run looks
        # perfectly healthy while testing nothing, which is exactly what happened at 00:05.
        # A default-path miss stays a warning; an explicit --general-replay that yields
        # nothing is a hard error.
        if args.general_replay:
            raise SystemExit(
                "--general-replay %s produced ZERO chunks. Expected a `text` field, or "
                "`prompt` + `gold`. Training would silently proceed at 100%% domain and "
                "the replay arm would duplicate the no-replay arm."
                % GENERAL_REPLAY_PATH)
        print("[WARN] no general replay — training without annealing (100% domain)", flush=True)
        args.anneal_start = 1.0
        args.anneal_end = 1.0

    # Allocation
    allocation = allocate_chunks(corpus, n_total=total_domain_chunks, lam=args.lam)

    # Load model
    print("\n[model] loading...", flush=True)
    from transformers import AutoModelForCausalLM
    from rlv2.common.cost_meter import enforce_vram_ceiling
    from rlv2.judge.reward import unload_all_ollama_models

    unload_all_ollama_models()
    enforce_vram_ceiling(vram_frac=1.0)

    # DTYPE DEPENDS ON THE PARAMETERISATION, and getting it wrong fails at step 1.
    #
    # LoRA: base can stay fp16 because only the fp32 adapter params carry gradients.
    # FULL-PARAM: base weights ARE the trained params, so fp16 weights produce fp16
    # gradients, and GradScaler.unscale_ raises "Attempting to unscale FP16 gradients".
    # AMP's contract is fp32 master weights + autocast activations; amp=True in the
    # training loop supplies the autocast, so the weights must be fp32 here.
    # (§9/35: never half() a live model — the fp32 master precision is what holds
    # small updates.)
    load_dtype = torch.float32 if args.full_param else torch.float16
    print(f"[dtype] loading base in {load_dtype} "
          f"({'full-param: AMP needs fp32 masters' if args.full_param else 'LoRA: fp16 base is fine'})",
          flush=True)
    # Dropout is a CONFIG field, not a TrainingArguments knob, so it has to be set at
    # load time -- assigning it to a live model does nothing, the modules are already
    # built. This architecture (qwen3_5_text) exposes attention_dropout ONLY; there is no
    # hidden_dropout to sweep, so the "dropout axis" here is a single knob.
    _model_kw = {}
    if args.attention_dropout is not None:
        _model_kw["attention_dropout"] = args.attention_dropout
        print("[dropout] attention_dropout=%g (config override at load)"
              % args.attention_dropout, flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=load_dtype, trust_remote_code=True, **_model_kw
    )
    if args.attention_dropout is not None:
        got = getattr(model.config, "attention_dropout", None)
        assert got == args.attention_dropout, (
            "attention_dropout did not take: asked %r, config says %r. A silently ignored "
            "config override makes the whole dropout sweep a no-op that reports three "
            "identical numbers as 'dropout does not matter'."
            % (args.attention_dropout, got))

    if args.full_param:
        # Full-param CPT: adamw_8bit required to fit optimizer states in 16GB
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[model] full-param: {n_params:.1f}M parameters (all trainable)", flush=True)
        optim_name = "adamw_8bit"
    else:
        from peft import LoraConfig, get_peft_model
        # CPT LoRA targets all linear layers (knowledge goes everywhere)
        lora_cfg = LoraConfig(
            r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0,
            bias="none", task_type="CAUSAL_LM", target_modules="all-linear"
        )
        model = get_peft_model(model, lora_cfg)
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"[model] LoRA r={args.rank}: {n_tr:.1f}M trainable", flush=True)
        optim_name = "adamw_torch"

    # Build training dataset per epoch with annealing
    rng = random.Random(args.seed)
    all_epoch_chunks: list[list[int]] = []
    for ep in range(args.epochs):
        # Linear anneal: domain fraction increases from start to end
        if args.epochs > 1:
            t = ep / (args.epochs - 1)
        else:
            t = 1.0
        anneal_frac = args.anneal_start + t * (args.anneal_end - args.anneal_start)
        epoch_chunks = sample_epoch(corpus, allocation, general_chunks, anneal_frac, rng)
        all_epoch_chunks.extend(epoch_chunks)
        print(f"[epoch {ep+1}] anneal={anneal_frac:.0%} domain | {len(epoch_chunks)} chunks", flush=True)

    # Each chunk is a sequence of token IDs for next-token prediction (labels = input_ids; the loss
    # shifts by one). Sequences are VARIABLE length, so the batch a record lands in decides how much
    # padding the whole batch pays for.
    _record_lengths = [len(c) for c in all_epoch_chunks]

    total_steps = len(all_epoch_chunks) // (args.batch_size * args.grad_accum)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    # §7.3 rule 12: compute warmup_steps from a fraction, never pass warmup_ratio --
    # it is deprecated in this HF version and its warning trains you to ignore warnings
    # that matter. The FRACTION is the sweepable knob; steps are derived from it.
    warmup_steps = int(args.warmup_frac * total_steps)

    # Scheduler SHAPE arguments. num_cycles belongs to the cosine family, min_lr_rate to
    # cosine_with_min_lr -- handing one to a scheduler that does not accept it raises, so
    # each is forwarded only when actually set.
    sched_kwargs: dict = {}
    if args.num_cycles is not None:
        sched_kwargs["num_cycles"] = args.num_cycles
    if args.min_lr_rate is not None:
        sched_kwargs["min_lr_rate"] = args.min_lr_rate
    if args.lr_scheduler == "warmup_stable_decay":
        # REQUIRED, not optional: `get_wsd_schedule()` raises
        # "missing 1 required positional argument: 'num_decay_steps'" without it, and the
        # failure lands after the model is loaded -- an HPO trial burns a slot to discover
        # a missing kwarg. Half the post-warmup budget decays, so the schedule is
        # warmup -> stable -> decay in roughly equal thirds, which is the shape's point.
        post_warmup = max(1, total_steps - warmup_steps)
        sched_kwargs["num_decay_steps"] = max(1, post_warmup // 2)

    print(f"\n[train] {len(all_epoch_chunks)} samples, {total_steps} steps, "
          f"warmup={warmup_steps} ({args.warmup_frac:.0%}), sched={args.lr_scheduler}"
          f"{sched_kwargs or ''}, wd={args.weight_decay}, clip={args.max_grad_norm}, "
          f"betas=({args.adam_beta1},{args.adam_beta2}), eps={args.adam_eps}, "
          f"attn_dropout={args.attention_dropout}, lr={args.lr}, optim={optim_name}",
          flush=True)

    # DERIVE the batch size now that the model and optimizer are actually on the card.
    # Effective batch (batch x accum) is held CONSTANT so this is purely a memory decision
    # and never a silent change to the optimisation -- halving the batch without doubling
    # accumulation would change the gradient-noise scale and confound every lr result.
    if args.auto_batch:
        effective = args.batch_size * args.grad_accum
        max_len = max(_record_lengths) if _record_lengths else args.chunk_size
        # Put the model on the card FIRST. Trainer would do this later, so measuring before
        # it reports allocated=0.00 GB and every downstream number is fiction.
        if torch.cuda.is_available():
            model.to("cuda")
        n_params_total = sum(p.numel() for p in model.parameters())
        new_bs, bi = derive_batch_size(max_len, model.config.vocab_size, args.batch_size,
                                       n_params=n_params_total,
                                       full_param=bool(args.full_param))
        if bi.get("measured"):
            print("[batch] logits/seq=%.0f MB at max_len=%d vocab=%d | weights=%.2f GB "
                  "+ grads/optim=%.2f GB pending | headroom=%.2f GB -> fits %d, requested %d"
                  % (bi["per_seq_bytes"] / 1e6, max_len, model.config.vocab_size,
                     bi["allocated"] / 1e9, bi["pending"] / 1e9,
                     bi["headroom"] / 1e9, bi["fits"], bi["requested"]), flush=True)
        if new_bs != args.batch_size:
            args.grad_accum = max(1, effective // new_bs)
            print("[batch] %d x %d -> %d x %d (effective %d held constant)"
                  % (args.batch_size, effective // args.batch_size, new_bs,
                     args.grad_accum, new_bs * args.grad_accum), flush=True)
            args.batch_size = new_bs
        else:
            print("[batch] keeping %d x %d" % (args.batch_size, args.grad_accum), flush=True)

    os.makedirs(args.out, exist_ok=True)

    # PRE-FLIGHT DISK CHECK. Full-param fp32 weights are ~4 GB per checkpoint, and the final save
    # is the only one, so ~4 GB must be free BEFORE the run starts. (The HF Trainer used to write
    # two more per full run, ~12 GB in all.) Twice now a run has trained to completion and then died
    # writing shards -- 33 minutes of GPU time discarded at the last step, with no
    # traceback, because safetensors cannot serialise into a full disk. Checking here costs
    # microseconds and fails while nothing is lost.
    import shutil as _shutil
    _keep = 1                                     # the final save is the only checkpoint
    _need = _keep * 4.3e9
    _free = _shutil.disk_usage(os.path.abspath(args.out)).free
    print("[disk] %.1f GB free, need ~%.1f GB for %d checkpoint(s)"
          % (_free / 1e9, _need / 1e9, _keep), flush=True)
    if _free < _need:
        raise SystemExit(
            "REFUSING TO START: %.1f GB free but this run needs ~%.1f GB to save %d "
            "checkpoint(s) of ~4.3 GB each. Training would finish and then lose "
            "everything at the write. Free space first, or pass --max-steps to run in "
            "throwaway mode (no per-epoch checkpoints)."
            % (_free / 1e9, _need / 1e9, _keep))

    # Batches pad to their OWN longest sequence, never a global maximum: the cross-entropy logits are
    # batch x seq x 248,320 vocab, which is what OOM'd at 2.03 GB when every sequence was forced to
    # 512. Labels are -100 on pad positions (rlv2.common.loop.collate): padding that reaches the
    # loss teaches the model to predict PAD, which looks like a converging loss curve and is not
    # learning. CPT pads on the right whatever the tokenizer's side is.
    from rlv2.common.loop import collate, train_batches

    def _batch(idxs):
        return collate([{"input_ids": all_epoch_chunks[i], "attention_mask": [1] * len(all_epoch_chunks[i]),
                         "labels": all_epoch_chunks[i]} for i in idxs], tok, pad_side="right")

    # QUANTILE DYNAMIC BATCHING -- 4 length-alike blocks per epoch, each batched to a
    # constant TOKEN budget using that block's OWN longest record.
    #
    #     batch_b = token_budget // max_len(block_b)
    #
    # so a short block runs many records per step and a long block runs few, while
    # tokens-per-step stays roughly constant. Padding is to the block max, not the global
    # max: that is the entire saving, and it is why the logits tensor stops being sized by
    # the single longest record in the corpus.
    #
    # `quantile_blocks` is the INCUMBENT in src/rlv2/common/lengths.py -- it already returns
    # (block_max_len, indices) sorted short -> long and asserts no row is lost. It is
    # extended here, not reimplemented.
    from rlv2.common.lengths import quantile_blocks

    class QuantileBatchSampler(torch.utils.data.Sampler):
        """Yields BATCHES (lists of indices), not indices -- a batch_sampler.

        Blocks are rotated and their contents reshuffled every epoch so the optimiser does
        not see all short records and then all long ones, which would correlate sequence
        length with training step.
        """

        def __init__(self, lengths, token_budget, n_blocks=4, seed=0):
            self.blocks = quantile_blocks(list(range(len(lengths))), lengths, n_blocks)
            self.token_budget = token_budget
            self.sizes = [max(1, token_budget // bmax) for bmax, _ in self.blocks]
            self.rng = random.Random(seed)
            self.epoch = 0

        def batches(self):
            order = list(range(len(self.blocks)))
            self.rng.shuffle(order)             # rotate blocks between epochs
            out = []
            for b in order:
                _, idxs = self.blocks[b]
                idxs = list(idxs)
                self.rng.shuffle(idxs)          # without replacement inside the block
                bs = self.sizes[b]
                for i in range(0, len(idxs), bs):
                    out.append(idxs[i:i + bs])
            return out

        def __iter__(self):
            return iter(self.batches())

        def __len__(self):
            return sum((len(idxs) + self.sizes[b] - 1) // self.sizes[b]
                       for b, (_, idxs) in enumerate(self.blocks))

    _token_budget, _tb_info = derive_token_budget(
        model.config.vocab_size,
        n_params=sum(p.numel() for p in model.parameters()),
        full_param=bool(args.full_param))
    _sampler = QuantileBatchSampler(_record_lengths, _token_budget, seed=args.seed)
    if _tb_info.get("measured"):
        print("[batch] token budget %d/step (headroom %.2f GB x %.2f safety, "
              "%.0f KB per token of logits)"
              % (_token_budget, _tb_info["headroom"] / 1e9, _tb_info["safety"],
                 _tb_info["per_token_bytes"] / 1e3), flush=True)
    print("[batch] %d quantile blocks | %s"
          % (len(_sampler.blocks),
             "  ".join("max=%d n=%d bs=%d" % (bmax, len(idxs), _sampler.sizes[b])
                       for b, (bmax, idxs) in enumerate(_sampler.blocks))), flush=True)

    # HALF-EPOCH F1 IN EVERY PHASE (owner requirement 2026-08-27, task 7a). CPT previously
    # had NO eval callback at all -- it reported one number at the end, so the SHAPE of the
    # curve, which is the instrument the experiment reads for lift, was invisible.
    from rlv2.eval.half_epoch_eval import HalfEpochEval
    _eval_cb = HalfEpochEval(model, tok, evals_per_epoch=args.evals_per_epoch,
                             tag="CPT lr=%g" % args.lr, epochs=args.epochs)

    # One pass over the sampler = every pre-built epoch. A step-bounded run (--max-steps) is an HPO
    # trial: it stops there, and is scored from the FINAL save below. The schedule spans the steps
    # that will actually run.
    _n_steps = total_steps if args.max_steps else -(-len(_sampler) // args.grad_accum)

    print("[train] starting CPT...", flush=True)
    t0 = time.time()
    losses = train_batches(
        model, (_batch(idxs) for idxs in _sampler), num_steps=_n_steps,
        lr=args.lr, lr_scheduler=args.lr_scheduler, warmup_steps=warmup_steps,
        weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm,
        betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps,
        scheduler_kwargs=sched_kwargs, seed=args.seed, callbacks=[_eval_cb], tok=tok,
        logging_steps=max(1, total_steps // 20), grad_accum=args.grad_accum, amp=True, optim=optim_name)
    elapsed = (time.time() - t0) / 60
    print(f"\n[train] DONE in {elapsed:.1f} min", flush=True)

    # Save final checkpoint
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[save] checkpoint → {args.out}", flush=True)

    # Report what happened
    if losses:
        print(f"[loss] start={losses[0]:.4f} → end={losses[-1]:.4f} "
              f"(Δ={losses[-1]-losses[0]:+.4f})", flush=True)

    print("\nNext: score with src/rlv2/eval/rl_calibration_eval.py to check post-trainability.", flush=True)
    print("The BEST CPT checkpoint is the one that produces the best downstream SFT/RL,", flush=True)
    print("NOT the one with lowest perplexity.", flush=True)


if __name__ == "__main__":
    main()
