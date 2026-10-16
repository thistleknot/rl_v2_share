"""src/rlv2/data/pools.py — §6.11 R3-only three-pool generation over the 3 trees-ensembles skills.

Notes: docs/SETTLED.md; docs/dispositions.md "skills_model PROVENANCE" (2026-09-11). Slides: docs/PROVENANCE.md rows for
RL deck slides 4 to 6.

Generates self-contained (no-context) QA from the 24 chunks, stratified by chunk, with a
running exclusion list, a 0.80 cosine gate, deixis rejection, and full provenance.

Build order is strictly C -> A -> B (§6.11: "the eval must be fixed before anything can
leak into it"). The running exclusion list and the cosine gate persist across all phases.

    C (eval)   : 3 items, ONE PER SKILL, chosen for centrality to that skill and NOT for
                 chunk coverage. §3.5.1 (owner, 2026-08-23) withdrew the n≈40 power target:
                 "Do not reintroduce it." The answer to a coarse eval is k≈8 repeated
                 sampling (§6.10), not more items. Do NOT grow this pool.
    A (CPT QA) : chunk-stratified over all 24 chunks, --n/2 items.
    B (SFT/RL) : chunk-stratified over all 24 chunks, --n/2 items.

--n sizes pools A+B only; C is fixed at 3.

MEASURED (2026-08-23, qwen3.5-oc:4b, 64.3 min): this corpus saturates. Asking for 78
A+B yielded 60 (A=54, B=6) from 22 chunks; several random-forest chunks exhausted all
6 tries and gave up. Cost was 350 regenerations, of which 340 were cosine rejections
(parse=10, deixis=0) -- i.e. GATE-bound, not latency-bound.

Do not read an early-run attempts/accept ratio as steady state: rejections are ~0 for
the first few dozen items and climb steeply, because each new item must clear the 0.80
gate against every prior item. Accepted items end up crowded at the bar (mean cosine
0.747, median 0.766, 41/63 at >=0.75), so raising --n past ~60 A+B buys near-duplicates
rather than coverage. To get more items, add source chunks -- not a bigger --n.

Slice gate first, then scale:
    python build_pools_r3.py --n 8       # cheap gate — eyeball quality
    python build_pools_r3.py --n 150     # full run
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

import numpy as np

from rlv2.judge import nvidia_llm

# GENERATION RUNS LOCAL. Every QA item in this pool was derived from the local 9b models
# and that does not change -- mixing a hosted model into the same pool would make
# `gen_model` provenance meaningless across records (§1.5: one variable).
#
# The default was "nvidia" until 2026-08-27, which was wrong twice over: steering records
# the hosted endpoint as MEASURED UNRELIABLE for batch generation, and its breaker opened
# on QuotaExhausted after 12 consecutive failures while the run sat at 1% GPU looking
# alive. Worse, `_chat_model` forwards the OLLAMA model id to the hosted endpoint, and
# `gen_model` is stamped from CATEGORY_MODELS unconditionally -- so a hosted-generated
# record would carry a local 9b label. Local by default closes all three.
#
# Embeddings were already local and stay there: the 0.80 cosine gate and every saturation
# number in this file's header were calibrated on nomic-embed-text, so swapping the
# embedder would silently redefine the gate.
# 2026-08-29: "openrouter" added on owner direction. The local 9b cannot supply the
# remaining items at the volume 240-per-phase needs -- and the note above about mixing
# backends is answered explicitly: varied GENERATION models are acceptable, what must stay
# static is the JUDGE. Provenance is preserved because resolved_gen_model() stamps the real
# id (see below), so a mixed pool stays auditable per record rather than silently mislabelled.
GEN_BACKEND = os.environ.get("POOL_GEN_BACKEND", "ollama")  # "ollama" | "nvidia" | "openrouter"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3.5-9b")
HOST = "http://localhost:11434"  # embeddings, and local generation
# The owner's sanctioned set is exactly two models (§6.11.1). The previous fallback was
# `qwen3.5-oc:4b`, which is in NEITHER -- a 4b standing in for a 9b would have been
# invisible in the receipt because the stamp comes from CATEGORY_MODELS, not from here.
GEN_MODEL = os.environ.get("POOL_GEN_MODEL", "qwen3.5-reasoning-thinking-256k:9b")
EMBED_MODEL = "nomic-embed-text"

CHUNKS = "data/rl/cpt_skills_chunked.jsonl"
EXEMPLARS = "data/rl/eval_canonical.jsonl"
OUT = "data/rl/pools_r3.jsonl"

# --- Few-shot exemplars (§6.11 step 1) ------------------------------------------------
# eval_canonical.jsonl IS the VibeThinker eval: 70 records, {'general': 56, 'domain': 14}.
# The 18 ('R3','general') rows are the population the spec names. The previous selector took
# source=="domain" with an `or r3` fallback -- and since 4 ('R3','domain') rows exist the
# fallback never fired, so all 78 prior items were shaped by 3 domain-derived strings and
# every real VibeThinker exemplar was silently skipped.
EXEMPLAR_SOURCE = "general"
# Measured R3 category profile; normalised over its sum (0.544) at point of use.
# THE R3 ROW ALONE IS THE WRONG BASIS. These four numbers are the R3 row of the owner's
# regime x category sheet, and using them means the eval mix is shaped by one regime rather
# than by the corpus as a whole. Kept because `boxcox_weights` and existing callers still
# reference it, and because the pool that already exists was drawn under it.
R3_SHAPE_WEIGHTS = {"Math": 0.229, "Code": 0.135, "Instruction": 0.095, "STEM": 0.085}

# THE CORRECT BASIS: the AVERAGE across R1-R4 per category (owner, 2026-08-29), Box-Cox at
# lambda=0.5, renormalised. Agentic (0.046) and ReadingComp (0.04933) are in the sheet but
# not in our four categories; dropping them before or after the transform gives the same
# PDF, because Box-Cox-then-renormalise is scale-equivariant.
#
#   category      avg across R1-R4    ->  PDF
#   Math                    0.1175            0.3075
#   Code                    0.0715            0.2399
#   Instruction             0.06533           0.2293
#   STEM                    0.0620            0.2234
#
# Difference from the R3 row it replaces: Math 0.331 -> 0.308, STEM 0.202 -> 0.223.
CATEGORY_AVG_WEIGHTS = {"Math": 0.1175, "Code": 0.0715,
                        "Instruction": 0.06533, "STEM": 0.0620}


def category_pdf(lam: float = 0.5) -> dict:
    """Box-Cox(lam) of CATEGORY_AVG_WEIGHTS, renormalised. THE allocation basis.

    Used for the eval draw and for the generation quota, so both are shaped by the same
    distribution instead of one being derived and the other assumed.
    """
    tot = sum(CATEGORY_AVG_WEIGHTS.values())
    bc = {c: (CATEGORY_AVG_WEIGHTS[c] / tot) ** lam for c in CATEGORY_ORDER}
    s = sum(bc.values())
    return {c: bc[c] / s for c in CATEGORY_ORDER}


def category_quota(n: int) -> dict:
    """How many items of each category an n-item pool should hold. Largest remainder."""
    w = category_pdf()
    exact = {c: n * w[c] for c in CATEGORY_ORDER}
    out = {c: int(exact[c]) for c in CATEGORY_ORDER}
    for c in sorted(CATEGORY_ORDER, key=lambda c: exact[c] - out[c], reverse=True):
        if sum(out.values()) < n:
            out[c] += 1
    return out
EXEMPLAR_SEED = 20260826  # exemplar rotation must be reproducible (§6.10 fixed sampling seed)
EXEMPLAR_GOLD_CAP = 600   # real golds run to multi-line derivations; 3 uncapped crowd out the chunk


def resolved_gen_model() -> str:
    """The model that actually generated an item.

    GEN_MODEL is the *ollama fallback* id. It was written to every record unconditionally,
    so all 78 prior items are labelled "qwen3.5-oc:4b" while generation in fact ran on the
    NVIDIA endpoint throughout (§6.11 step 6).
    """
    if GEN_BACKEND == "openrouter":
        return OPENROUTER_MODEL
    return nvidia_llm.MODEL if GEN_BACKEND == "nvidia" else GEN_MODEL

# The tight 3-skill cluster (§3.5.1). Kept at 3 on 2026-08-26 after the category axis
# below removed the need to widen: asking each chunk four DIFFERENT kinds of question
# multiplies distinct-item capacity 4x, giving 24 chunks * 4 = 96 flavour-slots for a
# 70-item budget. Widening the cluster was the pre-category workaround; it is not needed.
#
# NOTE on corpus size: cpt_skills_chunked.jsonl is a pre-chunked SAMPLE (262 skills /
# 1208 chunks) of a much larger master corpus (1,524 .md files). The trees-ensembles
# family alone has 29 skills / 165,428 chars, of which only 6 were ever chunked. So the
# "24 chunks" here is a property of the chunked sample, NOT a ceiling on the domain --
# if more headroom is ever needed, chunk more of the family rather than assuming the
# corpus is exhausted.
TARGET_SKILLS = [
    # TRAINING corpus skills. Widened 3 -> 17 on 2026-08-26 because the 3-skill corpus
    # SATURATED: at 147 items, 36% of accepted items sat at >= 0.78 against the 0.80 gate
    # and the last twelve included 0.800/0.799/0.798. Splitting the same passages finer
    # (24 -> 42 chunks) did NOT help -- the very next item still landed at 0.798, because
    # splitting text does not create new CONCEPTS, only smaller pieces of the same ones.
    #
    # §1.6 names this remedy exactly: "the fix is MORE SOURCE CHUNKS ... never a relaxed
    # gate and never a bigger --n". All 17 stay inside the trees-ensembles family.
    #
    # CONSEQUENCE, stated plainly: training items now span 17 skills while the eval
    # (EVAL_SKILLS / pool C) still probes 3. The eval is frozen by §1.4 and must not grow;
    # the added skills are near neighbours of the original three, so they act as related
    # domain content rather than a different subject.
    "trees-ensembles/ensemble-method",
    "trees-ensembles/random-forest",
    "trees-ensembles/boosted-tree",
    "trees-ensembles/gini-impurity-index",
    "trees-ensembles/expected-gini",
    "trees-ensembles/decision-tree",
    "trees-ensembles/bagging",
    "trees-ensembles/boosting",
    "trees-ensembles/gradient-boosting",
    "trees-ensembles/adaboost",
    "trees-ensembles/stacking-ensemble",
    "trees-ensembles/xgboost",
    "trees-ensembles/ensemble-learning",
    "trees-ensembles/ensemble-of-tree",
    "trees-ensembles/classification-tree",
    "trees-ensembles/cost-complexity-pruning",
]

# The EVAL skill set, deliberately separate from the training skill set.
#
# §1.4 freezes the eval for the whole pipeline, so pool C must stay at these three skills
# and these three items no matter how the TRAINING corpus is widened. Before this split,
# Phase C keyed off TARGET_SKILLS -- so adding source skills to relieve gate saturation
# (§1.6's sanctioned remedy) would silently have demanded a new eval item per added skill,
# growing pool C and swapping the instrument mid-experiment.
EVAL_SKILLS = [
    "trees-ensembles/ensemble-method",
    "trees-ensembles/random-forest",
    "trees-ensembles/boosted-tree",
]

COSINE_MAX = 0.80
MAX_TRIES = 6
POOL_C_SIZE = 3  # §1.4/§3.5.1: one R3 eval item per EVAL skill. Frozen. Never grows.

# --- Question-category axis (owner, 2026-08-26) ----------------------------------------
# Every prior run asked the SAME KIND of question ("in gradient boosting, how is X
# computed?"), so items collided semantically and piled against the cosine gate: median
# max_cosine_to_prior 0.77, max 0.799, modal bin the top one, 340/350 then 162/166
# regenerations rejected. Asking a derivation, an implementation, a causal why and a
# procedural how about the SAME chunk yields items that are far apart by construction.
# So this axis is both the requested feature and the fix for saturation -- it multiplies
# usable items per chunk without touching COSINE_MAX, which stays 0.80 (§3.5.1a).
#
# Names match R3_SHAPE_WEIGHTS deliberately: those are the VibeThinker R3 domains the
# few-shot exemplars are drawn from, so style block and category pull the same direction.
R3_CATEGORIES = {
    "Math": (
        "CATEGORY = MATH. Ask for a derivation, a formula, or a small numeric computation. "
        "The answer must be a specific expression, value, or derivation step -- not prose."
    ),
    "Code": (
        "CATEGORY = CODING. Ask about implementation: what a parameter does, how a step is "
        "coded, or what a described snippet computes. The answer must name the concrete "
        "mechanism, parameter, or returned value."
    ),
    "STEM": (
        "CATEGORY = CONCEPTUAL. Ask WHY a mechanism behaves as it does, or what property it "
        "guarantees. The answer must state the causal reason -- a definition is NOT acceptable."
    ),
    "Instruction": (
        "CATEGORY = PROCEDURAL. Ask how to carry out or configure the procedure correctly. "
        "The answer must give the concrete steps or the correct setting."
    ),
}
CATEGORY_ORDER = ["Math", "Code", "STEM", "Instruction"]

# Per-category generator models (owner, 2026-08-26: "pick different models for different
# dataset types that make the most sense"). Local ollama, because the NVIDIA endpoint
# measured 122s for a 16-TOKEN completion and returned nothing at all for 12 concurrent
# batch calls in 10 minutes -- matching the do_not_retry.md:377 precedent where the same
# endpoint hung and the work was abandoned.
#
# MEASURED, not assumed (§6.11 flags model choice as UNMEASURED and records two earlier
# probes that produced zero data): see archive/src/probe_gen_models.py and data/rl/gen_model_probe_local.json.
# Owner decision 2026-08-26: use ONLY the two strongest 9B models available, accepting the
# latency. The probe measured qwopus-coding at 1451s/batch vs 160s for qwen3.5-oc:4b -- the
# owner saw that number and chose capability over speed, so the -thinking- caveat recorded in
# §6.11.1 stands as a cost note, not as a veto.
CATEGORY_MODELS = {
    "Math": "qwen3.5-reasoning-thinking-256k:9b",
    # Code was routed to qwopus-coding-thinking-256k:9b (the code specialist) until
    # 2026-08-26. MEASURED throughput with it in the mix: 4 items in 46 min = 11.5
    # min/item, i.e. ~18 HOURS for the remaining 94 items, because 20 of 28 remaining
    # calls were qwopus and it runs ~9x slower (1451 s vs ~160 s per batch).
    # Both models remain inside the owner's sanctioned set; this trades the code
    # specialist for completing the dataset. Revert = restore the line below.
    "Code": "qwen3.5-reasoning-thinking-256k:9b",
    "STEM": "qwen3.5-reasoning-thinking-256k:9b",
    "Instruction": "qwen3.5-reasoning-thinking-256k:9b",
}
# Kept for a one-line revert / future comparison, NOT referenced by the run.
CODE_SPECIALIST_MODEL = "qwopus-coding-thinking-256k:9b"
# Same two models only -- never silently drop to a weaker one mid-run.
CATEGORY_MODEL_FALLBACK = "qwen3.5-reasoning-thinking-256k:9b"

# A batch of N asked in ONE call comes back internally near-duplicated -- MEASURED
# within-batch max cosine 0.94 (reasoning-9b) and 0.83 (oc-4b), both ABOVE the 0.80 gate.
# So a batch is a pool of CANDIDATES, never N accepted items: every item is still gated
# individually and greedily against everything already accepted. This is the amendment to
# §6.11 step 3, which banned batching outright on exactly this failure -- the ban is
# replaced by measurement plus per-item gating, not waived.
BATCH_SIZE = 10

# --- Generation length cap (owner, 2026-08-26) -----------------------------------------
# Answers are capped at VibeThinker's own answer length, taken as a ROBUST PREDICTION
# interval rather than a guess:
#
#   lengths  = token counts of the 18 R3/general golds in eval_canonical.jsonl
#   box-cox  = (x**0.5 - 1) / 0.5          (lam=0.5, the §6.7 transform)
#   bound_y  = median(y) + 2 * MAD(y) * 1.4826     <- MAD -> sigma, 2-sigma prediction
#   cap      = inverse-box-cox(bound_y)
#
# MEASURED 2026-08-26: n=18, raw tokens min 2 / median 53 / max 368;
# box-cox median 12.625, MAD 9.783, sigma 14.505 -> cap = 476 tokens per answer.
# (Recompute rather than trust this if eval_canonical.jsonl ever changes.)
#
# Why a cap at all: without an output bound a 256k-context "thinking" GGUF generates until
# it decides to stop, which blew past the 180 s HTTP timeout on EVERY call -- the request
# timed out, retried, timed out again, and a 25-minute run wrote ZERO items while sitting
# at 100% GPU. `think:false` does not suppress it on these builds (§6.11.1).
ANSWER_TOKEN_CAP = 476


# Reasoning-trace allowance, SEPARATE from the answer cap.
#
# The Box-Cox cap sizes the ANSWER. It says nothing about the <think> block these GGUFs
# emit regardless of think:false (§6.11.1), and num_predict bounds TOTAL output -- trace
# included. Budgeting zero for the trace is why Phase C kept dying: the reasoning ate the
# whole 645-token budget, the trace was truncated mid-tag, strip_reasoning correctly
# discarded an unterminated block, and there was no payload left to parse. Six tries, give
# up, no eval item for that skill.
#
# It bites unevenly, which is what made it look nondeterministic: qwen3.5-reasoning-9b
# thinks briefly and usually squeaked through, while qwopus-coding-9b (measured 1451 s for
# one batch vs 160 s) thinks at length and failed nearly every time. Same code, same
# budget, different model -- so the failure tracked WHICH CATEGORY was being generated.
THINK_TOKEN_BUDGET = 4096

# Generation HTTP timeout, and why retries are 1 here.
#
# _post defaults to timeout=180 with 3 retries -- sane for a fast embed call, actively
# harmful for a bounded long generation. MEASURED: qwopus-coding-9b took 1451 s for a
# batch of 10 at ~3000 tokens; at num_predict 10546 it runs longer still. Against a 900 s
# timeout every Code call timed out, retried, and timed out again -- 49 minutes of wall
# clock with zero items written and no error, because the retry loop never exhausted.
#
# A generation that is bounded by num_predict cannot run away, so a long call is WORK, not
# a hang. Give it room, and do NOT retry it: re-issuing a 25-minute request that was
# progressing fine is how an afternoon disappears.
GEN_TIMEOUT = 5400


def num_predict_for(batch_n: int) -> int:
    """Output-token budget for a batch of `batch_n` items.

    Trace allowance (paid once per call) + per item the answer cap plus ~40 tokens of
    question text, then +25% for the JSON scaffolding wrapping the list.
    """
    return int(THINK_TOKEN_BUDGET + (ANSWER_TOKEN_CAP + 40) * max(1, batch_n) * 1.25)

# Context-dependent deixis — an R3 question must stand alone (§6.11 step 5).
DEIXIS = re.compile(
    r"\b(in|from|per|according to)\s+the\s+(passage|text|excerpt|chunk|section|article|document|"
    r"above|following|snippet)\b"
    r"|\bthe\s+(passage|excerpt|chunk|snippet)\b"
    r"|\b(passage|text|section|article|document|example|figure|table|paragraph)\s+above\b"
    r"|\bthis\s+(dataset|passage|text|chunk|section|article|document|example|excerpt|snippet)\b"
    r"|\bthe\s+author\b|\bas\s+described\b"
    r"|\b(described|mentioned|shown|discussed|given|listed|outlined|stated|presented|defined"
    r"|introduced|referenced|depicted|illustrated|provided)\s+"
    r"(above|below|earlier|previously|here)\b"
    r"|\bthe\s+(above|aforementioned)\b"
    # "the <arbitrary noun> above/below" — the fixed noun list above cannot cover this.
    # LEAKED A REAL ITEM 2026-08-26: "In the boosting pseudocode above, what is the exact
    # mathematical operation ...". 'pseudocode' is not in any list, so the gate passed a
    # question that is only answerable with the source in hand -- the precise failure R3
    # exists to prevent. Allows 1-3 words between 'the' and the locator.
    # The negative lookahead keeps legitimate comparisons ("weights above the mean",
    # "error below a threshold") out of the gate: those continue into a noun phrase,
    # whereas a deictic reference ends the clause.
    r"|\bthe\s+(?:[\w-]+\s+){1,3}(?:above|below)"
    r"(?!\s+(?:the|a|an|its|his|her|their|this|that|\d|some|any|each|every|all))\b",
    re.I,
)

# Schema leakage -- the model answering questions ABOUT the JSON envelope it was asked to
# fill in, not about the domain. fetch_batch() requests a `distinct_facts` array with a
# per-item `fact_index`; measured 2026-08-29, 99 of 509 generated items treated those
# field names as subject matter ("What is the year corresponding to the value stored in the
# first element of the distinct_facts list?"). A prompt instruction is a request; this
# rejects. Checked against the ANSWER as well as the question -- the leak often lands only
# in the gold ("elderberry", index positions).
SCHEMA_LEAK = re.compile(
    r"\bdistinct[_\s]?facts\b|\bfact[_\s]?index\b"
    r"|\b(?:list|array)\s+of\s+(?:distinct\s+)?facts\b"
    r"|\bfacts?\s+(?:list|array)\b"
    r"|\b(?:at|in|from)\s+index\s+\d+\b"
    r"|\b(?:first|second|third|fourth|fifth|sixth|last)\s+(?:element|entry|item)\s+of\s+the\b"
    r"|\b(?:element|entry|item|value|string)\s+(?:stored\s+)?(?:at|in)\s+(?:position|index)\b"
    # BRACKET-INDEXED SCAFFOLD REFERENCE. The literal-plural alternatives above are defeated
    # by a single typo: r3-0706 leaked as "the string at distinct_fats[8]" -- `distinct_fats`
    # is not `distinct_facts`, so nothing above fired and the item reached the 12-item rubric
    # split, where it became the judge's own calibration material. This matches the
    # `<scaffold_identifier>[<n>]` SHAPE instead of an exact spelling, so any misspelling of
    # the generator's own field family is caught.
    #
    # DELIBERATELY NOT a general `\w+\[\d+\]`: array indexing is legitimate subject matter in
    # the Code category ("what does arr[3] evaluate to"), and a gate that rejected it would
    # delete a real category. The identifier must belong to the scaffold family --
    # `distinct_*`, anything containing `fact`, or anything ending `_index` -- which no
    # domain question introduces on its own.
    r"|\b(?:distinct[_\s]?\w*|\w*fact\w*|\w+_index)\s*\[\s*\d+\s*\]",
    re.I,
)

# Code-reference deixis -- the same self-containment failure as DEIXIS, but pointing at
# passage INTERNALS rather than the passage itself, so none of the classic prose triggers
# fire. Measured 2026-08-29: DEIXIS missed every one of "In the provided Python code, the
# variable `skf` is instantiated as ...", "The outer loop iterates using `for j, clf in
# enumerate(clfs):`", "The code uses a temporary array `dataset_blend_test_j`".
#
# The discriminator is a DEFINITE reference to a specific named entity that exists only in
# the source passage (`skf`, `clfs`, `dataset_blend_test_j`), not general domain content.
# Legitimate Code-category items -- "Write a Python function that computes ...", "In
# gradient boosting, what does the learning rate control?" -- introduce their identifiers
# indefinitely ("a function `foo`") and must keep passing. Erring toward precision on
# purpose: a false reject costs one item, a false accept poisons the corpus.
CODE_DEIXIS = re.compile(
    # "the provided/given/following [python] code|snippet|pseudocode|..."
    r"\bthe\s+(?:provided|given|following|above|preceding|original|source|sample|accompanying|"
    r"attached|shown)\s+(?:python\s+|sklearn\s+|numpy\s+)?"
    r"(?:code|snippet|script|program|listing|pseudocode|implementation|function|loop)\b"
    # "in|from|within the code|snippet|codebase|..."
    r"|\b(?:in|from|within|inside|per|using|given|throughout)\s+the\s+(?:python\s+)?"
    r"(?:code|snippet|script|codebase|pseudocode|listing)\b"
    # "the code [adverb] uses|defines|initializes|..." -- a claim about what the source does
    r"|\bthe\s+(?:code|snippet|script|program|pseudocode|listing|implementation)\s+"
    r"(?:\w+ly\s+)?"
    r"(?:uses?|used|defines?|defined|initiali[sz]es?|creates?|imports?|iterates?|calls?|sets?|"
    r"assigns?|contains?|includes?|applies|instantiates?|constructs?|builds?|computes?|shows?|"
    r"stores?|passes|performs?|declares?|returns?|prints?|fits?|trains?|loops?)\b"
    r"|\bthe\s+(?:code|source)\s+(?:snippet|block|sample|listing|fragment|line)\b"
    # loops and lines that exist only in the passage
    r"|\bthe\s+(?:outer|inner|main|nested|first|second|final|for|while|training|evaluation|"
    r"cross-validation)\s+loop\b"
    r"|\b(?:within|inside|in)\s+the\s+loop\b|\bthe\s+loop\s+over\b"
    r"|\bthe\s+line\s+`"
    # "the variable `skf`" / "the `clfs` list" -- a DEFINITE reference to a named local.
    # 'parameter'/'argument' are deliberately absent: "the parameter `max_depth`" is a
    # public sklearn name and is legitimate general domain content.
    r"|\bthe\s+(?:variable|array|list|object|function|method|class|dict|dictionary|matrix|"
    r"instance|call|line|statement)s?\s+`[^`]+`"
    r"|\bthe\s+`[^`]+`\s+(?:variable|array|list|object|loop|call|line|statement|block)\b"
    r"|\bthe\s+(?:variable|function|method|class|array|loop|object)\s+"
    r"(?:defined|declared|instantiated|initiali[sz]ed|created|assigned|named)\b"
    # "the evaluation print statements", "the import statements"
    r"|\bthe\s+(?:\w+\s+){0,2}(?:print|import|assignment|return)\s+statements?\b",
    re.I,
)


def gate_reason(question: str, answer: str = "") -> str | None:
    """Content-rejection reason for one candidate, or None to accept.

    Single decision point for every self-containment gate so the two call sites (the
    single-item retry loop and the batch gate) can never drift apart. The return value IS
    the stats key, so each class is booked distinctly and the receipt says which one killed
    the item.
    """
    if SCHEMA_LEAK.search(question) or SCHEMA_LEAK.search(answer or ""):
        return "schema"
    if DEIXIS.search(question):
        return "deixis"
    if CODE_DEIXIS.search(question):
        return "codedeixis"
    return None


# LaTeX -> ASCII pseudo-code. Applied to every generated question and gold before the
# item is accepted, because rule 5 in the template is a REQUEST and models ignore requests.
_GREEK = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
          "pi rho sigma tau upsilon phi chi psi omega Gamma Delta Theta Lambda Xi Pi "
          "Sigma Phi Psi Omega").split()
# The full Greek block in codepoint order, so no letter can be forgotten. U+03B1..U+03C9
# and U+0391..U+03A9; the final sigma (U+03C2) is handled separately below.
_GREEK_LOWER = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
                "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
                "final_sigma", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"]
_GREEK_UPPER = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
                "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi", "Rho",
                "", "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega"]

_UNICODE_MATH = {
    "\u03b1": "alpha", "\u03b2": "beta", "\u03b3": "gamma", "\u03b4": "delta",
    "\u03b5": "epsilon", "\u03b8": "theta", "\u03bb": "lambda", "\u03bc": "mu",
    "\u03c0": "pi", "\u03c1": "rho", "\u03c3": "sigma", "\u03c4": "tau",
    "\u03c6": "phi", "\u03c9": "omega", "\u0394": "Delta", "\u03a3": "sum",
    "\u03a0": "prod", "\u2211": "sum", "\u220f": "prod", "\u221a": "sqrt",
    "\u2264": "<=", "\u2265": ">=", "\u2260": "!=", "\u2248": "~=", "\u00d7": "*",
    "\u00f7": "/", "\u2212": "-", "\u2192": "->", "\u21d2": "=>", "\u2208": "in",
    "\u221e": "inf", "\u2202": "d", "\u2032": "'", "\u00b7": "*",
}
_SUPERSCRIPT = {"\u00b2": "^2", "\u00b3": "^3", "\u2070": "^0", "\u00b9": "^1",
                "\u2074": "^4", "\u2075": "^5", "\u2076": "^6", "\u2077": "^7",
                "\u2078": "^8", "\u2079": "^9"}
_SUBSCRIPT = {"\u2080": "_0", "\u2081": "_1", "\u2082": "_2", "\u2083": "_3",
              "\u2084": "_4", "\u2085": "_5", "\u2086": "_6", "\u2087": "_7",
              "\u2088": "_8", "\u2089": "_9", "\u1d62": "_i", "\u2c7c": "_j"}


def to_ascii_math(text: str) -> str:
    """Rewrite LaTeX / unicode math as ASCII pseudo-code.

    Why this is a mechanism and not a prompt line: the OpenRouter 9b was measured emitting
    "$\\sum p_i^2$" and "$C$" on 2026-08-29 while the local 9b wrote plain text. Mixed
    notation between an ANSWER and its GOLD is a spurious difference the judge can see, so
    the corpus has to be normalised on one side of the wire.
    """
    if not text:
        return text
    t = text
    # \frac{a}{b} -> (a)/(b), innermost first
    for _ in range(4):
        new = re.sub(r"\\(?:d|t)?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", t)
        if new == t:
            break
        t = new
    t = re.sub(r"\\(?:sqrt)\s*\{([^{}]*)\}", r"sqrt(\1)", t)
    t = re.sub(r"\\(?:text|mathrm|mathbf|mathit|mathbb|operatorname)\s*\{([^{}]*)\}",
               r"\1", t)
    t = re.sub(r"\\(?:left|right|displaystyle|limits|,|;|!)", "", t)
    t = t.replace("\\sum", "sum").replace("\\prod", "prod").replace("\\int", "integral")
    t = t.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    t = t.replace("\\leq", "<=").replace("\\geq", ">=").replace("\\neq", "!=")
    t = t.replace("\\approx", "~=").replace("\\in", " in ").replace("\\infty", "inf")
    t = t.replace("\\log", "log").replace("\\ln", "ln").replace("\\exp", "exp")
    t = t.replace("\\to", "->").replace("\\Rightarrow", "=>").replace("\\partial", "d")
    for g in _GREEK:
        t = t.replace("\\" + g, g)
    # EVERY Greek letter, not a hand-picked subset. The curated _UNICODE_MATH map missed
    # eta, and 5 items shipped with a raw U+03B7 before this was widened -- the same class
    # of gap would recur for zeta, xi, chi and the rest. Generate the whole block instead.
    for _i, _name in enumerate(_GREEK_LOWER):
        t = t.replace(chr(0x3B1 + _i), _name)
    for _i, _name in enumerate(_GREEK_UPPER):
        t = t.replace(chr(0x391 + _i), _name)
    # strip $...$ / \(...\) / \[...\] delimiters, keep the content
    t = re.sub(r"\\\[|\\\]|\\\(|\\\)", "", t)
    t = re.sub(r"\$+", "", t)
    for k, v in list(_UNICODE_MATH.items()) + list(_SUPERSCRIPT.items()) + list(_SUBSCRIPT.items()):
        t = t.replace(k, v)
    t = t.replace("{", "").replace("}", "")           # leftover grouping braces
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def _post(path: str, payload: dict, timeout: int = 180, retries: int = 3) -> dict:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                HOST + path,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 400:
                raise
            last = e
        except Exception as e:
            last = e
        # Exponential backoff with jitter (owner, 2026-08-26: "if timeouts occur, use
        # exponential decay"). Was linear 1.5/3.0/4.5s, which barely backs off at all when
        # the real problem is a saturated GPU still chewing on the previous request --
        # retrying into contention makes the contention worse.
        time.sleep(1.5 * (2 ** attempt) + random.uniform(0, 1.0))
    raise last


class RemoteMiss(RuntimeError):
    """The endpoint returned NOTHING -- distinct from 'the model wrote something unusable'."""


def _openrouter_chat(prompt: str, system: str, temperature: float,
                     timeout: int = 180, retries: int = 3) -> str:
    """OpenRouter chat completion.

    REASONING IS DISABLED, and that is not a preference. qwen/qwen3.5-9b is a thinking
    model: measured 2026-08-29, the default spent every one of 16 allotted tokens on
    reasoning and returned content=None. Three settings were compared on the same prompt:

        default max_tokens=512      3.1s   158 reasoning tokens   $2.8e-05
        reasoning.exclude=True      2.3s   185 reasoning tokens   $5.3e-05   <- hides, still pays
        reasoning.enabled=False     1.2s     0 reasoning tokens   $2.0e-06   <- shipped

    `exclude` only suppresses the text in the response; the tokens are still generated and
    billed. `enabled: False` is the one that actually turns it off -- 25x cheaper.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 1200,
        "reasoning": {"enabled": False},
    }
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                OPENROUTER_URL, data=json.dumps(payload).encode("utf-8"),
                headers={"Authorization": "Bearer " + key,
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            txt = (d.get("choices") or [{}])[0].get("message", {}).get("content")
            if txt:
                return txt.strip()
            # Empty content with a 200 is the thinking-model failure mode, not a parse
            # error. Surface it as a miss so the caller retries instead of booking a
            # malformed item.
            last = RemoteMiss("openrouter returned empty content")
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403):
                raise
            last = e
        except Exception as e:                                   # noqa: BLE001
            last = e
        time.sleep(1.5 * (2 ** attempt) + random.uniform(0, 1.0))
    raise RemoteMiss("openrouter exhausted retries: %s" % last)


def batch_schema(n: int, cats: list[str]) -> dict:
    """Strict schema for one batched ask (REQ-G3/G4/G5).

    `distinct_facts` FIRST, then items tagged with a `fact_index` into it. Measured
    2026-08-29: an unconstrained batch of 10 from one chunk collapsed to about THREE
    distinct facts -- four questions on the log-rank criterion, three on the Kaplan-Meier
    leaf -- and the cosine gate then rejected ~89%. Making the model enumerate the facts
    first, and forcing each item to name which one it used, is what separates them.

    ASCII rules live in the field DESCRIPTIONS so they are part of the contract rather than
    a line of prose the model may skip; `to_ascii_math` still strips anything that leaks.
    """
    return {
        "type": "object",
        "properties": {
            "distinct_facts": {
                "type": "array", "items": {"type": "string"},
                "description": ("Every separate, non-overlapping testable fact in this "
                                "passage. Be exhaustive; do not invent facts."),
            },
            "items": {
                "type": "array",
                "description": "Exactly %d question/answer pairs." % n,
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "enum": cats},
                        "fact_index": {
                            "type": "integer",
                            "description": ("Index into distinct_facts. Every item must use "
                                            "a DIFFERENT fact."),
                        },
                        "question": {
                            "type": "string",
                            "description": ("Self-contained exam question. Plain ASCII only: "
                                            "no LaTeX, no $ delimiters, no backslash "
                                            "commands. Formulas as ASCII pseudo-code, "
                                            "e.g. sum(p_i^2)."),
                        },
                        "answer": {
                            "type": "string",
                            "description": ("Verifiable answer in 1-3 sentences. Plain "
                                            "ASCII, same formula rules. No reasoning and no "
                                            "self-correction."),
                        },
                    },
                    "required": ["category", "fact_index", "question", "answer"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["distinct_facts", "items"],
        "additionalProperties": False,
    }


def _openrouter_structured(prompt: str, system: str, schema: dict,
                           temperature: float = 0.8, timeout: int = 240,
                           retries: int = 3) -> dict | None:
    """OpenRouter chat constrained by a strict JSON schema.

    `require_parameters: true` keeps routing to endpoints that support structured outputs --
    all 5 of this model's endpoints do, verified 2026-08-29. Reasoning stays disabled for
    the measured reason in `_openrouter_chat`: `enabled: false` costs 25x less than
    `exclude: true`, which still generates and bills the tokens.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 6000,
        "reasoning": {"enabled": False},
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "qa_batch", "strict": True,
                                            "schema": schema}},
        "provider": {"require_parameters": True},
    }
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                OPENROUTER_URL, data=json.dumps(payload).encode("utf-8"),
                headers={"Authorization": "Bearer " + key,
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            txt = (d.get("choices") or [{}])[0].get("message", {}).get("content")
            if txt:
                return json.loads(txt)
            last = RemoteMiss("openrouter returned empty content")
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403):
                raise
            last = e
        except Exception as e:                                   # noqa: BLE001
            last = e
        time.sleep(1.5 * (2 ** attempt) + random.uniform(0, 1.0))
    raise RemoteMiss("openrouter structured exhausted retries: %s" % last)


def dedupe_by_fact(items: list, stats: dict | None = None) -> list:
    """Drop ITEMS that reuse a fact_index. Never drop the batch (REQ-G5).

    JSON Schema cannot express cross-item uniqueness, so the constraint has to be enforced
    here. Measured 2026-08-29: 13 of 24 sampled batches reused an index, up to FOUR times on
    `random-forest` -- rejecting the batch there would have discarded 4 good questions to
    kill 4 bad ones.
    """
    seen, out = set(), []
    for it in items or []:
        fi = it.get("fact_index")
        if fi is not None and fi in seen:
            if stats is not None:
                stats["dupfact"] = stats.get("dupfact", 0) + 1
            continue
        if fi is not None:
            seen.add(fi)
        out.append(it)
    return out


def _chat(prompt: str, system: str, temperature: float = 0.8) -> str:
    if GEN_BACKEND == "openrouter":
        return _openrouter_chat(prompt, system, temperature)
    if GEN_BACKEND == "nvidia":
        out = nvidia_llm.complete(
            system, prompt, max_tokens=1200, temperature=temperature
        )
        if out is None:
            # complete() returns None ONLY after exhausting MAX_RETRIES. Coercing that to ""
            # made a dead endpoint indistinguishable from a model emitting garbage: the caller
            # booked it as a PARSE failure, burned MAX_TRIES on every chunk, and reported
            # saturation over an empty pool. That is the 2026-08-23 three-hour empty run.
            raise RemoteMiss("nvidia_llm.complete() exhausted retries -- no response at all")
        return out.strip()
    d = _post(
        "/api/chat",
        {
            "model": GEN_MODEL,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": temperature, "num_ctx": 16384},
        },
    )
    return (d.get("message") or {}).get("content", "").strip()


def _chat_model(prompt: str, system: str, model: str, temperature: float = 0.8,
                tries: int = 4, num_predict: int = 1024) -> str:
    """One completion from an EXPLICIT model, with exponential backoff.

    Backoff per the owner's instruction, but bounded: nvidia_llm's breaker docstring says a
    caller that sees RemoteMiss must stop rather than retry, and grinding a rate-limited
    endpoint is what produced the earlier self-DoS. Four tries, doubling from 4s, jittered.
    """
    delay = 4.0
    for attempt in range(tries):
        try:
            if GEN_BACKEND == "openrouter":
                # THE PER-CATEGORY `model` ARG IS DELIBERATELY IGNORED HERE.
                # CATEGORY_MODELS routes Math/Code/etc to different LOCAL ollama ids; on
                # OpenRouter we run one model for the whole pool, and resolved_gen_model()
                # stamps that id on every record so provenance stays honest. Forwarding an
                # ollama id to OpenRouter is exactly the mislabelling the module header
                # warns about for the nvidia path.
                return _openrouter_chat(prompt, system, temperature)
            if GEN_BACKEND == "nvidia":
                out = nvidia_llm.complete(system, prompt, max_tokens=3000,
                                          temperature=temperature, model=model)
                if out is None:
                    raise RemoteMiss("nvidia_llm.complete() exhausted retries")
                return out.strip()
            d = _post("/api/chat", {
                "model": model,
                "stream": False,
                "think": False,   # kiro Req 3.7: /v1-style reasoning returns empty content
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "options": {
                    "temperature": temperature,
                    "num_ctx": 16384,
                    # BOUND THE OUTPUT. Without num_predict a 256k-context "thinking" model
                    # generates until it decides to stop, which for these GGUFs exceeded the
                    # 180 s _post timeout -- so the call timed out, retried, timed out again,
                    # and the run produced ZERO items while looking busy at 100% GPU for 25
                    # minutes. `think:false` does NOT suppress it on these builds (§6.11.1).
                    "num_predict": num_predict,
                },
            }, timeout=GEN_TIMEOUT, retries=1)
            return (d.get("message") or {}).get("content", "").strip()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(delay + random.uniform(0, 1.5))
            delay *= 2
    return ""


def embed(texts: list[str]) -> np.ndarray:
    out = []
    for t in texts:
        d = _post("/api/embed", {"model": EMBED_MODEL, "input": t[:6000]})
        out.append(d["embeddings"][0])
    v = np.asarray(out, dtype=np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(n, 1e-9, None)


def load_chunks() -> list[dict]:
    rows = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
    ch = [r for r in rows if r.get("skill") in TARGET_SKILLS]
    # stable chunk ids, per-skill ordinal
    per: dict[str, int] = {}
    for r in ch:
        s = r["skill"]
        per[s] = per.get(s, 0) + 1
        r["chunk_id"] = "%s#%02d" % (s.split("/")[-1], per[s])
    return ch


def allocate(chunks: list[dict], total: int) -> dict[str, int]:
    """Proportional to chunk content, summing EXACTLY to `total`.

    The per-chunk floor is `min(2, total // n_chunks)`, not a hard 2.

    A hard floor of 2 over 24 chunks silently forces >=48 items regardless of what
    `total` says, and the old down-adjust loop refused to cut below 2, so it gave up
    and returned an over-allocation. Pool A (built first) therefore ate 48+ items out
    of a 37-item half-budget and pool B was left with 6, all from a single skill.
    The floor can no longer exceed the budget: floor * n_chunks <= total by
    construction, so the down-adjust loop always has slack to cut.
    """
    n_chunks = len(chunks)
    if n_chunks == 0:
        return {}
    if total <= 0:
        return {c["chunk_id"]: 0 for c in chunks}

    floor = min(2, total // n_chunks)
    lens = np.array([len(c["text"]) for c in chunks], dtype=float)
    raw = lens / lens.sum() * total
    n = np.maximum(floor, np.floor(raw)).astype(int)

    while n.sum() > total:
        for i in np.argsort(-(n - raw)):
            if n[i] > floor:
                n[i] -= 1
                break
        else:
            break
    while n.sum() < total:
        i = int(np.argmax(raw - n))
        n[i] += 1
        raw[i] -= 1e-6

    assert n.sum() == total, "allocate() returned %d for total=%d" % (n.sum(), total)
    return {c["chunk_id"]: int(k) for c, k in zip(chunks, n)}


def allocate_by_skill(chunks: list[dict], total: int) -> dict[str, int]:
    """Split `total` EQUALLY across skills, then across each skill's chunks.

    Owner instruction (2026-08-23): "3 skills and divide that 70 between them."

    The older allocate() distributed proportional to chunk CHARACTER COUNT, which
    weights a skill by how verbose its source text happens to be rather than giving
    each skill equal representation. At total=70 that yields

        ensemble-method 13,811 chars -> 28    random-forest 11,125 -> 23
        boosted-tree     9,307       -> 19

    i.e. boosted-tree gets 2/3 of ensemble-method's share for no reason connected to
    the experiment. Equal thirds is what was asked for and is the defensible default:
    the eval is one item per skill, so each skill must be trained comparably.

    Remainders go to the longest chunks (deterministic, no RNG) so repeated runs at
    the same --n produce the same allocation.
    """
    by_skill: dict[str, list[dict]] = collections.OrderedDict()
    for c in chunks:
        by_skill.setdefault(c["skill"], []).append(c)

    out = {c["chunk_id"]: 0 for c in chunks}
    skills = sorted(by_skill)
    if not skills or total <= 0:
        return out

    base, rem = divmod(total, len(skills))
    for i, s in enumerate(skills):
        share = base + (1 if i < rem else 0)
        sc = by_skill[s]
        per, r = divmod(share, len(sc))
        for c in sc:
            out[c["chunk_id"]] = per
        for j in sorted(range(len(sc)), key=lambda j: -len(sc[j]["text"]))[:r]:
            out[sc[j]["chunk_id"]] += 1

    assert sum(out.values()) == total, "allocate_by_skill returned %d for total=%d" % (
        sum(out.values()), total)
    return out


def _round_robin(chunks: list[dict], alloc: dict[str, int]) -> list[dict]:
    """Chunk order that cycles 1,2,..,N,1,2,.. rather than emitting a chunk's whole
    quota back-to-back. Consecutive items from the same chunk are the most likely
    pair to trip the 0.80 cosine gate, so spreading them lowers the rejection rate."""
    left = {c["chunk_id"]: alloc.get(c["chunk_id"], 0) for c in chunks}
    out: list[dict] = []
    while any(v > 0 for v in left.values()):
        for c in chunks:
            if left[c["chunk_id"]] > 0:
                out.append(c)
                left[c["chunk_id"]] -= 1
    return out


def plan_tasks(chunks: list[dict], budget: int) -> list[tuple[dict, str]]:
    """Interleaved A/B work list: [(chunk, pool), ...].

    THE STARVATION FIX. The old loop was

        for pool, total in (("A", a_budget), ("B", budget - a_budget)):

    which ran pool A to exhaustion over all 24 chunks before B emitted anything.
    Every accepted item joins the running exclusion list and the cosine gate, so B
    began against an already-saturated corpus, burned MAX_TRIES chunk after chunk,
    and finished with 6 items from a single skill while A took 54.

    Interleaving A,B,A,B makes both pools face the same gate pressure at the same
    time. A corpus that saturates then starves them *symmetrically* instead of
    spending its entire headroom on whichever pool happens to run first.
    """
    a_budget = budget // 2
    qa = _round_robin(chunks, allocate_by_skill(chunks, a_budget))
    qb = _round_robin(chunks, allocate_by_skill(chunks, budget - a_budget))
    plan: list[tuple[dict, str, str]] = []
    # Category rotates per chunk AND is OFFSET by the chunk's index.
    #
    # Both halves are load-bearing. Rotation alone (every chunk starting at Math) is
    # degenerate at this budget: 70 items over 36 chunks gives most chunks exactly two
    # draws, so every chunk yields Math then Code and the other two flavours never
    # appear -- measured Math 33 / Code 33 / STEM 2 / Instruction 2. Offsetting the
    # start by chunk index spreads the short draws across all four flavours instead.
    order = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    seen: dict[str, int] = collections.defaultdict(int)

    def _tag(c: dict) -> tuple[dict, str]:
        cid = c["chunk_id"]
        idx = (order.get(cid, 0) + seen[cid]) % len(CATEGORY_ORDER)
        seen[cid] += 1
        return c, CATEGORY_ORDER[idx]

    for i in range(max(len(qa), len(qb))):
        if i < len(qa):
            c, cat = _tag(qa[i])
            plan.append((c, "A", cat))
        if i < len(qb):
            c, cat = _tag(qb[i])
            plan.append((c, "B", cat))
    return plan


def _shape_counts(k: int, pool_domains: set[str]) -> dict[str, int]:
    """Largest-remainder allocation of k exemplar slots over the R3 domain profile.

    Only domains actually present in the pool take part, so the weights renormalise instead
    of silently dropping slots. For k=3 over the full profile this yields Math/Code/
    Instruction = 1/1/1 (STEM's 0.085 share loses the final remainder).
    """
    w = {d: v for d, v in R3_SHAPE_WEIGHTS.items() if d in pool_domains}
    if not w or k <= 0:
        return {}
    tot = sum(w.values())
    exact = {d: k * v / tot for d, v in w.items()}
    counts = {d: int(x) for d, x in exact.items()}
    left = k - sum(counts.values())
    # ties broken by weight then name so the allocation is stable across runs
    order = sorted(exact, key=lambda d: (-(exact[d] - int(exact[d])), -w[d], d))
    for d in order[:left]:
        counts[d] += 1
    return {d: c for d, c in counts.items() if c}


def style_exemplars(k: int = 3) -> list[str]:
    """Few-shot exemplars drawn from the real VibeThinker R3 eval, shaped to R3's profile.

    Prior behaviour selected source=="domain" and never reached its `or r3` fallback, so every
    generated item was styled on 3 domain rows and none of the 18 genuine R3 exemplars was
    ever shown to the generator (§6.11 step 1).
    """
    if not os.path.exists(EXEMPLARS):
        return []
    rs = [json.loads(l) for l in open(EXEMPLARS, encoding="utf-8")]
    pool = [
        r for r in rs
        if str(r.get("regime")) == "R3" and str(r.get("source")) == EXEMPLAR_SOURCE
    ]
    if not pool:
        return []

    by_dom: dict[str, list[dict]] = {}
    for r in pool:
        by_dom.setdefault(str(r.get("domain")), []).append(r)

    rng = random.Random(EXEMPLAR_SEED)
    picked: list[dict] = []
    for dom, n in sorted(_shape_counts(k, set(by_dom)).items()):
        cand = sorted(by_dom[dom], key=lambda r: str(r.get("question") or r.get("prompt")))
        picked += rng.sample(cand, min(n, len(cand)))
    # top up from the whole pool if the profile under-filled (tiny/degenerate eval files)
    if len(picked) < k:
        rest = sorted(
            (r for r in pool if r not in picked),
            key=lambda r: str(r.get("question") or r.get("prompt")),
        )
        picked += rest[: k - len(picked)]

    out = []
    for r in picked[:k]:
        q = " ".join(str(r.get("question") or r.get("prompt")).split())[:200]
        a = " ".join(str(r.get("gold") or "").split())[:EXEMPLAR_GOLD_CAP]
        out.append("Q: %s\n    A: %s" % (q, a) if a else "Q: %s" % q)
    return out


EXEMPLARS_PER_CATEGORY = 3


def style_exemplars_by_category(k: int = EXEMPLARS_PER_CATEGORY) -> dict[str, list[str]]:
    """Few-shot exemplars MATCHED to the category being asked for.

    §6.11 step 1 samples R3 exemplars from the VibeThinker eval to shape phrasing
    (VibeThinker contributes no ITEMS to any pool -- style only). But the selector
    built ONE global block and showed it to every item, so a prompt saying
    "CATEGORY = PROCEDURAL" was illustrated with Code and Math questions: the
    instruction said one thing and the demonstration showed another, which is the
    weakest possible way to specify a format.

    Matching them means a Math item sees real Math phrasing and a Code item sees
    real Code phrasing. The R3/general pool carries Math 5 / Code 5 / STEM 4 /
    Instruction 4 -- enough to give every category its own block.

    Deterministic (sorted, then seeded per category) so runs stay comparable (§6.10).
    """
    if not os.path.exists(EXEMPLARS):
        return {c: [] for c in CATEGORY_ORDER}
    rs = [json.loads(l) for l in open(EXEMPLARS, encoding="utf-8")]
    pool = [
        r for r in rs
        if str(r.get("regime")) == "R3" and str(r.get("source")) == EXEMPLAR_SOURCE
    ]

    by_dom: dict[str, list[dict]] = {}
    for r in pool:
        by_dom.setdefault(str(r.get("domain")), []).append(r)

    out: dict[str, list[str]] = {}
    for cat in CATEGORY_ORDER:
        cand = sorted(by_dom.get(cat, []),
                      key=lambda r: str(r.get("question") or r.get("prompt")))
        if not cand:
            # Fall back to the mixed block rather than showing nothing, and SAY so --
            # silence here would be indistinguishable from a deliberate style choice.
            print("  [style] no %s exemplars in the R3/general pool; using mixed block" % cat)
            out[cat] = style_exemplars(k)
            continue
        rng = random.Random(EXEMPLAR_SEED + sum(map(ord, cat)))
        picked = rng.sample(cand, min(k, len(cand)))
        rendered = []
        for r in picked:
            q = " ".join(str(r.get("question") or r.get("prompt")).split())[:200]
            a = " ".join(str(r.get("gold") or "").split())[:EXEMPLAR_GOLD_CAP]
            rendered.append("Q: %s\n    A: %s" % (q, a) if a else "Q: %s" % q)
        out[cat] = rendered
    return out


SYSTEM = (
    "You write exam questions that test whether a model has internalised a machine-learning "
    "concept. You output STRICT JSON only: {\"question\": \"...\", \"answer\": \"...\"}. No prose."
)

TEMPLATE = """Source material (for YOUR reference only — the reader will NOT see it):
---
{chunk}
---

Write ONE exam question about the concept above, and its answer.

HARD REQUIREMENTS:
1. SELF-CONTAINED. The reader never sees the source material. Never write "the passage",
   "the text above", "this dataset", "the author", "as described". The question must make
   full sense on its own to someone who has studied machine learning.
2. VERIFIABLE. The answer must be a specific, checkable fact or mechanism — not an opinion
   and not open-ended discussion. Answer in 1-3 sentences.
3. NON-TRIVIAL. It should test real understanding of the concept, not vocabulary.
4. DO NOT duplicate or paraphrase any question in the exclusion list below.
5. PLAIN ASCII ONLY. No LaTeX, no MathJax, no markdown. Write formulas as ASCII
   pseudo-code: "sum(p_i^2)" not "$\\sum p_i^2$", "alpha" not "$\\alpha$",
   "RSS(T) + alpha * |T|" not "$RSS(T) + \\alpha|T|$". No $ delimiters, no backslash
   commands, no unicode math symbols.
6. {category}

{style}
{exclusion}
Output STRICT JSON only: {{"question": "...", "answer": "..."}}"""


BATCH_SYSTEM = (
    "You write exam questions that test whether a model has internalised a machine-learning "
    "concept. You output STRICT JSON only: a LIST of objects, "
    '[{"question": "...", "answer": "..."}, ...]. No prose, no markdown fence.'
)

BATCH_TEMPLATE = """Source material (for YOUR reference only — the reader will NOT see it):
---
{chunk}
---

Write {n} DIFFERENT exam questions about the concept above, each with its answer.

HARD REQUIREMENTS:
1. CLOSED-BOOK AND SELF-CONTAINED. This is a closed-book exam: the reader has NO access to
   the source material, and never will. Two consequences, both mandatory:
   (a) The question must CARRY ITS OWN SETUP. Every definition, symbol, value or condition
       needed to answer must appear IN the question. If the source says "with the parameters
       above", the question must restate those parameters explicitly.
   (b) NEVER allude to the source as if it could be consulted. Banned: "the passage", "the
       text above", "this dataset", "the author", "as described", "in the given example",
       "the section", "the document". Anyone who has studied machine learning must be able
       to answer from memory alone.
2. VERIFIABLE. Each answer must be a specific, checkable fact or mechanism — not an opinion
   and not open-ended discussion. Answer in 1-3 sentences.
3. NON-TRIVIAL. Test real understanding of the concept, not vocabulary.
4. The {n} questions must be MAXIMALLY DIFFERENT from each other — a different sub-topic and
   a different angle each. Do NOT paraphrase one question {n} times.
5. {category}
6. DO NOT duplicate or paraphrase any question in the exclusion list below.

{style}
{exclusion}
Output STRICT JSON only: a list of {n} objects, each {{"question": "...", "answer": "..."}}"""


def parse_batch(raw: str) -> list[dict]:
    """Pull a LIST of {question, answer} out of model text, tolerantly.

    Batch replies arrive fenced, prefixed with prose, or with a trailing comma far more
    often than single-item replies do, so this is deliberately more forgiving than
    parse_json() -- but it applies the same minimum-quality bar per item.
    """
    if not raw:
        return []
    s = strip_reasoning(raw)
    m = re.search(r"\[.*\]", s, re.S)
    cand = m.group(0) if m else s
    try:
        data = json.loads(cand)
    except Exception:
        data = []
        for om in re.finditer(r"\{[^{}]*\}", cand, re.S):
            try:
                data.append(json.loads(om.group(0)))
            except Exception:
                pass
    if isinstance(data, dict):
        data = [data]
    out = []
    for d in data or []:
        if not isinstance(d, dict):
            continue
        q, a = d.get("question"), d.get("answer")
        if isinstance(q, str) and isinstance(a, str) and len(q.strip()) >= 15 and a.strip():
            # ASCII BEFORE THE GATE. The cosine gate embeds the question text, so LaTeX and
            # its ASCII equivalent would embed differently and dedup differently.
            out.append({"question": to_ascii_math(q.strip()),
                        "answer": to_ascii_math(a.strip())})
    return out


def strip_reasoning(s: str) -> str:
    """Remove <think>...</think> traces before any JSON extraction.

    These GGUFs emit reasoning despite think:false (§6.11.1). It matters for BOTH parsers:
    parse_json's greedy `\\{.*\\}` swallows the whole trace when the reasoning contains
    braces, json.loads then fails, and all MAX_TRIES burn as parse failures -- which is how
    Phase C gave up on ensemble-method and would have shipped an eval MISSING a skill.
    An unclosed trace means num_predict truncated the model mid-reasoning: there is no
    payload after it, so drop from the opening tag on rather than scrape scratch work.
    """
    if not s:
        return ""
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S | re.I)
    s = re.sub(r"<think>.*$", "", s, flags=re.S | re.I)
    return re.sub(r"^```(?:json)?|```$", "", s.strip(), flags=re.M).strip()


def parse_json(s: str) -> dict | None:
    s = strip_reasoning(s)
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        try:
            d = json.loads(m.group(0).replace("\n", " "))
        except Exception:
            return None
    q = to_ascii_math(str(d.get("question", "")).strip())
    a = to_ascii_math(str(d.get("answer", "")).strip())
    if len(q) < 15 or len(a) < 5:
        return None
    return {"question": q, "answer": a}


def central_chunks_per_skill(chunks: list[dict]) -> dict[str, dict]:
    """Pick the most representative chunk for each skill.

    The eval item for a skill should probe that skill's core, not a
    peripheral aside, so we take the chunk whose embedding is closest to
    its own skill's centroid. Returns {skill: chunk}.
    """
    by_skill: dict[str, list[dict]] = collections.defaultdict(list)
    for c in chunks:
        by_skill[c["skill"]].append(c)

    out: dict[str, dict] = {}
    for skill, cs in by_skill.items():
        if len(cs) == 1:
            out[skill] = cs[0]
            continue
        V = embed([c["text"][:4000] for c in cs])
        centroid = V.mean(axis=0)
        centroid = centroid / (float(np.linalg.norm(centroid)) + 1e-12)
        out[skill] = cs[int(np.argmax(V @ centroid))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="total items across all pools")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--resume", action="store_true",
                    help="append to --out and skip already-satisfied quotas (never truncates)")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent generation calls; owner cap is 4 (GPU contention)")
    ap.add_argument("--batch", type=int, default=BATCH_SIZE,
                    help="candidates requested per call; each is still gated individually")
    ap.add_argument("--force-overwrite", action="store_true", dest="force_overwrite",
                    help="DESTROY an existing --out pool and start fresh (hours of GPU); "
                         "without this the run refuses rather than truncate")
    ap.add_argument("--only-category", default=None, dest="only_category",
                    help="generate ONLY this category (Math|Code|STEM|Instruction); "
                         "for repairing a flavour deficit that generic rotation preserves")
    ap.add_argument("--chunks", default=None,
                    help="override the chunk source (e.g. the finer split); skills must "
                         "still match TARGET_SKILLS exactly")
    args = ap.parse_args()

    global CHUNKS
    if args.chunks:
        CHUNKS = args.chunks
        print("chunk source: %s (override)" % CHUNKS)

    chunks = load_chunks()
    # Was `== 24`. The count is a property of the chunk FILE, not a law: §1.6's remedy for
    # gate saturation is more source chunks, and `split_chunks_finer.py` turns the same 24
    # passages into 42 without changing skills or content. What must NOT drift is the skill
    # set -- that is what the eval probes -- so the coverage assert below stays exact.
    assert len(chunks) >= 24, "expected at least 24 chunks, got %d" % len(chunks)
    covered = {c["skill"] for c in chunks}
    assert covered == set(TARGET_SKILLS), (
        "chunk coverage does not match TARGET_SKILLS; missing=%s extra=%s"
        % (sorted(set(TARGET_SKILLS) - covered), sorted(covered - set(TARGET_SKILLS)))
    )
    # One HOMOGENEOUS exemplar block per category: a Math prompt is shown only Math
    # exemplars, a Code prompt only Code. Mixing types inside one block makes the
    # demonstration contradict the CATEGORY instruction.
    by_cat = style_exemplars_by_category()
    style_blocks: dict[str, str] = {}
    for _cat, _ex in by_cat.items():
        style_blocks[_cat] = (
            "Style reference — real %s questions (phrasing only, different topics):\n" % _cat
            + "\n".join("  - %s" % e for e in _ex) + "\n"
        ) if _ex else ""
    print("few-shot exemplars per category: %s"
          % {c: len(by_cat.get(c, [])) for c in CATEGORY_ORDER})

    accepted: list[dict] = []
    vecs: np.ndarray | None = None
    stats = {"regen": 0, "deixis": 0, "codedeixis": 0, "schema": 0,
             "cosine": 0, "parse": 0, "miss": 0}
    t0 = time.time()

    # --- resume (owner-critical durability) -------------------------------------------
    # The generator has always opened its output "w", so every restart TRUNCATED the pool.
    # That cost 142 items once and smaller amounts several times over, and it is what made
    # the run5 zombie so destructive. With --resume the accepted items, their embeddings
    # (re-embedded, since vectors are not persisted) and the id counter are reloaded, the
    # file is opened for APPEND, and already-satisfied (chunk, category) quotas are skipped.
    resumed = 0
    if args.resume and os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                # A partial/corrupt tail is a fact about the file, not something to paper
                # over: refuse rather than silently resume onto damaged data.
                raise SystemExit(
                    "REFUSING TO RESUME: %s contains an unparseable line. Inspect and "
                    "quarantine it first (a zombie writer produced exactly this once)."
                    % args.out)
            if rec.get("question") and rec.get("pool"):
                accepted.append(rec)
        if accepted:
            vecs = embed([a["question"] for a in accepted])
            gen_index = max(int(a.get("generation_index", 0)) for a in accepted) + 1
            resumed = len(accepted)
            print("RESUMED %d items from %s (gen_index -> %d)"
                  % (resumed, args.out, gen_index))
            print("  pools: %s" % dict(collections.Counter(a["pool"] for a in accepted)))

    # REFUSE TO TRUNCATE AN EXISTING POOL.
    #
    # This file opened its output "w" unconditionally for its whole life, so any run
    # without --resume destroyed the pool instantly. That cost 142 items on 2026-08-26 and
    # smaller amounts repeatedly. A generated dataset is expensive (hours of GPU) and
    # irreproducible (LLM sampling is not seeded), so overwriting one must be a DELIBERATE
    # act, never the default behaviour of a mistyped command.
    if not resumed and os.path.exists(args.out) and not args.force_overwrite:
        existing = sum(1 for l in open(args.out, encoding="utf-8") if l.strip())
        raise SystemExit(
            "REFUSING TO OVERWRITE %s (%d existing items).\n"
            "  To ADD to it:      --resume\n"
            "  To replace it:     --force-overwrite  (destroys the pool; back it up first)\n"
            "  Manifest + hashes: data/rl/pools_r3.MANIFEST.json"
            % (args.out, existing))

    fout = open(args.out, "a" if resumed else "w", encoding="utf-8")

    # DERIVE the counter from what is already accepted -- never reset it here.
    # A bare `gen_index = 0` sat AFTER the resume block and silently wiped the resumed
    # value, so a resumed run restarted ids at r3-0000 and wrote FOUR records that
    # collided with existing ones. Ids are the only handle for tracing or excising an
    # item, so duplicates corrupt provenance even though every count still looks right.
    gen_index = (max(int(a.get("generation_index", -1)) for a in accepted) + 1
                 if accepted else 0)

    def emit(c: dict, pool: str, category: str) -> bool:
        """Generate one accepted item from chunk `c` into `pool`, in `category`.

        Returns False if MAX_TRIES was exhausted without a candidate that
        clears the deixis and cosine gates.
        """
        nonlocal vecs, gen_index
        excl = ""
        if accepted:
            excl = "Exclusion list (do NOT duplicate or paraphrase):\n" + "\n".join(
                "  - %s" % a["question"] for a in accepted
            ) + "\n"
        prompt = TEMPLATE.format(
            chunk=c["text"][:4000],
            style=style_blocks.get(category, ""),
            exclusion=excl,
            category=R3_CATEGORIES[category],
        )

        cand = None
        maxcos = 0.0
        cvec = None
        for attempt in range(MAX_TRIES):
            try:
                raw = _chat_model(prompt, SYSTEM,
                              CATEGORY_MODELS.get(category, CATEGORY_MODEL_FALLBACK),
                              temperature=0.7 + 0.06 * attempt, tries=2,
                              num_predict=num_predict_for(1))
            except RemoteMiss:
                # Booked separately from `parse`: a miss says the ENDPOINT is not answering,
                # a parse failure says the MODEL answered badly. Conflating them sends you
                # tuning prompts when the real fault is the model id or the quota.
                stats["miss"] += 1
                stats["regen"] += 1
                continue
            d = parse_json(raw)
            if d is None:
                stats["parse"] += 1
                stats["regen"] += 1
                continue
            why = gate_reason(d["question"], d.get("answer", ""))
            if why:
                stats[why] += 1
                stats["regen"] += 1
                continue
            v = embed([d["question"]])
            mc = 0.0 if vecs is None else float((vecs @ v[0]).max())
            if mc >= COSINE_MAX:
                stats["cosine"] += 1
                stats["regen"] += 1
                continue
            cand, maxcos, cvec = d, mc, v
            break

        if cand is None:
            print("  !! chunk %s: gave up after %d tries" % (c["chunk_id"], MAX_TRIES))
            return False

        rec = {
            "id": "r3-%04d" % gen_index,
            "pool": pool,
            "regime": "R3",
            "question": cand["question"],
            "gold": cand["answer"],
            "category": category,
            "source_skill": c["skill"],
            "source_chunk_id": c["chunk_id"],
            "generation_index": gen_index,
            "max_cosine_to_prior": round(maxcos, 4),
            # STAMP THE MODEL THAT ACTUALLY RAN, not the routing table's local id.
            # CATEGORY_MODELS maps a category to an OLLAMA id; when GEN_BACKEND is
            # openrouter (or nvidia) that id never ran, and writing it makes provenance a
            # lie -- the failure the module header calls out for the nvidia path. Verified
            # 2026-08-29: two OpenRouter items were stamped "qwen3.5-...-256k:9b" before
            # this fix.
            "gen_model": (resolved_gen_model() if GEN_BACKEND != "ollama"
                          else CATEGORY_MODELS.get(category, CATEGORY_MODEL_FALLBACK)),
        }
        accepted.append(rec)
        vecs = cvec if vecs is None else np.vstack([vecs, cvec])
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        gen_index += 1
        print("[%3d/%3d] %-22s %s %-11s | cos=%.2f | %s"
              % (gen_index, args.n, c["chunk_id"], pool, category, maxcos,
                 cand["question"][:60]))
        return True

    # ---- Phase C FIRST: the eval set, one item per skill ----------------
    # C must precede A/B. Every training item generated afterwards is
    # cosine-tested against the eval questions already in `vecs`, so
    # contamination is *rejected at generation time* rather than
    # discovered afterwards when the only fix is to throw work away.
    # On --resume the eval already exists and is FROZEN (§1.4). Regenerating it would both
    # grow pool C past one-per-skill and silently swap the instrument mid-experiment, so a
    # resumed run must skip this phase entirely rather than "top it up".
    have_c = {a["source_skill"] for a in accepted if a["pool"] == "C"}
    if have_c >= set(EVAL_SKILLS):
        print("\n=== PHASE C already complete (%d skills) — skipping (resumed) ==="
              % len(have_c))
    else:
        # EVAL skills only — pool C must not acquire an item for a skill that was added
        # to the TRAINING corpus to relieve gate saturation.
        central = central_chunks_per_skill(
            [c for c in chunks if c["skill"] in set(EVAL_SKILLS)])
        print("\n=== PHASE C (eval, one item per skill) ===")
        for i, skill in enumerate(sorted(central)):
            if skill in have_c:
                continue
            emit(central[skill], "C", CATEGORY_ORDER[i % len(CATEGORY_ORDER)])

    # FAIL CLOSED. Pool C is the measuring instrument: §1.6 says one R3 item per skill,
    # and Gate A reads per-skill rates off it. A skill that gave up here yields an eval
    # that silently cannot measure that skill, while every downstream count still looks
    # plausible -- so this must abort the run, not warn. (Run7 hit exactly this: the
    # single-item parser had no <think> stripping, ensemble-method burned all 6 tries as
    # parse failures, and generation carried on into A+B with a 2-skill eval.)
    c_skills = {a["source_skill"] for a in accepted if a["pool"] == "C"}
    missing = sorted(set(EVAL_SKILLS) - c_skills)
    assert not missing, (
        "PHASE C INCOMPLETE — no eval item for %s. Pool C must cover every skill "
        "(§1.6); refusing to build training pools against a broken eval." % missing
    )
    # HARD CAP, not just a skip. The skip above is an optimisation; THIS is the guarantee.
    # A silently-failed edit meant the skip did not apply on the first --resume attempt and
    # the run began regenerating eval items -- pool C would have reached 6, breaking §1.6
    # and swapping the frozen instrument mid-experiment. A guard that depends on another
    # branch working is not a guard.
    n_c = sum(1 for a in accepted if a["pool"] == "C")
    assert n_c == POOL_C_SIZE, (
        "POOL C SIZE VIOLATION — %d items, expected exactly %d (one per skill, §1.6). "
        "The eval is frozen; refusing to continue." % (n_c, POOL_C_SIZE)
    )
    print("phase C complete: %d/%d eval skills covered (pool C = %d)"
          % (len(c_skills), len(EVAL_SKILLS), n_c))

    # ---- Phases A (CPT) and B (SFT/RL), INTERLEAVED ----------------------
    # plan_tasks() is the starvation fix (§6.11): it emits A,B,A,B so both pools face the
    # same gate pressure at the same time. It existed and was unit-tested but main() never
    # called it -- main still ran the old A-to-exhaustion-then-B loop that produced
    # A=54 / B=6-from-one-skill. Wired up 2026-08-26.
    budget = max(0, args.n - len(accepted))
    plan = plan_tasks(chunks, budget)

    def fetch_batch(c: dict, category: str) -> str:
        """The I/O half: one batched call, no state mutation. Safe to run in parallel."""
        model = CATEGORY_MODELS.get(category, CATEGORY_MODEL_FALLBACK)
        excl = ""
        if accepted:
            recent = [a["question"] for a in accepted][-60:]
            excl = "Exclusion list (do NOT duplicate or paraphrase):\n" + "\n".join(
                "  - %s" % q for q in recent) + "\n"
        prompt = BATCH_TEMPLATE.format(
            chunk=c["text"][:4000], n=args.batch, category=R3_CATEGORIES[category],
            style=style_blocks.get(category, ""), exclusion=excl)

        if GEN_BACKEND == "openrouter":
            # SCHEMA-CONSTRAINED PATH. Returns the same STRING shape the ollama path does,
            # so `parse_batch` and `gate_candidates` are untouched -- the only new
            # behaviour is the schema, the fact-index dedup and the empty retry.
            #
            # RETRY ON EMPTY (REQ-G8): 2 of 26 sampled chunks returned zero items. A silent
            # skip loses a whole chunk's capacity, which at ~5 accepted each is not a
            # rounding error.
            schema = batch_schema(args.batch, [category])
            for attempt in range(3):
                out = _openrouter_structured(prompt, BATCH_SYSTEM, schema,
                                             temperature=0.8 + 0.05 * attempt)
                items = dedupe_by_fact((out or {}).get("items"), stats)
                if items:
                    return json.dumps(items, ensure_ascii=False)
                stats["regen"] += 1
            raise RemoteMiss("openrouter returned no usable items after 3 tries")

        return _chat_model(prompt, BATCH_SYSTEM, model,
                           num_predict=num_predict_for(args.batch))

    def gate_candidates(c: dict, category: str, want: int, raw: str,
                        pool_cycle: list[str]) -> int:
        """The stateful half: gate + accept. MUST run serially (see run_wave)."""
        nonlocal vecs, gen_index
        model = CATEGORY_MODELS.get(category, CATEGORY_MODEL_FALLBACK)
        cands = parse_batch(raw)
        if not cands:
            stats["parse"] += 1
            print("  !! %s [%s] no parseable items" % (c["chunk_id"], category))
            return 0

        got = 0
        for d in cands:
            if got >= want:
                break
            why = gate_reason(d["question"], d.get("answer", ""))
            if why:
                stats[why] += 1
                continue
            v = embed([d["question"]])
            mc = 0.0 if vecs is None else float((vecs @ v[0]).max())
            if mc >= COSINE_MAX:
                stats["cosine"] += 1
                continue
            pool = pool_cycle.pop(0)
            rec = {
                "id": "r3-%04d" % gen_index,
                "pool": pool,
                "regime": "R3",
                "question": d["question"],
                "gold": d["answer"],
                "category": category,
                "source_skill": c["skill"],
                "source_chunk_id": c["chunk_id"],
                "generation_index": gen_index,
                "max_cosine_to_prior": round(mc, 4),
                # Same rule as above: `model` is the routing-table id, which is only
                # the truth on the local backend.
                "gen_model": (resolved_gen_model() if GEN_BACKEND != "ollama" else model),
            }
            accepted.append(rec)
            vecs = v if vecs is None else np.vstack([vecs, v])
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            gen_index += 1
            got += 1
            print("[%3d/%3d] %-22s %s %-11s | cos=%.2f | %s"
                  % (gen_index, args.n, c["chunk_id"], pool, category, mc,
                     d["question"][:60]))
        stats["regen"] += len(cands) - got
        return got

    def emit_batch(c: dict, category: str, want: int, pool_cycle: list[str]) -> int:
        """Serial fetch+gate for a single job (the width-1 wave)."""
        try:
            raw = fetch_batch(c, category)
        except Exception as e:
            stats["miss"] += 1
            print("  !! %s [%s] call failed: %s" % (c["chunk_id"], category, type(e).__name__))
            return 0
        return gate_candidates(c, category, want, raw, pool_cycle)

    # Work list: one (chunk, category) job per plan entry, deduped -- the plan says how many
    # items each chunk/category owes, and one batched call can satisfy several of them.
    jobs: dict[tuple[str, str], int] = collections.OrderedDict()
    job_chunk: dict[str, dict] = {}
    pool_cycle: list[str] = []
    for c, pool, cat in plan:
        jobs[(c["chunk_id"], cat)] = jobs.get((c["chunk_id"], cat), 0) + 1
        job_chunk[c["chunk_id"]] = c
        pool_cycle.append(pool)

    # Order jobs by MODEL, keeping the interleaved order within each model group.
    #
    # Category rotation is right for pool/flavour balance, but with a different model per
    # category it would swap two ~8 GB models on a 16 GB card on nearly every call -- ~95
    # reloads. Grouping by model costs one swap total while preserving the A/B interleave
    # and chunk rotation inside each group.
    # Resume: an already-satisfied (chunk, category) quota must not be regenerated.
    done = collections.Counter((a.get("source_chunk_id"), a.get("category"))
                               for a in accepted if a["pool"] in ("A", "B"))
    for key in list(jobs):
        jobs[key] = max(0, jobs[key] - done.get(key, 0))
        if jobs[key] == 0:
            del jobs[key]

    # Targeted top-up of ONE category.
    #
    # The category axis only delivers its diversity if the four flavours are actually
    # balanced. MEASURED 2026-08-26 at 200 items: Code 19 vs STEM 65 / Instruction 69 /
    # Math 44, because Code was routed to the 9x-slower qwopus model for most of the run.
    # Generic generation cannot repair that -- it rotates all four evenly and preserves
    # the deficit. Only a category-restricted pass closes it.
    if args.only_category:
        want_cat = args.only_category
        assert want_cat in R3_CATEGORIES, "unknown category %r" % want_cat
        # BUILD the job list, do not FILTER it. plan_tasks() rotates categories by chunk
        # index, so only ~1/4 of chunks are ever assigned a given flavour -- filtering that
        # plan yielded ZERO jobs. A deficit repair must ask EVERY chunk for the missing
        # flavour, which is precisely what generic rotation will not do.
        per = max(1, -(-budget // max(1, len(chunks))))   # ceil
        jobs = collections.OrderedDict(
            ((c["chunk_id"], want_cat), per) for c in chunks)
        job_chunk = {c["chunk_id"]: c for c in chunks}
        pool_cycle = [("A" if i % 2 == 0 else "B")
                      for i in range(sum(jobs.values()) + 8)]
        print("ONLY-CATEGORY=%s -> %d jobs over %d chunks (%d each)"
              % (want_cat, len(jobs), len(chunks), per))

    # --- GLOBAL CATEGORY QUOTA (deficit-driven `want`) --------------------------------
    #
    # `want` used to be the plan's per-(chunk, category) count, which is ~1. A batched call
    # asks for --batch (16) candidates, so gate_candidates stopped accepting after the first
    # one and the other ~15 were thrown away UNGATED: measured on the 2026-08-29 smoke run,
    # regenerations=49 (parse=0 deixis=0 cosine=1 miss=0) for 3 accepted items. Per-chunk
    # capacity at ask-16 is ~9 accepted, so the plan was discarding most of the yield.
    #
    # `want` is now the REMAINING GLOBAL QUOTA for that category -- so a batch is allowed to
    # fill as much of the category's deficit as it can clear the gates for, and the natural
    # cap is the number of candidates the call returned. Quota state is derived from
    # `accepted` on every read, never held in a counter, so --resume needs no extra state.
    quota = category_quota(args.n)

    def cat_have() -> dict:
        h = collections.Counter(a.get("category") for a in accepted)
        return {c: h.get(c, 0) for c in CATEGORY_ORDER}

    def want_for(cat: str) -> int:
        """Remaining global quota for `cat`. 0 means the category is done."""
        return max(0, quota.get(cat, 0) - cat_have().get(cat, 0))

    def cat_deficit(cat: str) -> float:
        """Normalised shortfall, the same measure deficit_draw() in partition_r3_pools.py
        uses: categories with different targets have to compete on one scale."""
        t = quota.get(cat, 0)
        return (t - cat_have().get(cat, 0)) / max(t, 1e-9)

    print("\ncategory quota (target / accepted / remaining):")
    for c in CATEGORY_ORDER:
        print("  %-12s %4d %4d %4d" % (c, quota[c], cat_have()[c], want_for(c)))

    by_model = collections.Counter(CATEGORY_MODELS.get(c, CATEGORY_MODEL_FALLBACK)
                                   for (_, c) in jobs)
    print("\n=== PHASES A+B — %d items via %d batched calls (batch=%d) ==="
          % (budget, len(jobs), args.batch))
    for m, n in by_model.items():
        print("   %-42s %d calls" % (m, n))

    # Concurrency, capped at --workers (owner: "no more than 4 at a time" -- we are
    # contending with the owner's own use of the same GPU).
    #
    # The split matters: the CALLS run in parallel because they are pure I/O against
    # ollama, but the GATING stays serial. Cosine acceptance is order-dependent -- each
    # item is tested against everything already accepted, including items accepted
    # earlier in the same wave -- so gating concurrently would let two near-duplicates
    # both pass by racing each other's insert.
    #
    # `want` is resolved at GATE time, not when the wave is submitted: two jobs in the same
    # category can be in flight at once, and the second must see what the first accepted.
    def run_wave(wave: list[tuple[tuple[str, str], int]]) -> None:
        if len(wave) == 1:
            ((cid, cat), _) = wave[0]
            emit_batch(job_chunk[cid], cat, want_for(cat), pool_cycle)
            return
        with cf.ThreadPoolExecutor(max_workers=len(wave)) as ex:
            futs = {
                ex.submit(fetch_batch, job_chunk[cid], cat): (cid, cat)
                for (cid, cat), _ in wave
            }
            done = []
            for fut in cf.as_completed(futs):
                cid, cat = futs[fut]
                try:
                    done.append((job_chunk[cid], cat, fut.result()))
                except Exception as e:
                    stats["miss"] += 1
                    print("  !! %s [%s] call failed: %s"
                          % (cid, cat, type(e).__name__))
            for c, cat, raw in done:
                w = want_for(cat)
                if w <= 0:
                    continue
                gate_candidates(c, cat, w, raw, pool_cycle)

    def skill_have() -> collections.Counter:
        """Accepted items per source skill, derived from `accepted` on every read so a
        --resume run picks up the spread it already has instead of restarting it."""
        return collections.Counter(a.get("source_skill") for a in accepted)

    def job_skill(cid: str) -> str:
        return job_chunk[cid].get("skill", "")

    # DEFICIT-DRIVEN SCHEDULING with SKILL SPREAD. The category furthest below its quota is
    # asked first, and the ordering is recomputed after every wave because acceptance moves
    # the deficits. A category whose quota is already met is dropped from the work list
    # rather than generated and discarded. Ties keep the model grouping (one 8 GB model
    # swap, not ~95).
    #
    # Category deficit alone drains ONE skill at a time: every job in a category ties on
    # deficit, so the original file order decides, and a run measured at 168 items put 58
    # of them (35%) through `expected-gini` while all 29 fresh round-2/3 chunks got zero
    # calls. Two costs: it breaks the spread rule, and it burns yield -- consecutive
    # same-skill chunks clear the 0.80 cosine gate at 52% against 65% cross-skill (measured
    # 2026-08-29 over 6 `boosted-tree` chunks: 62%->38% as mean max-cosine rose
    # 0.790->0.865). So the least-used skill goes first inside the neediest category.
    #
    # Two counters, because they are in different units. `picked` counts CALLS and is
    # advanced the moment a job is chosen, which is what actually makes the order
    # round-robin -- a wave of 4 ranked on accepted items alone would happily be 4 chunks
    # of the same skill, since none of them has returned anything yet. `skill_have()`
    # counts ITEMS off `accepted` and breaks the round-robin tie, so a --resume run leads
    # with whichever skill is genuinely behind instead of restarting the rotation blind.
    picked: collections.Counter = collections.Counter()
    pending = list(jobs.items())
    while pending:
        # Stop on the TOTAL target, not on `budget`. `budget` is the REMAINING work
        # (args.n - already accepted), so comparing cumulative accepted against it is a
        # unit mismatch: on a resumed run with 142 items and budget 98, 142-3 >= 98 was
        # true on the first iteration and the run exited having generated nothing.
        if len(accepted) >= args.n:
            break
        pending = [kv for kv in pending if want_for(kv[0][1]) > 0]
        if not pending:
            print("all category quotas met -- nothing left to generate")
            break
        have = skill_have()
        wave: list = []
        while pending and len(wave) < args.workers:
            pending.sort(key=lambda kv: (-cat_deficit(kv[0][1]),
                                         picked[job_skill(kv[0][0])],
                                         have[job_skill(kv[0][0])],
                                         CATEGORY_MODELS.get(kv[0][1],
                                                             CATEGORY_MODEL_FALLBACK)))
            kv = pending.pop(0)
            wave.append(kv)
            picked[job_skill(kv[0][0])] += 1
        run_wave(wave)

    fout.close()
    dt = time.time() - t0

    # ---------------- receipt (§6.11) ----------------
    print("\n" + "=" * 72)
    # Report the models that ACTUALLY produced the items, read off the records --
    # resolved_gen_model() returns the single global fallback id, which is simply wrong
    # once generation is per-category. Same provenance-lie that mislabelled all 78 items
    # in an earlier run; a receipt that names the wrong model is worse than none.
    used = collections.Counter(a.get("gen_model", "?") for a in accepted)
    print("RECEIPT — %s  (%.1f min)" % (args.out, dt / 60))
    for m, k in used.most_common():
        print("   %-46s %d items" % (m, k))
    print("=" * 72)

    pc = collections.Counter(a["pool"] for a in accepted)
    print("per-pool counts:", dict(pc))
    for p in ("C", "A", "B"):
        sk = collections.Counter(a["source_skill"].split("/")[-1] for a in accepted if a["pool"] == p)
        ch = len({a["source_chunk_id"] for a in accepted if a["pool"] == p})
        print("  %s: n=%-3d chunks=%-3d skills=%s" % (p, pc[p], ch, dict(sk)))

    # Per-pool x per-category. A pool that is 90% one flavour has not actually used the
    # category axis, and since that axis is what buys the cosine headroom (§3.5.1a), a
    # lopsided table is the early warning that the next run will saturate again.
    print("\ncategory coverage (pool x category):")
    print("  %-4s %s" % ("pool", " ".join("%-12s" % c for c in CATEGORY_ORDER)))
    for p in ("C", "A", "B"):
        cc = collections.Counter(
            a.get("category") for a in accepted if a["pool"] == p
        )
        print("  %-4s %s" % (p, " ".join("%-12d" % cc.get(c, 0) for c in CATEGORY_ORDER)))
    # Quota outturn: what the run was asked for per category, what it got, what is still
    # owed. This is the line that says whether the deficit-driven schedule actually closed.
    final_have = collections.Counter(a.get("category") for a in accepted)
    print("\ncategory quota (target / accepted / remaining):")
    for c in CATEGORY_ORDER:
        print("  %-12s %4d %4d %4d"
              % (c, quota[c], final_have.get(c, 0),
                 max(0, quota[c] - final_have.get(c, 0))))

    missing = [c for c in CATEGORY_ORDER
               if not any(a.get("category") == c for a in accepted)]
    if missing:
        print("  !! categories with ZERO items: %s" % ", ".join(missing))

    if vecs is not None and len(accepted) > 1:
        S = vecs @ vecs.T
        np.fill_diagonal(S, -1.0)
        idx = {p: [i for i, a in enumerate(accepted) if a["pool"] == p] for p in "CAB"}

        def blk(p, q):
            i, j = idx[p], idx[q]
            if not i or not j:
                return float("nan")
            return float(S[np.ix_(i, j)].max())

        print("max cosine within-A=%.3f within-B=%.3f within-C=%.3f"
              % (blk("A", "A"), blk("B", "B"), blk("C", "C")))
        print("max cosine  A*B=%.3f  A*C=%.3f  B*C=%.3f"
              % (blk("A", "B"), blk("A", "C"), blk("B", "C")))
        print("GATE (all pairs < %.2f): %s" % (COSINE_MAX, "PASS" if S.max() < COSINE_MAX else "FAIL"))

    print("regenerations=%d (parse=%d deixis=%d codedeixis=%d schema=%d cosine=%d miss=%d)"
          % (stats["regen"], stats["parse"], stats["deixis"], stats["codedeixis"],
             stats["schema"], stats["cosine"], stats["miss"]))
    if stats["miss"]:
        print("  !! %d regenerations were ENDPOINT MISSES, not content rejections -- the pool "
              "below is short because inference failed, NOT because the corpus saturated"
              % stats["miss"])

    # disjointness assertion in code, not assumed (SO-2)
    qs = collections.Counter(a["question"].strip().lower() for a in accepted)
    dupes = [q for q, n in qs.items() if n > 1]
    print("exact duplicate questions:", len(dupes))
    assert not dupes, "duplicate questions leaked across pools"

    poolC = [a for a in accepted if a["pool"] == "C"]
    h = hashlib.sha256(
        json.dumps([a["question"] for a in sorted(poolC, key=lambda x: x["id"])],
                   ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    print("POOL C EVAL HASH: %s  (n=%d)" % (h, len(poolC)))


if __name__ == "__main__":
    main()
