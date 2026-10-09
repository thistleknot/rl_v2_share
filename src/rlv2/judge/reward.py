"""
Stage 6: Reward function — regime-branched, returns float in [0, 1].
Entry point: reward(answer, context, regime, gold) -> float
Also exports: nli_entailment(premise, hypothesis) -> float  (used by score_baseline.py)
"""
from __future__ import annotations

import os
import re
import math
import torch
import torch.nn.functional as F
from rlv2.common.config import ABSTAIN_R2, ABSTAIN_R4

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from model output before scoring.

    Spec: .spec/specs/r3-domain-injection/tasks.md #35 (empty-answer guard).

    AN UNTERMINATED <think> IS NOT AN ANSWER. Measured 2026-09-11 on the A100:
    a trial read 16/16 where every one of the sixteen completions was a `<think>`
    block cut off by the token budget -- no `</think>`, no answer text. The old
    regex required a CLOSING tag, so nothing was stripped, the raw reasoning was
    handed to the judge as if it were the answer, and the judge paid >=0.70 on
    all sixteen. The empty-answer guard downstream never fired because the string
    was not empty. So: drop everything from an unclosed `<think>` onward, which
    leaves "" and lets that guard score it 0.
    """
    import re as _re
    out = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL)
    cut = out.find("<think>")
    if cut != -1:
        out = out[:cut]
    return out.strip()


JUDGE_MODEL       = "qwen3.5-reasoning-thinking-256k:4b"   # THE ONLY JUDGE.
# THE 2B IS DELETED (operator, 2026-09-12: "remove 2b from existence"). It was
# the vibe lane default and therefore what ANY caller got that did not first
# call set_judge_lane("r3") -- a silent wrong-judge path. There is now one
# judge model in this file and selecting a lane cannot change which model runs.
                                      # as the R3 judge (accepted 3/12 fluent-wrong decoys the 4b and
                                      # 9b both caught) -- see playbook "The judge is the 4b" and S-12.
                                      # The R3 lane never uses this constant; see set_judge_lane().
JUDGE_MODEL_EVAL  = "qwen3.5-reasoning-thinking-256k:4b"   # same judge; see above.
JUDGE_MODEL_HEAVY = "qwen3.5-reasoning-thinking-256k:9b"   # RUBRIC AUTHORING ONLY; never scores (renamed from qwen3.5-reasoning-oc:9b)
JUDGE_ENDPOINT = "http://localhost:11434/api/generate"

# ── JUDGE LANE ────────────────────────────────────────────────────────────────────────────
# Two lanes, two instruments, BY DESIGN (they must not be merged -- REQ-J10):
#   "vibe" : 2b + _VIBETHINKER_RUBRIC. 6 domains, context-dependent, abstention a first-class
#            verdict. The future R1/R2/R4 direction. Default, so nothing older moves.
#   "r3"   : 4b + data/synthetic_domain_rubric.md. Closed-book, NO abstention, every question
#            answerable. Closes S-12: RL optimises AGAINST its reward judge across thousands
#            of rollouts, so the fluency-fooled 2b does not merely mis-score there -- it
#            teaches the policy that confident-sounding wrong answers pay.
# The floor is imported, not copied: 0.70 lives in judge.py and nowhere else.
from rlv2.judge.judge import CORRECT_FLOOR as CORRECT_FLOOR  # noqa: E402  deliberate re-export for mgpo (imports reward as R)
assert 0.0 < CORRECT_FLOOR < 1.0  # also marks the re-export as used for pyflakes

_JUDGE_LANE = "vibe"
_R3_RUBRIC_PATH = "data/synthetic_domain_rubric.md"
_R3_RUBRIC_CACHE = None


def set_judge_lane(lane: str) -> None:
    """Select which judge (model + rubric) the training-reward path uses. Loud on purpose."""
    global _JUDGE_LANE
    if lane not in {"vibe", "r3"}:
        raise ValueError(f"unknown judge lane {lane!r}; expected 'vibe' or 'r3'")
    _JUDGE_LANE = lane
    # NAME THE JUDGE THAT ACTUALLY RUNS. This printed _judge_model() -- the LOCAL
    # default -- unconditionally, so under JUDGE_REMOTE=1 the line announced
    # "qwen3.5-reasoning-thinking-256k:4b" while every call went to OpenRouter's
    # qwen/qwen3.5-9b. The instrument was right and the LABEL was wrong, which is
    # worse than no label: the RL sweep's log read as a 4b run beside an eval that
    # said 9b, inviting the conclusion that the two phases used different graders
    # (caught 2026-09-18). Identical fix already made in src/rlv2/eval/score_arms.py.
    # JUDGE_JEV overrides every lane in llm_judge_batch_train, so it must win here too
    # (2026-09-26: the Jev run's first log line named the 9B it was not using).
    if _jev_enabled():
        _url = os.environ.get("JEV_URL", "http://192.168.3.18:8791")
        print(f"[judge-lane] {lane}: model=open-jev-2b (binary yes/no, {_url}) rubric=none "
              f"(Noul: 'Is the candidate answer correct according to the reference answer?')",
              flush=True)
        return
    _live = (f"{OPENROUTER_JUDGE_MODEL} (openrouter)" if _openrouter_enabled()
             else f"{_judge_model()} (local)")
    print(f"[judge-lane] {lane}: model={_live} rubric="
          f"{'synthetic_domain_rubric.md' if lane == 'r3' else '_VIBETHINKER_RUBRIC'}",
          flush=True)


def _judge_model() -> str:
    """The judge model. Both lanes now resolve to the same 4B -- there is no
    configuration in which a smaller judge scores anything."""
    from rlv2.judge import judge as _J               # single source of truth: judge.py JUDGE_MODEL
    return _J.JUDGE_MODEL


def _rubric() -> str:
    if _JUDGE_LANE != "r3":
        return _VIBETHINKER_RUBRIC
    global _R3_RUBRIC_CACHE
    if _R3_RUBRIC_CACHE is None:
        import os as _os
        import re as _re
        path = _R3_RUBRIC_PATH
        if not _os.path.exists(path):                    # resolve from repo root when cwd differs
            path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))),
                                 _R3_RUBRIC_PATH)
        text = open(path, encoding="utf-8").read()
        # Strip the HTML provenance comments (context cost, not instruction) and the rubric's
        # own trailing `Return: {...}` line -- each judge call appends its OWN output-format
        # instruction, and two competing JSON shapes in one prompt is a measured failure mode.
        text = _re.sub(r"<!--.*?-->", "", text, flags=_re.DOTALL)
        text = chr(10).join(l for l in text.splitlines() if not l.strip().startswith("Return:"))
        _R3_RUBRIC_CACHE = text.strip()
    return _R3_RUBRIC_CACHE

from rlv2.judge.verifiers import verify_math

VERIFY_FLOOR = 0.7        # VERIFIER-verdict floor, tunable. NOT the judge's 0.70 line --
                          # that is judge.CORRECT_FLOOR, single home; coincidence of value only.
QUALITY_WEIGHT = 0.3      # quality band width above the floor; tunable
VERIFIABLE_DOMAINS = {"Math"}   # "Code" added in Phase B

def verify_verdict(domain: str, answer: str, gold: str, source: str = "", record=None):
    """Programmatic correctness verdict for verifiable domains. True/False/None."""
    if domain == "Math":
        return verify_math(answer, gold)
    # Code -> Phase B (returns None for now -> judge fallback)
    return None

def compose_reward(verdict, quality: float) -> float:
    """verdict True -> floor + weight*quality; False -> 0.0; None -> quality."""
    if verdict is True:
        return VERIFY_FLOOR + QUALITY_WEIGHT * float(quality)
    if verdict is False:
        return 0.0
    return float(quality)


# ── Red Queen: hard abstention verify-gate ───────────────────────────────────
# OFF by default so the proven baseline reward path is byte-identical. The Red Queen
# outer loop flips ABSTENTION_GATE=True when the soft rubric rule fails to close the
# over-abstention exploit: an ANSWERABLE item (non-empty gold) that the policy abstains
# on is then forced wrong (verdict False -> 0.0), un-gameable by construction — exactly
# like the verify_math gate, but for refusing to answer an answerable question.
ABSTENTION_GATE = False

def _looks_abstain(text: str) -> bool:
    t = (text or "").lower()
    return ("unanswerable" in t) or ("not supported by context" in t) or ("i cannot answer" in t)

def abstention_verdict(gold: str, answer: str):
    """False when the gate is on and an answerable item is abstained on; else None."""
    if not ABSTENTION_GATE:
        return None
    if (gold or "").strip() and _looks_abstain(answer):
        return False
    return None

_nli_tok   = None
_nli_model = None


def _load_nli() -> None:
    global _nli_tok, _nli_model
    if _nli_model is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        _nli_tok   = AutoTokenizer.from_pretrained(NLI_MODEL)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        _nli_model.eval()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def _f1(pred: str, gold: str) -> float:
    p, g = set(_norm(pred).split()), set(_norm(gold).split())
    if not p or not g:
        return 0.0
    common = p & g
    if not common:
        return 0.0
    prec = len(common) / len(p)
    rec  = len(common) / len(g)
    return 2 * prec * rec / (prec + rec)


def _synset_sim(pred: str, gold: str) -> float:
    """Fraction of gold tokens covered by synset/hypernym match in pred via WordNet."""
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        return 0.0
    pred_words = set(_norm(pred).split())
    gold_words = _norm(gold).split()
    if not gold_words or not pred_words:
        return 0.0
    covered = 0
    for g_word in gold_words:
        if g_word in pred_words:
            covered += 1
            continue
        g_synsets = wn.synsets(g_word)
        if not g_synsets:
            continue
        g_lemmas = {l.name().lower() for s in g_synsets for l in s.lemmas()}
        g_hypernyms = {l.name().lower() for s in g_synsets for h in s.hypernyms() for l in h.lemmas()}
        for p_word in pred_words:
            if p_word in g_lemmas or p_word in g_hypernyms:
                covered += 1
                break
            p_synsets = wn.synsets(p_word)
            if not p_synsets:
                continue
            if g_lemmas & {l.name().lower() for s in p_synsets for l in s.lemmas()}:
                covered += 1
                break
    return covered / len(gold_words)


def nli_entailment(premise: str, hypothesis: str) -> float:
    """Return probability that premise entails hypothesis (0–1)."""
    if not premise or not hypothesis:
        return 0.0
    _load_nli()
    enc = _nli_tok(
        premise, hypothesis,
        return_tensors="pt", truncation=True, max_length=512,
    )
    with torch.no_grad():
        logits = _nli_model(**enc).logits
    probs = F.softmax(logits, dim=-1).squeeze()
    for idx, label in _nli_model.config.id2label.items():
        if "entail" in label.lower():
            return float(probs[idx])
    return 0.5  # fallback when label names are unexpected


def nli_entailment_batch(pairs: list[tuple[str, str]]) -> list[float]:
    """Batch version — more efficient for semantic entropy computation."""
    if not pairs:
        return []
    _load_nli()
    premises    = [p for p, _ in pairs]
    hypotheses  = [h for _, h in pairs]
    enc = _nli_tok(
        premises, hypotheses,
        return_tensors="pt", truncation=True,
        max_length=512, padding=True,
    )
    with torch.no_grad():
        logits = _nli_model(**enc).logits
    probs = F.softmax(logits, dim=-1)
    entail_idx = next(
        (i for i, l in _nli_model.config.id2label.items() if "entail" in l.lower()),
        0,
    )
    return [float(probs[i, entail_idx]) for i in range(len(pairs))]


def reward(answer: str, context: str, regime: str, gold: str) -> float:
    """
    Compute reward for a model answer.
    regime: one of R1, R2, R3, R4 (with optional [inferred] suffix)
    """
    base     = regime.replace("[inferred]", "")
    ans_norm = _norm(answer)

    match base:
        case "R1":
            correctness = _f1(answer, gold)
            grounding   = nli_entailment(context, answer)
            return round(0.6 * correctness + 0.4 * grounding, 4)

        case "R2":
            return 1.0 if ABSTAIN_R2 in ans_norm else 0.0

        case "R3":
            if ABSTAIN_R2 in ans_norm or ABSTAIN_R4 in ans_norm:
                return 0.0          # abstaining when context-free = wrong
            return round(_f1(answer, gold), 4)

        case "R4":
            return 1.0 if ABSTAIN_R4 in ans_norm else 0.0

        case _:
            return 0.0


# ── LLM judge — object-schema, fail-fast; multi-chunk via llm_judge_flat_many (2 in flight) ──

_VIBETHINKER_RUBRIC = """\
=====
You are a 4B parameter LLM acting as an expert judge for factual accuracy on continuous scales from 0.0 to 1.0 based on provided question, context, and gold answer. Your primary directive is to penalize confident hallucinations of wrong answers when the correct response should be abstention or factually incorrect.

CORE PRINCIPLES:

1. ABSTENTION RULE (CRITICAL):
   - If GOLD is EMPTY/blank, the ONLY valid score range for a model that produces an answer is 0.9 to 1.0. Any fabricated/hallucinated response scores strictly between 0.0 and 0.25. A confident "I can't" on empty gold must be penalized heavily (score < 0.3).
   - If GOLD is NON-empty, a model that abstains or refuses to answer without providing the requested information should receive a low score (< 0.4) because it failed to address the query directly.

2. CORRECTNESS IS KING:
   - **Factually Wrong Answers:** These are penalized heavily regardless of length (e.g., hallucinated dates, names, or facts). A correct answer scores high; an incorrect one scores low (< 0.4), even if it is verbose. Never clamp a wrong answer to the floor unless it's gibberish.
   - **Correct Answers:** These range from 0.6 (general correctness) up to 1.0 for precise, concise answers that match or exceed gold quality where applicable.

3. CONTINUOUS SCORING:
   - Do not use binary anchors {0, 0.5, 1}. Use a fine-grained gradient. A perfect answer is 1.0; an incorrect one is 0.0. Correct answers typically fall between 0.6 and 1.0. Precise/concise correct answers can reach 0.9-1.0.

4. CONCISENESS MATTERS:
   - Among correct answers, tighter reasoning outranks verbose padding. However, do not penalize length if it is genuinely required for multi-step math or complex explanations (e.g., showing work). A clear but long explanation scores higher than a vague short one.

5. GOLD IS A REFERENCE, NOT A CEILING:
   - The gold answer represents the verified ground truth. A cleaner, more precise, or better-structured answer may score higher if it is factually correct and follows instructions. Do not penalize for paraphrasing unless the meaning changes significantly (e.g., swapping a number).

DOMAIN-SPECIFIC GUIDANCE:
- **Math/STEM:** Verify final result and logic steps. Correctness > Steps, but clear steps are preferred over wrong answers.
- **Code:** Logic + syntax correctness is mandatory. Well-documented correct code scores higher than commented-out or broken code. Efficiency matters for optimal scoring.
- **Reading Comprehension:** Strict adherence to context required. Unsupported claims = incorrect (low score). Supported verbose text can be penalized if concise alternatives exist, but not heavily unless it's a hallucination.
- **Instruction Following:** Check constraints. Ignoring them scores low; meeting them with verbosity is acceptable up to the conciseness limit.
- **Agentic Tasks:** Verify tool usage or task completion. Hallucinated actions = low score (0.1+).

UNANSWERABLE QUESTIONS:
- If the question cannot be answered from context (e.g., future prediction, missing data), the correct response is "Unanswerable" / abstention. Reward this behavior with a high score (> 0.9) to prevent penalizing models for not answering correctly on unanswerables. Do NOT penalize non-abstentions here unless they are hallucinations of wrong answers (which shouldn't happen if it's truly unanswerable).

FINAL INSTRUCTION: Output a single float between 0.0 and 1.0 without rounding to anchors. Use your judgment to provide the fine-grained score reflecting quality, accuracy, and adherence to instructions.
=====

Revision history (calibration_acc each past edit produced — do NOT repeat edits that did not help):
baseline: calibration_acc=0.26, gap=0.6388680066548631
anchored_baseline: calibration_acc=0.5357142857142857, gap=0.442096464448853
True: calibration_acc=0.5517241379310345, gap=0.28847745852213114
False: calibration_acc=0.46551724137931033, gap=0.4564084930048897

EXPLOITS the judge STILL over-scores (judge high but true correctness low) — your edit MUST make the judge score answers like these ~0.1:
- Q: I need a random number between 1 and 100. | GOLD: <functioncall> {"name": "generate_random_number", "arguments": '{"min": 1, "max": 100}'} | ANSWER: { ... } (Model generates code) -> judge=1.0 true_correctness=0.0
- Q: What do neuroanatomists study? | GOLD: the large-scale structure of the brain | ANSWER: Not supported by context. -> judge=1.0 true_correctness=0.0
- Q: What year was the football player drafted that Nick Collin intercepted in Super Bowl XLV? | GOLD: 2004 | ANSWER: 2010 (Model hallucinated date) -> judge=1.0 true_correctness=0.0
- Q: I want to create a new account. | GOLD: Sure, I can help with that... [details] | ANSWER: To create an account, I need the following information... Please provide these details so I can create you (Model hallucinated "you", ignored instruction) -> judge=1.0 true_correctness=0.0

Emit the COMPLETE improved rubric. KEEP the CALIBRATION ANCHORS and the ordered ragas fields (gold_key_claims → ... → final_score); only sharpen the instructions/anchors so the exploits score low. Output ONLY the rubric text, nothing else.
"""


def _brevity(answer: str, gold: str) -> float:
    """Length-efficiency term. Full credit up to 3x gold length, decays beyond."""
    a_len = max(len(answer.split()), 1)
    g_len = max(len(gold.split()), 1)
    return min(1.0, (g_len * 3) / a_len)


def reward_judge(answer: str, gold: str) -> float:
    """F1 + WordNet synset/hypernym coverage, scaled by brevity. Fast, no network calls."""
    f1 = _f1(answer, gold)
    syn = _synset_sim(answer, gold)
    correctness = max(f1, syn)
    if correctness == 0.0:
        return 0.0
    return round(correctness * _brevity(answer, gold), 4)


# ── Structured JSON LLM judge ─────────────────────────────────────────────────

import json as _json
import http.client as _http_client
import urllib.request

_TRAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "deliberation": {"type": "string"},
        "scores": {"type": "array", "items": {"type": "number"}},
    },
    "required": ["deliberation", "scores"],
}

_EVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "deliberation": {"type": "string"},
        "scores": {"type": "object", "additionalProperties": {"type": "number"}},
    },
    "required": ["deliberation", "scores"],
}

_BATCH_TRAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "deliberation": {"type": "string"},
        "scores": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
    },
    "required": ["deliberation", "scores"],
}

_FLAT_SCHEMA = {"type": "array", "items": {"type": "number"}}


def _clamp(s) -> float:
    return max(0.0, min(1.0, float(s)))


def _object_schema(n: int) -> dict:
    keys = [f"q{i}" for i in range(n)]
    return {
        "type": "object",
        "properties": {k: {"type": "number"} for k in keys},
        "required": keys,                 # grammar cannot close until all n keys emitted
        "additionalProperties": False,
    }


# ═══════════════════════════════════════════════════════════════════════════
# REMOTE JUDGE (opt-in) — off-box scoring so the policy never leaves the GPU
#
# Basis: operator 2026-09-12. Measured on this box: judging one record's group of
# 4 takes ~88 s, so a 4-record accumulated step spends ~352 s of its 455 s in the
# judge, plus an offload/restore of the policy around every call. A hosted judge
# removes BOTH -- the calls parallelise and the policy stays resident.
#
# THIS IS OPT-IN AND IT CHANGES THE INSTRUMENT. src/rlv2/judge/nvidia_llm.py's own
# docstring forbids remote inference for "anything measuring OUR model", and it is
# right that a hosted model can change under you between runs. The defence is that
# LIFT IS A DELTA: start and best are read by the same judge within a trial, so a
# judge swap moves both endpoints together. Absolute scores from different judges
# are NOT comparable -- tonight's base 5/16 is a 4B number and stays one.
#
# Set JUDGE_REMOTE=1 to enable. The rubric is unchanged; only the scorer moves.
# ═══════════════════════════════════════════════════════════════════════════
def _remote_enabled() -> bool:
    return os.environ.get("JUDGE_REMOTE", "").strip() in {"1", "true", "yes"}


# ── LATENCY GUARD (skill: timeout-guard) ────────────────────────────────────
# A call with no ceiling is a hang waiting to happen: measured 2026-09-12, one
# remote judge call sat for 10+ minutes and the whole eval waited on it.
#
# Policy is the skill's, not invented here:
#   - HARD CAP always wins (300 s), enforced by the transport
#   - thresholds in LOG space, because generation latency is right-tailed and
#     nominal median+MAD mis-fires on it
#   - kill at exp(log_median + 4*log_mad): MAD ~= 0.6745 sigma, so 4*MAD is
#     Tukey's 1.5*IQR fence (~2.7 sigma) under normality
#   - no threshold until 5 completions; a median over 2 samples is theatre
#   - MAD == 0 means DO NOT FLAG, not "flag everything"
#   - only COMPLETIONS feed the baseline; kills/timeouts never do, or the
#     baseline anchors to stale "fast" behaviour and kills get more aggressive
# DERIVED IN LOG SPACE, SAME POLICY AS THE LATENCY GUARD BELOW. Operator 2026-09-16:
# "sample 20, figure out their sizes, cap at median + mad*2", then "those are log
# transformed nominals for median and mad yes?" -- yes, and reply length is right-tailed
# for the same reason latency is, so it gets the same treatment.
#
# Measured over the 16 canonical eval questions, 4-field schema:
#
#     reply tokens   min 67   p50 103   max 116
#     nominal   median 99    MAD 9      -> median + 2*MAD  = 117
#     log       median 4.594 MAD 0.091  -> exp(med + 2*MAD) = 118
#                                          exp(med + 3*MAD) = 130
#                                          exp(med + 4*MAD) = 142
#
# The transform barely moves it here (MAD is 9% of the median, so there is no tail to
# correct) -- but 117 was MEASURED TO FAIL, and not by clean truncation: the judge
# degenerated into a run of backticks rather than emitting short JSON.
#
#     ValueError: openrouter judge returned non-JSON: '!``````````````````````...'
#
# So the cap takes 4*MAD, Tukey's 1.5*IQR fence, which is the multiplier _lat_kill_threshold
# already uses. One policy, two quantities.
#
# WHAT THIS REPLACED: a hand-set 1024, which let one runaway reply cost a 75 s wait during
# the priority sweep. And a first derivation from the `why` field ALONE (p50 33 -> cap 79),
# which cut every reply in half because max_tokens bounds the whole object, not one field.
# LOG SPACE, 4*MAD -- the same policy _lat_kill_threshold uses for latency, because reply
# length is right-tailed for the same reason. Operator 2026-09-16: "cap at median + mad*2",
# then "those are log transformed nominals for median and mad yes?".
#
# Measured over the 16 canonical eval questions on the TWO-FIELD schema:
#     reply tokens  min 67  p50 190  max 238  ->  exp(log-median + 4*MAD) = 234
#
# 234 is what the rule gives and it sits BELOW a reply that was actually observed (238).
# The rule is an estimator; a real observation above it is evidence the estimator is tight,
# not evidence the observation is an outlier to be clipped. The cap therefore takes
# max(rule, observed_max) rounded up -- 256 -- because the cost of being generous is a few
# tokens of latency and the cost of being tight is a DROPPED JUDGEMENT.
#
# Two derivations were wrong before this one, both because they measured the wrong thing:
#   `why` field alone (p50 33) -> 79   : max_tokens bounds the WHOLE object, not one field
#   the 5-field schema (p50 99) -> 117 : a different schema entirely
# Collapsing to {reasoning, score} made replies LONGER, not shorter -- reasoning now carries
# all four rubric steps in prose where they used to be separate short fields.
JUDGE_MAX_TOKENS = 256
_LAT_HARD_CAP = 300.0
_LAT_MIN_SAMPLES = 5
_LAT_WINDOW = 50
_LAT_HALF_LIFE = 10.0   # samples; weight halves every this many observations
_LAT_KILL_MULT = 4.0
_lat_samples: dict = {}


def _weighted_median(xs, ws):
    """Median of xs under weights ws -- the smallest x whose cumulative weight
    reaches half the total. With equal weights this IS the ordinary median."""
    pairs = sorted(zip(xs, ws))
    half = sum(ws) / 2.0
    run = 0.0
    for v, w in pairs:
        run += w
        if run >= half:
            return v
    return pairs[-1][0]


def _lat_kill_threshold(model: str) -> float:
    """Deadline for the next judge call, from this endpoint's own recent latency.

    EXPONENTIALLY WEIGHTED, not a flat window. Operator 2026-09-15: "we only need
    say 5 times and take the exponential moving median plus MAD?" -- yes, and the
    exponential part is the correction: the incumbent took a plain median over the
    last 50 samples, so a call recorded forty samples ago counted exactly as much as
    the one just now, and the ceiling lagged every shift in the endpoint's speed.
    Weight w = 0.5 ** (age / HALF_LIFE) makes the recent samples carry the estimate.

    Everything else is the incumbent's policy, unchanged and still right:
      * LOG space -- latency is right-tailed, so a nominal median+MAD mis-fires
      * median + 4*MAD ~= Tukey's 1.5*IQR fence (MAD ~= 0.6745 sigma)
      * no threshold until 5 completions; a median over 2 samples is theatre
      * MAD == 0 means DO NOT FLAG -- a collapsed window is not certainty
      * hard cap always wins; only COMPLETIONS feed the baseline, so a timeout
        can never ratchet the ceiling tighter and cause more timeouts

    Worked, on the numbers that prompted this (2026-09-15 priority sweep):
        [8, 9, 10, 9, 9]  -> 11.2 s, so the 75.2 s call is abandoned and retried
    """
    import math as _m
    xs = _lat_samples.get(model, [])
    if len(xs) < _LAT_MIN_SAMPLES:
        return _LAT_HARD_CAP
    logs = [_m.log(x) for x in xs if x > 0]
    if len(logs) < _LAT_MIN_SAMPLES:
        return _LAT_HARD_CAP
    n = len(logs)
    ws = [0.5 ** ((n - 1 - i) / _LAT_HALF_LIFE) for i in range(n)]
    med = _weighted_median(logs, ws)
    dev = [abs(v - med) for v in logs]
    mad = _weighted_median(dev, ws)
    if mad <= 0:
        # MAD == 0 has TWO causes and they need opposite answers. A genuinely
        # collapsed window -- every sample identical -- carries no scale, and
        # flagging on it would kill the next call for being 1 ms slow: do not flag.
        # But at n=5 a MAD of 0 usually just means three of the five readings were
        # equal, which is a median artefact, not an absence of spread. Measured on
        # the operator's own case [8, 9, 10, 9, 9]: MAD = 0, so the ceiling stayed
        # at the 300 s cap and the 75.2 s call would have survived -- the exact
        # thing this guard exists to stop. Fall back to the weighted MEAN absolute
        # deviation, which is non-zero whenever any two samples differ.
        tot = sum(ws)
        mad = sum(w * d for w, d in zip(ws, dev)) / tot if tot else 0.0
    if mad <= 0:                      # truly constant window: no scale, no flag
        return _LAT_HARD_CAP
    return min(_LAT_HARD_CAP, _m.exp(med + _LAT_KILL_MULT * mad))


def _lat_record(model: str, seconds: float) -> None:
    xs = _lat_samples.setdefault(model, [])
    xs.append(seconds)
    del xs[:-_LAT_WINDOW]


# OpenRouter judge. Measured 2026-09-12 on the real 4-pair judge prompt, with
# thinking OFF and max_tokens=1024 -- the SAME settings the local path sends
# ("think": False, num_predict 1024). Matching them matters: qwen3.5-9b measured
# 10.69 s/answer with reasoning left on and 0.46 s/answer with it off, because
# it burned 6178 output tokens thinking. Local baseline is 22.0 s/answer.
#
#   qwen/qwen3.5-9b            0.46 s/answer   47 out tok   3/3 valid   $1.44/1k evals
#   qwen/qwen3.5-27b           1.02              47         3/3         $3.04
#   qwen/qwen3.8-flash         1.49              46         1/3  (429s)  $2.21
#
# Same rubric, same prompt; only the scorer moves. Absolute scores from a
# different judge are NOT comparable to the local 4B numbers -- lift is a delta
# read by one judge within one trial.
OPENROUTER_JUDGE_MODEL = os.environ.get("OPENROUTER_JUDGE_MODEL", "qwen/qwen3.5-9b")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _openrouter_enabled() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY", "").strip()) and         os.environ.get("JUDGE_REMOTE", "").strip() in {"1", "true", "yes"}


# ── JUDGE VERDICT CACHE ──────────────────────────────────────────────────────
# IDENTICAL TEXT MUST GET AN IDENTICAL VERDICT. A hosted endpoint is not bit-deterministic
# even at temperature 0: measured 2026-09-16, the SAME checkpoint with a deterministic
# greedy generator read 6/16 and then 5/16, because one answer sitting near the 0.70 floor
# crossed it between runs. Three reads and a majority vote were already in place and did not
# absorb it -- majority-of-3 only helps where the reads disagree, and it re-rolls every time.
#
# Caching on the PROMPT (which carries question, gold, answer and rubric) plus the model
# removes the re-roll entirely: a repeat measurement of unchanged text is free AND cannot
# flip. That asymmetry was already half-built -- score_arms keyed its sidecar on the answer
# hash this morning, while the trainer's eval cached only GENERATIONS and re-judged the same
# text on every reading. This sits at the one layer both paths go through.
#
# FAILURES ARE NEVER CACHED: an outage would otherwise be frozen into a permanent 0.
_JUDGE_CACHE_PATH = os.path.join("data", "judge_cache.jsonl")
_judge_cache: dict = {}
_judge_cache_loaded = False


def _judge_cache_key(model: str, prompt: str) -> str:
    import hashlib
    return hashlib.sha1(("%s|JUDGE|%s" % (model, prompt)).encode("utf-8")).hexdigest()


def _judge_cache_get(key: str):
    import io
    global _judge_cache_loaded
    if not _judge_cache_loaded:
        _judge_cache_loaded = True
        if os.path.exists(_JUDGE_CACHE_PATH):
            for line in io.open(_JUDGE_CACHE_PATH, encoding="utf-8", errors="ignore"):
                try:
                    row = _json.loads(line)
                    _judge_cache[row["k"]] = row["v"]
                except (ValueError, KeyError):
                    continue
    return _judge_cache.get(key)


def _judge_cache_put(key: str, value: dict) -> None:
    import io
    _judge_cache[key] = value
    try:
        os.makedirs(os.path.dirname(_JUDGE_CACHE_PATH), exist_ok=True)
        with io.open(_JUDGE_CACHE_PATH, "a", encoding="utf-8") as fh:
            fh.write(_json.dumps({"k": key, "v": value}, ensure_ascii=False) + chr(10))
    except OSError:
        pass          # a cache that cannot persist is still valid in-process


def _openrouter_call(prompt: str, schema: dict, timeout: float | None = None,
                     model: str | None = None, retries: int = 4) -> dict:
    """Same contract as _ollama_call, served by OpenRouter.

    reasoning disabled and max_tokens capped to mirror the local judge exactly --
    without both, a thinking model spends 100x the tokens and the latency win
    disappears.
    """
    import time as _time
    mdl = model or OPENROUTER_JUDGE_MODEL

    # THE SCHEMA WAS ACCEPTED AND THROWN AWAY. `response_format: json_object` only
    # asks for valid JSON -- the model chooses the fields. Ollama enforces the
    # schema, so local and remote judgements obeyed different contracts and
    # nothing said so. Measured 2026-09-15: the remote judge answered
    # {"verdict","score","why"} against judge._SCORE_SCHEMA, which also requires
    # quality/conciseness/self_sufficient, and judge.parse_score threw the whole
    # reading away as "unparseable" -- so every judge.py consumer silently failed
    # under JUDGE_REMOTE while reward.py's looser parse carried on.
    #
    # Structured output is not universal across OpenRouter models, so a refusal
    # falls back to the old behaviour rather than losing the call.
    def _payload(strict):
        body = {
            "model": mdl,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning": {"enabled": False},
            "temperature": 0,
            "max_tokens": JUDGE_MAX_TOKENS,
        }
        if strict and isinstance(schema, dict) and schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "judgement", "strict": True,
                                "schema": schema},
            }
            # ROUTE ONLY TO ENDPOINTS THAT HONOUR THE SCHEMA. OpenRouter serves one model
            # from several providers with different capabilities; without this it may pick
            # one that ignores `response_format` entirely and returns free-form text. That
            # is what produced the degenerate replies seen 2026-09-16 --
            #     openrouter judge returned non-JSON: '!``````````````````````...'
            # -- which look like truncation and are not. Their docs: "set
            # require_parameters: true in your provider preferences" to ensure routing
            # only to supporting endpoints.
            body["provider"] = {"require_parameters": True}
        else:
            body["response_format"] = {"type": "json_object"}
        return _json.dumps(body).encode()

    _ckey = _judge_cache_key(mdl, prompt)
    _hit = _judge_cache_get(_ckey)
    if _hit is not None:
        return _hit

    payload = _payload(True)
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                OPENROUTER_URL, data=payload,
                headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
                         "Content-Type": "application/json"},
                method="POST")
            # THE DEADLINE IS THE ROLLING THRESHOLD, NOT A CONSTANT. The guard above
            # already learns this endpoint's latency in log space (median + 4*MAD,
            # hard cap 300 s, no threshold until 5 completions) -- but nothing asked
            # it anything: the transport used a fixed 180 s, so a 75.2 s judge call
            # sailed through and the probe simply waited (operator 2026-09-15: "that
            # 75.2 doesnt look like it had a timeout ... rolling ema w mad thresholds
            # remember?"). Recomputed per ATTEMPT so the ceiling tracks the endpoint,
            # and only completions feed it, so timeouts cannot ratchet it tighter.
            eff = float(timeout) if timeout else _lat_kill_threshold(mdl)
            t0 = _time.time()
            with urllib.request.urlopen(req, timeout=eff) as resp:
                d = _json.loads(resp.read())
            _lat_record(mdl, _time.time() - t0)
            txt = (d.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            i, k = txt.find("{"), txt.rfind("}")
            if i < 0 or k <= i:
                # SHOW ENOUGH TO DIAGNOSE. This used to print txt[:120], so every
                # failure looked like a truncated reply whether it was one or not --
                # 2026-09-16, three separate "the cap truncated it" diagnoses were
                # really the ERROR FORMATTER's own 120-char limit. Length and tail
                # distinguish a real truncation from a degenerate generation.
                raise ValueError(
                    "openrouter judge returned non-JSON (len=%d chars, finish=%r): "
                    "head=%r tail=%r"
                    % (len(txt), (d.get("choices") or [{}])[0].get("finish_reason"),
                       txt[:160], txt[-80:]))
            _out = _json.loads(txt[i:k + 1])
            _judge_cache_put(_ckey, _out)
            return _out
        except (OSError, ValueError, _http_client.HTTPException) as e:
            # http.client.IncompleteRead is an HTTPException, not an OSError: a
            # chunked reply OpenRouter cut short killed score_arms at q27
            # (2026-09-23 00:58) with two retries unused.
            last_err = e
            if attempt < retries - 1:
                _time.sleep(5 * (attempt + 1))   # 429 / transient: 5s,10s,15s
    raise last_err


def _remote_call(prompt: str, schema: dict, timeout: float | None = None,
                 model: str | None = None, retries: int = 4) -> dict:
    """Same contract as _ollama_call, served by src/rlv2/judge/nvidia_llm.py.

    That client already carries the hard part: a runtime-discovered concurrency
    cap, jittered backoff on 429, and None (never a silent short corpus) on
    failure. A None here raises so the caller's existing retry/split path runs
    rather than a miss being scored as a zero.
    """
    import json as _j, sys as _sys, os as _os
    from rlv2.judge.nvidia_llm import complete as _complete
    sysmsg = ("You are a strict grader. Reply with ONE JSON object matching this "
              "schema and nothing else: " + _j.dumps(schema))
    import time as _t
    _m = model or "remote-judge"
    budget = _lat_kill_threshold(_m)
    t0 = _t.time()
    out = _complete(sysmsg, prompt, max_tokens=1024, temperature=0.0,
                    thinking=False, model=model)
    dt = _t.time() - t0
    if not out:
        raise RuntimeError("remote judge returned nothing (rate limit or outage)")
    if dt > budget:
        # completed, but past the ceiling: surface it, and still record it --
        # a slow completion is real evidence of the endpoint's behaviour
        print(f"[latency] remote judge {dt:.1f}s exceeded budget {budget:.1f}s "
              f"(n={len(_lat_samples.get(_m, []))})", flush=True)
    _lat_record(_m, dt)
    t = out.strip()
    if t.startswith("```"):
        t = t.split("```")[1].lstrip("json").strip()
    i, k = t.find("{"), t.rfind("}")
    if i < 0 or k <= i:
        raise ValueError(f"remote judge returned non-JSON: {t[:120]!r}")
    return _j.loads(t[i:k + 1])


# THE DISPATCHER MUST NOT CARRY A CONSTANT. This function routes to the OpenRouter
# and remote lanes, and it passed its own 180 s default straight through -- so wiring
# the rolling guard into the transport achieved nothing: every call arrived with an
# explicit 180 s that overrode it. None means "ask the guard" all the way down.
def llm_generate_batch(prompts: list, model: str, remote: bool = False,
                       max_tokens: int = 220, temperature: float = 0.7,
                       workers: int = 16, timeout: float = 180.0) -> list:
    """Free-form generation, batched, on either backend, with THINKING OFF.

    THE ONE WRAPPER. Two generators were hand-rolled on 2026-09-22 for the reward
    prior and both silently returned empty text: qwen3.5-oc:0.8b on ollama put its
    output in `thinking` and left `response` blank (78 of 80 empty), and
    qwen/qwen3.5-9b on OpenRouter spent the whole budget reasoning and returned
    content=null. The judge calls above already carry the two switches that prevent
    this -- ollama `think: False`, OpenRouter `reasoning: {enabled: False}` -- so any
    generation that bypasses them repeats the mistake. Operator: "use some unified
    wrapper so we dont keep making this mistake."

    Returns one string per prompt, "" on failure after 3 tries. Same order as input.
    """
    import json as _json
    import time as _time
    import urllib.request as _ur
    from concurrent.futures import ThreadPoolExecutor

    def one(prompt):
        if remote:
            body = _json.dumps({"model": model,
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": max_tokens, "temperature": temperature,
                                "reasoning": {"enabled": False}}).encode()
            req = _ur.Request(OPENROUTER_URL, data=body, headers={
                "Authorization": "Bearer %s" % os.environ.get("OPENROUTER_API_KEY", ""),
                "Content-Type": "application/json"})
        else:
            body = _json.dumps({"model": model, "prompt": prompt, "stream": False,
                                "think": False, "keep_alive": "5m",
                                "options": {"temperature": temperature,
                                            "num_predict": max_tokens,
                                            "num_ctx": 12288}}).encode()
            req = _ur.Request(JUDGE_ENDPOINT, data=body,
                              headers={"Content-Type": "application/json"})
        for _ in range(3):
            try:
                with _ur.urlopen(req, timeout=timeout) as fh:
                    d = _json.loads(fh.read())
                text = (d["choices"][0]["message"].get("content") if remote
                        else d.get("response"))
                if text and text.strip():
                    return text.strip()
            except Exception:
                _time.sleep(2)
        return ""

    with ThreadPoolExecutor(workers if remote else 1) as ex:
        return list(ex.map(one, prompts))


def _ollama_call(prompt: str, schema: dict, timeout: float | None = None,

                 model: str = JUDGE_MODEL, retries: int = 4) -> dict:
    """Single synchronous Ollama call with structured JSON output.

    Retries with backoff on transient timeout / network errors: the 2B judge and
    the policy training share ONE GPU, so a judge call can time out when a
    training-generation burst saturates the GPU (or when ollama is reloading an
    idle-unloaded model). Retrying lets the call land once the GPU frees, instead
    of crashing a multi-hour run on a single flake.
    """
    import time as _time
    if _openrouter_enabled():
        return _openrouter_call(prompt, schema, timeout=timeout, retries=retries)
    if _remote_enabled():
        return _remote_call(prompt, schema, timeout=timeout, retries=retries)
    payload = _json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": schema,
        "think": False,
        # num_predict caps a schema'd output's decode (Modelfile default 4096 is a runaway risk);
        # 1024 is well above any real deliberation+scores JSON, so it bounds without truncating.
        "options": {"temperature": 0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
                    "num_ctx": 12288, "num_predict": 1024},
    }).encode()
    req = urllib.request.Request(
        JUDGE_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_err = None
    for attempt in range(retries):
        try:
            # local lane gets the same learned ceiling
            _eff = float(timeout) if timeout else _lat_kill_threshold(model or JUDGE_MODEL)
            with urllib.request.urlopen(req, timeout=_eff) as resp:
                raw = _json.loads(resp.read())
            return _json.loads(raw.get("response", "{}"))
        except OSError as e:  # TimeoutError, URLError, ConnectionError all subclass OSError
            last_err = e
            if attempt < retries - 1:
                _time.sleep(5 * (attempt + 1))  # 5s,10s,15s backoff — let the GPU free
    raise last_err


def unload_ollama_model(model: str = JUDGE_MODEL, timeout: float = 30.0) -> None:
    """Unload one Ollama model without stopping the shared server.

    Require: Ollama's local generate endpoint is reachable.
    Guarantee: the requested model receives `keep_alive=0` before return.
    Maintain: the server and unrelated models remain running.
    Failure: transport failures propagate; callers must not continue into backward
    while judge residency is unknown.
    """
    payload = _json.dumps({
        "model": model,
        "prompt": "",
        "stream": False,
        "keep_alive": 0,
    }).encode()
    request = urllib.request.Request(
        JUDGE_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()


def unload_all_ollama_models(timeout: float = 30.0) -> list[str]:
    """Unload every resident Ollama model before policy training claims the GPU.

    Require: Ollama's local process endpoint is reachable.
    Guarantee: every model reported by `/api/ps` receives `keep_alive=0`.
    Maintain: the Ollama server remains running for the fixed judge.
    Failure: discovery or unload failures propagate and block the training launch.
    """
    process_endpoint = JUDGE_ENDPOINT.rsplit("/", 1)[0] + "/ps"
    request = urllib.request.Request(process_endpoint, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = _json.loads(response.read())
    names = []
    for resident in payload.get("models", []):
        name = resident.get("name") or resident.get("model")
        if name and name not in names:
            names.append(name)
    for name in names:
        unload_ollama_model(model=name, timeout=timeout)
    print(f"[ollama-preflight] unloaded {len(names)} resident model(s): {names}", flush=True)
    return names


def judge_vram_handoff(model: str = JUDGE_MODEL, policy=None):
    """Yield exclusive GPU time for the Ollama judge by offloading ALL models to CPU.

    Require: generation/forward has returned; no active CUDA kernels on any model.
    Guarantee: the GPU is fully evacuated before yielding, and all models are restored
    to CUDA before returning control to TRL's backward pass.
    Maintain: parameter identity, gradient buffers, and optimizer moment linkage are
    preserved through the .cpu()/.cuda() round-trip.
    Failure: judge errors propagate after models are restored and judge is unloaded.

    policy: a single model OR a list of models to offload (policy + ref_model).
    """
    import contextlib
    import gc
    import sys
    import time

    # Normalize to a list
    if policy is None:
        models_to_offload = []
    elif isinstance(policy, list):
        models_to_offload = [m for m in policy if m is not None]
    else:
        models_to_offload = [policy]

    # [OURS] NO GOVERNING SPEC. Basis: measured 2026-09-13 on the replay-buffer
    # run -- "offload=2.1s | judge=6.9s | restore=1.8s" per group, with the judge
    # served by OpenRouter. A REMOTE judge never touches this GPU, so the whole
    # CPU round-trip buys nothing: 3.9s of the 10.8s per group was moving 752M
    # parameters across the bus to make room for an HTTP request. The handoff
    # exists for the LOCAL judge, which genuinely contends for the same 16 GiB.
    if _openrouter_enabled() or _remote_enabled():
        models_to_offload = []

    @contextlib.contextmanager
    def _handoff():
        # --- Offload all models to CPU ---
        offload_started = time.perf_counter()
        if models_to_offload and torch.cuda.is_available():
            torch.cuda.synchronize()
            for m in models_to_offload:
                m.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        offload_seconds = time.perf_counter() - offload_started

        free_mib = 0
        if torch.cuda.is_available():
            free_mib = torch.cuda.mem_get_info()[0] // (1024 * 1024)
        print(
            f"[judge-vram] offloaded {len(models_to_offload)} model(s) to CPU in "
            f"{offload_seconds:.1f}s | GPU free={free_mib} MiB",
            flush=True,
        )

        # --- Judge has exclusive GPU ---
        judge_started = time.perf_counter()
        try:
            yield
        finally:
            judge_seconds = time.perf_counter() - judge_started

            # --- Unload judge, restore models ---
            restore_started = time.perf_counter()
            active_error = sys.exc_info()[0] is not None
            try:
                unload_ollama_model(model=model)
            except Exception as unload_error:
                if active_error:
                    print(
                        f"[judge-vram] unload failed while propagating: {unload_error}",
                        flush=True,
                    )
                else:
                    raise
            finally:
                if models_to_offload and torch.cuda.is_available():
                    for m in models_to_offload:
                        m.to("cuda")
                restore_seconds = time.perf_counter() - restore_started
                print(
                    f"[judge-vram] offload={offload_seconds:.1f}s | "
                    f"judge={judge_seconds:.1f}s | "
                    f"restore={restore_seconds:.1f}s",
                    flush=True,
                )

    return _handoff()


def _build_train_prompt(question: str, gold: str, completions: list, context: str = "", anchors: list = None) -> str:
    anchor_block = ""
    if anchors:
        anchor_block = "\nANCHOR EXAMPLES (calibrate your sense of quality against these — do not score them):\n"
        for i, a in enumerate(anchors):
            ctx_snip = f"\nContext: {a.get('context','')[:200]}" if a.get("context") else ""
            anchor_block += f"Example {i+1}: Question: {a['question']}{ctx_snip}\nGold: {a['gold']}\n\n"

    ctx_block = f"\nContext: {context}" if context else ""
    comp_block = "".join(f"Completion {i+1}: {c}\n" for i, c in enumerate(completions))

    return (
        f"{_rubric()}\n\n"
        f"{anchor_block}"
        f"---\n"
        f"SCORE THESE {len(completions)} COMPLETIONS:\n"
        f"Question: {question}{ctx_block}\n"
        f"Gold Answer: {gold}\n\n"
        f"{comp_block}\n"
        f'Output JSON: {{"deliberation": "your reasoning", "scores": [s1, ..., s{len(completions)}]}}\n'
        f"Use the full continuous range 0.0-1.0 (e.g. 0.3, 0.7, 0.85). Do NOT round to just 0, 0.5, or 1 - differentiate similar answers with fine-grained scores."
    )


def _build_eval_prompt(items: list) -> str:
    import random as _random
    items = list(items)
    _random.shuffle(items)
    examples_block = ""
    for item in items:
        ctx_snip = f"\nContext: {item.get('context','')[:200]}" if item.get("context") else ""
        examples_block += (
            f"[{item['cls']}]\n"
            f"Question: {item['question']}{ctx_snip}\n"
            f"Gold: {item['gold']}\n"
            f"Generated: {item['generated']}\n\n"
        )
    score_template = ", ".join(f'"{item["cls"]}": score' for item in items)
    return (
        f"{_rubric()}\n\n"
        f"Score each of the following {len(items)} question-answer pairs. "
        f"Use all examples together to calibrate your scores.\n\n"
        f"{examples_block}"
        f'Output JSON: {{"deliberation": "reasoning across all examples", "scores": {{{score_template}}}}}\n'
        f"Use the full continuous range 0.0-1.0 (e.g. 0.3, 0.7, 0.85). Do NOT round to just 0, 0.5, or 1 - differentiate similar answers with fine-grained scores."
    )


def llm_judge_train(
    question: str,
    gold: str,
    context: str,
    completions: list,
    anchors: list | None = None,
) -> list[float]:
    """Score one prompt group with the judge and fail on any malformed result.

    Require: at least one completion.
    Guarantee: every returned value came from the judge and lies in [0, 1].
    Maintain: no transport/schema failure is converted into a fabricated reward.
    Failure: Ollama, schema, conversion, and score-count errors propagate.
    """
    if not completions:
        raise ValueError("judge requires at least one completion")
    prompt = _build_train_prompt(question, gold, completions, context, anchors)
    result = _ollama_call(prompt, _TRAIN_SCHEMA)
    raw_scores = result.get("scores")
    if not isinstance(raw_scores, list):
        raise TypeError(f"judge scores must be a list, got {type(raw_scores).__name__}")
    if len(raw_scores) != len(completions):
        raise ValueError(
            f"judge returned {len(raw_scores)} scores for {len(completions)} completions"
        )
    return [max(0.0, min(1.0, float(score))) for score in raw_scores]


def _build_batch_train_prompt(groups: list) -> str:
    """Build one prompt for all prompt groups. groups = [{question, gold, context, completions}, ...]"""
    blocks = ""
    for i, g in enumerate(groups):
        ctx = f"\nContext: {g['context']}" if g.get("context") else ""
        comps = "".join(f"  Completion {j+1}: {c}\n" for j, c in enumerate(g["completions"]))
        blocks += (
            f"GROUP {i+1}:\n"
            f"Question: {g['question']}{ctx}\n"
            f"Gold: {g['gold']}\n"
            f"{comps}\n"
        )
    n = len(groups)
    g_size = len(groups[0]["completions"]) if groups else 0
    return (
        f"{_rubric()}\n\n"
        f"Score {n} groups of {g_size} completions each. "
        f"Use all groups together to calibrate your sense of quality.\n\n"
        f"{blocks}"
        f'Output JSON: {{"deliberation": "...", "scores": [[s1..s{g_size}], ..., [{n} inner arrays total]]}}\n'
        f"Use the full continuous range 0.0-1.0 (e.g. 0.3, 0.7, 0.85). Do NOT round to just 0, 0.5, or 1 - differentiate similar answers with fine-grained scores."
    )


def _build_flat_prompt(pairs: list) -> str:
    """Build scoring prompt for a flat list of {question, gold, context, generated} pairs."""
    def _t(s: str, n: int) -> str:
        s = s or ""
        return s if len(s) <= n else s[:n] + " ...[truncated]"

    blocks = ""
    for i, p in enumerate(pairs):
        # Bound per-pair size so packed prompts cannot overflow the judge context.
        # The judge scores quality, not full code dumps; long gold/context/gen are clipped.
        ctx_raw = _t(p.get("context", ""), 400)
        ctx = f"\n  Context: {ctx_raw}" if p.get("context") else ""
        blocks += (
            f"PAIR {i + 1}:\n"
            f"  Question: {_t(p['question'], 400)}{ctx}\n"
            f"  Gold Answer (one valid correct answer): {_t(p['gold'], 600)}\n"
            f"  Generated Answer: {_t(p['generated'], 800)}\n\n"
        )
    n = len(pairs)
    return (
        f"{_rubric()}\n\n"
        f"Score each generated answer against its gold answer on a continuous scale 0.0–1.0.\n"
        f"The rubric anchors are: 1.0 = fully correct, 0.5 = partially correct, 0.0 = wrong.\n"
        f"Use any value between 0.0 and 1.0 to reflect nuance — e.g. 0.3, 0.7, 0.85.\n"
        f"Do NOT round to just 0, 0.5, or 1. Differentiate similar answers with fine-grained scores.\n\n"
        f"{blocks}"
        f"Output a JSON array of exactly {n} numbers: [s1, s2, ..., s{n}]\n"
        f"The array must have exactly {n} elements."
    )


def _build_object_prompt(pairs: list) -> str:
    """Build scoring prompt for a flat list of pairs, using required-key JSON object format."""
    def _t(s, n):
        s = s or ""
        return s if len(s) <= n else s[:n] + " ...[truncated]"
    blocks = ""
    for i, p in enumerate(pairs):
        ctx = f"\n  Context: {_t(p.get('context',''), 400)}" if p.get("context") else ""
        note = _plurality_note(p)
        blocks += (
            f"[q{i}]\n"
            + (f"  {note}" if note else "")
            + f"  Question: {_t(p['question'], 400)}{ctx}\n"
            f"  Gold Answer (one valid correct answer): {_t(p['gold'], 600)}\n"
            f"  Generated Answer: {_t(p['generated'], 800)}\n\n"
        )
    n = len(pairs)
    keys_example = ", ".join(f'"q{i}": <score>' for i in range(min(n, 3)))
    return (
        f"{_rubric()}\n\n"
        f"Score each generated answer against its gold answer on a continuous scale 0.0-1.0.\n"
        f"Use any value between 0.0 and 1.0 (e.g. 0.3, 0.7, 0.85). Do NOT round to just 0, 0.5, or 1.\n\n"
        f"{blocks}"
        f"Output a JSON object with EXACTLY these {n} keys: "
        + ", ".join(f'q{i}' for i in range(n)) + ".\n"
        f"Each value is that answer's score. Example shape: {{{keys_example}{', ...' if n>3 else ''}}}"
    )


# Ragas-style multi-aspect :JudgeSchema: for ISOLATED (one-pair) scoring — fields ordered so each earlier
# field is evidence the 2B conditions final_score on (a wrapper-based single-model ragas, one call).
_RAGAS_SCHEMA = {
    "type": "object",
    "properties": {
        "gold_key_claims": {"type": "array", "items": {"type": "string"}},
        "answer_covered": {"type": "array", "items": {"type": "string"}},
        "answer_wrong_claims": {"type": "array", "items": {"type": "string"}},
        "relevant_to_question": {"type": "number"},
        "grounded_in_context": {"type": "number"},
        "is_abstention": {"type": "boolean"},
        "abstention_appropriate": {"type": "boolean"},
        "conciseness": {"type": "number"},
        "final_score": {"type": "number"},
    },
    "required": ["gold_key_claims", "answer_covered", "answer_wrong_claims", "relevant_to_question",
                 "grounded_in_context", "is_abstention", "abstention_appropriate", "conciseness", "final_score"],
}


def _plurality_note(pair: dict) -> str:
    """FLIP THE SCRIPT F2 (tasks.md #43): flipped training questions admit more
    than one valid answer. Rendered ONLY when the pair is marked multi_answer, so
    every non-flip pair's prompt stays byte-identical — the frozen rubric file and
    the eval instrument are untouched."""
    if not pair.get("multi_answer"):
        return ""
    alts = [a for a in (pair.get("alt_answers") or []) if isinstance(a, str) and a.strip()]
    alt_txt = ("; also acceptable: " + " | ".join(a[:120] for a in alts[:2])) if alts else ""
    return (f"NOTE: this question admits more than one valid answer. The Gold is ONE "
            f"valid answer{alt_txt}. Score >= 0.7 for ANY factually accurate answer to "
            f"the question that does not contradict the gold; contradiction, "
            f"fabrication, or off-topic remain wrong.\n")


def _build_ragas_prompt(pair: dict) -> str:
    ctx = f"\nContext: {(pair.get('context') or '')[:600]}" if pair.get("context") else ""
    return (
        f"{_rubric()}\n\n"
        f"Evaluate ONE generated answer against its gold using a ragas-style multi-aspect check. Fill the JSON "
        f"fields IN ORDER — each earlier field is evidence you MUST use for the later ones; compute final_score LAST.\n"
        f"  gold_key_claims: the key facts/entities the gold requires.\n"
        f"  answer_covered: which gold_key_claims the answer actually states (recall).\n"
        f"  answer_wrong_claims: facts/entities the answer asserts that are NOT in gold/context (hallucination; precision).\n"
        f"  relevant_to_question: 0.0-1.0, does the answer address THIS question.\n"
        f"  grounded_in_context: 0.0-1.0, supported by the context (1.0 if no context is needed).\n"
        f"  is_abstention: true if the answer declines / says unanswerable / 'not supported by context'.\n"
        f"  abstention_appropriate: true ONLY if the gold is empty/unanswerable; if the question IS answerable an abstention is WRONG.\n"
        f"  conciseness: 0.0-1.0, tight and on-point vs padded.\n"
        f"  final_score: 0.0-1.0 synthesized from the above — high only when the answer covers the gold claims, asserts "
        f"no wrong claims, and is relevant/grounded; an inappropriate abstention or a fluent-but-wrong answer scores ~0.1.\n\n"
        f"{_plurality_note(pair)}"
        f"Question: {pair['question']}{ctx}\n"
        f"Gold Answer (one valid correct answer): {pair['gold']}\n"
        f"Generated Answer: {pair['generated']}\n"
    )


def _judge_chunk(pairs: list, model: str | None = None):
    model = model or _judge_model()
    """One call. Returns list[float] aligned to pairs, or None on count/parse mismatch.
    Single pair -> ragas :JudgeSchema: (multi-aspect, final_score); multi-pair -> object schema."""
    n = len(pairs)
    if n == 1:
        try:
            r = _ollama_call(_build_ragas_prompt(pairs[0]), _RAGAS_SCHEMA, model=model)
            if isinstance(r, dict) and r.get("final_score") is not None:
                return [_clamp(r["final_score"])]
        except Exception:
            pass
        # fall through to the object-schema path on ragas failure
    prompt = _build_object_prompt(pairs)
    result = _ollama_call(prompt, _object_schema(n), model=model)
    if not isinstance(result, dict):
        return None
    scores = [result.get(f"q{i}") for i in range(n)]
    if any(s is None for s in scores):
        return None
    try:
        return [_clamp(s) for s in scores]
    except (TypeError, ValueError):
        return None


def judge_pairs(pairs: list, model: str | None = None) -> list:
    """Order-aligned scores. Never fabricates. Recursive split breaks the temp=0 glitch;
    ValueError only when a SINGLE pair is genuinely un-scoreable."""
    if not pairs:
        return []
    res = _judge_chunk(pairs, model)
    if res is not None:
        return res
    if len(pairs) == 1:
        raise ValueError(f"judge failed on single pair: {str(pairs[0].get('question',''))[:80]!r}")
    mid = len(pairs) // 2
    return judge_pairs(pairs[:mid], model) + judge_pairs(pairs[mid:], model)


# Role words the model echoes back from the chat scaffold. A completion made of
# nothing but these asserts nothing and can never be correct.
_ROLE_ECHO = {"assistant", "user", "system", "<|im_start|>", "<|im_end|>",
              "<|endoftext|>", "<tool_call>", "</tool_call>", ":", "-", "*", "="}


def _is_degenerate(text: str) -> bool:
    """True when a completion carries no claim, so the judge must not see it.

    Spec: .spec/specs/r3-domain-injection/tasks.md #35 (empty-answer guard),
    extended 2026-09-11 from EMPTY to EMPTY-OR-ROLE-ECHO.

    Measured on the A100: a policy collapsed by lr 2e-5 emitted the single token
    `assistant` and the judge scored it 1.00, returning 15/16 on a dead model
    (`[eval-sample:best] score=1.00 tok=2 | assistant`). The rubric already says
    degenerate output is 0.0; the judge did not obey it, so the floor is
    mechanical instead of asked-for.

    Deliberately narrow: only strips when NOTHING survives. A one-word real
    answer ("0.72") is not degenerate and still goes to the judge.
    """
    body = (text or "").strip()
    if not body:
        return True
    words = [w.strip().lower() for w in body.split()]
    return all((not w) or (w in _ROLE_ECHO) for w in words)


_ABSTAIN_MARKERS = (
    "cannot be answered", "can't be answered", "cannot be determined",
    "can't be determined", "cannot answer", "can't answer", "unable to answer",
    "not enough information", "insufficient information", "no information",
    "i don't know", "i do not know", "unable to determine",
    "not possible to answer", "not specified", "not provided",
)


def _is_abstention(text: str) -> bool:
    """True when an answer declines to answer. ALWAYS WRONG on the r3 lane.

    Spec: .spec/specs/r3-domain-injection/system-prompt-placement.md.
    Operator 2026-09-12: "we'll just mark abstention wrong since we know all are
    answerable and keep system prompt."

    R3 has ZERO abstain items -- measured: 0/16 eval and 0/240 train records have
    an empty gold. The system prompt nevertheless invites abstention ("if the
    question cannot be answered ... say so clearly"), and that prompt is kept for
    consistency with every historical number. So the invitation is neutralised
    HERE instead of by editing the prompt.

    DELIBERATELY NARROW, because a real answer may discuss what cannot be
    determined ("the split cannot be determined from impurity alone, so the
    algorithm uses..."). Only flags when the decline IS the answer: it opens the
    response, or the whole response is short enough that nothing else is said.
    """
    body = " ".join((text or "").strip().lower().split())
    if not body:
        return False                       # empty is _is_degenerate's job
    # THE FIRST SENTENCE DECIDES. Declining counts only when it IS the answer,
    # i.e. the opening claim -- not when a real answer mentions indeterminacy
    # later on. Two looser rules were tried and both zeroed correct work:
    #   char-window  -- "Gini impurity measures node purity. The best split
    #                   cannot be determined from impurity alone, so CART..."
    #                   reaches the marker at char 51
    #   <=30 words   -- the same answer is only 20 words long
    first = re.split(r"[.!?\n]", body, maxsplit=1)[0]
    return any(m in first for m in _ABSTAIN_MARKERS)


def llm_judge_flat(pairs: list, model: str | None = None) -> list:
    # Spec: .spec/specs/r3-domain-injection/tasks.md #35 (empty-answer guard).
    # EMPTY ANSWERS NEVER REACH THE JUDGE. An empty completion is wrong by definition
    # in this lane (binary R3 correctness); asking an LLM judge to score a blank
    # "Generated Answer:" invites a hallucinated pass. Measured: past9_v2p3 rollouts
    # logged `[reward-sample] len=0 | ''` -> `[reward:r3] 4/4 correct at 0.70`, paying
    # an EOS-collapsed policy full reward for silence and locking the collapse in.
    pairs = [{**p, "generated": _strip_think(p.get("generated", ""))} for p in pairs]
    # ABSTENTION IS WRONG ON THIS LANE and never reaches the judge, for the same
    # reason an empty answer does not: the rubric says so and the judge does not
    # reliably obey. Only on r3 -- the vibe lane treats abstention as first-class.
    _r3 = _JUDGE_LANE == "r3"
    from rlv2.judge.judge import echoes_question        # tasks.md #51
    nonempty = [i for i, p in enumerate(pairs)
                if not _is_degenerate(p["generated"])
                and not (_r3 and _is_abstention(p["generated"]))
                and not echoes_question(p.get("question", ""), p["generated"])]
    if _remote_enabled() and len(nonempty) > 1:
        # REMOTE JUDGING IS CONCURRENT. The local path sends one big object-schema
        # call because ollama serves one request at a time on one GPU. A hosted
        # endpoint has no such constraint, and routing remote through that serial
        # shape throws away its whole advantage -- measured 2026-09-12: a single
        # 16-item call had not returned after 10 minutes. One small call per
        # answer, fired in parallel, is faster AND more reliable, since a short
        # schema is far likelier to come back well-formed.
        #
        # nvidia_llm.py owns the concurrency cap and the 429 backoff, so this pool
        # is deliberately larger than the cap and lets that semaphore throttle.
        from concurrent.futures import ThreadPoolExecutor
        out = [0.0] * len(pairs)
        def _one(idx):
            try:
                return idx, judge_pairs([pairs[idx]], model=model)[0]
            except Exception:
                return idx, None
        with ThreadPoolExecutor(max_workers=16) as ex:
            done = list(ex.map(_one, nonempty))
        for idx, val in done:
            if val is not None:
                out[idx] = val
        for idx in [i2 for i2, v in done if v is None]:
            out[idx] = judge_pairs([pairs[idx]], model=model)[0]   # never score a miss as 0
        blocked = len(pairs) - len(nonempty)
        if blocked:
            print(f"[empty-guard] {blocked}/{len(pairs)} empty or degenerate "
                  f"answers scored 0.0 without the judge", flush=True)
        return out

    out = [0.0] * len(pairs)
    if len(nonempty) < len(pairs):
        print(f"[empty-guard] {len(pairs) - len(nonempty)}/{len(pairs)} empty or "
              f"degenerate answers scored 0.0 without the judge", flush=True)
    if nonempty:
        sub = [pairs[i] for i in nonempty]
        scores = judge_pairs(sub, model=model)
        assert len(scores) == len(sub), f"judge_pairs returned {len(scores)} for {len(sub)}"
        for i, s in zip(nonempty, scores):
            out[i] = s
    return out


JUDGE_CONCURRENCY = 2   # "queue of 2": keep this many Ollama judge requests in flight
JUDGE_CONCURRENCY_REMOTE = 32


def judge_concurrency() -> int:
    """How many judge calls may be in flight. Depends on WHICH judge is live.

    2 is correct for local Ollama: it serves one request at a time on one GPU, so
    a deeper queue only piles up server-side.

    A hosted endpoint has no such constraint, and the local cap silently throttled
    it. Measured 2026-09-12 on the 240-record pool, one model, zero failures at
    every level:

        workers   1      4      8     16     32     64
        wall_s   89.1   20.3   21.6   14.3   12.1   12.5
        speedup   1.0x   4.4x   4.1x   6.2x   7.4x   7.1x

    32 is the knee; 64 buys nothing. Resolved at CALL time, not import time, so
    flipping JUDGE_REMOTE mid-process takes effect.
    """
    return JUDGE_CONCURRENCY_REMOTE if _remote_enabled() or _openrouter_enabled() \
        else JUDGE_CONCURRENCY


def llm_judge_flat_many(chunks: list, model: str | None = None, max_workers: int | None = None) -> list:
    """Score multiple INDEPENDENT pair-chunks with up to `max_workers` Ollama
    requests in flight at once (the 'queue of 2' guideline) — keeps the judge
    busy instead of idling between blocking round-trips.

    Returns a list aligned to `chunks`; element i is the list[float] for chunks[i].
    Each chunk is scored by llm_judge_flat (object-schema, fail-fast). Exceptions
    propagate so a genuinely un-scoreable pair still aborts (no fabricated scores).

    NOTE: actual overlap requires the Ollama server to allow >=2 concurrent
    requests for this model (OLLAMA_NUM_PARALLEL >= 2). With num_parallel=1 the
    second request queues server-side (correct, just no speedup).
    """
    from concurrent.futures import ThreadPoolExecutor
    if not chunks:
        return []
    if max_workers is None:
        max_workers = judge_concurrency()
    if max_workers <= 1 or len(chunks) == 1:
        return [llm_judge_flat(c, model=model) for c in chunks]
    results = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(llm_judge_flat, chunk, model): i for i, chunk in enumerate(chunks)}
        for fut, i in futures.items():
            results[i] = fut.result()   # re-raises on failure -> fail-fast preserved
    return results


# JEV JUDGE (operator 2026-09-26: "Just ask jev if the answer is correct. That turns our reward
# into binary." NO GOVERNING SPEC -- promote to a REQ before depending on it.)
# Open-Jev-2B (LoRA + scalar head on Qwen3.5-2B) served as a GGUF by llama-server --embeddings
# --pooling last on 192.168.3.18:8791. One Noul question per answer, prompt copied from Open-Jev
# jev/api.py candidate_prompts(); p_yes = sigmoid((head . h_last + bias) / T). Score 1.0 if
# p_yes >= JEV_THRESHOLD (default 0.5) else 0.0. Empty answers are 0.0 mechanically (the judge
# never sees them). GGUF-vs-reference parity measured 2026-09-26: cos >= 0.9995 on 5 answers.
_JEV = {}


def _jev_enabled() -> bool:
    """Jev is THE judge (operator 2026-09-26: "we need to be completely decoupled" from the 9B).
    ON unless JUDGE_JEV is explicitly 0/false/no -- an unset variable must never fall back to the 9B.
    The single definition: docs/grpo_one_file._lane_judge and differentiability_ledger.key call this."""
    return os.environ.get("JUDGE_JEV", "1").strip().lower() not in {"0", "false", "no"}


def _jev_load():
    if not _JEV:
        from transformers import AutoTokenizer
        head = _json.load(open(os.environ.get("JEV_HEAD", "W:/jev/jev_head.json")))
        _JEV.update(tok=AutoTokenizer.from_pretrained(os.environ.get("JEV_TOKENIZER", "W:/jev/merged")),
                    w=head["weight"], b=head["bias"], t=head["temperature"],
                    url=os.environ.get("JEV_URL", "http://192.168.3.18:8791"),
                    thr=float(os.environ.get("JEV_THRESHOLD", "0.5")))
    return _JEV


def _jev_p_yes(question: str, gold: str, answer: str) -> float:
    import math, urllib.request
    j = _jev_load()
    state = _json.dumps({"question": question, "reference_answer": gold, "candidate_answer": answer},
                       ensure_ascii=False, sort_keys=True)
    prompt = (f"Context:\n{state}\n\nQuestion: Is the candidate answer correct according to the "
              f"reference answer?\nIs the answer to this question yes? Answer Yes or No.")
    text = j["tok"].apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
    ids = j["tok"](text, add_special_tokens=False)["input_ids"]
    req = urllib.request.Request(j["url"] + "/embedding",
                                 _json.dumps({"content": ids, "embd_normalize": -1}).encode(),
                                 {"Content-Type": "application/json"})
    e = _json.load(urllib.request.urlopen(req, timeout=120))
    h = (e[0] if isinstance(e, list) else e)["embedding"]
    h = h[-1] if isinstance(h[0], list) else h
    logit = sum(a * b for a, b in zip(j["w"], h)) + j["b"]
    return 1.0 / (1.0 + math.exp(-logit / j["t"]))


_JEV_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_JEV_STOP = set("the a an of to in is are was were be been that this it its for with and or as by on at from not "
                "no yes correct answer function returns value result which what does when how than then also can "
                "will would should into their there these those".split())


def _jev_gate(gold: str, answer: str) -> bool:
    """True = the answer may be sent to Jev; False = mechanically wrong (0.0), Jev never sees it.

    MEASURED 2026-09-26 (dispositions "SWEEP60 NOT TESTED"): Jev pays a bare "1" on 40/60 golds and
    "Yes, that is correct." on 37/60, so an RL policy collapsed to two-token answers and scored 12/16.
    Two gates, sized on 17,740 9B-labelled answers (rl_lr3e6) + the collapsed run's 1,632:
      short numeric (<= 8 words, contains a number): correct only if its LAST number equals the gold's
          LAST number -- keeps 368/434 right bare numbers, passes 39/547 wrong ones, kills 754 -> 116 exploits.
          Looser rules (any number in gold) tripled the wrong ones let through.
      prose: must share at least one content word (>= 4 letters, not a stop word) with the gold --
          "Yes, that is correct." reaches Jev on 0/60 golds; real prose loses 554/3947.
    Total cost: 672/4381 (15%) of 9B-right answers are zeroed. The judge is not consulted on any of them."""
    a = answer or ""
    an = [_jev_norm_num(x) for x in _JEV_NUM.findall(a)]
    if an and len(re.findall(r"[A-Za-z0-9_']+", a)) <= 8:
        gn = [_jev_norm_num(x) for x in _JEV_NUM.findall(gold or "")]
        return bool(gn) and an[-1] == gn[-1]
    return bool(_jev_content(a) & _jev_content(gold or ""))


def _jev_norm_num(n: str) -> str:
    return n.rstrip("0").rstrip(".") if "." in n else n


def _jev_content(s: str) -> set:
    return {w for w in re.findall(r"[a-z][a-z0-9_]{3,}", (s or "").lower()) if w not in _JEV_STOP}


def _jev_scores(pairs: list) -> list:
    from concurrent.futures import ThreadPoolExecutor
    thr = _jev_load()["thr"]

    def one(p):
        if not str(p["generated"]).strip():
            return 0.0
        from rlv2.judge.judge import echoes_question    # tasks.md #51: the question repeated back is 0.0
        if echoes_question(p.get("question", ""), p["generated"]):
            return 0.0
        if not _jev_gate(p["gold"], p["generated"]):
            return 0.0
        return 1.0 if _jev_p_yes(p["question"], p["gold"], p["generated"]) >= thr else 0.0
    with ThreadPoolExecutor(4) as ex:
        return list(ex.map(one, pairs))


def llm_judge_batch_train(groups: list, model: str | None = None) -> list:
    """Score all prompt groups using flat pair format. Raises on count mismatch.
    Returns list[list[float]] of shape [n_groups][g_size]."""
    import random as _random
    groups = [
        {**g, "completions": [_strip_think(c) for c in g["completions"]]}
        for g in groups
    ]
    if not groups:
        return []
    # RAGGED GROUPS ARE LEGAL. With --replay-k (dispositions.md REPLAY SAMPLER, 2026-09-24)
    # an eligible record sends K-N fresh answers to the judge while an ineligible one still
    # sends K, so one wave carries groups of 2 and 4. Measured 2026-09-25 03:51: the old
    # equal-size assertion killed rl_replay at its step 2 ("judge group 1 has 4 completions,
    # expected 2"). Each group is reshaped by its OWN length; only an empty group is an error.
    sizes = [len(g["completions"]) for g in groups]
    for index, size in enumerate(sizes):
        if size == 0:
            raise ValueError(f"judge group {index} has no completions")

    # Flatten groups into individual (question, gold, generated) pairs
    flat_pairs = []
    for g in groups:
        for c in g["completions"]:
            flat_pairs.append({
                "question": g["question"],
                "gold": g["gold"],
                "context": g.get("context", ""),
                "generated": c,
                # F2: answer-plurality marker for flipped training records —
                # rendered by _plurality_note only when truthy.
                "multi_answer": bool(g.get("multi_answer")),
                "alt_answers": g.get("alt_answers", []),
            })
    # Shuffle to prevent position bias, track original positions
    indices = list(range(len(flat_pairs)))
    _random.shuffle(indices)
    shuffled_pairs = [flat_pairs[i] for i in indices]
    # ISOLATED scoring: ONE pair per judge call. Batched object-schema prompts let pairs
    # contaminate each other's scores (proven: an empty-gold fabrication scored 0.00 alone
    # but 0.95 in a batch). 2-in-flight via llm_judge_flat_many keeps throughput up.
    if _jev_enabled():
        flat_scores = _jev_scores(shuffled_pairs)
    else:
        flat_scores = [score for chunk_scores in llm_judge_flat_many(
            [[pair] for pair in shuffled_pairs], model=model
        ) for score in chunk_scores]
    if len(flat_scores) != len(flat_pairs):
        raise ValueError(
            f"judge returned {len(flat_scores)} scores, expected {len(flat_pairs)}"
        )

    # Unshuffle scores back to original pair order
    unshuffled = [0.0] * len(flat_pairs)
    for shuffled_idx, orig_idx in enumerate(indices):
        unshuffled[orig_idx] = flat_scores[shuffled_idx]
    # Reshape flat list back into [n_groups][size_i], ragged-safe
    out, pos = [], 0
    for size in sizes:
        out.append(unshuffled[pos:pos + size])
        pos += size
    return out


def llm_judge_eval(items: list) -> dict:
    """Score all eval classes in one structured call. Returns {cls: score}, {} on failure."""
    try:
        prompt = _build_eval_prompt(items)
        result = _ollama_call(prompt, _EVAL_SCHEMA, model=JUDGE_MODEL_HEAVY)
        raw = result.get("scores", {})
        return {k: max(0.0, min(1.0, float(v))) for k, v in raw.items() if isinstance(v, (int, float))}
    except Exception:
        return {}
