"""src/rlv2/train/rl/grpo.py — VibeThinker's Signal phase (arXiv:2511.06221), two-armed so the mechanism is falsifiable.

Concepts: Policy Gradients / Actor-Critic (docs/INSPIRATIONS.md). The 60-step, replay-sampler and crossing runs were trained
by docs/grpo_one_file.py (docs/PROVENANCE.md, "Where the cited material lives").

THE PAPER. VibeThinker-1.5B's Spectrum-to-Signal Principle splits SFT and RL by OBJECTIVE: SFT
cultivates a diverse spectrum of solutions, then MaxEnt-Guided Policy Optimization (MGPO)
"dynamically prioritis[es] problems where the model has high uncertainty".

WHY THAT MATTERS HERE, MEASURED NOT ASSUMED. The prior GRPO run in this repo logged

    frac_reward_zero_std : 0.2 - 0.3

meaning a fifth to a third of every batch produced groups where all rollouts scored identically.
GRPO's advantage is (reward - group mean) / group std, so a zero-variance group contributes EXACTLY
ZERO gradient. That compute is wasted. It is also the likely reason T9 measured reward climbing
0.306 -> 0.737 while held-out accuracy moved +0.021 (p=0.5637): the reward went up on prompts that
could not teach anything.

THE ADAPTATION, STATED PLAINLY. MGPO scores uncertainty as the ENTROPY of a binary pass rate,
maximal at p=0.5. Only Math has a programmatic verifier here, so that definition covers 475 of 2577
prompts. The generalisation used below is WITHIN-GROUP REWARD STANDARD DEVIATION:

    binary     std = sqrt(p(1-p))   -- same argmax as entropy, maximal at p = 0.5
    continuous std is literally what GRPO's advantage divides by

So it reduces to MaxEnt on the verifiable slice and extends to the rest without a second mechanism.

TWO ARMS, BECAUSE ONE ARM CANNOT ANSWER THE QUESTION.

    uniform : prompts sampled uniformly           (reproduces the T9 baseline)
    maxent  : prompts sampled weighted by std     (MGPO)

Identical steps, identical reward, identical everything else. This repo has already published one RL
null measured without its matched control; that is not repeated. The DISCRIMINATING readout is
`frac_reward_zero_std` -- if maxent does not drive it below uniform's, the mechanism never engaged
and any accuracy difference is noise, not evidence.

SELECTION IS ONLINE, IN ROUNDS. An offline pass-rate probe over 2577 prompts costs ~50 min of pure
generation AND goes stale as the policy improves. Instead the reward function observes every rollout
for free, and prompts are re-weighted between short rounds. That is also closer to the paper's word
"dynamically".

REWARD. Lane-dependent, selected by --lane (see reward.set_judge_lane):
  r3   (default) : the 4b judge + synthetic domain rubric scores each completion in an isolated
                   call; the reward is the BINARY correctness gate at judge.CORRECT_FLOOR plus a
                   zero-sum Long2Short brevity redistribution WITHIN the correct completions of
                   each prompt group (S-9 recommendation). Closes S-12: the 2b accepted 3/12
                   fluent-wrong decoys, and RL optimises AGAINST its reward judge across
                   thousands of rollouts -- a fluency-fooled judge there teaches the policy that
                   confident-sounding wrong answers pay.
  vibe           : the legacy frozen 2B judge (`qwen3.5-oc:2b`) with the VibeThinker rubric and
                   multiplicative brevity blend, preserved for the future R1/R2/R4 direction.
Judge transport, schema, metadata, or count failures raise in both lanes; there is no token-F1,
verifier, or fabricated-zero fallback.
Zero-variance groups still contribute zero advantage because GRPO's normalization has no signal to
extract, but that optimizer behavior does not replace or alter the judge reward.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
import time
from collections import defaultdict

# MUST precede the first CUDA allocation (same defect class as cpt.py). Measured on the
# 2026-08-31 r3 smoke: after the first backward, judge_vram_handoff's offload + empty_cache
# left only 4128 MiB free (15203 before) — ~11 GB pinned by allocator fragmentation, not by
# live tensors — and the ollama 4b judge, squeezed off the card, slowed 129s -> 371s per
# 4-completion group. expandable_segments lets empty_cache actually return that memory.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, ".")
from rlv2.judge import reward as R
from rlv2.common.config import MODEL_NAME

MIX = "data/rl/rl_mix.jsonl"
# SUBPROCESS CWD IS THE REPO ROOT, NOT src/.
# Both eval subprocesses below invoke the REPO-RELATIVE path "src/rlv2/eval/rl_calibration_eval.py".
# With cwd=<repo>/src that resolves to <repo>/src/src/... and the process dies with
# "can't open file" -- which is exactly how the RL run died at round=1 step=28 on
# 2026-08-28 ("patience eval failed"), 40 minutes into training.
#
# The SAME defect was fixed in vibe_sft.py earlier the same morning. Fixing the instance
# instead of grepping for the class cost a second 40-minute run. The pattern to grep is
# `cwd=os.path.dirname(os.path.abspath(__file__))` paired with a "src/..." argv entry.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

BASE = "skills_model"
OUT_DIR = "mgpo"

# Defaults are CLI-exposed below because each value changes the resulting policy.
ROUNDS = 2
STEPS_PER_ROUND = 12
NUM_GEN = 4                         # VibeThinker budget trade: G=4 rollouts per prompt.
LR = 1e-6                           # Measured stable floor; experiment overrides explicitly.
MAX_PROMPT = 512                    # Hardware/token-fit ceiling; prompts are selected, never clipped.
MAX_COMPLETION = 512                # 16 GB LoRA ceiling measured for this policy/judge pair.
LORA_R = 16
LORA_ALPHA = 64                     # Historical default; exact-support control passes alpha=32.
TARGET_POOL = STEPS_PER_ROUND
TOP_P = 0.95                        # VibeThinker §3.1.
TOP_K = -1                          # VibeThinker §3.1 disables top-k filtering.
BOXCOX_LAMBDA = 0.5                 # Current stratified allocation contract.
ZERO_STD_PATIENCE = 5               # More than five dead steps invalidates the run.
CLASS_COLLAPSE_PATIENCE = 10        # More than ten one-class steps invalidates the run.
OBSERVE_EVERY = 10                  # Operator must see cross-regime generations every ten steps.
DEFAULT_EVAL_FILE = "data/rl/rl_eval_v2.jsonl"
DEFAULT_MINI_EVAL_FILE = "data/rl/rl_eval_mini.jsonl"
DEFAULT_PARTITION_MANIFEST = "data/manifest.json"
DEFAULT_MIN_ADAPTER_B = 1e-4        # Production movement gate; smoke may pass 5e-5 explicitly.


def _banned(tok):
    """Token ids no phase may generate: reasoning blocks AND tool calls.

    Operator 2026-09-15: "Think and tool calling needs to be suppressed all phases,
    we've visited this b4". Both call sites here carried a literal [248068, 248069] --
    think-only, and wrong against any tokenizer but this one. Resolved from the live
    tokenizer through the single shared list in half_epoch_eval.
    """
    try:
        from rlv2.eval import half_epoch_eval as _he
        return _he.banned_ids(tok)
    except Exception:                       # never let a scorer import break training
        ids = [tok.convert_tokens_to_ids(t) for t in
               ("<think>", "</think>", "<tool_call>", "</tool_call>",
                "<tool_response>", "</tool_response>")]
        return sorted({i for i in ids if isinstance(i, int) and i >= 0})


def _load(p):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def _is_degenerate(text: str) -> bool:
    """Pre-judge degeneration gate (Req 38). Returns True if completion is repetitive garbage.

    Checks:
    1. Repetition ratio: most common 3-gram > 50% of all 3-grams
    2. Character entropy < 2.0 bits (near-uniform spam)
    3. Long (>200 chars) with low unique-token ratio (<0.3)

    Short valid abstentions ('unanswerable', 'I cannot answer') pass all checks.
    """
    if len(text) < 10:
        return False  # too short to be degenerate — could be a valid terse answer

    # Check 1: repetition ratio on 3-grams (word-level, punctuation-stripped)
    import re as _re
    cleaned = _re.sub(r'[^\w\s]', '', text.lower())
    words = cleaned.split()
    if len(words) >= 6:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        if trigrams:
            from collections import Counter
            counts = Counter(trigrams)
            most_common_count = counts.most_common(1)[0][1]
            if most_common_count / len(trigrams) > 0.3:
                return True

    # Check 1b: substring repetition — if any 8+ char substring repeats >15% of windows
    if len(text) >= 50:
        windows = [text[i:i+8] for i in range(len(text) - 7)]
        if windows:
            from collections import Counter
            win_counts = Counter(windows)
            top_count = win_counts.most_common(1)[0][1]
            if top_count > len(windows) * 0.15:
                return True

    # Check 2: character entropy
    if len(text) >= 20:
        from collections import Counter
        char_counts = Counter(text)
        total = len(text)
        entropy = -sum((c / total) * math.log2(c / total) for c in char_counts.values() if c > 0)
        if entropy < 2.5:
            return True

    # Check 3: long with low unique-token ratio
    if len(text) > 200:
        tokens = text.split()
        if tokens and len(set(tokens)) / len(tokens) < 0.3:
            return True

    return False


_ABSTENTION_RE = re.compile(
    r"unanswerable|cannot be answered|not answerable|no answer|cannot determine|"
    r"not supported|insufficient information|not possible to answer|unable to answer",
    re.IGNORECASE,
)


def _completion_class(text: str) -> str:
    """Classify a completion for collapse monitoring; never use this as reward."""
    return "abstain" if _ABSTENTION_RE.search(text) else "answer"


class TrainingHealthMonitor:
    """Enforce the observable-MGPO runtime contract.

    Require: one call per on-policy generation step with aligned records, completions, and scores.
    Guarantee: cross-regime tuples print every observation interval.
    Maintain: the frozen judge remains the sole reward authority.
    Failure: prolonged all-zero group variance, one-class collapse, or missing regime visibility raises.
    """

    def __init__(
        self,
        observe_every: int = OBSERVE_EVERY,
        zero_std_patience: int = ZERO_STD_PATIENCE,
        class_collapse_patience: int = CLASS_COLLAPSE_PATIENCE,
        watch_class_collapse: bool = True,
        min_regimes: int = 3,
    ) -> None:
        # watch_class_collapse=False is for the R3 lane, where ONE class is the SPEC, not a
        # collapse: closed-book, no abstention, every completion is class=answer by design.
        # The tripwire would kill an R3 run after class_collapse_patience steps of correct
        # behaviour. The zero-variance gate stays armed on every lane -- it watches reward
        # spread, which the binary gate + Long2Short still produce when learning happens.
        if min(observe_every, zero_std_patience, class_collapse_patience) <= 0:
            raise ValueError("health-monitor intervals must be positive")
        # min_regimes=1 is likewise the R3 lane: the mix is 100% R3 by construction, so
        # demanding three regimes at every observation step kills a correct run (measured:
        # 2026-08-31 smoke, "step 2 has only ['R3']"). Observability itself is kept — the
        # cross-regime print still fires with whatever regimes exist.
        if min_regimes < 1:
            raise ValueError("min_regimes must be >= 1")
        self.min_regimes = min_regimes
        self.watch_class_collapse = watch_class_collapse
        self.observe_every = observe_every
        self.zero_std_patience = zero_std_patience
        self.class_collapse_patience = class_collapse_patience
        self.step = 0
        self.zero_std_streak = 0
        self.class_streak = 0
        self.collapsed_class: str | None = None
        self.latest_by_regime: dict[str, tuple[dict, str, float]] = {}

    def record(self, records: list[dict], completions: list[str], scores: list[float]) -> None:
        """Record one generation step and raise immediately when a kill gate opens."""
        if not (len(records) == len(completions) == len(scores)) or not records:
            raise ValueError(
                "health monitor requires non-empty aligned records/completions/scores: "
                f"{len(records)}/{len(completions)}/{len(scores)}"
            )
        from rlv2.common.stratify import regime_of

        self.step += 1
        rewards_by_prompt: dict[str, list[float]] = defaultdict(list)
        classes: set[str] = set()
        for record, completion, score in zip(records, completions, scores):
            prompt_id = str(record.get("ptext") or record.get("prompt") or "")
            if not prompt_id:
                raise ValueError("health monitor record has no prompt identifier")
            rewards_by_prompt[prompt_id].append(float(score))
            classes.add(_completion_class(completion))
            self.latest_by_regime[regime_of(record)] = (record, completion, float(score))

        zero_groups = sum(max(values) == min(values) for values in rewards_by_prompt.values())
        frac_zero_std = zero_groups / len(rewards_by_prompt)
        self.zero_std_streak = self.zero_std_streak + 1 if frac_zero_std == 1.0 else 0
        if self.zero_std_streak > self.zero_std_patience:
            raise RuntimeError(
                "MGPO kill gate: frac_reward_zero_std=1.0 persisted for "
                f"{self.zero_std_streak} consecutive steps"
            )

        if self.watch_class_collapse and len(classes) == 1:
            current_class = next(iter(classes))
            if current_class == self.collapsed_class:
                self.class_streak += 1
            else:
                self.collapsed_class = current_class
                self.class_streak = 1
        else:
            self.collapsed_class = None
            self.class_streak = 0
        if self.class_streak > self.class_collapse_patience:
            raise RuntimeError(
                f"MGPO kill gate: all completions were class={self.collapsed_class!r} for "
                f"{self.class_streak} consecutive steps"
            )

        if self.step % self.observe_every == 0:
            regimes = sorted(self.latest_by_regime)
            if len(regimes) < self.min_regimes:
                raise RuntimeError(
                    f"MGPO observability gate: step {self.step} has only {regimes}; "
                    f"{self.min_regimes} regime(s) are required"
                )
            for regime in regimes[:3]:
                record, completion, score = self.latest_by_regime[regime]
                prompt = str(record.get("prompt", "")).replace("\n", " ")[:160]
                gold = str(record.get("gold", "")).replace("\n", " ")[:120]
                generated = completion.replace("\n", " ")[:200]
                print(
                    f"[observe] step={self.step} | {regime} | prompt={prompt!r} | "
                    f"gold={gold!r} | gen={generated!r} | reward={score:.4f}",
                    flush=True,
                )


def _normalized_question(record_or_text: dict | str) -> str:
    """Return the shared exact-match key used by runtime partition assertions."""
    if isinstance(record_or_text, dict):
        value = record_or_text.get("prompt", record_or_text.get("question", ""))
    else:
        value = record_or_text
    return " ".join(str(value).split()).casefold()


def assert_partition_disjoint(
    train_rows: list[dict],
    eval_files: list[str],
    manifest_path: str,
) -> None:
    """Reject train overlap with every eval and rubric partition before model loading.

    Require: eval JSONL files and the versioned partition manifest exist.
    Guarantee: no normalized training question occurs in eval or rubric partitions.
    Failure: missing partition artifacts, empty keys, or any overlap raises with examples.
    """
    if not train_rows:
        raise ValueError("MGPO train partition is empty")
    train_questions = {_normalized_question(row) for row in train_rows}
    if "" in train_questions:
        raise ValueError("MGPO train partition contains an empty question")

    for eval_file in eval_files:
        if not os.path.isfile(eval_file):
            raise FileNotFoundError(f"MGPO eval partition not found: {eval_file}")
        eval_questions = {_normalized_question(row) for row in _load(eval_file)}
        overlap = train_questions & eval_questions
        if overlap:
            raise AssertionError(
                f"MGPO train/eval leakage against {eval_file}: {len(overlap)} questions; "
                f"examples={sorted(overlap)[:3]}"
            )

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"MGPO partition manifest not found: {manifest_path}")
    manifest = json.load(io.open(manifest_path, encoding="utf-8"))
    try:
        rubric_questions = {
            _normalized_question(question)
            for question in manifest["splits"]["rubric"]["questions"]
        }
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid rubric partition in {manifest_path}") from exc
    overlap = train_questions & rubric_questions
    if overlap:
        raise AssertionError(
            f"MGPO train/rubric leakage against {manifest_path}: {len(overlap)} questions; "
            f"examples={sorted(overlap)[:3]}"
        )
    print(
        f"[disjoint] train={len(train_questions)} | eval_files={eval_files} | "
        f"rubric={len(rubric_questions)} | overlaps=0",
        flush=True,
    )


def _jsonl_count(path: str) -> int:
    """Count non-empty JSONL records for metric-tag provenance."""
    return len(_load(path))


def make_reward(meta, stats, offload_models=None, health_monitor=None,
                brevity_weight: float = 0.0, lane: str = "vibe",
                long2short_lambda: float = 0.2, rollout_ledger=None):
    """Return the sole MGPO reward path: isolated LLM-judge scores, lane-shaped.

    Require: each prompt resolves to question/gold/context metadata.
    Guarantee: each completion receives exactly one direct judge score.
    Maintain: no domain, regime, or failure path selects a proxy reward.
    Failure: metadata, judge transport, schema, or score-count errors propagate.

    offload_models: mutable list of models to move to CPU during judging.
    Populated after trainer init to include both policy and TRL's ref_model.
    """
    from rlv2.judge.reward import judge_vram_handoff, llm_judge_batch_train

    monitor = health_monitor or TrainingHealthMonitor()

    def fn(prompts, completions, **kw):
        if len(prompts) != len(completions):
            raise ValueError(
                f"prompt count {len(prompts)} does not match completion count {len(completions)}"
            )

        groups = []
        resolved_prompts = []
        resolved_records = []
        completion_texts = []
        for prompt, completion in zip(prompts, completions):
            text = completion if isinstance(completion, str) else completion[0].get("content", "")
            text = R._strip_think(text)
            if prompt not in meta:
                raise KeyError(
                    f"prompt not in meta ({len(meta)} keys) -- train/eval keying diverged; "
                    f"first 120 chars: {prompt[:120]!r}"
                )
            record = meta[prompt]
            groups.append({
                "question": record.get("prompt", record.get("question", "")),
                "gold": record.get("gold", ""),
                "context": record.get("context", ""),
                "completions": [text],
                # F2 (tasks.md #43): flipped records admit multiple valid answers;
                # the judge sees a per-pair NOTE, non-flip pairs stay byte-identical.
                "multi_answer": record.get("qa_kind") == "flip",
                "alt_answers": record.get("alt_answers", []),
            })
            resolved_prompts.append(prompt)
            resolved_records.append(record)
            completion_texts.append(text)

        # Observable outputs: print first completion sample every call (Smoke-Before-Commit rule)
        if groups:
            sample_text = groups[0]["completions"][0][:200]
            sample_len = len(groups[0]["completions"][0])
            print(f"[reward-sample] len={sample_len} | {sample_text!r}".encode("utf-8", errors="replace").decode("utf-8"), flush=True)

        # Repetition is a health signal, never a substitute reward. The fixed 2B judge
        # remains the sole authority for every completion, including degenerate text.
        n_degenerate = sum(_is_degenerate(group["completions"][0]) for group in groups)
        if n_degenerate:
            print(
                f"[degen-alarm] {n_degenerate}/{len(groups)} repetitive completions sent to judge",
                flush=True,
            )

        # NAME THE LANE'S JUDGE. judge_vram_handoff defaults to
        # reward.JUDGE_MODEL = qwen3.5-oc:2b (the VibeThinker lane). On the r3
        # lane that is the WRONG model to unload: it only succeeds because that
        # 2b happens to be installed locally. On a clean machine the unload 404s
        # and takes the run with it (measured on a Colab VM, 2026-09-11).
        # reward._judge_model() returns whichever judge set_judge_lane selected.
        with judge_vram_handoff(model=R._judge_model(), policy=offload_models):
            # THE JUDGE MUST BE NAMED (2026-09-12). Without model=, this uses
            # reward.JUDGE_MODEL = 'qwen3.5-oc:2b', the VibeThinker lane's judge,
            # while judge_vram_handoff loads/unloads the r3 4B around it. Every
            # number this trainer produced was scored by the 2B.
            batch_scores = llm_judge_batch_train(groups, model=R._judge_model())
        if len(batch_scores) != len(groups):
            raise RuntimeError(
                f"judge returned {len(batch_scores)} groups, expected {len(groups)}"
            )

        scores = []
        for index, row in enumerate(batch_scores):
            if len(row) != 1:
                raise RuntimeError(
                    f"judge group {index} returned {len(row)} scores for one completion"
                )
            scores.append(float(row[0]))

        # BREVITY: SHORTER CORRECT ANSWERS OUTRANK LONGER CORRECT ANSWERS.
        # The judge rubric already SAYS conciseness matters (reward.py _VIBETHINKER_RUBRIC
        # section 4, "tighter reasoning outranks verbose padding"), but an instruction in a
        # prompt is a request the judge may or may not honour, and the RL reward was the
        # raw judge score with no length term at all. This makes the preference mechanical.
        #
        # It MULTIPLIES rather than subtracts, which is the shape reward.py already uses in
        # `reward_judge` (correctness x brevity). Multiplication cannot promote a wrong
        # answer: a judge score near 0 stays near 0 however terse it is. Only the ordering
        # WITHIN correct answers moves, which is the intent.
        #
        # `_brevity` is the incumbent in reward.py -- full credit up to 3x the gold's
        # length, decaying beyond. Not reimplemented here.
        raw_scores = list(scores)   # judge output, pre-shaping; stats keeps THESE so the
                                    # threshold-spanning replay filter sees the judge, not
                                    # the shaped reward (which is 0/1-ish by design).

        # Task #37: SEEING IS BELIEVING (operator, 2026-09-04). Surface the question,
        # every completion, and its raw judge score per step, so "N/4 correct" is
        # auditable live from the log. Logging only — no behavior change.
        _seen_q = None
        for _g, _rs in zip(groups, raw_scores):
            _q = (_g["question"] or "")[:100]
            if _q != _seen_q:
                _seen_q = _q
                print(f"[judge] Q: {_q!r}".encode("utf-8", errors="replace").decode("utf-8"),
                      flush=True)
                print(f"[judge] gold: {(_g.get('gold') or '')[:100]!r}".encode(
                    "utf-8", errors="replace").decode("utf-8"), flush=True)
            print(f"[judge]   {_rs:.2f} | {_g['completions'][0][:100]!r}".encode(
                "utf-8", errors="replace").decode("utf-8"), flush=True)

        if lane == "r3":
            # CANON (canon.md §8): correctness and quality are NOT blended. The graded
            # verifier score decides ONE thing -- which side of the floor the answer falls
            # on. Quality then reorders only answers that already passed, zero-sum, so the
            # group mean is untouched. There is no critic in GRPO: that mean IS the baseline
            # the advantage subtracts, so a blended reward would change what "better than
            # average" means for every answer, not just which one wins. VibeThinker-3B gives
            # the same reason for the same design (Long2Short Math RL), and lambda=0.2 is
            # the paper's value. NOT TESTED here: the paper runs accuracy-only FIRST and
            # adds Long2Short as a SECOND stage; every run in this repo had lambda live from
            # step 1.
            # THE LADDER-ALIGNED REWARD (S-9 recommendation, recorded in playbook.md).
            # Binary gate at the ONE floor: correct -> 1.0, wrong -> 0.0. A raw-score
            # reward paid the model for confident wrongness -- measured: base raw-mean
            # 0.5608 vs SFT-ep3 0.5578, i.e. raw mean cannot even see the SFT lift the
            # binary ladder proved. Then Long2Short: zero-sum brevity redistribution
            # WITHIN the correct completions of the same prompt (s_i = 1/L_i, centred,
            # scaled to +/-long2short_lambda). Zero-sum is the point: preference among
            # correct answers is reshaped WITHOUT moving the group baseline GRPO's
            # advantage divides by. A wrong answer stays exactly 0.0 -- no brevity can
            # promote it.
            floor = R.CORRECT_FLOOR
            from collections import defaultdict as _dd
            _by_prompt = _dd(list)
            for _i, _pr in enumerate(resolved_prompts):
                _by_prompt[_pr].append(_i)
            shaped = [0.0] * len(scores)
            n_correct = 0
            for _idxs in _by_prompt.values():
                _correct = [_i for _i in _idxs if scores[_i] >= floor]
                n_correct += len(_correct)
                for _i in _correct:
                    shaped[_i] = 1.0
                if len(_correct) >= 2:
                    _inv = [1.0 / max(1, len(completion_texts[_i])) for _i in _correct]
                    _mu = sum(_inv) / len(_inv)
                    _dev = [_x - _mu for _x in _inv]
                    _md = max(abs(_d) for _d in _dev)
                    if _md > 0:
                        for _i, _d in zip(_correct, _dev):
                            shaped[_i] = 1.0 + long2short_lambda * _d / _md
            # sum(shaped) == n_correct is the zero-sum invariant; print the drift so a
            # regression is visible in the log, not just in a test.
            print("[reward:r3] %d/%d correct at %.2f | zero-sum drift %.1e"
                  % (n_correct, len(scores), floor, sum(shaped) - n_correct), flush=True)
            scores = shaped
        elif brevity_weight > 0.0:
            adjusted = []
            for group, raw in zip(groups, scores):
                b = R._brevity(group["completions"][0], group.get("gold", ""))
                # Blend toward the factor rather than applying it at full force, so one
                # knob spans "judge only" (0.0) and "full reward_judge shape" (1.0).
                adjusted.append(raw * ((1.0 - brevity_weight) + brevity_weight * b))
            n_short = sum(1 for a_, r_ in zip(adjusted, scores) if a_ < r_ - 1e-9)
            if n_short:
                print("[brevity] %d/%d penalised for length (w=%.2f, mean %.3f -> %.3f)"
                      % (n_short, len(scores), brevity_weight,
                         sum(scores) / max(1, len(scores)),
                         sum(adjusted) / max(1, len(adjusted))), flush=True)
            scores = adjusted

        for prompt, raw_value in zip(resolved_prompts, raw_scores):
            stats[prompt].append(raw_value)
        # Task #44 (Dyna-MGPO): judged rollouts stop being discarded. Every
        # (ptext, completion, raw judge score) triple lands in the round's replay
        # ledger; the round boundary consumes them as IPO preference pairs — the
        # Dyna move: act, store real judged experience, update offline from it.
        if rollout_ledger is not None:
            for prompt, text, raw_value in zip(resolved_prompts, completion_texts,
                                               raw_scores):
                rollout_ledger.append((prompt, text, raw_value))
        monitor.record(resolved_records, completion_texts, scores)
        return scores

    return fn


def bootstrap_cohort(
    pool: list[dict],
    n: int,
    seed: int,
    boxcox_lambda: float = BOXCOX_LAMBDA,
) -> list[dict]:
    """Hierarchical cross-hatch sampling with Box-Cox at each level.

    Level 1 (X): answer type (regime R1-R4) — Box-Cox over regime counts
    Level 2 (Y): domain (6 categories) within each regime — Box-Cox over domain counts
    Level 3 (Z): source dataset within each (regime, domain) — Box-Cox over source counts

    Each level applies its own Box-Cox transform on its own count vector to determine
    allocation. This guarantees representation at every level of the hierarchy without
    any single regime, domain, or source dominating.

    Require: n > 0, n <= distinct prompts in pool.
    Guarantee: returns n distinct records with hierarchical coverage.
    """
    from collections import defaultdict
    from rlv2.common.stratify import regime_of
    from rlv2.data.pull_datasets import calculate_allocations

    if n <= 0:
        raise ValueError(f"cohort size must be positive, got {n}")

    # Deduplicate by ptext
    unique: dict[str, dict] = {}
    for record in pool:
        key = record.get("ptext", "")
        if not key:
            raise ValueError("every MGPO record must have a non-empty ptext")
        unique.setdefault(key, record)
    if n > len(unique):
        # Ptext collisions reduced the effective pool. Use what's available rather than
        # crashing — the class balance is maintained at (n-collisions)/n accuracy.
        # The collision is logged for traceability.
        print(f"[cross-hatch] WARN: target_pool {n} reduced to {len(unique)} distinct ptext "
              f"({n - len(unique)} collision(s) — pre-filtering should have caught these)",
              flush=True)
        n = len(unique)

    candidates = list(unique.values())

    # Build the 3-level tree: regime → domain → source → [records]
    tree: dict[str, dict[str, dict[str, list[dict]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for r in candidates:
        regime = regime_of(r)
        # category IS domain under another name (R3 builder); same alias as manifold_sampler.
        domain = r.get("domain") or r.get("category", "unknown")
        source = r.get("source", "unknown")
        tree[regime][domain][source].append(r)

    # Level 1: Box-Cox over regime counts → how many from each regime
    regimes = sorted(tree.keys())
    regime_sizes = [sum(len(recs) for dom in tree[reg].values() for recs in dom.values())
                    for reg in regimes]
    regime_allocs = calculate_allocations(regime_sizes, n, lam=boxcox_lambda)

    # Level 2: within each regime, Box-Cox over domain counts
    domain_allocs: dict[str, dict[str, int]] = {}
    for reg, reg_n in zip(regimes, regime_allocs):
        if reg_n <= 0:
            domain_allocs[reg] = {}
            continue
        domains = sorted(tree[reg].keys())
        dom_sizes = [sum(len(recs) for recs in tree[reg][dom].values()) for dom in domains]
        dom_n_allocs = calculate_allocations(dom_sizes, reg_n, lam=boxcox_lambda)
        domain_allocs[reg] = {dom: alloc for dom, alloc in zip(domains, dom_n_allocs)}

    # Level 3: within each (regime, domain), Box-Cox over source counts
    source_allocs: dict[tuple[str, str], dict[str, int]] = {}
    for reg in regimes:
        for dom, dom_n in domain_allocs.get(reg, {}).items():
            if dom_n <= 0:
                continue
            sources = sorted(tree[reg][dom].keys())
            src_sizes = [len(tree[reg][dom][src]) for src in sources]
            if len(sources) == 1:
                source_allocs[(reg, dom)] = {sources[0]: dom_n}
            else:
                src_n_allocs = calculate_allocations(src_sizes, dom_n, lam=boxcox_lambda)
                source_allocs[(reg, dom)] = {src: alloc for src, alloc in zip(sources, src_n_allocs)}

    # Draw records according to the hierarchical allocation
    selected: list[dict] = []
    selected_ptexts: set = set()
    rng = random.Random(seed)

    for reg in regimes:
        for dom, dom_n in domain_allocs.get(reg, {}).items():
            for src, src_n in source_allocs.get((reg, dom), {}).items():
                pool_for_cell = [r for r in tree[reg][dom][src]
                                 if r["ptext"] not in selected_ptexts]
                draw_n = min(src_n, len(pool_for_cell))
                drawn = rng.sample(pool_for_cell, draw_n)
                selected.extend(drawn)
                selected_ptexts.update(r["ptext"] for r in drawn)

    # If rounding left us short, fill from the most underrepresented cells
    if len(selected) < n:
        remaining_pool = [r for r in candidates if r["ptext"] not in selected_ptexts]
        rng.shuffle(remaining_pool)
        selected.extend(remaining_pool[:n - len(selected)])
        selected_ptexts.update(r["ptext"] for r in selected[-(n - len(selected)):])

    # If rounding gave us too many, trim from the most overrepresented cells
    if len(selected) > n:
        selected = selected[:n]

    # Log for observability
    regime_counts: dict[str, int] = defaultdict(int)
    domain_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)
    for r in selected:
        regime_counts[regime_of(r)] += 1
        domain_counts[r.get("domain") or r.get("category", "unknown")] += 1
        source_counts[r.get("source", "unknown")] += 1
    print(f"[cross-hatch] n={len(selected)} | "
          f"regimes={dict(sorted(regime_counts.items()))} | "
          f"domains={dict(sorted(domain_counts.items()))} | "
          f"sources={len(source_counts)} distinct", flush=True)

    rng.shuffle(selected)
    return selected


def continuation_probabilities(
    cohort: list[dict],
    stats: dict[str, list[float]],
    mode: str,
    std_floor: float = 0.01,
    boxcox_lambda: float = BOXCOX_LAMBDA,
) -> dict[str, float]:
    """Return explicit round-two draw probabilities for the matched control or MGPO arm.

    Require: maxent has at least two judge rewards for every cohort prompt.
    Guarantee: probabilities are positive and sum to one.
    Maintain: uniform ignores reward history; maxent uses within-group reward standard deviation.
    """
    if mode not in {"uniform", "maxent"}:
        raise ValueError(f"unknown selection mode {mode!r}; expected 'uniform' or 'maxent'")
    if not cohort:
        raise ValueError("continuation cohort is empty")
    if std_floor <= 0:
        raise ValueError(f"std_floor must be positive, got {std_floor}")

    prompt_ids = [record.get("ptext", "") for record in cohort]
    if len(set(prompt_ids)) != len(prompt_ids) or any(not prompt_id for prompt_id in prompt_ids):
        raise ValueError("continuation cohort must contain unique non-empty ptext values")

    if mode == "uniform":
        probability = 1.0 / len(cohort)
        return {prompt_id: probability for prompt_id in prompt_ids}

    missing = [prompt_id for prompt_id in prompt_ids if len(stats.get(prompt_id, [])) < 2]
    if missing:
        raise ValueError(
            f"maxent requires reward history for every cohort prompt; missing {len(missing)}: "
            f"{missing[:3]}"
        )
    # Box-Cox diversity weighting: bucket rewards at 0.05 precision, count bucket frequencies,
    # then weight by the DIVERSITY of buckets (more spread across buckets = more uncertainty).
    # This ensures prompts with diverse reward distributions get prioritized over prompts that
    # always score the same (one bucket dominates). Box-Cox over bucket counts boosts rare
    # score patterns — a prompt that sometimes scores 0.1 and sometimes 0.8 (high diversity)
    # gets higher weight than one that always scores 0.5 (low diversity).
    from collections import Counter
    from rlv2.data.pull_datasets import calculate_allocations

    # RETENTION IS THRESHOLD-SPANNING, NOT BUCKET-DIVERSITY (playbook-replaybuffer-scores.md
    # sec 7). Bucket-diversity retained [0.65, 0.68] -- two buckets, both WRONG, a
    # zero-variance group under the binary gate and pure judge-call waste -- and near-dropped
    # [0.68, 0.72], the exact boundary prompt curriculum selection exists to find. A prompt
    # stays only if its judge history has outcomes on BOTH sides of the floor; Box-Cox
    # diversity then reweights WITHIN the retained set (sec 9: uncertainty as priority).
    from rlv2.judge.reward import CORRECT_FLOOR as _floor
    n_no_span = 0
    diversity_weights = []
    for prompt_id in prompt_ids:
        rewards = stats[prompt_id]
        if not (any(r >= _floor for r in rewards) and any(r < _floor for r in rewards)):
            n_no_span += 1
            diversity_weights.append(0.0)
            continue
        # Bucket at 0.05 precision
        buckets = [round(r / 0.05) * 0.05 for r in rewards]
        bucket_counts = Counter(buckets)
        n_distinct_buckets = len(bucket_counts)
        # Diversity = number of distinct 0.05 buckets occupied (more = more uncertain)
        # Box-Cox boost: if we have counts per bucket, use their distribution entropy
        if n_distinct_buckets <= 1:
            # All rewards in one bucket — zero diversity = zero learning signal.
            # DROP this prompt from round 2 entirely (weight=0, not floor).
            # ZeroVarMask: groups with identical scores produce advantage=0, wasted step.
            diversity_weights.append(0.0)
        else:
            # Weight by count spread across buckets: more even spread = higher weight
            counts = list(bucket_counts.values())
            # Use Box-Cox allocation as diversity measure: allocate 100 "points" across
            # the buckets. The more uniform the allocation, the more diverse the prompt.
            allocs = calculate_allocations(counts, 100, lam=boxcox_lambda)
            # Preserve Box-Cox occupancy balance while retaining the magnitude of score spread.
            # Occupancy alone makes [0,0,1,1] tie [0.4,0.5,0.4,0.5], contradicting the
            # documented within-group-STD MaxEnt generalization.
            max_share = max(allocs) / 100.0
            bucket_spread = statistics.pstdev(buckets)
            diversity_score = bucket_spread * (1.0 - max_share + std_floor)
            diversity_weights.append(diversity_score)

    total = sum(diversity_weights)
    if total == 0:
        raise RuntimeError(
            "MGPO MaxEnt gate failed: every bootstrap prompt has zero score diversity; "
            "round 2 would be uniform and cannot test MaxEnt selection"
        )
    n_dropped = sum(1 for w in diversity_weights if w == 0)
    if n_dropped:
        print(f"[maxent] dropped {n_dropped}/{len(diversity_weights)} prompts from round 2 "
              f"({n_no_span} not threshold-spanning at {_floor}, "
              f"{n_dropped - n_no_span} zero bucket-diversity)", flush=True)
    return {prompt_id: w / total for prompt_id, w in zip(prompt_ids, diversity_weights)}


def select_continuation(
    cohort: list[dict],
    stats: dict[str, list[float]],
    mode: str,
    rng: random.Random,
    n: int,
    std_floor: float = 0.01,
    boxcox_lambda: float = BOXCOX_LAMBDA,
) -> list[dict]:
    """Draw round-two prompts with replacement from the fixed bootstrap cohort."""
    if n <= 0:
        raise ValueError(f"continuation size must be positive, got {n}")
    probabilities = continuation_probabilities(
        cohort,
        stats,
        mode,
        std_floor=std_floor,
        boxcox_lambda=boxcox_lambda,
    )
    weights = [probabilities[record["ptext"]] for record in cohort]
    selected = rng.choices(cohort, weights=weights, k=n)
    summary = " ".join(
        f"{prompt_id[:16]!r}:{probability:.4f}"
        for prompt_id, probability in sorted(
            probabilities.items(), key=lambda item: item[1], reverse=True
        )[:5]
    )
    print(f"[select:{mode}] n={n} top_probabilities={summary}", flush=True)
    return selected


def per_continuation(cohort, stats, mix, args, rng):
    """PER-style round-2 curriculum: soft priorities + fresh injection (E8).

    Spec: playbook-uncertainty-menu.md E8 (operator design 2026-09-03, formalized
    against Prioritized Experience Replay, Schaul et al. 2015). RL STAGE ONLY by
    convention — PER is DQN-lineage replay prioritization; the SFT stage keeps its
    Box-Cox class-balanced sampling and is deliberately untouched.

    Schaul conventions followed: proportional prioritization P(i) = p_i^alpha /
    sum p_j^alpha with p_i = priority + per_eps (the epsilon-floor: NOTHING is ever
    ineligible — a zero-probability prompt could regress unseen); new items enter at
    max priority; priorities re-measured only when sampled. Documented deviation from
    DQN PER: no importance-sampling weights, because priorities steer which prompts
    get FRESH rollouts — no stored sample is ever retrained, so there is no gradient
    to bias-correct.

    Priority = variance of Beta(1 + wins, 1 + losses) over the prompt's judge history
    at CORRECT_FLOOR. Unseen prompts have no history -> Beta(1,1), the distribution's
    maximum — the operator's "50/50, we don't know" prior with no special-casing.
    Fresh injection: n_fresh = max(2, ceil(0.10 * |measured cache|)), capped at
    ceil(target_pool / 3), drawn without replacement from prompts outside the cohort
    (visit-ledger count 0 first when a ledger is configured).

    Require: stats holds round-1 judge rewards per cohort ptext.
    Guarantee: returns target_pool records; prints the pinned '[per]' line.
    """
    import math
    from rlv2.judge.reward import CORRECT_FLOOR as _floor

    def _beta_var(w, l):
        a, b = 1 + w, 1 + l
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    # LEDGER-SEEDED PRIORS (operator, 2026-09-07): after a full-coverage pass every
    # record has a measured outcome on disk. Without this, a follow-on run starts
    # blind — in-run `stats` is empty, so every prompt looks Beta(1,1)-unseen and the
    # uncertainty phase cannot re-sample "what we are unsure of". In-run evidence
    # always wins; the ledger only fills prompts this run has not rolled out yet.
    _led_wl = {}
    _lp = getattr(args, "visit_ledger", None)
    if _lp and os.path.exists(_lp):
        import hashlib as _hl2
        for _k, _e in json.load(open(_lp, encoding="utf-8")).items():
            if isinstance(_e, dict):
                _led_wl[_k] = (int(_e.get("w", 0)), int(_e.get("l", 0)))
        _key_of = lambda pt: _hl2.sha1(pt.encode("utf-8")).hexdigest()[:16]
    else:
        _key_of = None

    cand = {r["ptext"]: r for r in cohort}
    prios = {}
    for pt in cand:
        rewards = stats.get(pt, [])
        if rewards:
            w = sum(1 for r in rewards if r >= _floor)
            l = len(rewards) - w
        elif _key_of is not None:
            w, l = _led_wl.get(_key_of(pt), (0, 0))
        else:
            w = l = 0
        prios[pt] = _beta_var(w, l)

    n_measured = sum(1 for pt in cand if stats.get(pt))
    n_fresh = min(max(2, math.ceil(0.10 * max(1, n_measured))),
                  math.ceil(args.target_pool / 3))
    seen = set(cand)
    outside = [r for r in {r["ptext"]: r for r in mix}.values()
               if r["ptext"] not in seen]
    rng.shuffle(outside)
    fresh = outside[:n_fresh]
    # W2 (playbook.md "WORLD MODEL"): informed priors for UNSEEN prompts, gated by the
    # ensemble's own disagreement — part 6's uncertainty-masking trust rule applied to
    # priors. Low disagreement -> pseudo-counts Beta(1+2*p_mean, 1+2*(1-p_mean));
    # high -> keep flat Beta(1,1). SAMPLED prompts always use their real posteriors
    # above — the model never overrides evidence. Selection-only: no gradient flows
    # from these predictions.
    _wm_used = 0
    if fresh and getattr(args, "world_model", None):
        from rlv2.train.rl.prompt_world_model import load_ensemble, predict as _wm_predict
        from rlv2.data.pools import embed as _wm_embed
        _heads = load_ensemble(args.world_model)
        _pm, _pv = _wm_predict(_heads, _wm_embed(
            [r.get("question") or r.get("prompt", "") for r in fresh]))
        for r, m_hat, v_hat in zip(fresh, _pm, _pv):
            if v_hat < args.wm_trust:
                prios[r["ptext"]] = _beta_var(2 * m_hat, 2 * (1 - m_hat))
                cand[r["ptext"]] = r
                _wm_used += 1
        if _wm_used == 0:
            # The Dyna-Q ensemble's empty-imagined-batch warning: log it, proceed flat.
            print(f"[wm] no fresh prompt under trust threshold {args.wm_trust} "
                  f"(min var {float(min(_pv)):.4f}) — flat priors", flush=True)
    for r in fresh:
        if r["ptext"] not in cand:
            cand[r["ptext"]] = r
            prios[r["ptext"]] = _beta_var(0, 0)   # Beta(1,1): max priority, PER's rule
        elif r["ptext"] not in prios:
            prios[r["ptext"]] = _beta_var(0, 0)

    alpha, eps = args.per_alpha, args.per_eps
    keys = list(cand)
    weights = [(prios[k] + eps) ** alpha for k in keys]
    sel = rng.choices([cand[k] for k in keys], weights=weights, k=args.target_pool)
    top = sorted(zip(keys, weights), key=lambda kv: kv[1], reverse=True)[:5]
    print(f"[per] cache={n_measured} fresh={len(fresh)} candidates={len(keys)} | "
          f"alpha={alpha} eps={eps} | top_p "
          + " ".join(f"{k[:12]!r}:{w:.4f}" for k, w in top), flush=True)
    return sel


def coverage_first_cohort(mix, args, already, rng):
    """Operator's curriculum rule (2026-09-07): VISIT COUNT DOMINATES.

    'Train once and done until all other records have been visited' — one visit
    per record is the measurement; settled records wait for full pool coverage;
    fresh (never-visited) records therefore always enter at the front, which is
    how real-world experience trickles in and equalizes coverage over old data.
    Uncertainty (Beta-posterior variance from the ledger's w/l) only breaks
    ties among equal visit counts. Within a fully-tied top tier the incumbent
    cross-hatch draw (bootstrap_cohort) keeps stratification.

    `already` = ptexts selected earlier in THIS run: no prompt repeats within a
    run while unvisited records remain (pool exhausted -> revisit allowed,
    least-visited first). Ledger entries may be bare ints (probe path) or
    {w,l,visits} dicts (arm-end writer) — both normalize here.
    """
    import hashlib as _hl

    ledger = {}
    _lp = getattr(args, "visit_ledger", None)
    if _lp and os.path.exists(_lp):
        for _k, _e in json.load(open(_lp, encoding="utf-8")).items():
            if isinstance(_e, dict):
                ledger[_k] = (int(_e.get("visits", 0)),
                              int(_e.get("w", 0)), int(_e.get("l", 0)))
            else:
                ledger[_k] = (int(_e or 0), 0, 0)

    def _beta_var(w, l):
        a, b = 1 + w, 1 + l
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def _prio(r):
        k = _hl.sha1(r["ptext"].encode("utf-8")).hexdigest()[:16]
        v, w, l = ledger.get(k, (0, 0, 0))
        return (v, -_beta_var(w, l))

    dedup = list({r["ptext"]: r for r in mix}.values())
    cands = [r for r in dedup if r["ptext"] not in already]
    exhausted = not cands
    if exhausted:
        cands = dedup
    min_v = min(_prio(r)[0] for r in cands)
    tier = [r for r in cands if _prio(r)[0] == min_v]
    if len(tier) > args.target_pool and len({_prio(r) for r in tier}) == 1:
        sel = bootstrap_cohort(tier, n=args.target_pool,
                               seed=rng.randrange(1 << 30),
                               boxcox_lambda=args.boxcox_lambda)
    else:
        cands.sort(key=_prio)
        sel = cands[:args.target_pool]
    print(f"[coverage] excluded={len(already)} candidates={len(cands)} "
          f"min_visits={min_v}{' POOL-EXHAUSTED' if exhausted else ''} "
          f"-> {len(sel)} selected", flush=True)
    return sel


def dyna_ipo_step(policy, tok, ledger, args):
    """Task #44 — Dyna-MGPO's offline update, run at each round boundary.

    Dyna's shape (Model-Free vs Model-Based): act in the environment,
    keep a model of it learned from experience, do extra updates offline. Here
    the environment is the judge (a bandit over answer-trajectories, so a model
    of the judge IS the complete world model), the ledger holds the round's REAL
    judged rollouts, and the offline update is one mini IPO pass — the policy's
    own likelihood-ratios are fitted to rank answers the way the judge did
    (DPO-family: the policy carries an implicit model of the judge). Replaying
    real logged outcomes rather than hallucinated ones is the safe half of Dyna
    to keep under a foolable judge.

    Pair rules are dpo_pass.build_pairs verbatim (chosen = best correct
    non-degenerate; rejected = worst nonempty, judge-fooled spam admitted).
    Returns the (possibly updated) policy; skips with a printed reason when the
    round yielded too few pairs. LoRA + implicit PEFT reference, merged back
    in-process so the next GRPO round trains the updated full weights.
    """
    from collections import defaultdict
    from rlv2.train.rl.dpo_pass import build_pairs
    from rlv2.judge.reward import CORRECT_FLOOR

    by_prompt = defaultdict(lambda: ([], []))
    for prompt, text, score in ledger:
        by_prompt[prompt][0].append(text)
        by_prompt[prompt][1].append(score)
    pool = [{"id": p, "question": p} for p in by_prompt]
    gens = {p: v[0] for p, v in by_prompt.items()}
    scores = {p: v[1] for p, v in by_prompt.items()}
    pairs = build_pairs(pool, gens, scores, CORRECT_FLOOR)
    if len(pairs) < args.dyna_min_pairs:
        print(f"[dyna-ipo] skipped: {len(pairs)} pairs < min {args.dyna_min_pairs}",
              flush=True)
        return policy
    pairs = pairs[:32]

    # COMPAT SHIM MUST PRECEDE THE TRL IMPORT (crash 2026-09-05: shim after the
    # import reproduces exactly the failure dpo_pass.py already solved).
    import transformers.models.auto.modeling_auto as _auto
    if not hasattr(_auto, "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES"):
        _auto.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = {}
    from datasets import Dataset
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer
    if not hasattr(policy, "warnings_issued"):
        policy.warnings_issued = {}
    # The ledger key IS the training ptext (fewshot chat template) — prompt
    # parity with the rollouts, no re-templating.
    ds = Dataset.from_list([{"prompt": p["question"], "chosen": p["chosen"],
                             "rejected": p["rejected"]} for p in pairs])
    cfg = DPOConfig(
        output_dir=args.out + "/_dyna_work", loss_type=["ipo"], beta=0.1,
        learning_rate=5e-6, num_train_epochs=1, per_device_train_batch_size=1,
        gradient_accumulation_steps=4, gradient_checkpointing=True,
        logging_steps=4, save_strategy="no", report_to=[], seed=args.seed,
        max_length=448, max_prompt_length=300)
    peft = LoraConfig(r=16, lora_alpha=64, lora_dropout=0.0,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = DPOTrainer(model=policy, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=peft)
    trainer.train()
    merged = trainer.model.merge_and_unload()
    # CRASH PIN (2026-09-05, round 2 step 1: "No inf checks were recorded for
    # this optimizer"): merge_and_unload returns the base with EVERY param still
    # frozen from the PEFT wrap, so the next GRPO round has zero trainable
    # params and the fp16 scaler asserts. Re-enable grads, then re-apply the
    # same embedding freeze the main path uses.
    for _n, _p in merged.named_parameters():
        _p.requires_grad_(not (getattr(args, "freeze_embeddings", False)
                               and ("embed_tokens" in _n or "lm_head" in _n)))
    print(f"[dyna-ipo] {len(pairs)} pairs consumed; adapter merged into policy "
          f"(grads re-enabled, embeddings "
          f"{'frozen' if getattr(args, 'freeze_embeddings', False) else 'live'})",
          flush=True)
    import torch as _t
    _t.cuda.empty_cache()
    return merged


def probe_disagreement_cohort(mix, args, meta, policy, tok):
    """Two-model epistemic cohort selection + visit-count bonus (E1+E2).

    Spec: docs/plans/uncertainty-menu-hpo.md (approved 2026-09-03) · playbook.md
    "UNCERTAINTY MENU" task UE. Operator: "I really want to see if I can track
    multiple models."

    WHY. The incumbent spanning selection measures ALEATORIC uncertainty — one
    model's outcome variance across G samples. This adds the EPISTEMIC axis
    (ensemble disagreement: where two differently-trained models disagree, nobody
    knows — explore there; cheat-sheet §4.3-4.4 twin critics, RND) and a NOVELTY
    axis (MCTS/UCT count bonus, §8.1): both angles COMPOSE with spanning, they do
    not replace it — round-2 selection is untouched.

    Mechanism: draw a candidate superset (probe_candidates_mult x target_pool) via
    the incumbent hierarchical sampler; generate probe_samples completions per
    candidate from the LIVE policy (already resident) and from the PROBE checkpoint
    (loaded fp16 TRANSIENTLY and freed before the trainer builds its ref model —
    the 13.2 GiB co-residency peak this function must stay clear of; probe fp16
    ~1.6 GB beside the fp32 policy ~3 GB is ~5 GB, inside the measured ceiling);
    judge everything in one llm_judge_flat call; a prompt where the two models'
    any-correct outcomes DIFFER is the epistemic frontier and fills the cohort
    first. Ties and the remainder order by ascending visit count (E2), persisted
    in --visit-ledger across passes so chained runs explore unvisited prompts.

    Require: policy resident, meta[ptext] carries question/gold, probe path exists.
    Guarantee: returns target_pool distinct records; prints the pinned
    '[probe]' line; never leaves the probe model in VRAM.
    """
    import gc
    import hashlib
    import json as _json
    from transformers import AutoModelForCausalLM as _AM, GenerationConfig as _GC
    from rlv2.judge.reward import CORRECT_FLOOR as _floor, llm_judge_flat

    n_target = args.target_pool
    n_cand = min(max(1, args.probe_candidates_mult) * n_target, len({r["ptext"] for r in mix}))
    candidates = bootstrap_cohort(mix, n=n_cand, seed=args.seed,
                                  boxcox_lambda=args.boxcox_lambda)

    ledger_path = getattr(args, "visit_ledger", None)
    visits = {}
    if ledger_path and os.path.exists(ledger_path):
        visits = _json.load(open(ledger_path, encoding="utf-8"))
    _key = lambda pt: hashlib.sha1(pt.encode("utf-8")).hexdigest()[:16]

    # GENERATE ON THE GPU, EXPLICITLY. At probe-screen time TRL has not placed the
    # policy yet -- model.device is CPU, and the first two launches of this screen
    # silently ran 96 fp32 generations on CPU (measured 2026-09-03: err-log pad
    # warnings ticking, python at 4.1 GB SYSTEM ram, GPU at 1.0/16 GB, ~4 h pace,
    # read as a stall twice). Device is chosen here, never inferred from the model.
    _dev = "cuda" if torch.cuda.is_available() else "cpu"

    def _gen(model, records, tag):
        outs = []
        model.eval()
        with torch.no_grad():
            for _i, r in enumerate(records):
                ids = tok(r["ptext"], return_tensors="pt").input_ids.to(_dev)
                for _ in range(args.probe_samples):
                    g = model.generate(
                        ids, max_new_tokens=args.max_completion, do_sample=True,
                        temperature=args.temperature, top_p=args.top_p,
                        generation_config=_GC(suppress_tokens=_banned(tok)))
                    outs.append(tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True))
                if (_i + 1) % 6 == 0:
                    print(f"[probe-gen] {tag} {_i + 1}/{len(records)}", flush=True)
        return outs

    _policy_dev0 = next(policy.parameters()).device
    policy.to(_dev)
    policy_texts = _gen(policy, candidates, "policy")
    policy.to(_policy_dev0)   # hand back exactly what TRL expects to place itself
    # DYNAMIC SAMPLING (DAPO §3, "Dynamic Sampling"): a prompt whose samples all
    # land on the same side of the floor has zero group variance, so its advantage
    # is zero and the step teaches nothing. Measured here: 35-50% of prompts were
    # dead this way. Rather than pay for them, screen them out BEFORE training and
    # fill the cohort with prompts whose own samples come out MIXED. Single model,
    # so no probe checkpoint is loaded and the screen costs one generation pass.
    _dynamic = getattr(args, "dynamic_sampling", False)
    if _dynamic:
        probe_texts = []
    else:
        probe = _AM.from_pretrained(args.probe_base, dtype=torch.float16,
                                    trust_remote_code=True).to(_dev)
        probe_texts = _gen(probe, candidates, "probe")
        del probe
        gc.collect()
        torch.cuda.empty_cache()

    pairs = []
    for texts in ((policy_texts,) if _dynamic else (policy_texts, probe_texts)):
        for i, r in enumerate(candidates):
            m = meta[r["ptext"]]
            for s in range(args.probe_samples):
                pairs.append({"question": m.get("question") or m.get("prompt", ""),
                              "gold": m.get("gold", ""),
                              "context": m.get("context", "") or "",
                              "generated": texts[i * args.probe_samples + s]})
    # CHUNKED, with progress. The first version passed all pairs to one
    # llm_judge_flat call and STALLED for 95 minutes with frozen logs (measured
    # 2026-09-03, past9_v2): a ~96-question object-schema prompt is outside what
    # the judge reliably completes, and the recursive split retries silently.
    # score_arms never does this — it judges per-question. Chunks of 8 match the
    # judge's proven envelope; llm_judge_flat_many keeps 2 requests in flight.
    from rlv2.judge.reward import llm_judge_flat_many
    _chunks = [pairs[i:i + 8] for i in range(0, len(pairs), 8)]
    scores = []
    for _ci, _res in enumerate(llm_judge_flat_many(_chunks)):
        scores.extend(_res)
        print(f"[probe-judge] chunk {_ci + 1}/{len(_chunks)} done", flush=True)
    k, half = args.probe_samples, len(candidates) * args.probe_samples
    verdict = []
    for i in range(len(candidates)):
        _pol = scores[i * k:(i + 1) * k]
        if _dynamic:
            # MIXED means at least one sample each side of the floor. That is the
            # only condition under which the group has non-zero variance, which is
            # the only condition under which GRPO gets a gradient from it.
            _hit = sum(1 for s in _pol if s >= _floor)
            verdict.append((_hit > 0, _hit == len(_pol)))   # mixed <=> differ
        else:
            pol_ok = any(s >= _floor for s in _pol)
            prb_ok = any(s >= _floor for s in scores[half + i * k:half + (i + 1) * k])
            verdict.append((pol_ok, prb_ok))

    # Disagreement first (epistemic frontier), then ascending visit count (novelty),
    # then the hierarchical order the sampler already chose.
    order = sorted(range(len(candidates)),
                   key=lambda i: (0 if verdict[i][0] != verdict[i][1] else 1,
                                  visits.get(_key(candidates[i]["ptext"]), 0), i))
    cohort = [candidates[i] for i in order[:n_target]]
    n_dis = sum(1 for v in verdict if v[0] != v[1])
    n_dis_sel = sum(1 for i in order[:n_target] if verdict[i][0] != verdict[i][1])
    _label = "mixed" if _dynamic else "disagree"
    print(f"[probe] candidates={len(candidates)} {_label}={n_dis} | "
          f"cohort={n_target} ({n_dis_sel} {_label}, "
          f"{n_target - n_dis_sel} novelty/hierarchical) | "
          f"{'dynamic-sampling (single model)' if _dynamic else 'probe=' + str(args.probe_base)}",
          flush=True)
    if _dynamic and n_dis == 0:
        raise RuntimeError(
            "dynamic sampling found NO mixed prompt among "
            f"{len(candidates)} candidates at floor {_floor}: every candidate was "
            "uniformly right or uniformly wrong, so no cohort can produce a gradient. "
            "Raise --probe-candidates-mult or --probe-samples.")

    if ledger_path:
        for r in cohort:
            visits[_key(r["ptext"])] = visits.get(_key(r["ptext"]), 0) + 1
        os.makedirs(os.path.dirname(ledger_path) or ".", exist_ok=True)
        _json.dump(visits, open(ledger_path, "w", encoding="utf-8"))
    return cohort


def _release_round_vram(offload_models, tag=""):
    """Drop round N's GPU residents so round N+1's policy never co-resides with them.

    Spec: playbook.md Layer 7 task RV1 (round-boundary VRAM).

    ROOT CAUSE OF THE ROUND-2 OOM, measured 2026-09-08 across two runs (grad-accum
    12 and grad-accum 6, both dying at round 2 while round 1 completed at the same
    accumulation, which is what rules accumulation out as the driver):

        torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 410.00 MiB.
        GPU 0 has a total capacity of 16.00 GiB of which 0 bytes is free.
        14.78 GiB allowed; Of the allocated memory 13.67 GiB is allocated

    The 2026-09-02 `tr = None` fix released the TRAINER and nothing else.
    `_offload_models` -- the list the reward closure holds so it can push models to
    CPU while the judge runs -- still held round N's policy AND TRL's ref model, so
    two fp32 0.8B models (~3.2 GiB each, plus their .grad tensors) stayed pinned
    while `_reload_best` built a third. That is the ~3.8 GiB gap between the
    9.40 GiB peak measured on a no-reload run and the 13.2 GiB measured with one.

    Clearing the list is the fix. The caller must also drop its own `policy` and
    `tr` names first -- this releases only what the list pins.
    """
    import gc
    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    for m in list(offload_models or []):
        for p in getattr(m, "parameters", list)():
            p.grad = None
    if offload_models is not None:
        offload_models.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated()
        print(f"[vram] {tag} released {(before - after) / 2**30:.2f} GiB; "
              f"{after / 2**30:.2f} GiB still allocated", flush=True)


def _reload_best(args, best_dir, best_score, mode):
    """Reload the best patience checkpoint, as an ADAPTER or as a MODEL.

    THIRD SITE OF THE SAME BUG. Full-param `save_pretrained` writes a plain model with no
    adapter_config.json, so every `PeftModel.from_pretrained` call on a saved checkpoint dies
    with `Can't find 'adapter_config.json'`. That was fixed at the mid-eval call and at the
    guard, and then reappeared here -- twice, in two near-identical blocks -- and killed a
    full-param run AFTER it had trained successfully to 7/16.

    Two copies of a rule is how the third divergence happens, so the rule now lives once.
    Note the dtype must match the run's training dtype (fp32 master for full-param), or the
    reloaded policy silently differs from the one that produced best_score.
    """
    # Imported here, not at module scope: transformers is imported inside run_arm so the
    # CUDA-alloc env var above lands before torch initialises.
    from transformers import AutoModelForCausalLM
    if getattr(args, "full_param", False):
        policy = AutoModelForCausalLM.from_pretrained(
            best_dir, dtype=torch.float32, trust_remote_code=True)
        print(f"  [patience] loaded BEST full-param model from {best_dir} "
              f"(score={best_score:.4f})", flush=True)
        return policy
    from peft import PeftModel
    policy = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float16,
                                                  trust_remote_code=True)
    policy = PeftModel.from_pretrained(policy, best_dir, is_trainable=True)
    print(f"  [patience] loaded BEST adapter from {best_dir} (score={best_score:.4f})",
          flush=True)
    return policy


def run_arm(mode, mix, args,
    prompt_style: str = "abstain",
):
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    from rlv2.common.cost_meter import assert_gpu_exclusive, enforce_vram_ceiling
    from rlv2.judge.reward import unload_all_ollama_models
    work = f"{OUT_DIR}/{mode}"
    # Resume ONLY on an adapter whose recorded config matches this run. Existence alone is not
    # provenance: `maxent` was resumed off a killed 4x75 adapter while `uniform` trained 2x12, so the
    # two arms differed on the one axis held constant -- and because the resume skipped eval, the
    # stale 0.000 score survived a later fix to prompt construction. An UNTAGGED adapter predates
    # this check and is therefore exactly the suspect case: fail closed and retrain.
    # The tag must identify the INPUTS, not just the knobs. Tagging on rounds x steps alone let a
    # PREVIOUS night's `maxent` entry (same 2x20, different base model) survive the guard, so the arm
    # was skipped and its stale numbers were printed as this run's -- voiding the mechanism check.
    # Same defect class as the CPT stage tag that ignored corpus size, one file over.
    _sft_tag = f"|sft={os.path.basename(args.sft_adapter)}" if getattr(args, "sft_adapter", None) else ""
    tag = (
        f"{args.rounds}x{args.steps}p{args.target_pool}@{args.lr:g}|{args.base}|"
        f"{os.path.basename(args.mix)}:{len(mix)}|r{args.rank}a{args.alpha}{_sft_tag}|"
        f"g{args.num_generations}|t{args.temperature:g}|p{args.top_p:g}|k{args.top_k}|"
        f"bc{args.boxcox_lambda:g}|prompt{args.max_prompt}|completion{args.max_completion}|"
        f"beta{args.beta:g}|loss{args.loss_type}|obs{args.observe_every}|"
        f"fp16scale={args.fp16_init_scale if args.fp16_init_scale is not None else 'default'}|"
        f"minB{args.min_adapter_b:g}|seed{args.seed}|"
        f"eval={args.eval_file}|mini={args.mini_eval_file}"
    )
    stamp = os.path.join(work, "run_cfg.json")
    if os.path.exists(os.path.join(work, "adapter_model.safetensors")):
        prev = json.load(io.open(stamp, encoding="utf-8")).get("cfg") if os.path.exists(stamp) else None
        if prev == tag:
            print(f"[{mode}] resume (cfg={tag} matches)", flush=True)
            return work
        print(f"[{mode}] STALE adapter -- built at {prev or 'UNTAGGED'}, this run is {tag}; retraining",
              flush=True)
        shutil.rmtree(work, ignore_errors=True)
    elif os.path.exists(work):
        print(f"[{mode}] INCOMPLETE output without final adapter; removing {work}", flush=True)
        shutil.rmtree(work)
    assert_gpu_exclusive()
    unload_all_ollama_models()
    enforce_vram_ceiling(vram_frac=float(os.environ.get("HARNESS_VRAM_FRAC", "1.0")))
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Working TRL pattern for this repo (CLAUDE.md): wrap the adapter OURSELVES and pass the
    # PeftModel as `model=`, pass `processing_class=`, and pass NO `peft_config` -- peft_config makes
    # TRL re-wrap down a path that discards processing_class and then fails on an image processor.
    # DTYPE DEPENDS ON WHAT IS TRAINABLE. LoRA loads fp16 because the frozen base never
    # produces gradients and PEFT creates the adapter params in fp32 -- GradScaler unscales
    # those fp32 grads happily. FULL-PARAM has no such split: every weight is trainable, so
    # an fp16 load yields fp16 gradients and torch raises
    #     ValueError: Attempting to unscale FP16 gradients.
    # at the first clip_grad_norm_ (measured 2026-09-02, full-param probe). Mixed precision
    # requires fp32 master weights; fp16=True + autocast still does the matmuls in fp16, so
    # this costs memory (3.0 GB weights + 3.0 GB grads) but not speed. Combined with 8-bit
    # Adam the static total is ~7.5 GB, inside the ~13.4 GB budget.
    _policy_dtype = torch.float32 if getattr(args, "full_param", False) else torch.float16
    policy = AutoModelForCausalLM.from_pretrained(args.base, dtype=_policy_dtype,
                                                  trust_remote_code=True)
    if getattr(args, 'full_param', False):
        # Full-FT: ALL parameters trainable (attention + FFN + embeddings + norms).
        # Testing whether attention heads need to update alongside FFN for stability.
        #
        # EXCEPT the embedding, optionally. This model has vocab_size=248320 at hidden=1024
        # with tie_word_embeddings=True, so embed_tokens/lm_head is a single 254M-param
        # tensor -- 34% OF THE WHOLE 752M MODEL. Training it costs fp32 grads (~1.0 GB) plus
        # 8-bit Adam state (~0.5 GB) for a tensor an RL fine-tune has little reason to move:
        # GRPO is reweighting existing behaviour, not learning new vocabulary. Freezing it is
        # also the standard first move against fp16 NaN, since embedding rows accumulate
        # gradient from every token occurrence and overflow first.
        if getattr(args, "freeze_embeddings", False):
            _frozen_emb = 0
            for _n, _p in policy.named_parameters():
                if "embed_tokens" in _n or "lm_head" in _n:
                    _p.requires_grad_(False)
                    _frozen_emb += _p.numel()
            print(f"[{mode}] embeddings FROZEN: {_frozen_emb/1e6:.1f}M params held out "
                  f"(tied embed/lm_head, vocab={getattr(policy.config, 'vocab_size', '?')})",
                  flush=True)
        n_tr = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        n_frz = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
        print(f"[{mode}] FULL-PARAM all weights: {n_tr/1e6:.1f}M trainable, "
              f"{n_frz/1e6:.1f}M frozen", flush=True)
    elif getattr(args, 'sft_adapter', None):
        # CONTINUE TRAINING the SFT adapter with RL loss. Same adapter, same weights,
        # just switching objective from completion-loss to GRPO reward. No new adapter,
        # no merge, no stacking. The adapter picks up where SFT left off.
        from peft import PeftModel
        sft_path = args.sft_adapter
        if not os.path.exists(os.path.join(sft_path, "adapter_model.safetensors")):
            raise FileNotFoundError(
                f"--sft-adapter path has no adapter_model.safetensors: {sft_path}"
            )
        policy = PeftModel.from_pretrained(policy, sft_path, is_trainable=True)
        # Verify adapter geometry matches expected rank/alpha
        adapter_cfg = policy.peft_config["default"]
        if adapter_cfg.r != getattr(args, 'rank', LORA_R):
            raise ValueError(
                f"SFT adapter rank={adapter_cfg.r} != --rank {getattr(args, 'rank', LORA_R)}. "
                f"Alpha/rank must be SAME across all LoRA stages."
            )
        if adapter_cfg.lora_alpha != getattr(args, 'alpha', LORA_ALPHA):
            raise ValueError(
                f"SFT adapter alpha={adapter_cfg.lora_alpha} != --alpha {getattr(args, 'alpha', LORA_ALPHA)}. "
                f"Alpha must be SAME across all LoRA stages."
            )
        n_tr = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"[{mode}] CONTINUE SFT adapter from {sft_path} | "
              f"r={adapter_cfg.r} alpha={adapter_cfg.lora_alpha} | "
              f"{n_tr/1e6:.1f}M trainable (same adapter, RL objective)", flush=True)
    else:
        # Resolve target modules from the flag
        if getattr(args, 'target_modules', 'all-linear') == "all-linear":
            targets = ["in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj",
                       "q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
        else:
            targets = [t.strip() for t in args.target_modules.split(",")]
        lora_r = getattr(args, 'rank', LORA_R)
        lora_a = getattr(args, 'alpha', LORA_ALPHA)
        policy = get_peft_model(policy, LoraConfig(
            r=lora_r, lora_alpha=lora_a, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=targets))
        n_tr = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"[{mode}] LoRA r={lora_r} alpha={lora_a} over {len(targets)} module types: "
              f"{n_tr/1e6:.1f}M trainable", flush=True)
    # TRAIN AND EVAL MUST SEE THE SAME PROMPT FORMAT. This trained on raw prompt strings while eval
    # used the chat template + few-shot exemplars, which broke the run two ways:
    #   1. raw prompts collapse this model's output to ~4 tokens (measured: base scores 0.000 raw vs
    #      0.533 templated), so every completion in a group scores F1~0, within-group reward std is
    #      ~0 (frac_reward_zero_std 0.84-0.96), the GRPO advantage is 0, and grad_norm logged 0 --
    #      the policy never moved, which is why both arms tied at every seed;
    #   2. even had it learned, it optimised a format it was never scored on.
    # `ptext` is the exact string TRL returns to the reward function, so metadata must be keyed by
    # that exact value. A miss now raises before any score is assigned; silently substituting empty
    # metadata would corrupt the judge's question/gold/context tuple.
    from rlv2.eval import cpt_fewshot_lift as FS
    _shots = FS._squad_shots()

    # TRAIN/EVAL PROMPT PARITY. The eval runs with --prompt-style abstain, which builds
    # prompts via `abstain_messages` -- a style that PERMITS declining. Rollouts here used
    # `FS._fewshot_messages`, whose own docstring in rl_calibration_eval.py describes it as
    # instructing the model never to decline and showing it four exemplars that never
    # decline. Training calibration under a prompt that forbids abstention and then scoring
    # it under one that permits abstention measures the PROMPT, not the policy.
    #
    # That mismatch is not theoretical: it killed a run at 09:15 on 2026-08-28 with
    # "MGPO kill gate: all completions were class='answer' for 11 consecutive steps". The
    # gate was right -- under a forbidding prompt every completion IS an answer, so the
    # within-group variance this lane's reward depends on cannot exist.
    #
    # The confound was found on 2026-08-03 and fixed in the EVAL only; this carries the same
    # fix into the trainer.
    if prompt_style == "abstain":
        from rlv2.eval.rl_calibration_eval import abstain_messages as _build
    else:
        _build = FS._fewshot_messages
    for r in mix:
        r["ptext"] = tok.apply_chat_template(_build(r["prompt"], _shots),
                                             tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    # A ptext collision is only a DEFECT when the colliding records disagree -- that would hand the
    # reward function the wrong gold and silently score the wrong thing. Exact duplicates are benign
    # and are in fact the mechanism by which the mix is abstention-balanced: R2/R4 are upsampled by
    # repetition (79 -> 519, 76 -> 519), which is exactly how GRPO's per-step prompt sampling is
    # weighted toward them. `src/rlv2/common/stratify.py` does the same thing -- "rare cells are drawn WITH
    # replacement (repetition) to meet their allocation".
    #
    # The original `len(meta) == len(mix)` check conflated the two and killed a balanced run at
    # 883 duplicates. Assert the thing that actually matters: no two records under one key disagree.
    meta, dupes = {}, 0
    for r in mix:
        prev = meta.get(r["ptext"])
        if prev is not None:
            fields = ("gold", "context", "domain", "source")
            if tuple(prev.get(f) for f in fields) != tuple(r.get(f) for f in fields):
                raise AssertionError(
                    f"ptext collision with DISAGREEING records -- the reward would be computed "
                    f"against the wrong gold. prompt[:80]={r['prompt'][:80]!r}")
            dupes += 1
        meta[r["ptext"]] = r
    if dupes:
        print(f"[mix] {dupes} duplicate prompts collapse to {len(meta)} unique keys "
              f"(upsampling by repetition -- all colliding records verified identical)", flush=True)
    # TruncationFreeSampling (Invariant 4): SELECT prompts that fit, don't let TRL truncate.
    # A truncated few-shot prompt loses the system message and exemplars, producing garbage
    # input that the model responds to with a 4-token EOS — measured mean_length=4.75 tokens.
    max_prompt_tokens = args.max_prompt
    max_completion_tokens = args.max_completion
    pre_filter_n = len(mix)
    mix = [r for r in mix
           if len(tok.encode(r["ptext"])) <= max_prompt_tokens
           and len(tok.encode(r.get("gold", r.get("answer", "")))) <= max_completion_tokens]
    # Rebuild meta from surviving records only
    meta = {r["ptext"]: r for r in mix}
    dropped = pre_filter_n - len(mix)
    if dropped:
        print(f"[prefilter] dropped {dropped}/{pre_filter_n} records "
              f"(prompt>{max_prompt_tokens} OR gold>{max_completion_tokens} tokens) "
              f"— TruncationFreeSampling", flush=True)
    else:
        print(f"[prefilter] all {len(mix)} records fit within prompt={max_prompt_tokens} "
              f"completion={max_completion_tokens}", flush=True)
    # Regime filter: if --regimes specified, keep only those regimes in the training pool.
    # Default: all regimes. For answer-quality-only RL, pass --regimes R1 R3.
    if hasattr(args, 'regimes') and args.regimes:
        from rlv2.common.stratify import regime_of as _regime_filter
        pre_regime_n = len(mix)
        allowed = set(args.regimes)
        mix = [r for r in mix if _regime_filter(r) in allowed]
        meta = {r["ptext"]: r for r in mix}
        print(f"[regime-filter] kept {len(mix)}/{pre_regime_n} records "
              f"(regimes={sorted(allowed)})", flush=True)
    # Domain filter: if --domain specified, keep only that domain class.
    # Parallel-RL trains one adapter per domain; cross-hatch still balances answer_type within.
    if hasattr(args, 'domain') and args.domain:
        pre_domain_n = len(mix)
        target_domain = args.domain
        mix = [r for r in mix if (r.get("domain") or r.get("category", "unknown")) == target_domain]
        meta = {r["ptext"]: r for r in mix}
        if not mix:
            raise ValueError(
                f"--domain {target_domain!r} filtered ALL records. "
                f"Available domains: {sorted(set(r.get('domain','unknown') for r in _load(args.mix)))}"
            )
        print(f"[domain-filter] kept {len(mix)}/{pre_domain_n} records "
              f"(domain={target_domain!r})", flush=True)
    stats = defaultdict(list)
    rng = random.Random(args.seed if hasattr(args, 'seed') else 7)
    if getattr(args, "coverage_first", False):
        # Coverage-first (operator, 2026-09-07): round 0 draws the least-visited
        # records; on a fresh ledger this reduces to the incumbent cross-hatch.
        cohort = coverage_first_cohort(mix, args, set(), rng)
    elif getattr(args, "probe_base", None) or getattr(args, "dynamic_sampling", False):
        # Both routes screen candidates by generating and judging before training.
        # --probe-base ranks by disagreement between two models (epistemic);
        # --dynamic-sampling ranks by mixedness within ONE model's own samples,
        # which is the condition for a non-zero group variance (DAPO).
        cohort = probe_disagreement_cohort(mix, args, meta, policy, tok)
    else:
        cohort = bootstrap_cohort(
            mix,
            n=args.target_pool,
            seed=args.seed,
            boxcox_lambda=args.boxcox_lambda,
        )
    # Coverage-first bookkeeping: ptexts already selected THIS run — 'train once
    # and done until all other records have been visited'.
    _visited_this_run = set()
    print(
        f"[bootstrap] cohort={len(cohort)} distinct prompts | seed={args.seed} | "
        "round 1 uniform; round 2 mode-specific",
        flush=True,
    )
    zero_std_log, round_selections, t0 = [], [], time.time()
    health_monitor = TrainingHealthMonitor(
        observe_every=args.observe_every,
        class_collapse_patience=args.class_collapse_patience,
        # THE GATE COUNTS RECORDS, NOT OPTIMIZER STEPS (measured 2026-09-07: the
        # first full-coverage run died at step 6 of 20 on
        # "frac_reward_zero_std=1.0 persisted for 6 consecutive steps"). record()
        # is called once per PROMPT, so the default patience of 6 means six
        # consecutive uniform-outcome RECORDS — which a coverage walk hits by
        # construction, since ~40% of unvisited records are uniformly right or
        # uniformly wrong and only contested ones carry gradient. Under gradient
        # accumulation the unit that matters is the optimizer step, so the
        # patience scales with records-per-step: 6 dead STEPS, as intended.
        zero_std_patience=ZERO_STD_PATIENCE * max(1, getattr(args, "grad_accum", 1)),
        # On the r3 lane one class IS the spec (closed-book, no abstention); see __init__.
        watch_class_collapse=(getattr(args, "lane", "vibe") != "r3"),
        # The r3 mix is 100% R3 by construction; the vibe lane keeps the 3-regime demand.
        min_regimes=(1 if getattr(args, "lane", "vibe") == "r3" else 3),
    )
    # Mutable container for models to offload during judge scoring.
    # Populated after trainer init so the reward closure can reach both policy AND ref.
    _offload_models = []
    # Patience/lift state OUTSIDE the round loop: a lift streak that spans the round
    # boundary is still sequential, and the HPO summary below reads it after the loop.
    _best_score = [0.0]
    _best_dir = [None]
    _patience_counter = [0]
    # patience 0 = stop on the FIRST non-improvement (the HPO criterion is 3 sequential
    # lifts, so a config that ever plateaus is already out). Default 2 preserves the
    # production behaviour.
    _PATIENCE = max(0, getattr(args, "patience", 2))
    # Lift accounting for HPO: eval #1 seeds the baseline and is NOT a lift -- the old
    # best=0.0 seed made the first eval a guaranteed "improvement" and once handed the
    # scorer gate an untrained step-2 adapter as "best" (2026-08-31 smoke).
    _n_evals = [0]
    _lift_streak = [0, 0]  # [current consecutive lifts, max ever]
    _stopped_early = [False]
    # Task #36: every patience eval is RECORDED, not just printed — the per-step
    # numbers land in train_stats.json as eval_history so no question about a run's
    # curve ever requires grepping stdout logs again.
    _eval_history = []
    # Task #44: the Dyna replay ledger — filled by the reward path (real judged
    # rollouts), consumed by dyna_ipo_step at each round boundary, cleared after.
    _dyna_ledger = [] if getattr(args, "dyna_ipo", False) else None
    for rd in range(args.rounds):
        if rd > 0 and _dyna_ledger is not None and _dyna_ledger:
            # The ledger PERSISTS across rounds (textbook replay-buffer semantics).
            # Clearing it per round starved the first trial: one round at
            # target_pool 6 / G=4 yielded only 2 pairs ([dyna-ipo] skipped,
            # 2026-09-05) because most groups are uniformly right or wrong.
            policy = dyna_ipo_step(policy, tok, _dyna_ledger, args)
        if rd == 0:
            sel = list(cohort)
        elif getattr(args, "coverage_first", False):
            # Fresh least-visited cohort every round; no repeats within the run.
            sel = coverage_first_cohort(mix, args, _visited_this_run, rng)
        elif getattr(args, "per_sampling", False):
            sel = per_continuation(cohort, stats, mix, args, rng)
        else:
            sel = select_continuation(
                cohort,
                stats,
                mode=mode,
                rng=rng,
                n=args.target_pool,
                boxcox_lambda=args.boxcox_lambda,
            )
        round_selections.append([record["ptext"] for record in sel])
        _visited_this_run.update(record["ptext"] for record in sel)

        # ── Regime census assertion ──────────────────────────────────────────
        from rlv2.common.stratify import regime_of as _regime_of
        regimes_present = {_regime_of(r) for r in sel}
        if getattr(args, "lane", "vibe") == "r3":
            # The R3 lane trains on ONE regime by construction (closed-book, no abstention).
            # The census still runs -- inverted: anything BUT R3 in the cohort means the mix
            # or the partition leaked, which is worse than a missing regime.
            if regimes_present != {"R3"}:
                raise RuntimeError(
                    f"round {rd + 1} contains non-R3 regimes {sorted(regimes_present - {'R3'})}; "
                    "the r3 lane trains on the R3 partition only"
                )
        elif len(regimes_present) < 4:
            missing = {"R1", "R2", "R3", "R4"} - regimes_present
            raise RuntimeError(
                f"round {rd + 1} missing regimes {sorted(missing)}; "
                "the matched four-regime contract is not satisfied"
            )

        # ── Regime-interleave with within-regime length sort ──────────────────
        # Round-robin across regimes by descending count, shortest first within
        # each regime. One single TRL trainer per round — no multi-block cascade.
        # The prior proven 400-step run used one trainer and was stable;
        # the multi-block approach introduced degeneration at block boundaries.
        regime_buckets: dict[str, list[dict]] = {}
        for r in sel:
            reg = _regime_of(r)
            regime_buckets.setdefault(reg, []).append(r)
        for reg in regime_buckets:
            regime_buckets[reg].sort(key=lambda r: len(tok.encode(r["ptext"])))
        ordered: list[dict] = []
        while any(regime_buckets.values()):
            for reg in sorted(regime_buckets, key=lambda r: len(regime_buckets[r]), reverse=True):
                if regime_buckets[reg]:
                    ordered.append(regime_buckets[reg].pop(0))
            regime_buckets = {r: b for r, b in regime_buckets.items() if b}
        sel = ordered
        max_prompt_in_round = max(len(tok.encode(r["ptext"])) for r in sel)

        ds = Dataset.from_list([{"prompt": r["ptext"]} for r in sel])
        cfg = GRPOConfig(
            output_dir=f"{work}/run_{rd}",
            per_device_train_batch_size=args.num_generations,
            # RECORDS PER OPTIMIZER STEP (operator, 2026-09-07). Hardcoded 1 meant
            # one step = ONE prompt, so the 2x6 shape only ever touched 6 of RL's
            # own 240 records (2.5% coverage). Accumulation N makes one step
            # consume N prompts, which is what makes a full-coverage pass fit the
            # hard 2-round constraint (2 rounds x 10 steps x 12 = 240).
            gradient_accumulation_steps=max(1, getattr(args, "grad_accum", 1)),
            num_generations=args.num_generations,
            generation_batch_size=args.num_generations,
            max_steps=args.steps,
            learning_rate=args.lr,
            # Default ("linear") couples the decay horizon to steps-per-round, so changing
            # the round length silently changes the effective lr at every step — measured
            # as HPO mechanism 3 (trials 10/14, 2026-09-01). "constant" decouples them.
            lr_scheduler_type=getattr(args, "lr_scheduler", "linear"),
            max_prompt_length=max_prompt_in_round + 16,
            max_completion_length=args.max_completion,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=None if args.top_k < 0 else args.top_k,
            fp16=True,
            optim=getattr(args, "optim", "adamw_torch"),
            max_grad_norm=getattr(args, "max_grad_norm", 1.0),
            logging_steps=max(1, args.steps // OBSERVE_EVERY),
            save_strategy="no",
            report_to=[],
            gradient_checkpointing=True,
            beta=args.beta,
            epsilon=args.epsilon,
            epsilon_high=args.epsilon_high,
            seed=args.seed,
            data_seed=args.seed,
            dataloader_pin_memory=False,
            loss_type=args.loss_type,
            # Task #38 (BREAK THE 9 R1a): Dr. GRPO's finding is that dividing the
            # group advantage by its std re-weights prompts by difficulty; "none"
            # removes that scaling. Default "group" preserves every prior run.
            scale_rewards=args.scale_rewards,
            generation_kwargs={"suppress_tokens": _banned(tok)},
        )

        # ── Patience-based early stopping with mid-round canonical eval ──────
        # Optional canonical patience eval; disabled for bounded mechanism smokes.
        # Metric tags derive from the selected eval file cardinality.
        from transformers import TrainerCallback as _TC
        _eval_interval = args.eval_every if args.eval_every and args.eval_every > 0 else None
        # _best_score / _patience_counter / _lift_streak live OUTSIDE the round loop (above):
        # streaks and the baseline persist across rounds; only the callbacks are per-round.

        # r3 lane: patience compares on the LADDER metric -- binary rate at CORRECT_FLOOR
        # on the frozen 16-item eval -- through half_epoch_eval, the same judge (4b) and
        # rubric as every other stage's gate. min_delta is the instrument's MEASURED
        # resolution (score_arms.RESOLUTION_FLOOR = 0.0526): an 'improvement' smaller than
        # what the instrument can resolve must not reset patience, at any stage.
        _lane = getattr(args, "lane", "vibe")
        _r3_eval_rows = None

        def _r3_patience_eval(state, control, model):
            nonlocal _r3_eval_rows
            from rlv2.eval import half_epoch_eval as _he
            from rlv2.eval.score_arms import RESOLUTION_FLOOR as _min_delta
            from rlv2.judge.reward import CORRECT_FLOOR as _cf
            if _r3_eval_rows is None:
                _r3_eval_rows = _he.load_eval(args.eval_file)
            _tag = f"_rl_{mode}_rd{rd}_s{state.global_step}"
            _summary = _he.evaluate(model, tok, _r3_eval_rows, _tag)
            if not _summary:
                raise RuntimeError(f"r3 patience eval produced no summary at step {state.global_step}")
            _recs = _summary.get("records", [])
            _n = len(_recs)
            _corr = [float(r["reward"]) for r in _recs if r.get("verdict") == "correct"]
            _wrong = [float(r["reward"]) for r in _recs if r.get("verdict") != "correct"]
            _rate = len(_corr) / max(1, _n)
            _mc = sum(_corr) / len(_corr) if _corr else float("nan")
            # KL(correct || incorrect) via the shared estimator -- separation diagnostic only.
            try:
                from rlv2.eval.score_arms import kl_divergence as _kl_fn
                _kl = _kl_fn(_corr, _wrong)
            except Exception:                                    # noqa: BLE001
                _kl = float("nan")
            print(f"  [patience-eval:r3] step={state.global_step} "
                  f"binary@{_cf:.2f}={_rate:.4f} ({len(_corr)}/{_n})  "
                  f"mean|correct={_mc:.4f}  KL={_kl:.4f}  "
                  f"best={_best_score[0]:.4f}  min_delta={_min_delta}  "
                  f"patience={_patience_counter[0]}/{_PATIENCE}", flush=True)
            _eval_history.append({
                "round": rd, "step": int(state.global_step), "lane": "r3",
                "correct": len(_corr), "n": _n, "rate": _rate,
                "mean_correct": _mc, "kl": _kl,
            })
            _n_evals[0] += 1
            if _n_evals[0] == 1:
                # Eval #1 is the BASELINE, not a lift: seed best with the input model's own
                # rate (no min_delta -- it is not an improvement claim, it is the anchor).
                _best_score[0] = _rate
                print(f"  [patience] baseline seeded at {_rate:.4f}; lift counting starts "
                      f"next eval", flush=True)
                return
            if _rate > _best_score[0] + _min_delta:
                _best_score[0] = _rate
                _patience_counter[0] = 0
                _lift_streak[0] += 1
                _lift_streak[1] = max(_lift_streak[1], _lift_streak[0])
                _best_path = f"{work}/_best_rd{rd}"
                if os.path.exists(_best_path):
                    shutil.rmtree(_best_path)
                os.makedirs(_best_path, exist_ok=True)
                model.save_pretrained(_best_path)
                tok.save_pretrained(_best_path)
                _best_dir[0] = _best_path
                print(f"  [patience] NEW BEST saved -> {_best_path} "
                      f"| lifts={_lift_streak[0]} (max {_lift_streak[1]})", flush=True)
            else:
                _lift_streak[0] = 0
                _patience_counter[0] += 1
                if _patience_counter[0] >= _PATIENCE:
                    print(f"  [patience] EARLY STOP at step {state.global_step} "
                          f"(best={_best_score[0]:.4f} at {_best_dir[0]}) "
                          f"| max_sequential_lifts={_lift_streak[1]}", flush=True)
                    control.should_training_stop = True
                    _stopped_early[0] = True

        class _PatienceEval(_TC):
            def on_train_begin(self, _args, state, control, model=None, **kw):
                """Anchor on the INPUT model, before any update (task RV2).

                Without this the first eval fires AFTER step 1, so step 1's reading
                becomes the baseline and its weights are never saved. Measured
                2026-09-08 (cov240e): step 1 read 8/16, the run stopped at step 6
                with `EARLY STOP ... (best=0.5000 at None)` — the best number the
                run produced had no checkpoint behind it. Round 1 only: later
                rounds inherit the anchor from the round before.
                """
                if not getattr(args, "eval_at_start", False) or rd != 0 or _lane != "r3":
                    return
                from types import SimpleNamespace as _SN
                _r3_patience_eval(_SN(global_step=0), control, model)

            def on_step_end(self, _args, state, control, model=None, **kw):
                if (
                    _eval_interval is None
                    or state.global_step % _eval_interval != 0
                    or state.global_step == 0
                ):
                    return
                if _lane == "r3":
                    _r3_patience_eval(state, control, model)
                    return
                _tmp = f"{work}/_patience_rd{rd}_step{state.global_step}"
                os.makedirs(_tmp, exist_ok=True)
                try:
                    model.save_pretrained(_tmp)
                    tok.save_pretrained(_tmp)
                    import subprocess as _sp
                    _cmd = [
                        sys.executable,
                        "-m", "rlv2.eval.rl_calibration_eval",
                        "--adapters",
                        _tmp,
                        "--model",
                        args.base,
                        "--tag",
                        f"_mid_{mode}_rd{rd}_s{state.global_step}",
                        "--eval-file",
                        args.eval_file,
                        "--prompt-style",
                        "abstain",
                        "--min-adapter-b",
                        str(args.min_adapter_b),
                    ]
                    _r = _sp.run(_cmd, capture_output=True, text=True, errors="replace",
                                 timeout=600, cwd=_REPO_ROOT)
                    if _r.returncode == 0:
                        import pathlib as _pl
                        _cal = json.loads(_pl.Path("data/rl/calibration.json").read_text(encoding="utf-8"))
                        _t = f"_mid_{mode}_rd{rd}_s{state.global_step}"
                        _m = _cal.get(_t, {}).get("macro_f1", {})
                        _pr = _cal.get(_t, {}).get("per_regime", {})
                        _score = _m.get("macro_f1", 0.0)
                        print(f"  [patience-eval] step={state.global_step} "
                              f"macroF1@{_jsonl_count(args.eval_file)}={_score:.4f}  "
                              f"f1_ans={_m.get('f1_answer', 0):.4f}  "
                              f"f1_abs={_m.get('f1_abstain', 0):.4f}  "
                              f"best={_best_score[0]:.4f}  patience={_patience_counter[0]}/{_PATIENCE}",
                              flush=True)
                        _eval_history.append({
                            "round": rd, "step": int(state.global_step), "lane": "vibe",
                            "macro_f1": _score,
                            "f1_answer": _m.get("f1_answer", 0),
                            "f1_abstain": _m.get("f1_abstain", 0),
                        })
                        for _reg in ["R1", "R2", "R3", "R4"]:
                            _rv = _pr.get(_reg, {})
                            if _rv:
                                print(f"    {_reg}: n={_rv.get('n', '?')} reward={_rv.get('reward', 0):.4f}",
                                      flush=True)
                        # Patience logic
                        if _score > _best_score[0]:
                            _best_score[0] = _score
                            _patience_counter[0] = 0
                            # Save best adapter
                            _best_path = f"{work}/_best_rd{rd}"
                            if os.path.exists(_best_path):
                                shutil.rmtree(_best_path)
                            os.rename(_tmp, _best_path)
                            _best_dir[0] = _best_path
                            _tmp = None  # don't delete renamed dir
                            print(f"  [patience] NEW BEST saved -> {_best_path}", flush=True)
                        else:
                            _patience_counter[0] += 1
                            if _patience_counter[0] >= _PATIENCE:
                                print(f"  [patience] EARLY STOP at step {state.global_step} "
                                      f"(best={_best_score[0]:.4f} at {_best_dir[0]})", flush=True)
                                control.should_training_stop = True
                    else:
                        raise RuntimeError(
                            f"patience eval failed with exit {_r.returncode}: "
                            f"stdout={_r.stdout[-500:]} stderr={_r.stderr[-500:]}"
                        )
                except Exception as _e:
                    raise RuntimeError(
                        f"patience eval failed at round={rd + 1} step={state.global_step}"
                    ) from _e
                finally:
                    if _tmp and os.path.exists(_tmp):
                        shutil.rmtree(_tmp, ignore_errors=True)

        class _UpdateDiagnostics(_TC):
            """Expose fp16 scaler skips and LoRA-B movement without changing optimization."""

            def __init__(self) -> None:
                self.trainer = None
                self.last_scale = None
                self.pre_opt = {}

            def attach(self, trainer) -> None:
                self.trainer = trainer
                scaler = trainer.accelerator.scaler
                self.last_scale = float(scaler.get_scale()) if scaler is not None else None
                print(
                    f"[update-diag] round={rd + 1} initial_scale={self.last_scale} "
                    f"lora_B_tensors={sum(1 for n, _ in trainer.model.named_parameters() if 'lora_B' in n)}",
                    flush=True,
                )

            def on_pre_optimizer_step(self, _args, state, control, model=None, **kw):
                if model is None:
                    raise RuntimeError("update diagnostics require Trainer to provide the active model")
                b_params = [p for n, p in model.named_parameters() if "lora_B" in n]
                live = [p.grad for p in b_params if p.grad is not None]
                self.pre_opt = {
                    "step": state.global_step + 1,
                    "none": len(b_params) - len(live),
                    "present": len(live),
                    "nan": sum(bool(torch.isnan(g).any().item()) for g in live),
                    "inf": sum(bool(torch.isinf(g).any().item()) for g in live),
                    "finite_nonzero": sum(
                        bool(torch.isfinite(g).all().item()) and bool(g.abs().max().item() > 0)
                        for g in live
                    ),
                }

            def on_step_end(self, _args, state, control, model=None, **kw):
                if self.trainer is None:
                    raise RuntimeError("update diagnostics were not attached to the active trainer")
                active_model = model or self.trainer.model
                scaler = self.trainer.accelerator.scaler
                scale = float(scaler.get_scale()) if scaler is not None else None
                backed_off = (
                    scale is not None
                    and self.last_scale is not None
                    and scale < self.last_scale
                )
                skipped = bool(self.trainer.accelerator.optimizer_step_was_skipped)
                b_values = [
                    p.detach().abs().max().item()
                    for n, p in active_model.named_parameters()
                    if "lora_B" in n
                ]
                stats = self.pre_opt or {
                    "none": "?",
                    "present": "?",
                    "nan": "?",
                    "inf": "?",
                    "finite_nonzero": "?",
                }
                print(
                    f"[update-diag] step={state.global_step} scale={scale} "
                    f"backed_off={backed_off} optimizer_skipped={skipped} "
                    f"grad_none={stats['none']} grad_present={stats['present']} "
                    f"grad_nan={stats['nan']} grad_inf={stats['inf']} "
                    f"grad_finite_nonzero={stats['finite_nonzero']} "
                    f"max|B|={max(b_values) if b_values else float('nan'):.8e}",
                    flush=True,
                )
                self.last_scale = scale
                self.pre_opt = {}

        # Half-epoch F1 with per-category support and generated outputs (task 7a). RL
        # printed sample completions but never a per-CATEGORY score, so a stage whose whole
        # purpose is calibration ACROSS question types could not be read per type.
        # HalfEpochEval scores the R3 DOMAIN eval with the 4b judge. OFF by default here:
        # this lane is scored on the 62-row rl_eval_v2, and a second resident judge fights
        # the 2B reward judge for VRAM on a 16 GB card where policy and judge are already
        # choreographed to never co-reside.
        _callbacks = [_PatienceEval()]
        if getattr(args, "evals_per_epoch", 0) > 0:
            from rlv2.eval.half_epoch_eval import HalfEpochEval
            _callbacks.append(HalfEpochEval(policy, tok,
                                            evals_per_epoch=args.evals_per_epoch,
                                            tag="RL lr=%g" % args.lr))
        _update_diagnostics = None
        if args.diagnose_updates:
            _update_diagnostics = _UpdateDiagnostics()
            _callbacks.append(_update_diagnostics)

        tr = GRPOTrainer(
            model=policy,
            reward_funcs=[make_reward(
                meta,
                stats,
                offload_models=_offload_models,
                health_monitor=health_monitor,
                brevity_weight=getattr(args, "brevity_weight", 0.0),
                lane=getattr(args, "lane", "vibe"),
                long2short_lambda=getattr(args, "long2short_lambda", 0.2),
                rollout_ledger=_dyna_ledger,
            )],
            args=cfg, train_dataset=ds, processing_class=tok,
            callbacks=_callbacks,
        )
        if args.fp16_init_scale is not None:
            scaler = tr.accelerator.scaler
            if scaler is None:
                raise RuntimeError("--fp16-init-scale requires an active CUDA fp16 GradScaler")
            if scaler._scale is not None:
                raise RuntimeError("fp16 GradScaler initialized before the requested initial scale was applied")
            scaler._init_scale = float(args.fp16_init_scale)
            print(f"[fp16] initial loss scale set to {scaler._init_scale:g}", flush=True)
        if _update_diagnostics is not None:
            _update_diagnostics.attach(tr)
        _offload_models.clear()
        _offload_models.append(policy)
        if getattr(tr, 'ref_model', None) is not None:
            _offload_models.append(tr.ref_model)
        tr.train()
        # Patience early-stop ends the ARM, not just the round. Two reasons, both measured
        # 2026-08-31: (1) at patience 0 the config has already failed the lift criterion, so
        # a further round spends GPU proving nothing; (2) a mid-round stop leaves cohort
        # prompts with no reward history, and round-2 maxent selection fails its
        # every-prompt-history contract on them (trials 2 and 4 died there).
        # BEST DIRS ARE NOT DELETED UNTIL THE ARM ENDS. Fourth manifestation of the
        # full-param checkpoint bug, measured 2026-09-02 (fpv2_hpo_lr1e-5): _reload_best
        # sets the loaded policy's config._name_or_path to the best dir, and TRL's next
        # GRPOTrainer(...) rebuilds its REFERENCE model from exactly that path string.
        # Deleting the dir right after loading made the round-2 trainer die with
        #     HFValidationError: Repo id must be in the form ... '<...>/_best_rd0'
        # (isdir failed on the freshly-deleted path, so transformers fell through to the
        # hub). Round 1 never hit it because its _name_or_path was args.base, which
        # exists. Cleanup of {work}/_best_rd* happens once, after the final save.
        if _stopped_early[0] and rd + 1 < args.rounds:
            print(f"[{mode}] patience stop ends the arm after round {rd + 1}/{args.rounds}; "
                  f"skipping remaining round(s)", flush=True)
            if _best_dir[0] and os.path.exists(_best_dir[0]):
                tr = None
                policy = None
                _release_round_vram(_offload_models, tag=f"patience stop rd{rd + 1}")
                policy = _reload_best(args, _best_dir[0], _best_score[0], mode)
            else:
                policy = tr.model
            break
        # Use the BEST checkpoint from patience eval if available, else use final
        if _best_dir[0] and os.path.exists(_best_dir[0]):
            # FREE THE OLD TRAINER FIRST. Measured 2026-09-02 (fpv2_hpo_lr1e-5): reloading
            # best while `tr` still held the round's policy put old policy + new policy +
            # TRL's next ref model briefly co-resident -- peak 13.22 GiB vs 9.40 on a run
            # with no reload, failing the x1.3 Memory gate on a run that survived by luck.
            # `tr = None`, NOT `del tr`: downstream `... if tr else []` (round summary)
            # still evaluates the NAME, and del made that an UnboundLocalError (measured,
            # T4 first attempt). Rebinding to None releases the trainer identically.
            # `tr = None` alone was NOT enough (2026-09-08): _offload_models still
            # pinned this round's policy and ref model. See _release_round_vram.
            tr = None
            policy = None
            _release_round_vram(_offload_models, tag=f"round {rd + 1} -> {rd + 2}")
            policy = _reload_best(args, _best_dir[0], _best_score[0], mode)
        else:
            policy = tr.model

        # ── Round summary ────────────────────────────────────────────────────
        batch_regimes = [_regime_of(sel[i]) for i in range(min(args.num_generations, len(sel)))]
        print(f"[{mode}] round {rd+1}/{args.rounds} | optimizer_steps={args.steps} | "
              f"selected_prompts={len(sel)} | first_batch_regimes={batch_regimes} | "
              f"max_prompt_tok={max_prompt_in_round}", flush=True)

        # ── Mid-epoch sample printing (half-epoch visibility) ────────────────
        # Show 3 sample completions from the last block's log so the operator
        # can see what the model generates without waiting for full eval.
        last_logs = tr.state.log_history if tr else []
        z = [l.get("frac_reward_zero_std") for l in last_logs
             if isinstance(l, dict) and "frac_reward_zero_std" in l]
        rm = [l.get("reward") for l in last_logs
              if isinstance(l, dict) and "reward" in l]
        comp_lens = [l.get("completions/mean_length") for l in last_logs
                     if isinstance(l, dict) and "completions/mean_length" in l]
        zm = sum(z) / len(z) if z else float("nan")
        zero_std_log.append(zm)
        if zm > 0.80:
            print(f"[WARN] dead signal: frac_reward_zero_std={zm:.3f} > 0.80 — "
                  f"check class balance and base abstention rate", flush=True)
        avg_comp_len = sum(comp_lens) / len(comp_lens) if comp_lens else float("nan")
        print(f"  frac_zero_std={zm:.3f}  reward={rm[-1] if rm else float('nan'):.3f}  "
              f"comp_len_avg={avg_comp_len:.0f}  "
              f"{(time.time()-t0)/60:.1f} min", flush=True)

        # ── Between-rounds canonical eval (Req 41) ───────────────────────────
        # Save a temporary adapter, score the configured mini eval, print its cardinality-tagged
        # macroF1, then resume. This diagnostic is a kill gate, not the final verdict.
        mid_eval_dir = f"{work}/_mid_eval_rd{rd}"
        os.makedirs(mid_eval_dir, exist_ok=True)
        policy.save_pretrained(mid_eval_dir)
        tok.save_pretrained(mid_eval_dir)
        try:
            import subprocess as _sp
            # FULL-PARAM SAVES A MODEL, NOT AN ADAPTER. `policy.save_pretrained` on a plain
            # (non-PEFT) policy writes no adapter_config.json, so passing the directory as
            # --adapters made the evaluator raise
            #     FileNotFoundError: No adapter_config.json at .../_mid_eval_rd0
            # at the FIRST round boundary -- after round 1 had trained successfully. Measured
            # 2026-09-02 on the full-param lr sweep; both trials exited 1 here.
            # For full-param the saved dir IS the model; --adapters stays empty.
            _fp = bool(getattr(args, "full_param", False))
            _eval_cmd = [
                sys.executable,
                "-m", "rlv2.eval.rl_calibration_eval",
            ] + ([] if _fp else ["--adapters", mid_eval_dir]) + [
                "--model",
                mid_eval_dir if _fp else args.base,
                "--tag",
                f"_mid_{mode}_rd{rd}",
                "--eval-file",
                args.mini_eval_file,
                "--prompt-style",
                "abstain",
                "--min-adapter-b",
                str(args.min_adapter_b),
            ]
            _eval_result = _sp.run(
                _eval_cmd, capture_output=True, text=True, errors="replace",
                timeout=600, cwd=_REPO_ROOT,
            )
            if _eval_result.returncode == 0:
                import pathlib as _pathlib
                _cal = json.loads(_pathlib.Path("data/rl/calibration.json").read_text(encoding="utf-8"))
                _mid_tag = f"_mid_{mode}_rd{rd}"
                _mid_metrics = _cal.get(_mid_tag, {}).get("macro_f1", {})
                _mid_f1 = _mid_metrics.get("macro_f1", "?")
                _mid_ans = _mid_metrics.get("f1_answer", "?")
                _mid_abs = _mid_metrics.get("f1_abstain", "?")
                print(
                    f"  [mid-eval] round {rd+1} "
                    f"macroF1@{_jsonl_count(args.mini_eval_file)}={_mid_f1:.4f}  "
                    f"f1_answer={_mid_ans:.4f}  f1_abstain={_mid_abs:.4f}",
                    flush=True,
                )
            else:
                raise RuntimeError(
                    f"mid-eval failed with exit {_eval_result.returncode}: "
                    f"stdout={_eval_result.stdout[-500:]} stderr={_eval_result.stderr[-500:]}"
                )
        except Exception as _eval_err:
            raise RuntimeError(f"mid-eval failed after round {rd + 1}") from _eval_err
        finally:
            shutil.rmtree(mid_eval_dir, ignore_errors=True)
    # Machine-readable summary for the RL HPO driver: the criterion is max SEQUENTIAL lifts
    # on the ladder metric (eval #1 is the baseline and never counts as a lift).
    if args.eval_every and getattr(args, "lane", "vibe") == "r3":
        print(f"[patience-summary] arm={mode} max_sequential_lifts={_lift_streak[1]} "
              f"best={_best_score[0]:.4f} evals={_n_evals[0]}", flush=True)
    # TRUE PEAK, from the allocator, at the END of the arm. An `nvidia-smi` poll partway
    # through a run reads whatever the workload happened to be holding at that instant and is
    # NOT the peak -- reading one as a peak is what produced a false "full-param fits"
    # retraction on 2026-09-02. §5's Memory gate (free >= peak x 1.3) needs this number.
    if torch.cuda.is_available():
        _peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        _resv = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(f"[vram-peak] arm={mode} allocated={_peak:.2f} GiB reserved={_resv:.2f} GiB "
              f"(torch.cuda.max_memory_allocated; NOT an nvidia-smi sample)", flush=True)
    os.makedirs(work, exist_ok=True)
    policy.save_pretrained(work)
    tok.save_pretrained(work)
    # W0 (playbook.md "WORLD MODEL"): persist per-prompt win/loss into the visit
    # ledger at arm end. Training-prompt judge outcomes previously DIED with each
    # run's logs, which is why the world model (E10) had no labels — every run now
    # accrues them for free. Backward compatible: old ledgers held bare ints
    # (visit counts); entries upgrade to {w, l, visits} on first touch.
    _ledger_path = getattr(args, "visit_ledger", None)
    if _ledger_path:
        import hashlib as _hl
        import json as _j
        from rlv2.judge.reward import CORRECT_FLOOR as _lfloor
        _led = {}
        if os.path.exists(_ledger_path):
            _led = _j.load(open(_ledger_path, encoding="utf-8"))
        for _pt, _rewards in stats.items():
            _k = _hl.sha1(_pt.encode("utf-8")).hexdigest()[:16]
            _e = _led.get(_k)
            if not isinstance(_e, dict):
                _e = {"w": 0, "l": 0, "visits": int(_e) if _e else 0}
            _w = sum(1 for _r in _rewards if _r >= _lfloor)
            _e["w"] += _w
            _e["l"] += len(_rewards) - _w
            _e["visits"] = _e.get("visits", 0) + 1
            _e["prompt"] = meta[_pt].get("question") or meta[_pt].get("prompt", "")
            _led[_k] = _e
        os.makedirs(os.path.dirname(_ledger_path) or ".", exist_ok=True)
        _j.dump(_led, open(_ledger_path, "w", encoding="utf-8"))
        print(f"[ledger] {len(stats)} prompts -> {_ledger_path} "
              f"({len(_led)} total entries)", flush=True)
    # Deferred cleanup of best-checkpoint dirs. They MUST outlive the round loop: after
    # _reload_best, the policy's config._name_or_path is the best dir, and TRL rebuilds the
    # next round's REFERENCE model from that path -- deleting it mid-arm crashed round-2
    # trainer construction with HFValidationError (measured 2026-09-02). Safe here: the
    # final policy is saved to `work` above and nothing else will dereference the path.
    import glob as _glob
    for _bd in _glob.glob(os.path.join(work, "_best_rd*")):
        shutil.rmtree(_bd, ignore_errors=True)
    # HARD GATE, added 2026-08-02. LoRA's B inits to EXACTLY zero, so max|B| is an unfakeable measure
    # of whether this run trained at all. The 2x12 @ 1e-6 arms saved cleanly with max|B| = 8e-6 --
    # an effective |dW| of 1e-6, which fp16 cannot represent -- and were then evaluated for weeks as
    # though they were policies. Fail here rather than ship a no-op that reads as a clean null.
    if not args.full_param:
        from rlv2.common.cost_meter import assert_adapter_trained
        assert_adapter_trained(
            work,
            min_abs_b=args.min_adapter_b,
            label=f"mgpo/{mode}",
        )
    # The receipt records final selection probabilities. An early-stopped arm has judged
    # only part of the cohort; computing over unjudged prompts violates the
    # every-prompt-history contract (trial 5, 2026-08-31), so restrict to what was seen.
    _judged = [r for r in cohort if r["ptext"] in stats]
    if len(_judged) < len(cohort):
        print(f"[{mode}] receipt probabilities over {len(_judged)}/{len(cohort)} judged "
              f"prompts (early stop left the rest unjudged)", flush=True)
    try:
        probabilities = (
            continuation_probabilities(_judged, stats, mode=mode,
                                       boxcox_lambda=args.boxcox_lambda)
            if _judged else {}
        )
    except RuntimeError as _span_err:
        # Nothing threshold-spanning: fatal for SELECTION (training uniform would be a lie),
        # but this is the post-run receipt -- record the fact instead of dying on it.
        print(f"[{mode}] receipt probabilities unavailable: {_span_err}", flush=True)
        probabilities = {}
    reward_history = {prompt: list(values) for prompt, values in stats.items()}
    json.dump(
        {
            "mode": mode,
            "zero_std_by_round": zero_std_log,
            "prompts_seen": len(stats),
            "wall_min": (time.time() - t0) / 60,
            "cohort": [record["ptext"] for record in cohort],
            "round_selections": round_selections,
            "continuation_probabilities": probabilities,
            "reward_history": reward_history,
            "eval_history": _eval_history,
        },
        io.open(f"{work}/train_stats.json", "w", encoding="utf-8"),
    )
    # Written LAST, after the adapter is safely on disk, so a run killed mid-save leaves an untagged
    # -- and therefore rejected -- directory rather than a tagged partial one.
    json.dump({"cfg": tag, "mode": mode}, io.open(stamp, "w", encoding="utf-8"))
    del policy
    torch.cuda.empty_cache()
    return work


def main():
    ap = argparse.ArgumentParser(description="Train MGPO; score checkpoints externally with src/rlv2/eval/rl_calibration_eval.py")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--mix", default=MIX)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--steps", type=int, default=STEPS_PER_ROUND)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--arms", nargs="+", default=["uniform", "maxent"])
    ap.add_argument("--rank", type=int, default=LORA_R,
                    help="LoRA rank r (default: 16)")
    ap.add_argument("--alpha", type=int, default=LORA_ALPHA,
                    help="LoRA alpha (default: 64, convention alpha=4r for this stage)")
    ap.add_argument("--target-modules", default="all-linear",
                    help="LoRA target modules: 'all-linear' or comma-separated")
    ap.add_argument("--full-param", action="store_true",
                    help="train FFN weights directly instead of a LoRA adapter")
    ap.add_argument("--max-grad-norm", type=float, default=1.0, dest="max_grad_norm",
                    help="gradient clipping norm. NOT cosmetic on the full-param path: "
                         "measured grad norms run 20-45 there, so at the 1.0 default ~95%% of "
                         "the gradient MAGNITUDE is discarded and the step size is set by the "
                         "clipper rather than by lr")
    ap.add_argument("--freeze-embeddings", action="store_true",
                    help="full-param only: hold the tied embed_tokens/lm_head out of "
                         "training. 254M of 752M params on this model (vocab 248320), so it "
                         "saves ~1.5 GB of grad+optimizer state and removes the most common "
                         "fp16 NaN source")
    ap.add_argument("--allow-full-param-oom", action="store_true",
                    help="override the full-param refusal. Measured to OOM at step 3 of 6 on "
                         "this 16 GB card even with 8-bit Adam; intended for bigger hardware")
    ap.add_argument("--optim", default="adamw_torch",
                    choices=["adamw_torch", "adamw_bnb_8bit"],
                    help="optimizer. adamw_bnb_8bit quantises Adam m/v state and is what "
                         "makes --full-param fit on this 16 GB rig (~6.0 GB -> ~1.5 GB of "
                         "optimizer state on a 752M model)")
    ap.add_argument("--out", default=None,
                    help="override output directory (default: mgpo/)")
    ap.add_argument("--prompt-style", choices=["abstain", "cb"], default="abstain",
                    help="rollout prompt style. MUST MATCH the style the eval scores with "
                         "(rl_calibration_eval --prompt-style). `cb` forbids declining and "
                         "collapses every completion to class=answer, which trips the "
                         "class-collapse kill gate and makes calibration untrainable")
    ap.add_argument("--lane", choices=["r3", "vibe"], default="r3",
                    help="judge lane. r3 = 4b + synthetic domain rubric, binary-gated "
                         "Long2Short reward, patience on the frozen 16-item eval (the "
                         "pare-down default). vibe = legacy 2b + VibeThinker rubric for "
                         "the future R1/R2/R4 abstention direction.")
    ap.add_argument("--long2short-lambda", type=float, default=0.2, dest="long2short_lambda",
                    help="max zero-sum brevity shift within correct completions (r3 lane)")
    ap.add_argument("--brevity-weight", type=float, default=0.5,
                    help="how strongly shorter CORRECT answers are preferred. 0 = raw "
                         "judge score, the configuration the recorded 0.452 was measured "
                         "with; 1 = full reward_judge shape (score x brevity). Default "
                         "0.5. NOTE: any non-zero value changes the RL objective, so such "
                         "a run is an IMPROVEMENT on the recorded lane, not a replication "
                         "of it -- report it as such")
    ap.add_argument("--evals-per-epoch", type=int, default=0,
                    help="R3 half-epoch readings (4b judge). 0 = OFF, the default -- this "
                         "lane scores on rl_eval_v2 and the reward judge already owns the "
                         "VRAM budget")
    ap.add_argument("--seed", type=int, default=7,
                    help="selection and training seed (default: 7)")
    ap.add_argument("--max-prompt", type=int, default=MAX_PROMPT,
                    help="maximum prompt tokens; records above the cap are rejected")
    ap.add_argument("--max-completion", type=int, default=MAX_COMPLETION,
                    help="maximum completion tokens")
    ap.add_argument("--target-pool", type=int, default=TARGET_POOL,
                    help="distinct bootstrap prompts and round-two draw count")
    ap.add_argument("--num-generations", type=int, default=NUM_GEN,
                    help="on-policy completions per prompt (G)")
    ap.add_argument("--top-p", type=float, default=TOP_P,
                    help="nucleus-sampling probability")
    ap.add_argument("--top-k", type=int, default=TOP_K,
                    help="top-k sampling cutoff; -1 disables filtering")
    ap.add_argument("--boxcox-lambda", type=float, default=BOXCOX_LAMBDA,
                    help="lambda for hierarchical and MaxEnt Box-Cox allocation")
    ap.add_argument("--beta", type=float, default=0.10,
                    help="KL penalty strength (0.10 is the proven stable path)")
    # PPO-style clip range on the policy ratio (Raschka Ch7 S7.4; Ch7 S7.6.3 item 5).
    # Was silently inherited from TRL's GRPOConfig default (epsilon=0.2) -- never once
    # exposed or swept across this project's full-param RL campaigns. DeepSeek-R1's
    # published reference uses clip_eps=10.0 (effectively unclipped, per Ch7 S7.5.2's
    # discussion of that value), against our beta=0.3 which is 15x their kl_coeff=0.02 --
    # both axes are candidates for the DeepSeek-perturbation HPO sweep, not just beta.
    ap.add_argument("--epsilon", type=float, default=0.2,
                    help="GRPO clip range (TRL default 0.2 = standard PPO clip; "
                         "DeepSeek-R1 reference uses 10.0, effectively unclipped)")
    ap.add_argument("--epsilon-high", type=float, default=None, dest="epsilon_high",
                    help="asymmetric upper clip bound (TRL default: mirrors --epsilon)")
    # E1+E2 epistemic/novelty cohort selection (docs/plans/uncertainty-menu-hpo.md).
    ap.add_argument("--probe-base", default=None, dest="probe_base",
                    help="second checkpoint for two-model disagreement cohort selection; "
                         "None = incumbent hierarchical sampling only")
    ap.add_argument("--dynamic-sampling", action="store_true", dest="dynamic_sampling",
                    help="DAPO dynamic sampling: screen candidates with --probe-samples "
                         "generations each and keep only prompts whose OWN samples come out "
                         "mixed (some above the floor, some below). A uniform prompt has zero "
                         "group variance and contributes no gradient; 35-50%% of prompts were "
                         "dead this way in measured runs. Single model -- no --probe-base "
                         "checkpoint is loaded. Implies the probe-selection path.")
    ap.add_argument("--probe-samples", type=int, default=2, dest="probe_samples",
                    help="completions per candidate per model in the probe screen")
    ap.add_argument("--probe-candidates-mult", type=int, default=4,
                    dest="probe_candidates_mult",
                    help="candidate superset size as a multiple of --target-pool")
    ap.add_argument("--visit-ledger", default=None, dest="visit_ledger",
                    help="JSON path persisting per-prompt visit counts across passes "
                         "(UCT-style novelty bonus); None disables the counter")
    # E8: PER-style round-2 curriculum (playbook-uncertainty-menu.md; Schaul 2015).
    # RL stage only by convention -- SFT keeps Box-Cox class-balanced sampling.
    ap.add_argument("--per-sampling", action="store_true", dest="per_sampling",
                    help="round 2 uses PER soft priorities (Beta-posterior variance, "
                         "epsilon-floor, fresh injection) instead of the hard spanning gate")
    ap.add_argument("--per-alpha", type=float, default=0.6, dest="per_alpha",
                    help="PER prioritization exponent (Schaul's proportional-variant value)")
    ap.add_argument("--per-eps", type=float, default=1e-3, dest="per_eps",
                    help="PER epsilon-floor: no prompt is ever sampling-ineligible")
    # Coverage-first curriculum (operator, 2026-09-07): visit count dominates,
    # uncertainty breaks ties, one visit per record until the pool is covered.
    ap.add_argument("--grad-accum", type=int, default=1, dest="grad_accum",
                    help="prompts consumed per optimizer step (gradient accumulation). "
                         "1 = the historical shape (one step, one record). 12 with "
                         "--steps 10 --rounds 2 walks 240 records = full coverage")
    ap.add_argument("--eval-at-start", action="store_true", dest="eval_at_start",
                    help="evaluate the INPUT model once before training (round 1 only) so "
                         "the patience baseline is the model you started from. Without it "
                         "the first eval fires after step 1, and step 1's reading is spent "
                         "as the anchor instead of being saved as a checkpoint.")
    ap.add_argument("--coverage-first", action="store_true", dest="coverage_first",
                    help="every round draws a FRESH least-visited cohort (no repeats "
                         "within a run while unvisited records remain); uses "
                         "--visit-ledger for cross-run visit counts. Overrides "
                         "--per-sampling/--probe-base selection")
    # W2: ensemble world model informs UNSEEN-prompt priors (selection-only).
    ap.add_argument("--world-model", default=None, dest="world_model",
                    help="prompt_world_model .npz; informs fresh-injection priors in "
                         "--per-sampling mode (never overrides measured posteriors)")
    ap.add_argument("--wm-trust", type=float, default=0.02, dest="wm_trust",
                    help="max ensemble disagreement (variance) at which the model's "
                         "prior is trusted; above it, flat Beta(1,1) — part 6's "
                         "uncertainty-masking rule")
    ap.add_argument("--eval-file", default=DEFAULT_EVAL_FILE,
                    help="canonical eval partition used by optional patience eval")
    ap.add_argument("--mini-eval-file", default=DEFAULT_MINI_EVAL_FILE,
                    help="between-round diagnostic eval partition")
    ap.add_argument("--partition-manifest", default=DEFAULT_PARTITION_MANIFEST,
                    help="manifest containing the disjoint rubric partition")
    ap.add_argument("--min-adapter-b", type=float, default=DEFAULT_MIN_ADAPTER_B,
                    help="minimum max absolute LoRA-B value required before scoring")
    ap.add_argument("--regimes", nargs="*", default=None,
                    help="restrict training pool to these regimes (e.g. --regimes R1 R3)")
    ap.add_argument("--domain", default=None,
                    help="restrict training pool to one domain class for Parallel-RL "
                         "(e.g. --domain Math). Cross-hatch still operates on answer_type within.")
    ap.add_argument(
        "--loss-type",
        choices=["grpo", "dr_grpo", "dapo", "bnpo"],
        default="dapo",
        help="TRL GRPO loss normalization; dapo is the installed-version default",
    )
    ap.add_argument(
        "--dyna-ipo", action="store_true",
        help="Task #44 Dyna-MGPO: keep each round's judged rollouts in a replay "
             "ledger and run a mini offline IPO step at every round boundary "
             "(act -> store real judged experience -> update offline)")
    ap.add_argument("--dyna-min-pairs", type=int, default=2,
                    help="skip the round-boundary IPO step below this many pairs")
    ap.add_argument(
        "--scale-rewards",
        choices=["group", "batch", "none"],
        default="group",
        help="TRL advantage scaling (task #38, BREAK THE 9 R1a): 'group' divides by "
             "the group std (TRL default, every prior run), 'none' is Dr. GRPO's "
             "recommendation (no difficulty re-weighting), 'batch' is PPO-Lite",
    )
    ap.add_argument("--temperature", type=float, default=0.65,
                    help="rollout generation temperature (default: 0.65, Zhu et al. exploration-"
                         "interference sweet spot; VibeThinker paper used 1.0)")
    ap.add_argument("--lr-scheduler", default="linear", dest="lr_scheduler",
                    choices=["linear", "constant", "cosine", "constant_with_warmup"],
                    help="lr schedule; 'constant' decouples effective lr from steps-per-round "
                         "(the decay horizon otherwise moves with the round length)")
    ap.add_argument("--patience", type=int, default=2,
                    help="consecutive non-improving patience evals before early stop; "
                         "0 = stop on the first (RL HPO criterion: 3 sequential lifts)")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="canonical patience-eval interval; 0 disables expensive mid-round eval")
    ap.add_argument("--observe-every", type=int, default=OBSERVE_EVERY,
                    help="steps between three-regime generation/reward observations")
    ap.add_argument(
        "--diagnose-updates",
        action="store_true",
        help="print per-step fp16 scaler, LoRA-B gradient, skip, and movement diagnostics",
    )
    ap.add_argument(
        "--fp16-init-scale",
        type=float,
        default=None,
        help="override CUDA GradScaler initial loss scale; omit to preserve the PyTorch default",
    )
    ap.add_argument(
        "--class-collapse-patience",
        type=int,
        default=CLASS_COLLAPSE_PATIENCE,
        help="consecutive one-class steps before kill gate fires (default: 10; set high to disable)",
    )
    ap.add_argument("--sft-adapter", default=None,
                    help="path to a FROZEN SFT LoRA adapter to stack under the trainable RL adapter. "
                         "Loads base → SFT adapter (frozen) → new RL adapter (trainable). "
                         "No merge step needed — both adapters active during forward pass.")
    args = ap.parse_args()

    R.set_judge_lane(args.lane)
    if args.lane == "r3":
        # The R3 lane is closed-book: no abstention class exists, so the abstain prompt
        # style and the 62-row abstention eval are the WRONG instruments there. Swap the
        # defaults loudly; explicit non-default values are respected.
        if args.prompt_style == "abstain":
            print("[lane:r3] prompt-style abstain -> cb (R3 has no abstention class)", flush=True)
            args.prompt_style = "cb"
        if args.eval_file == DEFAULT_EVAL_FILE:
            args.eval_file = "data/rl/r3_eval_canonical.jsonl"
            print("[lane:r3] eval-file -> %s (frozen 16-item R3 eval)" % args.eval_file,
                  flush=True)

    if args.steps <= 0:
        raise ValueError(f"--steps must be positive, got {args.steps}")
    if args.observe_every <= 0:
        raise ValueError(f"--observe-every must be positive, got {args.observe_every}")
    if args.target_pool <= 0:
        raise ValueError(f"--target-pool must be positive, got {args.target_pool}")
    if args.num_generations < 2:
        raise ValueError(f"--num-generations must be at least 2, got {args.num_generations}")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError(f"--top-p must be in (0, 1], got {args.top_p}")
    if args.max_prompt <= 0 or args.max_completion <= 0:
        raise ValueError("--max-prompt and --max-completion must be positive")
    if args.rank <= 0 or args.alpha <= 0 or args.min_adapter_b <= 0:
        raise ValueError("--rank, --alpha, and --min-adapter-b must be positive")
    if args.fp16_init_scale is not None and args.fp16_init_scale <= 0:
        raise ValueError("--fp16-init-scale must be positive when provided")
    if args.rounds != 2:
        raise ValueError(
            f"valid online MGPO requires exactly 2 rounds (bootstrap + continuation), got {args.rounds}"
        )
    if args.full_param and not args.allow_full_param_oom:
        # RE-CONFIRMED BY MEASUREMENT 2026-09-02, after an earlier version of this guard
        # wrongly suggested 8-bit Adam was the way through. It is not. The full lever set --
        # --optim adamw_bnb_8bit (Adam m/v 6.0 -> 1.5 GB), fp32 master weights, gradient
        # checkpointing -- was applied and the run STILL dies:
        #
        #   step 1  grad_norm 4.383
        #   step 2  grad_norm nan
        #   step 3  grad_norm nan -> torch.OutOfMemoryError: tried to allocate 1.89 GiB,
        #                            0 bytes free, 13.06 GiB allocated (pool=6, G=4)
        #
        # Two independent failures: it exhausts VRAM mid-run AND produces NaN gradients
        # before it gets there. The static arithmetic (~7.5 GB with 8-bit Adam) looks
        # survivable and is not, because `expandable_segments` is UNSUPPORTED ON WINDOWS --
        # reserved-but-unallocated memory cannot be reclaimed, so paper headroom is not
        # reachable headroom. DeepSpeed is not an escape either: ZeRO 1/2 shard across
        # devices and this is a single GPU; the Windows CUDA-op build is broken.
        #
        # The escape hatch exists so a bigger card can run this without editing source, and
        # is named for what it costs rather than for what it enables.
        raise ValueError(
            "full-parameter RL does NOT fit on this 16 GB rig. Re-confirmed 2026-09-02 WITH "
            "8-bit Adam + fp32 master + gradient checkpointing: OOM at step 3 of 6 (13.06 GiB "
            "allocated, 0 free) plus grad_norm=nan on 2 of the first 3 steps. Neither "
            "adamw_bnb_8bit nor DeepSpeed changes this (expandable_segments is unsupported on "
            "Windows; ZeRO 1/2 needs multiple GPUs). Pass --allow-full-param-oom only on "
            "hardware with more VRAM.")
    invalid_modes = sorted(set(args.arms) - {"uniform", "maxent"})
    if invalid_modes:
        raise ValueError(f"unknown --arms values: {invalid_modes}")

    mix = _load(args.mix)
    if not mix:
        raise ValueError(f"MGPO mix is empty: {args.mix}")
    # NORMALISE FIELD NAMES ONCE, HERE. The R3 builder writes `question`/`category`; this
    # trainer and everything it calls read `prompt`/`domain`. Without this every R3 row
    # collapsed to ONE distinct prompt (`.get("prompt", "")` -> "") and one "unknown" domain
    # cell -- the same alias already applied in sprawl/manifold_sampler for the same reason.
    for _r in mix:
        if not _r.get("prompt") and _r.get("question"):
            _r["prompt"] = _r["question"]
        if not _r.get("domain") and _r.get("category"):
            _r["domain"] = _r["category"]
    assert_partition_disjoint(
        mix,
        [args.eval_file, args.mini_eval_file],
        args.partition_manifest,
    )
    distinct_prompts = len({record.get("prompt", "") for record in mix})
    if args.target_pool > distinct_prompts:
        raise ValueError(
            f"--target-pool {args.target_pool} exceeds {distinct_prompts} distinct prompts"
        )

    global OUT_DIR
    if args.out:
        OUT_DIR = args.out
    os.makedirs(OUT_DIR, exist_ok=True)

    update_scale = args.lr * args.alpha / args.rank
    print(
        f"mix={len(mix)} rows ({distinct_prompts} distinct prompts) | "
        f"target_pool={args.target_pool} | {args.rounds} rounds x {args.steps} steps | "
        f"G={args.num_generations} top_p={args.top_p} top_k={args.top_k} "
        f"boxcox_lambda={args.boxcox_lambda} | r={args.rank} alpha={args.alpha} "
        f"lr={args.lr:g} update_scale={update_scale:g} | "
        f"reward={R._judge_model()} isolated judge (lane={args.lane})",
        flush=True,
    )
    for mode in args.arms:
        checkpoint = run_arm(mode, mix, args, prompt_style=args.prompt_style)
        stats = json.load(io.open(f"{checkpoint}/train_stats.json", encoding="utf-8"))
        print(
            f"[{mode}] TRAINED | checkpoint={checkpoint} | "
            f"prompts_seen={stats['prompts_seen']} | "
            f"zero_std={stats['zero_std_by_round']}",
            flush=True,
        )

    print(
        "Training complete. Score each checkpoint with src/rlv2/eval/rl_calibration_eval.py; "
        "src/rlv2/train/rl/grpo.py has no alternate evaluation instrument.",
        flush=True,
    )


if __name__ == "__main__":
    main()
