"""One metric across CPT / SFT / RL — the DECLARED one, enforced.

Spec: playbook.md "The three projections — ALWAYS report all three, at EVERY stage" +
the S-9 recommendation (binary gate + Long2Short) + S-12 (the RL reward judge).
Task: plan ancient-wondering-elephant.md, 2026-08-31.

The failure this file guards is not a crash — it is three stages quietly measuring three
different quantities and the lift table comparing them anyway. Measured consequence: the raw
judge-score mean cannot even SEE the SFT lift the binary ladder proved (base 0.5608 vs SFT-ep3
0.5578), because the rubric deliberately parks plausible-wrong at 0.4–0.6. A previous version
of this work invented a fourth metric ("gated mean") to fix that; it was retracted — the
ladder is the binary at the one floor, declared before any run.
"""
from __future__ import annotations
import os as _o, sys as _s

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ── one floor, one home ────────────────────────────────────────────────────────────────────

def test_the_floor_has_exactly_one_home():
    """Every 0.70-style correctness floor in src/ must be judge.CORRECT_FLOOR or an import
    of it. Two constants that must agree, enforced nowhere, is how the 0.5 threshold in
    rl_calibration_eval drifted from the 0.70 line without anyone noticing."""
    offenders = []
    # Different-semantic constants, each carrying a comment saying it is NOT the judge floor:
    allow = {("reward.py", "VERIFY_FLOOR"), ("config.py", "HIGH_REWARD_THRESHOLD")}
    for f in (ROOT / "src").glob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for t in node.targets:
                name = getattr(t, "id", "")
                if not name or "FLOOR" not in name.upper():
                    continue
                if isinstance(node.value, ast.Constant) and node.value.value in (0.7, 0.70):
                    if f.name == "judge.py" and name == "CORRECT_FLOOR":
                        continue                      # the one home
                    if (f.name, name) in allow:
                        continue
                    offenders.append("%s:%s" % (f.name, name))
    assert not offenders, "floor re-declared outside judge.py: %s" % offenders


def test_score_arms_and_categories_use_the_judge_floor():
    from rlv2.judge import judge as J
    from rlv2.eval import score_arms as SA
    from rlv2.eval import score_r3_categories as SC
    assert SA.FLOOR == J.CORRECT_FLOOR
    assert SC.TAU == J.CORRECT_FLOOR, (
        "the legacy fallback disagrees with the live floor; judge.combine maps incorrect "
        "verdicts strictly below CORRECT_FLOOR, so anything else re-opens the fluent-wrong leak")


def test_reward_reexports_the_floor():
    from rlv2.judge import judge as J
    from rlv2.judge import reward as R
    assert R.CORRECT_FLOOR == J.CORRECT_FLOOR


# ── the judge lane (S-12) ──────────────────────────────────────────────────────────────────

def test_r3_lane_selects_the_4b_and_the_synthetic_rubric():
    """The 2b accepted 3/12 fluent-wrong decoys the 4b and 9b both rejected. RL optimises
    AGAINST its reward judge across thousands of rollouts, so on the R3 lane the 2b would
    teach the policy that confident-sounding wrong answers pay."""
    from rlv2.judge import judge as J
    from rlv2.judge import reward as R
    R.set_judge_lane("r3")
    try:
        assert R._judge_model() == J.JUDGE_MODEL, "r3 lane not on judge.py's 4b"
        rub = R._rubric()
        assert "closed-book" in rub or "0.7 is the line" in rub, "not the synthetic domain rubric"
        assert "Return:" not in rub, (
            "the rubric's own output-format line must be stripped -- each judge call appends "
            "its own, and two competing JSON shapes in one prompt is a measured failure mode")
        assert "<!--" not in rub, "HTML provenance comments are context cost, not instruction"
    finally:
        R.set_judge_lane("vibe")


def test_vibe_lane_is_untouched():
    """Abstention machinery is KEPT (owner: R1/R2/R4 is a future direction, not a cut)."""
    from rlv2.judge import reward as R
    R.set_judge_lane("vibe")
    assert R._judge_model() == "qwen3.5-oc:2b"
    assert R._rubric() is R._VIBETHINKER_RUBRIC


def test_unknown_lane_raises():
    from rlv2.judge import reward as R
    with pytest.raises(ValueError):
        R.set_judge_lane("r2")


# ── the r3 reward shape (S-9 recommendation: binary gate + Long2Short) ─────────────────────

def _shape(scores, lengths, floor=0.70, lam=0.2):
    """Reference implementation of the reward shaping in mgpo.make_reward, one prompt group."""
    correct = [i for i, s in enumerate(scores) if s >= floor]
    shaped = [0.0] * len(scores)
    for i in correct:
        shaped[i] = 1.0
    if len(correct) >= 2:
        inv = [1.0 / max(1, lengths[i]) for i in correct]
        mu = sum(inv) / len(inv)
        dev = [x - mu for x in inv]
        md = max(abs(d) for d in dev)
        if md > 0:
            for i, d in zip(correct, dev):
                shaped[i] = 1.0 + lam * d / md
    return shaped


def test_wrong_answers_reward_exactly_zero():
    """The raw-score reward paid ~0.5 for confident wrongness; the gate pays 0.0, and no
    brevity term may promote a wrong answer."""
    shaped = _shape([0.55, 0.45, 0.68], [10, 500, 40])
    assert shaped == [0.0, 0.0, 0.0]


def test_long2short_is_zero_sum_within_the_correct_set():
    """The recorded reason Long2Short beat KL-as-objective: shifts cancel, so the GRPO group
    baseline (which advantage divides by) does not move."""
    scores = [0.95, 0.80, 0.72, 0.40]
    shaped = _shape(scores, [50, 200, 400, 30])
    n_correct = 3
    assert abs(sum(shaped) - n_correct) < 1e-9, "brevity redistribution moved the baseline"
    # shorter correct answer outranks longer correct answer
    assert shaped[0] > shaped[1] > shaped[2]
    assert shaped[3] == 0.0


def test_single_correct_answer_gets_exactly_one():
    """With nothing to redistribute against, the gate alone decides."""
    assert _shape([0.9, 0.3], [100, 100]) == [1.0, 0.0]


def test_mgpo_source_implements_this_shape():
    """The reference above must be the shape mgpo actually ships, not a lookalike."""
    src = (ROOT / "src" / "rlv2" / "train" / "rl" / "grpo.py").read_text(encoding="utf-8")
    assert 'if lane == "r3":' in src
    assert "long2short_lambda * _d / _md" in src, "the centred, lambda-scaled shift is gone"
    assert "raw_scores = list(scores)" in src, (
        "stats must record the RAW judge score -- the spanning filter compares history "
        "against the floor, and the shaped reward is 0/1-ish by design")


# ── threshold-spanning retention (playbook-replaybuffer-scores.md sec 7) ───────────────────

def _probs(history_by_prompt):
    from rlv2.train.rl import grpo as mgpo
    cohort = [{"ptext": k} for k in history_by_prompt]
    return mgpo.continuation_probabilities(cohort, history_by_prompt, "maxent")


def test_both_sides_of_the_floor_is_the_retention_rule():
    probs = _probs({
        "spans": [0.68, 0.72, 0.95, 0.40],     # outcomes on both sides -> retained
        "all_wrong": [0.65, 0.68, 0.45, 0.55], # two buckets, ZERO binary variance -> dropped
        "solved": [0.90, 0.95, 0.92, 0.88],    # clearly solved -> dropped
    })
    assert probs["spans"] == pytest.approx(1.0)
    assert probs["all_wrong"] == 0.0, (
        "bucket-diversity retained this; it is judge-call waste -- both outcomes are wrong")
    assert probs["solved"] == 0.0


def test_nothing_spanning_raises_rather_than_training_uniform():
    with pytest.raises(RuntimeError):
        _probs({"a": [0.1, 0.2], "b": [0.9, 0.95]})


# ── patience resolution ────────────────────────────────────────────────────────────────────

def test_resolution_floor_is_the_measured_constant():
    from rlv2.eval import score_arms as SA
    assert SA.RESOLUTION_FLOOR == pytest.approx(0.0526), (
        "this is a MEASURED property of the instrument (identical checkpoint as 4 arms, "
        "max spread 0.0526); if the instrument changed, re-measure, do not edit")


def test_patience_ignores_sub_resolution_improvements():
    """The comparison the r3 patience path makes, extracted: an improvement the instrument
    cannot resolve must not reset the counter."""
    from rlv2.eval import score_arms as SA
    best, counter = 0.30, 0
    for rate in (0.32, 0.33):                  # +0.02, +0.03 -- both inside the noise
        if rate > best + SA.RESOLUTION_FLOOR:
            best, counter = rate, 0
        else:
            counter += 1
    assert counter == 2, "sub-resolution noise reset patience"
    assert best == 0.30
    assert 0.36 > best + SA.RESOLUTION_FLOOR, "a real improvement must still reset it"


def test_mgpo_patience_uses_the_ladder_not_macro_f1_on_r3():
    src = (ROOT / "src" / "rlv2" / "train" / "rl" / "grpo.py").read_text(encoding="utf-8")
    assert "_r3_patience_eval" in src
    assert "RESOLUTION_FLOOR as _min_delta" in src
    i = src.index("def _r3_patience_eval")
    body = src[i:i + 3000]
    assert "macro_f1" not in body, "the abstention instrument leaked into the r3 patience path"


def _lifts(rates, min_delta=0.0526, patience=0):
    """Reference implementation of the patience/lift accounting in mgpo's r3 patience eval:
    eval #1 seeds the baseline (never a lift), a lift needs > best + min_delta, a miss
    resets the streak, and `patience` misses stop the run."""
    best, streak, mx, misses = None, 0, 0, 0
    for r in rates:
        if best is None:
            best = r
            continue
        if r > best + min_delta:
            best, streak, misses = r, streak + 1, 0
            mx = max(mx, streak)
        else:
            streak = 0
            misses += 1
            if misses > patience or (patience == 0 and misses >= 1):
                break
    return mx


def test_three_sequential_lifts_is_countable_and_baseline_is_not_a_lift():
    """The HPO pass criterion (owner, 2026-08-31): 3 sequential lifts at patience 0.
    The first eval is the input model's own rate -- counting it as a lift would hand every
    config one free lift and once handed the scorer gate an untrained 'best' adapter."""
    assert _lifts([0.30, 0.36, 0.42, 0.48]) == 3          # baseline + 3 real lifts
    assert _lifts([0.30, 0.36, 0.42, 0.42]) == 2          # plateau breaks the streak
    assert _lifts([0.30, 0.33, 0.39, 0.45]) == 0          # +0.03 is sub-resolution: stop
    assert _lifts([0.30]) == 0                            # baseline alone proves nothing


def test_patience_stop_ends_the_arm_not_just_the_round():
    """Measured 2026-08-31 (HPO trials 2 and 4): a mid-round patience stop leaves cohort
    prompts with no reward history, and round-2 maxent selection then dies on its
    every-prompt-history contract. The stop must break the round loop."""
    src = (ROOT / "src" / "rlv2" / "train" / "rl" / "grpo.py").read_text(encoding="utf-8")
    assert "_stopped_early[0] = True" in src, "early stop must set the arm-level flag"
    i = src.index("patience stop ends the arm")
    assert "break" in src[i:i + 900], "the flag must break the round loop"
    # And the post-run receipt must not re-violate the contract on the unjudged remainder,
    # nor die when nothing among the judged prompts spans the threshold.
    assert 'r["ptext"] in stats' in src, "receipt must restrict to judged prompts"
    assert "receipt probabilities unavailable" in src, "no-span receipt must record, not raise"


def test_mgpo_source_seeds_baseline_and_counts_lifts():
    src = (ROOT / "src" / "rlv2" / "train" / "rl" / "grpo.py").read_text(encoding="utf-8")
    assert "baseline seeded" in src, "eval #1 must anchor, not lift"
    assert "max_sequential_lifts" in src, "HPO driver reads this summary token"
    assert '_PATIENCE = max(0, getattr(args, "patience", 2))' in src
    assert '"--patience"' in src
    # HPO mechanism 3 (trials 10/14): default linear decay ties effective lr to
    # steps-per-round; the scheduler must be selectable and reach GRPOConfig.
    assert 'lr_scheduler_type=getattr(args, "lr_scheduler", "linear")' in src
    assert '"--lr-scheduler"' in src


def test_observability_gate_is_lane_aware():
    """Measured on the 2026-08-31 smoke: 'step 2 has only [R3]; three regimes are required'
    killed a correct r3 run at the first observation step. The r3 mix is 100% R3 by
    construction, so the monitor takes min_regimes=1 there; the vibe default stays 3."""
    from rlv2.train.rl import grpo as mgpo
    recs = [{"ptext": "p1", "regime": "R3", "gold": "g"}] * 2
    comps = ["an answer", "another answer"]
    scores = [0.0, 1.0]                                 # non-zero spread: no kill gate
    ok = mgpo.TrainingHealthMonitor(observe_every=1, watch_class_collapse=False,
                                    min_regimes=1)
    ok.record(recs, comps, scores)                      # must not raise
    strict = mgpo.TrainingHealthMonitor(observe_every=1, watch_class_collapse=False)
    with pytest.raises(RuntimeError, match="observability gate"):
        strict.record(recs, comps, scores)
    src = (ROOT / "src" / "rlv2" / "train" / "rl" / "grpo.py").read_text(encoding="utf-8")
    assert 'min_regimes=(1 if getattr(args, "lane", "vibe") == "r3" else 3)' in src


def test_mgpo_sets_expandable_segments_before_torch_import():
    """Measured on the 2026-08-31 r3 smoke: allocator fragmentation pinned ~11 GB after the
    first backward (offload + empty_cache left 4128 MiB free vs 15203 before it), and the
    ollama 4b judge slowed 129s -> 371s per group. The knob only works if it is set before
    the first CUDA allocation, i.e. textually before `import torch`."""
    src = (ROOT / "src" / "rlv2" / "train" / "rl" / "grpo.py").read_text(encoding="utf-8")
    knob = src.index('os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")')
    assert knob < src.index("\nimport torch"), "alloc conf set after torch import -- inert"


def test_class_collapse_gate_is_disarmed_only_on_r3():
    """On R3 every completion is class=answer BY DESIGN (closed-book, no abstention); the
    collapse tripwire would kill a correct run. The zero-variance gate must stay armed."""
    from rlv2.train.rl import grpo as mgpo
    m = mgpo.TrainingHealthMonitor(watch_class_collapse=False)
    assert m.watch_class_collapse is False
    src = (ROOT / "src" / "rlv2" / "train" / "rl" / "grpo.py").read_text(encoding="utf-8")
    assert 'watch_class_collapse=(getattr(args, "lane", "vibe") != "r3")' in src
    assert "self.watch_class_collapse and len(classes) == 1" in src
