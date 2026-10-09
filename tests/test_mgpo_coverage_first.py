"""tests/test_mgpo_coverage_first.py — the coverage-first curriculum.

Spec: docs/plans (coverage-first curriculum, approved 2026-09-07); operator rule:
'train once and done until all other records have been visited' — visit count
dominates, uncertainty breaks ties, no repeats within a run while unvisited
records remain, fresh records always enter at the front.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import random


from rlv2.train.rl import grpo as M

SRC = (Path(__file__).resolve().parents[1] / "src" / "rlv2" / "train" / "rl" / "grpo.py").read_text(encoding="utf-8")


def _mix(n):
    return [{"ptext": "prompt-%02d" % i, "question": "q%d" % i} for i in range(n)]


def _args(ledger, pool=4):
    return SimpleNamespace(visit_ledger=ledger, target_pool=pool,
                           boxcox_lambda=M.BOXCOX_LAMBDA)


def _ledger_file(tmp_path, entries):
    """entries: {ptext: int-or-dict} written under the arm-end sha1[:16] key."""
    led = {hashlib.sha1(pt.encode("utf-8")).hexdigest()[:16]: e
           for pt, e in entries.items()}
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps(led), encoding="utf-8")
    return str(p)


def test_least_visited_first_and_uncertainty_breaks_ties(tmp_path):
    mix = _mix(6)
    ledger = _ledger_file(tmp_path, {
        "prompt-00": {"w": 3, "l": 0, "visits": 2},   # settled, much visited
        "prompt-01": {"w": 1, "l": 1, "visits": 1},   # contested, visited once
        "prompt-02": {"w": 2, "l": 0, "visits": 1},   # settled, visited once
        "prompt-03": {"w": 0, "l": 0, "visits": 0},   # never visited
        # prompt-04, prompt-05 absent from ledger = never visited
    })
    sel = M.coverage_first_cohort(mix, _args(ledger, pool=4), set(), random.Random(7))
    names = [r["ptext"] for r in sel]
    # the three never-visited records fill first, in some order
    assert set(names[:3]) == {"prompt-03", "prompt-04", "prompt-05"}
    # among visits=1, the CONTESTED one (higher Beta variance) outranks the settled
    assert names[3] == "prompt-01"
    assert "prompt-00" not in names, "the most-visited record must wait for coverage"


def test_no_repeats_across_rounds_while_unvisited_remain(tmp_path):
    mix = _mix(10)
    ledger = _ledger_file(tmp_path, {
        "prompt-%02d" % i: {"w": i % 2, "l": (i + 1) % 2, "visits": 0}
        for i in range(10)})
    args = _args(ledger, pool=4)
    r0 = M.coverage_first_cohort(mix, args, set(), random.Random(7))
    already = {r["ptext"] for r in r0}
    r1 = M.coverage_first_cohort(mix, args, already, random.Random(7))
    assert not ({r["ptext"] for r in r1} & already), \
        "a record repeated while unvisited records remained"


def test_pool_exhausted_allows_revisit_least_visited_first(tmp_path):
    mix = _mix(4)
    ledger = _ledger_file(tmp_path, {
        "prompt-00": {"w": 1, "l": 0, "visits": 3},
        "prompt-01": {"w": 0, "l": 1, "visits": 1},
        "prompt-02": {"w": 1, "l": 1, "visits": 2},
        "prompt-03": {"w": 0, "l": 0, "visits": 1},
    })
    already = {r["ptext"] for r in mix}          # everything visited this run
    sel = M.coverage_first_cohort(mix, _args(ledger, pool=2), already,
                                  random.Random(7))
    assert [r["ptext"] for r in sel][0] in {"prompt-01", "prompt-03"}, \
        "revisit must start from the least-visited records"
    assert len(sel) == 2


def test_mixed_int_and_dict_ledger_entries_normalize(tmp_path):
    """The probe path writes bare ints, the arm-end writer dicts; a ledger
    touched by both must not TypeError (the latent bug the explorer found)."""
    mix = _mix(3)
    ledger = _ledger_file(tmp_path, {
        "prompt-00": 5,                                  # bare int (probe path)
        "prompt-01": {"w": 1, "l": 1, "visits": 1},      # dict (arm-end path)
        # prompt-02 absent = unvisited
    })
    sel = M.coverage_first_cohort(mix, _args(ledger, pool=3), set(),
                                  random.Random(7))
    assert [r["ptext"] for r in sel] == ["prompt-02", "prompt-01", "prompt-00"]


def test_flag_defaults_off_and_dispatch_preserves_incumbent_branches():
    """Default OFF: without the flag, cohort creation and the round dispatch are
    unchanged — including the literal per_continuation call other tests pin."""
    assert 'action="store_true", dest="coverage_first"' in SRC
    loop = SRC.split("for rd in range(args.rounds):")[1]
    i_cov = loop.find('getattr(args, "coverage_first", False)')
    i_per = loop.find('getattr(args, "per_sampling", False)')
    assert 0 <= i_cov < i_per, \
        "coverage-first must be checked before the PER branch"
    assert "per_continuation(cohort, stats, mix, args, rng)" in loop, \
        "the pinned PER call expression must survive"
    assert "_visited_this_run.update(record[\"ptext\"] for record in sel)" in loop, \
        "visited bookkeeping must run for every branch"


# --------------------------------------------------------------------------
# Records per optimizer step + ledger-seeded uncertainty (operator, 2026-09-07)
# --------------------------------------------------------------------------
def test_grad_accum_defaults_to_one_and_reaches_the_trainer_config():
    """The hardcoded gradient_accumulation_steps=1 is why the winning shape only
    ever touched 6 of RL's own 240 records. The flag makes coverage reachable;
    default 1 keeps every historical recipe byte-identical."""
    assert '"--grad-accum", type=int, default=1' in SRC
    assert 'gradient_accumulation_steps=max(1, getattr(args, "grad_accum", 1))' in SRC


def test_per_priors_fall_back_to_the_ledger_but_never_override_in_run_evidence(tmp_path):
    """After a full-coverage pass the outcomes live on disk. Without ledger
    seeding the next run sees every prompt as unseen and cannot re-sample what
    it is unsure of. In-run rollouts still win where they exist."""
    ledger = _ledger_file(tmp_path, {
        "prompt-00": {"w": 3, "l": 3, "visits": 1},   # ledger-contested
        "prompt-01": {"w": 6, "l": 0, "visits": 1},   # ledger-settled
        "prompt-02": {"w": 6, "l": 0, "visits": 1},   # settled on disk...
    })
    cohort = [{"ptext": "prompt-0%d" % i, "question": "q%d" % i} for i in range(3)]
    args = SimpleNamespace(visit_ledger=ledger, target_pool=3, per_alpha=0.6,
                           per_eps=1e-3, world_model=None,
                           boxcox_lambda=M.BOXCOX_LAMBDA)
    stats = {"prompt-02": [0.9, 0.1]}                  # ...but contested in-run
    seen = {}
    real_choices = random.Random.choices

    def spy(self, population, weights=None, k=1):
        seen["w"] = dict(zip([r["ptext"] for r in population], weights))
        return real_choices(self, population, weights=weights, k=k)

    import pytest as _p
    monkey = _p.MonkeyPatch()
    monkey.setattr(random.Random, "choices", spy)
    try:
        M.per_continuation(cohort, stats, cohort, args, random.Random(7))
    finally:
        monkey.undo()
    w = seen["w"]
    assert w["prompt-00"] > w["prompt-01"], \
        "ledger-contested must outrank ledger-settled"
    assert w["prompt-02"] > w["prompt-01"], \
        "in-run evidence must override a settled ledger entry"


def test_zero_std_kill_gate_counts_optimizer_steps_not_records():
    """Measured 2026-09-07: the first full-coverage run died at step 6 of 20 on
    'frac_reward_zero_std=1.0 persisted for 6 consecutive steps'. record() runs
    once per PROMPT, so the default patience meant six consecutive uniform
    RECORDS — which a coverage walk hits by construction (~40% of unvisited
    records are uniformly right or wrong; only contested ones carry gradient).
    Under accumulation the gate must count dead OPTIMIZER STEPS."""
    assert ("zero_std_patience=ZERO_STD_PATIENCE * max(1, getattr(args, "
            '"grad_accum", 1))') in SRC, \
        "the kill gate no longer scales with records-per-step"
    # the monitor itself still fires on a genuinely dead streak
    m = M.TrainingHealthMonitor(observe_every=1, zero_std_patience=2,
                                watch_class_collapse=False, min_regimes=1)
    rec = [{"ptext": "p", "regime": "R3", "domain": "d"}]
    for _ in range(2):
        m.record(rec, ["same"], [1.0])          # one prompt, uniform outcome
    with pytest.raises(RuntimeError, match="frac_reward_zero_std"):
        m.record(rec, ["same"], [1.0])


# ── Round-boundary VRAM release (playbook Layer 7, task RV1) ─────────────────
# Two OOMs at round 2 (grad-accum 12 and 6) with 13.67 GiB allocated against a
# 14.78 GiB ceiling: _offload_models pinned round N's policy and TRL's ref model
# while _reload_best loaded a third. `tr = None` released only the trainer.

class _FakeParam:
    def __init__(self):
        self.grad = object()


class _FakeModel:
    def __init__(self, n=3):
        self._params = [_FakeParam() for _ in range(n)]

    def parameters(self):
        return self._params


def test_release_round_vram_empties_the_offload_list():
    models = [_FakeModel(), _FakeModel()]
    offload = list(models)
    M._release_round_vram(offload, tag="test")
    assert offload == []


def test_release_round_vram_drops_gradients_on_every_held_model():
    models = [_FakeModel(2), _FakeModel(4)]
    offload = list(models)
    M._release_round_vram(offload, tag="test")
    assert [p.grad for m in models for p in m.parameters()] == [None] * 6


def test_release_round_vram_tolerates_none_and_paramless_holders():
    M._release_round_vram(None, tag="test")
    holder = [object()]
    M._release_round_vram(holder, tag="test")
    assert holder == []


def test_every_mid_arm_reload_releases_the_round_first():
    """No _reload_best call inside the round loop without a release above it.

    The 2026-09-02 fix put `tr = None` here and it read as sufficient for six
    days. Pin the real invariant: the offload list is cleared before a third
    model is built.
    """
    lines = SRC.splitlines()
    call_lines = [i for i, ln in enumerate(lines)
                  if "_reload_best(args," in ln and "def " not in ln]
    assert len(call_lines) == 2, f"expected 2 mid-arm reload sites, found {len(call_lines)}"
    for i in call_lines:
        window = "\n".join(lines[max(0, i - 6):i])
        assert "_release_round_vram(" in window, (
            f"_reload_best at line {i + 1} is not preceded by a VRAM release")
        assert "policy = None" in window, (
            f"_reload_best at line {i + 1} does not drop the old policy reference")


# ── Anchor on the input model (playbook Layer 7, task RV2) ───────────────────
# cov240e read 8/16 at step 1 and stopped at step 6 with `best=0.5000 at None`:
# the first eval fires after step 1, so step 1 was consumed as the baseline and
# its weights were never saved.

def test_eval_at_start_flag_defaults_off():
    ap = M.build_argparser() if hasattr(M, "build_argparser") else None
    if ap is None:
        assert '"--eval-at-start"' in SRC or "'--eval-at-start'" in SRC
        assert 'action="store_true", dest="eval_at_start"' in SRC
        return
    assert ap.parse_args([]).eval_at_start is False


def test_pre_training_anchor_runs_on_round_one_only():
    """on_train_begin must gate on the flag, on round 0, and on the r3 lane."""
    i = SRC.index("def on_train_begin")
    body = SRC[i:SRC.index("def on_step_end", i)]
    assert 'getattr(args, "eval_at_start", False)' in body
    assert "rd != 0" in body
    assert '_lane != "r3"' in body
    assert "global_step=0" in body, "the anchor eval must report step 0"


def test_baseline_is_seeded_before_any_lift_check():
    """Eval #1 seeds and returns; only eval #2+ can save a checkpoint."""
    i = SRC.index("if _n_evals[0] == 1:")
    seed = SRC[i:SRC.index("if _rate > _best_score[0]", i)]
    assert "_best_score[0] = _rate" in seed
    assert seed.rstrip().endswith("return")
