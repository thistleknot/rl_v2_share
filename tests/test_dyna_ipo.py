"""tests/test_dyna_ipo.py — Dyna-MGPO: the round-boundary offline update.

Spec: tasks.md #44. The skip path runs without trl/torch training (heavy imports
sit AFTER the pair-count gate), so the ledger->pairs plumbing tests cheaply.
"""
import sys
from pathlib import Path
from types import SimpleNamespace


from rlv2.train.rl import grpo as M

SRC = (Path(__file__).resolve().parents[1] / "src" / "rlv2" / "train" / "rl" / "grpo.py").read_text(encoding="utf-8")
GOOD = "a full correct answer that restates the fact at proper length for the gate"


def test_reward_path_feeds_the_ledger_with_raw_scores():
    block = SRC.split("def make_reward")[1].split("\ndef ")[0]
    assert "rollout_ledger.append((prompt, text, raw_value))" in block, \
        "judged rollouts are still being discarded"
    assert "rollout_ledger=None" in block.split(")")[0] + ")", \
        "ledger must be optional so non-dyna runs are unchanged"


def test_dyna_step_runs_before_continuation_selection_each_round():
    loop = SRC.split("for rd in range(args.rounds):")[1]
    i_dyna = loop.find("dyna_ipo_step(policy, tok, _dyna_ledger, args)")
    i_sel = loop.find("per_continuation(cohort, stats, mix, args, rng)")
    assert 0 <= i_dyna < i_sel, \
        "the offline update must land BEFORE the next round's prompt selection"
    # Starvation fix (2026-09-05, first trial logged '[dyna-ipo] skipped: 2 pairs
    # < min 4'): the ledger PERSISTS across rounds — replay buffers accumulate
    # experience, they don't forget it each round.
    assert "_dyna_ledger.clear()" not in loop, \
        "ledger cleared per round — that starved the in-loop update"


def test_skip_path_returns_policy_untouched_below_min_pairs():
    policy = object()
    ledger = [("q1", "wrong answer", 0.1), ("q1", "also wrong", 0.2)]  # no chosen
    args = SimpleNamespace(dyna_min_pairs=2, out="X", seed=7)
    assert M.dyna_ipo_step(policy, None, ledger, args) is policy


def test_ledger_grouping_reuses_dpo_pass_pair_rules(monkeypatch):
    captured = {}
    from rlv2.train.rl import dpo_pass as D
    def fake_build_pairs(pool, gens, scores, floor):
        captured.update(pool=pool, gens=gens, scores=scores, floor=floor)
        return []   # forces the skip path — no training machinery touched
    monkeypatch.setattr(D, "build_pairs", fake_build_pairs)
    ledger = [("q1", GOOD, 0.9), ("q1", "bad", 0.1), ("q2", "meh", 0.3)]
    args = SimpleNamespace(dyna_min_pairs=1, out="X", seed=7)
    M.dyna_ipo_step(object(), None, ledger, args)
    assert captured["gens"]["q1"] == [GOOD, "bad"]
    assert captured["scores"]["q1"] == [0.9, 0.1]
    assert captured["scores"]["q2"] == [0.3]
    assert captured["floor"] >= 0.7


def test_flag_defaults_off_so_prior_recipes_are_unchanged():
    assert '"--dyna-ipo", action="store_true"' in SRC
    assert '_dyna_ledger = [] if getattr(args, "dyna_ipo", False) else None' in SRC


def test_compat_shim_precedes_the_trl_import():
    """Pins the 2026-09-05 crash: the vision-mapping shim placed AFTER
    `from trl import` reproduces the import failure dpo_pass already solved.
    The skip-path tests can't catch this (they never reach the imports)."""
    body = SRC.split("def dyna_ipo_step")[1].split("\ndef ")[0]
    i_shim = body.find("MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES")
    i_trl = body.find("from trl import")
    assert 0 <= i_shim < i_trl, "shim must run before trl is imported"


def test_merge_reenables_grads_with_the_embedding_freeze_reapplied():
    """Pins the round-2 crash ('No inf checks were recorded for this optimizer'):
    merge_and_unload returns every param frozen from the PEFT wrap; without
    re-enabling grads the next GRPO round has nothing to train."""
    body = SRC.split("def dyna_ipo_step")[1].split("\ndef ")[0]
    i_merge = body.find("merge_and_unload()")
    i_grads = body.find("requires_grad_")
    assert 0 <= i_merge < i_grads, "grads must be re-enabled AFTER the merge"
    assert '"embed_tokens" in _n or "lm_head" in _n' in body, \
        "the embedding freeze must be re-applied post-merge"
