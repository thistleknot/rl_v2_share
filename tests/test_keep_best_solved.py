"""--keep-best-solved saves at the first full-coverage step, then on strict new highs.

Spec: .spec/specs/r3-domain-injection/tasks.md #55. Measured 2026-09-28: rl_b_echoguard set
its solved high (3) before full coverage, only tied it after, and never wrote best_solved.
"""
import importlib.util
import sys

import pytest

pytest.importorskip("torch")


@pytest.fixture(scope="module")
def gof():
    spec = importlib.util.spec_from_file_location("gof_bs", "docs/grpo_one_file.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["gof_bs"] = m
    saved, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(m)
    sys.argv = saved
    return m


def test_no_save_before_coverage(gof):
    assert gof.keep_best_solved(False, 9, -1) == (False, -1)


def test_first_full_coverage_step_saves_even_at_a_tie_with_the_run_high(gof):
    # the echoguard trace: solved 3 pre-coverage, 3 at the first full-coverage step
    assert gof.keep_best_solved(True, 3, -1) == (True, 3)


def test_tie_after_coverage_does_not_overwrite(gof):
    assert gof.keep_best_solved(True, 3, 3) == (False, 3)


def test_strict_new_high_after_coverage_saves(gof):
    assert gof.keep_best_solved(True, 5, 3) == (True, 5)


def test_replay_of_the_echoguard_run(gof):
    # (covered_all, solved) for steps 9..12 of rl_b_echoguard
    best, saves = -1, []
    for step, (cov, s) in enumerate([(False, 2), (False, 3), (True, 3), (True, 3)], 9):
        ok, best = gof.keep_best_solved(cov, s, best)
        if ok:
            saves.append(step)
    assert saves == [11]
