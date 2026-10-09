"""Long2Short pays nothing extra when every correct answer has the same length.

Spec: .spec/specs/r3-domain-injection/tasks.md #49 (brevity shaping on equal lengths).
Live evidence 2026-09-27: "[reward] 3/4 correct at 0.70 | zero-sum drift 6.0e-01" -- float
residue in the deviations was divided by itself and paid every correct answer 1.2.
"""
import importlib.util
import sys

import pytest

pytest.importorskip("torch")


@pytest.fixture(scope="module")
def gof():
    spec = importlib.util.spec_from_file_location("gof_l2s", "docs/grpo_one_file.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["gof_l2s"] = m
    saved, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(m)
    sys.argv = saved
    return m


def _cfg(gof):
    return dict(gof.DEFAULTS)


@pytest.mark.parametrize("length", [20, 39, 166])   # lengths whose 3-way mean leaves float residue
def test_equal_length_correct_answers_are_paid_one(gof, length):
    answers = ["x" * length] * 3 + ["y" * 50]
    shaped, _ = gof.score_group("q", "g", answers, _cfg(gof), raw=[1.0, 1.0, 1.0, 0.0])
    assert shaped[:3] == [1.0, 1.0, 1.0]
    assert shaped[3] == 0.0


def test_different_lengths_still_shaped_and_zero_sum(gof):
    answers = ["x" * 100, "x" * 200, "x" * 300, "y"]
    shaped, _ = gof.score_group("q", "g", answers, _cfg(gof), raw=[1.0, 1.0, 1.0, 0.0])
    assert shaped[0] == pytest.approx(1.2)          # shortest gets the full +lambda
    assert shaped[0] > shaped[1] > shaped[2]
    assert sum(shaped[:3]) == pytest.approx(3.0)


def test_one_char_spread_at_length_1000_is_still_shaped(gof):
    answers = ["x" * 999, "x" * 1000, "z"]
    shaped, _ = gof.score_group("q", "g", answers, _cfg(gof), raw=[1.0, 1.0, 0.0])
    assert shaped[0] == pytest.approx(1.2)
    assert shaped[1] == pytest.approx(0.8)
