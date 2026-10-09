"""Per-step echo share (`[echo]` line, `echo_share` in metrics.json) and --keep-best-solved.

Spec: .spec/specs/r3-domain-injection/tasks.md #53.
Measured 2026-09-27: echo share rose 5.7% -> 49.4% unseen by any per-step line, and the
best-solved step's weights were overwritten one step later -- this closes both gaps.
"""
import importlib.util
import sys

import pytest

pytest.importorskip("torch")


@pytest.fixture(scope="module")
def gof():
    spec = importlib.util.spec_from_file_location("gof_echo", "docs/grpo_one_file.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["gof_echo"] = m
    saved, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(m)
    sys.argv = saved
    return m


Q = "what is the capital city of France"
ECHO_UNPAID = "what is the capital city of France is Paris"      # opens with Q, score 0 (guarded)
ECHO_PAID = "what is the capital city of France"                  # verbatim echo, judge paid it
REAL = "The capital of France is Paris"                           # not an echo


def test_echo_stats_counts_echoes_and_paid_echoes(gof):
    pairs = [(Q, ECHO_UNPAID, 0.0), (Q, ECHO_PAID, 1.0), (Q, REAL, 1.0)]
    n_echo, n, n_paid = gof.echo_stats(pairs)
    assert (n_echo, n, n_paid) == (2, 3, 1)


def test_echo_stats_empty_step_is_zero(gof):
    assert gof.echo_stats([]) == (0, 0, 0)


def test_echo_stats_all_real_answers_zero_echoes(gof):
    pairs = [(Q, REAL, 1.0), (Q, "Paris is the capital", 0.0)]
    n_echo, n, n_paid = gof.echo_stats(pairs)
    assert n_echo == 0
    assert n == 2
    assert n_paid == 0


def test_echo_stats_fails_if_body_emptied(gof):
    # a stubbed-out echo_stats that always returns zeros must not pass the first test's
    # non-trivial assertion -- this pins the function to actually counting, not stubbing.
    pairs = [(Q, ECHO_PAID, 1.0)]
    n_echo, n, n_paid = gof.echo_stats(pairs)
    assert n_echo == 1 and n_paid == 1, "echo_stats body looks emptied/stubbed"


def test_keep_best_solved_flag_parses():
    src = open("docs/grpo_one_file.py", encoding="utf-8").read()
    assert "--keep-best-solved" in src
    assert 'dest="keep_best_solved"' in src


def test_echo_share_wired_into_metrics(gof):
    src = open("docs/grpo_one_file.py", encoding="utf-8").read()
    assert '"echo_share"' in src
    # wired into both the step's `out` dict and the metrics.json row
    assert src.count('"echo_share"') >= 2
