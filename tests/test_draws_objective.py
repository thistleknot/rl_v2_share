"""Scoring a config on how cheaply it produces differentiable groups.

Operator 2026-09-14: "I think i found a new way to explore hpo parms / Parms that
give us the best differentiable signal (least amount of redraws)".

WHY IT BEATS THE EVAL AS A SEARCH SIGNAL. The eval gives ONE reading per trial,
on 16 questions, with a measured +-1 same-seed wobble -- so a whole 45-minute
trial can return a number indistinguishable from noise. Draw efficiency gives a
reading per step, and it is what the trials are actually spending their time on:
measured from base, 85-88% of probed groups came back uniform and two trials were
killed at the cap having completed 2 and 1 steps.

THE FAILURE MODE IT MUST NOT HAVE. Fewest-redraws on its own is maximised by
cranking sampling temperature: incoherent rollouts disagree with each other, so
every group looks "live" and none of them teach anything. The score is therefore
live_rate * mean_reward -- spread bought by destroying the model collapses the
second factor.
"""
import io

import pytest

SRC = "archive/tools/rl_hpo_optuna.py"


def score(live_rate, mean_reward):
    """Mirror of the draws objective in rl_hpo_optuna.run_trial."""
    return live_rate * mean_reward


def _src():
    return io.open(SRC, encoding="utf-8").read()


def test_a_config_with_no_dead_groups_and_real_reward_scores_highest():
    assert score(1.0, 0.6) > score(0.5, 0.6)
    assert score(1.0, 0.6) > score(1.0, 0.3)


def test_spread_bought_by_incoherence_scores_near_zero():
    """The guard: every group live, but the model is producing garbage."""
    garbage = score(1.0, 0.02)
    honest = score(0.4, 0.5)
    assert garbage < honest, (garbage, honest)


def test_an_all_dead_config_scores_zero():
    assert score(0.0, 0.5) == 0.0


def test_the_objective_is_monotone_in_live_rate_at_fixed_quality():
    vals = [score(r / 10.0, 0.5) for r in range(11)]
    assert vals == sorted(vals)


def test_the_objective_is_monotone_in_quality_at_fixed_live_rate():
    vals = [score(0.7, q / 10.0) for q in range(11)]
    assert vals == sorted(vals)


def test_the_flag_exists_and_defaults_to_the_lift_based_score():
    s = _src()
    assert 'choices=("floor", "draws")' in s
    assert 'ap.add_argument("--objective", default="floor"' in s, \
        "lift stays the default; draw efficiency is opt-in"


def test_sampling_axes_are_searched():
    """temperature and top_p decide how different four rollouts of the SAME
    prompt are -- the direct lever on whether a group is live. They were flags
    the whole project and never once searched."""
    s = _src()
    assert 'suggest_float("temperature"' in s
    assert 'suggest_float("top_p"' in s


def test_the_searched_temperature_range_brackets_the_current_default():
    """0.9 is what every number on the board was measured at; the range must
    contain it so the incumbent is reachable, not replaced."""
    import re
    m = re.search(r'suggest_float\("temperature", ([0-9.]+), ([0-9.]+)\)', _src())
    assert m, "temperature must be searched"
    lo, hi = float(m.group(1)), float(m.group(2))
    assert lo <= 0.9 <= hi, (lo, hi)
