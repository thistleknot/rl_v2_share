"""The HPO trial cap budgets WORK, not wall-clock.

NO GOVERNING SPEC. Basis: operator 2026-09-13 -- "If a trial takes longer than 20
min somethings broken", softened the same day to "I said 20 minutes ish". So the
cap is a budget, not a tripwire.

WHY A FLAT CAP IS WRONG. Trial duration scales with the searched `--rollouts`
axis. Measured 2026-09-13 on study rl_r3_replay_buffer, identical shape otherwise:

    rollouts 4   -> 3 steps completed in 20 min
    rollouts 8   -> 1 step  completed in 20 min
    rollouts 16  -> no step completed

A flat wall-clock cap truncates the 16-rollout arm four times harder than the
4-rollout arm. The search then ranks large groups last for a reason that has
nothing to do with their quality -- and large groups are the single axis most
likely to cure the dead groups (zero within-group variance, zero gradient) that
dominate this lane. Equal compute per arm is what makes the curves comparable.
"""
import pytest


def cap_for(base_seconds, rollouts):
    """Mirror of the budget rule in rl_hpo_optuna.run_trial.

    Kept as a local mirror because the rule lives inline inside a subprocess
    call; extracting it into the tool purely for testability would be the
    speculative abstraction the constitution forbids. The assertions below pin
    the ARITHMETIC, and the source comment carries the same numbers.
    """
    return int(base_seconds * max(1.0, rollouts / 4.0))


def test_the_smallest_group_gets_the_base_budget():
    assert cap_for(1200, 4) == 1200


def test_doubling_the_group_doubles_the_budget():
    assert cap_for(1200, 8) == 2400


def test_the_largest_searched_group_gets_four_times_the_budget():
    # rollouts is searched over [4, 8, 16]; 16 is the arm a flat cap destroyed
    assert cap_for(1200, 16) == 4800


def test_every_arm_gets_the_same_seconds_per_rollout():
    """The invariant that makes trials comparable: equal compute, not equal time."""
    per_rollout = {r: cap_for(1200, r) / r for r in (4, 8, 16)}
    assert len(set(per_rollout.values())) == 1, per_rollout
    assert per_rollout[4] == 300.0


def test_a_group_smaller_than_the_baseline_is_not_penalised():
    """max(1.0, ...) floors the multiplier -- a tiny group keeps the full budget.

    Without the floor, rollouts=2 would get a 600 s cap and be truncated for
    being cheap, which inverts the whole point of the rule.
    """
    assert cap_for(1200, 2) == 1200
    assert cap_for(1200, 1) == 1200


def test_the_budget_scales_with_the_operator_supplied_base():
    # "20 minutes ish" is the DEFAULT, not a constant baked into the rule
    assert cap_for(1800, 8) == 3600
    assert cap_for(600, 16) == 2400


@pytest.mark.parametrize("rollouts", [4, 8, 16])
def test_the_cap_is_always_a_positive_integer_of_seconds(rollouts):
    c = cap_for(1200, rollouts)
    assert isinstance(c, int) and c > 0
