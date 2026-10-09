"""The redraw bracket: Wilson intervals, non-overlap eviction, shared sequence.

Basis, operator 2026-09-14: "We do a set of these trials and then compare them
relative to each other ... whoever has the most redraws by that point is voted
off the island", "maybe 3 trials is too small. Maybe we need 5", and on the
statistics, "W 5 we can use stats like means and sdev and those clearly outside a
95 CI?"

THREE THINGS THIS FILE PROTECTS, each one a way the bracket could look like it
works while ranking noise.

1. WILSON, NOT THE NORMAL APPROXIMATION. The count of draws-to-fill is negative
   binomial and skewed, but the underlying data is Bernoulli per draw, so the
   live rate has an exact interval. The normal approximation is too narrow, and
   at k=0 it has ZERO width -- it would declare certainty about the most broken
   config in the bracket.

2. EVICTION AGAINST THE LEADER, NOT AGAINST THE SPREAD OF ARMS. An interval taken
   across the five arms describes how different the configs are; the best arm sits
   outside it on the good side. That rule evicts the winner.

3. NOBODY DIES ON THIN EVIDENCE. At one round (~20 draws) the interval spans most
   of the range and no eviction should fire. That is not a bug to tune away -- it
   is the correct answer at that sample size.
"""
import importlib.util
import io
import sys

import pytest

SRC = "archive/tools/bracket_redraw.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("bracket_redraw", SRC)
    m = importlib.util.module_from_spec(spec)
    sys.modules["bracket_redraw"] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def src():
    return io.open(SRC, encoding="utf-8").read()


# ---------------------------------------------------------------- Wilson ----
def test_wilson_matches_the_hand_computed_interval(mod):
    """k=4, n=20 -> [0.081, 0.416], the figure the design was argued from."""
    lo, hi = mod.wilson(4, 20)
    assert abs(lo - 0.081) < 0.002, lo
    assert abs(hi - 0.416) < 0.002, hi


def test_wilson_is_wider_than_the_normal_approximation_it_replaces(mod):
    """The normal interval at k=4/n=20 is [0.025, 0.375] -- narrower AND shifted
    low, so it would evict arms the evidence cannot separate."""
    lo, hi = mod.wilson(4, 20)
    assert hi > 0.375, "Wilson must not be narrower than the approximation"
    assert lo > 0.025, "and it must not sit as low"


def test_wilson_has_non_zero_width_when_nothing_was_live(mod):
    """k=0 is the badly-broken config. The normal approximation gives width 0
    there and would declare certainty about the arm we most need to measure."""
    lo, hi = mod.wilson(0, 20)
    assert lo == 0.0
    assert hi > 0.15, "a zero-success arm must still carry real uncertainty"


def test_wilson_handles_the_all_live_end(mod):
    lo, hi = mod.wilson(20, 20)
    assert hi == 1.0
    assert lo < 0.9, "certainty is not warranted at n=20 either"


def test_wilson_never_leaves_the_unit_interval(mod):
    for k, n in ((0, 1), (1, 1), (0, 3), (3, 3), (1, 500), (499, 500)):
        lo, hi = mod.wilson(k, n)
        assert 0.0 <= lo <= hi <= 1.0, (k, n, lo, hi)


def test_wilson_tightens_as_draws_accumulate(mod):
    """The whole reason eviction is deferred rather than forced at round 1."""
    widths = [mod.wilson(int(0.2 * n), n)[1] - mod.wilson(int(0.2 * n), n)[0]
              for n in (20, 50, 100, 400)]
    assert widths == sorted(widths, reverse=True), widths
    assert widths[0] > 0.30, "one round cannot separate anything"
    assert widths[-1] < 0.10, "four hundred draws can"


def test_an_empty_arm_is_maximally_uncertain(mod):
    assert mod.wilson(0, 0) == (0.0, 1.0)


# ------------------------------------------------------------- eviction ----
def _evicts(mod, leader, challenger):
    """Mirror of the rule in main(): challenger dies iff its upper bound sits
    strictly below the leader's lower bound."""
    l_lo, _ = mod.wilson(*leader)
    _, c_hi = mod.wilson(*challenger)
    return c_hi < l_lo


def test_nobody_is_evicted_after_a_single_round(mod):
    """~20 draws each: 40% vs 10% live still overlap. Correct at that n."""
    assert not _evicts(mod, (8, 20), (2, 20))


def test_a_two_fold_difference_separates_once_draws_accumulate(mod):
    """The scale of difference worth acting on, at the budget the bracket has."""
    assert _evicts(mod, (40, 100), (10, 100))


def test_a_small_difference_never_separates(mod):
    """20% vs 25% is not worth compute, and the rule declines to pretend."""
    assert not _evicts(mod, (25, 100), (20, 100))
    assert not _evicts(mod, (100, 400), (80, 400))


def test_a_totally_dead_arm_dies_at_the_first_checkpoint(mod):
    """The one case that SHOULD fire early, and does.

        leader 8/20 -> [0.219, 0.613]
        dead   0/20 -> [0.000, 0.161]     0.161 < 0.219, no overlap

    Zero live groups in twenty draws is decisive even at that sample size, and
    an arm that cannot produce a gradient at all should not consume a second
    round. This is the counterpart to the small-difference case: the rule is
    deliberately slow about 20%-vs-25% and fast about 40%-vs-0%.
    """
    assert _evicts(mod, (8, 20), (0, 20)), "a dead arm dies immediately"
    assert _evicts(mod, (40, 100), (0, 100))


def test_the_leader_can_never_evict_itself(mod):
    assert not _evicts(mod, (40, 100), (40, 100))


# -------------------------------------------------------------- wiring ----


def test_arm_zero_is_the_incumbent(mod):
    """A bracket of five novel guesses has no reference point."""
    import random
    arm = mod.sample_arm(random.Random(0), 0)
    assert arm["name"] == "incumbent"
    assert set(arm) == {"name"}, "the incumbent overrides nothing"


def test_other_arms_vary_only_axes_that_can_change_a_rollout(mod):
    import random
    keys = set()
    for i in range(1, 6):
        keys.update(mod.sample_arm(random.Random(i), i))
    keys.discard("name")
    assert keys <= {"temperature", "top_p", "num_rollouts", "lr", "grad_clip",
                    "optimizer"}, keys
    assert "temperature" in keys, "the only knob that changes generated text"
    assert "grad_clip" in keys, "measured binding: gnorm 137-278 against clip 1.0"


def test_the_default_arm_count_is_five(src):
    assert '"--arms", type=int, default=5' in src, \
        "operator: 3 is too few to separate anything"


def test_the_fill_loop_counts_every_probe_not_just_the_accepted_ones(mod):
    """`draws` is the ranking statistic -- if it counted only successes it would
    be identical for every arm by construction."""
    class _Gof:
        @staticmethod
        def compute_grpo_loss(model, ref, tok, rec, cfg, stub=False):
            # every third record is differentiable
            return {"dead": rec["i"] % 3 != 0, "raw": [0.0, 1.0]}

    records = [{"i": i} for i in range(30)]
    chosen, draws, live = mod.fill_one_step(_Gof, None, None, {}, records, 4)
    assert len(chosen) == 4
    assert live == 4
    assert draws == 10, "four live records at every third probe = ten probes"


def test_the_fill_loop_stops_when_the_sequence_runs_out(mod):
    class _Gof:
        @staticmethod
        def compute_grpo_loss(model, ref, tok, rec, cfg, stub=False):
            return {"dead": True, "raw": [0.0, 0.0]}

    chosen, draws, live = mod.fill_one_step(
        _Gof, None, None, {}, [{"i": i} for i in range(7)], 4)
    assert chosen == [] and live == 0
    assert draws == 7, "a step that cannot fill still reports what it spent"


# ------------------------------------------------- shared queue / cursors ----
def test_one_queue_is_shuffled_once_not_per_round(src):
    """Operator 2026-09-14: "ordering is crucial".

    The earlier version reshuffled after round 0, so later rounds compared arms
    that differed in WHICH records they saw as well as in their weights -- two
    variables producing one number.
    """
    assert "queue = list(pool)" in src
    assert "random.Random(a.seed).shuffle(queue)" in src
    assert "random.Random(a.seed + rnd)" not in src, \
        "no per-round reshuffle; the order is fixed for the whole bracket"


def test_each_arm_keeps_its_own_cursor(src):
    assert 'cursor = {arm["name"]: 0 for arm in arms}' in src
    assert 'cursor[arm["name"]] = c + draws' in src, \
        "the cursor advances by draws SPENT, not by records accepted"


def test_arms_that_consume_different_amounts_still_see_the_same_records(mod):
    """The property the queue buys: unequal consumption, identical data.

    Arm A spends 12 probes, arm B spends 20. A's twelve are B's first twelve --
    so a comparison at any matched draw count is exact.
    """
    queue = ["r%d" % i for i in range(40)]
    cursor_a = cursor_b = 0
    seen_a, seen_b = [], []
    for spend_a, spend_b in ((12, 20), (9, 5)):
        seen_a += queue[cursor_a:cursor_a + spend_a]
        seen_b += queue[cursor_b:cursor_b + spend_b]
        cursor_a += spend_a
        cursor_b += spend_b
    n = min(len(seen_a), len(seen_b))
    assert seen_a[:n] == seen_b[:n], "the k-th probe must be the same record"
    assert cursor_a != cursor_b, "but the arms are at different positions"


def test_the_cursor_advances_by_probes_spent_not_records_kept(mod):
    """If it advanced by accepted records, a rejecting arm would re-probe the
    same dead records forever and its reject rate would be self-fulfilling."""
    class _Gof:
        @staticmethod
        def compute_grpo_loss(model, ref, tok, rec, cfg, stub=False):
            return {"dead": rec["i"] % 4 != 0, "raw": [0.0, 1.0]}

    queue = [{"i": i} for i in range(40)]
    _, draws, live = mod.fill_one_step(_Gof, None, None, {}, queue, 2)
    assert live == 2
    assert draws == 5, "two live at every fourth record = five probes"
    # the next step must start AFTER all five, not after the two kept
    assert queue[draws]["i"] == 5


def test_an_exhausted_queue_retires_the_arm_rather_than_wrapping(src):
    """Wrapping would re-probe records the arm has already seen, quietly turning
    the reject rate into a measure of memorisation."""
    assert "queue exhausted" in src
    assert 'arm["alive"] = False' in src


def test_retirement_is_not_applied_during_the_bracket(src):
    """A retired record would be skipped by one arm and not another, breaking
    the cursor alignment the whole design rests on."""
    assert "RETIREMENT IS OFF HERE" in src
    assert "is_retired" not in src
    assert "eligible_pool" not in src
