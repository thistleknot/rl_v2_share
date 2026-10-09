"""Boltzmann sampling on prior sdev, with unvisited as its own case.

Spec: operator 2026-09-13.

  "simply tracking the priors sdev is all we need. Than use this as a type of
   boltzman sampler"
  "If sdev is null, it hasnt been visited"          <- unvisited is a CASE
  "anneal like a decay or Warmup ratio"             <- T high early, low late

WHY NOT A HARD FILTER. maxent_weights gives weight 0.0 to any prompt that does not
straddle the floor, which excludes exactly the two groups worth keeping reachable:
an all-failure record (dropped where the model is weakest, possibly never revisited
before patience ends the run) and a perfect record (never re-checked, so
catastrophic forgetting on it is invisible).

WHY UNVISITED IS NOT A PRIOR. An earlier version scored unvisited records 0.5 so
they would outrank settled ones -- which made them tie EXACTLY with a genuine
straddler, a coincidence that hid the fact that a number was doing the logic's job.

This is prioritized experience replay (Schaul et al. 2015) over prompts; the
"every step must be differentiable" rule beside it is DAPO's dynamic sampling.
"""
from __future__ import annotations

import importlib.util
import os as _o
import random
import sys as _s

for _d in ("docs",):
    _p = _o.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))), _d)
    _p not in _s.path and _s.path.insert(0, _p)

import pytest  # noqa: E402


@pytest.fixture(scope="module")
def gof():
    saved = _s.argv
    _s.argv = ["x"]
    spec = importlib.util.spec_from_file_location(
        "gof_bz", _o.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))),
                               "docs", "grpo_one_file.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    _s.argv = saved
    return m


VISITED = [{"question": "straddles", "category": "Math"},
           {"question": "always_wrong", "category": "Math"},
           {"question": "always_right", "category": "Math"}]
# 4 observations = 1 coverage visit + 3 REPLAYS, which is the retirement clock
HIST = {"straddles": [0.0, 1.0, 0.0, 1.0],
        "always_wrong": [0.0, 0.0, 0.0, 0.0],
        "always_right": [1.0, 1.0, 1.0, 1.0]}


# --------------------------------------------------------------------------
# unvisited is a CASE, not a number
# --------------------------------------------------------------------------
def test_spread_is_none_exactly_when_unvisited(gof):
    assert gof.visit_stats(HIST, "never_seen") == (0, None, None)
    assert gof.visit_stats({"q": [0.5]}, "q")[1] is None, "one sample is not a spread"
    # REQ-R7: spread is the DEBIASED expected gradient, never pstdev of the rewards.
    #
    # THREE CHANGES, THREE SCALES, all deliberate:
    #   Beta(3,3) over two bands        9/42   = 0.21   (retired)
    #   Dirichlet over three bands      raw    = 0.19   (retired)
    #   Dirichlet, shrinkage inverted   debias = 0.23   (retired)
#   MLE variance of the rewards     s2     = 0.25   (current)
    #
    # [0, 1, 0, 1] is 2 wrong + 2 hq. The debias exists because the raw posterior
    # moved LEVEL with n -- a record scored +0.014 higher for having been looked at
    # more often, worth ~8% extra draw weight through the Boltzmann and earned by
    # nothing (operator 2026-09-19: "more rollouts should buy precision, never
    # level", the p_hat = k/n result). Inverting the shrinkage recovers the sample
    # variance, which is level-stable at fixed rate.
    #
    # This number moving is the change landing. It is not a constant to be "fixed"
    # back to match an older scale.
    n, sd, mean = gof.visit_stats(HIST, "straddles")
    assert n == 4 and sd == 0.25 and mean == 0.5


def test_values_are_rounded_to_two_decimals(gof):
    # without rounding, float noise makes near-identical spreads DISTINCT ranks,
    # so the ECDF tie handling never fires and the bloc never splits its mass
    h = {"q": [0.3333333, 0.6666667, 0.3333333, 0.6666667]}
    n, sd, mean = gof.visit_stats(h, "q")
    assert sd == round(sd, 2) and mean == round(mean, 2)


def test_unknown_ranks_highest_of_all_states(gof):
    """UNKNOWN IS TOP RANK. Operator 2026-09-19, stating the ordering directly:

        "Maxent > straddle > .5 > 0 > 1 ... maxent would not be the majority
         selection when null is no longer the majority (null = unknown as we haven't
         scored it) but it would still be the highest chance"

    SUPERSEDES the previous contract, which let Beta(1,1)'s mean (0.1667) place
    unvisited THIRD of four states -- below a measured straddler at 0.2143. That
    placement was a consequence of scoring EXPECTED learnable signal with no term for
    not knowing, even though an unvisited record carries 4.2x the posterior variance
    (Var[p] 0.0833 vs 0.0198 after one visit).

    Measured cost of the old placement, 2026-09-19: coverage sat at 232/240 for five
    consecutive steps while 76 known straddlers out-ranked the 8 unknowns, and the
    exhaustion gate -- which cannot arm before full coverage -- never started.

    What falls away as the pool is explored is unknown's share of a batch by COUNT,
    not its per-record rank. That is the operator's distinction and it is what this
    asserts.
    """
    pool = [{"question": "straddles"}, {"question": "always_wrong"},
            {"question": "brand_new"}]
    w, unvisited = gof.boltzmann_weights(pool, HIST, temp=0.24)
    assert unvisited == 1
    assert w[2] > w[0] > w[1], "unknown > straddle > dead: %r" % (w,)


def test_unknown_is_pinned_to_the_maximum_priority(gof):
    """Top rank is structural. It used to be that the unvisited prior EQUALLED the
    sup of the score, giving rank exactly 1.0 -- but sup(s2) is also _PRIOR_MAX, so
    a perfect straddle normalised to 1.0 and TIED unvisited. The operator's ladder
    puts unknown first (2026-09-19), so the prior clears the ceiling instead."""
    assert gof._UNVISITED_PRIOR > gof._PRIOR_MAX
    assert gof._UNVISITED_PRIOR > 0.2143, "must outrank a measured straddler"


# --------------------------------------------------------------------------
# nothing visited is excluded
# --------------------------------------------------------------------------
def test_a_straddler_is_preferred_among_visited(gof):
    # RANKING COMPRESSES, and these are the measured numbers. sdevs 0.5/0/0
    # become ranks 0.75/0.375/0.375, so at T=0.25 the straddler is e^1.5 = 4.48x
    # likelier. The ECDF says "ranks above", never "by how much": a straddler at
    # sdev 0.5 and one at 0.05 would rank identically above the zeros. That is
    # the price of scale-independence, and T is what buys it back.
    w, _u = gof.boltzmann_weights(VISITED, HIST, temp=0.25)
    import math
    assert w[0] > w[1]
    assert w[0] / w[1] > 2.0


def test_lowering_the_temperature_buys_back_the_separation(gof):
    # the compression is a DIAL, not a loss: T is the knob, and because ranks are
    # scale-free the same T means the same thing whatever the reward looks like
    sharp, _u = gof.boltzmann_weights(VISITED, HIST, temp=0.1)
    assert sharp[0] / sharp[1] > 10


def test_an_all_failure_record_stays_reachable(gof):
    w, _u = gof.boltzmann_weights(VISITED, HIST, temp=0.25)
    assert w[1] > 0.0, "dropped exactly where the model is weakest"


def test_a_perfect_record_is_still_checked_occasionally(gof):
    # forgetting means perfect today is not perfect tomorrow
    w, _u = gof.boltzmann_weights(VISITED, HIST, temp=0.25)
    assert w[2] > 0.0


def test_failures_outrank_winners(gof):
    # THIS TEST WAS INVERTED. It previously asserted they rank TOGETHER, and this
    # file called that "the point of the whole thing". Operator 2026-09-13:
    # "Maxent > differentiable > failures (all 0s) > all winners (all 1 ...)".
    # Both have spread 0, but an all-failure record still has something to teach
    # while an all-winner one only needs an occasional forgetting check.
    w, _u = gof.boltzmann_weights(VISITED, HIST, temp=0.25)
    assert w[1] > w[2], "failures must outrank winners at equal spread"
    # THE ORDER HOLDS; THE 1.5x DOES NOT, AND CANNOT. That ratio was written when
    # flat records still had NON-ZERO spread (all-fail scored 0.119 and the x0.25
    # solved discount made a 4x raw gap). Under the MLE variance every uniform group
    # is exactly 0 -- they genuinely tie on gradient -- so the only separation left
    # is the ladder bonus, and sizing that for 1.5x needs a gap of
    # ln(1.5) * T * PRIOR_MAX = 0.025 while the smallest REAL spread is 0.019. A
    # 1.5x ladder would rank a flat record above a differentiable one, which is the
    # harder rule. Ordering by a nudge is the correct shape for a preference among
    # equals; a lexicographic sort is the open fix for making it strict.
    assert 1.0 < w[1] / w[2] < 1.5


def test_the_mean_tiebreak_never_crosses_a_real_spread_difference(gof):
    # tiebreak 0.001 against spreads rounded to 0.01: a worse-spread record can
    # never jump a better-spread one just by having a lower mean
    better_spread_high_mean = gof.priority_score(0.01, 1.0)
    worse_spread_low_mean = gof.priority_score(0.0, 0.0)
    assert better_spread_high_mean > worse_spread_low_mean


def test_an_unvisited_record_has_no_priority_score(gof):
    assert gof.priority_score(None, None) is None


def test_weights_form_a_distribution(gof):
    w, _u = gof.boltzmann_weights(VISITED, HIST, temp=0.25)
    assert sum(w) == pytest.approx(1.0)


def test_an_all_unvisited_pool_is_uniform(gof):
    # every record shares rank 1.0, so they split the mass equally -- the bloc
    # dynamic: nulls dominate while numerous and dilute as they are consumed
    w, unvisited = gof.boltzmann_weights([{"question": "a"}, {"question": "b"}], {},
                                         temp=0.25)
    assert unvisited == 2
    assert w[0] == pytest.approx(w[1])
    assert sum(w) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# temperature and its annealing
# --------------------------------------------------------------------------
def test_low_temperature_sharpens_high_temperature_flattens(gof):
    sharp, _u = gof.boltzmann_weights(VISITED, HIST, temp=0.1)
    flat, _u2 = gof.boltzmann_weights(VISITED, HIST, temp=5.0)
    assert sharp[0] / sharp[1] > flat[0] / flat[1]


def test_temperature_starts_high_and_ends_low(gof):
    # early estimates rest on 2-3 samples; sharp sampling on noise is overconfident
    assert gof.anneal_temp(0.0) > gof.anneal_temp(1.0)
    assert gof.anneal_temp(0.0) == pytest.approx(1.0)
    assert gof.anneal_temp(1.0) == pytest.approx(0.15)


def test_the_anneal_is_monotone_across_the_run(gof):
    xs = [gof.anneal_temp(i / 10.0) for i in range(11)]
    assert all(a >= b for a, b in zip(xs, xs[1:])), "temperature must never rise"


def test_linear_and_cosine_share_endpoints_but_differ_in_shape(gof):
    # They CROSS at the midpoint -- both give k=0.5 there, 0.57 measured -- so the
    # midpoint proves nothing. Compare a quarter in, where cosine is still high
    # (slow start, 0.88) and linear has already dropped (0.79).
    assert gof.anneal_temp(0.25, shape="cosine") > gof.anneal_temp(0.25, shape="linear")
    assert gof.anneal_temp(0.5, shape="cosine") == \
        pytest.approx(gof.anneal_temp(0.5, shape="linear"))
    for sh in ("linear", "cosine"):
        assert gof.anneal_temp(0.0, shape=sh) == pytest.approx(1.0)
        assert gof.anneal_temp(1.0, shape=sh) == pytest.approx(0.15)


def test_constant_shape_disables_the_anneal(gof):
    assert gof.anneal_temp(0.0, shape="constant") == \
        pytest.approx(gof.anneal_temp(1.0, shape="constant"))


def test_progress_outside_zero_to_one_is_clamped(gof):
    assert gof.anneal_temp(-5.0) == pytest.approx(gof.anneal_temp(0.0))
    assert gof.anneal_temp(99.0) == pytest.approx(gof.anneal_temp(1.0))


# --------------------------------------------------------------------------
# coverage survives all of it
# --------------------------------------------------------------------------
def test_domain_coverage_holds_under_boltzmann(gof):
    pool = [{"question": "%s%d" % (c, i), "category": c}
            for c in ("Math", "Code", "STEM", "Instruction") for i in range(4)]
    b = gof.boltzmann_batch(pool, {}, 4, random.Random(7))
    assert sorted(r["category"] for r in b) == ["Code", "Instruction", "Math", "STEM"]


def test_records_within_a_step_are_distinct(gof):
    pool = [{"question": "%s%d" % (c, i), "category": c}
            for c in ("Math", "Code", "STEM", "Instruction") for i in range(4)]
    b = gof.boltzmann_batch(pool, {}, 8, random.Random(3))
    assert len({r["question"] for r in b}) == len(b)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# --------------------------------------------------------------------------
# rank normalisation (operator 2026-09-13: "rank normalize and null just is max")
# --------------------------------------------------------------------------
def test_the_draw_no_longer_rank_normalises(gof):
    """SUPERSEDED by REQ-R7. rank_normalise turned the priority into an ORDINAL
    position and discarded its magnitude -- right for an uncalibrated sdev, wrong for
    a posterior where 0.214 vs 0.119 is a real 1.8x difference in learnable signal.
    It was also pool-relative: with every record dead the least-dead still ranked 1.0.

    The weight path now softmaxes the posterior against a CONSTANT (_PRIOR_MAX), so a
    uniformly dead pool scores uniformly low instead of renormalising to look healthy.
    """
    dead_pool = [{"question": "always_wrong"}, {"question": "always_right"}]
    w, _u = gof.boltzmann_weights(dead_pool, HIST, temp=0.24)
    assert all(x == x and abs(x) != float("inf") for x in w), "finite"
    # the whole pool is dead; no record is promoted to "top" by renormalisation
    assert max(w) < 0.9, "a dead pool must not manufacture a strong preference"


def test_ties_share_a_rank_so_they_split_the_mass(gof):
    r = gof.rank_normalise([0.0, 0.0, 0.0, 0.5])
    assert r[0] == r[1] == r[2], "tied sdevs must not be ordered arbitrarily"
    assert r[3] > r[0]


def test_every_unvisited_record_shares_the_same_prior(gof):
    """Unvisited records tie at the TOP rank and split that mass evenly.

    The tie is the part that survives the 2026-09-19 change; what moved is where they
    tie. They still dilute as they are consumed -- which is exactly the operator's
    "maxent would not be the majority selection when null is no longer the majority,
    but it would still be the highest chance": the SHARE falls with the count, the
    RANK does not."""
    pool = [{"question": "new_a"}, {"question": "new_b"}, {"question": "straddles"}]
    w, unvisited = gof.boltzmann_weights(pool, HIST, temp=0.24)
    assert unvisited == 2
    assert abs(w[0] - w[1]) < 1e-12, "unvisited records must tie exactly"
    assert w[0] > w[2], "an unmeasured record outranks a measured straddler"


def test_ordering_follows_sdev(gof):
    r = gof.rank_normalise([0.0, 0.1, 0.3, 0.5])
    assert r == sorted(r), "higher sdev must not rank lower"


def test_always_right_and_always_wrong_rank_together_at_the_bottom(gof):
    # both have sdev 0: known and unknowable sink equally, which is the point
    r = gof.rank_normalise([0.0, 0.0, 0.25, None])
    assert r[0] == r[1] == min(r)


def test_the_scale_of_sdev_does_not_change_the_ranks(gof):
    # THE reason for ranking: a binary reward tops out at 0.5, a graded one does
    # not, and a temperature tuned to one scale is wrong for the other
    small = gof.rank_normalise([0.0, 0.1, 0.2, 0.5])
    large = gof.rank_normalise([0.0, 10.0, 20.0, 50.0])
    assert small == large


def test_an_empty_pool_returns_nothing(gof):
    assert gof.rank_normalise([]) == []


def test_a_single_known_value_is_well_defined(gof):
    r = gof.rank_normalise([0.3])
    assert 0.0 < r[0] <= 1.0


def test_the_temperature_is_interpretable_on_ranks(gof):
    # ranks span [0,1], so the best/worst odds ratio is exp(1/T) whatever the
    # reward scale -- which is what lets T be held constant across runs
    import math
    w, _u = gof.boltzmann_weights(VISITED + [{"question": "brand_new"}], HIST,
                                  temp=0.5)
    # exp(1.1/T), not exp(1.0/T): the rank RANGE is 1.1 now, because unvisited sits
    # above the ceiling rather than on it. T still prices how much the scores
    # deserve to be trusted; only the top of the scale moved.
    assert max(w) / min(w) < math.exp(1.1 / 0.5) + 1e-6


# --------------------------------------------------------------------------
# retirement and termination (operator 2026-09-13)
# --------------------------------------------------------------------------
def test_the_replay_clock_does_not_count_rollouts(gof):
    # THE defect this closes: reward_history is EXTENDED with every rollout
    # score, so its length counts rollouts. Counting replays off it retires a
    # record after ONE visit at any group size >= 4 -- the hard filter that was
    # rejected. One visit of 8 dead rollouts is ONE look, not eight.
    vlog = {}
    gof.record_visit(vlog, "hard", [0.0] * 8)
    assert gof.n_replays(vlog, "hard") == 0, "eight rollouts is still one visit"
    assert gof.is_retired(vlog, "hard", retire_after=3) is False


def test_a_record_retires_after_three_dead_replays(gof):
    vlog = {}
    for visit in range(4):                       # 1 coverage visit + 3 replays
        gof.record_visit(vlog, "hard", [0.0] * 4)
        expected = visit >= 3
        assert gof.is_retired(vlog, "hard", retire_after=3) is expected,             "visit %d: replays=%d" % (visit + 1, gof.n_replays(vlog, "hard"))


def test_a_live_visit_resets_the_hard_no(gof):
    # retirement reads CONSECUTIVE dead replays -- one group with spread means
    # the record is still teaching something, whatever it did before
    vlog = {}
    for _ in range(3):
        gof.record_visit(vlog, "q", [0.0] * 4)
    gof.record_visit(vlog, "q", [0.0, 1.0, 0.0, 1.0])
    gof.record_visit(vlog, "q", [0.0] * 4)
    assert gof.n_replays(vlog, "q") == 4
    assert gof.is_retired(vlog, "q", retire_after=3) is False


def test_an_unvisited_record_is_never_retired(gof):
    assert gof.is_retired({}, "never_seen", retire_after=3) is False


def test_a_record_with_spread_is_never_retired(gof):
    vlog = {}
    for _ in range(9):
        gof.record_visit(vlog, "live", [0.0, 1.0, 1.0, 0.0])
    assert gof.is_retired(vlog, "live", retire_after=3) is False


def test_the_eligible_pool_drops_retired_records(gof):
    vlog = {}
    for _ in range(4):
        gof.record_visit(vlog, "always_wrong", [0.0] * 4)
        gof.record_visit(vlog, "straddles", [0.0, 1.0, 1.0, 0.0])
    pool = [{"question": "straddles"}, {"question": "always_wrong"},
            {"question": "brand_new"}]
    live, retired = gof.eligible_pool(pool, vlog, retire_after=3)
    assert retired == 1
    assert [r["question"] for r in live] == ["straddles", "brand_new"]


def test_an_empty_eligible_pool_signals_the_run_is_done(gof):
    # "if we run out of differentiable records we are done"
    vlog = {}
    for _ in range(4):
        gof.record_visit(vlog, "always_wrong", [0.0] * 4)
        gof.record_visit(vlog, "always_right", [1.0] * 4)
    pool = [{"question": "always_wrong", "category": "Math"},
            {"question": "always_right", "category": "Code"}]
    live, retired = gof.eligible_pool(pool, vlog, retire_after=3)
    assert live == [] and retired == 2
    import random as _r
    assert gof.boltzmann_batch(pool, HIST, 4, _r.Random(1), visit_log=vlog) == []


def test_the_ema_separates_records_that_pooled_stats_call_identical(gof):
    # same four observations, opposite order: one record is learning, the other
    # is forgetting. Pooled mean/sdev cannot tell them apart -- that is the whole
    # reason for an EMA, and it is why the window has to reach visit_stats.
    hist = {"learning": [0.0, 0.0, 1.0, 1.0], "forgetting": [1.0, 1.0, 0.0, 0.0]}
    pooled = [gof.visit_stats(hist, q) for q in ("learning", "forgetting")]
    assert pooled[0] == pooled[1], "pooled stats are blind to order, by construction"

    ema = [gof.visit_stats(hist, q, n_window=3) for q in ("learning", "forgetting")]
    assert ema[0][2] > ema[1][2], "recent successes must lift the EMA mean"
    assert ema[0] != ema[1]


def test_the_window_reaches_the_weights_through_the_public_call(gof):
    # the defect this closes: ema_window sat in DEFAULTS and NOTHING read it, so
    # every EMA claim was true of the helper and false of the run
    hist = {"learning": [0.0, 0.0, 1.0, 1.0], "forgetting": [1.0, 1.0, 0.0, 0.0]}
    pool = [{"question": "learning"}, {"question": "forgetting"}]
    flat, _u = gof.boltzmann_weights(pool, hist, temp=0.5)
    assert abs(flat[0] - flat[1]) < 1e-9, "without a window the two are tied"
    wina, _u = gof.boltzmann_weights(pool, hist, temp=0.5, ema_window=3)
    assert abs(wina[0] - wina[1]) > 1e-6, "with a window the tie must break"


def test_an_ema_spread_is_too_slow_to_be_a_retirement_signal(gof):
    # why retirement counts CONSECUTIVE dead visits instead of reading the EMA:
    # a decayed spread approaches zero geometrically. One live reading followed
    # by flat ones needs ~18 observations to round to 0.00 at a 3-step span --
    # longer than the whole run, so nothing would ever retire.
    h = [0.0, 1.0] + [0.0] * 20
    assert gof.ema_stats(h[:10], 3)[1] > 0.0, "still 0.04 after 8 flat readings"
    assert gof.ema_stats(h[:18], 3)[1] == 0.0, "only flat at 2dp by the 16th"


# --------------------------------------------------------------------------
# the buffer is only TESTABLE when visits exceed the pool (operator 2026-09-13)
# --------------------------------------------------------------------------
def test_the_buffer_is_inert_while_every_draw_is_a_first_look(gof):
    """Measured 2026-09-13: 24 visits into 240 records logged `STEM:new
    Instruction:new Math:new Code:new` on every draw, all three rounds.

    n_replays never reaches 1, so retirement, the ladder and the EMA never
    execute. That is not a tuned constant, it is an untested one -- this test
    pins the condition so the distinction cannot be lost again.
    """
    doms = ("Math", "Code", "STEM", "Instruction")
    pool = [{"question": "q%d" % i, "category": doms[i % 4]} for i in range(240)]
    vlog, hist = {}, {}
    import random as _r
    rng = _r.Random(0)
    for _ in range(6):                       # 6 steps x 4 = 24 visits
        for rec in gof.boltzmann_batch(pool, hist, 4, rng, visit_log=vlog):
            gof.record_visit(vlog, rec["question"], [0.0, 1.0, 0.0, 1.0])
    assert max(gof.n_replays(vlog, q) for q in vlog) == 0, \
        "240 records cannot be revisited in 24 draws -- the buffer is untestable"
    live, retired = gof.eligible_pool(pool, vlog, retire_after=1)
    assert retired == 0, "nothing can retire when nothing is ever seen twice"


def test_capping_the_pool_below_the_visit_budget_makes_the_buffer_live(gof):
    # same 24 visits, pool capped to 16 -> every record is seen, then resampled
    doms = ("Math", "Code", "STEM", "Instruction")
    pool = [{"question": "q%d" % i, "category": doms[i % 4]} for i in range(16)]
    vlog, hist = {}, {}
    import random as _r
    rng = _r.Random(0)
    for _ in range(6):
        for rec in gof.boltzmann_batch(pool, hist, 4, rng, visit_log=vlog):
            gof.record_visit(vlog, rec["question"], [0.0] * 4)   # all dead
    assert max(gof.n_replays(vlog, q) for q in vlog) >= 1, "records must recur"
    live, retired = gof.eligible_pool(pool, vlog, retire_after=1)
    assert retired > 0, "dead records revisited must now actually retire"


def test_the_ema_span_is_per_record_visits_not_run_steps(gof):
    """Operator 2026-09-13: "Idk if n should be based on step or n_replays".

    It is n_replays. The EMA averages ONE RECORD's visits, so the span must be
    in units of that record's observations. Today's run had 32 visits over 16
    records -- 2 visits each -- but used a span of 8 (rounds x steps), a window
    sized for observations the record never had.
    """
    def span(rounds, steps, accum, pool_size):
        return max(1, int(round(rounds * steps * accum / pool_size)))

    assert span(2, 4, 4, 16) == 2, "32 visits over 16 records is 2 each"
    assert span(2, 4, 4, 16) != 2 * 4, "NOT rounds x steps -- that was the bug"
    assert span(2, 10, 12, 240) == 1, "full coverage is one visit per record"

    # and the span actually changes what the priority says
    hist = {"q": [0.0, 0.0, 1.0, 1.0]}
    assert gof.visit_stats(hist, "q", n_window=2)[2] > \
           gof.visit_stats(hist, "q", n_window=8)[2], \
        "a short span weights the recent successes harder"
