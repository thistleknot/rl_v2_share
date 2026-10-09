"""Where UNKNOWN sits in the ladder depends on whether anything is known.

NO GOVERNING SPEC. Basis: operator 2026-09-15 -- "I want gotchas addressed".

THE DEFECT, MEASURED. `rank_normalise` put every unvisited record at rank 1.0, the top.
That was correct when NOTHING had a prior: the whole pool ties, splits its mass, and
dilutes as it is consumed, so coverage stays an invariant.

With a persistent ledger it inverts. Measured 2026-09-15 on the RL pool: 79 records
carried priors, 161 did not, and the 161 outranked all 79. An arm draws ~12 records, so
the unknown bloc never dilutes:

    picked from prior: 0 of 4
    [boltzmann] T=0.57 STEM:new Instruction:new Math:new Code:new
    live rate ~25% against the ledger's measured 68%

The ledger loaded, matched on every key, and was outranked into irrelevance. "Prior loads"
and "prior steers" are different claims and only the first was ever checked.

AFTER: unknown sits at the MEDIAN of known ranks -- because unknown now means "we have
evidence about others and none about this one". Known-live outrank it, known-dead fall
below it, coverage still happens, just after the records we know can teach.

    from the prior: 51 of 120 (42%)      known-LIVE: 47 (39%)
"""
import importlib.util
import sys

import pytest


@pytest.fixture(scope="module")
def gof():
    spec = importlib.util.spec_from_file_location("gof_rank", "docs/grpo_one_file.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["gof_rank"] = m
    saved, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(m)
    sys.argv = saved
    return m


def test_unknown_outranks_a_known_dead_record(gof):
    """Unknown is not bad -- a record we have never tried still deserves a turn
    ahead of one measured to teach nothing."""
    ranks = gof.rank_normalise([0.0, None, 0.5])
    assert ranks[1] > ranks[0]


def test_unknown_does_not_outrank_a_known_live_record(gof):
    """The whole defect. A record measured to straddle must beat one we know
    nothing about, or the ledger is decoration."""
    ranks = gof.rank_normalise([0.0, None, 0.5])
    assert ranks[2] > ranks[1], "a high-spread record must outrank unknown"


def test_with_nothing_known_unknown_is_still_top(gof):
    """The original behaviour, preserved exactly: when no record has a prior,
    every one is unknown, they tie at 1.0, split their mass and dilute."""
    ranks = gof.rank_normalise([None, None, None])
    assert ranks == [1.0, 1.0, 1.0]


def test_a_lone_unknown_among_knowns_sits_in_the_middle(gof):
    known = [0.0, 0.1, 0.2, 0.8, 0.9]
    ranks = gof.rank_normalise(known + [None])
    unknown = ranks[-1]
    assert min(ranks[:-1]) < unknown < max(ranks[:-1])


def test_the_draw_actually_lands_on_the_prior(gof):
    """The end-to-end claim, not the unit one: `prior loads` and `prior steers`
    are different, and only the first was ever verified."""
    import random
    hist = {"q%d" % i: ([0.0, 1.0] if i % 2 else [0.0, 0.0]) for i in range(40)}
    pool = [{"question": "q%d" % i, "category": ["Math", "Code", "STEM",
                                                 "Instruction"][i % 4]}
            for i in range(120)]          # 40 known, 80 unknown -- the real shape
    hits = 0
    for seed in range(25):
        for r in gof.boltzmann_batch(pool, hist, 4, random.Random(seed), visit_log={}):
            hits += r["question"] in hist
    rate = hits / (25 * 4)
    assert rate > 0.30, "prior-backed records drawn only %.0f%% of the time" % (100 * rate)


def test_the_sampler_runs_in_round_zero_when_a_prior_was_loaded():
    """The condition was `rd > 0`: the Boltzmann draw only ran from round 1,
    because round 0 "has no history yet". A persistent ledger falsifies that --
    measured 2026-09-15, 79 records carried priors before step 1 and round 0 took
    the flat branch anyway, so NO [boltzmann] line appeared in the log at all.

    With `--rounds 3 --steps 1` round 0 is a third of an HPO arm.

    Cold start is unchanged: with an empty reward_history the condition is false in
    round 0, exactly as before.
    """
    import io as _io
    src = _io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    assert 'if rd > 0 and cfg.get("boltzmann", True):' not in src
    assert 'cfg.get("boltzmann", True) and (rd > 0 or reward_history)' in src
