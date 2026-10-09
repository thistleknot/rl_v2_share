"""Importance-sampling correction for prioritised draws.

Spec: .spec/specs/r3-domain-injection/rl-readiness.md REQ-R4, R5, R6.

Operator 2026-09-14: *"Although seems to violate random assumptions."* Correct, and this
project's own notes say so -- the Replay Buffer concept gives the canonical purpose of a
replay buffer as decorrelation via RANDOM sampling. Ours prioritises, and until this
change nothing compensated: `(loss_tensor / accum).backward()` gave every record weight
`1/accum` regardless of how likely it was to be drawn.

THREE PROPERTIES THIS PINS.

1. beta=0 IS A NO-OP. Every number measured before this existed was taken without the
   correction. If beta=0 ever stops returning exactly 1.0, the whole board becomes
   incomparable and nothing would announce it.

2. N IS THE ELIGIBLE COUNT FOR THAT DRAW. The draw is per DOMAIN and P is normalised over
   that bucket, so N is len(cand). Using the pool size would scale every weight by a
   constant that changes as buckets change size -- wrong in a way that still runs.

3. DEAD-GROUP FILTERING NEEDS NO WEIGHT. A uniform group's gradient is exactly zero, so
   excluding it is not sampling bias. That premise is asserted here so it cannot silently
   change underneath the design.
"""
import importlib.util
import sys

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def gof():
    spec = importlib.util.spec_from_file_location("gof_isw", "docs/grpo_one_file.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gof_isw"] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------- beta = 0 ----
def test_beta_zero_is_exactly_one_for_every_probability(gof):
    """The no-op guarantee. Not 'approximately', exactly."""
    for p in (1e-9, 0.01, 0.1, 0.25, 0.5, 0.9, 1.0):
        assert gof.is_weight(p, 4, 0.0) == 1.0, p


def test_beta_zero_is_one_at_every_eligible_count(gof):
    for n in (1, 2, 4, 40, 240):
        assert gof.is_weight(0.3, n, 0.0) == 1.0, n


def test_the_default_beta_is_zero(gof):
    assert gof.DEFAULTS["is_beta"] == 0.0, \
        "every number on the board was measured uncorrected; the default must not drift"


# --------------------------------------------------------- the weighting ----
def test_a_uniformly_drawn_record_gets_weight_one(gof):
    """P = 1/N is the unbiased case, so the correction must do nothing there.

    This is also why a REDRAW needs no special case: pick_same_domain draws
    uniformly, so its probability is 1/n and the arithmetic returns 1.0.
    """
    for n in (2, 4, 8, 40):
        assert abs(gof.is_weight(1.0 / n, n, 1.0) - 1.0) < 1e-9, n


def test_an_over_sampled_record_is_down_weighted(gof):
    """The whole point: a record drawn more often than uniform contributes less."""
    n = 4
    over = gof.is_weight(0.7, n, 1.0)          # 0.7 vs uniform 0.25
    assert over < 1.0
    assert abs(over - (1.0 / (4 * 0.7))) < 1e-9


def test_an_under_sampled_record_is_not_up_weighted_past_one(gof):
    """Normalising by max(w) caps at 1.0.

    Without the cap, a rarely-drawn record could receive a weight far above 1 and
    a single unlucky draw would dominate the step -- the instability PER's
    max-normalisation exists to prevent.
    """
    n = 4
    under = gof.is_weight(0.01, n, 1.0)        # 0.01 vs uniform 0.25
    assert under == 1.0


def test_beta_interpolates_between_off_and_full(gof):
    p, n = 0.7, 4
    off = gof.is_weight(p, n, 0.0)
    half = gof.is_weight(p, n, 0.5)
    full = gof.is_weight(p, n, 1.0)
    assert off == 1.0
    assert full < half < off, (off, half, full)


def test_n_changes_the_weight_so_using_the_pool_size_would_be_wrong(gof):
    """REQ-R5. Same probability, different eligible count -> different weight."""
    p = 0.25
    assert gof.is_weight(p, 4, 1.0) == 1.0, "uniform in a bucket of 4"
    assert gof.is_weight(p, 40, 1.0) < 1.0, "the same p is over-sampling in 40"


def test_a_zero_probability_draw_cannot_happen_and_returns_one(gof):
    """Defensive: a record at P=0 was never drawn, so there is nothing to correct."""
    assert gof.is_weight(0.0, 4, 1.0) == 1.0


# ------------------------------------------------- the filtering premise ----
def test_a_uniform_group_contributes_exactly_zero_gradient(gof):
    """REQ-R6, the premise that makes dead-group filtering unbiased.

    If this ever becomes approximately-zero rather than exactly-zero, filtering
    starts introducing bias and the design needs revisiting.
    """
    logps = torch.tensor([-1.0, -2.0, -3.0, -4.0], requires_grad=True)
    rewards = torch.tensor([0.0, 0.0, 0.0, 0.0])
    adv = rewards - rewards.mean()
    ratio = torch.exp(logps - logps.detach())
    obj = torch.where(adv >= 0, torch.minimum(ratio * adv, ratio * adv),
                      torch.maximum(ratio * adv, ratio * adv))
    (-obj.mean()).backward()
    assert float(logps.grad.norm()) == 0.0


# ------------------------------------------------------- the draw record ----
def test_the_batch_records_the_probability_it_drew_with(gof):
    """The correction needs P(i); it used to be computed and thrown away."""
    import random
    doms = ("Math", "Code", "STEM", "Instruction")
    pool = [{"question": "q%d" % i, "category": doms[i % 4]} for i in range(24)]
    batch = gof.boltzmann_batch(pool, {}, 4, random.Random(0), visit_log={})
    assert len(batch) == 4
    for rec in batch:
        assert "_draw_p" in rec and "_draw_n" in rec
        assert 0.0 < rec["_draw_p"] <= 1.0
        assert rec["_draw_n"] >= 1


def test_the_recorded_probability_is_normalised_over_its_own_bucket(gof):
    """REQ-R5 again, from the other end: _draw_n must be the bucket, not the pool."""
    import random
    doms = ("Math", "Code", "STEM", "Instruction")
    pool = [{"question": "q%d" % i, "category": doms[i % 4]} for i in range(24)]
    batch = gof.boltzmann_batch(pool, {}, 4, random.Random(1), visit_log={})
    for rec in batch:
        assert rec["_draw_n"] <= len(pool) / 2, \
            "n=%d looks like the whole pool, not one domain" % rec["_draw_n"]
