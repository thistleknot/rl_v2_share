"""KL weighted by the prioritised replay buffer.

NO GOVERNING SPEC. Basis: operator 2026-09-20 -- "there is a way to weight the kl
divergence based on the prioritized replay buffer, I want to try that during full".

THE ANCHOR SHOULD BE STRONGEST WHERE THERE IS NOTHING TO LEARN. A flat KL penalty
pulls the policy back toward the reference equally on every record, INCLUDING the
contested ones the replay buffer drew RL to. That is the regulariser fighting the
objective: a record is selected BECAUSE its rollouts straddle, and the KL then
penalises exactly the movement straddle exists to enable.

    w_i    = 1 - priority_i          priority is E[p(1-p)] / 0.25, already on [0,1]
    beta_i = beta * w_i

    priority 1.0  maximally contested -> w 0.0  free to move
    priority 0.0  settled or dead     -> w 1.0  fully anchored

Same logic as prioritised replay itself, applied to the regulariser rather than the
draw: protect what is already right (forgetting is what KL exists to prevent), step
out of the way where the gradient is.
"""
import importlib.util
import sys

import pytest


@pytest.fixture(scope="module")
def gof():
    spec = importlib.util.spec_from_file_location("gof_klp", "docs/grpo_one_file.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["gof_klp"] = m
    saved, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(m)
    sys.argv = saved
    return m


def test_every_pool_record_is_stamped_with_its_priority(gof):
    """The value must ride on the record, or the loss cannot see it without
    threading the sampler's state through every call."""
    pool = [{"question": "q%d" % i} for i in range(5)]
    gof.boltzmann_weights(pool, {}, temp=1.0)
    assert all("_priority" in r for r in pool)
    assert all(0.0 <= r["_priority"] <= 1.0 for r in pool)


def test_an_unvisited_record_cannot_produce_a_negative_weight(gof):
    """_UNVISITED_PRIOR is deliberately ABOVE _PRIOR_MAX so unknown records outrank
    every measured one. Unclamped, 1 - priority would go NEGATIVE and flip the KL
    anchor into a repulsor -- pushing the policy AWAY from the reference on exactly
    the records it knows least about."""
    assert gof._UNVISITED_PRIOR > gof._PRIOR_MAX
    pool = [{"question": "never_seen"}]
    gof.boltzmann_weights(pool, {}, temp=1.0)
    assert pool[0]["_priority"] <= 1.0
    assert 1.0 - pool[0]["_priority"] >= 0.0


def test_a_contested_record_is_anchored_less_than_a_settled_one(gof):
    """THE WHOLE POINT. A record whose rollouts straddle the floor carries the
    gradient RL was drawn to; a record answered identically every time carries none.
    The first must be freer to move than the second."""
    hist = {"contested": [0.95, 0.20, 0.90, 0.15],   # straddles -> high priority
            "settled": [0.98, 0.97, 0.99, 0.98]}     # uniform   -> low priority
    pool = [{"question": "contested"}, {"question": "settled"}]
    gof.boltzmann_weights(pool, hist, temp=1.0, floor=0.7)
    w_contested = 1.0 - pool[0]["_priority"]
    w_settled = 1.0 - pool[1]["_priority"]
    assert w_contested < w_settled, (w_contested, w_settled)


def test_priority_is_refreshed_not_cached(gof):
    """A record's priority moves as its history fills. An anchor computed from a
    stale priority protects the wrong records."""
    pool = [{"question": "q"}]
    gof.boltzmann_weights(pool, {}, temp=1.0)
    first = pool[0]["_priority"]
    gof.boltzmann_weights(pool, {"q": [0.9, 0.9, 0.9, 0.9]}, temp=1.0, floor=0.7)
    assert pool[0]["_priority"] != first


def test_the_weighting_is_off_by_default(gof):
    """It changes what --beta MEANS, so every beta measured under the flat form
    would otherwise silently become incomparable."""
    assert gof.DEFAULTS.get("kl_priority_weighted", False) in (False, None)


def test_the_weight_reaches_the_loss_and_the_row(gof):
    import inspect
    src = inspect.getsource(gof)
    assert 'cfg["beta"] * _kl_w' in src, "both estimators must use the weight"
    assert src.count('cfg["beta"] * _kl_w') == 2, "k1 AND k3"
    assert '"kl_weight": float(_kl_w)' in src, "surface it or it cannot be checked"
