"""A zero loss must not be readable as a dead gradient.

NO GOVERNING SPEC. Basis: this trainer prints `loss=0.0000` on every step even
when the group is live and `adv_std=0.514`. That is arithmetic, not a defect:

    old_logps = logps.detach()            -> ratio = exp(logp - logp) = 1.0
    obj       = 1.0 * advantage
    pg_loss   = -mean(advantage)          -> 0, advantages are zero-mean

The VALUE is zero by construction; the GRADIENT is not, because `logps` carries
grad and `old_logps` does not. Reading the printed loss as "nothing is learning"
is a misdiagnosis this project already made once in session -- so the step logs
the clipped gradient norm, which `clip_grad_norm_` was computing and discarding.

That number is the discriminator between the two failure modes that look
identical in the log:
    dead group, nothing to learn        -> adv_std 0, gnorm ~0
    live group, gradient not arriving   -> adv_std > 0, gnorm ~0   <- the real bug
"""
import math

import pytest

torch = pytest.importorskip("torch")


def _pg_loss(logps, advantages, eps=0.2):
    """The file's objective, one inner epoch, transcribed."""
    old = logps.detach()
    ratio = torch.exp(logps - old)
    clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
    adv = advantages.detach()
    unclipped, clip = ratio * adv, clipped * adv
    obj = torch.where(adv >= 0, torch.minimum(unclipped, clip),
                      torch.maximum(unclipped, clip))
    return -torch.mean(obj)


def test_the_loss_is_zero_on_a_live_group_by_construction():
    """Not a bug: zero-mean advantages times a unit ratio."""
    logps = torch.tensor([-1.0, -2.0, -3.0, -4.0], requires_grad=True)
    adv = torch.tensor([1.5, 0.5, -0.5, -1.5])      # zero-mean, real spread
    assert abs(float(adv.mean())) < 1e-6, "precondition: advantages are zero-mean"
    loss = _pg_loss(logps, adv)
    assert abs(float(loss)) < 1e-6, "the printed loss is expected to read 0.0000"


def test_the_gradient_is_not_zero_on_that_same_group():
    """The whole point: zero value, non-zero gradient."""
    logps = torch.tensor([-1.0, -2.0, -3.0, -4.0], requires_grad=True)
    adv = torch.tensor([1.5, 0.5, -0.5, -1.5])
    _pg_loss(logps, adv).backward()
    gnorm = float(logps.grad.norm())
    assert gnorm > 0.1, "a live group must move weights: gnorm=%r" % gnorm
    # each rollout's gradient is its own advantage, scaled by the group size
    expected = float((adv / len(adv)).norm())
    assert math.isclose(gnorm, expected, rel_tol=1e-5), (gnorm, expected)


def test_a_dead_group_really_does_produce_no_gradient():
    """The contrast case -- identical rewards, so zero advantage and zero grad."""
    logps = torch.tensor([-1.0, -2.0, -3.0, -4.0], requires_grad=True)
    adv = torch.zeros(4)
    _pg_loss(logps, adv).backward()
    assert float(logps.grad.norm()) == 0.0


def test_gradient_scales_with_advantage_spread():
    """gnorm is informative, not a constant: a wider spread pushes harder."""
    def gnorm_for(scale):
        logps = torch.tensor([-1.0, -2.0, -3.0, -4.0], requires_grad=True)
        _pg_loss(logps, torch.tensor([1.5, 0.5, -0.5, -1.5]) * scale).backward()
        return float(logps.grad.norm())

    small, large = gnorm_for(0.1), gnorm_for(1.0)
    assert large > small * 5, (small, large)


def test_the_trainer_prints_the_norm_it_computes():
    """Guards the regression: clip_grad_norm_'s return must not be discarded."""
    import io
    src = io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    assert "_gnorm = float(torch.nn.utils.clip_grad_norm_" in src, \
        "the norm must be captured, not thrown away"
    assert "gnorm={out['grad_norm']" in src, "and it must reach the step line"
