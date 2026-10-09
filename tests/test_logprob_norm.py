"""Sequence log-prob: summed (length-scaled) vs per-token (TRL-style).

NO GOVERNING SPEC. Basis: measured 2026-09-13 in an HPO trial's own step line,

    [round 1 step 1] loss=0.0000 reward=0.372 adv_std=0.444 gnorm=137 | dead 0/4

against `clip_grad_norm_(model.parameters(), 1.0)`. A raw norm of 137 into a clip
of 1.0 means the update is renormalized ~137x, so the step is `lr x unit-vector`:
the advantage's MAGNITUDE contributes nothing, only its direction, and lr is the
only knob left alive. That is exactly the cleanly-monotone lr response every
sweep this session produced.

The cause is `selected[prompt_len - 1:].sum()` -- a SUMMED sequence log-prob, so
the gradient scales with answer length. The file already noted this for the KL
term ("TRL normalizes per token, this file sums") and never applied the same
reading to the policy gradient.

DEFAULT STAYS "sum". Every number on this board was measured under it; flipping
the default silently would make the history incomparable.
"""
import importlib.util
import sys

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def gof():
    spec = importlib.util.spec_from_file_location("gof_lp",
                                                  "docs/grpo_one_file.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gof_lp"] = mod
    spec.loader.exec_module(mod)
    return mod


class _FixedLogits(torch.nn.Module):
    """A model whose per-token log-probs are known, so the math is checkable."""

    def __init__(self, vocab=8, seq=12):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(seq, vocab))

    def forward(self, ids):
        return type("O", (), {"logits": self.w.unsqueeze(0)[:, :ids.shape[-1]]})()


def _score(gof, model, n_answer_tokens, norm):
    prompt_len = 2
    ids = torch.arange(prompt_len + n_answer_tokens) % 8
    return gof.sequence_logprob(model, ids, prompt_len, {"logprob_norm": norm})


def test_the_default_is_still_sum(gof):
    assert gof.DEFAULTS["logprob_norm"] == "sum", \
        "today's board was measured under sum; the default must not drift"


def test_sum_scales_with_answer_length(gof):
    """The defect, stated as a measurement."""
    m = _FixedLogits()
    short = float(_score(gof, m, 2, "sum"))
    long = float(_score(gof, m, 8, "sum"))
    assert abs(long / short - 4.0) < 0.01, (short, long)


def test_mean_does_not_scale_with_answer_length(gof):
    """The fix, stated as the same measurement."""
    m = _FixedLogits()
    short = float(_score(gof, m, 2, "mean"))
    long = float(_score(gof, m, 8, "mean"))
    assert abs(long - short) < 1e-5, (short, long)


def test_mean_is_the_sum_divided_by_the_answer_length(gof):
    m = _FixedLogits()
    n = 6
    assert abs(float(_score(gof, m, n, "sum")) / n
               - float(_score(gof, m, n, "mean"))) < 1e-5


def test_the_gradient_magnitude_follows_the_same_ratio(gof):
    """Norm, not just value -- the 137 was a GRADIENT norm."""
    def gnorm(n, norm):
        m = _FixedLogits()
        _score(gof, m, n, norm).backward()
        return float(m.w.grad.norm())

    assert gnorm(8, "sum") > gnorm(2, "sum") * 1.5, "summed grad must grow"
    assert gnorm(8, "mean") < gnorm(2, "mean"), "per-token grad must not"


def test_an_empty_answer_does_not_divide_by_zero(gof):
    """A zero-length answer is real: the eos-gate exists because of them."""
    m = _FixedLogits()
    v = _score(gof, m, 0, "mean")
    assert torch.isfinite(v).all()


def test_an_unknown_setting_falls_back_to_sum(gof):
    """Fail toward the measured default, never toward a silent new behaviour."""
    m = _FixedLogits()
    assert abs(float(_score(gof, m, 4, "nonsense"))
               - float(_score(gof, m, 4, "sum"))) < 1e-6
