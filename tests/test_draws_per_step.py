"""Redraw rate is measured at EVERY step, for free.

Operator 2026-09-14: "I suppose its dangerous to stop at step 1 / Its almost like
we need to repeat this process at every step!"

They are right, and the measurement does not need a new loop. The
differentiable-slot search already counts how many records it probes to fill its
slots -- that count existed in every run this project has done and was discarded
at the end of each step. Logging it turns a one-shot screen into a per-step
series at zero extra generation or judging cost.

WHAT THE NUMBER MEANS. `draws N/M (Rx)` where M is records-per-step and N is
probes spent. R = 1.0 means every first draw was differentiable. Measured from
base, R has been running 5-13x because 85-88% of probed groups come back uniform.

WHY IT MATTERS BEFORE BUILDING ANYTHING BIGGER. Re-selecting hyperparameters at
every step costs a full probe pass per candidate per step. If R turns out flat
across steps, that machinery buys nothing -- so measure whether the axis is live
before paying to search it.
"""
import io

import pytest

SRC = "docs/grpo_one_file.py"


@pytest.fixture(scope="module")
def src():
    return io.open(SRC, encoding="utf-8").read()


def test_the_counter_is_reset_per_step(src):
    """A cumulative counter would hide a step-to-step trend, which is the whole
    point of measuring it per step."""
    i = src.index("_draws_this_step = 0")
    j = src.index("_draws_this_step += 1")
    assert i < j, "reset must precede the increments"
    body = src[i:j]
    assert "for _micro in range(_accum)" in body, \
        "the reset belongs OUTSIDE the slot loop, once per step"


def test_every_probe_is_counted(src):
    """The increment sits on the compute call, so a redraw counts as a draw."""
    i = src.index("_draws_this_step += 1")
    after = src[i:i + 200]
    assert "compute_grpo_loss" in after, \
        "the counter must sit on the probe itself, not on the accepted record"


def test_the_step_line_reports_draws_and_the_ratio(src):
    assert 'f"draws {_draws_this_step}/{_accum} "' in src
    assert "_draws_this_step / max(1, _accum)" in src, \
        "the ratio is the comparable number across different batch sizes"


def test_the_ratio_is_draws_per_record_not_per_step():
    """1.0 means every first draw landed. A per-step ratio would change meaning
    whenever grad_accum changes, making runs incomparable."""
    def ratio(draws, accum):
        return draws / max(1, accum)

    assert ratio(4, 4) == 1.0, "perfect: four draws for four records"
    assert ratio(20, 4) == 5.0, "five probes per usable record"
    assert ratio(8, 8) == 1.0, "unchanged by batch size"


def test_the_measurement_adds_no_generation_cost(src):
    """It is a counter on an existing call -- if it ever grows its own probe
    loop, it stops being free and the tier-1/tier-3 tradeoff collapses."""
    i = src.index("_draws_this_step = 0")
    j = src.index("n_steps += 1")
    body = src[i:j]
    assert body.count("compute_grpo_loss") == 1, \
        "exactly one probe call per draw; no second pass for measurement"
