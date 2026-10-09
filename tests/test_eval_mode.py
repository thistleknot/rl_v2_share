"""evaluate() must run with dropout OFF, and must restore the caller's mode.

Spec: playbook campaign T6 (RL HPO), defect found 2026-09-13.

The trainer calls `model.train()` three lines before its baseline eval, and
`evaluate()` did not set eval mode. Every reading in the RL sweep was therefore
taken with dropout active. Measured across four trials, every eval, before and
after training:

    RL trainer            finished_on_eos=11/16  median 45 tokens
    ReFT, same checkpoint finished_on_eos=16/16  median 31 tokens

Dropout lengthened generations and stopped them terminating; the EOS gate scored
those 5 answers 0 regardless of content. Ceiling 11/16 instead of 16, and all
four configs landed on exactly 4 -- a broken instrument reading as a tie.

The restore half matters as much: handing back a model stuck in eval mode would
silently disable dropout for the remainder of training.
"""
from __future__ import annotations

import importlib.util
import os as _o
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
        "gof_t", _o.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))),
                              "docs", "grpo_one_file.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    _s.argv = saved
    return m


class FakeModel:
    """Records the mode evaluate() ran the body under."""
    def __init__(self, training):
        self.training = training
        self.mode_during_body = None

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode


def _patch_body(gof, model, monkeypatch, boom=False):
    def _inner(m, *a, **k):
        model.mode_during_body = m.training
        if boom:
            raise RuntimeError("generation blew up")
        return 7
    monkeypatch.setattr(gof, "_evaluate_inner", _inner)


def test_body_runs_with_dropout_off_even_when_caller_was_training(gof, monkeypatch):
    m = FakeModel(training=True)
    _patch_body(gof, m, monkeypatch)
    gof.evaluate(m, None, [], {})
    assert m.mode_during_body is False, "eval ran with dropout ACTIVE"


def test_training_mode_is_restored_afterwards(gof, monkeypatch):
    m = FakeModel(training=True)
    _patch_body(gof, m, monkeypatch)
    gof.evaluate(m, None, [], {})
    assert m.training is True, "model left in eval mode disables dropout for training"


def test_a_model_already_in_eval_mode_stays_there(gof, monkeypatch):
    m = FakeModel(training=False)
    _patch_body(gof, m, monkeypatch)
    gof.evaluate(m, None, [], {})
    assert m.training is False


def test_the_mode_is_restored_even_when_generation_raises(gof, monkeypatch):
    m = FakeModel(training=True)
    _patch_body(gof, m, monkeypatch, boom=True)
    with pytest.raises(RuntimeError):
        gof.evaluate(m, None, [], {})
    assert m.training is True, "an exception must not strand the model in eval mode"


def test_the_score_is_passed_straight_through(gof, monkeypatch):
    m = FakeModel(training=True)
    _patch_body(gof, m, monkeypatch)
    assert gof.evaluate(m, None, [], {}) == 7


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
