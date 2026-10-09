"""Unit tests: src/rlv2/eval/half_epoch_eval.py — the half-epoch F1 instrument.

This callback is a MEASUREMENT device, so its failure modes are quiet and expensive: a
schedule that fires once per run reports a number with no curve; a generator that leaves
the model in eval() silently halves the rest of training; an exception that escapes kills a
30-minute run for a reading nobody needed. All three are pinned here.
"""
from __future__ import annotations
import os as _o, sys as _s

import json

import pytest

from rlv2.eval import half_epoch_eval as he
from rlv2.judge import judge as _J


@pytest.fixture(autouse=True)
def _stub_judge(monkeypatch):
    """The eval callback now scores through the judge. These tests are about the CALLBACK
    (schedule, mode restoration, report shape), not about the judge, so the judge is stubbed
    deterministically -- a live ollama call here would make the suite slow and flaky."""
    def fake(question, gold, answer, category="", **kw):
        ok = "decoded" in (answer or "")
        return {"verdict": "correct" if ok else "incorrect",
                "score": 0.9 if ok else 0.2, "quality": 0.8, "conciseness": 0.8,
                "self_sufficient": True, "why": "stub", "ok": True}
    monkeypatch.setattr(_J, "judge", fake)


# --------------------------------------------------------------------------
# doubles
# --------------------------------------------------------------------------
class FakeTok:
    padding_side = "right"
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, texts, **kw):
        class Enc(dict):
            def to(self, *_a, **_k):
                return self
        n = max(len(t.split()) for t in texts)
        return Enc(input_ids=_FakeTensor([[1] * n for _ in texts]))

    def decode(self, ids, skip_special_tokens=True):
        return "decoded"

    def convert_tokens_to_ids(self, t):
        # generate() ends the answer at the assistant turn boundary; the double must
        # answer this or every generation test dies in the stop-token lookup.
        return {"<|im_end|>": 151645, "<|endoftext|>": 151643}.get(t, -1)

    @property
    def eos_token_id(self):
        return 151643


class _FakeTensor(list):
    @property
    def shape(self):
        return (len(self), len(self[0]) if self else 0)


class FakeModel:
    device = "cpu"

    def __init__(self, raises=False):
        self.training = True
        self.raises = raises
        self.mode_log = []

    def eval(self):
        self.training = False
        self.mode_log.append("eval")

    def train(self):
        self.training = True
        self.mode_log.append("train")

    def generate(self, **kw):
        if self.raises:
            raise RuntimeError("CUDA OOM during generate")
        n = len(kw["input_ids"])
        return _FakeTensor([[1] * (kw["input_ids"].shape[1] + 3) for _ in range(n)])


class Args:
    num_train_epochs = 3


class State:
    def __init__(self, step=0, max_steps=48, epoch=0.0):
        self.global_step = step
        self.max_steps = max_steps
        self.epoch = epoch


def _rows(n=4):
    cats = ["Math", "Code", "STEM", "Instruction"]
    return [{"item_id": "i%d" % i, "category": cats[i % 4], "prompt": "q %d here" % i,
             "question": "q %d here" % i, "gold": "g", "context": "", "regime": "R3"}
            for i in range(n)]


# --------------------------------------------------------------------------
# the schedule — a curve needs >= 2 points per epoch
# --------------------------------------------------------------------------
def test_fires_twice_per_epoch_by_default():
    """48 steps over 3 epochs is 16/epoch, so half-epoch is every 8 steps."""
    cb = he.HalfEpochEval(FakeModel(), FakeTok(), eval_file="nope.jsonl")
    cb.on_train_begin(Args(), State(max_steps=48), None)
    assert cb._every == 8


def test_evals_per_epoch_is_tunable_for_short_runs():
    """A 24-step run over 3 epochs is 8 steps/epoch — at 4/epoch that is every 2 steps.
    Without this knob a short HPO trial produces too few points to read a shape."""
    cb = he.HalfEpochEval(FakeModel(), FakeTok(), eval_file="nope.jsonl",
                          evals_per_epoch=4)
    cb.on_train_begin(Args(), State(max_steps=24), None)
    assert cb._every == 2


def test_schedule_never_degenerates_to_zero_steps():
    """A tiny run must not compute `every=0` — modulo by zero would crash the trainer."""
    cb = he.HalfEpochEval(FakeModel(), FakeTok(), eval_file="nope.jsonl",
                          evals_per_epoch=8)
    cb.on_train_begin(Args(), State(max_steps=2), None)
    assert cb._every >= 1


def test_a_step_is_never_evaluated_twice():
    cb = he.HalfEpochEval(FakeModel(), FakeTok(), eval_file="nope.jsonl")
    cb.on_train_begin(Args(), State(max_steps=48), None)
    st = State(step=8, max_steps=48)
    cb.on_step_end(Args(), st, None)
    cb.on_step_end(Args(), st, None)
    assert cb._fired == {8}


def test_non_boundary_steps_do_not_fire():
    cb = he.HalfEpochEval(FakeModel(), FakeTok(), eval_file="nope.jsonl")
    cb.on_train_begin(Args(), State(max_steps=48), None)
    cb.on_step_end(Args(), State(step=7, max_steps=48), None)
    assert cb._fired == set()


# --------------------------------------------------------------------------
# generation must not corrupt training state
# --------------------------------------------------------------------------
def test_model_is_returned_to_training_mode():
    """Leaving the model in eval() silently disables dropout for the REST of the run —
    a measurement that changes what it measures."""
    m = FakeModel()
    he.generate(m, FakeTok(), ["a b", "c d"])
    assert m.training is True
    assert m.mode_log == ["eval", "train"]


def test_training_mode_restored_even_when_generation_raises():
    m = FakeModel(raises=True)
    with pytest.raises(RuntimeError):
        he.generate(m, FakeTok(), ["a b"])
    assert m.training is True, "an OOM mid-eval must not leave the model in eval()"


def test_padding_side_is_left_during_generation_then_restored():
    """Right padding makes a batched decoder generate from PAD positions and return
    truncated text — which reads as 'the model learned nothing'."""
    seen = {}
    tok = FakeTok()

    class Watch(FakeModel):
        def generate(self, **kw):
            seen["side"] = tok.padding_side
            return super().generate(**kw)

    he.generate(Watch(), tok, ["a b"])
    assert seen["side"] == "left"
    assert tok.padding_side == "right", "padding_side must be restored"


# --------------------------------------------------------------------------
# never raise into the training loop
# --------------------------------------------------------------------------
def test_missing_eval_file_is_skipped_not_fatal():
    assert he.load_eval("does_not_exist.jsonl") == []
    assert he.evaluate(FakeModel(), FakeTok(), [], "t") is None


def test_generation_failure_is_swallowed_and_reported(capsys):
    """A 30-minute run must not die for a reading nobody needed."""
    out = he.evaluate(FakeModel(raises=True), FakeTok(), _rows(), "t")
    assert out is None
    assert "SKIPPED" in capsys.readouterr().out


# --------------------------------------------------------------------------
# the report — categories WITH support, and the generations themselves
# --------------------------------------------------------------------------
def test_judge_failures_are_flagged_not_silently_counted_correct(monkeypatch, capsys):
    """A judge outage that silently read as `correct` would manufacture lift. It must be
    counted not_attempted AND announced."""
    monkeypatch.setattr(_J, "judge", lambda *a, **k: dict(_J.FAILED, why="down"))
    he.evaluate(FakeModel(), FakeTok(), _rows(4), "t")
    txt = capsys.readouterr().out
    assert "judge calls FAILED" in txt
    assert "floor, not a measurement" in txt


def test_report_shows_every_category_with_its_support(capsys):
    s = he.evaluate(FakeModel(), FakeTok(), _rows(8), "t")
    assert s is not None
    txt = capsys.readouterr().out
    for cat in ("Math", "Code", "STEM", "Instruction"):
        assert cat in txt, "category %s missing from the report" % cat
    assert s["per_category"]["Math"]["n_items"] == 2, "support must be counted per category"
    assert s["n_items"] == 8


def test_report_echoes_generated_outputs_not_only_scores(capsys):
    """A table of zeros cannot distinguish a wrong answer from a degenerate loop from an
    empty string, and those need different fixes."""
    he.evaluate(FakeModel(), FakeTok(), _rows(), "t")
    txt = capsys.readouterr().out
    assert "GENERATED" in txt
    assert "gold:" in txt and "got :" in txt


def test_history_accumulates_so_a_curve_exists():
    cb = he.HalfEpochEval(FakeModel(), FakeTok(), eval_file="nope.jsonl")
    cb.rows = _rows()
    cb.on_train_begin(Args(), State(max_steps=48), None)
    for step in (8, 16, 24):
        cb.on_step_end(Args(), State(step=step, max_steps=48, epoch=step / 16), None)
    assert len(cb.history) == 3
    assert [h["step"] for h in cb.history] == [8, 16, 24]


# --------------------------------------------------------------------------
# the live frozen eval
# --------------------------------------------------------------------------
EVAL = _o.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))),
                    "data", "rl", "r3_eval_canonical.jsonl")
EVAL_SPLIT = _o.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))),
                          "data", "rl", "r3_eval.jsonl")


@pytest.mark.skipif(not _o.path.exists(EVAL), reason="eval not built")
def test_live_eval_has_the_fields_this_callback_needs():
    rows = he.load_eval(EVAL)
    # Sized against the partitioner's eval split, never a literal. This assertion
    # read `== 20` until 2026-08-29 and passed against an orphan file that no script
    # wrote and that overlapped the training splits.
    expect = sum(1 for l in open(EVAL_SPLIT, encoding="utf-8") if l.strip())
    assert len(rows) == expect
    for r in rows:
        assert r.get("prompt") or r.get("question")
        assert r.get("item_id") or r.get("id")
        assert r.get("category")
