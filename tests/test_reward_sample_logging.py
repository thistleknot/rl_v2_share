"""Rollout generations must show what the reward actually PAID, and why.

NO GOVERNING SPEC. Two operator rules, read together:

  2026-09-11 -- "just automatically reject generations or better yet they fail if
                 they dont have eos ... kind of like rlvr"
  2026-09-13 -- "We need to see rollout generations"

THE DEFECT THIS FILE PINS. One HPO trial emitted 57 `[reward] n/4 correct` lines
and not one showed what the model wrote. The first fix logged the JUDGE's score
from inside `score_group` -- which runs BEFORE the EOS gate. So a rollout printed
as 0.00 could be either:

    a wrong answer                     -> the reward is working, the model is not
    a generation that never stopped    -> mechanically zeroed, content irrelevant

Opposite defects, opposite fixes, and neither the aggregate count nor a
pre-gate score can tell them apart. The sample line therefore carries all four
facts that decide the reward: judge score, amount PAID, stop-token status, and
the budget the generation had.
"""
import importlib.util
import sys

import pytest


@pytest.fixture(scope="module")
def gof():
    spec = importlib.util.spec_from_file_location(
        "gof_rs", "docs/grpo_one_file.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gof_rs"] = mod
    spec.loader.exec_module(mod)
    return mod


class _Tok:
    """Minimal tokenizer stand-in: one token per whitespace-separated word."""

    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        n = max(1, len((text or "").split()))
        return type("E", (), {"input_ids": list(range(n))})()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join("w" for _ in ids)

    def apply_chat_template(self, msgs, **kw):
        return " ".join(m.get("content", "") for m in msgs)


def _run(gof, monkeypatch, answers, finished, capsys, cfg_over=None):
    """Drive compute_grpo_loss's logging path with scripted rollouts."""
    cfg = dict(gof.DEFAULTS)
    cfg["num_rollouts"] = len(answers)
    cfg.update(cfg_over or {})

    seq = list(zip(answers, finished))
    calls = {"i": 0}

    def fake_sample(model, tokenizer, prompt, c, budget=None):
        text, ok = seq[calls["i"]]
        calls["i"] += 1
        return ([1, 2, 3], 1, text, ok)

    monkeypatch.setattr(gof, "sample_response", fake_sample)
    monkeypatch.setattr(gof, "build_prompt", lambda r, t: "prompt")
    monkeypatch.setattr(gof, "answer_budget", lambda r, t, c: 53)

    def fake_score(question, gold, ans, c, offload=None, stub=False):
        raw = [0.15 if a else 0.0 for a in ans]
        return list(raw), list(raw)

    monkeypatch.setattr(gof, "score_group", fake_score)

    record = {"question": "q?", "gold": "g"}
    try:
        gof.compute_grpo_loss(None, None, _Tok(), record, cfg, stub=True)
    except Exception:
        pass                      # the loss math needs a real model; logging ran
    return capsys.readouterr().out


def test_a_no_eos_rollout_is_labelled_NO_EOS(gof, monkeypatch, capsys):
    """The whole point: 0.00 from a rambler must not look like 0.00 from a miss."""
    out = _run(gof, monkeypatch, ["rambling answer"] * 4, [False] * 4, capsys)
    assert "NO-EOS" in out, "a generation that never stopped must say so"


def test_a_finished_rollout_is_labelled_eos(gof, monkeypatch, capsys):
    out = _run(gof, monkeypatch, ["clean answer"] * 4, [True] * 4, capsys)
    sample = [ln for ln in out.splitlines() if ln.startswith("[reward-sample:")]
    assert sample
    assert all(" eos " in ln for ln in sample)
    assert all("NO-EOS" not in ln for ln in sample)


def test_the_paid_reward_is_shown_separately_from_the_judge_score(gof, monkeypatch,
                                                                 capsys):
    """judge=0.15 paid=0.00 is the signature of an EOS-gated rollout.

    Logging only the judge's opinion hides the gate entirely; logging only the
    paid amount hides WHY it was zero.
    """
    out = _run(gof, monkeypatch, ["answer"] * 4, [False] * 4, capsys)
    sample = [ln for ln in out.splitlines() if ln.startswith("[reward-sample:")]
    assert sample
    assert any("judge=0.15" in ln and "paid=0.00" in ln for ln in sample), sample


def test_the_budget_is_shown_so_truncation_is_attributable(gof, monkeypatch,
                                                           capsys):
    """Without the budget, NO-EOS cannot be blamed on anything in particular."""
    out = _run(gof, monkeypatch, ["answer"] * 4, [False] * 4, capsys)
    assert "budget=53" in out


def test_an_empty_generation_reports_zero_length(gof, monkeypatch, capsys):
    """Guards the old defect: the judge once paid full reward for empty answers."""
    out = _run(gof, monkeypatch, ["", "", "", ""], [True] * 4, capsys)
    assert "len=0" in out


def test_both_extremes_print_even_when_the_group_is_uniform(gof, monkeypatch,
                                                            capsys):
    """A dead group is exactly the case worth reading.

    Only the text distinguishes four genuine attempts landing the same way from
    one degenerate output repeated four times.
    """
    out = _run(gof, monkeypatch, ["same"] * 4, [True] * 4, capsys)
    assert out.count("[reward-sample:") == 2


def test_newlines_are_flattened_to_keep_one_rollout_on_one_line(gof, monkeypatch,
                                                                capsys):
    out = _run(gof, monkeypatch, ["first\nsecond\nthird"] * 2, [True] * 2, capsys)
    for ln in out.splitlines():
        if ln.startswith("[reward-sample:"):
            assert "first second third" in ln


def test_the_length_is_the_generation_not_the_preview(gof, monkeypatch, capsys):
    """len= must describe the GENERATION.

    Reporting the 160-char preview's length would make every long rollout read
    as 160 and hide runaway decoding -- and it already caused one misread, where
    a preview cut mid-word was reported as model truncation.
    """
    out = _run(gof, monkeypatch, ["x" * 500] * 2, [True] * 2, capsys)
    assert "len=500" in out
