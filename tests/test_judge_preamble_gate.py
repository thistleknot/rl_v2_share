"""The preamble gate: a mechanical post-process, deliberately NOT a rubric edit.

NO GOVERNING SPEC. Basis: operator 2026-09-15 --

> "We have hard filters (no eos = 0), abstain answer is also 0, no need to change the
>  prompt"
> "Now put this in a docstring as a post process ... Not in the prompt!"

WHAT HAPPENED. A 14-line band was added to data/synthetic_domain_rubric.md telling the
judge that a plan or preamble scores 0.0. That edit was uncommitted, changed the frozen
eval instrument mid-campaign (instructions.md 1.4, REQ-J9), and was redundant twice over:
the committed rubric ALREADY carries a 0.00-0.10 "content-free: agrees, defers, restates
the question" band, and the prompt wrapper's step 3 already says restating the question is
not stating the gold's claim. It was reverted; the rule lives here instead, in code, where
it is deterministic, testable, and costs no judge call.

THE ASYMMETRY IS THE DESIGN. A false zero on a CORRECT answer corrupts the measurement in
the direction that cannot be recovered -- it looks exactly like a model that failed. A
missed preamble is caught downstream by the rubric's content-free band. So this gate fires
only on high-confidence cases and lets ambiguous ones through to the judge.
"""
import io
import json
import sys

import pytest

from rlv2.judge import judge as J  # noqa: E402


# ------------------------------------------------- what it must catch ----
@pytest.mark.parametrize("text,label", [
    ("Thinking Process:", "a plan header with nothing after it"),
    ("Reasoning:", "any of the recognised headers"),
    ("Thinking Process:\n1. Analyze the Request\n2. Recall the formula\n3. Compute",
     "the measured 2026-09-11 shape: header + list of intentions"),
    ("1. Analyze the request\n2. Identify the base case\n- Check the formula",
     "bare enumeration with no header"),
    ("   ", "whitespace"),
])
def test_a_response_that_commits_to_nothing_is_zeroed(text, label):
    assert J.never_states_an_answer(text), label


# ------------------------------- what it must NOT catch (the expensive way) ----
@pytest.mark.parametrize("text,label", [
    ("variance - averaging high-variance trees", "a terse correct answer"),
    ("Let me compute. 1-sum(p^2) = 0.375, and 2*(3/4)*(1/4)=0.375, so they agree.",
     "OPENS with planning language then answers -- zeroing this is the bad failure"),
    ("Thinking Process:\nThe answer is 0.375 by the Gini formula.",
     "header, but the next line commits"),
    ("1. First the impurity is 0.375\nand the two-class form agrees",
     "enumerated but contains prose"),
    ("The weight ratio is below 1 for correctly classified samples.", "plain prose"),
])
def test_an_answer_that_states_something_is_never_zeroed(text, label):
    assert not J.never_states_an_answer(text), label


# ------------------------------------------------------- the wiring ----
def test_the_gate_runs_before_the_judge_is_called():
    """It must cost no judge call -- that is half the point of a mechanical gate,
    and it also means the verdict cannot depend on judge availability."""
    out = J.judge("What is the Gini impurity?", "0.375",
                  "Thinking Process:\n1. Analyze\n2. Compute", category="Math")
    assert out["score"] == 0.0
    assert out["verdict"] == "incorrect"
    assert out["ok"] is True, "this IS a measurement, not a judge failure"
    assert "never states an answer" in out["why"]


def test_it_sits_beside_the_other_mechanical_gates(  ):
    """Empty, trailing-colon stub, and preamble are one family: the LLM judge pays
    for silence unless something deterministic stops it first."""
    src = open("src/rlv2/judge/judge.py", encoding="utf-8").read()
    body = src[src.index("def judge(question"):src.index("for _ in range(max(1, retries))")]
    for guard in ("not answer.strip()", 'endswith(":")', "never_states_an_answer("):
        assert guard in body, guard


def test_the_rule_is_gated_mechanically_whether_or_not_the_rubric_says_it():
    """The rule lives in CODE. Whether the rubric ALSO states it is a separate,
    campaign-level decision recorded in configs/instrument.json.

    Settled 2026-09-15: the rubric band is restored, because it is the instrument the
    2026-09-12 campaign froze (sha 9470adb9f39935ac) and restoring it keeps that
    campaign's numbers valid. The mechanical gate stays too -- it fires BEFORE the
    judge, so it costs nothing and cannot be talked out of by a persuasive answer.

    That makes this a KNOWN DELTA from 2026-09-12, which had the band but not the
    gate. Expected effect is nil (the gate only fires where the band already scores
    0.0) but expected-nil is not identical, and instrument.json says so rather than
    letting a receipt claim parity it does not have.
    """
    cfg = json.loads(io.open("configs/instrument.json", encoding="utf-8").read())
    assert "never_states_an_answer" in " ".join(cfg["mechanical_gates"])
    assert "never_states_an_answer" in cfg["gate_delta_note"],         "the delta from the frozen campaign must be recorded, not assumed harmless"
