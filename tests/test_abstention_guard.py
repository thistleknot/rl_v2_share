"""Abstention is WRONG on the r3 lane, and the system prompt stays.

Spec: .spec/specs/r3-domain-injection/system-prompt-placement.md.
Operator 2026-09-12: "to keep things consistent. We'll just mark abstention wrong
since we know all are answerable and keep system prompt."

R3 has ZERO abstain items (0/16 eval, 0/240 train have an empty gold), but the
system prompt invites abstention. The prompt is KEPT -- editing it changes the
instrument and discards comparability with every historical number -- so the
invitation is neutralised in scoring instead.

The guard must be NARROW. A real answer may legitimately discuss what cannot be
determined; flagging those would silently zero correct work, which is the more
expensive failure of the two.
"""
from __future__ import annotations

import os as _o
import sys as _s

for _d in ("docs",):
    _p = _o.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))), _d)
    _p not in _s.path and _s.path.insert(0, _p)

import pytest  # noqa: E402

from rlv2.judge.reward import _is_abstention, _is_degenerate  # noqa: E402
from rlv2.eval.half_epoch_eval import SYSTEM, chat_wrap  # noqa: E402


# --------------------------------------------------------------------------
# the prompt is unchanged
# --------------------------------------------------------------------------
def test_the_system_prompt_is_present_in_the_rollout():
    assert "<|im_start|>system" in chat_wrap(None, "What is gini impurity?")


def test_the_system_prompt_is_the_one_cpt_rendered_with():
    # every historical number was measured under this exact text
    assert SYSTEM == ("You are a helpful assistant. Answer the question directly "
                      "and concisely. If the question cannot be answered from "
                      "available information, say so clearly.")


# --------------------------------------------------------------------------
# declining IS the answer -> abstention
# --------------------------------------------------------------------------
def test_a_bare_refusal_is_an_abstention():
    for t in ("I cannot answer this question.",
              "This cannot be determined from the available information.",
              "I don't know.",
              "There is not enough information to answer."):
        assert _is_abstention(t) is True, t


def test_a_refusal_that_opens_the_answer_is_caught_even_if_long():
    t = ("The question cannot be answered from the information given. " +
         "Additional context would be required. " * 12)
    assert _is_abstention(t) is True


# --------------------------------------------------------------------------
# a real answer that MENTIONS uncertainty is not an abstention
# --------------------------------------------------------------------------
def test_a_real_answer_discussing_indeterminacy_is_not_flagged():
    t = ("Gini impurity measures node purity. The best split cannot be determined "
         "from impurity alone, so CART evaluates every candidate threshold and "
         "picks the one with the largest weighted impurity decrease across the "
         "two child nodes, which is why the algorithm is greedy rather than "
         "globally optimal over the tree structure.")
    assert _is_abstention(t) is False


def test_an_ordinary_correct_answer_is_not_flagged():
    assert _is_abstention("The number of candidate features becomes log2(64) = 8.") is False


def test_a_short_answer_with_no_refusal_is_not_flagged():
    assert _is_abstention("0.72") is False


# --------------------------------------------------------------------------
# the two guards stay in their lanes
# --------------------------------------------------------------------------
def test_empty_is_degenerate_not_abstention():
    # empty is _is_degenerate's job; double-counting would hide which fired
    assert _is_degenerate("") is True
    assert _is_abstention("") is False


def test_case_and_whitespace_do_not_evade_the_guard():
    assert _is_abstention("   I CANNOT ANSWER   this.  ") is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_the_head_window_does_not_swallow_a_real_answer():
    # regression: an 80-char window flagged this, because a legitimate answer
    # reaches "cannot be determined" at char ~52. Only an OPENING refusal counts.
    t = ("Gini impurity measures node purity. The best split cannot be determined "
         "from impurity alone, so CART evaluates every candidate threshold.")
    assert t.lower().find("cannot be determined") > 40
    assert _is_abstention(t) is False


def test_a_later_refusal_clause_never_flags_a_real_answer():
    # both earlier rules zeroed this: the marker is at char 51 AND the answer is
    # only 20 words, so neither a char-window nor a word-count rule survives
    t = ("Gini impurity measures node purity. The best split cannot be determined "
         "from impurity alone, so CART evaluates every candidate threshold.")
    assert _is_abstention(t) is False


def test_a_refusal_in_the_first_sentence_flags_regardless_of_length():
    t = ("The question cannot be answered from the information given, and no "
         "further detail is available anywhere in the provided material here.")
    assert _is_abstention(t) is True
