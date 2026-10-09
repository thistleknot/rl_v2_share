"""Unit tests: src/rlv2/eval/policy_eval.py — the turn-boundary trim behind the stop-token fix.

Measured 2026-08-28: 48 of 48 rubric generations ran past the assistant turn into the next
one, and `skip_special_tokens=True` deleted the `<|im_end|>` marker, so the leak was
invisible in the text — it read as "...answer. assistant   <a second, unrelated answer>".
The judge scored the transcript. Every number in data/rubric_scored_pairs.txt was void.

The real fix is `eos_token_id=_stop_ids(tok)` on the generate call. `_cut_at_turn` is the
belt-and-braces behind it, for checkpoints whose tokenizer lacks those tokens. Its risk is
the opposite of the bug: trimming too eagerly would silently truncate a CORRECT answer that
happens to use the word "user". Both directions are pinned here.
"""
from __future__ import annotations
import os as _o, sys as _s

import pytest

from rlv2.eval import policy_eval as pe


# ── it CUTS a real turn leak ───────────────────────────────────────────────────────
# Each case is a verbatim shape observed in data/rubric_variance.json.
@pytest.mark.parametrize("raw,expected", [
    # the dominant observed signature: role word, run of spaces, second answer
    ("A split is permitted only if X differs. assistant   The code then sums the residuals",
     "A split is permitted only if X differs."),
    # markers intact (tokenizer that does not strip them)
    ("F_0 = 0 and each w_i = 0.01.<|im_end|><|im_start|>assistant\nSecond answer",
     "F_0 = 0 and each w_i = 0.01."),
    # role word on its own line
    ("Bagging reduces variance.\nassistant\nmore junk", "Bagging reduces variance."),
    # a bare end marker with nothing after it
    ("Gini impurity is 1 - sum(p_i^2).<|im_end|>", "Gini impurity is 1 - sum(p_i^2)."),
    # 'user' turn rather than 'assistant'
    ("No, they should not proceed.  user   next question", "No, they should not proceed."),
])
def test_cuts_turn_leak(raw, expected):
    assert pe._cut_at_turn(raw) == expected


# ── it does NOT cut a legitimate answer ────────────────────────────────────────────
# Over-trimming is the failure mode that would be mistaken for a bad model.
@pytest.mark.parametrize("raw", [
    "The user provides a labelled dataset and the model fits it.",
    "F_0 = 0 (zero baseline), and each w_i = 1/100 = 0.01.",
    "max_depth limits tree depth; min_samples_leaf sets the minimum node size.",
    "A system of weak learners is combined additively.",
    "Boosting fits residuals sequentially, so one user-supplied seed changes nothing.",
])
def test_keeps_clean_answer(raw):
    assert pe._cut_at_turn(raw) == raw.strip()


def test_single_space_before_role_word_is_not_a_seam():
    """One space is ordinary prose; the leak signature is 2+ spaces or a newline.

    Requiring the wider gap is what lets 'the user provides...' survive while
    '...answer. assistant   junk' is cut.
    """
    assert pe._cut_at_turn("Ask the user for a value.") == "Ask the user for a value."


def test_empty_and_whitespace_are_safe():
    assert pe._cut_at_turn("") == ""
    assert pe._cut_at_turn("   \n  ") == ""


def test_stop_ids_are_wired_into_generation():
    """The trim is the backstop; eos_token_id is the actual fix.

    Pins that both generation paths request stop ids, so a refactor that drops them fails
    here rather than silently reintroducing 48/48 contamination.
    """
    import inspect
    for fn in (pe._generate_one, pe.generate_greedy_batch):
        src = inspect.getsource(fn)
        assert "eos_token_id=_stop_ids(" in src, f"{fn.__name__} lost its stop ids"
        assert "_cut_at_turn(" in src, f"{fn.__name__} lost the turn trim"
