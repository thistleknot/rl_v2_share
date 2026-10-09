"""tests/test_harvest_sft.py — the ReST harvest's guards and selection logic.

Spec: docs/plans/break-the-9-advanced-rl-ladder.md rung R2 · Task: tasks.md #40.
"""
import sys
from pathlib import Path


import pytest

from rlv2.train.rl import harvest_sft as H


def test_eval_leakage_tripwire_fails_closed_on_a_planted_question():
    pool = [{"id": "1", "question": "What is bagging?", "gold": "g"},
            {"id": "2", "question": "THE EVAL QUESTION", "gold": "g"}]
    eval_rows = [{"question": "THE EVAL QUESTION", "gold": "g"}]
    with pytest.raises(AssertionError, match="EVAL LEAKAGE"):
        H.assert_no_eval_leakage(pool, eval_rows)


def test_tripwire_passes_on_disjoint_pools():
    H.assert_no_eval_leakage([{"id": "1", "question": "q1"}],
                             [{"question": "q2"}])


def test_select_best_prefers_highest_score_then_shortest():
    samples = [
        {"completion": "a long correct answer indeed", "score": 0.95},
        {"completion": "short right", "score": 0.95},
        {"completion": "wrong", "score": 0.10},
        {"completion": "medium correct one", "score": 0.80},
    ]
    best = H.select_best(samples, floor=0.70)
    assert best["completion"] == "short right"


def test_select_best_returns_none_when_nothing_clears_the_floor():
    samples = [{"completion": "near miss", "score": 0.69},
               {"completion": "miss", "score": 0.0}]
    assert H.select_best(samples, floor=0.70) is None


def test_select_best_never_returns_an_empty_completion():
    # Belt-and-braces with the judge's empty-answer guard: even if a score leaked
    # through high, an empty completion is not training data.
    samples = [{"completion": "   ", "score": 0.99},
               {"completion": "real answer", "score": 0.75}]
    assert H.select_best(samples, floor=0.70)["completion"] == "real answer"


def test_select_best_rejects_degenerate_digit_spam_even_when_judge_passed_it():
    """Pins the 2026-09-05 spot-read: the judge scored '1.0.0.1.0.1...' >= 0.70 on
    6 of 10 sampled harvest rows. Degenerate junk never becomes training data."""
    spam = "1.0.0.1.0.1.0.1.1.0.1.1.0.1.1.0.1.1.0.1.1.0.1.1.0.1.1.0.1.1.0.1.1."
    samples = [{"completion": spam, "score": 0.95},
               {"completion": "the split minimizes weighted impurity", "score": 0.80}]
    assert H.select_best(samples, floor=0.70)["completion"] == \
        "the split minimizes weighted impurity"
    assert H.select_best([{"completion": spam, "score": 0.99}], floor=0.70) is None


def test_sft_row_matches_the_deposit_schema():
    row = H.to_sft_row(
        {"id": "7", "question": "Q?", "gold": "G", "category": "analysis",
         "source_skill": "boosted-tree"},
        "the harvested answer")
    assert row == {"prompt": "Q?", "gold": "G", "answer": "the harvested answer",
                   "context": "", "domain": "analysis", "skill": "boosted-tree"}
