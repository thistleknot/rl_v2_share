"""tests/test_dpo_pass.py — preference-pair construction for the IPO rung.

Spec: docs/plans/break-the-9-advanced-rl-ladder.md rung R3 · Task: tasks.md #41.
"""
import sys
from pathlib import Path


from rlv2.train.rl import dpo_pass as D

SPAM = "1.0.0.1.0.1.0.1.1.0.1.1.0.1.1.0.1.1.0.1.1.0.1.1.0.1.1.0.1.1.0.1.1."


def _fixtures():
    pool = [{"id": "1", "question": "q1", "gold": "g"},
            {"id": "2", "question": "q2", "gold": "g"},
            {"id": "3", "question": "q3", "gold": "g"}]
    gens = {
        "1": ["right answer", "wrong answer", SPAM],
        "2": ["all wrong", "also wrong"],          # no chosen -> no pair
        "3": ["only right", "another right"],      # no rejected -> no pair
    }
    scores = {"1": [0.9, 0.2, 0.95], "2": [0.1, 0.3], "3": [0.8, 0.9]}
    return pool, gens, scores


def test_pair_needs_both_a_correct_and_a_failed_sample():
    pool, gens, scores = _fixtures()
    pairs = D.build_pairs(pool, gens, scores, floor=0.7)
    assert [p["question"] for p in pairs] == ["q1"]


def test_chosen_is_the_correct_answer_never_the_high_scored_spam():
    pool, gens, scores = _fixtures()
    p = D.build_pairs(pool, gens, scores, floor=0.7)[0]
    # SPAM scored 0.95 (judge-fooled) but is degenerate; chosen must be the real one.
    assert p["chosen"] == "right answer"


def test_rejected_is_the_worst_scoring_failed_sample():
    pool = [{"id": "1", "question": "q", "gold": "g"}]
    gens = {"1": ["good", "bad", "worse"]}
    scores = {"1": [0.9, 0.5, 0.1]}
    p = D.build_pairs(pool, gens, scores, floor=0.7)[0]
    assert p["rejected"] == "worse"


def test_judge_fooled_spam_is_admitted_as_rejected_and_ranked_first():
    """Pins the 61->4 pair collapse: spam scores ABOVE the floor (judge-fooled),
    so a score-only rejected filter never admits it. Degenerate junk is exactly
    the behavior to push away from — it outranks an honest low-scoring miss."""
    pool = [{"id": "1", "question": "q", "gold": "g"}]
    gens = {"1": ["right answer", "honest miss", SPAM]}
    scores = {"1": [0.9, 0.3, 0.95]}
    p = D.build_pairs(pool, gens, scores, floor=0.7)[0]
    assert p["chosen"] == "right answer"
    assert p["rejected"] == SPAM


def test_empty_completions_are_never_a_rejected_sample():
    pool = [{"id": "1", "question": "q", "gold": "g"}]
    gens = {"1": ["good", "   "]}
    scores = {"1": [0.9, 0.0]}
    assert D.build_pairs(pool, gens, scores, floor=0.7) == []
