"""
Test regime-branched reward function.
"""
import os as _o, sys as _s
from rlv2.judge.reward import reward

def test_r2_abstention_reward():
    """R2: abstention string → 1.0, else 0."""
    rec = {"answer": "not supported by context"}
    reward_val = reward(rec["answer"], "", "R2", "none")
    assert reward_val == 1.0

def test_r4_unanswerable():
    """R4: unanswerable string → 1.0."""
    rec = {"answer": "Unanswerable"}
    reward_val = reward(rec["answer"], "", "R4", "none")
    assert reward_val == 1.0