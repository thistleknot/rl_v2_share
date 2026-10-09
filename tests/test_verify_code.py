"""Unit tests: verify_code — executing candidate code against dataset asserts.

Spec: .spec/specs/r3-domain-injection/rlvr-math-code.md (acceptance).

THE POINT OF THIS FILE IS THAT THE PROBE CAN GO RED. A verifier that cannot fail
on wrong code is worse than no verifier: it converts an unknown into a confident
wrong reward, and every gradient downstream trusts it. compile() alone passes on
code that returns the wrong answer, which is why execution is the score and
compile is only a gate.

Each test breaks the candidate in ONE specific way and asserts the verdict.
"""
from __future__ import annotations

import os as _o
import sys as _s
import time

for _d in ("tools",):
    _p = _o.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))), _d)
    _p not in _s.path and _s.path.insert(0, _p)

import pytest  # noqa: E402

from rlv2.judge.verifiers import CODE_TIMEOUT_S, rlvr_score, verify_code  # noqa: E402

MBPP = {"tests": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]}
GOLD = "def add(a, b):\n    return a + b"

HUMANEVAL = {
    "entry_point": "add",
    "tests": ["def check(candidate):\n    assert candidate(1, 2) == 3\n"
              "    assert candidate(-1, 1) == 0\n"],
}


# --------------------------------------------------------------------------
# it says YES to correct code
# --------------------------------------------------------------------------
def test_gold_solution_passes():
    assert verify_code(GOLD, MBPP) is True


def test_humaneval_check_function_is_actually_called():
    # HumanEval ships `check(candidate)` but never calls it. If the harness
    # forgets the call, EVERY candidate passes -- the probe silently dies open.
    assert verify_code(GOLD, HUMANEVAL) is True
    assert verify_code("def add(a, b):\n    return 999", HUMANEVAL) is False


def test_code_inside_a_markdown_fence_is_unwrapped():
    fenced = "Here you go:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
    assert verify_code(fenced, MBPP) is True


def test_think_block_is_stripped_before_execution():
    answer = "<think>let me reason</think>\ndef add(a, b):\n    return a + b"
    assert verify_code(answer, MBPP) is True


# --------------------------------------------------------------------------
# it says NO to wrong code — the cases compile() would wave through
# --------------------------------------------------------------------------
def test_wrong_answer_fails_even_though_it_compiles():
    wrong = "def add(a, b):\n    return a * b"
    assert compile(wrong, "<t>", "exec") is not None   # compiles fine
    assert verify_code(wrong, MBPP) is False           # and is still rejected


def test_passing_one_assert_but_not_all_is_a_failure():
    # add(1,2)==3 holds, add(-1,1)==0 does not: all-or-nothing
    half = "def add(a, b):\n    return 3 if (a, b) == (1, 2) else 42"
    assert verify_code(half, MBPP) is False


def test_syntax_error_fails_without_crashing():
    assert verify_code("def add(a, b) return a + b", MBPP) is False


def test_runtime_exception_fails():
    assert verify_code("def add(a, b):\n    raise ValueError('boom')", MBPP) is False


def test_right_function_never_defined_fails():
    # prints the right answer, defines nothing -- a classic reward hack
    assert verify_code("print(3)", MBPP) is False


def test_empty_and_blank_answers_fail():
    assert verify_code("", MBPP) is False
    assert verify_code("   \n  ", MBPP) is False
    assert verify_code(None, MBPP) is False


def test_prose_with_no_code_fails():
    assert verify_code("The answer is to add the two numbers together.", MBPP) is False


# --------------------------------------------------------------------------
# it cannot be hung
# --------------------------------------------------------------------------
def test_infinite_loop_fails_by_timeout_and_the_timeout_is_enforced():
    started = time.time()
    verdict = verify_code("def add(a, b):\n    while True:\n        pass", MBPP)
    elapsed = time.time() - started
    assert verdict is False
    assert elapsed < CODE_TIMEOUT_S + 15, "timeout did not bound the run"


# --------------------------------------------------------------------------
# it abstains rather than lying — Article X
# --------------------------------------------------------------------------
def test_no_tests_returns_none_not_false():
    # an absent test is NOT evidence the code is wrong
    assert verify_code(GOLD, {"tests": []}) is None
    assert verify_code(GOLD, {}) is None
    assert verify_code(GOLD, None) is None


def test_a_record_without_tests_abstains_even_for_garbage_code():
    assert verify_code("!!! not code at all", {"tests": []}) is None


# --------------------------------------------------------------------------
# tier routing
# --------------------------------------------------------------------------
def test_rlvr_score_routes_code_records_to_the_code_tier():
    assert rlvr_score(GOLD, "", MBPP) == (True, "code")
    assert rlvr_score("def add(a, b):\n    return a * b", "", MBPP) == (False, "code")


def test_rlvr_score_routes_math_to_the_numeric_tier():
    assert rlvr_score("the total is 16", "#### 16") == (True, "numeric")
    assert rlvr_score("the total is 99", "#### 16") == (False, "numeric")


def test_rlvr_score_falls_back_to_the_judge_when_nothing_verifies():
    verdict, method = rlvr_score("some prose", "a gold that is long enough prose "
                                 "to carry no extractable key at all")
    assert verdict is None and method == "judge"


def test_a_record_declaring_judge_is_never_mechanically_scored():
    # a 2-word PROSE gold looks exactly like a STEM key; only the declaration
    # separates them, and guessing wrong scores an unverifiable record as WRONG
    assert rlvr_score("some prose", "other prose",
                      {"verifier": "judge"}) == (None, "judge")


def test_a_record_declaring_short_answer_uses_the_stem_tier():
    rec = {"verifier": "short_answer"}
    assert rlvr_score("the answer is deductive", "deductive", rec) == (True, "short_answer")
    assert rlvr_score("inductive", "deductive", rec) == (False, "short_answer")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# --------------------------------------------------------------------------
# complexity — a tie-breaker among CORRECT solutions, never a rescue
# --------------------------------------------------------------------------
from rlv2.judge.verifiers import code_simplicity, score_code  # noqa: E402

BLOATED = ("def add(a, b):\n"
           "    result = 0\n"
           "    for _ in range(abs(a)):\n"
           "        result = result + (1 if a > 0 else -1)\n"
           "    for _ in range(abs(b)):\n"
           "        result = result + (1 if b > 0 else -1)\n"
           "    total = result\n"
           "    return total")


def test_elegant_correct_beats_bloated_correct():
    assert verify_code(BLOATED, MBPP) is True, "fixture must be CORRECT, only ugly"
    assert score_code(GOLD, {**MBPP, "answer": GOLD}) > \
           score_code(BLOATED, {**MBPP, "answer": GOLD})


def test_a_wrong_answer_scores_zero_however_elegant_it_is():
    # one line, beautiful, wrong -- elegance must not buy anything back
    assert score_code("def add(a, b):\n    return a * b",
                      {**MBPP, "answer": GOLD}) == 0.0


def test_correct_but_bloated_still_earns_the_correctness_floor():
    s = score_code(BLOATED, {**MBPP, "answer": GOLD})
    assert 0.8 <= s < 1.0, "correctness alone is worth 0.8"


def test_matching_the_reference_earns_full_credit():
    assert score_code(GOLD, {**MBPP, "answer": GOLD}) == 1.0


def test_shrinking_below_the_reference_wins_nothing_extra():
    # THE anti-code-golf cap: without it, the optimum is a stripped one-liner
    assert code_simplicity("def add(a,b):return a+b", BLOATED) == 1.0
    assert code_simplicity(GOLD, BLOATED) == 1.0


def test_bloat_is_penalised_in_proportion():
    s = code_simplicity(BLOATED, GOLD)
    assert 0.0 < s < 1.0


def test_unparseable_side_yields_no_signal_rather_than_a_penalty():
    assert code_simplicity("def broken(", GOLD) == 1.0
    assert code_simplicity(GOLD, "def broken(") == 1.0


def test_score_code_abstains_when_there_are_no_tests():
    assert score_code(GOLD, {"tests": [], "answer": GOLD}) is None


# --------------------------------------------------------------------------
# math quality — same shape as code, anchored on the reference solution
# --------------------------------------------------------------------------
from rlv2.judge.verifiers import math_conciseness, score_math, score_rlvr  # noqa: E402

GOLD_MATH = ("Mimi has 2 x 12 = 24 shells.\nKyle has 24 x 2 = 48 shells.\n"
             "Leigh has 48 / 3 = 16 shells.\n#### 16")


def test_wrong_number_scores_zero_however_short():
    assert score_math("#### 99", GOLD_MATH) == 0.0


def test_a_concise_correct_answer_earns_full_credit():
    assert score_math(GOLD_MATH, GOLD_MATH) == 1.0


def test_rambling_correct_answer_still_clears_the_correctness_floor():
    rambling = ("Let me think about this very carefully and at great length. " * 20) + "#### 16"
    s = score_math(rambling, GOLD_MATH)
    assert 0.8 <= s < 1.0


def test_concise_beats_rambling_when_both_are_correct():
    rambling = ("Well, first we should consider every possibility here. " * 15) + "#### 16"
    assert score_math(GOLD_MATH, GOLD_MATH) > score_math(rambling, GOLD_MATH)


def test_answering_shorter_than_the_reference_wins_nothing_extra():
    assert math_conciseness("#### 16", GOLD_MATH) == 1.0


def test_think_block_is_not_counted_against_length():
    # reasoning inside <think> is stripped before generation is scored elsewhere
    assert math_conciseness("<think>" + "x " * 500 + "</think> #### 16", GOLD_MATH) == 1.0


def test_score_rlvr_prefers_the_code_tier_for_records_with_tests():
    assert score_rlvr(GOLD, "", {**MBPP, "answer": GOLD}) == (1.0, "code")


def test_score_rlvr_uses_math_when_there_are_no_tests():
    reward, method = score_rlvr(GOLD_MATH, GOLD_MATH)
    assert method == "numeric" and reward == 1.0


def test_score_rlvr_abstains_to_the_judge_on_prose():
    assert score_rlvr("some prose", "other prose") == (None, "judge")


def test_no_reward_ever_exceeds_one_or_falls_below_zero():
    for r, _ in [score_rlvr(GOLD, "", {**MBPP, "answer": GOLD}),
                 score_rlvr("def add(a,b): return a*b", "", {**MBPP, "answer": GOLD}),
                 score_rlvr(GOLD_MATH, GOLD_MATH),
                 score_rlvr("#### 99", GOLD_MATH)]:
        assert r is None or 0.0 <= r <= 1.0


# --------------------------------------------------------------------------
# STEM short-answer — exact match on a 2-word span
# --------------------------------------------------------------------------
from rlv2.judge.verifiers import SHORT_ANSWER_MAX_WORDS, verify_short_answer  # noqa: E402


def test_exact_gold_matches():
    assert verify_short_answer("deductive", "deductive") is True


def test_gold_embedded_in_a_sentence_matches():
    assert verify_short_answer("The answer is deductive.", "deductive") is True
    assert verify_short_answer("It is caused by air pollution", "air pollution") is True


def test_a_different_answer_fails():
    assert verify_short_answer("inductive", "deductive") is False
    assert verify_short_answer("The answer is inductive reasoning", "deductive") is False


def test_multiword_gold_must_be_contiguous_and_in_order():
    # "pollution ... air" must NOT satisfy gold "air pollution"
    assert verify_short_answer("pollution of the air", "air pollution") is False
    assert verify_short_answer("air pollution", "air pollution") is True


def test_case_and_punctuation_are_normalised():
    assert verify_short_answer("AIR-POLLUTION!", "air pollution") is True


def test_empty_answer_fails_but_empty_gold_abstains():
    assert verify_short_answer("", "deductive") is False
    assert verify_short_answer("anything", "") is None


def test_a_long_prose_gold_abstains_rather_than_guessing():
    # R3-length golds scored AUC 0.580 on this instrument -- abstain, use the judge
    long_gold = " ".join("word%d" % i for i in range(SHORT_ANSWER_MAX_WORDS + 1))
    assert verify_short_answer("whatever", long_gold) is None


def test_think_block_does_not_supply_the_answer():
    assert verify_short_answer("<think>deductive</think> inductive", "deductive") is False


def test_asserts_using_math_isclose_work_without_a_dataset_import():
    # MBPP asserts call math.isclose but the dataset ships no import; without a
    # preamble a CORRECT gold dies on NameError and scores False (10/421 did)
    rec = {"tests": ["assert math.isclose(circ(10), 62.830000000000005, rel_tol=0.001)"]}
    assert verify_code("def circ(r):\n    return 2 * math.pi * r", rec) is True


def test_the_preamble_does_not_rescue_a_wrong_answer():
    rec = {"tests": ["assert math.isclose(circ(10), 62.83, rel_tol=0.001)"]}
    assert verify_code("def circ(r):\n    return 999.0", rec) is False
