"""The judge's max_tokens is derived from full replies, and must not truncate one.

NO GOVERNING SPEC. Basis: operator 2026-09-16 -- "sample 20, figure out their sizes, cap
description at median + mad*2", replacing a hand-set 1024 that let a runaway reply cost a
75 s wait.

MEASURED TWICE, because the first derivation sampled the wrong thing:

    `why` field alone   p50  33  MAD  7  -> 79   TRUNCATED; the reply died mid-JSON
                        ValueError: openrouter judge returned non-JSON:
                        '{"gold_claim": "training error is monotonically non-decreasing...'
    full reply          p50 150  MAD 24  -> 199  16/16 parsed, 0 no-score

max_tokens bounds the WHOLE emitted object and the live schema carries five fields, so a
cap derived from one field cuts the reply in half.
"""
import io
import sys



def test_the_cap_is_a_named_constant_not_a_literal():
    src = io.open("src/rlv2/judge/reward.py", encoding="utf-8").read()
    assert '"max_tokens": JUDGE_MAX_TOKENS' in src
    assert '"max_tokens": 1024' not in src, "the hand-set 1024 is back"


def test_the_cap_clears_the_whole_schema_not_one_field():
    """The cap must clear the longest reply ACTUALLY OBSERVED, not just the MAD rule.
    Measured on the two-field schema: p50 190, max 238, while exp(log-med + 4*MAD) = 234 --
    the rule sat below a real reply. A tight cap costs a dropped judgement; a generous one
    costs a few tokens of latency."""
    from rlv2.judge import reward as R
    assert R.JUDGE_MAX_TOKENS >= 190, "below the median reply (p50 190)"
    assert R.JUDGE_MAX_TOKENS >= 238, "below the longest reply actually observed"


def test_the_cap_is_not_wastefully_large():
    """Every token is judge latency, on every read, of every answer, of every arm."""
    from rlv2.judge import reward as R
    assert R.JUDGE_MAX_TOKENS <= 500


def test_the_schema_is_two_fields():
    """Operator 2026-09-16: "Its two fields / Reasoning / Score"."""
    from rlv2.judge import judge as J
    assert set(J._SCORE_SCHEMA["required"]) == {"reasoning", "score"}


def test_a_bare_number_is_not_a_judgement():
    """The rubric's ordering -- gold claim, answer claim, does-it-state-it, shortcoming --
    now lives inside `reasoning`. A reply with a score and no reasoning means that ordering
    never happened, so it must fail closed rather than be trusted (rubric item r3-0057, a
    procedural deferral that scored correct when nothing forced the check)."""
    from rlv2.judge import judge as J
    assert J.parse_score({"score": 0.95})["ok"] is False
    assert J.parse_score({"score": 0.95, "reasoning": "   "})["ok"] is False
    assert J.parse_score({"score": 0.95, "reasoning": "states the gold claim"})["ok"] is True


def test_the_verdict_cannot_disagree_with_the_score():
    """One number, two views of it."""
    from rlv2.judge import judge as J
    assert J.parse_score({"score": 0.71, "reasoning": "x"})["verdict"] == "correct"
    assert J.parse_score({"score": 0.69, "reasoning": "x"})["verdict"] == "incorrect"


def test_routing_is_pinned_to_endpoints_that_honour_the_schema():
    """OpenRouter serves one model from several providers; one that ignores
    response_format returns free-form text that looks like truncation."""
    src = io.open("src/rlv2/judge/reward.py", encoding="utf-8").read()
    assert '"provider"] = {"require_parameters": True}' in src.replace("body[", "")
