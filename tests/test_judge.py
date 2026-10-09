"""Unit tests: src/rlv2/judge/judge.py — the verdict+score instrument.

The dangerous failure is a judge that defaults to `correct` when it cannot parse a
response: an ollama outage would then manufacture lift, and the resulting number would be
indistinguishable from a real one. Every malformed-input path is pinned to fail closed.
"""
from __future__ import annotations
import os as _o, sys as _s

import pytest

from rlv2.judge import judge as J


# --------------------------------------------------------------------------
# fail closed — every one of these must NOT be `correct`
# --------------------------------------------------------------------------
@pytest.mark.parametrize("raw", [
    None,                                    # transport failure
    {},                                      # empty object
    {"score": 0.9},                          # verdict missing
    {"verdict": "CORRECT-ISH"},              # not in the enum
    {"verdict": ""},                         # blank
    "not a dict",
    ["correct", 1.0],
])
def test_malformed_verdict_fails_closed(raw):
    out = J.parse({"quality": 0.5, "conciseness": 0.5}, raw)
    assert out["ok"] is False
    assert out["verdict"] == "not_attempted"
    assert out["score"] == 0.0
    assert J.is_correct(out) is False, "a parse failure must never read as correct"


@pytest.mark.parametrize("q_raw", [
    None,                                    # quality call failed entirely
    {},                                      # no quality emitted
    {"quality": "high"},                     # not numeric
    {"quality": None},
    {"conciseness": 0.5},                    # conciseness without quality
])
def test_malformed_quality_fails_closed(q_raw):
    """Both calls must succeed. A verdict without a quality score is not a usable
    measurement, and defaulting the missing half would invent a number."""
    out = J.parse(q_raw, {"verdict": "correct"})
    assert out["ok"] is False
    assert J.is_correct(out) is False


def test_nan_score_fails_closed():
    assert J.parse({"quality": float("nan")}, {"verdict": "correct"})["ok"] is False


def test_judge_returns_not_attempted_when_the_service_is_down(monkeypatch):
    monkeypatch.setattr(J, "_call", lambda *a, **k: None)
    out = J.judge("q", "g", "a")
    assert out["ok"] is False
    assert out["verdict"] == "not_attempted"
    assert J.is_correct(out) is False


# --------------------------------------------------------------------------
# well-formed responses
# --------------------------------------------------------------------------
@pytest.mark.parametrize("verdict,expected", [
    ("correct", 0.85),          # 0.70 + 0.30*0.5
    # 0.69 * 0.5, NOT 0.70 * 0.5. The incorrect band tops out strictly BELOW the correct
    # floor: at quality 1.0 the old arithmetic returned exactly 0.7000, putting a fluent
    # wrong answer ON the correct line. `is_correct` reads the verdict so the gate never
    # miscounted it, but the score is what RL consumes as reward.
    ("incorrect", 0.345),       # 0.69 * 0.5
    # `not_attempted` is deliberately NOT in VERDICTS -- R3 is closed-book with every item
    # answerable, so there is no abstention class and an answer that attempts and misses is
    # `incorrect`, never "no attempt". It survives only as a parse-failure sentinel, which
    # `test_parse_fails_closed_*` covers.
])
def test_verdict_selects_the_band_and_quality_positions_within_it(verdict, expected):
    out = J.parse({"quality": 0.5, "conciseness": 0.5}, {"verdict": verdict})
    assert out["ok"] is True
    assert out["verdict"] == verdict
    assert out["score"] == pytest.approx(expected)


def test_a_score_field_from_the_judge_is_IGNORED():
    """The judge does not get to hand us a final score — it reports quality and verdict,
    and the combination happens in code. A judge-supplied `score` must not leak through."""
    out = J.parse({"quality": 0.5, "conciseness": 0.5},
                  {"verdict": "correct", "score": 0.01})
    assert out["score"] == pytest.approx(0.85)


def test_verdict_is_case_insensitive_and_trimmed():
    assert J.parse({"quality": 0.5, "conciseness": 0.5}, {"verdict": "  Correct ", "score": 0.8})["verdict"] == "correct"


def test_quality_is_clamped_to_the_unit_interval():
    assert J.parse({"quality": 1.7, "conciseness": 1.7}, {"verdict": "correct"})["quality"] == 1.0
    assert J.parse({"quality": -0.5, "conciseness": -0.5}, {"verdict": "correct"})["quality"] == 0.0


def test_integer_quality_is_accepted():
    assert J.parse({"quality": 1, "conciseness": 1}, {"verdict": "correct"})["score"] == 1.0


def test_a_correct_answer_never_scores_below_an_incorrect_one():
    """The band boundary is the whole design: only the verdict may cross it."""
    worst_correct = J.parse({"quality": 0.0, "conciseness": 0.0}, {"verdict": "correct"})
    best_incorrect = J.parse({"quality": 1.0, "conciseness": 1.0}, {"verdict": "incorrect"})
    assert worst_correct["score"] >= best_incorrect["score"]


def test_conciseness_moves_the_score_shorter_is_better():
    terse = J.parse({"quality": 0.8, "conciseness": 0.9}, {"verdict": "correct"})["score"]
    padded = J.parse({"quality": 0.8, "conciseness": 0.2}, {"verdict": "correct"})["score"]
    assert terse > padded, "shorter answers must be preferred"


def test_missing_conciseness_falls_back_to_quality_rather_than_failing():
    out = J.parse({"quality": 0.6}, {"verdict": "correct"})
    assert out["ok"] is True
    assert out["conciseness"] == pytest.approx(0.6)


def test_why_is_captured_and_bounded():
    out = J.parse({"quality": 0.5, "conciseness": 0.5}, {"verdict": "correct", "score": 0.9, "why": "x" * 500})
    assert out["why"].startswith("x")
    assert len(out["why"]) <= 200


# --------------------------------------------------------------------------
# the binary the gate reads
# --------------------------------------------------------------------------
def test_only_correct_counts_as_correct():
    """`not_attempted` and `incorrect` are different failures needing different fixes,
    which is why the verdict keeps three values — but the gate collapses both to False."""
    assert J.is_correct({"verdict": "correct"}) is True
    assert J.is_correct({"verdict": "incorrect"}) is False
    assert J.is_correct({"verdict": "not_attempted"}) is False
    assert J.is_correct({}) is False


def test_a_high_score_does_not_make_a_verdict_correct():
    """The score must never be thresholded into the decision — that is the whole design."""
    assert J.is_correct(J.parse({"quality": 0.5, "conciseness": 0.5}, {"verdict": "incorrect", "score": 0.99})) is False


# --------------------------------------------------------------------------
# retry behaviour
# --------------------------------------------------------------------------
def test_a_transient_failure_is_retried_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky(prompt, schema, *_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            return None                       # first scoring call fails
        return {"gold_claim": "averaging cuts variance",
                "answer_claim": "averaging cuts variance",
                "answer_states_gold_claim": True,
                "shortcoming": "none", "score": 0.94}

    monkeypatch.setattr(J, "_call", flaky)
    out = J.judge("q", "g", "a", retries=2)
    assert out["ok"] is True
    assert out["score"] == pytest.approx(0.94)
    assert out["verdict"] == "correct", "verdict is DERIVED from the score, not asserted"
    # ONE call per attempt now (the verdict+quality pair was retired 2026-08-29), so a
    # single retry means exactly two calls -- not more.
    assert calls["n"] == 2


def test_retries_are_bounded(monkeypatch):
    calls = {"n": 0}

    def always_bad(prompt, schema, *_a, **_k):
        calls["n"] += 1
        return None

    monkeypatch.setattr(J, "_call", always_bad)
    J.judge("q", "g", "a", retries=2)
    assert calls["n"] == 2, "2 retries x 1 call each; must not retry forever"


# --------------------------------------------------------------------------
# the rubric is one file, hashed (REQ-J9 / J10)
# --------------------------------------------------------------------------
def test_the_quality_prompt_NEVER_contains_the_gold():
    """This is the whole mechanism. Positional blindness (quality-first in one call) was
    measured at 7/10 gold recognition and sd 0.071; withholding the gold gave 10/10 and
    restored the spread. If the gold leaks back in, the instrument silently reverts."""
    p = J.build_quality_prompt("what is gini impurity?", "an answer", "STEM")
    assert "SECRETGOLD" not in p
    p2 = J.build_quality_prompt("q", "a", "STEM")
    assert "Gold" not in p2, "the quality call must not mention a gold at all"


def test_prompt_embeds_the_rubric_and_all_four_fields():
    p = J.build_verdict_prompt("what is gini?", "a purity measure", "no idea", "STEM")
    assert "what is gini?" in p
    assert "a purity measure" in p
    assert "no idea" in p
    assert "STEM" in p
    assert "verdict" in p, "the rubric itself must be in the prompt"


def test_empty_answer_is_marked_rather_than_sent_blank():
    assert "<EMPTY>" in J.build_verdict_prompt("q", "g", "   ")
    assert "<EMPTY>" in J.build_quality_prompt("q", "   ")


def test_rubric_hash_is_stable_and_changes_with_content(tmp_path):
    a = tmp_path / "r.md"
    a.write_text("rubric one", encoding="utf-8")
    h1 = J.rubric_sha(str(a))
    assert h1 == J.rubric_sha(str(a)), "hash must be stable"
    b = tmp_path / "r2.md"
    b.write_text("rubric two", encoding="utf-8")
    assert J.rubric_sha(str(b)) != h1


def test_missing_rubric_reports_rather_than_crashing():
    assert J.rubric_sha("no/such/rubric.md") == "MISSING"


def test_instrument_records_what_produced_a_number():
    inst = J.instrument()
    assert inst["judge_model"]
    assert inst["rubric"].endswith(".md")
    assert len(inst["rubric_sha256_16"]) in (16, 7)


# --------------------------------------------------------------------------
# the shipped rubric
# --------------------------------------------------------------------------
RUBRIC = _o.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))),
                      "data", "synthetic_domain_rubric.md")


@pytest.mark.skipif(not _o.path.exists(RUBRIC), reason="rubric not present")
def test_shipped_rubric_states_the_three_verdicts_and_the_gold_band():
    txt = open(RUBRIC, encoding="utf-8").read()
    for v in J.VERDICTS:
        assert v in txt
    assert "0.85" in txt, "the gold band must be stated (REQ-J4)"
    assert "SHORTER IS BETTER" in txt or "Shorter is preferred" in txt,         "the conciseness preference must be stated"
    assert "not measuring anything" in txt or "not all equally tight" in txt,         "saturation must be explicitly warned against"
    for cat in ("Math", "Code", "STEM", "Instruction"):
        assert cat in txt, "per-category criteria missing for %s (REQ-J7)" % cat


def test_stub_answer_ending_in_a_colon_scores_zero_without_the_llm(monkeypatch):
    """2026-09-06 final board: 'The prediction is:' and 'The relationship is:' were
    graded CORRECT by the LLM — a truncated frame asserts nothing, same class as
    the empty answer. The guard must fire BEFORE any network call."""
    monkeypatch.setattr(J, "_call",
                        lambda *a, **k: pytest.fail("stub reached the LLM judge"))
    for stub in ("The prediction is:", "  The relationship is:  ", "The rule is used because:"):
        out = J.judge("q", "gold", stub)
        assert out["ok"] and out["score"] == 0.0 and out["verdict"] == "incorrect"
        assert "stub" in out["why"]


def test_short_but_contentful_answers_still_reach_the_judge(monkeypatch):
    """The gate is about ASSERTING NOTHING, not about length: 'The value is 0.25.'
    is 18 characters and correct — it must not be gated."""
    seen = {}
    def fake_call(prompt, schema, model, timeout=None, **k):
        seen["called"] = True
        # raw judge-output shape (parse_score derives the verdict from it)
        return {"score": 0.95, "answer_states_gold_claim": True,
                "quality": 0.9, "conciseness": 0.9, "self_sufficient": True,
                "why": "x"}
    monkeypatch.setattr(J, "_call", fake_call)
    out = J.judge("q", "0.25", "The value is 0.25.")
    assert seen.get("called"), "a contentful short answer was gated from the judge"
    assert out["score"] == 0.95


def test_single_answer_judging_reserves_room_for_a_long_gold(monkeypatch):
    """Measured 2026-09-07 on eval item r3-0640 (gold ~300 chars, prompt 13.6k):
    at num_predict=256 the schema's gold_claim echo cut the reply mid-JSON, so
    the parse failed, three retries later the cell read 'judge unavailable', and
    that question became a silent zero on every arm. With room it parses in 31s."""
    seen = {}
    def fake_call(prompt, schema, model, timeout, num_predict=256):
        seen["num_predict"] = num_predict
        return {"score": 0.9, "answer_states_gold_claim": True, "quality": 0.9,
                "conciseness": 0.9, "self_sufficient": True, "why": "x"}
    monkeypatch.setattr(J, "_call", fake_call)
    J.judge("q", "g" * 300, "a real answer")
    assert seen["num_predict"] >= 512, \
        "single-answer judging is back to a budget that truncates long golds"


# --------------------------------------------------------------------------
# the GPU handoff is for the LOCAL judge only (measured 2026-09-13)
# --------------------------------------------------------------------------
def test_a_remote_judge_does_not_move_the_policy_off_the_gpu(monkeypatch):
    """Measured: offload=2.1s | judge=6.9s | restore=1.8s, judge on OpenRouter.

    36% of every group's wall-clock was shuttling 752M parameters across the bus
    to free VRAM for an HTTP request. The handoff exists for the local judge,
    which really does contend for the same card.
    """
    import sys, types
    sys.path.insert(0, "src")
    from rlv2.judge import reward

    moved = []

    class _FakeModel:
        def to(self, device):
            moved.append(device)
            return self

    monkeypatch.setenv("JUDGE_REMOTE", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert reward._openrouter_enabled(), "precondition: the remote gate is open"

    with reward.judge_vram_handoff(policy=_FakeModel()):
        pass
    assert moved == [], "a remote judge must not touch the policy's placement"


def test_a_local_judge_still_gets_the_gpu(monkeypatch):
    import sys
    sys.path.insert(0, "src")
    from rlv2.judge import reward

    moved = []

    class _FakeModel:
        def to(self, device):
            moved.append(device)
            return self

    monkeypatch.delenv("JUDGE_REMOTE", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert not reward._openrouter_enabled() and not reward._remote_enabled()

    with reward.judge_vram_handoff(policy=_FakeModel()):
        pass
    assert "cpu" in moved, "the local judge shares one card -- it needs the room"
