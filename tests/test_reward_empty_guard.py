"""tests/test_reward_empty_guard.py — empty answers never reach the judge, score 0.0.

Spec: .spec/specs/r3-domain-injection/tasks.md #35 (empty-answer guard).
Pins the measured defect: past9_v2p3 logged `[reward-sample] len=0 | ''` scored
`4/4 correct at 0.70` — the judge hallucinated a pass for a blank answer and paid
an EOS-collapsed policy full reward for silence.
"""
import sys
from pathlib import Path


from rlv2.judge import reward as R


def _patch_judge(monkeypatch, calls, score=0.95):
    """A judge that scores every pair it is shown 0.95 — a hallucinated pass.
    The guard must make it impossible for an empty answer to receive that score."""
    def fake_judge_pairs(pairs, model=None):
        calls.append([p["generated"] for p in pairs])
        return [score] * len(pairs)
    monkeypatch.setattr(R, "judge_pairs", fake_judge_pairs)


def test_empty_answer_scores_zero_even_when_judge_would_pass_it(monkeypatch):
    calls = []
    _patch_judge(monkeypatch, calls)
    scores = R.llm_judge_flat([
        {"question": "q1", "gold": "g1", "generated": ""},
        {"question": "q2", "gold": "g2", "generated": "a real answer"},
        {"question": "q3", "gold": "g3", "generated": "   \n\t "},
    ])
    assert scores == [0.0, 0.95, 0.0]


def test_empty_answers_are_never_sent_to_the_judge(monkeypatch):
    calls = []
    _patch_judge(monkeypatch, calls)
    R.llm_judge_flat([
        {"question": "q1", "gold": "g1", "generated": ""},
        {"question": "q2", "gold": "g2", "generated": "answer"},
    ])
    sent = [g for call in calls for g in call]
    assert sent == ["answer"], f"judge saw {sent!r}"


def test_all_empty_skips_the_judge_entirely(monkeypatch):
    calls = []
    _patch_judge(monkeypatch, calls)
    scores = R.llm_judge_flat([
        {"question": "q1", "gold": "g1", "generated": ""},
        {"question": "q2", "gold": "g2", "generated": "\n"},
    ])
    assert scores == [0.0, 0.0]
    assert calls == [], "judge was called for a batch of only empty answers"


def test_nonempty_ordering_preserved_around_empties(monkeypatch):
    calls = []
    def fake_judge_pairs(pairs, model=None):
        calls.append(True)
        return [0.1 * (i + 1) for i in range(len(pairs))]
    monkeypatch.setattr(R, "judge_pairs", fake_judge_pairs)
    scores = R.llm_judge_flat([
        {"question": "q1", "gold": "g1", "generated": "first"},
        {"question": "q2", "gold": "g2", "generated": ""},
        {"question": "q3", "gold": "g3", "generated": "second"},
    ])
    assert scores == [0.1, 0.0, 0.2]


def test_flip_pair_ragas_prompt_carries_the_plurality_note():
    """FLIP THE SCRIPT F2: flipped pairs tell the judge the gold is ONE valid
    answer; alternatives that are accurate and non-contradicting score correct."""
    pair = {"question": "Q?", "gold": "G", "generated": "A", "context": "",
            "multi_answer": True, "alt_answers": ["alt answer one"]}
    p = R._build_ragas_prompt(pair)
    assert "NOTE: this question admits more than one valid answer" in p
    assert "alt answer one" in p


def test_non_flip_ragas_prompt_is_byte_identical_to_unmarked():
    """The eval instrument must not drift: a pair without the marker renders
    exactly as before the F2 change."""
    base = {"question": "Q?", "gold": "G", "generated": "A", "context": ""}
    marked_false = {**base, "multi_answer": False, "alt_answers": []}
    assert R._build_ragas_prompt(base) == R._build_ragas_prompt(marked_false)
    assert "NOTE:" not in R._build_ragas_prompt(base)


def test_object_prompt_notes_only_the_flipped_pair():
    pairs = [
        {"question": "Q1?", "gold": "G1", "generated": "A1"},
        {"question": "Q2?", "gold": "G2", "generated": "A2",
         "multi_answer": True, "alt_answers": []},
    ]
    p = R._build_object_prompt(pairs)
    assert p.count("NOTE: this question admits") == 1
    q1_block = p.split("[q0]")[1].split("[q1]")[0]
    assert "NOTE:" not in q1_block


def test_batch_train_carries_the_plurality_marker_to_flat_pairs(monkeypatch):
    seen = []
    def fake_many(chunks, model=None):
        for chunk in chunks:
            seen.extend(chunk)
        return [[0.5] * len(c) for c in chunks]
    monkeypatch.setattr(R, "llm_judge_flat_many", fake_many)
    R.llm_judge_batch_train([{"question": "q", "gold": "g", "context": "",
                              "completions": ["a"], "multi_answer": True,
                              "alt_answers": ["x"]}])
    assert seen[0]["multi_answer"] is True
    assert seen[0]["alt_answers"] == ["x"]


def test_eval_judge_scores_empty_answer_incorrect_without_llm(monkeypatch):
    # Same guard on the eval/receipt instrument: judge.judge must never send an
    # empty answer to the LLM, and must fail closed to a deterministic incorrect.
    from rlv2.judge import judge as J

    def explode(*a, **k):
        raise AssertionError("LLM was called for an empty answer")

    monkeypatch.setattr(J, "_call", explode)
    for empty in ("", "   ", "\n\t"):
        v = J.judge("q", "gold", empty)
        assert v["verdict"] == "incorrect"
        assert v["score"] == 0.0
        assert v["ok"] is True


def test_think_only_completion_is_empty_after_strip(monkeypatch):
    # _strip_think removes the reasoning block; a completion that is ONLY a think
    # block is an empty answer and must be guarded like one.
    calls = []
    _patch_judge(monkeypatch, calls)
    scores = R.llm_judge_flat([
        {"question": "q1", "gold": "g1", "generated": "<think>musing</think>"},
    ])
    assert scores == [0.0]
    assert calls == []
