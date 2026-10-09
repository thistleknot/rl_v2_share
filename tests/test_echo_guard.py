"""The question repeated back is scored 0.0 without the judge, on every judge path.

Spec: .spec/specs/r3-domain-injection/tasks.md #51 (echo guard).
Measured 2026-09-27: the capped transfer run's answers opened with the question verbatim
16.4% of the time (490 of 2996) and the Jev judge paid 326 of them.
"""
import io
import json
import sys

import pytest

from rlv2.judge import judge as J  # noqa: E402

Q = ("Write a function `check_model_type` that validates the input target variable y. "
     "If y contains only strings representing categories, the function must confirm the model.")


def test_verbatim_echo_is_caught():
    assert J.echoes_question(Q, Q)
    assert J.echoes_question(Q, Q[:120])                 # a truncated copy too


def test_real_answer_is_not_caught():
    assert not J.echoes_question(Q, "It is a supervised learning model for categorical outcomes.")
    assert not J.echoes_question(Q, "0.72")               # short answers pass through


def test_judge_scores_echo_zero_without_calling_the_llm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the LLM judge must not see an echo")
    monkeypatch.setattr(J, "_call", boom)
    r = J.judge(Q, "It is a supervised classifier.", Q)
    assert r["score"] == 0.0 and r["ok"] is True and "repeats the question" in r["why"]


@pytest.mark.parametrize("path", ["data/rl/r3_eval_canonical.jsonl", "data/rl/r3_rl_transfer40.jsonl"])
def test_no_real_gold_is_rejected(path):
    rows = [json.loads(l) for l in io.open(path, encoding="utf-8")]
    assert [r["id"] for r in rows if J.echoes_question(r["question"], r["gold"])] == []


def test_only_the_broken_training_golds_are_flagged():
    rows = [json.loads(l) for l in io.open("data/rl/r3_rl.jsonl", encoding="utf-8")]
    flagged = sorted(r["id"] for r in rows if J.echoes_question(r["question"], r["gold"]))
    assert flagged == ["r3-0567", "r3-0570", "r3-0571"]   # gold == question text


def test_jev_path_zeroes_echo_before_the_server(monkeypatch):
    from rlv2.judge import reward as R
    monkeypatch.setattr(R, "_jev_load", lambda: {"thr": 0.5})
    monkeypatch.setattr(R, "_jev_p_yes", lambda *a: (_ for _ in ()).throw(
        AssertionError("Jev must not see an echo")))
    out = R._jev_scores([{"question": Q, "gold": "It is a supervised classifier.",
                          "generated": Q}])
    assert out == [0.0]
