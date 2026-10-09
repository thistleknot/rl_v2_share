"""Jev binary judge: routing and scoring contract (operator 2026-09-26: 'Just ask jev if the answer
is correct'). NO GOVERNING SPEC. Live-server test is skipped only when 192.168.3.18 is unreachable."""
import os, sys, urllib.request
import pytest
from rlv2.judge import reward


def test_jev_on_by_default_and_only_explicit_zero_disables(monkeypatch):
    monkeypatch.delenv("JUDGE_JEV", raising=False)
    assert reward._jev_enabled() is True          # unset must never fall back to the 9B
    for off in ("0", "false", "no", "NO"):
        monkeypatch.setenv("JUDGE_JEV", off)
        assert reward._jev_enabled() is False
    monkeypatch.setenv("JUDGE_JEV", "1")
    assert reward._jev_enabled() is True


def test_binary_scores_and_empty_guard(monkeypatch):
    monkeypatch.setattr(reward, "_jev_load", lambda: {"thr": 0.5})
    monkeypatch.setattr(reward, "_jev_p_yes", lambda q, g, a: 0.9 if "right" in a else 0.1)
    gold = "the right answer mentions gradient boosting"
    pairs = [{"question": "q", "gold": gold, "generated": x} for x in ("gradient boosting is right", "gradient boosting is wrong", "   ")]
    assert reward._jev_scores(pairs) == [1.0, 0.0, 0.0]


def test_batch_routes_through_jev(monkeypatch):
    monkeypatch.setenv("JUDGE_JEV", "1")
    monkeypatch.setattr(reward, "_jev_scores", lambda pairs: [1.0 if "ok" in p["generated"] else 0.0 for p in pairs])
    out = reward.llm_judge_batch_train([{"question": "q", "gold": "g", "completions": ["ok", "no", "ok", "no"]}])
    assert out == [[1.0, 0.0, 1.0, 0.0]]


def test_lane_line_names_jev_when_enabled(monkeypatch, capsys):
    monkeypatch.setenv("JUDGE_JEV", "1")
    reward.set_judge_lane("r3")
    line = capsys.readouterr().out
    assert "open-jev-2b" in line and "9b" not in line, line


def test_judge_py_routes_through_jev(monkeypatch):
    from rlv2.judge import judge as J
    monkeypatch.setenv("JUDGE_JEV", "1")
    monkeypatch.setattr(reward, "_jev_load", lambda: {"thr": 0.5})
    monkeypatch.setattr(reward, "_jev_p_yes", lambda q, g, a: 0.8 if "depth" in a else 0.2)
    gold = "max_depth limits how deep the tree can grow"
    good = J.judge("q?", gold, "it limits the tree depth")
    bad = J.judge("q?", gold, "it sets the tree learning rate")
    assert (good["score"], bad["score"]) == (1.0, 0.0)
    assert "open-jev-2b" in good["why"]


def test_instrument_pins_jev(monkeypatch):
    from rlv2.train.rl import instrument
    monkeypatch.setenv("JUDGE_JEV", "1")
    assert instrument.observed()["judge_model"] == "open-jev-2b"
    assert not any("judge" in d for d in instrument.diff())
    monkeypatch.delenv("JUDGE_JEV")
    assert not any("judge" in d for d in instrument.diff())        # unset = Jev, still on-pin
    monkeypatch.setenv("JUDGE_JEV", "0")
    assert any("pinned open-jev-2b" in d for d in instrument.diff())  # only an explicit opt-out drifts


def test_ledger_key_separates_jev_from_9b(monkeypatch):
    from rlv2.train.rl import differentiability_ledger as L
    monkeypatch.setattr(L, "phase_state", lambda c: "sha123")
    monkeypatch.setenv("JUDGE_JEV", "0")
    old = L.key("ck", "data/rl/r3_rl.jsonl", L.MEASURES[0])
    monkeypatch.delenv("JUDGE_JEV", raising=False)
    new = L.key("ck", "data/rl/r3_rl.jsonl", L.MEASURES[0])
    assert old != new and new.endswith("|judge=open-jev-2b+gate1") and new.startswith(old)


def test_jev_gate_blocks_exploits_and_keeps_real_answers():
    gold = "The sum of squared residuals is 1^2 + (-1)^2 + 0^2 = 2."
    assert reward._jev_gate(gold, "1") is False            # the collapse answer: last number 1 != 2
    assert reward._jev_gate(gold, "2") is True             # right bare number
    assert reward._jev_gate(gold, "The sum is 2.") is True
    assert reward._jev_gate(gold, "Yes, that is correct.") is False   # affirmation stub, no content overlap
    assert reward._jev_gate(gold, "The squared residuals sum to two.") is True
    assert reward._jev_gate("p = 1 / (1 + np.exp(-F))", "1") is True    # gold's last number IS 1: a known leak, accepted
    assert reward._jev_gate("The value used is 8.", "8") is True
    assert reward._jev_gate("The value used is 8.", "7") is False


def test_jev_scores_apply_gate_before_judge(monkeypatch):
    monkeypatch.setattr(reward, "_jev_load", lambda: {"thr": 0.5})
    calls = []
    monkeypatch.setattr(reward, "_jev_p_yes", lambda q, g, a: calls.append(a) or 0.9)
    gold = "The sum of squared residuals is 1^2 + (-1)^2 + 0^2 = 2."
    out = reward._jev_scores([{"question": "q", "gold": gold, "generated": x} for x in ("1", "Yes, that is correct.", "The sum is 2.")])
    assert out == [0.0, 0.0, 1.0] and calls == ["The sum is 2."]


def _up():
    try:
        return b"ok" in urllib.request.urlopen("http://192.168.3.18:8791/health", timeout=3).read()
    except Exception:
        return False


@pytest.mark.skipif(not _up(), reason="Jev server 192.168.3.18:8791 unreachable")
def test_live_jev_separates_right_from_wrong():
    q = "What does max_depth control in a decision tree?"
    gold = "The maximum depth of the tree, limiting how many splits deep it can grow."
    good = reward._jev_p_yes(q, gold, "It limits how deep the tree can grow, i.e. the maximum number of split levels.")
    bad = reward._jev_p_yes(q, gold, "It sets the learning rate of the tree.")
    assert good > 0.5 > bad, (good, bad)
