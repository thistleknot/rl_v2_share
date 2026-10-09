"""Unit tests: src/rlv2/eval/score_arms.py — the one-invocation multi-arm scorer.

The dangerous failure here is a comparison that LOOKS valid. A shuffle that does not shuffle,
a permutation that is not undone, a KL that returns nan and gets read as zero, or a CI so
tight it manufactures significance. Each is pinned below, and each would be invisible in the
output.
"""
from __future__ import annotations
import os as _o, sys as _s

import json
import math
import random

import pytest

from rlv2.eval import score_arms as SA


# --------------------------------------------------------------------------
# auc — the MONOTONE separation measure
# --------------------------------------------------------------------------
def test_auc_is_one_when_every_correct_outranks_every_wrong():
    assert SA.auc([0.9, 0.8], [0.4, 0.1]) == 1.0


def test_auc_is_zero_when_the_ordering_is_inverted():
    """Not 0.5. An inverted judge is maximally wrong, and must not read as 'no signal'."""
    assert SA.auc([0.1], [0.9]) == 0.0


def test_auc_is_half_on_ties():
    assert SA.auc([0.5], [0.5]) == 0.5


def test_auc_grows_monotonically_as_the_classes_separate():
    """This is the property KL does NOT have, and the reason auc exists alongside it."""
    correct = [0.80, 0.78]
    a = SA.auc(correct, [0.79, 0.77])      # heavily overlapping
    b = SA.auc(correct, [0.60, 0.55])      # touching
    c = SA.auc(correct, [0.10, 0.05])      # disjoint
    assert a < b <= c


def test_auc_is_nan_when_a_class_is_empty():
    assert math.isnan(SA.auc([], [0.1]))
    assert math.isnan(SA.auc([0.9], []))


# --------------------------------------------------------------------------
# kl_divergence — overlap, NOT distance
# --------------------------------------------------------------------------
def test_kl_is_zero_for_identical_distributions():
    xs = [0.1, 0.3, 0.5, 0.7, 0.9]
    assert SA.kl_divergence(xs, list(xs)) == pytest.approx(0.0, abs=1e-12)


def test_kl_is_finite_when_a_bin_is_empty():
    """Laplace smoothing exists so an empty bin cannot produce inf and poison a receipt."""
    v = SA.kl_divergence([0.99] * 5, [0.01] * 5)
    assert math.isfinite(v) and v > 0


def test_kl_returns_nan_rather_than_a_misleading_zero_on_a_tiny_class():
    """A one-member class has no distribution. nan says so; 0.0 would read as 'no separation'."""
    assert math.isnan(SA.kl_divergence([0.9], [0.1, 0.2, 0.3]))


def test_kl_SATURATES_and_is_therefore_not_a_distance():
    """Pinning the measured surprise, so nobody reads KL as 'how far apart'.

    Measured 2026-08-30: correct ~0.95 against wrong ~0.72 scores HIGHER than the same correct
    against wrong ~0.05 -- further apart, lower KL. KL asks how badly Q fails to cover P, and
    once the supports are disjoint that is already maximal. `auc` is the monotone one.
    """
    correct = [0.95, 0.92, 0.98]
    near = SA.kl_divergence(correct, [0.72, 0.75, 0.71])
    far = SA.kl_divergence(correct, [0.05, 0.10, 0.02])
    assert far <= near, "if KL became monotone, auc's justification needs re-checking"


# --------------------------------------------------------------------------
# bootstrap_ci — no bare point estimates
# --------------------------------------------------------------------------
def test_bootstrap_ci_brackets_the_observed_rate():
    lo, hi = SA.bootstrap_ci([1, 0] * 6, reps=2000)
    assert lo < 0.5 < hi


def test_bootstrap_ci_is_degenerate_when_every_observation_agrees():
    """All-ones resamples to ones. A zero-width interval is correct here, not a bug."""
    assert SA.bootstrap_ci([1] * 20, reps=500) == (1.0, 1.0)


def test_bootstrap_ci_narrows_as_n_grows():
    """If more observations did not buy tighter bounds, raising k bought nothing."""
    small = SA.bootstrap_ci([1, 0] * 8, reps=3000)
    large = SA.bootstrap_ci([1, 0] * 64, reps=3000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_bootstrap_ci_is_reproducible_for_a_fixed_seed():
    xs = [1, 0, 1, 1, 0, 1, 0, 0]
    assert SA.bootstrap_ci(xs, reps=1000, seed=7) == SA.bootstrap_ci(xs, reps=1000, seed=7)


# --------------------------------------------------------------------------
# score_question — the shuffle is a CORRECTNESS property, not a nicety
# --------------------------------------------------------------------------
class _PositionScoringResponse:
    """Stands in for ollama, scoring purely by the POSITION an answer was SHOWN in.

    That is what a position-biased judge would do, so any scorer that fails to undo its own
    permutation is caught here rather than in a run.
    """

    def __init__(self, n):
        self._n = n

    def __enter__(self):
        payload = {"scores": [{"label": i + 1, "states_gold_claim": i == 0,
                               "note": "shown at position %d" % (i + 1),
                               "score": round(1.0 - 0.1 * i, 3)}
                              for i in range(self._n)]}
        self._body = json.dumps({"response": json.dumps(payload)}).encode()
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._body


@pytest.fixture
def stub_judge(monkeypatch):
    monkeypatch.setattr(SA.J, "load_rubric", lambda *a, **k: "RUBRIC BODY")
    monkeypatch.setattr(SA.J, "JUDGE_MODEL", "stub-model")
    seen = {"n": 2}

    def fake_urlopen(req, timeout=None):
        seen["prompt"] = json.loads(req.data.decode())["prompt"]
        return _PositionScoringResponse(seen["n"])

    monkeypatch.setattr(SA.urllib.request, "urlopen", fake_urlopen)
    return seen


ROW = {"question": "Why does bagging reduce variance?", "gold": "Averaging decorrelated trees."}


def test_permutation_is_undone_so_the_arm_shown_first_gets_the_top_score(stub_judge):
    """The silent failure this whole test file exists for.

    The judge gives ANSWER 1 the highest score. Whichever ARM was shown there must receive it.
    If the mapping is dropped, the first arm in the dict wins every time and it reads as a
    real effect.
    """
    stub_judge["n"] = 4
    entries = [("base", "a"), ("sft_5e6", "b"), ("sft_1e5", "c"), ("sft_2e5", "d")]
    for seed in range(30):
        out = SA.score_question(ROW, entries, random.Random(seed))
        shown_first = out["permutation"][0]
        assert out["per_arm"][shown_first]["score"] == pytest.approx(1.0)
        assert set(out["per_arm"]) == {"base", "sft_5e6", "sft_1e5", "sft_2e5"}


def test_the_shuffle_actually_shuffles_across_seeds(stub_judge):
    """A fixed order makes position bias indistinguishable from a genuine arm effect."""
    stub_judge["n"] = 4
    entries = [("a", "1"), ("b", "2"), ("c", "3"), ("d", "4")]
    perms = {tuple(SA.score_question(ROW, entries, random.Random(s))["permutation"])
             for s in range(30)}
    assert len(perms) > 1, "identical permutation for every seed"


def test_no_arm_reaches_the_top_slot_every_time(stub_judge):
    """Over many seeds each arm should sometimes go first. A biased shuffle is still a bias."""
    stub_judge["n"] = 4
    entries = [("a", "1"), ("b", "2"), ("c", "3"), ("d", "4")]
    firsts = collections_counter(
        SA.score_question(ROW, entries, random.Random(s))["permutation"][0] for s in range(80))
    assert len(firsts) == 4, "some arm never appeared first: %s" % dict(firsts)


def collections_counter(it):
    import collections
    return collections.Counter(it)


def test_arm_names_never_leak_into_the_judge_prompt(stub_judge):
    """If the judge can read which model wrote an answer, the shuffle is decorative."""
    stub_judge["n"] = 2
    SA.score_question(ROW, [("BASELINE_MODEL", "x"), ("SFT_WINNER", "y")], random.Random(0))
    assert "BASELINE_MODEL" not in stub_judge["prompt"]
    assert "SFT_WINNER" not in stub_judge["prompt"]


def test_identical_answers_are_shown_identically(stub_judge):
    """The degenerate case: the SAME checkpoint entered twice.

    The prompt must contain the two answers verbatim and indistinguishable. If the harness
    marked them apart in any way, a judge could separate them and 'arm A beat arm B' would be
    manufacturable from nothing.
    """
    stub_judge["n"] = 2
    SA.score_question(ROW, [("armA", "identical text"), ("armB", "identical text")],
                      random.Random(3))
    assert stub_judge["prompt"].count("identical text") == 2


def test_an_empty_generation_is_rendered_explicitly(stub_judge):
    """An empty answer must show as <EMPTY>, not blank space the judge may silently skip."""
    stub_judge["n"] = 2
    SA.score_question(ROW, [("a", ""), ("b", "real answer")], random.Random(0))
    assert "<EMPTY>" in stub_judge["prompt"]


def test_the_gold_and_question_reach_the_prompt(stub_judge):
    """Scoring without the gold would be a different measurement entirely."""
    stub_judge["n"] = 2
    SA.score_question(ROW, [("a", "x"), ("b", "y")], random.Random(0))
    assert ROW["gold"] in stub_judge["prompt"]
    assert ROW["question"] in stub_judge["prompt"]


def test_the_ranking_rule_comes_from_the_RUBRIC_not_this_prompt(monkeypatch):
    """One artifact, one hash.

    The ranking instruction used to be duplicated in score_arms' prompt, so this file and
    rescore_rubric.py read ONE rubric under TWO framings. The rubric is hashed into every
    receipt; a scoring rule stated only in the caller is invisible to that hash and drifts
    silently. The prompt must DEFER to the rubric, not restate it.
    """
    monkeypatch.setattr(SA.J, "load_rubric", lambda *a, **k: "<<<RUBRIC BODY MARKER>>>")
    monkeypatch.setattr(SA.J, "JUDGE_MODEL", "stub-model")
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["prompt"] = json.loads(req.data.decode())["prompt"]
        return _PositionScoringResponse(2)

    monkeypatch.setattr(SA.urllib.request, "urlopen", fake_urlopen)
    SA.score_question(ROW, [("a", "x"), ("b", "y")], random.Random(0))

    assert "<<<RUBRIC BODY MARKER>>>" in seen["prompt"], "the rubric must reach the judge"
    assert "RANK THEM AGAINST EACH OTHER" not in seen["prompt"], (
        "ranking rule restated in the caller — it belongs in the hashed rubric")


def test_retries_ESCALATE_the_timeout_rather_than_repeating_it(monkeypatch):
    """Retrying a timeout with the same timeout reproduces the same failure.

    Measured on eval q8, 2026-08-30: the identical call took 192.3 s cold and 18.3 s warm --
    same 202-token output, same done_reason, parses fine both times. Latency is BIMODAL
    (ollama KV cache on a 2808-token prompt), so a budget learned from warm calls evicts cold
    ones. The original loop then retried three times at one timeout, guaranteeing three
    identical failures. Each attempt must get materially more room.
    """
    monkeypatch.setattr(SA.J, "load_rubric", lambda *a, **k: "RUBRIC")
    monkeypatch.setattr(SA.J, "JUDGE_MODEL", "stub")
    monkeypatch.setattr(SA.J, "BUDGET", SA.J.LatencyBudget(bootstrap=30.0))
    seen = []

    def fake_call(prompt, schema, model, timeout=None, num_predict=None):
        seen.append(timeout)
        return None                      # always fail, so all three attempts run

    monkeypatch.setattr(SA.J, "_call", fake_call)
    # 2026-09-06: the single-answer fallback (no-missing-evals law) would fire after
    # the comparative attempts and route through J._call again; stub it out so this
    # test keeps measuring ONLY the comparative escalation ladder.
    monkeypatch.setattr(SA.J, "judge", lambda *a, **k: {"score": None})
    out = SA.score_question(ROW, [("a", "x"), ("b", "y")], random.Random(0))

    assert len(seen) == 3, "expected three attempts, got %d" % len(seen)
    assert seen[0] < seen[1] < seen[2], "timeouts did not escalate: %s" % seen
    assert seen[2] >= 4 * seen[0], "final attempt barely wider than the first: %s" % seen
    assert out["failed"] is True, "exhausted retries must mark the question failed, not score it"


def test_num_predict_scales_with_the_number_of_arms(monkeypatch):
    """~90 output tokens per answer. A fixed 256 cut 4-arm calls mid-JSON and dropped 44% of
    them -- a truncation that looked exactly like a timeout from outside."""
    monkeypatch.setattr(SA.J, "load_rubric", lambda *a, **k: "RUBRIC")
    monkeypatch.setattr(SA.J, "JUDGE_MODEL", "stub")
    monkeypatch.setattr(SA.J, "BUDGET", SA.J.LatencyBudget(bootstrap=30.0))
    caps = {}

    def fake_call(prompt, schema, model, timeout=None, num_predict=None):
        caps[len(prompt)] = num_predict
        return {"scores": []}

    monkeypatch.setattr(SA.J, "_call", fake_call)
    # 2026-09-06: stub the single-answer fallback for the same reason as above —
    # its internal call would overwrite `caps` with a num_predict this test
    # does not measure.
    monkeypatch.setattr(SA.J, "judge", lambda *a, **k: {"score": None})
    SA.score_question(ROW, [("a", "x"), ("b", "y")], random.Random(0))
    two = list(caps.values())[-1]
    caps.clear()
    SA.score_question(ROW, [("a", "x"), ("b", "y"), ("c", "z"), ("d", "w")], random.Random(0))
    four = list(caps.values())[-1]
    assert four > two, "num_predict did not grow with arm count (%s vs %s)" % (four, two)
    assert four >= 256 + 160 * 4


def test_judge_score_is_clamped_to_unit_interval(monkeypatch):
    """Task #39: measured leak — data/past9_per_receipt.json r3-0007 carried
    score=3.0 while the judge's own note said "Scored 0.98", inflating
    mean|correct to an impossible 1.362. Scores are clamped at ingestion."""
    def fake_call(prompt, schema, model, timeout=0, num_predict=0):
        return {"scores": [
            {"label": 1, "score": 3.0, "states_gold_claim": True, "note": "Scored 0.98."},
            {"label": 2, "score": -0.5, "states_gold_claim": False, "note": "wrong"},
        ]}
    monkeypatch.setattr(SA.J, "_call", fake_call)
    r = SA.score_question(
        {"prompt": "q", "gold": "g", "context": ""},
        [("armA", "answer a"), ("armB", "answer b")],
        random.Random(0),
    )
    scores = sorted(v["score"] for v in r["per_arm"].values())
    assert scores == [0.0, 1.0], f"scores not clamped to [0,1]: {scores}"


# --------------------------------------------------------------------------
# score_question — NO MISSING EVALS (operator, 2026-09-06: "I have 16 eval
# questions and I fucking expect 16 evaluations")
# --------------------------------------------------------------------------
_ROW = {"question": "Q?", "gold": "G", "category": "Math"}


def _entry_scores(labels_scores):
    return {"scores": [{"label": l, "score": s, "states_gold_claim": True, "note": ""}
                       for l, s in labels_scores]}


def test_partial_parse_escalates_until_every_answer_is_scored(monkeypatch):
    """A judge reply covering only SOME answers used to be accepted silently,
    leaving unscored cells with no log line (surpass11 receipt, r3-0042)."""
    calls = []
    def fake_call(prompt, schema, model, timeout, num_predict):
        calls.append(timeout)
        if len(calls) == 1:
            return _entry_scores([(1, 0.9)])                 # partial: 1 of 3
        return _entry_scores([(1, 0.9), (2, 0.2), (3, 0.8)])  # complete
    monkeypatch.setattr(SA.J, "_call", fake_call)
    monkeypatch.setattr(SA.J, "judge",
                        lambda *a, **k: pytest.fail("fallback must not fire — "
                                                    "the retry completed the parse"))
    r = SA.score_question(_ROW, [("a", "x"), ("b", "y"), ("c", "z")], random.Random(7))
    assert len(calls) == 2, "partial parse must escalate like a timeout"
    assert not r["failed"] and r["missing"] == []
    assert set(r["per_arm"]) == {"a", "b", "c"}
    assert not any(v.get("fallback") for v in r["per_arm"].values())


def test_single_answer_fallback_fills_cells_the_comparative_call_never_scored(monkeypatch):
    monkeypatch.setattr(SA.J, "_call",
                        lambda *a, **k: _entry_scores([(1, 0.9), (2, 0.2)]))  # never label 3
    seen = []
    def fake_single(question, gold, answer, category=""):
        seen.append((question, gold, answer, category))
        return {"score": 0.75, "verdict": "correct"}
    monkeypatch.setattr(SA.J, "judge", fake_single)
    r = SA.score_question(_ROW, [("a", "x"), ("b", "y"), ("c", "z")], random.Random(7))
    assert not r["failed"] and r["missing"] == []
    fb = [arm for arm, v in r["per_arm"].items() if v.get("fallback")]
    assert len(fb) == 1, "exactly the never-scored arm takes the fallback protocol"
    assert r["per_arm"][fb[0]]["score"] == 0.75
    assert "single-answer fallback" in r["per_arm"][fb[0]]["note"]
    assert len(seen) == 1 and seen[0][3] == "Math"


def test_a_cell_no_protocol_could_score_marks_the_question_failed(monkeypatch):
    monkeypatch.setattr(SA.J, "_call", lambda *a, **k: None)
    monkeypatch.setattr(SA.J, "judge", lambda *a, **k: {"score": None})
    r = SA.score_question(_ROW, [("a", "x"), ("b", "y")], random.Random(7))
    assert r["failed"] and set(r["missing"]) == {"a", "b"}
    assert r["per_arm"] == {}


def test_main_refuses_to_complete_an_incomplete_receipt():
    """BREAK FAST: the receipt persists its evidence (complete/missing fields),
    then the process exits nonzero so no pipeline reads a floor as a verdict."""
    src = open("src/rlv2/eval/score_arms.py", encoding="utf-8").read()
    body = src.split("def main")[1]
    i_dump = body.find('"complete": not missing_cells')
    i_exit = body.find("sys.exit(2)")
    assert 0 <= i_dump < i_exit, "receipt must persist completeness, THEN fail"
    assert '"missing": [list(c) for c in missing_cells]' in body
    assert "RECEIPT INCOMPLETE" in body


# --------------------------------------------------------------------------
# --protocol single — THE LOCKED MECHANISM (operator, 2026-09-06: "you need a
# clear scoring mechanism ... so we don't double guess ourselves")
# --------------------------------------------------------------------------
def _single_rig(tmp_path, monkeypatch, k=1):
    """Two fake arms with cached generations so main() never touches a GPU."""
    import os as _os
    for arm in ("good", "bad"):
        d = tmp_path / arm
        d.mkdir()
        (d / "config.json").write_text("{}", encoding="utf-8")
    ev = tmp_path / "eval.jsonl"
    ev.write_text(json.dumps({"item_id": "e1", "question": "Q1", "gold": "G1"}) + "\n" +
                  json.dumps({"item_id": "e2", "question": "Q2", "gold": "G2"}) + "\n",
                  encoding="utf-8")
    out = tmp_path / "receipt.json"
    gens = {"good": [["right one", "right two"]], "bad": [["wrong one", "wrong two"]]}
    (tmp_path / "receipt.json.generations.json").write_text(json.dumps(gens),
                                                            encoding="utf-8")
    argv = ["score_arms", "--arm", "good=%s" % (tmp_path / "good"),
            "--arm", "bad=%s" % (tmp_path / "bad"),
            "--eval", str(ev), "--k", str(k), "--temperature", "0.0",
            "--protocol", "single", "--out", str(out)]
    monkeypatch.setattr(_s, "argv", argv)
    return out


def test_single_protocol_majority_counts_and_receipt_shape(tmp_path, monkeypatch):
    calls = []
    def fake_judge(question, gold, answer, category=""):
        calls.append(answer)
        # real judge replies always carry ok; the locked path now requires it
        return {"score": 0.9 if "right" in answer else 0.1, "ok": True}
    monkeypatch.setattr(SA.J, "judge", fake_judge)
    out = _single_rig(tmp_path, monkeypatch)
    SA.main()
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["protocol"] == "single"
    assert d["complete"] is True
    assert d["summary"]["good"]["n_correct"] == 2
    assert d["summary"]["bad"]["n_correct"] == 0
    assert len(calls) == 2 * 2 * 3, "each cell must be judged exactly 3 times"
    # per-cell reads persisted in both the records and the resume sidecar
    assert all(len(v["reads"]) == 3 for r in d["records"] for v in r["scores"].values())
    side = out.parent / (out.name + ".cells.jsonl")
    assert sum(1 for _ in open(side, encoding="utf-8")) == 4


def test_single_protocol_resumes_from_the_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(SA.J, "judge",
                        lambda *a, **k: {"score": 0.9, "ok": True})
    out = _single_rig(tmp_path, monkeypatch)
    SA.main()
    # second run: judging again would be a resume failure
    monkeypatch.setattr(SA.J, "judge",
                        lambda *a, **k: pytest.fail("sidecar cells were re-judged"))
    SA.main()
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["summary"]["good"]["n_correct"] == 2


def test_single_protocol_breaks_fast_when_a_cell_has_no_majority(tmp_path, monkeypatch):
    monkeypatch.setattr(SA.J, "judge", lambda *a, **k: {"score": None})
    out = _single_rig(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as e:
        SA.main()
    assert e.value.code == 2
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["complete"] is False and len(d["missing"]) == 4


def test_unparseable_judge_reply_is_a_missing_reading_not_a_zero(tmp_path, monkeypatch):
    """Measured 2026-09-07: 'The value is 0.5^2 = 0.25.' against gold '0.25' read
    0.0 three times. judge.FAILED carries score 0.0 with ok=False, and the locked
    path was recording it as a genuine WRONG answer — silently depressing every
    count. A failed parse is retried once, then dropped from the vote."""
    calls = []
    def flaky_judge(question, gold, answer, category=""):
        calls.append(answer)
        # first attempt of every reading fails to parse, retry succeeds
        if len(calls) % 2 == 1:
            return {"score": 0.0, "ok": False, "verdict": "not_attempted"}
        return {"score": 0.95, "ok": True, "verdict": "correct"}
    monkeypatch.setattr(SA.J, "judge", flaky_judge)
    out = _single_rig(tmp_path, monkeypatch)
    SA.main()
    d = json.loads(out.read_text(encoding="utf-8"))
    # every cell recovered on its retry -> both arms fully correct, nothing zeroed
    assert d["summary"]["good"]["n_correct"] == 2
    assert d["summary"]["bad"]["n_correct"] == 2
    assert d["complete"] is True


def test_cells_whose_readings_all_fail_to_parse_break_the_receipt(tmp_path, monkeypatch):
    """If no reading survives, the cell has no verdict — it must show up as
    MISSING and fail the run, never as a confident zero."""
    monkeypatch.setattr(SA.J, "judge",
                        lambda *a, **k: {"score": 0.0, "ok": False})
    out = _single_rig(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as e:
        SA.main()
    assert e.value.code == 2
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["complete"] is False and len(d["missing"]) == 4


def test_resume_re_judges_cells_that_could_not_vote(tmp_path, monkeypatch):
    """Measured 2026-09-07 on data/cov119_locked.json: a cell whose three
    readings all failed to parse was written to the sidecar as [null,null,null],
    and every rerun then 'resumed' it as done — so the receipt could never be
    completed. A cached cell counts as done only if it can still vote."""
    out = _single_rig(tmp_path, monkeypatch)
    side = out.parent / (out.name + ".cells.jsonl")
    # one dead cell + one good cell, pre-seeded as if a prior run wrote them
    side.write_text(
        json.dumps({"arm": "good", "qid": "e1", "rep": 0,
                    "reads": [None, None, None]}) + "\n" +
        json.dumps({"arm": "good", "qid": "e2", "rep": 0,
                    "reads": [0.9, 0.9, 0.9]}) + "\n", encoding="utf-8")
    seen = []
    def fake_judge(question, gold, answer, category=""):
        seen.append(answer)
        return {"score": 0.95, "ok": True}
    monkeypatch.setattr(SA.J, "judge", fake_judge)
    SA.main()
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["complete"] is True, "the dead cell was never re-judged"
    assert d["summary"]["good"]["n_correct"] == 2
    # the good cached cell was NOT re-judged; the dead one was
    assert "right two" not in seen, "a usable cached cell was needlessly re-judged"
    assert "right one" in seen, "the dead cell was skipped on resume"
