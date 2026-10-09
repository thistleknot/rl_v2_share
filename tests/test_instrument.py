"""The pinned instrument. Drift fails at second 1, not at hour 3.

NO GOVERNING SPEC. Basis: operator 2026-09-15 -- "Just give me a unified approach where we
dont have these late gotcha regressions".

FOUR DRIFTS IN ONE SESSION, every one found by running into it:

  1. judge.py had no JUDGE_REMOTE handling -> score_arms scored the eval on the local 4b
     while every straddle number came from OpenRouter's 9b.
  2. _openrouter_call accepted a schema and discarded it -> the remote judge was never
     schema-constrained and every remote reading was thrown away as unparseable.
  3. The rubric carried an uncommitted edit; reverting moved us to a THIRD instrument.
  4. The banner printed the local judge's name unconditionally -> the log named the wrong
     instrument and was twice read as confirmation.

All four are one-line comparisons against a pinned value. That is what this module is.
"""
import io
import json
import sys

import pytest

from rlv2.train.rl import instrument as I  # noqa: E402


@pytest.fixture
def cfg():
    return I.load()


# ------------------------------------------------------------ the pin ----
def test_the_pinned_rubric_is_the_campaign_the_playbook_froze(cfg):
    """playbook.md:4051 records 9470adb9f39935ac. It is the STRIPPED hash -- a
    byte-for-byte compare against git matches nothing, which is exactly how the
    campaign's own instrument came to be called unrecoverable. Twice."""
    assert cfg["rubric_sha_stripped"] == "9470adb9f39935ac"
    assert I._rubric_sha_stripped(cfg["rubric_path"]) == cfg["rubric_sha_stripped"]


def test_the_live_instrument_matches_the_pin(cfg):
    """If this fails, either something drifted or the pin was changed without a
    campaign bump. Both are things to look at before trusting any number."""
    import os
    os.environ["JUDGE_REMOTE"] = "1"
    assert I.diff(cfg) == []


# ----------------------------------------------- what it must refuse ----
def test_it_refuses_when_the_remote_judge_is_not_enabled(cfg, monkeypatch):
    """Defect 1, as a check. Without JUDGE_REMOTE the eval silently scores on the
    local 4b -- a different judge from the one every other number used."""
    monkeypatch.delenv("JUDGE_REMOTE", raising=False)
    d = I.diff(cfg)
    assert any("JUDGE_REMOTE" in x for x in d), d


def test_it_refuses_when_the_rubric_moved(cfg, tmp_path, monkeypatch):
    """Defect 3. A one-line edit to the rubric changes what every past number meant."""
    moved = tmp_path / "rubric.md"
    moved.write_text("a different rubric entirely", encoding="utf-8")
    c = dict(cfg, rubric_path=str(moved))
    assert any("rubric:" in x for x in I.diff(c))


def test_it_refuses_when_a_mechanical_gate_is_missing(cfg, monkeypatch):
    """The gates are part of the instrument: remove one and answers that scored 0.0
    start reaching the judge, which pays for silence."""
    monkeypatch.setattr(I, "_judge_src", lambda: "def judge(): pass")
    assert any("gates missing" in x for x in I.diff(cfg))


def test_it_refuses_when_the_eval_set_changes_size(cfg, tmp_path):
    """instructions.md 1.4: the eval is FIXED for the pipeline."""
    ev = tmp_path / "eval.jsonl"
    ev.write_text('{"prompt":"x","gold":"y"}\n', encoding="utf-8")
    c = dict(cfg, eval_file=str(ev))
    assert any("eval:" in x for x in I.diff(c))


def test_verify_raises_rather_than_warning_by_default(cfg, monkeypatch):
    """A warning gets scrolled past in a 4000-line log. The whole point is that a
    drifted run does not happen at all."""
    monkeypatch.delenv("JUDGE_REMOTE", raising=False)
    with pytest.raises(SystemExit):
        I.verify()


# ------------------------------------------------------- the receipt ----
def test_the_stamp_carries_everything_needed_to_reproduce_a_number():
    """7/16 became unverifiable because its receipt named a rubric sha and nothing
    else -- no judge, no protocol, no gates. A number without its instrument is a
    number nobody can check."""
    import os
    os.environ["JUDGE_REMOTE"] = "1"
    s = I.stamp()
    for key in ("campaign", "rubric_sha_stripped", "judge_model", "judge_remote",
                "eval_file", "eval_n", "protocol", "reads_per_cell",
                "k_samples_per_question", "correct_floor", "gates"):
        assert key in s, key
    assert s["drift"] is None


def test_the_config_records_the_known_delta_from_the_old_campaign(cfg):
    """never_states_an_answer did not exist on 2026-09-12. Expected effect is nil --
    it fires only where the rubric's band already scores 0.0 -- but 'expected nil'
    is not 'identical', and a receipt must not claim parity it does not have."""
    assert "gate_delta_note" in cfg
    assert "never_states_an_answer" in cfg["gate_delta_note"]


def test_the_config_records_the_measured_resolution_limit(cfg):
    """3 of 96 cells had reads on both sides of 0.70, each worth 6.25 points on a
    16-question score. A 1-question difference is inside the judge's own noise."""
    assert "resolution_note" in cfg
    assert "6.25" in cfg["resolution_note"]


# -------------------------------------------------------- the wiring ----
@pytest.mark.parametrize("path", [
    "src/rlv2/eval/score_arms.py", "archive/tools/straddle_probe.py", "archive/tools/phase_runner.py"])
def test_every_scorer_verifies_before_it_scores(path):
    src = io.open(path, encoding="utf-8").read()
    assert "_verify_instrument()" in src, path
