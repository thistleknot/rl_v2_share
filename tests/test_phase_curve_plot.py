"""The two-axis phase plot: straddle and accuracy, per epoch/step, per phase.

NO GOVERNING SPEC. Basis: operator 2026-09-15 -- "I want a plot showing / Straddle x
accuracy (/16) / For each epoch/step / For the best hpo parms we did full with / For both
phases / Sft / rl".

WHY TWO AXES AND NOT ONE. They are different objectives and they are ALLOWED to disagree:
SFT that raises straddle and leaves accuracy flat has done its job; RL that raises accuracy
by spending straddle has done its. Collapsing them to one line hides the trade, which is
the only thing the plot is for.

WHY MISSING POINTS STAY MISSING. A gap here is a measurement nobody took. Interpolating it
turns NOT TESTED into a data point, which is the Article X failure in picture form.
"""
import importlib.util
import io
import json
import os
import sys

import pytest

SRC = "archive/tools/phase_curve_plot.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("phase_curve_plot", SRC)
    m = importlib.util.module_from_spec(spec)
    sys.modules["phase_curve_plot"] = m
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------- readers ----
def test_straddle_receipts_are_keyed_by_checkpoint(mod, tmp_path):
    p = tmp_path / "straddle_x.json"
    p.write_text(json.dumps([
        {"checkpoint": "sft_hpo_4_epoch2", "straddle": 9, "records": 24}]),
        encoding="utf-8")
    got = mod.straddle_rows(str(tmp_path / "straddle_*.json"))
    assert got == {"sft_hpo_4_epoch2": (9, 24)}


def test_eval_receipts_collapse_on_the_checkpoint_not_the_arm_nickname(mod, tmp_path):
    """The same checkpoint is scored under different labels across runs -- `arm4`
    in one invocation, `sft_hpo_4` in another. Keyed on the nickname they would
    plot as two different models."""
    for i, nick in enumerate(("arm4", "sft_hpo_4")):
        (tmp_path / ("evalacc_%d.json" % i)).write_text(json.dumps({
            "arms": {nick: "sft_arms/sft_hpo_4_epoch2"},
            "summary": {nick: {"n_correct": 2}}}), encoding="utf-8")
    got = mod.eval_rows(str(tmp_path / "evalacc*.json"))
    assert got == {"sft_hpo_4_epoch2": 2}, got


def test_an_unscored_arm_is_omitted_rather_than_counted_zero(mod, tmp_path):
    """A receipt with no n_correct is an unfinished measurement, not a zero --
    the arm2 case, where 15 of 16 generations were empty."""
    (tmp_path / "evalacc_z.json").write_text(json.dumps({
        "arms": {"a": "ckpt/a"}, "summary": {"a": {"n_obs": 16}}}), encoding="utf-8")
    assert mod.eval_rows(str(tmp_path / "evalacc*.json")) == {}


def test_a_corrupt_receipt_is_skipped_not_fatal(mod, tmp_path):
    (tmp_path / "straddle_bad.json").write_text("{ not json", encoding="utf-8")
    assert mod.straddle_rows(str(tmp_path / "straddle_*.json")) == {}


def test_the_rl_curve_comes_from_the_runs_own_lift_lines(mod, tmp_path):
    log = tmp_path / "rl.log"
    log.write_text("\n".join([
        "[lift] step 9: 2/16 (start 3, best 3)",
        "noise",
        "[lift] step 19: 6/16 (start 3, best 6)"]), encoding="utf-8")
    assert mod.rl_curve(str(log)) == [(9, 2), (19, 6)]


def test_a_missing_rl_log_is_an_empty_curve_not_a_crash(mod, tmp_path):
    assert mod.rl_curve(str(tmp_path / "nope.log")) == []


# ---------------------------------------------------------- the picture ----
def test_it_reports_what_it_could_not_measure(mod, tmp_path, monkeypatch):
    """The Article X requirement in picture form: a gap is a measurement nobody
    took, and the plot must SAY so rather than joining the line across it."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("sft_arms/stem_epoch1")
    os.makedirs("sft_arms/stem_epoch2")
    os.makedirs("data")
    # straddle for epoch2 only; accuracy for neither
    io.open("data/straddle_a.json", "w", encoding="utf-8").write(json.dumps(
        [{"checkpoint": "stem_epoch2", "straddle": 9, "records": 24}]))
    missing = mod.build(str(tmp_path / "out.png"), "stem", str(tmp_path / "none.log"))
    assert os.path.exists(tmp_path / "out.png")
    assert any("stem_epoch1" in m for m in missing)
    assert any("stem_epoch2" in m for m in missing), "accuracy is absent for both"


def test_a_phase_with_no_data_at_all_still_renders(mod, tmp_path, monkeypatch):
    """RL before its full run has nothing to show, and an empty panel that says
    NOT MEASURED YET is more useful than a crash or a blank file."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("data")
    mod.build(str(tmp_path / "empty.png"), "nothing", str(tmp_path / "none.log"))
    assert os.path.exists(tmp_path / "empty.png")


def test_the_two_metrics_are_on_separate_axes(mod):
    src = io.open(SRC, encoding="utf-8").read()
    assert "twinx()" in src, "one shared axis would imply they are the same quantity"
    assert "accuracy  (correct of %d)" in src and "straddle rate" in src
