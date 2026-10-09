"""The Stop hook that keeps the session monitoring.

NO GOVERNING SPEC. Basis: operator 2026-09-14 -- "Dont stop monitoring. Continue to
monitor every 300s until e2e confirmed. This should be a rule you hook thru."

WHY THIS IS A HOOK. The tool layer refuses chained sleeps, and a background-task
notification ENDS the turn -- so a session paced by notifications reads as an agent that
keeps quitting. A Stop hook is the only construct that can sleep and then REFUSE the
stop, which keeps the turn alive without spinning.

THE THREE WAYS IT COULD FAIL SILENTLY, each pinned below:
  1. blocking forever because a sentinel was never written
  2. sleeping on an idle GPU because "running" was never actually checked
  3. blocking without showing the operator anything, which is a held session with no
     information -- worse than stopping (Article XI.5)
"""
import importlib.util
import io
import json
import os
import sys

import pytest

SRC = ".claude/hooks/e2e_watch.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("e2e_watch", SRC)
    m = importlib.util.module_from_spec(spec)
    sys.modules["e2e_watch"] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def src():
    return io.open(SRC, encoding="utf-8").read()


# ------------------------------------------------------------- the exits ----
def test_the_sentinel_releases_the_session(mod, src):
    """Without a release the operator cannot end a session at all."""
    assert "if os.path.exists(SENTINEL):" in src
    i = src.index("if os.path.exists(SENTINEL):")
    assert "sys.exit(0)" in src[i:i + 200]
    assert src.index("os.path.exists(SENTINEL)") < src.index("time.sleep(INTERVAL)"), \
        "the sentinel must be checked BEFORE the sleep, not after"


def test_it_never_sleeps_on_an_idle_gpu(mod, src):
    """CLAUDE.md 1.1: "If nothing is running ... say so explicitly rather than
    sleeping on an empty GPU." Blocking with no process alive holds the session
    open for a run that already finished."""
    assert "if not live():" in src
    i = src.index("if not live():")
    assert "sys.exit(0)" in src[i:i + 160]
    assert src.index("if not live():") < src.index("time.sleep(INTERVAL)")


def test_there_is_a_cap_so_a_forgotten_sentinel_cannot_hold_forever(mod):
    assert mod.MAX_BLOCKS > 0
    # 300s x 240 = 20h: far above any real campaign, but finite.
    assert mod.INTERVAL * mod.MAX_BLOCKS >= 12 * 3600


def test_the_interval_is_the_one_the_operator_asked_for(mod):
    assert mod.INTERVAL == 300


# --------------------------------------------------------- what it watches ----
def test_every_phase_trainer_counts_as_running(mod):
    """Windows reports them all as python.exe, so the match is on the command
    line. Missing one means the hook releases the session mid-phase."""
    for tool in ("phase_runner.py", "vibe_sft.py", "straddle_probe.py",
                 "reft_hpo.py", "grpo_one_file.py"):
        assert tool in mod.WATCHED, tool


def test_a_block_always_carries_log_lines(src):
    """Article XI.5: a held session with no visible evidence is worse than a
    stopped one. Both block branches must call newest_lines()."""
    body = src[src.index("time.sleep(INTERVAL)"):]
    assert body.count("newest_lines(") == 2, \
        "both the still-running and the exited branch must surface lines"
    assert '"decision": "block"' in body


def test_the_signal_filter_keeps_the_lines_that_carry_results(mod):
    """A tail of raw log is mostly progress bars. These are the lines a verdict
    is read from."""
    keep = [
        "[straddle] sft_hpo_0 straddle  9/24 ( 38%)  solved 2  missed 13",
        "[step  2] 6/16  vs start 5/16  lift +1",
        "[eval] 7/16 correct",
        "[priority] 60 measured, 56 live (93%)",
        "  -> start=5 best=6 lift=+1",
        "Traceback (most recent call last):",
        "CUDA out of memory. Tried to allocate 2.00 GiB",
    ]
    for line in keep:
        assert mod.SIGNAL.search(line), line
    drop = [
        "  0%|          | 0/13 [00:00<?, ?it/s]",
        "Loading weights:  40%|####      | 128/320",
        "The fast path is not available because one of the required library",
    ]
    for line in drop:
        assert not mod.SIGNAL.search(line), line


def test_the_exited_branch_names_both_bars(src):
    """instructions.md 1.2a: an HPO pass and a phase win are different claims and
    must never share a threshold. The wake-up message is where that gets
    conflated, so it states both."""
    body = src[src.index("EXITED"):]
    assert "BOTH bars" in body
    assert "its own start" in body and "previous phase" in body


# -------------------------------------------------------- the registration ----
def test_it_is_registered_on_stop_with_a_timeout_above_its_own_sleep(mod):
    """A timeout below the sleep kills the hook mid-sleep and the stop goes
    through unblocked -- which is indistinguishable from the hook working."""
    cfg = json.loads(io.open(".claude/settings.local.json", encoding="utf-8").read())
    entries = [h for group in cfg.get("hooks", {}).get("Stop", [])
               for h in group.get("hooks", [])
               if "e2e_watch.py" in h.get("command", "")]
    assert entries, "the hook exists but nothing calls it"
    assert entries[0]["timeout"] > mod.INTERVAL, entries[0]["timeout"]


def test_the_superseded_campaign_hook_is_gone():
    """chain_sleep_watch.py did this correctly for one hardcoded campaign
    (pid 28972, .rl_chain3.log). Two watchers would sleep in series."""
    assert not os.path.exists(".claude/hooks/chain_sleep_watch.py")


def test_the_filter_sees_progress_during_a_long_measurement(mod):
    """The defect this closes: a 24-record x K=4 straddle measurement runs ~16
    minutes emitting only reward, judge-vram and per-record progress lines. None
    matched the original filter, so every wake-up during the longest stretch of
    the campaign reported "(no signal lines yet)" on a perfectly healthy run --
    a monitoring instrument that goes blind exactly when it is needed.
    """
    for line in ["[reward] 2/4 correct at 0.70 | zero-sum drift 0.0e+00",
                 "  [sft_hpo_0_epoch3] 5/24 generated",
                 "  [ledger] skills_model: 0 of 24 from ledger, 24 to generate"]:
        assert mod.SIGNAL.search(line), line


def test_the_tail_is_long_enough_to_reach_a_progress_line(mod, src):
    """Progress prints every 5 records; between them the judge emits several
    hundred characters per rollout, so a short tail can contain none of them."""
    assert "tail=20000" in src
