"""The wall that stops a tidy-up from destroying GPU-hours.

NO GOVERNING SPEC. Basis: operator 2026-09-15 -- "Ensure we dont have a repeat of this
issue", after `rm -rf sft_arms` destroyed five trained SFT checkpoints as a "clean slate"
step before a re-run. The JSON receipts survived; the weights did not, which turned an
optional ~15 min re-score into a forced ~3h re-train.

WHY A HOOK AND NOT A RULE. The intent was never "delete the weights" -- it was "start
clean", and the weights were collateral. A rule cannot catch a command whose author does
not realise what it is about to hit. A PreToolUse deny can, because it reads the
filesystem instead of the intent.

THE ASYMMETRY IT ENCODES. A stale checkpoint costs disk. A deleted one costs the GPU-hours
that produced it, and the loss surfaces only when something downstream asks for it.
"""
import io
import json
import os
import subprocess
import sys

import pytest

HOOK = ".claude/hooks/protect_checkpoints.py"
RETIRE = "archive/tools/retire.py"


def run_hook(command, tool="Bash"):
    """Feed the hook a tool call the way the harness does; return its decision."""
    payload = json.dumps({"tool_name": tool, "tool_input": {"command": command}})
    out = subprocess.run([sys.executable, HOOK], input=payload,
                         capture_output=True, text=True).stdout.strip()
    if not out:
        return None
    return json.loads(out)["hookSpecificOutput"]


@pytest.fixture
def ckpt_dir(tmp_path, monkeypatch):
    """A directory that looks like a checkpoint store."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("protect", HOOK)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    d = tmp_path / "arms" / "arm0_epoch3"
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"\x00" * 32)
    (d / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(m, "ROOT", str(tmp_path))
    return m, str(tmp_path / "arms")


# --------------------------------------------------- what it must block ----
def test_it_blocks_the_exact_command_that_cost_us_the_checkpoints():
    """`rm -rf sft_arms; mkdir -p sft_arms` -- a clean-slate idiom, and it took five
    trained checkpoints with it."""
    d = run_hook("rm -rf sft_arms; mkdir -p sft_arms")
    assert d and d["permissionDecision"] == "deny"
    assert "model weight file" in d["permissionDecisionReason"]
    assert "archive/tools/retire.py" in d["permissionDecisionReason"], \
        "a deny must name the thing to do instead, or it just gets worked around"


@pytest.mark.parametrize("cmd", [
    "rm -rf sft_arms",
    "rm -fr sft_arms",
    "rm -r sft_arms",
    "cd /x && rm -rf sft_arms",
    "Remove-Item -Recurse -Force sft_arms",
])
def test_every_spelling_of_a_recursive_delete_is_caught(cmd):
    """One uncaught spelling is the whole guard defeated."""
    d = run_hook(cmd, tool="PowerShell" if "Remove-Item" in cmd else "Bash")
    assert d and d["permissionDecision"] == "deny", cmd


# ----------------------------------------------- what it must NOT block ----
@pytest.mark.parametrize("cmd", [
    "rm -rf .tmp/scratch",
    "rm -rf data/generations_cache",
    "rm -f data/sft_sweep_evalacc.json",
    "ls -la sft_arms",
    "python archive/tools/retire.py sft_arms",
])
def test_it_stays_out_of_the_way_of_everything_else(cmd):
    """A guard that blocks ordinary work gets disabled, and then it guards nothing.
    Directories with no weights, single files, and reads all pass."""
    assert run_hook(cmd) is None, cmd


def test_naming_one_weight_file_explicitly_is_allowed(ckpt_dir):
    """If you name a single file you have looked at it. The block is on RECURSIVE
    deletes that sweep up weights as a side effect of tidying something else."""
    m, arms = ckpt_dir
    assert m.weight_count(os.path.join(arms, "arm0_epoch3", "model.safetensors")) == 0


def test_it_counts_weights_recursively(ckpt_dir):
    m, arms = ckpt_dir
    assert m.weight_count(arms) == 1
    assert m.weight_count(os.path.join(arms, "arm0_epoch3")) == 1


def test_a_missing_path_is_not_an_error(ckpt_dir):
    m, arms = ckpt_dir
    assert m.weight_count(os.path.join(arms, "does_not_exist")) == 0


# ------------------------------------------------------- the way out ----
def test_retire_moves_rather_than_removes(tmp_path, monkeypatch):
    """The deny is only legitimate because there IS a safe alternative."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("retire", RETIRE)
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "arms"
    src.mkdir()
    (src / "model.safetensors").write_bytes(b"\x00" * 16)

    dest = r.retire("arms", stamp="T0")
    assert not os.path.exists("arms"), "the original is out of the way"
    assert os.path.exists(os.path.join(dest, "model.safetensors")), \
        "and the weights are still on disk"


def test_purging_is_a_separate_deliberate_act(tmp_path, monkeypatch):
    """Retiring must not quietly become deleting -- that would rebuild the defect
    one level down. Freeing space is its own command with its own stamp."""
    src_ = io.open(RETIRE, encoding="utf-8").read()
    assert '"--purge"' in src_, "there must BE a way to free space, or this gets bypassed"
    assert "rmtree" in src_, "and it must actually delete, not pretend to"
    # the flag's own help has to say why it is separate -- a future reader who does not
    # know the 2026-09-15 loss will otherwise fold purge back into retire as a 'cleanup'
    i = src_.index('ap.add_argument("--purge"')
    assert "separate and deliberate" in src_[i:i + 500].lower()


# ------------------------------------------------------ registration ----
def test_the_hook_is_actually_wired_to_pretooluse():
    """An unregistered guard is a file, not a wall. chain_sleep_watch.py sat
    unregistered for two weeks doing nothing."""
    cfg = json.loads(io.open(".claude/settings.local.json", encoding="utf-8").read())
    entries = [h for g in cfg.get("hooks", {}).get("PreToolUse", [])
               for h in g.get("hooks", [])
               if "protect_checkpoints" in h.get("command", "")]
    assert entries, "the hook exists but nothing calls it"
