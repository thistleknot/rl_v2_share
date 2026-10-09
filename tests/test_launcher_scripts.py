r"""Guards the five pipeline launchers against silent rot.

The 2026-08-24 restructure moved root scripts into src/ and sprawl/. The .py
importers were swept, the .ps1 launchers were not, and the breakage hid for two
days: run_chain.ps1 still invoked $PSScriptRoot\hpo.py while its own log lines
already printed the new archive/src/hpo.py path, and three pipeline scripts called
`from rlv2.judge.reward import ...` after reward.py moved to src/ -- swallowed by
`2>&1 | Out-Null`, so Ollama silently stopped unloading.

Both defects are mechanically detectable from the launcher text alone. These
tests do that, so the next move breaks the suite instead of a training run.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# The pipeline is exactly these four launchers, in this order (instructions.md
# section 6A). A fifth launcher at root means the sprawl is growing back.
EXPECTED_LAUNCHERS = {
    "run_synthetic.ps1",
    "run_cpt.ps1",
    "run_sft.ps1",
    "run_rl.ps1",
}

# `& $PY -m rlv2 <stage> [sub] <args...>` -- how every launcher invokes python.
INVOKE_RE = re.compile(r"^\s*&\s+\$PY\s+-m\s+rlv2\s+(?P<stage>[a-z]+)(?:\s+(?P<sub>[a-z][a-z-]*))?(?P<rest>.*)$")
FLAG_RE = re.compile(r"\s--([a-z0-9-]+)")
ARGPARSE_RE = re.compile(r"""add_argument\(\s*["']--([a-z0-9-]+)["']""")


def _launchers() -> list[Path]:
    return sorted(ROOT.glob("run_*.ps1"))


def _stage_script(stage: str, sub: str | None) -> str:
    """The file `python -m rlv2 <stage> [sub]` runs, from the entrypoint's own table."""
    from rlv2.__main__ import STAGES
    target = STAGES[stage]
    if isinstance(target, dict):
        target = target[sub]
    return target.replace(".", "/") + ".py"


def _invocations(ps1: Path) -> list[tuple[str, str]]:
    """(script_path, remaining_args) for each python call in a launcher."""
    out = []
    for line in ps1.read_text(encoding="utf-8", errors="replace").splitlines():
        m = INVOKE_RE.match(line)
        if m:
            isdict = isinstance(__import__("rlv2.__main__", fromlist=["STAGES"]).STAGES[m.group("stage")], dict)
            sub = m.group("sub") if isdict else None
            rest = m.group("rest") if isdict else ((" " + m.group("sub") if m.group("sub") else "") + m.group("rest"))
            out.append((_stage_script(m.group("stage"), sub), rest))
    return out


def test_exactly_the_four_pipeline_launchers_exist():
    """Four stages, four launchers. Catches sprawl regrowth and renames."""
    found = {p.name for p in _launchers()}
    assert found == EXPECTED_LAUNCHERS, (
        f"unexpected={sorted(found - EXPECTED_LAUNCHERS)} "
        f"missing={sorted(EXPECTED_LAUNCHERS - found)}"
    )


@pytest.mark.parametrize("ps1", _launchers(), ids=lambda p: p.name)
def test_launcher_invokes_at_least_one_script(ps1: Path):
    """A launcher that invokes nothing has been gutted; the path checks below
    would then vacuously pass."""
    assert _invocations(ps1), f"{ps1.name} invokes no python script"


@pytest.mark.parametrize("ps1", _launchers(), ids=lambda p: p.name)
def test_every_invoked_script_exists(ps1: Path):
    """The run_chain.ps1 defect: invoking a path that moved."""
    missing = [s for s, _ in _invocations(ps1) if not (ROOT / s).is_file()]
    assert not missing, f"{ps1.name} invokes non-existent script(s): {missing}"


@pytest.mark.parametrize("ps1", _launchers(), ids=lambda p: p.name)
def test_every_flag_is_accepted_by_its_target(ps1: Path):
    """A flag the target's argparse does not define dies at launch, after the
    model has already loaded."""
    problems = []
    for script, rest in _invocations(ps1):
        target = ROOT / script
        if not target.is_file():
            continue  # covered by test_every_invoked_script_exists
        known = set(ARGPARSE_RE.findall(target.read_text(encoding="utf-8", errors="replace")))
        for flag in FLAG_RE.findall(rest):
            if flag not in known:
                problems.append(f"{script} does not accept --{flag}")
    assert not problems, f"{ps1.name}: " + "; ".join(problems)


@pytest.mark.parametrize("ps1", _launchers(), ids=lambda p: p.name)
def test_no_bare_root_module_imports(ps1: Path):
    """The Unload-Ollama defect: `python -c "from rlv2.judge.reward import ..."` kept
    working by accident until reward.py moved to src/. Any inline import must
    set sys.path first."""
    text = ps1.read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(r'-c\s+"([^"]*)"', text):
        body = m.group(1)
        if re.search(r"\b(from|import)\s+(reward|config|policy_eval|stratify|cost_meter)\b", body):
            assert "sys.path" in body, (
                f"{ps1.name}: inline import of a moved module without a sys.path "
                f"insert: {body[:80]!r}"
            )


@pytest.mark.parametrize("ps1", _launchers(), ids=lambda p: p.name)
def test_launcher_sets_utf8_and_anchors_cwd(ps1: Path):
    """Console is cp1252; every stage writes non-ASCII. And every invocation
    uses repo-relative paths, so cwd must be pinned to the repo root."""
    text = ps1.read_text(encoding="utf-8", errors="replace")
    assert "PYTHONIOENCODING" in text, f"{ps1.name} does not set PYTHONIOENCODING"
    assert "Set-Location $PSScriptRoot" in text, f"{ps1.name} does not anchor cwd"
