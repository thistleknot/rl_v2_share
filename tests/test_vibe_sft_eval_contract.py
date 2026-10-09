"""Which eval file src/rlv2/train/sft/train.py actually scores, and why the default must NOT move.

BACKGROUND. src/rlv2/train/sft/train.py's epoch-loop comment claimed the between-epoch eval ran on
"the full 62-item balanced eval", and the code read

    _eval_file = getattr(a, 'eval', 'data/rl/rl_eval_v2.jsonl')

That fallback is unreachable -- argparse always populates `a.eval` from
`default=EVAL`, and EVAL is data/rl/rl_eval.jsonl = 112 rows. So a direct run scored
112 rows while the source advertised 62. Nine SFT checkpoints in this project already
carry a macro_f1 that describes something other than what the label says; this is the
same failure with a different mechanism.

WHY THE DEFAULT IS NOT SIMPLY FLIPPED TO v2. Full caller audit (2026-08-16):

    archive/harness/hpo_sft_ffn.py:111    --eval rl_eval_mini.jsonl   explicit
    src/rlv2/train/hpo/sft.py:37    --eval rl_eval.jsonl        explicit
    archive/harness/hpo_sft_lr62.py:70    --eval rl_eval_v2.jsonl     explicit
    archive/harness/train_harness.py:114  --eval job["eval_file"]     explicit
    run_crosshatch_v2.ps1:44,87   --eval $evalFile            explicit
    run_overnight.ps1:38          (none)                      RELIES ON DEFAULT
    run_pipeline.ps1:26,43        (none)                      RELIES ON DEFAULT

Every Python harness passes --eval and is immune to the default. The two callers that
would silently change metric if EVAL moved are the PowerShell pipeline scripts. Flipping
EVAL to the 62-row set would re-point those at a different eval with nothing in their
output announcing the switch, making their new numbers incomparable to the ones already
recorded. The default therefore stays at 112 and the run announces what it scored.

NOTE: an earlier revision of this docstring -- and the comment still sitting at
vibe_sft.py:381-387 -- justified the freeze by claiming hpo_sft_ffn.py and
hpo_sft_full.py depend on the default. That is false; both pass --eval explicitly. The
decision is right, the stated reason was not. The source comment needs the same fix.

test_default_reliant_callers_still_omit_eval is the tripwire: when run_overnight.ps1 and
run_pipeline.ps1 start passing --eval, flipping the default becomes safe.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = (ROOT / "src/rlv2/train/sft/train.py").read_text(encoding="utf-8")
TREE = ast.parse(SRC)


def _ps1_vibe_sft_commands(text: str) -> list[str]:
    """Every src/rlv2/train/sft/train.py invocation in a .ps1, following backtick continuations."""
    lines = text.splitlines()
    cmds = []
    for i, line in enumerate(lines):
        if "src/rlv2/train/sft/train.py" not in line:
            continue
        buf, j = [line], i
        while lines[j].rstrip().endswith("`") and j + 1 < len(lines):
            j += 1
            buf.append(lines[j])
        cmds.append("\n".join(buf))
    return cmds


def _count_rows(path: Path) -> int:
    """Same rule src/rlv2/train/sft/train.py uses: non-blank lines."""
    with open(path, encoding="utf-8") as fh:
        return sum(1 for ln in fh if ln.strip())


class TestEvalDefault:
    def test_eval_default_is_still_the_112_row_file(self):
        """Pinned deliberately. Flipping this silently re-points two sibling harnesses."""
        m = re.search(r'^EVAL\s*=\s*["\'](.+?)["\']', SRC, re.M)
        assert m, "EVAL constant not found in src/rlv2/train/sft/train.py"
        assert m.group(1) == "data/rl/rl_eval.jsonl"

    def test_dead_v2_fallback_is_gone(self):
        """getattr(a,'eval', <v2>) advertised a default that could never fire.

        AST, not substring: the explanatory comment at vibe_sft.py:381 quotes the old
        expression verbatim, and a naive `in SRC` check flags that comment as live code.
        """
        for node in ast.walk(TREE):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) == 3):
                continue
            key = node.args[1]
            if isinstance(key, ast.Constant) and key.value == "eval":
                pytest.fail(
                    f"live getattr(..., 'eval', <default>) at line {node.lineno}: "
                    f"argparse always populates a.eval, so the fallback is unreachable "
                    f"and misadvertises which eval set runs."
                )

    def test_eval_file_is_read_straight_off_argparse(self):
        assert re.search(r"^\s*_eval_file\s*=\s*a\.eval\s*$", SRC, re.M)


class TestAnnounceLine:
    """The run must state what it scored, so a mislabel cannot hide."""

    def test_announce_line_exists(self):
        assert "[epoch-eval] scoring against" in SRC

    def test_announce_reports_both_file_and_row_count(self):
        m = re.search(r"\[epoch-eval\] scoring against \{(\w+)\} \(\{(\w+)\} rows\)", SRC)
        assert m, "announce line must interpolate both the path and the row count"
        assert m.group(1) == "_eval_file"
        assert m.group(2) == "_eval_n"

    def test_unreadable_eval_warns_and_does_not_crash_the_run(self):
        """A missing eval file must degrade to a warning, not kill a trained epoch."""
        assert "except OSError" in SRC
        assert "WARNING cannot read" in SRC
        assert re.search(r"_eval_n\s*=\s*-1", SRC)


class TestDataContract:
    """The row counts the whole objective rests on."""

    V2 = ROOT / "data" / "rl" / "rl_eval_v2.jsonl"
    V1 = ROOT / "data" / "rl" / "rl_eval.jsonl"

    @pytest.mark.skipif(not (ROOT / "data" / "rl" / "rl_eval_v2.jsonl").exists(),
                        reason="rl_eval_v2.jsonl absent")
    def test_v2_really_has_62_rows(self):
        """If this is not 62, the '62-item balanced eval' objective is misnamed."""
        assert _count_rows(self.V2) == 62

    @pytest.mark.skipif(not (ROOT / "data" / "rl" / "rl_eval.jsonl").exists(),
                        reason="rl_eval.jsonl absent")
    def test_default_eval_really_has_112_rows(self):
        assert _count_rows(self.V1) == 112

    @pytest.mark.skipif(not ((ROOT / "data" / "rl" / "rl_eval.jsonl").exists()
                             and (ROOT / "data" / "rl" / "rl_eval_v2.jsonl").exists()),
                        reason="eval files absent")
    def test_the_two_eval_sets_are_actually_different_sizes(self):
        """Guards the premise: if they were the same, none of this would matter."""
        assert _count_rows(self.V1) != _count_rows(self.V2)


class TestCallerAudit:
    """Documents the real reason the default is frozen. Revisit when these fail."""

    # The only two callers that omit --eval. These are what freeze EVAL at 112.
    DEFAULT_RELIANT = ["run_overnight.ps1", "run_pipeline.ps1"]

    # Every Python harness passes --eval and is immune to the default's value.
    EXPLICIT_PY = [
        "src/rlv2/train/hpo/sft.py",
    ]

    @pytest.mark.parametrize("name", DEFAULT_RELIANT)
    def test_default_reliant_callers_still_omit_eval(self, name):
        p = ROOT / name
        if not p.exists():
            pytest.skip(f"{name} absent")
        cmds = _ps1_vibe_sft_commands(p.read_text(encoding="utf-8"))
        assert cmds, f"{name} no longer invokes src/rlv2/train/sft/train.py"
        for cmd in cmds:
            assert "--eval" not in cmd, (
                f"{name} now passes --eval explicitly. It was the reason EVAL stayed "
                f"frozen at the 112-row set; if no caller relies on the default any "
                f"more, flipping it to rl_eval_v2.jsonl is now safe."
            )

    @pytest.mark.parametrize("rel", EXPLICIT_PY)
    def test_python_harnesses_pass_eval_explicitly(self, rel):
        """These are immune to the default -- and must stay that way."""
        p = ROOT / rel
        if not p.exists():
            pytest.skip(f"{rel} absent")
        txt = p.read_text(encoding="utf-8")
        assert "rlv2.train.sft.train" in txt, f"{rel} no longer calls rlv2.train.sft.train"
        assert '"--eval"' in txt or "'--eval'" in txt, (
            f"{rel} stopped passing --eval and now inherits EVAL={{112 rows}} "
            f"silently. Pass it explicitly."
        )

    def test_lr62_driver_targets_the_62_row_set(self):
        """Our sweep's objective must be the balanced v2 eval, not the 112-row set."""
        p = ROOT / "archive" / "harness" / "hpo_sft_lr62.py"
        if not p.exists():
            pytest.skip("hpo_sft_lr62.py absent")
        txt = p.read_text(encoding="utf-8")
        assert "--eval" in txt
        assert "rl_eval_v2.jsonl" in txt

    def test_ffn_harness_scores_the_mini_set_not_the_default(self):
        """hpo_sft_ffn scores rl_eval_mini -- a third eval set. Pinned so the
        three-way distinction cannot quietly collapse."""
        p = ROOT / "archive" / "harness" / "hpo_sft_ffn.py"
        if not p.exists():
            pytest.skip("hpo_sft_ffn.py absent")
        txt = p.read_text(encoding="utf-8")
        assert "rl_eval_mini.jsonl" in txt
