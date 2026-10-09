"""--epsilon / --epsilon-high actually reach GRPOConfig, not just argparse.

NO GOVERNING SPEC. Basis: operator instruction 2026-09-03 ("modify mgpo for all those
clear ENHANCEMENTS identified in ch6/7... follow the advice for deepseek"). Ch7 S7.4
covers PPO-style clipped policy ratios; TRL's GRPOConfig has always supported this
(epsilon=0.2 default) but mgpo.py never exposed it -- every full-param RL run this
project has produced inherited the untuned library default silently.

This project has a recorded, repeated failure mode for exactly this shape of bug: a
flag added to argparse but never threaded into TrainingArguments/GRPOConfig
(tests/test_vibe_sft_knobs.py's B-1/B-2 lesson -- "the evidence was in a line I had
already read on every trial... reading a log line is not the same as acting on it").
Pin the wiring so it cannot silently regress the same way twice.
"""
from __future__ import annotations
import os as _o
import sys as _s


import ast
import io

MGPO = _o.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))),
                    "src", "rlv2", "train", "rl", "grpo.py")
SRC = io.open(MGPO, encoding="utf-8").read()


def test_epsilon_flags_exist_with_the_documented_defaults():
    tree = ast.parse(SRC)
    seen = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
            continue
        if not node.args:
            continue
        flag = getattr(node.args[0], "value", None)
        if flag in ("--epsilon", "--epsilon-high"):
            kw = {k.arg: k.value for k in node.keywords}
            seen[flag] = getattr(kw.get("default"), "value", "MISSING")
    assert seen.get("--epsilon") == 0.2, (
        "--epsilon must default to TRL's own 0.2 so existing runs stay reproducible "
        "unless the flag is explicitly overridden")
    assert "--epsilon-high" in seen and seen["--epsilon-high"] is None, (
        "--epsilon-high must default to None (TRL mirrors --epsilon when unset)")


def test_epsilon_flags_are_passed_to_grpoconfig_not_just_parsed():
    assert "epsilon=args.epsilon" in SRC, \
        "--epsilon is parsed but never reaches GRPOConfig -- the B-1/B-2 bug shape"
    assert "epsilon_high=args.epsilon_high" in SRC, \
        "--epsilon-high is parsed but never reaches GRPOConfig -- the B-1/B-2 bug shape"


def test_dr_grpo_loss_type_remains_a_valid_untuned_choice():
    """Ch7 S7.6.3 item 7 (Dr. GRPO's std-normalization removal). Already wired via
    --loss-type; this test only pins that the choice keeps existing -- do not narrow
    it back to {grpo, dapo} while it is still an open backlog item."""
    assert '"dr_grpo"' in SRC and "choices=" in SRC.split('"--loss-type"')[1][:120], \
        "dr_grpo dropped from --loss-type choices"
