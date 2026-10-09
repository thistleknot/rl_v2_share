"""Unit tests: the nine optimizer settings actually REACH the trainer.

WHY THIS FILE EXISTS. Until 2026-08-30 `vibe_sft.py` accepted exactly one optimizer flag,
`--lr`. Warmup was hardcoded to `total_steps // 10`, the scheduler to the literal "cosine",
and weight decay / grad clipping / adam betas / label smoothing were never passed at all --
they sat on HuggingFace defaults while `hpo_r3.py` believed it was sweeping them. Every trial
printed "knobs held at src/rlv2/train/sft/train.py defaults" and that line was read and not acted on.

So "SFT HPO" was a learning-rate sweep wearing a bigger name, and the conclusion drawn from it
-- that the configuration is wrong -- was never supported by a one-axis sweep.

**A flag that PARSES but never LANDS is worse than a missing flag**: it reports a sweep that
did not happen. These tests assert the values arrive in `TrainingArguments`, which is the only
thing that makes a sweep real.
"""
from __future__ import annotations
import os as _o, sys as _s

import ast
import pathlib
import re

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "rlv2" / "train" / "sft" / "train.py"
TEXT = SRC.read_text(encoding="utf-8")

# The owner's nine, as flag -> the train_bins(...) keyword (or model-config field) it must reach.
NINE = {
    "--lr": "lr",
    "--warmup-frac": "warmup_steps",
    "--weight-decay": "weight_decay",
    "--lr-scheduler": "lr_scheduler",
    "--max-grad-norm": "max_grad_norm",
    "--adam-beta1": "betas",
    "--label-smoothing": "label_smoothing",
    "--num-cycles": "scheduler_kwargs",
    "--attention-dropout": "attention_dropout",      # model config, applied at load
}


def _training_args_call() -> str:
    """The source text of the train_bins(...) call (kept under its old helper name)."""
    i = TEXT.index("train_bins(\n")
    depth, j = 0, i + len("train_bins")
    for j in range(j, len(TEXT)):
        if TEXT[j] == "(":
            depth += 1
        elif TEXT[j] == ")":
            depth -= 1
            if depth == 0:
                return TEXT[i:j + 1]
    raise AssertionError("unbalanced train_bins call")


@pytest.mark.parametrize("flag", sorted(NINE))
def test_every_one_of_the_nine_settings_is_accepted(flag):
    """argparse must define it. Half the requirement."""
    assert 'add_argument("%s"' % flag in TEXT, "%s is not a CLI flag" % flag


@pytest.mark.parametrize("flag,field", sorted(NINE.items()))
def test_every_setting_actually_lands_in_the_trainer(flag, field):
    """The half that matters.

    Parsing a flag and then not passing it produces a sweep that reports values it never
    used -- which is exactly what happened to warmup, scheduler, decay, clipping, betas and
    label smoothing before this file existed.
    """
    where = TEXT if field == "attention_dropout" else _training_args_call()
    assert field in where, (
        "%s parses but %s never reaches the trainer -- the sweep would be fictional" % (flag, field))


def _strip_comments(src: str) -> str:
    """Comments are not code. The first version of the test below matched the string
    `total_steps // 10` inside a comment EXPLAINING that the hardcoding was removed, and
    failed on correct code -- a test asserting on prose rather than behaviour."""
    return "\n".join(re.sub(r"#.*$", "", line) for line in src.splitlines())


def test_warmup_is_derived_from_the_flag_not_a_hardcoded_tenth():
    """`warmup_steps=max(1, total_steps // 10)` ignored --warmup-frac entirely."""
    call = _strip_comments(_training_args_call())
    assert "total_steps // 10" not in call, "warmup is still hardcoded to a tenth"
    assert "warmup_frac" in call, "warmup_steps does not read the flag"


def test_scheduler_is_not_the_hardcoded_literal():
    call = _strip_comments(_training_args_call())
    assert 'lr_scheduler="cosine"' not in call, "scheduler is still the literal 'cosine'"
    assert "a.lr_scheduler" in call


def test_scheduler_choices_include_the_warmup_decay_variant():
    """`warmup_decay` in the owner's list is the warmup_stable_decay scheduler variant."""
    assert "warmup_stable_decay" in TEXT


def test_dropout_is_applied_at_MODEL_LOAD_not_passed_to_TrainingArguments():
    """It is a model-config field. Passing it to TrainingArguments would be silently ignored,
    which is the same class of bug as not passing it at all."""
    assert "attention_dropout" not in _strip_comments(_training_args_call())
    assert "_model_kwargs" in TEXT and "attention_dropout" in TEXT


def test_defaults_match_huggingface_so_an_unswept_axis_does_not_move():
    """Adding a flag must not silently change behaviour for runs that do not set it.

    If a new default differed from HF's, every previously-recorded number would become
    non-comparable -- a One-Variable violation introduced by the plumbing itself.
    """
    tree = ast.parse(TEXT)
    defaults = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)):
            for kw in node.keywords:
                if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                    defaults[node.args[0].value] = kw.value.value
    assert defaults.get("--weight-decay") == 0.0
    assert defaults.get("--max-grad-norm") == 1.0
    assert defaults.get("--adam-beta1") == 0.9
    assert defaults.get("--adam-beta2") == 0.999
    assert defaults.get("--label-smoothing") == 0.0
    assert defaults.get("--lr-scheduler") == "cosine"
    assert defaults.get("--warmup-frac") == 0.1, "was hardcoded 1/10; default must preserve it"


def test_hpo_driver_no_longer_restricts_the_sft_knob_set():
    """`STAGES["sft"]["knobs"] = {"lr"}` existed only because the trainer took one flag.
    Leaving it would keep discarding eight axes while the flags sat there unused."""
    hpo = (pathlib.Path(__file__).resolve().parents[1] / "archive" / "src" / "hpo_r3.py").read_text(
        encoding="utf-8")
    sft_block = hpo[hpo.index('"sft":'):hpo.index('"sft":') + 700]
    assert '"knobs": {"lr"}' not in sft_block, "SFT is still restricted to the lr axis"
    assert '"knobs": None' in sft_block
