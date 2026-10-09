"""Think AND tool-call tokens are suppressed in EVERY phase, from one shared list.

NO GOVERNING SPEC. Basis: operator 2026-09-15 -- "Think and tool calling needs to be
suppressed all phases, we've visited this b4", after the eval scorer read 0/16 on a
healthy SFT checkpoint because 9 of 16 completions were unclosed <think> blocks that
strip_reasoning deleted entirely.

WHAT THIS PINS, and why each one is a defect that actually shipped:
  * the list covers tool calls, not think alone -- tool markers were suppressed in NO
    phase before this
  * the ids are resolved from the LIVE tokenizer, never hardcoded -- src/rlv2/train/rl/grpo.py and
    src/rlv2/train/sft/train.py both carried a literal [248068, 248069]
  * every phase agrees, so a checkpoint cannot behave one way in training and another
    at scoring time
"""
import sys

import pytest


MARKERS = ("<think>", "</think>", "<tool_call>", "</tool_call>",
           "<tool_response>", "</tool_response>")


class FakeTok:
    """Ids that deliberately do NOT match the production literals."""
    TABLE = {m: 900 + i for i, m in enumerate(MARKERS)}

    def convert_tokens_to_ids(self, t):
        return self.TABLE.get(t, -1)


def test_banned_list_covers_think_and_tool_calls():
    from rlv2.eval import half_epoch_eval as he
    assert set(he.BANNED_MARKERS) == set(MARKERS)


def test_ids_come_from_the_tokenizer_not_a_literal():
    from rlv2.eval import half_epoch_eval as he
    got = he.banned_ids(FakeTok())
    assert got == sorted(FakeTok.TABLE.values())
    assert 248068 not in got, "hardcoded production id leaked into the resolved list"


def test_unknown_markers_are_dropped_not_passed_as_minus_one():
    from rlv2.eval import half_epoch_eval as he

    class Missing:
        def convert_tokens_to_ids(self, t):
            return -1

    assert he.banned_ids(Missing()) == []


@pytest.mark.parametrize("mod", ["rlv2.train.rl.grpo", "rlv2.train.sft.train"])
def test_every_phase_agrees_with_the_shared_list(mod):
    import importlib

    from rlv2.eval import half_epoch_eval as he
    m = importlib.import_module(mod)
    assert m._banned(FakeTok()) == he.banned_ids(FakeTok())


def test_think_ids_is_the_banned_list(monkeypatch):
    """_think_ids is kept as an alias; it must not narrow back to think-only."""
    from rlv2.eval import half_epoch_eval as he
    assert he._think_ids(FakeTok()) == he.banned_ids(FakeTok())
