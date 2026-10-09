"""Regression contract: the SFT subset draw must exclude eval prompts from the pool.

The disjointness assert in build_subset is a post-hoc tripwire, not a filter. It passed
for years against the 112-row rl_eval.jsonl only because those prompts happen not to live
in POOL. Against the 62-row rl_eval_v2.jsonl, 30 of 62 eval prompts were in POOL and the
draw selected them -- which would have made its macro-F1 a training score. These tests pin
the exclusion so the guard cannot silently become decorative again.
"""
from __future__ import annotations
import os as _o, sys as _s

import json

import pytest

from rlv2.train.sft import train as vibe_sft


def _pool_records() -> list[dict]:
    """Eight records spanning the regime x domain grid, with stable prompt ids."""
    return [
        {"prompt": "q1", "context": "ctx", "gold": "a", "domain": "Math"},
        {"prompt": "q2", "context": "ctx", "gold": "", "domain": "Math"},
        {"prompt": "q3", "context": "", "gold": "a", "domain": "Math"},
        {"prompt": "q4", "context": "", "gold": "", "domain": "Math"},
        {"prompt": "q5", "context": "ctx", "gold": "a", "domain": "Code"},
        {"prompt": "q6", "context": "ctx", "gold": "", "domain": "Code"},
        {"prompt": "q7", "context": "", "gold": "a", "domain": "Code"},
        {"prompt": "q8", "context": "", "gold": "", "domain": "Code"},
    ]


def _write_jsonl(path, records) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return str(path)


@pytest.fixture()
def wired(tmp_path, monkeypatch):
    """Point build_subset's module-global input/output at temp files."""
    pool = _write_jsonl(tmp_path / "pool.jsonl", _pool_records())
    monkeypatch.setattr(vibe_sft, "POOL", pool)
    monkeypatch.setattr(vibe_sft, "SUBSET", str(tmp_path / "subset.jsonl"))
    return tmp_path


def test_eval_prompts_are_excluded_from_the_draw(wired) -> None:
    """Prompts present in the eval file must never be selected for training."""
    eval_path = _write_jsonl(
        wired / "eval.jsonl",
        [r for r in _pool_records() if r["prompt"] in {"q1", "q4", "q7"}],
    )

    sel = vibe_sft.build_subset(5, seed=7, eval_path=eval_path)

    picked = {r["prompt"] for r in sel}
    assert picked == {"q2", "q3", "q5", "q6", "q8"}
    assert picked.isdisjoint({"q1", "q4", "q7"})


def test_excluded_prompts_are_otherwise_selectable(wired) -> None:
    """Control: with a non-overlapping eval, the same three prompts are drawable.

    Without this, test_eval_prompts_are_excluded could pass for the wrong reason -- e.g.
    if q1/q4/q7 were simply unreachable by the MMR draw.
    """
    eval_path = _write_jsonl(wired / "eval.jsonl", [{"prompt": "not-in-pool"}])

    sel = vibe_sft.build_subset(8, seed=7, eval_path=eval_path)

    picked = {r["prompt"] for r in sel}
    assert {"q1", "q4", "q7"} <= picked
    assert len(sel) == 8


def test_draw_shrinks_to_available_rows_after_exclusion(wired) -> None:
    """Asking for more rows than survive exclusion yields the survivors, not a crash."""
    eval_path = _write_jsonl(
        wired / "eval.jsonl",
        [r for r in _pool_records() if r["prompt"] in {"q1", "q2", "q3", "q4"}],
    )

    sel = vibe_sft.build_subset(8, seed=7, eval_path=eval_path)

    assert {r["prompt"] for r in sel} == {"q5", "q6", "q7", "q8"}


def test_subset_file_written_matches_returned_rows(wired) -> None:
    """The persisted subset is what downstream training actually reads."""
    eval_path = _write_jsonl(wired / "eval.jsonl", [{"prompt": "not-in-pool"}])

    sel = vibe_sft.build_subset(4, seed=7, eval_path=eval_path)

    written = [json.loads(l) for l in open(vibe_sft.SUBSET, encoding="utf-8")]
    assert [r["prompt"] for r in written] == [r["prompt"] for r in sel]
