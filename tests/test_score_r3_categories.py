"""Unit tests: src/rlv2/eval/score_r3_categories.py — the R3 objective readout.

This is the instrument the whole experiment reads. Its failure modes are quiet: a
miscounted cell reports lift that is not there, and a mean over 20 items hides a
category that never gets anything right. Both are pinned here.
"""
from __future__ import annotations
import os as _o, sys as _s

import json
import os

import pytest

from rlv2.eval import score_r3_categories as sc


def _recs(spec):
    """spec: {item_id: (category, [rewards...])} -> flat sample rows."""
    out = []
    for item, (cat, rewards) in spec.items():
        for r in rewards:
            out.append({"item_id": item, "category": cat, "reward": r})
    return out


# --------------------------------------------------------------------------
# binary threshold
# --------------------------------------------------------------------------
def test_correct_is_reward_at_or_above_tau():
    """The fallback boundary is judge.CORRECT_FLOOR, inclusive, and nothing else.

    TAU was 0.5 until 2026-08-31; `judge.combine` maps an `incorrect` verdict to strictly
    below 0.70, so at 0.5 a fluent wrong answer with quality >= 0.72 counted correct -- the
    exact leak the verdict exists to catch. The fallback now equals the live floor, so the
    two ways of deciding correctness cannot disagree (metric-unification, playbook
    2026-08-31). Inclusivity at the boundary still matters: an off-by-one silently
    reclassifies every borderline generation."""
    from rlv2.judge import judge as J
    assert sc.TAU == J.CORRECT_FLOOR
    s = sc.summarise(_recs({"a": ("Math", [0.69, 0.70, 0.71])}), k=3)
    assert s["correct"] == 2, "0.70 must count as correct (inclusive boundary)"
    assert s["n_obs"] == 3


def test_rate_is_over_observations_not_items():
    """20 items at k=8 is 160 observations. Dividing by items would inflate the
    rate 8x and make any checkpoint look like a breakthrough."""
    s = sc.summarise(_recs({
        "a": ("Math", [1.0] * 8),
        "b": ("Math", [0.0] * 8),
    }), k=8)
    assert s["n_items"] == 2
    assert s["n_obs"] == 16
    assert s["rate"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# per-category cells — the objective's axis
# --------------------------------------------------------------------------
def test_every_category_cell_is_reported_even_when_empty():
    """An absent category must appear as 0, not vanish. A missing row reads as
    'not measured' and is easy to mistake for 'nothing to report'."""
    s = sc.summarise(_recs({"a": ("Math", [1.0])}), k=1)
    assert set(s["per_category"]) == set(sc.CATEGORY_ORDER)
    assert s["per_category"]["Code"]["n_items"] == 0
    assert s["per_category"]["Code"]["rate"] == 0.0


def test_category_rates_are_computed_independently():
    s = sc.summarise(_recs({
        "m1": ("Math", [1.0, 1.0]),      # 2/2
        "c1": ("Code", [0.0, 0.0]),      # 0/2
        "s1": ("STEM", [1.0, 0.0]),      # 1/2
    }), k=2)
    assert s["per_category"]["Math"]["rate"] == pytest.approx(1.0)
    assert s["per_category"]["Code"]["rate"] == pytest.approx(0.0)
    assert s["per_category"]["STEM"]["rate"] == pytest.approx(0.5)
    assert s["rate"] == pytest.approx(3 / 6)


def test_a_dead_category_is_visible_despite_a_healthy_mean():
    """THE failure §6.10 exists to prevent: an overall rate that looks fine while
    one question type never lands. The per-category cell must expose it."""
    s = sc.summarise(_recs({
        "m1": ("Math", [1.0] * 8),
        "m2": ("Math", [1.0] * 8),
        "c1": ("Code", [0.0] * 8),
    }), k=8)
    assert s["rate"] > 0.6, "overall looks healthy"
    assert s["per_category"]["Code"]["correct"] == 0, "…but Code is dead"


# --------------------------------------------------------------------------
# per-item detail
# --------------------------------------------------------------------------
def test_per_item_exposes_variance_across_k():
    """3/8-3/8-3/8 and 8/8-0/8-1/8 share a mean and are completely different
    results. Per-item counts plus spread are what separate them."""
    s = sc.summarise(_recs({
        "even1": ("Math", [1, 1, 1, 0, 0, 0, 0, 0]),
        "even2": ("Math", [1, 1, 1, 0, 0, 0, 0, 0]),
        "split1": ("Code", [1] * 8),
        "split2": ("Code", [0] * 8),
    }), k=8)
    assert s["per_item"]["even1"]["correct"] == 3
    assert s["per_item"]["split1"]["correct"] == 8
    assert s["per_item"]["split2"]["correct"] == 0
    assert s["per_item"]["split1"]["reward_sd"] == pytest.approx(0.0)
    assert s["per_item"]["even1"]["reward_sd"] > 0.0


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def test_render_marks_items_that_are_never_correct():
    s = sc.summarise(_recs({"dead": ("Code", [0.0] * 8),
                            "live": ("Math", [1.0] * 8)}), k=8)
    txt = sc.render(s, "t")
    assert "never correct" in txt
    assert "f1_abstain: 0.000" in txt, "legacy field retained for compatibility"


def test_render_shows_delta_against_a_baseline():
    base = sc.summarise(_recs({"a": ("Math", [0.0] * 4)}), k=4)
    now = sc.summarise(_recs({"a": ("Math", [1.0] * 4)}), k=4)
    txt = sc.render(now, "after", base)
    assert "+1.000" in txt, "per-category delta vs baseline must be shown"


# --------------------------------------------------------------------------
# the live eval file
# --------------------------------------------------------------------------
EVAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "rl", "r3_eval_canonical.jsonl")


@pytest.mark.skipif(not os.path.exists(EVAL), reason="eval not built")
def test_eval_file_carries_the_fields_the_objective_needs():
    """`category` was dropped by the original converter, which made the objective's
    own axis unreportable. `item_id` is needed to group k samples per question."""
    rows = [json.loads(l) for l in open(EVAL, encoding="utf-8") if l.strip()]
    # Sized against the partitioner's eval split, never a literal -- see the same
    # note in tests/test_half_epoch_eval.py.
    split = os.path.join(os.path.dirname(EVAL), "r3_eval.jsonl")
    assert len(rows) == sum(1 for l in open(split, encoding="utf-8") if l.strip())
    for field in ("prompt", "context", "gold", "domain", "category", "item_id"):
        assert all(field in r for r in rows), "missing %s" % field
    assert all(r["regime"] == "R3" for r in rows)
    assert all(r["context"] == "" for r in rows), "R3 items must carry no context"
