"""The reward bands are the operator's, set on best practice, and are CONSTANTS.

Operator 2026-09-14: "I set my values for a reason ... This is what best
practice said >=0.9 -> 1.0, >=0.7 -> 0.5, else -> 0."

    judge >= 0.9   -> 1.0     high quality and correct
    judge >= 0.7   -> 0.5     correct, below high quality
    judge <  0.7   -> 0.0     wrong, at any quality

THE DEFECT THIS PINS. grpo_one_file implemented the scheme correctly and carried
0.9 / 0.5 as defaults. archive/tools/rl_hpo_optuna.py then SEARCHED over them --
quality_bands across {False, True}, hq_floor across 0.8-0.95, lq_reward across
0.3-0.7 -- so most trials trained against reward values the operator never chose,
and some ran with no bands at all. A value fixed on external grounds is a
constant; sweeping it overwrites the decision rather than tuning around it.
"""
import importlib.util
import io
import sys

import pytest


@pytest.fixture(scope="module")
def gof():
    spec = importlib.util.spec_from_file_location("gof_qb",
                                                  "docs/grpo_one_file.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gof_qb"] = mod
    spec.loader.exec_module(mod)
    return mod


def _bands(gof, raws):
    """Run score_group's banding over scripted judge scores."""
    cfg = dict(gof.DEFAULTS)
    cfg["quality_bands"] = True
    floor = cfg["correct_floor"]
    hq, lo = cfg["hq_floor"], cfg["lq_reward"]
    out = []
    for r in raws:
        out.append(0.0 if r < floor else (1.0 if r >= hq else lo))
    return out


def test_the_defaults_are_the_operators_values(gof):
    assert gof.DEFAULTS["hq_floor"] == 0.9
    assert gof.DEFAULTS["lq_reward"] == 0.5
    assert gof.DEFAULTS["correct_floor"] == 0.7


def test_the_three_bands_pay_what_was_specified(gof):
    assert _bands(gof, [0.95, 0.9, 0.85, 0.7, 0.69, 0.45, 0.0]) == \
        [1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0]


def test_the_boundaries_are_inclusive(gof):
    """>= 0.9 and >= 0.7, not > -- a judge score landing exactly on a boundary
    is common because the 3-read majority produces quantised values."""
    assert _bands(gof, [0.9])[0] == 1.0
    assert _bands(gof, [0.7])[0] == 0.5


def test_a_wrong_answer_pays_nothing_at_any_quality(gof):
    """The reason the scheme is shaped this way: no gradient may ever point
    toward sounding confident while wrong."""
    assert all(v == 0.0 for v in _bands(gof, [0.69, 0.6, 0.45, 0.15, 0.0]))


def test_the_live_reward_path_produces_the_same_bands(gof, capsys):
    """Not just the arithmetic -- score_group itself."""
    cfg = dict(gof.DEFAULTS)
    cfg["quality_bands"] = True
    shaped, raw = gof.score_group("q?", "gold", ["a", "b", "c", "d"], cfg,
                                  stub=True)
    for s, r in zip(shaped, raw):
        expected = 0.0 if r < 0.7 else (1.0 if r >= 0.9 else 0.5)
        assert abs(s - expected) < 1e-9, (r, s, expected)


def test_the_hpo_does_not_use_bands_at_all():
    """Dropped 2026-09-14. Operator: "If we set those bands before we fixed
    things. The bands are meaningless and need to be dropped."

    The bands are dated 2026-09-12; the replay buffer was fixed 09-13/14. They
    were chosen against a reward distribution that no longer exists, so they
    carry no information about the current one. RL reward is binary at the
    correctness floor until bands are re-derived from current data.

    The seed path must not smuggle them back either -- the enqueued trial
    predates the decision and still carries hq_floor / lq_reward in its params.
    """
    src = io.open("archive/tools/rl_hpo_optuna.py", encoding="utf-8").read()
    assert 'flags["--quality-bands"]' not in src
    assert 'suggest_float("hq_floor"' not in src
    assert 'suggest_float("lq_reward"' not in src
    assert 'suggest_categorical("quality_bands"' not in src
    assert '_params.pop("hq_floor"' in src, "the seed must drop them too"
    assert '_params.pop("lq_reward"' in src


def test_the_trainer_still_supports_bands_for_when_they_are_re_derived():
    """Dropping them from the SEARCH is not deleting the mechanism.

    grpo_one_file keeps --quality-bands and its defaults, so bands can be
    re-derived from the current reward distribution and re-enabled without
    rebuilding anything.
    """
    src = io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    assert '"--quality-bands"' in src
    assert 'cfg.get("quality_bands")' in src
