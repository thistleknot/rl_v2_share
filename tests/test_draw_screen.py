"""The screen ranks settings by reject rate, before any weights move.

Operator 2026-09-14: "what set of parms get us to the most efficient redraw rate
(the model wouldn't be adjusted until we actually do 1 step)" and "track up to
100 records drawn and cap at that / Then we train on those settings".

WHY IT EXISTS. Measured the same day from base, two 45-minute HPO trials were
killed at the cap having completed 2 steps and 1 step between them, with 85-88%
of probed groups coming back uniform. The trials were spending their whole budget
discovering the reject rate. Measuring it directly costs a fraction of that.

TWO PROPERTIES THIS FILE PROTECTS.

1. THE MODEL IS FROZEN. No optimizer, no backward, no checkpoint written. If the
   screen ever trains, its numbers stop being comparable across settings because
   each row would be measuring a different model.

2. EVERY SETTING SEES THE SAME RECORDS. The rng is re-seeded per candidate, so
   row-to-row differences come from the setting and not from which records were
   drawn. This is the one-variable rule applied to a screen.
"""
import io

import pytest

SRC = "archive/tools/screen_draw_efficiency.py"


@pytest.fixture(scope="module")
def src():
    return io.open(SRC, encoding="utf-8").read()


@pytest.fixture(scope="module")
def mod():
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("screen_de", SRC)
    m = importlib.util.module_from_spec(spec)
    sys.modules["screen_de"] = m
    spec.loader.exec_module(m)
    return m


def test_the_model_is_frozen(src):
    """Match CALLS, not words.

    An earlier version of this test grepped for the bare word "optimizer" and
    failed on the module docstring's own sentence "no optimizer is constructed" --
    a test that fails on its subject's explanation of why it passes.
    """
    assert "model.eval()" in src, "the screen must not run in train mode"
    for call in ("optimizer.step(", "AdamW(", ".backward(", "save_pretrained("):
        assert call not in src, "%s means it is training, not screening" % call


def test_every_candidate_draws_the_same_records(src):
    assert "random.Random(a.seed)" in src, \
        "the rng is re-seeded per candidate so rows differ only by setting"


def test_the_cap_defaults_to_a_hundred_records(mod):
    import argparse
    ap = [a for a in dir(mod) if a == "main"]
    assert ap
    assert '"--cap", type=int, default=100' in io.open(SRC, encoding="utf-8").read()


def test_no_learning_rate_family_parameter_is_screened(mod):
    """With frozen weights they cannot change a rollout, so screening them would
    produce identical rows and waste the budget."""
    keys = set()
    for c in mod.CANDIDATES:
        keys.update(c.keys())
    for banned in ("lr", "beta", "epsilon", "epsilon_high", "weight_decay",
                   "lr_schedule", "warmup_steps"):
        assert banned not in keys, banned


def test_the_screened_axes_are_the_ones_that_change_the_text(mod):
    keys = set()
    for c in mod.CANDIDATES:
        keys.update(c.keys())
    assert "temperature" in keys, "the only knob that changes the generated text"
    assert "rollouts" in keys, "group size decides how many chances to disagree"
    assert "max_completion" in keys, "truncation forces zeros via the EOS gate"


def test_the_incumbent_setting_is_included(mod):
    """The current 0.9/G4 must be a row, or the screen has no reference point."""
    names = [c["name"] for c in mod.CANDIDATES]
    assert any("base" in n for n in names), names
    base = [c for c in mod.CANDIDATES if "base" in c["name"]][0]
    assert "temperature" not in base, "the incumbent uses the default temperature"
    assert base["rollouts"] == 4


def test_candidates_are_ordered_cheapest_first(mod):
    """Cost is cap x G, so the G=16 row costs 4x a G=4 row. Useful numbers must
    land before the expensive rows run."""
    gs = [c.get("rollouts", 4) for c in mod.CANDIDATES]
    assert gs[0] == 4
    assert gs[-1] == max(gs), gs


def test_results_are_written_after_every_candidate(src):
    """A screen that only writes at the end loses everything when it is killed --
    which is exactly what happened to the trials this replaces."""
    i = src.index("rows.append(r)")
    j = src.index("=== RANKED BY FEWEST REJECTS ===")
    assert "json.dumps(rows" in src[i:j], "receipt must be written inside the loop"
