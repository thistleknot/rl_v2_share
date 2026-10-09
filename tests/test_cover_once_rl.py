"""RL sees every record exactly once when --cover-once is set.

NO GOVERNING SPEC. Basis: operator 2026-09-16 -- "We need full coverage / Id say
patience of 0", after the full run was measured consuming 70 record-visits against a
240-record pool that carried 79 straddling records.

WHY A FLAG AND NOT ARITHMETIC. boltzmann_batch excludes repeats WITHIN a batch but not
across steps, and `retire_after` only drops records whose last N replays had zero
spread -- a deadness filter, not a coverage rule. Unvisited records do rank top
(rank 1.0), but that is a softmax weight, not a guarantee: drawing 240 times from 240
records with replacement returns 240*(1-(1-1/240)^240) ~= 152 unique, so a run sized
"20 steps x 12 records" would have LOOKED like full coverage and delivered 63% of it.
"""
import math
import sys

sys.path.insert(0, "docs")


def test_sampling_with_replacement_does_not_cover_the_pool():
    """The arithmetic that makes the flag necessary."""
    n = 240
    expected_unique = n * (1 - (1 - 1.0 / n) ** n)
    assert 145 < expected_unique < 160
    assert expected_unique < n * 0.7


def test_cover_once_is_exact_arithmetic():
    """With the pool shrinking, coverage is steps * records-per-step, no estimate."""
    pool, per_step = 240, 12
    assert pool % per_step == 0
    assert pool // per_step == 20


def _src():
    import io
    return io.open("docs/grpo_one_file.py", encoding="utf-8").read()


def test_the_draw_pool_excludes_already_covered_records():
    src = _src()
    assert 'if r.get("question", "") not in _covered' in src
    assert "_step_batch = boltzmann_batch(\n                        _draw_pool" in src, \
        "the draw still reads the full pool"


def test_drawn_records_are_marked_covered():
    src = _src()
    assert '_covered.add(_r.get("question", ""))' in src


def test_exhaustion_ends_the_RUN_not_just_the_inner_loop():
    """A bare `break` exits the step loop and leaves the round loop spending a judged
    eval per round on an empty pool."""
    src = _src()
    assert "_coverage_done = True" in src
    assert "        if _coverage_done:\n            break" in src


def test_the_flag_defaults_off_so_existing_recipes_are_unchanged():
    src = _src()
    i = src.index('"--cover-once"')
    assert 'action="store_true"' in src[i:i + 200]


def test_the_runner_only_passes_the_flag_when_asked():
    import io
    src = io.open("archive/tools/phase_runner.py", encoding="utf-8").read()
    assert '(["--cover-once"] if getattr(a, "cover_once", False) else [])' in src


# --------------------------------------------------- patience vs coverage ----
def test_the_coverage_gate_does_not_depend_on_the_sampler_mode():
    """Operator 2026-09-16: "we dont track patience until all records visited at least
    once?" -- correct, and it must hold however the records are drawn.

    The first version read `(not cover_once) or _coverage_done`, so the hold EVAPORATED
    whenever --cover-once was off -- which is precisely the run where the sampler may
    repeat and therefore covers the pool more slowly. The gate now counts what has actually
    been visited."""
    src = _src()
    assert "_covered_all = _coverage_done or len(_covered) >= _pool_size" in src
    assert "(not cfg.get(\"cover_once\")) or _coverage_done" not in src,         "patience hold is tied to the flag again"


def test_coverage_is_recorded_on_every_draw():
    """If _covered only fills under --cover-once, the gate above can never open."""
    src = _src()
    i = src.index("for _r in _step_batch or []:")
    assert "cfg.get(\"cover_once\")" not in src[i - 200:i],         "coverage recording is conditioned on the flag"


def test_patience_cannot_fire_before_the_pool_is_covered():
    """Operator 2026-09-16: "we shouldn't have done patience of 5 / We should have
    waited till full coverage / Then some patience".

    Measured the same day: patience 5 stopped the full run at 70 record-visits of a
    240-record pool. It had met 45 of the 79 straddling records SFT built for it, and
    the five "no new best" readings it stopped on were taken over a quarter of the
    data. A flat stretch on data the model was never shown is not convergence.
    """
    src = _src()
    assert "if args.patience and stale >= args.patience and _covered_all:" in src


def test_a_held_stop_is_announced_not_silent():
    """A suppressed stop the operator cannot see is indistinguishable from a bug."""
    src = _src()
    assert "[patience] HELD" in src


def test_patience_does_fire_once_the_pool_is_covered():
    """The gate DELAYS early stopping; it must not disable it. Once every record has been
    seen, a flat stretch is real evidence and patience is allowed to end the run."""
    src = _src()
    i = src.index("_covered_all = ")
    line = src[i:src.index(chr(10), i)]
    assert "len(_covered) >= _pool_size" in line, \
        "coverage must be able to satisfy the gate, or patience could never fire"
    assert "_coverage_done or" in line, \
        "an exhausted --cover-once pool must also satisfy it"


def test_a_judged_record_counts_as_covered_even_if_it_was_discarded():
    """A record that was generated and judged has been SEEN, whether or not its group
    carried gradient.

    MEASURED 2026-09-16: _covered.add fired only for the originally-drawn slots, while the
    defer-and-redraw path judged ~41 records per 16 live. Coverage was therefore understated
    by ~2.5x, the patience gate stayed shut long past real coverage, and the run would have
    ground to its 100-step ceiling instead of stopping when the pool was exhausted."""
    src = _src()
    # the covered-set must be updated where EVERY attempt passes, beside reward_history
    i = src.index("reward_history.setdefault(record.get(\"question\", \"\"), []).extend(")
    j = src.index("deferred.append", i)
    assert "_covered.add(record.get(\"question\", \"\"))" in src[i:j],         "discarded records are invisible to the coverage gate again"


def test_coverage_is_recorded_in_both_places():
    """The drawn batch AND every judged attempt -- one without the other undercounts."""
    src = _src()
    assert src.count("_covered.add(") == 2
