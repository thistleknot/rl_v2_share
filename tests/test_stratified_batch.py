"""One question per domain per step — the opening round's draw.

Spec: operator 2026-09-13: "1st step is one question from each of our 4 domains
each step / The only time we do uniform".

Before this, a 4-record step was 4 random draws from all 240. The R3 pool's
`category` counts are uneven (Math 74 / Code 58 / STEM 53 / Instruction 55), so a
random 4-draw MISSES Code entirely about one time in seven. With 3 steps per HPO
trial, a domain could go untrained for a whole trial — and an axis that was never
sampled reads as a flat axis (CLAUDE.md 6b, partition variance).
"""
from __future__ import annotations

import importlib.util
import os as _o
import io
import random
import sys as _s

for _d in ("docs",):
    _p = _o.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))), _d)
    _p not in _s.path and _s.path.insert(0, _p)

import pytest  # noqa: E402


@pytest.fixture(scope="module")
def gof():
    saved = _s.argv
    _s.argv = ["x"]
    spec = importlib.util.spec_from_file_location(
        "gof_strat", _o.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))),
                                  "docs", "grpo_one_file.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    _s.argv = saved
    return m


# the real R3 proportions
POOL = ([{"category": "Math", "id": "m%d" % i} for i in range(74)]
        + [{"category": "Code", "id": "c%d" % i} for i in range(58)]
        + [{"category": "STEM", "id": "s%d" % i} for i in range(53)]
        + [{"category": "Instruction", "id": "i%d" % i} for i in range(55)])


def cats(batch):
    return sorted(r["category"] for r in batch)


def test_a_four_record_step_takes_one_from_every_domain(gof):
    for seed in range(25):
        b = gof.stratified_batch(POOL, 4, random.Random(seed))
        assert cats(b) == ["Code", "Instruction", "Math", "STEM"], \
            "seed %d drew %s" % (seed, cats(b))


def test_the_old_random_draw_really_does_miss_domains(gof):
    # the defect this replaces: uniform_batch misses a domain often
    misses = sum(1 for seed in range(200)
                 if len(set(r["category"] for r in
                            gof.uniform_batch(POOL, 4, random.Random(seed)))) < 4)
    assert misses > 20, "uniform should miss a domain frequently; got %d/200" % misses


def test_a_sixteen_record_step_is_four_of_each(gof):
    b = gof.stratified_batch(POOL, 16, random.Random(3))
    assert cats(b).count("Math") == 4
    assert cats(b).count("Code") == 4
    assert cats(b).count("STEM") == 4
    assert cats(b).count("Instruction") == 4


def test_fewer_draws_than_domains_never_doubles_up(gof):
    for seed in range(20):
        b = gof.stratified_batch(POOL, 2, random.Random(seed))
        assert len(b) == 2
        assert len(set(r["category"] for r in b)) == 2, "drew two from one domain"


def test_records_are_distinct_within_a_step(gof):
    b = gof.stratified_batch(POOL, 8, random.Random(11))
    assert len({r["id"] for r in b}) == len(b)


def test_an_unlabelled_pool_falls_back_instead_of_raising(gof):
    plain = [{"id": i} for i in range(20)]
    b = gof.stratified_batch(plain, 4, random.Random(1))
    assert len(b) == 4


def test_a_single_domain_pool_falls_back(gof):
    one = [{"category": "Math", "id": i} for i in range(20)]
    b = gof.stratified_batch(one, 4, random.Random(1))
    assert len(b) == 4


def test_asking_for_more_than_the_pool_holds_does_not_hang(gof):
    tiny = ([{"category": "Math", "id": 1}, {"category": "Code", "id": 2}])
    b = gof.stratified_batch(tiny, 10, random.Random(1))
    assert 0 < len(b) <= 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# --------------------------------------------------------------------------
# coverage is an INVARIANT of every step; maxent chooses WITHIN a domain
# --------------------------------------------------------------------------
# Operator 2026-09-13: "always have full coverage each step. Thats the key. So
# if one doesnt have entropy, we swap out that record for another of the same
# type / This way we always have coverage"
#
# The first implementation stratified round 1 only and then handed the batch to
# maxent, which weights the WHOLE pool and can return four Math records.
def _hist(pool, spanning=(), settled=()):
    """Reward history: spanning records straddle the floor, settled ones don't."""
    h = {}
    for r in pool:
        q = r.get("question", r["id"])
        if q in spanning:
            h[q] = [0.2, 0.9]        # straddles 0.7 -> still uncertain
        elif q in settled:
            h[q] = [0.95, 0.98]      # settled above the floor -> no entropy
    return h


def _qpool():
    out = []
    for cat, n in (("Math", 5), ("Code", 5), ("STEM", 5), ("Instruction", 5)):
        for i in range(n):
            out.append({"category": cat, "id": "%s%d" % (cat, i),
                        "question": "%s%d" % (cat, i)})
    return out


def test_every_domain_appears_even_when_only_one_domain_has_entropy(gof):
    pool = _qpool()
    # ONLY Math records still span the floor; everything else is settled
    h = _hist(pool, spanning=["Math0", "Math1"],
              settled=[r["question"] for r in pool if r["category"] != "Math"])
    b = gof.covered_maxent_batch(pool, h, 0.7, 4, random.Random(5))
    assert cats(b) == ["Code", "Instruction", "Math", "STEM"], \
        "maxent starved the other domains: %s" % cats(b)


def test_within_a_domain_the_uncertain_record_is_preferred(gof):
    pool = _qpool()
    h = _hist(pool, spanning=["Code3"],
              settled=[r["question"] for r in pool
                       if r["category"] == "Code" and r["question"] != "Code3"])
    picks = [next(r for r in gof.covered_maxent_batch(pool, h, 0.7, 4,
                                                      random.Random(s))
                  if r["category"] == "Code")["id"]
             for s in range(12)]
    assert picks.count("Code3") >= 9, "maxent did not favour the spanning record"


def test_a_domain_with_no_entropy_is_swapped_not_dropped(gof):
    pool = _qpool()
    # every STEM record settled; the slot must still be filled by a STEM record
    h = _hist(pool, settled=[r["question"] for r in pool if r["category"] == "STEM"])
    for seed in range(10):
        b = gof.covered_maxent_batch(pool, h, 0.7, 4, random.Random(seed))
        assert "STEM" in cats(b)


def test_records_within_a_step_are_distinct(gof):
    pool = _qpool()
    b = gof.covered_maxent_batch(pool, {}, 0.7, 8, random.Random(2))
    assert len({r["id"] for r in b}) == len(b)


def test_sixteen_draws_stay_balanced_across_domains(gof):
    pool = _qpool()
    b = gof.covered_maxent_batch(pool, {}, 0.7, 16, random.Random(4))
    for c in ("Math", "Code", "STEM", "Instruction"):
        assert cats(b).count(c) == 4


def test_an_unlabelled_pool_falls_back_to_plain_maxent(gof):
    plain = [{"id": i, "question": "q%d" % i} for i in range(10)]
    b = gof.covered_maxent_batch(plain, {}, 0.7, 4, random.Random(1))
    assert len(b) == 4


# --------------------------------------------------------------------------
# every slot carries a DIFFERENTIABLE record (operator 2026-09-13)
# --------------------------------------------------------------------------
# "each step is always 4 differentiable signals ... we save those [all-0
# records] for the end and dont even count them towards differentiable
# iterations". Measured before this: 8-10 of every 12 records were dead, so a
# "4-record step" was often one real gradient plus three of nothing.
def test_a_replacement_comes_from_the_same_domain(gof):
    pool = _qpool()
    dead = next(r for r in pool if r["category"] == "Code")
    repl = gof.pick_same_domain(pool, "Code", random.Random(3), exclude=[dead])
    assert repl["category"] == "Code", "coverage broke when replacing a dead record"
    assert repl["id"] != dead["id"], "redrew the record we just deferred"


def test_replacements_avoid_everything_already_deferred(gof):
    pool = _qpool()
    code = [r for r in pool if r["category"] == "Code"]
    deferred = code[:-1]                      # only one Code record left
    repl = gof.pick_same_domain(pool, "Code", random.Random(1), exclude=deferred)
    assert repl["id"] == code[-1]["id"]


def test_when_a_whole_domain_is_deferred_it_still_returns_one(gof):
    # exhaustion must not return None and blow up the step
    pool = _qpool()
    code = [r for r in pool if r["category"] == "Code"]
    repl = gof.pick_same_domain(pool, "Code", random.Random(2), exclude=code)
    assert repl is not None and repl["category"] == "Code"


def test_an_unknown_domain_returns_none_rather_than_a_wrong_domain(gof):
    pool = _qpool()
    assert gof.pick_same_domain(pool, "Nonexistent", random.Random(1)) is None


def test_the_attempt_cap_is_bounded_and_defaulted(gof):
    # an all-settled domain would otherwise loop forever, and every attempt costs
    # a full generation + judging pass
    assert gof.DEFAULTS["max_draw_attempts"] >= 1
    src = io.open(_o.path.join(_o.path.dirname(_o.path.dirname(
        _o.path.abspath(__file__))), "docs", "grpo_one_file.py"),
        encoding="utf-8").read()
    assert "exhausted %d draws, all uniform" in src, \
        "exhaustion must announce itself, not silently accept a dead record"
