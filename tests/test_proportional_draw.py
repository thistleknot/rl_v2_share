"""Slots are allocated from the POOL's domain profile, not cycled uniformly.

NO GOVERNING SPEC IN THE TRAINER; the rule is judge-rubric.md:225 -- "measured R3 profile
-- {Math 0.331, Code 0.254, STEM 0.202, Instruction 0.213} -- which at n=15 yields
{Math 5, Code 4, STEM 3, Instruction 3}. It is derived, never hardcoded."

MEASURED 2026-09-16 on the live RL run, which was using uniform cycling:

    drawn   STEM 25%  Instruction 25%  Code 25%  Math 25%
    pool    Math 31%  Code 24%  Instruction 23%  STEM 22%

Math records were therefore visited less often than STEM records for no reason anyone
chose. Operator recalled the intended split from memory -- "I think its like 5, 4, 4, 3" --
before it was measured.
"""
import collections
import importlib.util
import random
import sys


POOL = {"Math": [0] * 74, "Code": [0] * 58, "Instruction": [0] * 55, "STEM": [0] * 53}


def _mod():
    spec = importlib.util.spec_from_file_location("gof", "docs/grpo_one_file.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _counts(n, seed=7):
    return collections.Counter(_mod().proportional_order(POOL, n, random.Random(seed)))


def test_sixteen_slots_give_the_operators_split():
    assert dict(_counts(16)) == {"Math": 5, "Code": 4, "Instruction": 4, "STEM": 3}


def test_the_allocation_always_sums_to_n_draw():
    """Rounding each share independently does NOT sum: 4.96+3.87+3.67+3.53 -> 5+4+4+4 = 17.
    Largest-remainder is why this holds."""
    for n in (1, 3, 4, 7, 12, 16, 31, 64):
        assert sum(_counts(n).values()) == n, n


def test_a_bigger_domain_never_gets_fewer_slots():
    for n in (8, 16, 32):
        c = _counts(n)
        assert c["Math"] >= c["Code"] >= c["Instruction"] >= c["STEM"], (n, dict(c))


def test_it_tracks_the_pool_rather_than_a_hardcoded_table():
    """Change the pool, the allocation must follow -- 'derived, never hardcoded'."""
    m = _mod()
    flipped = {"Math": [0] * 10, "Code": [0] * 10, "Instruction": [0] * 10,
               "STEM": [0] * 200}
    c = collections.Counter(m.proportional_order(flipped, 16, random.Random(7)))
    assert c["STEM"] > c["Math"], dict(c)


def test_uniform_cycling_is_gone_from_EXECUTABLE_code():
    """The docstring quotes the old line on purpose, so this checks live code only --
    a naive whole-file grep fails for the wrong reason."""
    import io as _io
    live = [l for l in _io.open("docs/grpo_one_file.py", encoding="utf-8")
            if not l.lstrip().startswith("#")]
    bad = [l for l in live if "order = [names[i %" in l]
    assert not bad, bad


def test_both_samplers_use_the_same_allocation():
    """boltzmann_batch and maxent_batch both draw per domain. If only one is
    proportional the mix changes with the sampler -- a hidden variable."""
    import io as _io
    src = _io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    assert src.count("order = proportional_order(buckets, n_draw, rng)") == 2


def test_every_domain_still_appears_when_slots_allow():
    """Proportional must not starve a small domain out of a batch big enough for it."""
    c = _counts(16)
    assert all(c[k] >= 1 for k in POOL), dict(c)
