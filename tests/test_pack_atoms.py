"""Unit tests: cpt.pack_atoms — record-aware packing for the CPT corpus.

The defect these pin: `build_corpus` used to join every record with "\\n\\n" and re-slice
at a fixed 512 tokens, cutting across record boundaries and mid-sentence. The skill chunks
arrive already segmented on paragraph boundaries, so that re-slice threw the segmentation
away AND mixed unrelated skills into one training sequence. Nothing in the loss curve shows
that; it only shows up as a model that never learned a skill it was shown.
"""
from __future__ import annotations
import os as _o, sys as _s

import pytest

from rlv2.train.cpt import train as cpt


class FakeTok:
    """One token per whitespace word. Keeps the test about packing, not tokenisation."""

    def encode(self, text, add_special_tokens=False):
        return [hash(w) % 1000 for w in text.split()]


TOK = FakeTok()


def _texts(*sizes):
    return [" ".join("w%d" % i for i in range(n)) for n in sizes]


# --------------------------------------------------------------------------
# the budget invariant
# --------------------------------------------------------------------------
def test_no_output_record_exceeds_the_budget():
    out, _ = cpt.pack_atoms(_texts(30, 40, 50, 60), TOK, budget=100)
    assert all(len(r) <= 100 for r in out), [len(r) for r in out]


def test_atoms_are_never_split_across_records():
    """A record of exactly the budget must come back whole, not as budget-1 + 1."""
    out, _ = cpt.pack_atoms(_texts(10, 100, 10), TOK, budget=100)
    assert 100 in [len(r) for r in out], "the 100-token atom was split"


def test_greedy_fill_actually_packs_rather_than_one_atom_per_record():
    """If each atom got its own record we would waste most of the window -- that is the
    naive fix to the concatenate bug and it is nearly as wasteful."""
    out, stats = cpt.pack_atoms(_texts(*([20] * 10)), TOK, budget=100)
    assert len(out) == 2, "10 x 20 tokens should pack into 2 records of 100"
    assert stats["fill"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# conservation — the assert that makes a packing bug loud
# --------------------------------------------------------------------------
def test_every_atom_is_placed_exactly_once():
    sizes = [7, 13, 31, 5, 64, 2]
    out, stats = cpt.pack_atoms(_texts(*sizes), TOK, budget=64)
    assert stats["atoms"] == len(sizes)
    assert sum(len(r) for r in out) == sum(sizes), "tokens lost or duplicated"


def test_empty_and_whitespace_records_are_dropped_not_counted():
    out, stats = cpt.pack_atoms(["", "   ", "a b c"], TOK, budget=10)
    assert stats["atoms"] == 1
    assert sum(len(r) for r in out) == 3


# --------------------------------------------------------------------------
# the over-budget atom — reported, never silently dropped
# --------------------------------------------------------------------------
def test_an_oversized_atom_fails_closed_rather_than_truncating():
    """The 512-token cap is enforced UPSTREAM by the length gate, which chunks skills and
    drops everything else with a printed receipt. An over-cap record arriving here means
    that gate was bypassed, and truncating would silently restore the mid-sentence cut the
    gate exists to prevent -- so this raises rather than repairs."""
    with pytest.raises(ValueError, match="length gate"):
        cpt.pack_atoms(_texts(500), TOK, budget=100)


def test_the_failure_names_the_offending_size_and_the_fix():
    """A fail-closed error that does not say what to do next just moves the confusion."""
    with pytest.raises(ValueError) as e:
        cpt.pack_atoms(_texts(10, 500, 10), TOK, budget=100)
    msg = str(e.value)
    assert "500" in msg and "100" in msg, "must name the actual sizes"
    assert "rlv2 data cpt-mix" in msg, "must name the command that fixes it"


# --------------------------------------------------------------------------
# contract
# --------------------------------------------------------------------------
def test_non_positive_budget_fails_fast():
    with pytest.raises(ValueError):
        cpt.pack_atoms(_texts(5), TOK, budget=0)


def test_empty_input_is_not_a_crash():
    out, stats = cpt.pack_atoms([], TOK, budget=64)
    assert out == [] and stats["atoms"] == 0


# --------------------------------------------------------------------------
# the live corpus — the thing that actually trains
# --------------------------------------------------------------------------
CORPUS = _o.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))),
                      "data", "rl", "cpt_mix_r3.jsonl")


@pytest.mark.skipif(not _o.path.exists(CORPUS), reason="CPT mix not built")
def test_live_corpus_packs_without_losing_records():
    import json
    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8") if l.strip()]
    texts = [r["text"] for r in rows if (r.get("text") or "").strip()]
    out, stats = cpt.pack_atoms(texts, TOK, budget=512)
    assert stats["atoms"] == len(texts)
    assert all(len(r) <= 512 for r in out)
    assert stats["fill"] > 0.5, "packing should fill more than half the window on average"
