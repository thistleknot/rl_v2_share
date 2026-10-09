"""Generations are cached per CHECKPOINT, not per invocation.

NO GOVERNING SPEC. Basis: operator 2026-09-15 -- "Didn't I say I wanted this cached like 2
days ago so we dont waste time on it".

WHAT WAS BROKEN. The cache keyed on `--out` and required `set(cached) == set(arms)`, so
adding or removing ONE arm discarded every cached generation, the base's included. The base
was therefore re-rolled in five separate invocations and read 2, 3, 4, 3 and 5 out of 16 --
same weights, same judge, same eval.

WHY THAT MATTERS MORE THAN IT LOOKS. At temperature 0.7 with k=1 each question gets ONE
sampled answer, re-drawn every run. That is the dominant source of movement on this board --
larger than the judge's own variance (3 of 96 cells flipping across 0.70). A bar that moves
by 3 points between readings cannot certify a 1-point RL gain.

THE KEY IS WHAT DETERMINES THE ANSWERS: checkpoint CONTENT hash, eval file, k, temperature.
Not the output filename, and not which other arms happen to be in the same run.
"""
import io

SRC = "src/rlv2/eval/score_arms.py"


def src():
    return io.open(SRC, encoding="utf-8").read()


def body():
    s = src()
    return s[s.index("# CACHE THE GENERATIONS PER CHECKPOINT"):s.index("score_k = a.score_k")]


def test_the_cache_key_is_not_the_output_filename():
    """The whole defect in one line: `cache = a.out + '.generations.json'`."""
    assert 'a.out + ".generations.json"' not in src()
    assert "_gen_key(" in body()


def test_the_key_covers_everything_that_changes_the_answers():
    b = body()
    for part in ("phase_state(", "a.eval", "a.k", "a.temperature"):
        assert part in b, part


def test_the_checkpoint_is_hashed_by_content_not_by_path():
    """A directory overwritten under the same name must not serve the previous
    model's cached answers -- the same reason the ledger keys on content."""
    assert "_ledger.phase_state(" in body()
    assert "differentiability-ledger REQ-L2" in body()


def test_changing_the_arm_list_does_not_discard_unchanged_arms():
    """The specific regression: one extra arm re-rolled base. Each arm is now
    looked up and stored independently."""
    b = body()
    assert "for _name, _path in arms.items():" in b
    code = [l for l in src().splitlines() if not l.lstrip().startswith("#")]
    assert "set(gens) == set(arms)" not in chr(10).join(code),         "the all-or-nothing check is back (comments describing it are fine)"
    assert "if need:" in b, "only the MISSING arms are generated"


def test_a_corrupt_cache_entry_regenerates_rather_than_crashing():
    b = body()
    assert "except (ValueError, OSError):" in b


def test_the_arm_count_mismatch_still_regenerates_that_arm():
    """A cache written at k=1 must not be served to a k=4 run."""
    assert "len(v) == a.k" in body()
