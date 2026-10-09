"""RL early stopping: counted from step 1, ties are stale, 0 disables.

NO GOVERNING SPEC. Basis: operator 2026-09-14 --

> "in rl full we cap concerns at 3 and just keep running steps until we hit patience
>  of 5. A patience of 5 handles this."
> "i meant to say will it automatically consider step 3 in its patience of 5 and yes
>  it will unless it continues gaining for much longer"

Promote to a REQ before the RL phase depends on it.

WHY THERE IS NO "START COUNTING AFTER STEP N" KNOB. Patience 5 cannot fire before the
6th evaluation, so every early step is reached and evaluated regardless. A second knob
would be two ways to express one thing, and they would drift.

WHY THE FLOOR MATTERS AT ALL. This lane DIPS before it climbs -- full-param SFT reads
0.172 at epoch 1, BELOW the 0.271 base, and peaks at 0.424 by epoch 3. A patience of 1
or 2 kills it at the dip (instructions.md 1.2a, which exists because that nearly
happened twice).
"""
import io
import re

SRC = "docs/grpo_one_file.py"


def src():
    return io.open(SRC, encoding="utf-8").read()


def _loop():
    """The ROUND LOOP body only. Slicing from the baseline print would include the
    pre-loop initialisation and the end-of-run save, and both look like the things
    these tests are trying to forbid."""
    s = src()
    return s[s.index("for rd in range(cfg[\"rounds\"])"):s.index("if stop_early:")]


def test_zero_disables_so_every_measured_run_is_unaffected(_=None):
    """Every RL number on the board was taken with no early stopping. If the
    default ever becomes non-zero, the board stops being comparable and nothing
    would announce it."""
    assert 'ap.add_argument("--patience", type=int, default=0,' in src()
    assert "if args.patience and stale >=" in _loop(), \
        "a falsy patience must skip the check entirely, not compare against 0"


def test_a_new_best_resets_the_counter(_=None):
    body = _loop()
    i = body.index("if score > best:")
    j = body.index("else:", i)
    assert "stale = 0" in body[i:j], "progress must clear the counter"


def test_a_tie_counts_as_stale_and_saves_nothing(_=None):
    """A tie is not progress and not a regression. It must not reset patience --
    otherwise a model oscillating at one score never stops -- and it must not
    save, because `best` already holds a checkpoint at that score."""
    body = _loop()
    i = body.index("else:", body.index("if score > best:"))
    tail = body[i:]
    assert "stale += 1" in tail
    assert "save_pretrained" not in tail, "a tie must not write a second copy"


def test_the_counter_is_not_reset_anywhere_inside_the_loop(_=None):
    """The classic defect: re-initialising the counter at the top of each round,
    so patience silently never accumulates across rounds."""
    body = _loop()
    assert body.count("stale = 0") == 1, \
        "exactly one reset, on a new best -- a per-round reset defeats patience"


def test_stopping_breaks_out_of_the_round_loop_not_just_the_step_loop(_=None):
    """Breaking only the inner loop would start the next round and keep training
    after the stop was announced."""
    s = src()
    assert "stop_early = True" in s
    assert re.search(r"if stop_early:\s*\n\s*break", s), \
        "the outer round loop must break too"


def test_the_best_checkpoint_is_still_written_before_stopping(_=None):
    """Patience decides WHEN to stop; it must not change WHAT survives. A prior
    run stopped at step 6 with `best=0.5000 at None` -- no weights for the best
    number it produced -- and that failure must not come back via this path."""
    body = _loop()
    i = body.index("if score > best:")
    j = body.index("else:", i)
    assert "save_pretrained" in body[i:j]
    assert "NEW BEST" in body[i:j]


def test_patience_five_cannot_fire_before_the_sixth_reading(_=None):
    """The arithmetic the operator relied on, as executable arithmetic rather
    than a comment: with patience 5, step 3 is always reached."""
    patience, stale, stopped_at = 5, 0, None
    for step in range(1, 21):
        improved = step == 1          # worst case: nothing improves after step 1
        stale = 0 if improved else stale + 1
        if stale >= patience:
            stopped_at = step
            break
    assert stopped_at == 6, stopped_at
    assert stopped_at > 3, "step 3 must always be reached and evaluated"


# ------------------------------------------------- where the wrapper sets it ----
def test_hpo_arms_never_stop_early_but_the_full_run_does():
    """The asymmetry is the point, and it is easy to get backwards.

    An HPO arm exists to produce a curve COMPARABLE to its siblings at a fixed
    step count; stopping one early makes the bracket compare a 4-step arm against
    a 6-step arm, which is the one thing it cannot tolerate. The full run has no
    sibling, and every extra step costs judging.
    """
    import importlib.util
    import random
    import sys
    spec = importlib.util.spec_from_file_location("pr2", "archive/tools/phase_runner.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["pr2"] = m
    spec.loader.exec_module(m)

    class _A:
        base, mix = "ckpt", "pool.jsonl"
        hpo_steps, rps, eval_reads, seed = 6, 4, 3, 7
    hpo_arm = m.PHASES["rl"]["arms"](3, random.Random(0))[0]
    cmd = m.PHASES["rl"]["cmd"](hpo_arm, _A, "rl_hpo_0")
    assert cmd[cmd.index("--patience") + 1] == "0", "an HPO arm must run to its cap"

    full_arm = dict(hpo_arm)
    full_arm["_patience"] = 5
    cmd = m.PHASES["rl"]["cmd"](full_arm, _A, "rl_full")
    assert cmd[cmd.index("--patience") + 1] == "5"


def test_the_full_run_sets_patience_from_a_flag_not_a_literal():
    src_ = io.open("archive/tools/phase_runner.py", encoding="utf-8").read()
    assert 'arm["_patience"] = a.full_patience' in src_
    assert '"--full-patience", type=int, default=5' in src_
