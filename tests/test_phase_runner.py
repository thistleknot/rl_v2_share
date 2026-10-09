"""The phase wrapper: proxy-or-real is an argument, HPO is structural.

Basis: operator 2026-09-14 --

> "Means you didnt create it at the right abstraction ... those two are interchangeable
>  for our rl wrapper, we just pass an argument if we're doing proxy (pyreft) or not."
> "Thats also how we never miss hpo. Phase wrapper. We always do hpo up front."
> "We can optionally load a config and skip hpo if its settled."

WHAT WENT WRONG THAT THIS PREVENTS. Two tools were built doing one job for two trainers --
bracket_redraw.py for RL and a reft_sweep.py for pyreft (since deleted). The duplication
was the symptom. The cause was treating "proxy" as a different KIND of run rather than a
parameter of the same run. And because HPO lived inside each tool instead of around them,
it was skipped entirely: every "pyreft HPO" this session was one configuration.

THREE PROPERTIES PINNED HERE.

1. THE PHASES SHARE EVERYTHING EXCEPT trainer, axes, parser. If any other logic becomes
   phase-specific, the abstraction has leaked and the duplication comes back.
2. HPO RUNS UNLESS A SETTLED CONFIG IS NAMED. Not a default, not a flag that defaults to
   off -- skipping requires pointing at an artifact.
3. THE TWO BARS STAY SEPARATE. Clearing your own start (HPO) is not clearing SFT's best
   (the thesis). Conflating them nearly killed full-parameter SFT twice, which is why
   instructions.md 1.2a exists.
"""
import importlib.util
import io
import random
import sys

import pytest

SRC = "archive/tools/phase_runner.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("phase_runner", SRC)
    m = importlib.util.module_from_spec(spec)
    sys.modules["phase_runner"] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def a():
    """The wrapper's parsed args. ONE stub -- seven copies of this drifted apart
    every time a flag was added, and each addition broke all of them at once."""
    class _Args:
        base, mix = "skills_model", "pool.jsonl"
        hpo_steps, rps, probe_cap, eval_reads, seed = 6, 4, 60, 3, 7
        # RECORDS PER OPTIMIZER STEP DURING THE SCREEN, separate from `rps`:
        # the screen may batch differently from the full run, and the RL cmd
        # reads this one. Missing here, so every RL-cmd test died on the stub
        # rather than on the tool.
        hpo_rps = 4
        sft_out_dir, sft_mix = "sft_arms", "data/rl/r3_sft_subset.jsonl"
        rl_mix, eval_file = "data/rl/r3_rl.jsonl", "data/rl/r3_eval_canonical.jsonl"
        straddle_cap, straddle_full = 24, 60
        bins_per_epoch, length_blocks, band_sigmas = 13, 10, 3.0
        batch_tokens = 1280
        # NO KL, PER-TOKEN LOG PROBS (operator 2026-09-20). These live on the one
        # shared stub for the reason the docstring above gives: a flag added to
        # seven private copies drifts in seven directions.
        beta = 0.0
        logprob_norm = "mean"
        # THE SCREEN'S OWN POOL AND CAPS. The screen trains on 16 records x 3
        # epochs and the full run on 237, so they are separate settings selected
        # by tag rather than by anyone remembering a flag.
        sft_hpo_mix = "data/rl/r3_sft_hpo16.jsonl"
        straddle_hpo_cap = 16
        gen_batch = 32
        judge_window = 32
        cover_once = False          # opt-in; see the argparse note
    return _Args


@pytest.fixture
def sft_arm(mod):
    """A REAL arm from the builder. Hand-written arm dicts drifted every time an
    axis was added and broke nine tests at once -- the same failure the single `a`
    fixture already fixed for the args."""
    return mod.PHASES["sft"]["arms"](2, random.Random(0))[0]


@pytest.fixture(scope="module")
def src():
    return io.open(SRC, encoding="utf-8").read()


# ------------------------------------------------------- the abstraction ----
def test_every_phase_is_registered_in_one_table(mod, a):
    """sft-phase.md REQ-S6: SFT goes through the same wrapper, so HPO cannot be
    skipped for it either -- which is exactly how it got skipped before."""
    assert set(mod.PHASES) == {"sft", "pyreft", "rl"}


def test_a_phase_supplies_only_a_trainer_and_its_axes(mod, a):
    """If a phase ever needs a third key, something non-phase-specific has leaked
    into it and the two tools start diverging again."""
    for name, spec in mod.PHASES.items():
        assert set(spec) == {"cmd", "arms"}, (name, sorted(spec))


def test_the_two_phases_invoke_different_trainers(mod, a):

    reft = mod.PHASES["pyreft"]["cmd"](
        {"name": "x", "rank": 4, "layers": "8;16", "lr": 5e-4}, a, "t")
    rl = mod.PHASES["rl"]["cmd"](
        {"name": "x", "lr": 1e-6, "rollouts": 4, "grad_clip": 1.0,
         "is_beta": 0.0}, a, "t")
    assert any("reft_hpo.py" in c for c in reft)
    assert any("grpo_one_file.py" in c for c in rl)
    assert not any("grpo_one_file.py" in c for c in reft)


def test_each_phase_sweeps_only_axes_its_trainer_has(mod, a):
    """A pyreft arm carrying `rollouts` would be meaningless -- the proxy is
    supervised and has no group. An RL arm carrying `rank` likewise."""
    rng = random.Random(0)
    reft_keys, rl_keys = set(), set()
    for arm in mod.PHASES["pyreft"]["arms"](5, rng):
        reft_keys.update(arm)
    for arm in mod.PHASES["rl"]["arms"](5, rng):
        rl_keys.update(arm)
    assert reft_keys == {"name", "rank", "layers", "lr"}, reft_keys
    assert rl_keys == {"name", "lr", "rollouts", "grad_clip", "is_beta"}, rl_keys
    assert "rollouts" not in reft_keys, "the proxy has no group"
    assert "rank" not in rl_keys, "full-param RL has no ReFT rank"


def test_arm_zero_is_the_incumbent_in_both_phases(mod, a):
    """Without the incumbent the sweep has no reference point, in either lane."""
    for phase in ("pyreft", "rl"):
        arms = mod.PHASES[phase]["arms"](5, random.Random(0))
        assert arms[0]["name"] == "incumbent", phase


def test_the_rl_incumbent_carries_the_measured_defaults(mod, a):
    """Every RL number on the board was taken at these values."""
    inc = mod.PHASES["rl"]["arms"](3, random.Random(0))[0]
    assert inc["grad_clip"] == 1.0, "the hardcoded clip every run used"
    assert inc["is_beta"] == 0.0, "uncorrected, as every prior run was"
    assert inc["rollouts"] == 4


def test_the_rl_sweep_reaches_above_the_measured_gradient_norm(mod, a):
    """gnorm ran 137-278 against a clip of 1.0. A range that never exceeds that
    could not test 'effectively do not clip', which is the live hypothesis."""
    clips = [a["grad_clip"] for a in mod.PHASES["rl"]["arms"](40, random.Random(1))]
    assert max(clips) > 150, max(clips)
    assert min(clips) < 2.0, min(clips)


# -------------------------------------------------------------- parsing ----
def test_a_pyreft_log_parses_to_start_and_curve(mod, a):
    log = "\n".join([
        "[patience] baseline is the INPUT model: 5/16",
        "[step  1] 6/16  vs start 5/16  lift +1  peak 6/16  stale 0",
        "[step  2] 5/16  vs start 5/16  lift +0  peak 6/16  stale 1"])
    start, curve, n = mod._parse("pyreft", log)
    assert (start, curve, n) == (5, [6, 5], 16)


def test_an_rl_log_parses_with_the_first_eval_as_the_start(mod, a):
    log = "\n".join(["[eval] 6/16 correct", "[eval] 6/16 correct",
                     "[eval] 7/16 correct"])
    start, curve, n = mod._parse("rl", log)
    assert (start, curve, n) == (6, [6, 7], 16)


def test_an_empty_log_yields_no_start_so_the_arm_is_not_tested(mod, a):
    for phase in ("pyreft", "rl"):
        start, curve, _ = mod._parse(phase, "nothing useful here")
        assert start is None and curve == [], phase


# ----------------------------------------------------------------- bars ----
def test_wilson_matches_the_project_figure(mod, a):
    lo, hi = mod.wilson(4, 20)
    assert abs(lo - 0.081) < 0.002 and abs(hi - 0.416) < 0.002


def test_six_and_seven_of_sixteen_are_not_separable(mod, a):
    """Why the wrapper must report 'separated from nobody' rather than a winner:
    a one-question difference on this eval is inside the interval."""
    assert mod.wilson(6, 16)[1] > mod.wilson(7, 16)[0]


def test_hpo_is_default_and_skipping_needs_a_named_artifact(src):
    assert '"--config", default=None' in src
    assert "HPO SKIPPED -- settled config from" in src, \
        "a skip must announce itself and name the file"
    assert "never skipped by default and never silently" in src


def test_the_two_bars_are_reported_separately(src):
    """instructions.md 1.2a. The screen reports HPO-PASSED against the arm's own
    start; the eval bar is a FULL-RUN claim, and the runner now says so rather
    than printing "CLEARS THE SFT BAR" off a number it never measured."""
    assert "--sft-bar" in src
    assert "HPO-PASSED" in src
    assert "NOT TESTED" in src, "the eval bar is named as untested, not as cleared"


def test_an_hpo_that_produces_no_curve_stops_rather_than_running_a_full(src):
    """Article X: an invalid run is not a negative result, and it is certainly
    not a config worth spending a full run on."""
    assert "NO ARM PRODUCED A CURVE" in src
    assert "stopping: HPO produced nothing to run" in src


def test_no_separation_is_reported_honestly(src):
    """The verdict stays; the word "honest" does not. A message that ASSERTS its own
    honesty and withholds the number is the thing the operator objected to
    (2026-09-19). Honesty is the interval being printed, not the adjective."""
    assert "SEPARATED FROM NOBODY" in src
    assert "every other arm's interval overlaps it" in src


def test_the_probe_matches_the_training_group_size(mod, a):
    """K=2 detects a p=0.5 record only 50% of the time vs 87.5% at K=4, and a
    false negative ranks that record BELOW every unprobed one. The 2x cost is
    paid once, not per arm, because the ledger caches it (REQ-L2)."""
    cmd = mod.PHASES["pyreft"]["cmd"](
        {"name": "x", "rank": 4, "layers": "8;16", "lr": 5e-4}, a, "t")
    assert cmd[cmd.index("--probe-rollouts") + 1] == "4"


# ------------------------------------------------------------ the sft phase ----
def test_sft_is_ranked_on_straddle_and_never_on_the_eval(mod, a, sft_arm):
    """REQ-S1/S2. Ranking SFT on eval accuracy would re-select epoch 2 (7/16 vs
    epoch 3's 5/16) and bury the thesis before it is tested. And the straddle
    measurement must read RL's pool: the eval would be leakage, SFT's own
    training set would measure memorisation."""
    cmds = mod.PHASES["sft"]["cmd"](
        sft_arm, a, "sft_hpo_0")
    assert len(cmds) == 3, "train, straddle on RL data, eval on the fixed 16"
    train, score, _evalc = cmds
    assert any("vibe_sft.py" in c for c in train)
    assert any("straddle_probe.py" in c for c in score)
    assert "data/rl/r3_rl.jsonl" in score, "straddle is measured on RL's pool"
    assert not any("eval" in str(c).lower() and "rl_eval" in str(c) for c in score)


def test_every_epoch_is_probed_and_the_base_is_not(mod, a, sft_arm):
    """EVERY EPOCH, so "pick the best epoch" is a real choice. Operator
    2026-09-19: one epoch on the 55-record Instruction pool IS full coverage, and
    the literature has pre-RL and post-RL peaks landing at different epochs.

    THE BASE IS DROPPED. It is identical across arms and already in the ledger, so
    re-probing it was a quarter of the bill for a number that cannot have moved --
    the arm's own start comes from its epoch 1 instead."""
    _train, score, _e = mod.PHASES["sft"]["cmd"](sft_arm, a, "sft_hpo_0")
    ckpts = score[score.index("--checkpoints") + 1].split(",")
    assert ckpts == ["sft_arms/sft_hpo_0_epoch%d" % e
                     for e in range(1, int(sft_arm["epochs"]) + 1)], ckpts
    assert a.base not in ckpts, "the base is measured once, not once per arm"


def test_sft_is_full_param_only(mod, a):
    """instructions.md 8: adapters for injection are FALSIFIED. This phase tunes
    the lane that works; it does not reopen a closed question."""
    rng = random.Random(0)
    keys = set()
    for arm in mod.PHASES["sft"]["arms"](6, rng):
        keys.update(arm)
        train = mod.PHASES["sft"]["cmd"](arm, a, "t")[0]
        assert "--full-param" in train
    # the sweep may grow optimizer axes; what it must NEVER grow is an adapter axis
    assert "rank" not in keys and "alpha" not in keys and "layers" not in keys


def test_an_sft_log_parses_the_first_reading_as_the_start(mod, a):
    """The straddle-count fallback, for a log with no available-gradient column."""
    log = "\n".join([
        "[straddle] skills_model     straddle  3/24 ( 13%)  solved  1  missed 20",
        "[straddle] sft_hpo_0        straddle  9/24 ( 38%)  solved  2  missed 13"])
    start, curve, n = mod._parse("sft", log)
    assert (start, curve, n) == (3, [3, 9], 24)


def test_every_epoch_reading_becomes_a_curve_point(mod, a):
    """THE DEFECT THIS CLOSES, measured 2026-09-19. The probe now measures every
    epoch and no base, so a 3-epoch arm logs THREE readings. The parser took two:
    epoch 1 as the start, epoch 2 as the whole curve, epoch 3 discarded -- and
    reported `start=450 best=441 lift=-9` for a curve that actually ran
    450 -> 441 -> 464. A rising arm was recorded as a regression and the winning
    epoch never appeared anywhere."""
    log = "\n".join([
        "[straddle] sft_hpo_0_epoch1 straddle  9/32 ( 28%)  solved 0  missed 23  "
        "grad   4.50 (mean 0.1406)  rollout-acc 0.06  0 min",
        "[straddle] sft_hpo_0_epoch2 straddle  8/32 ( 25%)  solved 0  missed 24  "
        "grad   4.41 (mean 0.1376)  rollout-acc 0.05  0 min",
        "[straddle] sft_hpo_0_epoch3 straddle 11/32 ( 34%)  solved 0  missed 21  "
        "grad   4.64 (mean 0.1451)  rollout-acc 0.08  0 min"])
    start, curve, n = mod._parse("sft", log)
    assert curve == [450, 441, 464], curve
    assert start == 450, "the arm\'s own start is its first epoch"
    assert max(curve) == 464 and curve.index(max(curve)) + 1 == 3, \
        "epoch 3 is the winner and must be findable from the curve"


def test_one_straddle_line_is_not_a_result(mod, a):
    """If the arm died before its own measurement, the input's line alone would
    otherwise read as a start with no curve -- or worse, as the arm's score."""
    log = "[straddle] skills_model     straddle  3/24 ( 13%)  solved  1  missed 20"
    assert mod._parse("sft", log) == (None, [], 0)


def test_a_failed_train_step_skips_the_scorer(src):
    """Scoring a checkpoint the trainer never finished writing would report a
    number for a model that does not exist. Article X: NOT TESTED, not bad."""
    assert "remaining steps skipped" in src
    assert "if rc != 0" in src


def test_sft_trains_on_the_domain_set_not_the_general_pool(mod, a, sft_arm):
    """Operator 2026-09-14: "we are not using rl_mix_vibe yet. Thats on non domain
    data" / "we are testing what matters domain knowledge ingestion keeping scope
    targeted". vibe_sft.py hardcodes a 2077-row general pool; this phase must
    override it, or the arms measure general-instruction tuning."""
    train, _score, _e = mod.PHASES["sft"]["cmd"](
        sft_arm, a, "t")
    assert "--subset" in train
    assert train[train.index("--subset") + 1] == "data/rl/r3_sft_subset.jsonl"
    assert not any("rl_mix_vibe" in c for c in train)


def test_the_three_datasets_stay_separate(mod, a, sft_arm):
    """Train on SFT's, measure on RL's, check disjointness against the eval. If
    any two of these collapse to one file the phase is measuring itself."""
    train, score, evalc = mod.PHASES["sft"]["cmd"](
        sft_arm, a, "t")
    assert train[train.index("--subset") + 1] != score[score.index("--mix") + 1]
    assert score[score.index("--mix") + 1] == "data/rl/r3_rl.jsonl"
    # The straddle probe must NEVER see the eval -- that is the leakage REQ-S2
    # bars, and it would look identical in the logs.
    assert "r3_eval_canonical.jsonl" not in " ".join(score)
    # ...and the eval scorer must never see RL's pool, or "accuracy" and
    # "straddle" quietly become the same measurement.
    assert "r3_rl.jsonl" not in " ".join(evalc)
    assert evalc[evalc.index("--eval") + 1] == "data/rl/r3_eval_canonical.jsonl"


def test_the_sft_training_set_is_r3_only_and_fully_answerable():
    """Operator 2026-09-14: "Were also doing r3 only types. I.e. no context field
    just self sufficient q, answerable (i.e. no abstentstion)). But we still treat
    abstentation as 0."

    An abstention row here would be actively harmful: vibe_sft.py encodes an empty
    gold as the literal refusal string, so the model would be TRAINED to say
    "unanswerable" on a lane where the reward pays 0.0 for exactly that.
    """
    import json
    import sys as _s
    _s.path.insert(0, "src")
    from rlv2.common.stratify import regime_of
    rows = [json.loads(l) for l in
            io.open("data/rl/r3_sft_subset.jsonl", encoding="utf-8") if l.strip()]
    assert len(rows) == 240
    assert all((r.get("gold") or "").strip() for r in rows), "an abstention row"
    assert not any((r.get("context") or "").strip() for r in rows), "not closed-book"
    assert {regime_of(r) for r in rows} == {"R3"}


def test_coverage_is_raised_by_bins_not_by_token_budget(mod, a, sft_arm):
    """Operator 2026-09-14: "increase the records PER epoch so we stay at 3".

    Two knobs reach records-per-epoch and only one is safe here. target_tokens
    sets the BIN size, so raising it grows the batch -- and the run peaks at
    11.7 GB against a 15.1 GB ceiling. bins_per_epoch adds bins at the same batch
    size: same VRAM, more steps, more records.

    Measured 2026-09-14 at 12 bins: 52/55/55 records = 162 of 229 (71%), despite
    the trainer flag's help claiming "3 epochs ~= one full pass".

    SUPERSEDED IN PART, 2026-09-20: bins is still the coverage lever, but the value
    is no longer a constant chosen here. It is DERIVED in the trainer from the pool
    it loaded, because a fixed count gives different coverage on every pool -- 16
    bins covered 38% of a 237-row pool and the whole of a 16-row one. 0 means derive.
    """
    train = mod.PHASES["sft"]["cmd"](
        sft_arm, a, "t")[0]
    assert train[train.index("--bins-per-epoch") + 1] == "0",         "bins is derived for full coverage, not set from a constant"
    assert "--target-tokens" not in train, "the batch-size knob is not the lever"
    assert train[train.index("--epochs") + 1] == "3", "epochs stays at 3"
    # THREE FULL PASSES IS THE DEFAULT. Operator 2026-09-19, settling it: "3 epochs
    # sees 237 x 3 times / Each epoch sees 237". An epoch is a full pass, so the
    # epochs RESAMPLE the whole pool; making every record appear once for the whole
    # RUN is the opt-in --cover-once case. Coverage WITHIN an epoch is the bins
    # knob's job -- 32 bins covers 236 records at the measured ~7.4 records/bin,
    # where 10 bins reached only 73 of them and quietly trained three thirds.
    assert "--cover-once" not in train, \
        "an epoch is a full pass; partitioning the pool must be opt-in"


def test_the_band_drops_records_and_the_blocks_do_not(mod, a, sft_arm):
    """Two knobs that look alike and are not. `quantile_blocks` only BATCHES --
    each block pads to its own max, so more blocks is free coverage-wise and
    strictly less wasted VRAM. `filter_outliers` DISCARDS rows.

    Measured on the R3 domain set, 2026-09-14:

        4 blocks, sigmas 2.0  ->  233 rows, 25842 padded slots (23.8% waste)
       10 blocks, sigmas 3.0  ->  237 rows, 22851 padded slots (11.7% waste)

    More records AND less padding, because one 480-token row was costing ~4000
    slots by itself. So the band is loosened and the blocks absorb the spread.
    """
    train = mod.PHASES["sft"]["cmd"](
        sft_arm, a, "t")[0]
    assert train[train.index("--length-blocks") + 1] == "10"
    assert float(train[train.index("--band-sigmas") + 1]) > 2.0, \
        "the band must be LOOSER than the trainer default, or coverage falls"


def test_the_scored_checkpoint_is_one_the_trainer_actually_writes(mod, a):
    """The bug this closes cost nothing only because it was caught before the
    sweep finished. vibe_sft writes `<out>_epoch1..N` and `<out>_run`; `<out>`
    itself is never created. Pointing the probe at `<out>` made straddle_probe
    log "SKIP -- not on disk", emit one straddle line instead of two, and the
    wrapper reported NOT TESTED for every arm -- after three hours of training.

    EVERY epoch is probed, so the list runs 1..N and the LAST element is the
    final epoch -- the path has to track whatever the arm carries, never a
    hardcoded _epoch3.
    """
    for epochs in (2, 3, 4):
        arm = dict(mod.PHASES["sft"]["arms"](2, random.Random(0))[0], epochs=epochs)
        _t, score, _e = mod.PHASES["sft"]["cmd"](arm, a, "sft_hpo_0")
        got = score[score.index("--checkpoints") + 1].split(",")
        assert len(got) == epochs, got
        assert got[-1].endswith("_epoch%d" % epochs), got


def test_an_sft_arm_reports_both_metrics_never_one(mod, a, sft_arm):
    """Operator 2026-09-15: "Eval is always eval. Straddle uses rl [data]. Two
    different metrics the entire time."

    The first sweep shipped straddle with NO eval number anywhere -- REQ-S1 said
    to record both and the step was never built. That left RL with no accuracy
    bar to clear, which is the entire point of running SFT first.

    straddle -> r3_rl.jsonl      what RL can LEARN from   -> ranks the arms
    eval     -> the fixed 16     the pipeline's score     -> RL's BAR
    """
    cmds = mod.PHASES["sft"]["cmd"](
        sft_arm, a, "sft_hpo_0")
    assert len(cmds) == 3, "train, straddle, eval"
    train, straddle, evalc = cmds
    assert any("vibe_sft.py" in c for c in train)
    assert any("straddle_probe.py" in c for c in straddle)
    assert any("score_arms.py" in c for c in evalc)

    # the two metrics must read DIFFERENT files, or one of them is a fiction
    assert straddle[straddle.index("--mix") + 1] == "data/rl/r3_rl.jsonl"
    assert evalc[evalc.index("--eval") + 1] == "data/rl/r3_eval_canonical.jsonl"
    assert straddle[straddle.index("--mix") + 1] != evalc[evalc.index("--eval") + 1]


def test_the_eval_uses_the_locked_scoring_mechanism(mod, a, sft_arm):
    """dispositions.md: one mechanism, every phase, or a quoted result is
    meaningless. `single` judges each (question, answer) alone, 3x, majority."""
    _t, _s, evalc = mod.PHASES["sft"]["cmd"](
        sft_arm, a, "sft_hpo_0")
    assert evalc[evalc.index("--protocol") + 1] == "single"
    assert "base=%s" % a.base in evalc, "the eval needs its own baseline row"


def test_a_killed_arm_is_not_tested_not_a_zero_lift(mod):
    """Article X. Measured 2026-09-15: pyreft arm 0 burned its 2700 s budget on the
    one-time priority probe and was killed during step 2. One step had landed, so
    the parser returned start=3/best=3 and the bracket wrote down `lift +0` -- an
    invalid run entered the board as a real negative, against which four valid arms
    would then have been ranked.
    """
    log = "\n".join([
        "[patience] baseline is the INPUT model: 3/16",
        "[step  1] 3/16  vs start 3/16  lift +0  peak 3/16  stale 1",
        "",
        "[phase] KILLED at 2700 s"])
    assert mod._parse("pyreft", log) == (None, [], 0)


def test_an_unkilled_run_with_the_same_curve_still_parses(mod):
    """The guard must key on the kill marker, not on a short curve -- a genuinely
    flat one-step arm is a real result and must survive."""
    log = "\n".join([
        "[patience] baseline is the INPUT model: 3/16",
        "[step  1] 3/16  vs start 3/16  lift +0  peak 3/16  stale 1"])
    assert mod._parse("pyreft", log) == (3, [3], 16)


def test_epochs_is_pinned_because_it_secretly_controlled_coverage(mod):
    """Operator 2026-09-15: "sft still needs to see all records and if it was 3
    epochs worth and you cut at 2. No bueno".

    Under --cover-once the epochs PARTITION the pool, so sweeping `epochs` swept
    COVERAGE at the same time. Measured: 2-epoch arms saw 196 of 237 records (83%),
    3-epoch arms saw all 237 -- and the sweep's winner was a 2-epoch arm, so its
    straddle advantage was confounded with training on 41 FEWER records. A
    one-variable-rule violation hiding inside an ordinary-looking hyperparameter.
    """
    for n in (2, 5, 9):
        arms = mod.PHASES["sft"]["arms"](n, random.Random(n))
        assert {a["epochs"] for a in arms} == {3}, \
            "every arm must cover the pool exactly once"


def test_the_full_optimizer_suite_is_swept_not_just_lr(mod):
    """Operator flagged this twice: "You know we have a full suit of optim parms.
    Did you again strip them to just lr". Freeing `epochs` from the sweep is what
    makes room for the axes that actually differ between arms."""
    keys = set()
    for arm in mod.PHASES["sft"]["arms"](12, random.Random(0)):
        keys.update(arm)
    for axis in ("adam_beta1", "adam_beta2", "adam_eps", "max_grad_norm",
                 "label_smoothing", "lr_scheduler", "warmup_frac", "weight_decay"):
        assert axis in keys, axis


def test_every_swept_axis_reaches_the_trainer(mod, a):
    """An axis in the arm dict that is never passed on the command line is a sweep
    that varies nothing -- the failure mode is silent and looks like a tie."""
    arm = mod.PHASES["sft"]["arms"](2, random.Random(1))[1]
    train = mod.PHASES["sft"]["cmd"](arm, a, "t")[0]
    for flag in ("--adam-beta1", "--adam-beta2", "--adam-eps", "--max-grad-norm",
                 "--label-smoothing", "--lr-scheduler"):
        assert flag in train, flag


def test_every_swept_scheduler_can_actually_be_constructed(mod, a):
    """`warmup_stable_decay` needs `num_decay_steps`, which vibe_sft never passes --
    so arms 2 and 3 both died with a TypeError before step 1 and came back NOT
    TESTED. A swept value the trainer cannot build is a wasted arm, and the sweep
    cannot tell that from a genuinely bad setting."""
    from transformers import SchedulerType
    buildable = {s.value for s in SchedulerType}
    seen = set()
    for arm in mod.PHASES["sft"]["arms"](40, random.Random(3)):
        seen.add(arm["lr_scheduler"])
    assert seen <= buildable, seen - buildable
    assert "warmup_stable_decay" not in seen, \
        "needs num_decay_steps; vibe_sft does not pass it"


def test_the_rl_arm_pays_the_three_reward_bands_not_brevity(mod, a):
    """THE DEFECT THIS CLOSES, measured 2026-09-19. `--quality-bands` existed only
    on the trainer, so every run routed through this wrapper fell through to the
    vibethinker Long2Short BREVITY shaping. One bracket paid 0.00 / 0.89 / 0.98 /
    1.00 / 1.19 / 1.20 and never printed the "[reward] bands:" line -- length was
    pricing the reward on a lane whose premise is domain correctness.

    Operator 2026-09-19, from their own notes: "there are actually 3 classes
    (0, .5, 1) based on 0, .7, .9 thresholds"."""
    cmd = mod.PHASES["rl"]["cmd"](
        {"name": "x", "lr": 1e-5, "rollouts": 4, "grad_clip": 1.0, "is_beta": 0.0},
        a, "t")
    assert "--quality-bands" in cmd, "a flag with no pass-through is silently off"
    assert float(cmd[cmd.index("--hq-floor") + 1]) == 0.9
    assert float(cmd[cmd.index("--lq-reward") + 1]) == 0.5


def test_the_bands_can_be_turned_off_explicitly(mod, a):
    """Off is a decision someone has to make on purpose, never a default that
    happens because nobody wired the flag."""
    # the `a` fixture hands back the CLASS, not an instance, so subclass it directly
    _Off = type("_Off", (a,), {"quality_bands": False})
    cmd = mod.PHASES["rl"]["cmd"](
        {"name": "x", "lr": 1e-5, "rollouts": 4, "grad_clip": 1.0, "is_beta": 0.0},
        _Off, "t")
    assert "--quality-bands" not in cmd


def test_the_separation_verdict_prints_the_interval_that_produced_it(src):
    """Operator 2026-09-19: "Wtf does Separated from nobody mean / Sounds like
    unqualified llm speak."

    It was never speak -- an arm counts as SEPARATED only when its 95% Wilson upper
    bound falls below the leader's lower bound, which is a real and fairly strict
    test. The defect was that the message stated the VERDICT and withheld the NUMBER,
    so a reader could not check it or see how wide the intervals actually were. At
    3/16 the interval is [0.07, 0.40] -- 1.1 to 6.3 questions -- and once that is on
    the page "nobody is separated" stops sounding like hedging and starts reading as
    the arithmetic it is."""
    assert "95%% Wilson on %d/%d is [%.2f, %.2f]" in src, \
        "the verdict must carry its own interval"
    assert "their upper bounds fall below %.2f" in src, \
        "the positive case must name the bound it cleared too"
    assert "l_lo, l_hi = wilson(" in src, "both bounds are needed to print the range"


def test_a_tie_on_best_is_broken_by_the_curve_not_by_list_order(mod, a):
    """Operator 2026-09-19, on being shown the same split three brackets running:
    "It sounds like you've raised a bug that we need to address."

    max() returns the FIRST maximum on ties, so two arms at best=3 were ranked by
    where they happened to sit in the list -- and the full run went three times to
    the arm whose curve read 3 1 1 over the one that read 3 3 3. `best` stays the
    primary key (instructions.md 1.2a); the mean of the curve is what separates an
    arm that HELD its gain from one that gave it back."""
    src = io.open(SRC, encoding="utf-8").read()
    assert 'key=lambda x: (x["best"],' in src, "best must remain the primary key"
    assert 'sum(x["curve"]) / max(1, len(x["curve"]))' in src, \
        "the tie-break must read the whole curve, not just its maximum"

    # and the arithmetic it implies
    held = {"best": 3, "curve": [3, 3, 3]}
    gave_back = {"best": 3, "curve": [3, 1, 1]}
    key = lambda x: (x["best"], sum(x["curve"]) / max(1, len(x["curve"])))
    assert max([gave_back, held], key=key) is held, \
        "listed first must not win when its curve is worse"
    # a genuinely better best still wins regardless of the curve
    better = {"best": 5, "curve": [5, 1, 1]}
    assert max([held, better], key=key) is better


def _ecdf_scores(vals):
    """Mirror of the midrank ECDF in phase_runner.hpo, for testing the contract."""
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
    out, i = [0.0] * n, 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        mid = (i + j) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = (mid + 0.5) / n
        i = j + 1
    return out


def test_the_midrank_ecdf_centres_duplicate_blocks(mod):
    """Operator 2026-09-19: "take the center of duplicate sets but treat the overall
    count as cumulative sum. I.e. value_counts."

    Ties must SHARE a position rather than being split by list order -- splitting on
    list order is the exact defect this key exists to remove."""
    assert _ecdf_scores([1.0, 2.0]) == [0.25, 0.75]
    # three-way tie: every member sits at the centre of the whole block
    assert _ecdf_scores([5.0, 5.0, 5.0]) == [0.5, 0.5, 0.5]
    # a tied pair at the bottom of THREE shares midrank 0.5 -> (0.5+0.5)/3 = 1/3,
    # and the singleton takes midrank 2 -> (2+0.5)/3 = 5/6. The block's centre moves
    # with n; it is not a fixed 0.25.
    got = _ecdf_scores([1.0, 1.0, 9.0])
    assert got[0] == got[1] == pytest.approx(1 / 3.0)
    assert got[2] == pytest.approx(5 / 6.0)


def test_the_ecdf_removes_the_scale_advantage_a_raw_average_would_give(mod):
    """The three signals live on different scales -- measured on the 2026-09-19
    bracket: grad ~37.5, straddle ~25, allfail ~40. Averaging them RAW lets the
    largest magnitude decide; the ECDF puts each on [0,1] against the other arms."""
    a_sig, b_sig = (37.45, 23.0, -41.0), (37.63, 26.0, -38.0)
    cols = [_ecdf_scores([a_sig[i], b_sig[i]]) for i in range(3)]
    a_score = sum(c[0] for c in cols) / 3
    b_score = sum(c[1] for c in cols) / 3
    assert b_score > a_score, "the arm winning all three must win the mean"
    assert (a_score, b_score) == (0.25, 0.75), "a 3-0 sweep is a clean 0.25 / 0.75"


def test_a_split_decision_goes_to_the_majority_not_to_the_biggest_number(mod):
    """THE CASE THAT SEPARATES THIS FROM LEXICOGRAPHIC. When grad favours A while
    straddle and all-fail favour B, lexicographic hands it to A on grad alone; the
    mean ECDF hands it to B, two signals to one."""
    a_sig = (99.0, 1.0, -99.0)    # wins grad only
    b_sig = (1.0, 99.0, -1.0)     # wins straddle and all-fail
    cols = [_ecdf_scores([a_sig[i], b_sig[i]]) for i in range(3)]
    a_score = sum(c[0] for c in cols) / 3
    b_score = sum(c[1] for c in cols) / 3
    assert b_score > a_score, "two signals out of three carry it"


def test_the_signal_score_never_outranks_the_eval(mod):
    """instructions.md 1.2a: the eval is the bar. The ECDF decides only what the
    eval and the curve left tied."""
    key = lambda x: (x["best"], sum(x["curve"]) / max(1, len(x["curve"])),
                     x.get("signal_ecdf", 0.0))
    strong_signal = {"best": 3, "curve": [3, 3, 3], "signal_ecdf": 1.0}
    better_eval = {"best": 4, "curve": [4, 4, 4], "signal_ecdf": 0.0}
    assert max([strong_signal, better_eval], key=key) is better_eval
    held = {"best": 3, "curve": [3, 3, 3], "signal_ecdf": 0.0}
    gave_back = {"best": 3, "curve": [3, 1, 1], "signal_ecdf": 1.0}
    assert max([gave_back, held], key=key) is held


def test_the_census_is_compared_by_dirichlet_not_by_rank(mod):
    """Operator 2026-09-19: "Why are we not using a direchlet type formula based on
    those 3, gradient, straddle, all_fail" -- and then the sharper version, "Isnt
    this where I use bernoulli variance but over 3 categories?"

    Two of the three ARE a composition: straddle/allfail/allwin partition the
    records that have been visited, so a Dirichlet is the natural posterior over
    their shares. grad is a SUM of per-record variances -- not a count, not a share
    -- so it stays a scalar and is ranked by ECDF.

    What the Dirichlet carries that a rank cannot: at 65 of 240 records covered, a
    31-vs-26 straddle count is a DIRECTION with a probability attached, not a
    verdict."""
    a = {"straddle": 26, "allfail": 61, "allwin": 0}
    b = {"straddle": 31, "allfail": 56, "allwin": 0}
    p = mod._census_beats(b, a, draws=20000)
    assert 0.70 < p < 0.88, p          # measured 0.789
    assert mod._census_beats(a, b, draws=20000) == pytest.approx(1 - p, abs=0.03)


def test_an_identical_census_is_a_coin_flip(mod):
    """No evidence of difference must read as no difference, not as a win for
    whichever arm is listed first -- the defect this whole key exists to remove."""
    c = {"straddle": 20, "allfail": 40, "allwin": 1}
    assert mod._census_beats(c, dict(c), draws=20000) == pytest.approx(0.5, abs=0.03)


def test_more_coverage_sharpens_the_same_proportion(mod):
    """THE PROPERTY A RANK CANNOT HAVE. The same 2:1 straddle advantage is far more
    certain at 300 records than at 30, and the Dirichlet says so."""
    thin_a, thin_b = {"straddle": 4, "allfail": 8, "allwin": 0}, {"straddle": 8, "allfail": 4, "allwin": 0}
    fat_a, fat_b = {"straddle": 40, "allfail": 80, "allwin": 0}, {"straddle": 80, "allfail": 40, "allwin": 0}
    assert mod._census_beats(fat_b, fat_a, draws=20000) > \
        mod._census_beats(thin_b, thin_a, draws=20000)


def test_an_unvisited_arm_claims_nothing(mod):
    """Article X: absence of evidence is not a result. An arm with no census must
    not win OR lose the tie-break on the strength of having measured nothing."""
    assert mod._census_beats({}, {"straddle": 30, "allfail": 30, "allwin": 0}) == 0.5


def test_hpo_writes_a_selection_receipt(src):
    """Operator 2026-09-19: "our hpo produces this file at the end ... we will need
    to ensure this formula works w real data".

    A ranking nobody can re-derive is a ranking nobody can check. The receipt holds
    every number the winner was chosen on, so the formula can be re-scored WITHOUT
    re-running a trial -- which is exactly how it was validated against the
    2026-09-19 bracket (grad ECDF 0.250/0.750, census P 0.211/0.790)."""
    assert 'data/%s_selection.json' % '' in src.replace('"%s_selection', '"_selection') \
        or "_selection.json" in src
    for field in ("mean_curve", "grad_mean", "straddle_mean", "allfail_mean",
                  "census_p", "signal_ecdf"):
        assert '"%s"' % field in src, "the receipt must carry %r" % field
    assert "selection receipt FAILED" in src, \
        "a receipt must never kill a bracket that already spent the GPU"


# ---------------------------------- no KL, per-token log probs ----------
def test_rl_cmd_passes_beta_and_logprob_norm_through(mod, a):
    """FIFTH PASS-THROUGH GAP. --quantile-sampling, --carry-visits, --adv-flat and
    --quality-bands each existed only on the trainer and were silently OFF for
    every run routed through this runner. --beta and --logprob-norm were the next
    two: without them an arm takes the trainer defaults beta=0.3 (KL ON) and
    logprob_norm="sum" (length-scaled), which is the opposite of what was asked.

    Operator 2026-09-20: "I want to be sure were not doing kl divergence / And /
    Were token token level log probs not sequence level"."""
    cmd = mod._rl_cmd({"lr": 1e-5, "rollouts": 4, "grad_clip": 1.0, "is_beta": 0.0},
                      a, "rl_hpo_0")
    assert "--beta" in cmd, "no KL pass-through -> arms silently run beta=0.3"
    assert float(cmd[cmd.index("--beta") + 1]) == 0.0
    assert "--logprob-norm" in cmd, "no norm pass-through -> arms silently sum"
    assert cmd[cmd.index("--logprob-norm") + 1] == "mean"


def test_rl_cmd_can_still_ask_for_kl_and_sum(mod, a):
    """The defaults changed; the capability did not. An explicit request for the
    old regime must still reach the trainer, or the board's earlier numbers become
    unreproducible."""
    a.beta = 0.3
    a.logprob_norm = "sum"
    cmd = mod._rl_cmd({"lr": 1e-5, "rollouts": 4, "grad_clip": 1.0, "is_beta": 0.0},
                      a, "rl_hpo_0")
    assert float(cmd[cmd.index("--beta") + 1]) == 0.3
    assert cmd[cmd.index("--logprob-norm") + 1] == "sum"


# ---------------------------------- the staged dimension-rotating search ----
def test_temperature_reaches_the_trainer_when_an_arm_sets_it(mod, a):
    """SIXTH PASS-THROUGH GAP. --temperature exists on the trainer and was never
    forwarded, so a phase sweeping temperature would have swept a DEAD AXIS --
    every arm pinned to the trainer default, every cell agreeing, which
    rules/085-partition-variance.md says to detect before spending the grid."""
    cmd = mod._rl_cmd({"lr": 1e-5, "rollouts": 4, "grad_clip": 1.0,
                       "is_beta": 0.0, "temperature": 1.0}, a, "rl_hpo_0")
    assert "--temperature" in cmd
    assert float(cmd[cmd.index("--temperature") + 1]) == 1.0


def test_an_arm_without_temperature_does_not_pin_it(mod, a):
    """Absent means "keep the trainer default", not "send a zero"."""
    cmd = mod._rl_cmd({"lr": 1e-5, "rollouts": 4, "grad_clip": 1.0, "is_beta": 0.0},
                      a, "rl_hpo_0")
    assert "--temperature" not in cmd


def test_the_arm_grid_varies_any_axis_not_just_lr(mod):
    """Operator 2026-09-20: the phases rotate onto NEW dimensions while holding
    what the previous phase chose. rl_lr_grid pins everything but lr and cannot
    express that."""
    arms = mod.rl_arm_grid([{"name": "p2a", "rollouts": 4, "temperature": 0.8},
                            {"name": "p2b", "rollouts": 8, "temperature": 1.0}])
    assert [x["rollouts"] for x in arms] == [4, 8]
    assert [x["temperature"] for x in arms] == [0.8, 1.0]


def test_the_arm_grid_fills_unstated_axes_from_the_incumbent(mod):
    """A phase writes only the axes it varies; everything else must be held
    IDENTICAL, or the comparison is confounded (instructions.md 1.5)."""
    arms = mod.rl_arm_grid([{"name": "only_lr", "lr": 5e-5}])
    assert arms[0]["rollouts"] == 4
    assert arms[0]["grad_clip"] == 1.0
    assert arms[0]["is_beta"] == 0.0


# ------------------------- accuracy ranks RL, never gradient ------------
def _row(arm, best, curve, signal, n_eval=16):
    return {"arm": arm, "best": best, "curve": curve, "n_eval": n_eval,
            "signal_ecdf": signal, "start": 0, "config": {}}


def test_rl_ranks_on_accuracy_not_available_gradient(mod):
    """Operator 2026-09-20: "trying to maximize avail gradient for rl hpo is not
    what we want ... Just max accuracy."

    RANKING ON REMAINING GRADIENT IS INVERTED, not merely off-spec. Gradient
    DECLINES as RL consumes it, so "most grad left at step 3" rewards the arm that
    learned LEAST. Measured on the phase-1 bracket: p1a took the better accuracy
    curve (2 2 2, mean 2.00) while p1b retained more gradient (7.305 vs 7.030) by
    consuming less (5.557 vs 5.942). Accuracy must win."""
    done = [_row("p1a", 2, [2, 2, 2], 0.490), _row("p1b", 2, [2, 1, 2], 0.656)]
    assert mod._rank_leader("rl", done)["arm"] == "p1a"


def test_a_high_signal_cannot_outrank_a_better_eval(mod):
    """The full run's bar is the eval (instructions.md 1.2a). No gradient statistic
    may override it, however wide the gap."""
    done = [_row("accurate", 9, [9, 9, 9], 0.01),
            _row("gradient_rich", 2, [2, 2, 2], 0.99)]
    assert mod._rank_leader("rl", done)["arm"] == "accurate"


def test_the_curve_mean_breaks_ties_before_signal(mod):
    """max() returns the FIRST maximum on ties, which made LIST ORDER the tie-break
    -- an arm that spikes and gives the gain back could not be told from one that
    holds it."""
    done = [_row("spikes", 2, [2, 1, 1], 0.99), _row("holds", 2, [2, 2, 2], 0.01)]
    assert mod._rank_leader("rl", done)["arm"] == "holds"


def test_beta_can_be_set_per_arm(mod, a):
    """optimizer settings.txt: "beta 0.10 @ lr 1e-4: rise then crash within 4-6
    steps (three independent collapses)", with beta=0.3 pinned as the stable KL
    anchor. beta is therefore a level to TEST, and testing it requires two arms in
    one bracket -- which a bracket-wide flag cannot express."""
    cmd = mod._rl_cmd({"lr": 1.2e-5, "rollouts": 4, "grad_clip": 0.05,
                       "is_beta": 0.0, "beta": 0.3}, a, "rl_hpo_0")
    assert float(cmd[cmd.index("--beta") + 1]) == 0.3


def test_an_arm_without_beta_falls_back_to_the_bracket_flag(mod, a):
    a.beta = 0.0
    cmd = mod._rl_cmd({"lr": 1.2e-5, "rollouts": 4, "grad_clip": 0.05,
                       "is_beta": 0.0}, a, "rl_hpo_0")
    assert float(cmd[cmd.index("--beta") + 1]) == 0.0


# ------------------- temperature and top_p are SFT axes (maxent) --------
def test_sft_arms_sweep_temperature_and_top_p(mod):
    """Operator 2026-09-20: "if we have temp in rl hpo we should have it in sft
    because sft is for maxent".

    SFT is ranked on STRADDLE -- the spread it leaves RL, not its accuracy
    (rl-readiness.md). Straddle is a property of the SAMPLED distribution, so
    temperature and top_p are the most direct levers on the very quantity SFT is
    selected for. They must vary across arms, not sit pinned at a default."""
    import random
    arms = mod.PHASES["sft"]["arms"](8, random.Random(0))
    assert all("temperature" in a and "top_p" in a for a in arms)
    assert len({a["temperature"] for a in arms}) > 1, "temperature must actually vary"
    assert len({a["top_p"] for a in arms}) > 1, "top_p must actually vary"


def test_the_sft_incumbent_keeps_its_original_measurement_point(mod):
    """Arm 0 is the board's reference point; it keeps the trainer default it was
    originally ranked at, or the board's history becomes incomparable."""
    import random
    arms = mod.PHASES["sft"]["arms"](4, random.Random(0))
    assert arms[0]["name"] == "incumbent"
    assert arms[0]["temperature"] == 0.9 and arms[0]["top_p"] == 0.9


def test_sft_passes_temperature_and_top_p_to_the_straddle_probe(mod, a, sft_arm):
    """SEVENTH PASS-THROUGH GAP, the only one that corrupts a SELECTION rather than
    pinning an axis. The probe took dict(gof.DEFAULTS) verbatim, so every SFT arm
    was ranked on straddle measured at temperature 0.9 while RL samples elsewhere."""
    cmds = mod.PHASES["sft"]["cmd"](sft_arm, a, "sft_hpo_0")
    probe = [c for c in cmds if any("straddle_probe" in str(x) for x in c)][0]
    assert "--temperature" in probe
    assert float(probe[probe.index("--temperature") + 1]) == sft_arm["temperature"]
    assert "--top-p" in probe
    assert float(probe[probe.index("--top-p") + 1]) == sft_arm["top_p"]


# ============ THE PASS-THROUGH GATE ====================================
# EIGHT of these in one session, each found by a run behaving wrongly rather than
# by a test: --quantile-sampling, --carry-visits, --adv-flat, --quality-bands,
# --beta, --logprob-norm, --temperature, --gen-batch.
#
# A flag that exists only on the child tool is silently OFF for every run routed
# through this wrapper, and the log still looks right. The failure is invisible by
# construction, which is why "review it carefully" does not work and a gate does.
# Operator 2026-09-20: "this is why we have wrappers, to unify how we handle
# processes once and best everywhere."
#
# The rule this encodes: a child flag is either PASSED THROUGH or listed below as
# deliberately-not. Silence is not a decision.

def _child_flags(path):
    """Every --flag the child tool declares."""
    import re
    src = io.open(path, encoding="utf-8").read()
    return set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', src))


def _wrapper_flags(cmds):
    return {str(x) for c in cmds for x in c if str(x).startswith("--")}


# Deliberately not forwarded, each with the reason it is the wrapper's business.
_SFT_NOT_FORWARDED = {
    "--help",
    "--checkpoints",        # the wrapper composes this from the arm's own epochs
    "--arm",                # ditto
    "--out",                # wrapper owns artifact paths
    "--no-reroll",          # reroll is part of the straddle definition, never off
}


def test_the_probe_has_no_unforwarded_performance_flags(mod, a, sft_arm):
    """--gen-batch existed on the probe with a default of 8 and was never passed,
    so every bracket generated in batches of 8 while the RL lane did 32 in one pass
    on the same card. Measured: 1.4 min to TRAIN an arm, ~27 min to probe it.

    Any probe flag that is neither forwarded nor explicitly excused fails here."""
    cmds = mod.PHASES["sft"]["cmd"](sft_arm, a, "sft_hpo_0")
    probe_cmd = [c for c in cmds if any("straddle_probe" in str(x) for x in c)][0]
    declared = _child_flags("archive/tools/straddle_probe.py")
    forwarded = _wrapper_flags([probe_cmd])
    missing = declared - forwarded - _SFT_NOT_FORWARDED
    assert not missing, (
        "straddle_probe flags neither forwarded nor excused: %s. Forward them in "
        "_sft_cmd, or add them to _SFT_NOT_FORWARDED with the reason."
        % sorted(missing))


def test_gen_batch_actually_reaches_the_probe(mod, a, sft_arm):
    """The specific one that cost the session's wall-clock."""
    cmds = mod.PHASES["sft"]["cmd"](sft_arm, a, "sft_hpo_0")
    probe = [c for c in cmds if any("straddle_probe" in str(x) for x in c)][0]
    assert "--gen-batch" in probe
    assert int(probe[probe.index("--gen-batch") + 1]) >= 32


# ================= RunShape: what an epoch and a bin MEAN =================
def test_an_epoch_is_a_full_pass_at_any_pool_size(mod, a, sft_arm):
    """Operator 2026-09-20: "an epoch means full coverage for whatever sample size
    the outer is (inner is bins for quantiles)".

    --bins-per-epoch is a count of token-budgeted STEPS, so the records it covers
    depends on the pool's own length distribution. One constant gives different
    coverage on every pool, and on 2026-09-20 it was wrong in BOTH directions:

        237-row pool, 16 bins -> 89 records/epoch = 38% of a pass
         16-row pool, 13 bins -> the whole pool inside epoch 1, then EXHAUSTED

    0 hands the sizing to the trainer, which knows the pool it loaded."""
    cmds = mod.PHASES["sft"]["cmd"](sft_arm, a, "sft_full")
    train = [c for c in cmds if any("vibe_sft" in str(x) for x in c)][0]
    assert train[train.index("--bins-per-epoch") + 1] == "0"


def test_the_screen_does_not_quantile_bin(mod, a, sft_arm):
    """16 rows across 10 length blocks is 1-2 rows per block -- a quantile of one.
    The screen runs a single block."""
    cmds = mod.PHASES["sft"]["cmd"](sft_arm, a, "sft_hpo_0")
    train = [c for c in cmds if any("vibe_sft" in str(x) for x in c)][0]
    assert train[train.index("--length-blocks") + 1] == "1"


def test_the_full_run_keeps_quantile_binning(mod, a, sft_arm):
    """237 rows over 10 blocks is ~24 per block -- genuinely length-alike, and the
    padding saving is real (measured 63% fewer slots)."""
    cmds = mod.PHASES["sft"]["cmd"](sft_arm, a, "sft_full")
    train = [c for c in cmds if any("vibe_sft" in str(x) for x in c)][0]
    assert train[train.index("--length-blocks") + 1] == str(a.length_blocks)


def test_rl_screen_also_skips_quantile_binning(mod, a):
    """Operator: "regardless if sft or rl". RL's screen is 4 records per step."""
    a.quantile_sampling = True
    hpo = mod._rl_cmd({"lr": 1e-5, "rollouts": 4, "grad_clip": 1.0, "is_beta": 0.0},
                      a, "rl_hpo_0")
    full = mod._rl_cmd({"lr": 1e-5, "rollouts": 4, "grad_clip": 1.0, "is_beta": 0.0},
                       a, "rl_full")
    assert "--quantile-sampling" not in hpo
    assert "--quantile-sampling" in full


def test_run_shape_reads_the_tag(mod):
    assert mod.RunShape("sft", "sft_hpo_3").is_hpo is True
    assert mod.RunShape("sft", "sft_full").is_hpo is False
    assert mod.RunShape("rl", "rl_hpo_0").is_hpo is True


def test_the_winner_config_carries_its_run_shape(mod, a):
    """Operator 2026-09-20: "hpo should emit w config as a product of this wrapped
    class".

    A receipt that records only hyperparameters cannot reproduce either run: the
    screen and the full run are different SHAPES (different pool, different binning).
    That gap is what let 247ab28's pool narrowing go unseen -- the receipts recorded
    lr and betas while the thing that changed was the data."""
    done = [{"arm": "a", "best": 2, "curve": [2, 2], "n_eval": 16, "start": 1,
             "config": {}, "signal_ecdf": 0.5}]
    lead = mod._rank_leader("sft", done)
    assert lead["arm"] == "a"
    # the shape is attached by hpo(); assert RunShape exposes what it needs
    sh = mod.RunShape("sft", "sft_hpo_0")
    assert sh.is_hpo and sh.length_blocks == 1 and sh.bins_per_epoch == 0
    assert "no quantile binning" in sh.describe()


# ============ the deposit is chosen on BOTH columns ======================
def _write(tmp_path, straddle, evals):
    import json as _j
    sp = tmp_path / "s.json"; ep = tmp_path / "e.json"
    sp.write_text(_j.dumps(straddle), encoding="utf-8")
    ep.write_text(_j.dumps({"summary": evals}), encoding="utf-8")
    return str(sp), str(ep)


def test_the_deposit_needs_accuracy_not_just_spread(mod, tmp_path):
    """Operator 2026-09-20: "the signal is a function of eval and straddle".

    Straddle alone has no accuracy floor: a model barely moved from base makes many
    0 < correct < K groups and scores well on spread while giving RL nothing to build
    on. MEASURED on the 2026-09-20 full run -- the maxent criterion preferred a 1/16
    checkpoint over a 3/16 one to gain 0.2 grad."""
    sp, ep = _write(
        tmp_path,
        [{"checkpoint": "e1", "straddle": 12, "records": 24, "available_gradient": 2.17},
         {"checkpoint": "e2", "straddle": 13, "records": 24, "available_gradient": 2.38}],
        {"e1": {"n_obs": 16, "n_correct": 3}, "e2": {"n_obs": 16, "n_correct": 1}})
    rows = mod.pick_deposit(sp, ep)
    assert rows[0]["checkpoint"] == "e1", "more gradient must not beat 3x the accuracy"


def test_zero_on_either_column_scores_zero(mod, tmp_path):
    """The product is the floor: no material is useless, and no accuracy to build on
    is equally useless. A weighted SUM would let spread compensate for zero accuracy,
    which is the failure being fixed."""
    sp, ep = _write(
        tmp_path,
        [{"checkpoint": "no_acc", "straddle": 20, "records": 24},
         {"checkpoint": "no_spread", "straddle": 0, "records": 24}],
        {"no_acc": {"n_obs": 16, "n_correct": 0},
         "no_spread": {"n_obs": 16, "n_correct": 8}})
    rows = mod.pick_deposit(sp, ep)
    assert all(r["score"] == 0.0 for r in rows)


def test_straddle_breaks_ties_at_zero_accuracy(mod, tmp_path):
    """Two checkpoints both at 0/16 both score 0. Without a tie-break the criterion
    goes blind exactly where the base model sits."""
    sp, ep = _write(
        tmp_path,
        [{"checkpoint": "lo", "straddle": 4, "records": 24},
         {"checkpoint": "hi", "straddle": 16, "records": 24}],
        {"lo": {"n_obs": 16, "n_correct": 0}, "hi": {"n_obs": 16, "n_correct": 0}})
    assert mod.pick_deposit(sp, ep)[0]["checkpoint"] == "hi"


def test_counts_are_read_from_summary_not_arms(mod, tmp_path):
    """`arms` is a name -> checkpoint-PATH map; reading it yields strings and the
    selector dies on .get(). Both keys sound like the right one."""
    import json as _j
    sp = tmp_path / "s.json"; ep = tmp_path / "e.json"
    sp.write_text(_j.dumps([{"checkpoint": "e1", "straddle": 9, "records": 24}]),
                  encoding="utf-8")
    ep.write_text(_j.dumps({"arms": {"e1": "sft_arms/e1"},
                            "summary": {"e1": {"n_obs": 16, "n_correct": 4}}}),
                  encoding="utf-8")
    assert mod.pick_deposit(str(sp), str(ep))[0]["score"] > 0


def test_an_arm_log_is_truncated_before_the_arm_runs(mod):
    """Every bracket reuses .tmp/<phase>_hpo_<i>.log, so a shorter bracket leaves the
    previous run's files sitting there looking current.

    MEASURED 2026-09-20, three times in one session: while arm 0 of a KL factorial
    was still running, .tmp/rl_hpo_1.log held a 49-minute-old TEMPERATURE arm whose
    numbers were read as a live result. They matched the earlier bracket exactly --
    because they WERE the earlier bracket. Nothing in the file says which experiment
    wrote it; only the mtime tells, and only if you think to look.

    An unstarted arm must read EMPTY, not plausibly stale: that is the difference
    between NOT TESTED and a wrong finding."""
    import inspect
    src = inspect.getsource(mod.run_one)
    i_open = src.index('io.open(log, "w"')
    i_cmd = src.index('PHASES[phase]["cmd"]')
    assert i_open < i_cmd, "truncate BEFORE building/dispatching the command"
