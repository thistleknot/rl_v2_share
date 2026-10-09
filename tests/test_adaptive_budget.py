"""The generation budget adapts to what the model actually finishes.

NO GOVERNING SPEC. Basis: operator 2026-09-19 --

> "generations should start with a known prior and adjust dynamically based on
>  box-cox prior generated lens + mad *4 (tukeys fence) ... This way we always have
>  the boundary accounted for giving the model plenty of room to think"

WHY IT HAD TO CHANGE, measured on the live run before the change:

    eos-gate fired on 136 of 344 rollouts = 39.5%

Two of every five generations were scored 0 for running out of budget rather than
for being wrong. A group whose answers are ALL truncated reads as all-wrong, is
filed dead, and contributes zero available gradient -- so a static budget derived
from the GOLDS was corrupting the quantity the sampler ranks on, and was punishing
exactly the behaviour the campaign wants (longer deliberation).
"""
import importlib.util
import sys

import pytest


@pytest.fixture(scope="module")
def gof():
    spec = importlib.util.spec_from_file_location("gof_ab", "docs/grpo_one_file.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["gof_ab"] = m
    saved, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(m)
    sys.argv = saved
    return m


class _Tok:
    """Word-count tokenizer -- enough to size a budget."""

    def __call__(self, text, add_special_tokens=True, **kw):
        n = len((text or "").split())
        return type("E", (), {"input_ids": [1] * n})()


def _cfg(gof, **kw):
    c = dict(gof.DEFAULTS)
    c.update(length_slack=10, min_completion=4, max_completion=1000, num_rollouts=4)
    c.update(kw)
    return c


def test_a_short_sample_cannot_fit_a_transform_and_returns_none(gof):
    """Fitting Box-Cox to three points is fitting noise; the prior must hold."""
    assert gof._boxcox_fence([100, 120, 140]) is None
    assert gof._boxcox_fence([]) is None


def test_the_fence_sits_above_every_ordinary_observation(gof):
    """The point of a fence is headroom: a typical answer must never land on it."""
    lens = [90, 95, 100, 105, 110, 98, 102, 97, 103, 99, 101, 96]
    cap = gof._boxcox_fence(lens, mads=4.0)
    assert cap is not None
    assert cap > max(lens), "a fence below the longest seen answer gives no room"


def test_a_wider_fence_gives_more_room(gof):
    """MADs is the knob the operator named (4, not Tukey's usual 1.5)."""
    lens = [90, 95, 100, 105, 110, 98, 102, 97, 103, 99, 101, 96]
    assert gof._boxcox_fence(lens, mads=6.0) > gof._boxcox_fence(lens, mads=2.0)


def test_the_cap_never_falls_below_an_observed_FINISHED_length(gof):
    """Every observation is a generation that FINISHED, so an observed 400 proves
    400 tokens were needed AND achievable. The Tukey fence alone puts this sample
    at 149 -- correctly, 200 and 400 are outliers against a 50-80 cluster -- but
    budgeting under an answer the model already completed guarantees truncating it
    next time, which is the failure the whole mechanism exists to stop."""
    skewed = [50, 52, 55, 58, 60, 62, 65, 70, 80, 120, 200, 400]
    cap = gof._boxcox_fence(skewed, mads=4.0)
    assert cap is not None
    assert cap >= 400, "a finished 400-token answer must remain producible"


def test_the_fence_still_bounds_a_tight_sample(gof):
    """The floor at max(observed) must not make the fence irrelevant: on a tight
    cluster the fence is what supplies headroom ABOVE anything seen so far."""
    tight = [100, 102, 98, 101, 99, 103, 97, 100, 102, 98, 101, 99]
    assert gof._boxcox_fence(tight, mads=4.0) > max(tight)


def test_the_budget_is_the_gold_prior_when_nothing_has_been_observed(gof):
    """Step one has no evidence, so the gold-derived prior stands unchanged."""
    tok = _Tok()
    rec = {"gold": "one two three four five", "question": "q"}
    assert gof.answer_budget(rec, tok, _cfg(gof)) == 15      # 5 gold + 10 slack


def test_observed_lengths_can_only_RAISE_the_budget(gof):
    """The adaptive term is headroom, never a squeeze. If the model finishes short,
    the gold prior still governs -- shrinking the budget on short answers would
    make a later long answer impossible to produce."""
    tok = _Tok()
    rec = {"gold": "one two three four five", "question": "q"}
    short = _cfg(gof, _observed_lens=[3, 4, 3, 5, 4, 3, 4, 5, 3, 4, 4, 3])
    assert gof.answer_budget(rec, tok, short) >= 15


def test_a_long_observed_sample_opens_the_budget_up(gof):
    """THE REPAIR. 39.5% of rollouts were being truncated; once the model's own
    finished lengths are in evidence the budget has to follow them."""
    tok = _Tok()
    rec = {"gold": "one two three four five", "question": "q"}
    base = gof.answer_budget(rec, tok, _cfg(gof))
    longs = [180, 190, 200, 210, 195, 205, 188, 202, 199, 191, 207, 198]
    grown = gof.answer_budget(rec, tok, _cfg(gof, _observed_lens=longs))
    assert grown > base * 2, (base, grown)


def test_the_ceiling_is_still_a_hard_cap(gof):
    """max_completion bounds the adaptive term -- an unbounded fence would let one
    rambling phase inflate every later budget."""
    tok = _Tok()
    rec = {"gold": "one two three", "question": "q"}
    huge = [5000] * 20
    c = _cfg(gof, _observed_lens=huge, max_completion=160)
    assert gof.answer_budget(rec, tok, c) == 160


def test_the_prior_yields_gradually_rather_than_at_the_first_observation(gof):
    """alpha = n / (n + w): twelve observations must not overwrite the prior the
    way two hundred do, or one unlucky early step sets the budget for the run."""
    tok = _Tok()
    rec = {"gold": "one two three four five", "question": "q"}
    longs = [200] * 12
    few = gof.answer_budget(rec, tok, _cfg(gof, _observed_lens=longs))
    many = gof.answer_budget(rec, tok, _cfg(gof, _observed_lens=[200] * 300))
    assert many > few, (few, many)


def test_only_finished_generations_are_collected(gof):
    """THE FEEDBACK LOOP THIS CLOSES. A truncated answer measures the BUDGET, not
    how long an answer wants to be. Learning from truncations means: model rambles
    -> budget grows -> rambling is never punished -> length inflates without bound,
    and the EOS gate is the only thing bounding length."""
    import inspect
    src = inspect.getsource(gof.compute_grpo_loss)
    assert '"finished_tok": [len(t[0]) - t[1] for t, ok in zip(token_sets, finished) if ok]' in src


# ------------------------------------------------- the batch-std EMA ----
def _ema(prev_var, step_var, n, w=192.0):
    """Mirror of the EMA in the trainer's step loop."""
    if prev_var is None:
        return step_var
    a = n / (n + w)
    return (1.0 - a) * prev_var + a * step_var


def test_the_batch_std_ema_damps_the_sampling_jitter(gof):
    """Operator 2026-09-19: "having a proper ema sdev based on steps and total number
    of calls ... gives us the best quality score."

    MEASURED, 7 steps of per-step batch-std: 0.3903 0.2864 0.2795 0.3903 0.3750
    0.3248 0.3046 -- mean 0.3358, a 33% swing. At n=16 a variance estimate carries
    18% relative error on the std, so that swing is what a PERFECTLY STATIONARY
    distribution looks like measured seven times. Per-step normalisation was
    injecting jitter, not tracking drift."""
    import statistics as st
    obs = [0.3903, 0.2864, 0.2795, 0.3903, 0.3750, 0.3248, 0.3046]
    raw_sd = st.pstdev(obs)
    var, smoothed = None, []
    for v in obs:
        var = _ema(var, v * v, 64)
        smoothed.append(var ** 0.5)
    assert st.pstdev(smoothed) < raw_sd / 2, (raw_sd, st.pstdev(smoothed))
    assert abs(st.mean(smoothed) - st.mean(obs)) < 0.05, "it must not shift the level"


def test_a_bigger_step_moves_the_ema_more(gof):
    """MLE weighting, not a flat decay: alpha = n/(n+w), so a step carrying 64
    rollouts moves the estimate four times as far as one carrying 16."""
    base = 0.09
    small = _ema(base, 0.25, 16)
    big = _ema(base, 0.25, 64)
    assert big > small > base


def test_the_first_step_has_nothing_to_average_with(gof):
    """No prior estimate means the step IS the estimate -- never a blend with zero,
    which would halve the first step's advantages for no reason."""
    assert _ema(None, 0.25, 64) == 0.25


def test_the_ema_is_wired_into_the_trainer(gof):
    import inspect
    src = inspect.getsource(gof)
    assert '_batch_var' in src, "the EMA needs its own state, separate from the std"
    assert 'std_prior_weight' in src
    assert 'EMA over %d rollouts' in src, "the log must show step vs smoothed"


# ------------------------------------------- the wave-1 size predictor ----
def _want(need, accum, wave, live_rate):
    """Mirror of the wave sizing in the trainer's draw loop."""
    naive = max(8, min(need * 2, need + 16))
    if wave == 1 and live_rate:
        w = min(int(accum / max(0.02, live_rate) * 1.2), naive * 4)
        return max(naive, w)
    return naive


def test_wave_one_is_sized_from_the_measured_live_rate(gof):
    """Operator 2026-09-19: "If we know start takes 9 waves. Can we batch 9 calls in
    one go? Or is that extremely wasteful? I dont think so as thats the number we
    iterated over."

    It is not wasteful: every candidate is generated and judged EITHER WAY, and waves
    decide only how many round trips carry them. Measured on step 1: 98 candidates
    across 7 waves -- the same 98 in one wave is 1 round trip."""
    # 16 slots at the measured 16% live rate -> ~120 candidates, not 32
    assert _want(16, 16, 1, 0.163) > 100
    # the naive draw would have opened at 32
    assert _want(16, 16, 1, None) == 32


def test_later_waves_keep_the_small_top_up_shape(gof):
    """Only wave 1 is predicted. Later waves are top-ups where the remaining need is
    small and KNOWN, so the old shape is right and a big draw would be pure
    overshoot."""
    assert _want(2, 16, 4, 0.163) == 8       # floor, not 120
    assert _want(6, 16, 3, 0.163) == 12


def test_a_bad_estimate_cannot_run_away(gof):
    """Capped at 4x the naive draw: a live rate of 2% would otherwise ask for 960
    candidates and spend an entire step's judge budget on one wave."""
    assert _want(16, 16, 1, 0.001) == 32 * 4


def test_a_richer_pool_asks_for_less(gof):
    """The predictor must fall as the pool gets easier, or it just becomes a bigger
    constant."""
    assert _want(16, 16, 1, 0.50) < _want(16, 16, 1, 0.16)


def test_the_predictor_is_wired_and_fed(gof):
    import inspect
    src = inspect.getsource(gof)
    assert '_live_rate' in src
    assert '0.5 * float(_prev_rate) + 0.5 * _rate' in src, \
        "the rate must smooth across steps, not jump on one unlucky draw"


# --------------------------------------- wave sizing from the proportion ----
def _want_wilson(need, seen, live, naive, z=1.0):
    """Mirror of the remainder sizing in the trainer's draw loop."""
    import math
    if seen < 8 or live <= 0:
        return naive
    p = live / float(seen)
    d = 1.0 + z * z / seen
    c = p + z * z / (2.0 * seen)
    m = z * math.sqrt(p * (1 - p) / seen + z * z / (4.0 * seen * seen))
    p_lo = max(0.02, (c - m) / d)
    return max(naive, min(int(need / p_lo), naive * 8))


def test_the_remainder_is_sized_from_the_LOWER_bound_not_the_point_estimate(gof):
    """Operator 2026-09-19: "As soon as we derive the first ratio of what's missing
    in the first 16 we should use standard error of a proportion to select the
    remainder."

    The old shape sized every wave to the point estimate, min(need*2, need+16),
    which undershoots half the time BY CONSTRUCTION -- that is why the waves halved
    32 22 12 8 8 and why step 1 took nine of them."""
    # wave 1 saw 5 live of 32; 11 slots left
    want = _want_wilson(need=11, seen=32, live=5, naive=22)
    point = int(11 / (5 / 32.0))          # what the point estimate would ask for
    assert want > point, (want, point)
    assert want > 100, want               # one wave, not eight


def test_a_richer_pool_asks_for_fewer(gof):
    """It must fall as the live rate rises, or it is just a bigger constant."""
    lean = _want_wilson(need=8, seen=32, live=4, naive=16)
    rich = _want_wilson(need=8, seen=32, live=20, naive=16)
    assert rich < lean, (rich, lean)


def test_an_unlucky_first_wave_cannot_demand_the_whole_pool(gof):
    """p_lo is floored at 0.02 and the draw at 8x naive: a wave 1 that found one
    live record must not ask for a thousand candidates."""
    assert _want_wilson(need=15, seen=32, live=1, naive=16) <= 16 * 8


def test_too_little_evidence_falls_back_to_the_naive_draw(gof):
    """Under 8 observations a proportion has no usable interval -- Wilson on 2 draws
    is wider than the range it is trying to bound."""
    assert _want_wilson(need=10, seen=4, live=1, naive=20) == 20
    assert _want_wilson(need=10, seen=32, live=0, naive=20) == 20


def test_the_wilson_sizing_is_wired(gof):
    import inspect
    src = inspect.getsource(gof)
    assert "_p_lo" in src and "math.sqrt(_p * (1 - _p) / _seen" in src


# ------------------------------------------ the persisted wave prior ----
def test_the_prior_is_dimensionless_across_step_sizes(gof, tmp_path):
    """THE DEFECT SIX-HATS SURFACED, 2026-09-20. The prior stored ABSOLUTE candidate
    counts in one global file keyed by nothing, while RL's screen draws 4 records per
    step and its full run draws 16 -- so a screen reading a full-run prior over-draws
    4x, and the reverse under-draws.

    Storing candidates PER SLOT makes the two comparable: a 16-slot run that needed
    96 candidates and a 4-slot run that needed 24 both describe 6 per slot."""
    f = str(tmp_path / "p.json")
    gof._save_wave_prior([96, 96, 96, 96], path=f, slots=16)
    assert gof._load_wave_prior(path=f, slots=16)[0] > 0
    big = gof._load_wave_prior(path=f, slots=16)
    small = gof._load_wave_prior(path=f, slots=4)
    assert min(big) >= 4 * min(small) - 4, "a 16-slot read is ~4x a 4-slot read"


def test_a_legacy_prior_is_refused_not_misread(gof, tmp_path):
    """A pre-normalisation file holds absolute counts and no slot count, so its units
    are unknowable. Refusing it opens wave 1 at the naive draw; trusting it mis-scales
    by up to 4x with no symptom."""
    import json as _j, io as _io
    f = str(tmp_path / "old.json")
    _io.open(f, "w", encoding="utf-8").write(_j.dumps(
        {"candidates": [100, 120], "median": 110, "mad": 10, "n": 2}))
    assert gof._load_wave_prior(path=f, slots=16) == []


def test_the_wave_prior_round_trips_and_pads(gof, tmp_path):
    """Operator 2026-09-19: "track an rl full config on disk that has the constant
    initial wave number so we know how much to retrieve on first pass ... id argue
    pad with 1 (+1, better yet based on stats)".

    The padding is the observed MAD, not a flat +1: the stored counts are what the
    fence fits, so a run whose counts were undershoots would teach the next run to
    undershoot. Adding the MAD biases the prior up by exactly the spread it saw."""
    f = str(tmp_path / "prior.json")
    counts = [98, 78, 80, 66, 66]
    gof._save_wave_prior(counts, path=f, slots=1)
    got = gof._load_wave_prior(path=f, slots=1)
    assert len(got) == len(counts)
    assert min(got) > min(counts), "every stored count is padded upward"
    import json as _j, io as _io
    d = _j.loads(_io.open(f, encoding="utf-8").read())
    assert d["median_per_slot"] == 78 and d["mad"] == 12


def test_a_missing_prior_is_not_fatal(gof, tmp_path):
    """Article X: no prior means open at the naive draw, exactly as before this
    existed -- never a crash and never a zero."""
    assert gof._load_wave_prior(path=str(tmp_path / "nope.json"), slots=16) == []


def test_one_count_is_not_a_prior(gof, tmp_path):
    """A single step cannot give a median and a MAD; refuse rather than persist a
    number the next run would trust."""
    f = str(tmp_path / "p.json")
    gof._save_wave_prior([98], path=f, slots=1)
    assert gof._load_wave_prior(path=f, slots=1) == []


def test_the_prior_is_saved_every_step_not_at_the_end(gof):
    """Runs get killed -- today, five times. A prior written only at exit learns
    nothing from any of them."""
    import inspect
    src = inspect.getsource(gof)
    i = src.index("_ch.append(_draws_this_step)")
    assert "_save_wave_prior(_ch, slots=" in src[i:i + 600],         "the prior must be saved WITH its slot count, or its units are unknowable"
