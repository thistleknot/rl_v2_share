"""Every generation carries a wall-clock deadline.

NO GOVERNING SPEC. Basis: operator 2026-09-15 --

> "Dude if given you my timeout rule multiple times and its on you for not applying it.
>  EVERY fucking generation has a timeout. If you see a stall. That means you had no
>  timeout!"

A STALL IS NOT AN EVENT TO DIAGNOSE -- it is the absence of a timeout, already proven.
Measured 2026-09-15: one record in a 24-record straddle probe ran 16 minutes at 100% GPU.
Nothing could interrupt it; recovery was noticing the log had stopped moving and killing
the process, which lost the whole measurement because the ledger only writes at the end.

WHY MaxTimeCriteria AND NOT A PYTHON TIMEOUT. `model.generate()` does not yield to the
interpreter between decoding steps in a way a signal or thread-timer can interrupt
portably. MaxTimeCriteria is checked BY the decoding loop, so it stops cleanly and returns
the tokens produced so far.

WHY A CUT GENERATION IS NOT A LOST SAMPLE. It has no stop token, so the EOS gate already
forces its reward to 0.0 -- the operator's own rule. A timeout is therefore a VERIFIABLE
failure, scored mechanically, not a hole in the data.
"""
import importlib.util
import inspect
import io
import sys

import pytest


@pytest.fixture(scope="module")
def gof():
    spec = importlib.util.spec_from_file_location("gof_to", "docs/grpo_one_file.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["gof_to"] = m
    saved, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(m)
    sys.argv = saved
    return m


def test_the_deadline_is_on_by_default(gof):
    """0 is what produced the 16-minute record. The default must protect a run
    that was launched without thinking about it."""
    assert gof.DEFAULTS["gen_timeout_s"] > 0
    assert gof.DEFAULTS["gen_timeout_s"] <= 300, "a deadline nobody hits is not one"


def test_generate_is_given_a_stopping_criterion(gof):
    src = inspect.getsource(gof.sample_response)
    assert "MaxTimeCriteria" in src
    assert "stopping_criteria=_crit" in src, \
        "building the criterion and not passing it is the same as having none"


def test_the_criterion_is_derived_from_the_config_not_a_literal(gof):
    """It reads the config through generation_deadline() rather than inlining a
    number -- which is also what makes the deadline per-record instead of flat."""
    src = inspect.getsource(gof.sample_response)
    assert "generation_deadline(_max_new, cfg)" in src
    assert 'cfg.get("gen_timeout_s")' in inspect.getsource(gof.generation_deadline)


def test_zero_disables_rather_than_setting_a_zero_deadline(gof):
    """A 0 that meant 'stop immediately' would silently zero every rollout."""
    src = inspect.getsource(gof.sample_response)
    assert "if _budget_s > 0:" in src


def test_a_breach_is_announced(gof):
    """A deadline that fires silently looks exactly like a short answer."""
    src = inspect.getsource(gof.sample_response)
    assert "[gen-timeout]" in src


def test_the_flag_exists_and_says_what_it_protects():
    src = io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    assert '"--gen-timeout-s"' in src
    i = src.index('"--gen-timeout-s"')
    assert "EOS gate" in src[i:i + 600], \
        "the help must say a cut generation is scored 0, not silently dropped"


def test_the_config_actually_carries_it_to_the_generator():
    """The flag existing and the cfg key existing are different things -- the
    generator reads cfg, not args."""
    src = io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    assert "gen_timeout_s=args.gen_timeout_s," in src


def test_a_cut_generation_is_scored_zero_by_the_existing_gate():
    """The timeout leans on the EOS gate rather than adding a second rule: no stop
    token -> reward 0. If that gate ever moves, this guarantee moves with it."""
    src = io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    assert "rewards_list = [r if ok else 0.0 for r, ok in zip(rewards_list, finished)]" in src


def test_the_module_can_actually_call_time(gof):
    """The timeout shipped with `time.time()` and no `import time`, so the FIRST
    rollout of the first arm raised NameError -- two arms NOT TESTED before it was
    caught. Importing the module is not enough; the name has to resolve."""
    assert hasattr(gof, "time"), "sample_response calls time.time()"
    assert callable(gof.time.time)


# ------------------------------------- the deadline is derived, not guessed ----
def test_the_deadline_scales_with_the_records_token_budget(gof):
    """Operator 2026-09-15: "that timeout is based on a rolling ema of q+ a lens of
    the quantile yes?" -- it was a flat 90s, which is wrong the same way a flat token
    budget was wrong. A record granted 700 tokens by its quantile legitimately needs
    more wall clock than one granted 50."""
    cfg = dict(gof.DEFAULTS)
    for _ in range(6):
        gof.observe_generation_rate(300, 6.0)     # settle EMA at 50 tok/s
    assert gof.generation_deadline(700, cfg) > gof.generation_deadline(400, cfg)
    assert gof.generation_deadline(400, cfg) > gof.generation_deadline(50, cfg)


def test_the_rate_is_an_ema_of_observed_generations(gof):
    """A constant cannot track a shared GPU or a thermal throttle; a fixed deadline
    then either fires on healthy work or never fires at all."""
    gof._GEN_RATE.update({"tps": 12.0, "n": 0})
    gof.observe_generation_rate(500, 10.0)        # first real sample replaces the seed
    assert abs(gof._GEN_RATE["tps"] - 50.0) < 1e-6
    gof.observe_generation_rate(100, 10.0)        # a slow one pulls it down, partially
    assert 10.0 < gof._GEN_RATE["tps"] < 50.0


def test_an_implausible_sample_is_ignored(gof):
    """A 3-token generation in 0.01s is cache warm-up, not a measurement of rate."""
    gof._GEN_RATE.update({"tps": 40.0, "n": 3})
    gof.observe_generation_rate(3, 0.01)
    assert gof._GEN_RATE["tps"] == 40.0


def test_there_is_a_floor_so_a_short_record_is_not_cut_cold(gof):
    """At 50 tok/s a 50-token record is ~1s of work. Without a floor the deadline
    would be ~3s, and a cold cache would breach it on healthy work."""
    cfg = dict(gof.DEFAULTS)
    for _ in range(6):
        gof.observe_generation_rate(300, 6.0)
    assert gof.generation_deadline(50, cfg) >= 15.0


def test_the_flag_is_a_cap_not_the_deadline(gof):
    """If the EMA spiked, a purely rate-derived deadline would collapse toward zero
    and cut every record before it said anything. The cap bounds the tail; the floor
    bounds the head."""
    cfg = dict(gof.DEFAULTS)
    assert gof.generation_deadline(10 ** 6, cfg) == cfg["gen_timeout_s"]


def test_zero_still_disables_everything(gof):
    assert gof.generation_deadline(500, dict(gof.DEFAULTS, gen_timeout_s=0)) == 0.0


def test_the_expected_length_is_a_boxcox_band_not_a_fixed_log(gof):
    """Operator 2026-09-15: "Literally look up MAD and box-cox".

    The incumbent is src/rlv2/data/cpt_mix.py:boxcox_length_band -- median +/-
    sigmas*MAD*1.4826 in BOX-COX space with lambda FITTED. I used log_mad_band
    (lambda assumed 0) and then did not use a band at all for the deadline, taking
    raw max_new_tokens. Length distributions are right-skewed; the fitted power is
    the difference between a band that contains the bulk and one that does not.
    """
    src = inspect.getsource(gof.boxcox_expected_tokens)
    assert "boxcox_length_band" in src, "must reuse the incumbent, not re-derive"
    # code only -- the docstring names log_mad_band to explain what was wrong
    code = src[src.index('"""', src.index('"""') + 3) + 3:]
    assert "log_mad_band" not in code
    assert "deadline_tokens" in inspect.getsource(gof.generation_deadline)


def test_a_pool_too_small_to_fit_falls_back_rather_than_crashing(gof):
    class _Tok:
        def __call__(self, text, add_special_tokens=False):
            class _E:
                input_ids = [0] * max(1, len(text.split()))
            return _E()
    assert gof.boxcox_expected_tokens([{"question": "a b", "gold": "c"}], _Tok()) == 0


def test_the_band_is_used_only_as_a_floor_on_the_hard_ceiling(gof):
    """max_new_tokens is still a hard cap: if a record is granted MORE than the
    band, the deadline must cover the grant, or a legitimately long record is cut."""
    cfg = dict(gof.DEFAULTS, deadline_tokens=216)
    for _ in range(6):
        gof.observe_generation_rate(300, 6.0)
    assert gof.generation_deadline(2000, cfg) > gof.generation_deadline(60, cfg)
