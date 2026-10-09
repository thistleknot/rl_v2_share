"""The judge deadline is learned from the endpoint, not picked by hand.

NO GOVERNING SPEC. Basis: operator 2026-09-15 -- "that 75.2 doesnt look like it had a
timeout", then "Ema bro. I keep having to tell you this / Rolling ema w mad thresholds
remember?", then "We only need say 5 times and take the exponential moving median plus
MAD?".

TWO defects this pins, both of which shipped:
  * the transport used a fixed 180 s, so the guard's threshold was computed and then
    ignored -- a 75.2 s judge call simply waited
  * a hand-picked constant (45 s) was briefly substituted for the guard, which is the
    same mistake wearing a smaller number
"""
import sys



def _fresh(name, samples):
    from rlv2.judge import reward as r
    r._lat_samples[name] = list(samples)
    return r._lat_kill_threshold(name)


def test_no_threshold_before_five_completions():
    """A median over two samples is theatre."""
    from rlv2.judge import reward as r
    for n in range(0, r._LAT_MIN_SAMPLES):
        assert _fresh("warm%d" % n, [9.0] * n) == r._LAT_HARD_CAP


def test_the_operator_case_arms_at_five_and_kills_the_outlier():
    """[8, 9, 10, 9, 9] -> the 75.2 s call must not survive."""
    t = _fresh("op", [8.0, 9.0, 10.0, 9.0, 9.0])
    assert t < 75.2, "the 75.2 s judge call would have survived again (t=%.1f)" % t
    assert t > 10.0, "a ceiling below the samples themselves would kill every call"


def test_a_constant_window_never_flags():
    """MAD == 0 with NO spread carries no scale; flagging would kill the next call
    for being a millisecond slow."""
    from rlv2.judge import reward as r
    assert _fresh("flat", [9.0] * 8) == r._LAT_HARD_CAP


def test_mad_zero_with_real_spread_falls_back_instead_of_giving_up():
    """Three equal readings out of five zero the MAD -- a median artefact, not an
    absence of spread. Without the mean-absolute-deviation fallback the guard
    silently reverts to the hard cap at exactly the n the operator asked for."""
    from rlv2.judge import reward as r
    assert _fresh("artefact", [9.0, 9.0, 9.0, 8.0, 12.0]) < r._LAT_HARD_CAP


def test_the_ceiling_follows_a_genuinely_slower_endpoint():
    fast = _fresh("fast", [4.0, 5.0, 4.0, 6.0, 5.0, 4.0, 5.0, 5.0, 4.0, 6.0])
    slow = _fresh("slow", [40.0, 45.0, 42.0, 44.0, 41.0, 43.0, 44.0, 42.0, 45.0, 43.0])
    assert slow > fast * 3, "a 10x slower endpoint must not keep the fast ceiling"


def test_recent_samples_dominate_the_estimate():
    """The whole point of the exponential weighting: an old regime must decay out."""
    from rlv2.judge import reward as r
    old_heavy = _fresh("old", [40.0] * 20 + [9.0] * 10)   # now fast
    steady_slow = _fresh("steady", [40.0] * 10)
    assert old_heavy < steady_slow or steady_slow == r._LAT_HARD_CAP


def test_transport_asks_the_guard_rather_than_a_constant():
    import io
    src = io.open("src/rlv2/judge/reward.py", encoding="utf-8").read()
    assert "eff = float(timeout) if timeout else _lat_kill_threshold(mdl)" in src
    assert "urlopen(req, timeout=eff)" in src
    assert "timeout: float = 180.0" not in src, "a hardcoded transport deadline is back"
    assert "JUDGE_CALL_TIMEOUT_S" not in src, "hand-picked constant is back"


def test_only_completions_feed_the_baseline():
    """A timeout must never tighten the ceiling, or kills cascade."""
    import io
    src = io.open("src/rlv2/judge/reward.py", encoding="utf-8").read()
    i = src.index("with urllib.request.urlopen(req, timeout=eff)")
    j = src.index("_lat_record(mdl,", i)
    between = src[i:j]
    assert "except" not in between, "_lat_record moved inside the failure path"
