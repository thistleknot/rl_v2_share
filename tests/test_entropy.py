"""Token entropy -- the direct measure of what straddle only proxies.

NO GOVERNING SPEC. Basis: operator 2026-09-20 --

> "We need to plot entropy (logprobs) since 'measure of uncertainty in a prob distro'"

WHY IT MATTERS HERE. This campaign selects on SPREAD. Straddle (0 < correct < K)
measures it discretely at K-resolution: with K=4 a record can only report 0, 1/4,
2/4, 3/4 or 1, and a whole probe run is needed to estimate it. Token entropy

    H = -SUM_v p_v log p_v

reads the same uncertainty off the distribution directly, per token, with no
rollouts at all. The function that computes the policy's log-probs has always been
docstring'd "sequence_logprob_and_entropy, the LOG-PROB HALF" -- the entropy half
existed in the reference implementation and was simply never kept.

Reference bands (reasoning-ch07): H > 1.5 exploring, H < 0.5 converging/collapsing.
"""
import importlib.util
import math
import sys

import pytest


@pytest.fixture(scope="module")
def gof():
    spec = importlib.util.spec_from_file_location("gof_H", "docs/grpo_one_file.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["gof_H"] = m
    saved, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(m)
    sys.argv = saved
    return m


class _FakeModel:
    """Returns fixed logits so entropy has a known closed form."""

    def __init__(self, logits):
        self._logits = logits

    def __call__(self, ids):
        import torch
        return type("O", (), {"logits": self._logits.unsqueeze(0)})()


def _run(gof, logits, prompt_len=1):
    import torch
    cfg = {"logprob_norm": "mean"}
    ids = torch.zeros(logits.shape[0], dtype=torch.long)
    gof.sequence_logprob(_FakeModel(logits), ids, prompt_len, cfg)
    return cfg["_last_entropy"]


def test_a_uniform_distribution_has_maximum_entropy(gof):
    """Uniform over V outcomes is exactly log(V) nats -- the closed form that makes
    this testable rather than merely plausible."""
    import torch
    V = 8
    H = _run(gof, torch.zeros(4, V))          # all logits equal -> uniform
    assert H == pytest.approx(math.log(V), abs=1e-4)


def test_a_peaked_distribution_has_near_zero_entropy(gof):
    """A policy that has collapsed onto one token carries no uncertainty, which is
    the < 0.5 'converging/collapsing' band."""
    import torch
    logits = torch.full((4, 8), -50.0)
    logits[:, 3] = 50.0
    assert _run(gof, logits) == pytest.approx(0.0, abs=1e-4)


def test_entropy_ranks_between_those_extremes(gof):
    """Sharpening the SAME distribution must lower entropy monotonically -- the
    ordering is what a plot reads, so it is the property that has to hold."""
    import torch
    base = torch.tensor([[2.0, 1.0, 0.0, -1.0]] * 4)
    hs = [_run(gof, base * k) for k in (0.5, 1.0, 2.0, 4.0)]
    assert hs == sorted(hs, reverse=True), hs


def test_entropy_covers_the_answer_not_the_prompt(gof):
    """Same completion-only rule the log-prob half follows: scoring prompt tokens
    would report the uncertainty of text the model was HANDED."""
    import torch
    logits = torch.zeros(6, 8)
    logits[:2] = torch.tensor([50.0] + [-50.0] * 7)   # prompt: near-zero entropy
    H = _run(gof, logits, prompt_len=3)
    assert H == pytest.approx(math.log(8), abs=1e-4), \
        "a peaked PROMPT must not drag the answer's entropy down"


def test_entropy_reaches_the_step_row_and_the_log(gof):
    """A metric that is computed and never surfaced is not an instrument."""
    import inspect
    src = inspect.getsource(gof)
    assert '"entropy": float(cfg.get("_last_entropy", 0.0))' in src
    assert '"entropy": sum(o.get("entropy", 0.0) for o in _outs) / len(_outs)' in src
    assert "H={out.get('entropy', 0.0):.3f}" in src


# ============ collapse is not exhaustion ================================
def test_the_collapse_exit_is_separate_from_the_exhaustion_exit(gof):
    """Operator 2026-09-20: "Idk if we should pivot to entropy though" -- for the
    EXHAUSTION exit the answer is no, and the reason is mechanical.

    GRPO's gradient is advantage * dlogpi with advantage = r - mean(r), so ZERO
    REWARD VARIANCE IS ZERO GRADIENT regardless of entropy. `grad` is the quantity
    that enters the update; entropy does not. Measured on the 2026-09-20 bracket, a
    model phrased one wrong answer four ways -- "[reward] bands: 0 hq / 0 lq / 4
    wrong", uniform, zero gradient -- while H sat near 1.3. An entropy-based exit
    would have kept training on records that can teach nothing.

    Entropy earns a DIFFERENT job. A mode-collapsed policy also emits uniform groups,
    so grad goes flat and the run exits announcing EXHAUSTED for a run that BROKE.
    Recording those identically is an Article X defect: a NOT TESTED logged as a
    clean finish."""
    import inspect
    src = inspect.getsource(gof)
    assert "COLLAPSED: token entropy" in src
    assert "EXHAUSTED: no new available-gradient low" in src
    assert "collapsed = True" in src, "the two exits must be distinguishable after"


def test_the_collapse_guard_is_not_armed_on_coverage(gof):
    """The exhaustion gate only counts after full coverage -- before that a flat
    stretch means "have not seen the pool yet". Collapse has no such excuse: a policy
    can degenerate at any step, and waiting for coverage to notice wastes the run."""
    import inspect
    src = inspect.getsource(gof)
    i_guard = src.index("COLLAPSED: token entropy")
    i_cov = src.index("if not _covered_all:")
    assert i_guard < i_cov, "the collapse check must run BEFORE the coverage branch"


def test_collapse_needs_three_consecutive_evals(gof):
    """One low reading is noise at these batch sizes; the gate requires a run of
    them, the same shape the exhaustion gate uses for grad."""
    import inspect
    src = inspect.getsource(gof)
    assert "_h_flat >= 3" in src
    assert "_h_flat = 0" in src, "a recovery must reset the counter"


def test_entropy_floor_defaults_to_the_published_band(gof):
    """reasoning-ch07: H < 0.5 converging/collapsing, H > 1.5 exploring."""
    import inspect
    src = inspect.getsource(gof)
    assert 'dest="entropy_floor"' in src and "default=0.5" in src


# ============ the book's adv_std, as a metric ============================
def test_the_book_adv_std_is_one_for_a_split_group_zero_for_a_uniform_one(gof):
    """ch07 normalises PER GROUP -- (r - mean) / (r.std() + 1e-4) -- which makes its
    adv_std near-binary, and that is exactly what gives it meaning:

        [1,1,0,0] -> std 0.577 -> advantages +-0.866 -> adv_std 1.0   has signal
        [0,0,0,0] -> std 0     -> 0/1e-4 = 0         -> adv_std 0.0   dead
        [1,1,1,1] -> same                            -> adv_std 0.0   dead

    Averaged over groups it is the FRACTION STILL CARRYING SIGNAL. The book's rule
    reads straight off it: "a nonzero (and stable) advantage std means there is
    still a usable learning signal". Ours divides by a batch-pooled EMA and floats
    near 1 even on a dead pool, so it cannot make that statement."""
    import torch
    for rewards, want in ([1., 1., 0., 0.], 1.0), ([0.] * 4, 0.0), ([1.] * 4, 0.0):
        r = torch.tensor(rewards)
        g = r - r.mean()
        sd = float(r.std())
        got = float((g / (sd + 1e-4)).std()) if sd > 0 else 0.0
        assert abs(got - want) < 0.02, (rewards, got, want)


def test_the_book_metric_never_touches_training(gof):
    """Operator 2026-09-20: "compute per-group std purely as a metric, change
    nothing about training". It must be computed BEFORE our normalisation and must
    not be assigned back into `advantages`."""
    import inspect
    src = inspect.getsource(gof)
    i_book = src.index('cfg["_last_adv_std_book"]')
    i_ours = src.index("advantages = advantages / float(_bstd)")
    assert i_book < i_ours, "the book metric is read off the raw rewards first"
    seg = src[i_book:i_ours]
    assert "advantages =" not in seg, "the metric must not rewrite advantages"


def test_the_live_fraction_is_surfaced(gof):
    import inspect
    src = inspect.getsource(gof)
    assert '"adv_std_book": out.get("adv_std_book", 0.0)' in src, "metrics.json"
    assert "live={out.get('adv_std_book', 0.0):.2f}" in src, "step log"


# ============ adv_avg: the book's bug detector ==========================
def test_adv_avg_is_zero_by_construction(gof):
    """ch07: "The average advantage is and stays zero ... if it drifted away from
    zero, that would point to a bug or other normalization bias issue."

    Advantages are r - mean(r), so the mean is zero BY CONSTRUCTION. That is what
    makes it useful as a check: it cannot drift for a legitimate reason, so any
    nonzero reading is a defect -- a masking bug, a reduction over the wrong axis,
    dead groups counted into the mean, or a normaliser applied twice."""
    import torch
    for rewards in ([1., 1., 0., 0.], [1., .5, 0., .5], [0.] * 4, [1.] * 4):
        r = torch.tensor(rewards)
        assert abs(float((r - r.mean()).mean())) < 1e-6, rewards


def test_dividing_by_a_constant_cannot_move_adv_avg(gof):
    """Our batch-pooled normalisation divides by a scalar, which scales the mean by
    that scalar -- and zero times anything is still zero. So the check stays valid
    under our variant, which is what lets it detect OUR bugs and not just the
    book's."""
    import torch
    r = torch.tensor([1., 1., 0., 0.])
    adv = (r - r.mean()) / 0.37
    assert abs(float(adv.mean())) < 1e-6


def test_the_drift_warning_only_fires_when_it_should(gof):
    """A sanity check that prints every step is noise; one that prints only on
    breakage is a tripwire. Threshold 1e-3 is far above float error and far below
    any real bias."""
    import inspect
    src = inspect.getsource(gof)
    assert "ADV_AVG_DRIFT" in src
    assert 'abs(float(out.get("adv_avg", 0.0))) > 1e-3' in src


def test_adv_avg_reaches_the_row(gof):
    import inspect
    src = inspect.getsource(gof)
    assert '"adv_avg": float(advantages.mean().detach())' in src
    assert '"adv_avg": out.get("adv_avg", 0.0)' in src


# ============ retirement is direction-blind; split it ====================
def test_retirement_cannot_tell_solved_from_hopeless(gof):
    """THE DEFECT THIS SPLIT EXISTS FOR. Operator 2026-09-20: "I feel we should have
    retired_all_win retired_all_fail."

    is_retired tests the POSTERIOR -- float(v[-1]) <= retire_below -- and E[p(1-p)]
    is low at BOTH ends. A record answered 4/4 and one answered 0/4 produce the same
    posterior, so the bare `retired` count reports opposite outcomes as one number."""
    vl_solved = {"q": [0.2, 0.08, 0.04, 0.02]}
    vl_hopeless = {"q": [0.2, 0.08, 0.04, 0.02]}
    assert gof.is_retired(vl_solved, "q", 3, 0.05) is True
    assert gof.is_retired(vl_hopeless, "q", 3, 0.05) is True, \
        "identical posteriors -- the visit log cannot distinguish direction"


def test_the_split_reads_direction_off_the_rewards(gof):
    """reward_history DOES carry the direction, which is what makes the split
    recoverable at all."""
    pool = [{"question": "won"}, {"question": "lost"}, {"question": "flip"}]
    vl = {q: [0.2, 0.08, 0.04, 0.02] for q in ("won", "lost", "flip")}
    hist = {"won": [0.95, 0.92, 0.98, 0.91],      # all above floor
            "lost": [0.10, 0.05, 0.20, 0.00],     # all below
            "flip": [0.95, 0.05, 0.92, 0.10]}     # spans it -- UNSTABLE
    c = gof._straddle_census(hist, pool, floor=0.7, visit_log=vl, retire_after=3)
    assert c["retired"] == 3
    assert c["retired_win"] == 1
    assert c["retired_fail"] == 1
    assert c["retired_flip"] == 1, \
        "a record flipping across visits is unstable, not settled"


def test_the_unstable_bucket_is_not_forced_into_either(gof):
    """The purity worry, verbatim: "the record could have flipped on its all one
    type of record past 3 no variance". Three consecutive UNIFORM visits can still
    be all-wrong, all-wrong, all-right. Forcing that into win or fail would claim
    knowledge the history does not support."""
    pool = [{"question": "f"}]
    vl = {"f": [0.2, 0.08, 0.04, 0.02]}
    c = gof._straddle_census({"f": [1.0, 1.0, 0.0, 0.0]}, pool, floor=0.7,
                             visit_log=vl, retire_after=3)
    assert c["retired_flip"] == 1
    assert c["retired_win"] == 0 and c["retired_fail"] == 0


# ============ retirement clips, it does not delete =======================
def test_a_retired_record_stays_in_the_pool(gof):
    """Operator 2026-09-20: "We never truly retire anything, we still occasionally
    check ... The way you have it now they simply drop off and thats bad."

    Deleting a record permanently asserts THE POLICY NEVER IMPROVES -- that something
    hopeless at step 5 is hopeless at step 30. The policy is the one thing that
    definitely changes. Measured on the live run: 119 of 240 records deleted by step
    15, none of them for being solved."""
    pool = [{"question": "dead"}, {"question": "live"}]
    vl = {"dead": [0.2, 0.08, 0.04, 0.02], "live": [0.2, 0.2, 0.2, 0.2]}
    live, n_ret = gof.eligible_pool(pool, vl, retire_after=3)
    assert n_ret == 1
    assert len(live) == 2, "the retired record must still be drawable"
    assert pool[0]["_retired"] is True and pool[1]["_retired"] is False


def test_the_clip_lowers_the_draw_weight_but_never_raises_it(gof):
    """min(), not max(): retirement may only ever REDUCE a record's chance. A record
    already scoring below the floor must not be promoted to it."""
    pool = [{"question": "dead"}, {"question": "live"}]
    hist = {"dead": [0.0, 0.0, 0.0, 0.0], "live": [0.95, 0.1, 0.9, 0.2]}
    vl = {"dead": [0.2, 0.08, 0.04, 0.02], "live": [0.2, 0.2, 0.2, 0.2]}
    gof.eligible_pool(pool, vl, retire_after=3)   # sets the _retired flag
    w, _ = gof.boltzmann_weights(pool, hist, temp=0.25, floor=0.7)
    assert w[0] < w[1], "a retired record draws less than a straddling one"
    assert w[0] > 0.0, "but it is NOT zero -- it must still be re-checkable"


def test_the_floor_is_small_but_nonzero(gof):
    """Small enough that retired records cost a few percent of the draw, large enough
    that each gets re-checked over a long run. Zero would be deletion by another
    name."""
    assert 0.0 < gof.RETIRED_RANK_FLOOR < 0.1


def test_a_retired_record_that_differentiates_rejoins_with_no_special_case(gof):
    """Nothing brings it back explicitly: a re-check that shows spread raises its
    posterior through the ordinary path, so it simply stops being retired on the next
    census."""
    pool = [{"question": "q"}]
    vl_dead = {"q": [0.2, 0.08, 0.04, 0.02]}
    gof.eligible_pool(pool, vl_dead, retire_after=3)
    assert pool[0]["_retired"] is True
    vl_revived = {"q": [0.2, 0.08, 0.04, 0.02, 0.21]}   # a differentiating re-check
    gof.eligible_pool(pool, vl_revived, retire_after=3)
    assert pool[0]["_retired"] is False


# ============ recent classification: the last 3 visits ===================
def _census(gof, rewards, K=4):
    pool = [{"question": "q"}]
    return gof._straddle_census({"q": rewards}, pool, floor=0.7, rollouts=K)


def test_recent_classifies_on_the_last_three_visits_not_all_history(gof):
    """Operator 2026-09-20: "Those will be records we know that landed on that
    category last 3, else we mark as unstable."

    THE DEFECT IT FIXES: the cumulative keys answer "has this record EVER straddled",
    which goes stale the moment behaviour changes. A record that straddled once, ten
    visits ago, and has failed uniformly since still counts as straddling."""
    old_straddle_then_dead = [1.0, 0.0, 1.0, 0.0] + [0.0] * 12
    c = _census(gof, old_straddle_then_dead)
    assert c["straddle"] == 1, "cumulative still says straddle -- unchanged, as promised"
    assert c["recent_fail"] == 1, "but RECENT correctly says it is dead now"
    assert c["recent_straddle"] == 0


def test_recent_pass_needs_every_one_of_the_last_three_visits(gof):
    c = _census(gof, [1.0] * 12)
    assert c["recent_pass"] == 1 and c["recent_unstable"] == 0


def test_flipping_between_uniform_visits_is_unstable_not_settled(gof):
    """Each visit uniform, the visits disagreeing. Calling that settled would assert
    something the history contradicts."""
    c = _census(gof, [1.0] * 4 + [0.0] * 4 + [1.0] * 4)
    assert c["recent_unstable"] == 1
    assert c["recent_pass"] == 0 and c["recent_fail"] == 0


def test_any_straddling_visit_wins_the_classification(gof):
    """A record that differentiated in ANY of the last three visits still carries
    gradient, which outranks how the other two looked."""
    c = _census(gof, [0.0] * 4 + [1.0, 0.0, 1.0, 0.0] + [0.0] * 4)
    assert c["recent_straddle"] == 1


def test_the_old_keys_are_untouched(gof):
    """Added additively so every existing plot, receipt and parser reads exactly what
    it read before -- the operator's condition was "without fucking up the plots"."""
    c = _census(gof, [1.0, 0.0, 1.0, 0.0])
    for k in ("straddle", "allfail", "allwin", "all_perfect", "all_correct",
              "unvisited", "retired", "retire_eligible", "grad", "rollout_acc"):
        assert k in c, k


# ============ two checkpoints: latest and best ==========================
def test_latest_is_saved_every_eval_and_best_only_on_improvement(tmp_path):
    """Operator 2026-09-20: "We should be saving two versions / Latest / Best".

    THE DEFECT: `best` is written only on a STRICT improvement, so a run that ties
    or dips -- the normal shape of RL -- leaves the newest weights nowhere on disk.
    Measured cost that day: 11 steps / ~4 GPU-hours discarded, because `best` sat at
    round 5 while the run was at step 17."""
    import re, io as _io
    src = _io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    body = src[src.index('_latest = Path(args.out) / "latest"'):]
    body = body[:body.index("[patience] NEW BEST")]

    save_at = body.index("model.save_pretrained(_latest)")
    guard_at = body.index("if score > best:")
    assert save_at < guard_at, "latest must be saved BEFORE the improvement guard"

    assert '"step": epoch_step' in body, "resume_state must record the step reached"
    assert 'Path(args.out) / "best"' in body, "best needs a stable path, not just best_rd{rd}"
    assert 'f"best_rd{rd}"' in body, "the per-round history is kept for existing tooling"


# ============ preflight: refuse a second trainer ========================
def _probe(monkeypatch, *cmdlines):
    """Stub the process enumerator with the PowerShell probe's output shape."""
    import subprocess

    class _R:
        returncode = 0
        stdout = chr(10).join(cmdlines) + chr(10)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _R())


_ME = ("C:/py310/python.exe docs/grpo_one_file.py --base W:/x "
       "--out W:/phase_runs/rl_full --rounds 70")
_OTHER = ("C:/py310/python.exe docs/grpo_one_file.py --base W:/other "
          "--out W:/phase_runs/rl_full --rounds 50")


def test_preflight_refuses_a_second_trainer_on_the_same_out(gof, monkeypatch):
    """The harm is TWO TRAINERS SHARING ONE --out. Measured 2026-09-20: the second
    run's replay posteriors were the previous campaign's, and 95 minutes became NOT
    TESTED without a single error."""
    monkeypatch.setattr(gof.sys, "argv", _ME.split())
    _probe(monkeypatch, _ME, _OTHER)
    with pytest.raises(SystemExit) as e:
        gof._refuse_if_another_trainer_is_running("W:/phase_runs/rl_full")
    assert "REFUSING TO START" in str(e.value)


def test_preflight_allows_a_run_on_a_different_out(gof, monkeypatch, capsys):
    """Two runs on separate directories are merely slow, not corrupting. Refusing
    them would be a false refusal, and false refusals train people to bypass the
    guard."""
    monkeypatch.setattr(gof.sys, "argv", _ME.split())
    _probe(monkeypatch, _ME,
           "C:/py310/python.exe docs/grpo_one_file.py --base W:/o --out W:/other_run")
    gof._refuse_if_another_trainer_is_running("W:/phase_runs/rl_full")
    assert "OK" in capsys.readouterr().out


def test_preflight_ignores_mere_mentions_of_the_filename(gof, monkeypatch, capsys):
    """A first cut matched any line containing "grpo_one_file" and caught its own
    shell history -- greps, editors, this source file."""
    monkeypatch.setattr(gof.sys, "argv", _ME.split())
    _probe(monkeypatch, _ME,
           'bash -c grep -n "out" docs/grpo_one_file.py',
           'vim docs/grpo_one_file.py',
           'python -c print(open("docs/grpo_one_file.py").read())')
    gof._refuse_if_another_trainer_is_running("W:/phase_runs/rl_full")
    assert "OK" in capsys.readouterr().out


def test_preflight_does_not_count_the_launcher_and_its_child_twice(gof, monkeypatch,
                                                                   capsys):
    """One run appears as a wrapper line and a child line with the same tail."""
    monkeypatch.setattr(gof.sys, "argv", _ME.split())
    _probe(monkeypatch, "C:/Git/env.exe PYTHONIOENCODING=utf-8 " + _ME, _ME)
    gof._refuse_if_another_trainer_is_running("W:/phase_runs/rl_full")
    assert "OK" in capsys.readouterr().out


def test_preflight_is_loud_when_no_probe_works(gof, monkeypatch, capsys):
    """wmic is absent on this box, and a wmic-only guard printed SKIPPED and let
    every launch through. Silence is the failure mode; say the guard is not
    protecting and name the check that is."""
    import subprocess

    def _boom(*a, **k):
        raise FileNotFoundError("no such tool")
    monkeypatch.setattr(subprocess, "run", _boom)
    gof._refuse_if_another_trainer_is_running("W:/phase_runs/rl_full")
    out = capsys.readouterr().out
    assert "NOT protecting" in out and "nvidia-smi" in out
# ============ patience counts SOLVED as progress =========================
def test_solved_high_counts_as_progress(gof):
    """Operator 2026-09-21: "We obviously have the wrong patience counter."

    Straddle apexed at 82 on step 9 and FELL to 58 by step 26 -- because straddlers
    convert into solved records, which is the run succeeding. all-win stayed 0
    throughout (p^4). Both original patience terms were unreachable, so the counter
    could only increment and the stop was guaranteed to fire on a healthy run."""
    b = _patience_block()
    assert "_solved_now > _solved_apex" in b, "a new solved high must be detected"
    import io as _io
    src = _io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    assert "_solved_apex = -1" in src, "the solved apex needs an initialiser"


def test_metrics_carry_across_a_resume(gof):
    """A resume used to overwrite metrics.json with a one-row file: _metrics started
    empty and the write mode is "w". Weights and visit log carried; the curve every
    plot and projection reads did not."""
    import io as _io
    src = _io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    blk = src[src.index("_metrics = []"):]
    blk = blk[:blk.index("def ", 100)] if "def " in blk[100:] else blk[:3000]
    assert 'carry_visits' in blk[:1200], "carrying is gated on a resume, not every run"
    assert '_metrics = _json.load' in blk[:1200], "prior rows must be loaded"


# ============ patience arming (operator's rule, 2026-09-21) ==============
def _patience_block():
    import io as _io
    src = _io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    b = src[src.index("_prog = []"):]
    return b[:b.index("print(\"[signal] apex")]


def test_patience_does_not_arm_before_full_coverage(gof):
    """Operator: "patience shouldn't start until / Coverage is at least 100%".
    A pool that has not been seen once has not had its chance."""
    b = _patience_block()
    assert "_covered_all = len(_covered) >= len(pool)" in b
    assert "if not (_covered_all and _harvesting):" in b


def test_patience_does_not_arm_until_solved_passes_straddle(gof):
    """Operator: "Solved passes straddle". Before that crossover the pool is still
    converting all-fail into contested records, so a flat solved count means
    nothing."""
    b = _patience_block()
    assert '_harvesting = _solved_now > _sig.get("recent_straddle", 0)' in b


def test_unarmed_patience_is_held_at_zero_not_merely_paused(gof):
    """Held at 0 so the counter cannot carry a head start out of the opening phase --
    on the live run it had already reached 17/20 before arming would have occurred."""
    b = _patience_block()
    unarmed = b[b.index("if not (_covered_all and _harvesting):"):]
    unarmed = unarmed[:unarmed.index("elif")]
    assert "_straddle_stale = 0" in unarmed


def test_once_armed_only_a_new_solved_high_resets_it(gof):
    """Operator: "Then we track when solved stops increasing." A new straddle apex
    must NOT reset it -- straddle falling is the run succeeding."""
    b = _patience_block()
    armed = b[b.index('elif "new solved high" in _prog:'):]
    assert "_straddle_stale = 0" in armed.split("else:")[0]
    assert "_straddle_stale += 1" in armed.split("else:")[1]


# ============ the census survives a resume ==============================
def test_reward_history_is_persisted_every_eval(gof):
    """Measured 2026-09-21. A resume carried the weights, the curve AND the replay
    posteriors, and still reset the operator's own census:

        step 30  straddling 51 | solved 57 | fail 131 | unstable  1
        step 31  straddling 31 | solved  3 | fail 172 | unstable 34

    The last-3-visits classification reads reward_history, which lived only in
    memory. visit_log holds decaying posteriors and a pooled [k, n]; neither can
    say WHICH rollouts were correct on each of the last three visits."""
    import io as _io
    src = _io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    assert 'reward_history.json' in src
    # The WRITE site, not the carry site -- both name the file.
    w = src.index('os.path.join(args.out, "reward_history.json")')
    blk = src[w - 900:w + 200]
    assert "_keep = 4 * max(1" in blk, "trim to what the classifier reads, 3 visits"
    assert "_rs[-_keep:]" in blk, "only the tail is kept"


def test_reward_history_is_carried_on_resume(gof):
    import io as _io
    src = _io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    blk = src[src.index("[carry] reward history for") - 1200:]
    blk = blk[:blk.index("_load_visit_log(args.carry_visits)")]
    assert "reward_history.setdefault" in blk, "prior rewards must be restored"
    assert "census RESTARTS" in blk, \
        "a missing file must be announced, not silently degrade the census"


def test_reward_history_is_defined_before_the_carry_block(gof):
    """Ordering bug this guards: the carry block extends reward_history, so a carry
    placed above its initialiser would NameError on every resume."""
    import io as _io
    src = _io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    assert src.index("reward_history = _prior_history(") < \
        src.index("[carry] reward history for")


# ============ the x axis cannot go backwards =============================
def test_eval_numbering_continues_across_a_resume(gof):
    """Operator 2026-09-21: "Im seeing weird diagonal lines."

    THE CAUSE, exactly: the plot's x axis was the trainer's own eval counter, which
    counts THIS process's evals and restarts at 1. After the resume the x array read

        [1, 2, ... 29, 30, 1]

    so every series ran out to x=30 and jumped back to x=1, drawing a line clear
    across all eight panels. Not a trend line and not the data -- a backtracking x."""
    import io as _io
    src = _io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    assert '"eval": _eval_offset + len(_adv_hist)' in src
    assert "_eval_offset = len(_metrics)" in src, "offset by the carried rows"


def test_plot_x_axis_is_position_not_the_eval_field(gof):
    """Belt and braces: even a run whose "eval" field is wrong must plot monotonically,
    because the rows are already one-per-eval in order."""
    import io as _io
    src = _io.open("archive/tools/plot_rl_metrics.py", encoding="utf-8").read()
    assert "x = list(range(1, len(rows) + 1))" in src
    assert 'x = [r["eval"] for r in rows]' not in src


def test_plot_x_is_monotonic_on_a_curve_with_a_reset_eval_field(gof, tmp_path):
    """The live failing case: 31 rows whose last one carries eval=1."""
    import importlib.util, json, matplotlib
    matplotlib.use("Agg")
    spec = importlib.util.spec_from_file_location("pl", "archive/tools/plot_rl_metrics.py")
    pl = importlib.util.module_from_spec(spec)
    sys.modules["pl"] = pl
    spec.loader.exec_module(pl)
    import matplotlib.pyplot as plt

    rows = []
    for i in range(31):
        rows.append({
            "eval": 1 if i == 30 else i + 1,          # <- the reset
            "score": 2, "reward": .5, "grad": 15.0, "adv_std": 1.1, "entropy": 1.5,
            "straddle": 50, "allfail": 130, "allwin": 0, "unvisited": 0,
            "covered": 240, "pool": 240, "straddle_apex": 82, "allwin_apex": 0,
            "answer_len": 120, "elapsed_s": 60.0 * (i + 1), "eval_n": 16,
            "recent_straddle": 50, "recent_pass": 40, "recent_fail": 130,
            "recent_unstable": 1, "best": 2, "loss": .004, "gnorm": 9.0,
            "live_frac": 1.0, "batch_std": .38, "retired": 0, "retire_eligible": 240,
        })
    pl.build(rows, str(tmp_path / "p.png"), 1.0)
    xs = list(plt.gcf().get_axes()[0].get_lines()[0].get_xdata())
    assert xs[0] == 1 and xs[-1] == 31
    assert all(xs[i + 1] > xs[i] for i in range(len(xs) - 1)), \
        "a backtracking x draws a line across the panel"


# ============ apex trackers are not plotted as -1 ========================
def test_apex_trackers_are_carried_across_a_resume(gof):
    """Operator 2026-09-21: "Accuracy and all win collapsed (green)".

    Neither did. rollout_acc ROSE through the resume (0.279 -> 0.285 -> 0.289 ->
    0.291) and allwin was 0 for all 33 steps -- there was nothing to collapse. What
    moved was allwin_apex 2 -> -1: a SENTINEL, initialised to -1 because no apex has
    been seen yet, written straight into metrics.json and plotted as a dip below zero
    on a count that cannot be negative.

    A tracker variable must never appear in a series read as data."""
    import io as _io
    src = _io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    assert '_carried_apex["_solved_apex"]' in src
    assert '_carried_apex.get("_allwin_apex", -1)' in src


def test_apexes_are_initialised_before_the_carry_block_reads_them(gof):
    """Ordering bug: the apexes were first written to read _carried_apex at the top
    of train(), 40 lines ABOVE where that dict is built -- a NameError on every run,
    resumed or not."""
    import io as _io
    src = _io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    sentinel = src.index("_straddle_apex = _allwin_apex = _solved_apex = -1")
    carried = src.index('_carried_apex.get("_straddle_apex"')
    assert sentinel < carried, "the sentinel must exist before the carried read"
    built = src.index("_carried_apex = {}")
    assert built < carried, "_carried_apex must be built before it is read"


# ============ optimizer state travels with the weights ==================
def test_optimizer_state_is_saved_with_latest(gof):
    """Operator 2026-09-21: "you didn't save optim state inbetween".

    Correct, and it was a judgement I made and stated as minor without checking.
    Measured across the resume, AdamW's moments restarting at zero looks like:

        loss   0.0128 -> 0.0000
        gnorm  18.68  -> 8.67

    Two of the twelve discontinuous fields. Weights without the optimizer is not a
    resume, it is a warm restart."""
    import io as _io
    src = _io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    blk = src[src.index('_latest = Path(args.out) / "latest"'):]
    blk = blk[:blk.index("resume_state.json")]
    assert 'optimizer.state_dict()' in blk
    assert '"optim_state.pt"' in blk
    assert 'scheduler.state_dict()' in blk, "the LR schedule position travels too"


def test_optimizer_state_is_restored_from_the_base(gof):
    """Keyed on --base, not --carry-visits: the moments are a property of the
    WEIGHTS, so they travel together or not at all."""
    import io as _io
    src = _io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    blk = src[src.index('_opt_path = os.path.join(args.base, "optim_state.pt")'):]
    blk = blk[:blk.index("model.train()")]
    assert "optimizer.load_state_dict" in blk
    assert "scheduler.load_state_dict" in blk


def test_a_missing_optimizer_state_says_warm_restart_out_loud(gof):
    """A fresh SFT deposit legitimately has none. A resume that silently gets none is
    the failure -- so the log names which one this is, every launch."""
    import io as _io
    src = _io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    blk = src[src.index('_opt_path = os.path.join(args.base, "optim_state.pt")'):]
    blk = blk[:blk.index("model.train()")]
    assert "WARM RESTART" in blk and "fresh optimizer" in blk


# ============ atomic writes ==============================================
def test_curve_and_history_are_written_atomically(gof):
    """Measured 2026-09-21: the watcher's backfill read metrics.json inside the
    trainer's write window and died with

        json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)

    open(path, "w") TRUNCATES FIRST, so every step had a window where the curve was
    zero bytes on disk -- and a crash inside that window loses the run's history."""
    import io as _io
    src = _io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    assert '_atomic_json(os.path.join(args.out, "metrics.json")' in src
    assert '_atomic_json(os.path.join(args.out, "reward_history.json")' in src
    blk = src[src.index("def _atomic_json("):]
    blk = blk[:blk.index("def _refuse_if_another")]
    assert "os.replace(tmp, path)" in blk, "rename is what makes it atomic"
    assert "os.fsync" in blk, "flush to the OS before renaming"


def test_atomic_json_round_trips_and_leaves_no_temp(gof, tmp_path):
    p = str(tmp_path / "curve.json")
    gof._atomic_json(p, [{"eval": 1, "score": 3}])
    import json as _json, io as _io, os as _os
    assert _json.load(_io.open(p, encoding="utf-8"))[0]["score"] == 3
    assert [f for f in _os.listdir(str(tmp_path)) if ".tmp" in f] == []


def test_atomic_json_replaces_rather_than_truncating(gof, tmp_path):
    """The old file must stay readable until the new one is complete."""
    import json as _json, io as _io
    p = str(tmp_path / "curve.json")
    gof._atomic_json(p, [{"eval": 1}])
    gof._atomic_json(p, [{"eval": 1}, {"eval": 2}])
    assert len(_json.load(_io.open(p, encoding="utf-8"))) == 2


def test_backfill_survives_an_unreadable_curve(gof, tmp_path, capsys):
    """A backfill must never be the thing that kills the watcher."""
    import importlib.util, sys as _sys
    spec = importlib.util.spec_from_file_location("bf", "archive/tools/backfill_entropy.py")
    bf = importlib.util.module_from_spec(spec)
    _sys.modules["bf"] = bf
    spec.loader.exec_module(bf)
    bad = tmp_path / "metrics.json"
    bad.write_text("", encoding="utf-8")          # the exact partial-write case
    assert bf._read_rows(str(bad)) is None
    assert "skipping" in capsys.readouterr().out


# ============ the book's average policy ratio (ch07 7.4.2) ==============
def test_policy_ratio_is_tracked_from_the_ratio_itself(gof):
    """Operator 2026-09-21: "I have a new var to add to our plot / Avg policy ratio."

    Book ch07 7.4.2 verbatim: "we compute the average policy ratio, policy_ratio =
    torch.mean(ratio), which we can track and report during the training run. Here,
    the result is a relatively large 5.6106." 1.0 means the update did not change how
    likely the sampled tokens are; distance from 1.0 is how hard the step moved."""
    import io as _io
    src = _io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    blk = src[src.index("ratio = torch.exp(logps_t - old_logps)"):]
    blk = blk[:blk.index("clipped_ratio")]
    assert 'cfg["_last_policy_ratio"] = float(ratio.mean())' in blk, \
        "must be mean(ratio), the book's definition, taken from the same tensor"
    assert "torch.no_grad()" in blk, "a metric must not join the graph"


def test_policy_ratio_reaches_metrics(gof):
    """It stays in the receipt, but it left the STEP LINE -- that slot now carries KL,
    which is the term that moves under beta 0.02."""
    import io as _io
    src = _io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    assert '"policy_ratio": round(cfg.get("_last_policy_ratio", 0.0), 4)' in src
    assert "pratio={cfg.get('_last_policy_ratio', 0.0):.3f}" not in src


def test_kl_loss_is_tracked_and_panelled_instead_of_the_ratio(gof):
    """Operator 2026-09-21 asked for the policy ratio (ch07 7.4.2). It was added, then
    reading the code showed it is CONSTANT 1.0 here: old_logps = logps.detach() on a
    single inner epoch, so exp(x - x) = 1 exactly and the 0.2 clip never engages.

    A panel that can only draw a flat line is worse than no panel -- it looks like a
    measurement. KL is the term that actually varies under beta 0.02."""
    import io as _io
    src = _io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    assert 'cfg["_last_kl"] = float(kl_loss)' in src
    assert '"kl_loss": round(cfg.get("_last_kl", 0.0), 6)' in src
    assert "kl={cfg.get('_last_kl', 0.0):.5f}" in src, "on the step line too"

    plt_src = _io.open("archive/tools/plot_rl_metrics.py", encoding="utf-8").read()
    g = plt_src.index('a.plot(x, [r["gnorm"] for r in rows]')
    k = plt_src.index("KL LOSS, not the policy ratio")
    assert g < k, "KL belongs beside gnorm: step size vs drift"
    assert "THE BOOK'S POLICY RATIO" not in plt_src, "the dead panel must be gone"


def test_policy_ratio_is_kept_as_a_tripwire_not_a_panel(gof):
    """It stays in metrics.json BECAUSE it is constant: if it ever leaves 1.0 without
    a rollout pool reused across minibatches, the loss changed under us."""
    import io as _io
    src = _io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    assert '"policy_ratio": round(cfg.get("_last_policy_ratio", 0.0), 4)' in src
    assert "CONSTANT 1.0 IN THIS CONFIGURATION" in src, "say why it cannot move"


def test_plot_survives_metrics_without_the_new_key(gof, tmp_path):
    """Every curve written before today lacks policy_ratio."""
    import importlib.util, matplotlib, json
    matplotlib.use("Agg")
    spec = importlib.util.spec_from_file_location("pl2", "archive/tools/plot_rl_metrics.py")
    pl = importlib.util.module_from_spec(spec); sys.modules["pl2"] = pl
    spec.loader.exec_module(pl)
    rows = [{"eval": i + 1, "score": 2, "reward": .5, "grad": 15.0, "adv_std": 1.1,
             "entropy": 1.5, "straddle": 50, "allfail": 130, "allwin": 0,
             "unvisited": 0, "covered": 240, "pool": 240, "straddle_apex": 82,
             "allwin_apex": 0, "answer_len": 120, "elapsed_s": 60.0 * (i + 1),
             "eval_n": 16, "best": 2, "loss": .004, "gnorm": 9.0, "live_frac": 1.0,
             "batch_std": .38, "retired": 0, "retire_eligible": 240,
             "recent_straddle": 50, "recent_pass": 40, "recent_fail": 130,
             "recent_unstable": 1} for i in range(6)]
    pl.build(rows, str(tmp_path / "p.png"), 1.0)     # must not KeyError


# ============ REQ-W1: every judged pair is persisted ====================
def test_judged_pairs_are_captured_at_the_judge_return(gof):
    """Spec: reward-world-model.md REQ-W1.

    R is the only non-trivial term of this MDP -- episodes are one step, so P is
    trivial -- and R is the metered remote call that killed a 14-step run with
    HTTP 402. Learning R is the world model, and this capture is its dataset.

    Before this, (question, answer, score) existed NOWHERE: the ledger holds
    {visits, hits, rewards:[...]}, reward_history holds scores per record, the logs
    hold two truncated samples per group. ~7,000 pairs from that run are gone."""
    import io as _io
    src = _io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    blk = src[src.index("_raws = _judge_many(_groups"):]
    blk = blk[:blk.index("[cost] wave")]
    assert '"judged_pairs.jsonl"' in blk, "capture must sit at the judge return"
    assert '"a",' in blk, "append-only; a run must never truncate earlier pairs"
    assert '"score": float(_sc)' in blk, "the RAW judge score, not just the band"
    assert '"q": _g["question"]' in blk and '"a": _ans' in blk, \
        "the PAIR is the point -- a score with no answer is what we already had"


def test_capture_failure_cannot_take_the_run_down(gof):
    """A dataset-collection convenience must never be able to kill training. The
    judge raising is fail-fast by design; the capture is not."""
    import io as _io
    src = _io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    blk = src[src.index('"judged_pairs.jsonl"') - 700:]
    blk = blk[:blk.index("[cost] wave")]
    assert "except Exception" in blk and "training unaffected" in blk


# ============ the KL anchor is the deposit, not the resume point =========
def test_kl_reference_can_be_pinned_independently_of_base(gof):
    """Operator spotted this in the plot, 2026-09-22: a huge red spike then zero.

        step 14  kl 0.008547   (4x the run's previous max)
        step 15  kl 0.000000   (resumed)

    The reference was always deepcopy(model) -- the weights --base loaded. On a fresh
    run that is the SFT deposit and correct. On a RESUME --base is the latest
    checkpoint, so the reference re-anchored to it and KL went to zero: every step of
    accumulated drift released for free, and again on every future resume. beta 0.02
    exists to hold the policy near the DEPOSIT."""
    import io as _io
    src = _io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    assert '"--ref-base"' in src, "there must be a way to pin it"
    blk = src[src.index("_kl_off = float(getattr(args"):]
    blk = blk[:blk.index("if _kl_off and not args.qlora:")]
    assert 'getattr(args, "ref_base", None) or args.base' in blk, \
        "default to --base so a fresh run is unchanged"
    assert "kl-anchor" in blk, "the log must name which anchor is in force"
    assert "AutoModelForCausalLM.from_pretrained" in blk, \
        "a pinned reference loads its own weights, not a copy of the policy"


def test_the_anchor_choice_is_announced_either_way(gof):
    """Same law as [optim-state]: a run must say which regime it is in, so a released
    anchor cannot be mistaken for a held one."""
    import io as _io
    src = _io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    blk = src[src.index("_kl_off = float(getattr(args"):]
    blk = blk[:blk.index("if _kl_off and not args.qlora:")]
    assert "reference = --base" in blk and "reference PINNED to" in blk
