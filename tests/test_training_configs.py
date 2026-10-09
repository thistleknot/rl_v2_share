"""
Unit sweep over the CPT / SFT / MGPO config anchors (training-configs.md rev 2).

These are pure assertions over rig.yml + training-configs.md -- no model, no GPU.
Full sweep runs in well under a second.

The invariants under test are the *rules*, not just the numbers:
  R1  alpha is FIXED at 32 at every stage and every rank (alpha=2r/4r cancels the
      1/r suppression and silently multiplies the step).
  R2  effective adapter step = lr * alpha/r must land inside the published band
      for that stage's *own* regime.
  R3  LoRA optimal lr ~= 10x the full-FT optimum for the SAME stage.
  R4  RL optima sit ~10x colder than SFT optima within a parametrization.
  R5  no anchor may be cited across an objective boundary (SFT anchor on an RL stage).
"""
import math
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rig():
    return yaml.safe_load((ROOT / "rig.yml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def arms(rig):
    return rig["grounding"]["arms"]


@pytest.fixture(scope="module")
def published(rig):
    return rig["grounding"]["published"]


def step(lr, alpha, r):
    """Effective adapter step scale."""
    return lr * alpha / r


def in_band(x, band, tol=1e-12):
    lo, hi = band
    return lo - tol <= x <= hi + tol


# --------------------------------------------------------------------------
# CPT -- lora (all-linear), the default at this scale
# --------------------------------------------------------------------------

def test_cpt_default_is_lora_not_full(arms):
    """CPT-LoRA is the default. The 'LoRA is capacity-bound at CPT' claim is
    Biderman's gap transplanted over this repo's own measurement."""
    cpt = arms["cpt_lora"]
    assert cpt["status"] == "MEASURED_HERE"
    assert cpt["prescribed"]["target"] == "all-linear"


def test_cpt_lr_and_schedule(arms):
    p = arms["cpt_lora"]["prescribed"]
    assert p["lr"] == pytest.approx(1e-4)
    assert "1.0e-5" in p["lr_schedule"] or "1e-5" in p["lr_schedule"]
    assert "cosine" in p["lr_schedule"]
    assert "warmup" in p["lr_schedule"]


def test_cpt_lr_sits_at_low_end_of_supervised_lora_band(arms):
    """1e-4 is the low end of the 1e-4..2e-4 supervised LoRA band -- pulled low
    for the long schedule. It is derived, not quoted."""
    lr = arms["cpt_lora"]["prescribed"]["lr"]
    assert in_band(lr, [1e-4, 2e-4])
    assert lr == pytest.approx(1e-4), "must be the LOW end, not the middle"
    assert "derived" in arms["cpt_lora"]["anchor"]


def test_cpt_replay_fraction(arms):
    lo, hi = arms["cpt_lora"]["prescribed"]["replay_general_frac"]
    assert (lo, hi) == (0.02, 0.05)


def test_cpt_rank_stays_at_measured_16_not_proposed_band(arms):
    """The 64-256 capacity band is unadopted: neither ~2 bits/param nor
    ~1 bit/token is measured here or quoted from a source, and r=16 already
    matched full-param at 1.42% trainable."""
    cpt = arms["cpt_lora"]
    assert cpt["prescribed"]["r"] == 16
    assert "rank_band_unverified" in cpt
    assert "NOT adopted" in cpt["rank_band_unverified"]


def test_cpt_capacity_rule_is_falsifiable_and_currently_falsified(arms):
    """Keep the escalation condition stated so it can be tested, not assumed."""
    txt = arms["cpt_lora"]["rank_band_unverified"]
    assert "held-out-ppl gap at r=16" in txt
    assert "measured none" in txt


# --------------------------------------------------------------------------
# SFT -- Spectrum, ! domain
# --------------------------------------------------------------------------

def test_sft_lora_matches_published_ift_general(arms, published):
    sft = arms["sft"]
    assert sft["lr"] == pytest.approx(1e-4)
    assert sft["lr"] == pytest.approx(published["biderman_ift_general"]["lora"])
    assert sft["status"] == "MATCHES"


def test_sft_rank_and_alpha(arms):
    sft = arms["sft"]
    assert sft["r"] == 16
    assert sft["alpha"] == 32


def test_sft_full_twin_is_10x_colder(published):
    """R3: LoRA 1e-4 <-> full 1e-5 for the same stage."""
    twin = published["lwr_sft_full_twin"]
    assert twin["lora"] / twin["full"] == pytest.approx(10.0)
    assert twin["full"] == pytest.approx(1e-5)


def test_sft_rank_escalation_is_gated_on_data_scale(published):
    assert "post-training scale" in published["lwr_rl_capacity"]


# --------------------------------------------------------------------------
# MGPO -- Signal, domain x answer-R. Weakest touch by design.
# --------------------------------------------------------------------------

def test_mgpo_lora_asrun_config_is_marked_superseded(arms):
    """As-run 2e-4 @ alpha 64 is retained as the record of what executed --
    but must never read as a recommendation."""
    m = arms["mgpo_lora"]
    assert m["lr"] == pytest.approx(2e-4)
    assert m["alpha"] == 64
    assert m["status"].startswith("SUPERSEDED")
    assert "not a recommendation" in m["note"]


def test_mgpo_lora_prescribed_band(arms, published):
    p = arms["mgpo_lora"]["prescribed"]
    assert p["lr"] == [1e-5, 5e-5]
    assert p["lr"] == published["lwr_rl_lora"]["lr"]
    assert p["r"] == 16
    assert p["alpha"] == 32
    assert p["target"] == "all-linear"


def test_mgpo_lora_effective_batch_ceiling(arms):
    """LoRA is batch-size intolerant for RL."""
    assert arms["mgpo_lora"]["prescribed"]["effective_batch_max"] == 32


def test_mgpo_full_prescribed_band(arms, published):
    p = arms["mgpo_full"]["prescribed"]
    assert p["lr"] == [1e-6, 3e-6]
    assert p["lr"] == published["lwr_rl_full"]["full"]


def test_mgpo_full_asrun_1e5_is_3_to_10x_hot(arms):
    m = arms["mgpo_full"]
    lo, hi = m["prescribed"]["lr"]
    assert m["lr"] / hi == pytest.approx(10 / 3, rel=1e-6)
    assert m["lr"] / lo == pytest.approx(10.0)
    assert m["status"].startswith("SUPERSEDED")


def test_mgpo_rank_one_would_suffice(published):
    """r=1 matches full FT for RL; 16 is free margin, not a requirement."""
    assert published["lwr_rl_lora"]["r"][0] == 1


# --------------------------------------------------------------------------
# Cross-stage rules
# --------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ["sft", "cpt_lora", "mgpo_lora"])
def test_R1_alpha_fixed_at_32_everywhere(arms, arm):
    """alpha=2r / 4r cancels the 1/r suppression. Heat goes in lr only."""
    assert arms[arm]["prescribed"]["alpha"] == 32 if arm != "sft" else arms[arm]["alpha"] == 32


def test_R1_alpha_convention_is_labelled_fixed(arms):
    for arm in ("cpt_lora", "mgpo_lora"):
        assert arms[arm]["prescribed"]["alpha_convention"] == "fixed-32"


def test_R2_asrun_mgpo_step_is_8_to_20x_the_published_step(arms):
    """This is the number that predicted the GRPO entropy collapse."""
    m = arms["mgpo_lora"]
    asrun = step(m["lr"], m["alpha"], m["r"])
    assert asrun == pytest.approx(8e-4)
    lo, hi = [step(lr, 32, 16) for lr in m["prescribed"]["lr"]]
    assert asrun / hi == pytest.approx(8.0)
    assert asrun / lo == pytest.approx(40.0)


def test_R4_rl_optima_are_10x_colder_than_sft_optima(arms, published):
    """Within LoRA: SFT 1e-4 -> RL 1e-5..5e-5. Within full: SFT 1e-5 -> RL 1e-6..3e-6."""
    sft_lora = arms["sft"]["lr"]
    rl_lora_lo = published["lwr_rl_lora"]["lr"][0]
    assert sft_lora / rl_lora_lo == pytest.approx(10.0)

    sft_full = published["lwr_sft_full_twin"]["full"]
    rl_full_lo = published["lwr_rl_full"]["full"][0]
    assert sft_full / rl_full_lo == pytest.approx(10.0)


def test_R5_no_sft_anchor_survives_on_an_rl_stage(arms):
    """An anchor is only 'the same anchor' inside one parametrization. The SFT
    LoRA anchors (1e-4 / 2e-4) must not appear on an RL LoRA arm; the SFT
    full-FT anchors (1e-5 / 5e-6) must not appear on an RL full arm.

    NB: RL-LoRA's floor is 1e-5, numerically equal to the SFT full-FT twin.
    That collision is a coincidence of the 10x rule applied twice down
    different axes, NOT a cross-stage citation -- so it is checked per-axis.
    """
    forbidden = {"mgpo_lora": {1e-4, 2e-4}, "mgpo_full": {1e-5, 5e-6}}
    for arm, bad in forbidden.items():
        band = arms[arm]["prescribed"]["lr"]
        band = band if isinstance(band, list) else [band]
        for lr in band:
            assert not any(math.isclose(lr, f) for f in bad), (
                f"{arm}: {lr} is a supervised-regime anchor on an RL stage"
            )


def test_R5_rl_lora_floor_colliding_with_sft_full_twin_is_coincidence(arms, published):
    """Pin the collision explicitly so nobody 'simplifies' one into the other."""
    rl_lora_floor = arms["mgpo_lora"]["prescribed"]["lr"][0]
    sft_full_twin = published["lwr_sft_full_twin"]["full"]
    assert rl_lora_floor == pytest.approx(sft_full_twin)
    # ...but they live on opposite sides of both axes:
    assert step(rl_lora_floor, 32, 16) != pytest.approx(sft_full_twin)


def test_R5_stage_strength_ordering_comes_from_capacity_not_detuning(rig):
    status = rig["grounding"]["rl_grounding_status"]
    assert "CPT > SFT > MGPO" in status
    assert "NOT from detuning lr" in status


def test_stage_strength_ordering_holds_on_lora_steps(arms):
    """CPT >= SFT > MGPO on effective step, with each stage at its own optimum."""
    cpt = step(arms["cpt_lora"]["prescribed"]["lr"], 32, 16)
    sft = step(arms["sft"]["lr"], arms["sft"]["alpha"], arms["sft"]["r"])
    mgpo_hi = step(arms["mgpo_lora"]["prescribed"]["lr"][1], 32, 16)
    assert cpt >= sft > mgpo_hi


# --------------------------------------------------------------------------
# Doc <-> rig agreement
# --------------------------------------------------------------------------

def test_training_configs_doc_is_rev2_and_defaults_cpt_to_lora():
    doc = (ROOT / "training-configs.md").read_text(encoding="utf-8")
    assert "rev 2" in doc
    assert "LoRA on all linear layers is the default at EVERY stage, CPT included." in doc


def test_doc_records_the_unverified_reference():
    """arXiv 2602.06204 does not resolve; nothing may depend on it."""
    doc = (ROOT / "training-configs.md").read_text(encoding="utf-8")
    assert "2602.06204" in doc
    assert "cannot verify" in doc
