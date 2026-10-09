"""The optimizer knobs from optimization.md, and the one that was binding.

Operator 2026-09-14: "Look at our markdown file in docs on optim settings and add
these but also pull in the others."

Source: train-llm-from-scratch/docs/foundations/optimization.md -- AdamW's beta1 /
beta2 / eps, the decay vs no-decay split, warmup + cosine decay, gradient
accumulation, and gradient clipping ("a stability guard, especially useful in
RL").

THE ONE THAT WAS BINDING. The clip was HARDCODED at 1.0 while this project's
measured gradient norm ran 137-278. `g <- g * min(1, c/||g||)` with c=1 and
||g||=211 scales the update by 1/211, so the clip -- not the learning rate -- set
every step size ever taken here. Every lr sweep was scaling a unit vector, which
is exactly why lr response looked so cleanly monotone.

THE NAMING TRAP. This file's `--beta` is the KL coefficient. Adam's are
`--adam-beta1` / `--adam-beta2`. Conflating them silently changes the KL penalty
while you believe you are tuning momentum.
"""
import io

import pytest

RL = "docs/grpo_one_file.py"
HPO = "archive/tools/rl_hpo_optuna.py"


@pytest.fixture(scope="module")
def rl():
    return io.open(RL, encoding="utf-8").read()


@pytest.fixture(scope="module")
def hpo():
    return io.open(HPO, encoding="utf-8").read()


def test_the_grad_clip_is_no_longer_hardcoded(rl):
    assert "clip_grad_norm_(model.parameters(),\n" in rl or \
           "clip_grad_norm_(model.parameters(), args.grad_clip)" in rl
    assert "clip_grad_norm_(model.parameters(), 1.0)" not in rl, \
        "the hardcoded 1.0 was scaling every update by ~1/200"
    assert '"--grad-clip"' in rl


def test_adam_moments_are_exposed(rl):
    for flag in ("--adam-beta1", "--adam-beta2", "--adam-eps"):
        assert flag in rl, flag
    assert "betas=(args.adam_beta1, args.adam_beta2)" in rl
    assert "eps=args.adam_eps" in rl


def test_the_kl_beta_and_adam_betas_stay_distinct(rl):
    """The collision is the point of the naming."""
    assert '"--beta"' in rl, "the KL coefficient keeps its name"
    assert "dest=\"adam_beta1\"" in rl
    assert "NOT --beta" in rl, "the trap must be documented at the flag"


def test_lion_is_selectable_and_adamw_is_the_default(rl):
    assert 'choices=("adamw", "lion")' in rl
    assert 'default="adamw"' in rl, "every number measured so far used AdamW"
    assert "from lion_pytorch import Lion" in rl


def test_the_decay_split_survives_both_optimizers(rl):
    """optimization.md: decay matrices, not biases or 1-D params. Both paths
    must receive the same groups, or the two arms are not comparable."""
    i = rl.index("_groups = [")
    j = rl.index("print(f\"[optim]")
    body = rl[i:j]
    assert body.count("_groups") >= 3, "both optimizers take the same groups"
    assert '"weight_decay": 0.0' in body


def test_the_clip_search_range_spans_the_measured_norms(hpo):
    """Measured gnorm 137-278. A range topping out below that could never test
    'do not clip at all', which is the hypothesis worth falsifying."""
    import re
    m = re.search(r'suggest_float\("grad_clip", ([0-9.]+), ([0-9.]+)', hpo)
    assert m, "grad_clip must be searched"
    lo, hi = float(m.group(1)), float(m.group(2))
    assert lo <= 1.0, "the incumbent 1.0 must stay reachable"
    assert hi >= 278.0, "the range must reach above the measured norms"


def test_the_optimizer_family_is_searched(hpo):
    assert 'suggest_categorical("optimizer", ["adamw", "lion"])' in hpo


def test_the_clip_arithmetic_explains_the_observed_behaviour():
    """Why this axis was worth finding: the scale factor at the old default."""
    def scale(norm, clip):
        return min(1.0, clip / norm)

    assert scale(211.0, 1.0) < 0.005, "the old default discarded 99.5% of it"
    assert scale(211.0, 300.0) == 1.0, "a clip above the norm is a no-op"
    assert scale(13.8, 1.0) < 0.08, "even the per-token norm was heavily clipped"
