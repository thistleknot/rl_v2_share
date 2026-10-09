"""NoRA on top of LoRA/QLoRA: column-normalized lora_A.

Spec: nora-vs-qlora-playbook.md (arXiv 2608.31036; PEFT fork Joluck/NoRA).
Operator 2026-09-13: "patch in nora as an option and/or addition to qlora".

THE ONE THING THAT IS EASY TO GET WRONG. lora_A is `nn.Linear(k, r)`, so its
weight is `(r, k)`. A "column" in the paper is A[:, j] -- the rank-dimension
vector feeding input dim j -- so the norm is taken over dim=0, the r rows.
Normalizing dim=1 instead would unit-norm each rank vector across inputs, a
different and wrong transform that still "runs" and still trains. These tests
pin the axis, because nothing at runtime would tell you it was the wrong one.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

import importlib.util
import sys


@pytest.fixture(scope="module")
def gof():
    spec = importlib.util.spec_from_file_location("gof_nora",
                                                  "docs/grpo_one_file.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gof_nora"] = mod
    spec.loader.exec_module(mod)
    return mod


def _tiny_peft():
    """A minimal real PEFT model -- no mock, per integration-first testing."""
    from peft import LoraConfig, get_peft_model

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(16, 16)

        def forward(self, x):
            return self.q_proj(x)

    cfg = LoraConfig(r=4, lora_alpha=4, target_modules=["q_proj"],
                     lora_dropout=0.0, bias="none")
    return get_peft_model(Tiny(), cfg)


def _a_weights(model):
    from peft.tuners.lora import LoraLayer
    out = []
    for m in model.modules():
        if isinstance(m, LoraLayer):
            for A in m.lora_A.values():
                out.append(A.weight)
    return out


def test_init_mode_unit_normalizes_columns(gof):
    model = gof.apply_nora(_tiny_peft(), "init")
    for W in _a_weights(model):
        norms = W.norm(dim=0)                      # one per input dim
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), norms


def test_the_norm_is_over_the_rank_axis_not_the_input_axis(gof):
    """Guards the silent-wrong-axis failure: dim=0 is r, dim=1 is k."""
    model = gof.apply_nora(_tiny_peft(), "init")
    W = _a_weights(model)[0]
    assert W.shape == (4, 16), "lora_A weight is (r, k)"
    assert torch.allclose(W.norm(dim=0), torch.ones(16), atol=1e-5)
    # the other axis must NOT be unit -- if it is, the wrong dim was normalized
    assert not torch.allclose(W.norm(dim=1), torch.ones(4), atol=1e-2)


def test_init_mode_leaves_the_weight_a_plain_parameter(gof):
    """init = normalize once, then vanilla LoRA. No parametrization overhead."""
    import torch.nn.utils.parametrize as P
    from peft.tuners.lora import LoraLayer
    model = gof.apply_nora(_tiny_peft(), "init")
    for m in model.modules():
        if isinstance(m, LoraLayer):
            for A in m.lora_A.values():
                assert not P.is_parametrized(A, "weight")


def test_init_mode_does_not_stay_normalized_under_training(gof):
    """The documented difference between the two modes, made observable."""
    model = gof.apply_nora(_tiny_peft(), "init")
    A = _a_weights(model)[0]
    with torch.no_grad():
        A.add_(torch.ones_like(A) * 0.5)           # stand in for an update
    assert not torch.allclose(A.norm(dim=0), torch.ones(16), atol=1e-3)


def test_forward_mode_renormalizes_on_every_access(gof):
    """`forward` must hold the invariant that `init` only establishes once."""
    import torch.nn.utils.parametrize as P
    from peft.tuners.lora import LoraLayer
    model = gof.apply_nora(_tiny_peft(), "forward")

    parametrized = [A for m in model.modules() if isinstance(m, LoraLayer)
                    for A in m.lora_A.values()]
    assert parametrized, "precondition: there is a lora_A to parametrize"
    for A in parametrized:
        assert P.is_parametrized(A, "weight")
        assert torch.allclose(A.weight.norm(dim=0), torch.ones(16), atol=1e-5)
        with torch.no_grad():                       # perturb the RAW parameter
            A.parametrizations.weight.original.add_(1.0)
        assert torch.allclose(A.weight.norm(dim=0), torch.ones(16), atol=1e-5), \
            "forward mode must re-normalize after the underlying weight moves"


def test_forward_mode_keeps_the_graph_differentiable(gof):
    """Gradients must flow THROUGH the norm -- that is the paper's point.

    lora_B IS ZERO-INITIALIZED, so dL/dA = B^T(...) = 0 on the very first step.
    That is plain LoRA and has nothing to do with NoRA; asserting a non-zero A
    gradient without moving B first tests the zero-init, not the parametrization.
    """
    from peft.tuners.lora import LoraLayer
    model = gof.apply_nora(_tiny_peft(), "forward")
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoraLayer):
                for B in m.lora_B.values():
                    B.weight.fill_(0.1)          # B != 0, so A can receive grad
    out = model(torch.randn(2, 16))
    out.sum().backward()
    grads = [A.parametrizations.weight.original.grad
             for m in model.modules()
             if type(m).__name__.startswith("Linear") and hasattr(m, "lora_A")
             for A in m.lora_A.values()]
    grads = [g for g in grads if g is not None]
    assert grads, "no gradient reached the raw lora_A parameter"
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_an_unknown_mode_is_rejected_rather_than_silently_ignored(gof):
    with pytest.raises(ValueError):
        gof.apply_nora(_tiny_peft(), "nope")


def test_zero_columns_do_not_divide_by_zero(gof):
    """lora_B is zero-init; a zero column in A is not impossible either."""
    model = _tiny_peft()
    with torch.no_grad():
        _a_weights(model)[0].zero_()
    gof.apply_nora(model, "init")
    W = _a_weights(model)[0]
    assert torch.isfinite(W).all(), "eps clamp must prevent NaN/inf"
