"""--optim-cpu keeps AdamW's moments in CPU RAM and must not change the math.

Spec: .spec/specs/r3-domain-injection/tasks.md #46 (optimizer moments on the CPU).
Operator 2026-09-27: "Wtf fix this shit" -- trial C spilled to shared GPU memory.

The CPU optimizer is only acceptable if a run cannot tell which one it got: same params
after N steps, a state_dict either optimizer can load, grads freed after the step.
"""
import copy
import importlib.util
import sys

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def gof():
    spec = importlib.util.spec_from_file_location("gof_optim_cpu", "docs/grpo_one_file.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["gof_optim_cpu"] = m
    saved, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(m)
    sys.argv = saved
    return m


def _net(device):
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(),
                               torch.nn.Linear(16, 1)).to(device)


def _groups(net):
    ps = list(net.parameters())
    return [{"params": [p for p in ps if p.dim() >= 2], "weight_decay": 0.01},
            {"params": [p for p in ps if p.dim() < 2], "weight_decay": 0.0}]


def _train(net, opt, steps=5):
    g = torch.Generator().manual_seed(1)
    dev = next(net.parameters()).device
    for _ in range(steps):
        opt.zero_grad()
        x = torch.randn(32, 8, generator=g).to(dev)
        net(x).pow(2).mean().backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()


DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_same_params_as_plain_adamw(gof):
    a, b = _net(DEV), _net(DEV)
    kw = dict(lr=1e-2, betas=(0.9, 0.999), eps=1e-8)
    _train(a, torch.optim.AdamW(_groups(a), **kw))
    _train(b, gof.CpuAdamW(_groups(b), **kw))
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-6), (pa - pb).abs().max()
    # and training actually moved them
    assert not torch.allclose(next(_net(DEV).parameters()), next(a.parameters()))


def test_moments_live_on_cpu_and_grads_are_freed(gof):
    net = _net(DEV)
    opt = gof.CpuAdamW(_groups(net), lr=1e-2)
    _train(net, opt, steps=2)
    st = opt.state_dict()["state"]
    assert len(st) == 4
    assert all(s["exp_avg"].device.type == "cpu" for s in st.values())
    assert all(p.grad is None for p in net.parameters())


def test_state_dict_loads_across_optimizers(gof):
    a = _net(DEV)
    plain = torch.optim.AdamW(_groups(a), lr=1e-2)
    _train(a, plain, steps=3)
    b = _net(DEV)
    b.load_state_dict(a.state_dict())
    cpu = gof.CpuAdamW(_groups(b), lr=1e-2)
    # deepcopy = what torch.load of optim_state.pt gives; an in-memory state_dict would
    # share the `step` tensor and both optimizers would advance one counter
    cpu.load_state_dict(copy.deepcopy(plain.state_dict()))
    _train(a, plain, steps=2)
    _train(b, cpu, steps=2)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-6)


def test_lr_scheduler_drives_cpu_optimizer(gof):
    net = _net(DEV)
    opt = gof.CpuAdamW(_groups(net), lr=1e-2)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 ** i)
    _train(net, opt, steps=1)
    sch.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(5e-3)
