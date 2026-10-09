"""tests/test_common_loop.py -- the shared CPT/SFT loop: accumulation, stepping, callbacks, failure.

Spec: C:/Users/user/.claude/plans/virtual-sauteeing-crown.md Layer 4 tasks 10-11.
NO GOVERNING SPEC in the repo's .spec tree; basis is the operator-approved restructure plan.
"""
import pytest
import torch
from transformers import TrainerCallback

from rlv2.common import loop

V = 11


class Tiny(torch.nn.Module):
    """Embedding + head; next token = (token + 1) % V, so a few steps visibly lower the loss."""

    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(V, 16)
        self.head = torch.nn.Linear(16, V)

    def gradient_checkpointing_enable(self):
        pass

    def forward(self, input_ids, attention_mask):
        import types
        return types.SimpleNamespace(logits=self.head(self.emb(input_ids)))


def _batch(n_rows, width, start=0):
    ids = torch.tensor([[(start + r + c) % V for c in range(width)] for r in range(n_rows)])
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


HP = dict(lr=0.05, lr_scheduler="constant", warmup_steps=0, weight_decay=0.0, max_grad_norm=1.0,
          betas=(0.9, 0.999), eps=1e-8, optim="adamw_torch")


def test_window_loss_is_summed_over_the_window_and_divided_by_its_targets():
    torch.manual_seed(0)
    m = Tiny()
    a, b = _batch(1, 3), _batch(4, 9)                     # CPT-shaped (labels = input_ids): 2 targets vs 32
    n = loop.target_tokens(a) + loop.target_tokens(b)
    windowed = (loop.calc_loss_batch(a, m, 0.0, n) + loop.calc_loss_batch(b, m, 0.0, n)).item()
    big = {k: torch.cat([torch.nn.functional.pad(a[k], (0, 6), value=(-100 if k == "labels" else 0)), b[k]])
           for k in a}                                    # the same 34 targets in one padded batch
    assert windowed == pytest.approx(loop.calc_loss_batch(big, m).item(), abs=1e-5)
    mean_of_means = (loop.calc_loss_batch(a, m).item() + loop.calc_loss_batch(b, m).item()) / 2
    assert abs(windowed - mean_of_means) > 1e-3           # and it is NOT the mean of the two means


def test_targets_are_counted_after_the_shift_so_the_first_token_of_a_sequence_is_not_one():
    torch.manual_seed(0)
    m = Tiny()
    b = _batch(2, 5)                                      # labels = input_ids, as in CPT: 10 labels, 8 targets
    assert loop.target_tokens(b) == 8                     # the old Trainer path counted 10
    mean = loop.calc_loss_batch(b, m).item()              # true mean over the 8 targets
    assert loop.calc_loss_batch(b, m, 0.0, loop.target_tokens(b)).item() == pytest.approx(mean, rel=1e-5)


def test_the_gradient_of_a_window_is_the_window_mean_not_divided_by_grad_accum_again(monkeypatch):
    """The old Trainer path divided it by grad_accum a second time (accelerate); corrected 2026-10-07."""
    monkeypatch.setattr(loop, "make_optimizer",
                        lambda model, lr, wd, betas, eps, optim: torch.optim.SGD(model.parameters(), lr=1.0))
    batches = [_batch(2, 5, 0), _batch(3, 6, 3)]

    def fresh():
        torch.manual_seed(0)
        return Tiny()

    trained = fresh()
    w0 = trained.head.weight.detach().clone()
    loop.train_batches(trained, iter(batches), num_steps=1, grad_accum=2, **{**HP, "max_grad_norm": 1e9})

    ref = fresh()
    n = sum(loop.target_tokens(b) for b in batches)
    sum(loop.calc_loss_batch(b, ref, 0.0, n) for b in batches).backward()
    got = trained.head.weight.detach().cpu()              # train_batches moves the model to the GPU when there is one
    assert torch.allclose(got, w0 - 1.0 * ref.head.weight.grad, atol=1e-6)            # the window mean
    assert not torch.allclose(got, w0 - 1.0 * ref.head.weight.grad / 2, atol=1e-6)    # not window mean / grad_accum


def test_one_loss_per_optimizer_step_and_the_loss_falls():
    torch.manual_seed(0)
    batches = [_batch(4, 6, s) for s in range(12)]
    losses = loop.train_batches(Tiny(), iter(batches), num_steps=6, grad_accum=2, **HP)
    assert len(losses) == 6                               # 12 batches / grad_accum 2
    assert losses[-1] < losses[0] * 0.8


def test_a_partial_last_window_still_takes_a_step():
    torch.manual_seed(0)
    losses = loop.train_batches(Tiny(), iter([_batch(2, 5, s) for s in range(5)]), num_steps=3, grad_accum=2, **HP)
    assert len(losses) == 3                               # windows of 2, 2, 1


def test_it_stops_at_num_steps_even_when_batches_remain():
    torch.manual_seed(0)
    losses = loop.train_batches(Tiny(), iter([_batch(2, 5, s) for s in range(20)]), num_steps=4, **HP)
    assert len(losses) == 4


def test_callbacks_see_the_trainer_protocol_with_a_rising_global_step():
    seen = []

    class Spy(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            seen.append(("begin", state.max_steps))

        def on_step_end(self, args, state, control, **kw):
            seen.append(("step", state.global_step, round(state.epoch, 3)))

        def on_train_end(self, args, state, control, **kw):
            seen.append(("end", state.global_step))

    torch.manual_seed(0)
    loop.train_batches(Tiny(), iter([_batch(2, 5, s) for s in range(3)]), num_steps=3, callbacks=[Spy()], **HP)
    assert seen == [("begin", 3), ("step", 1, 0.333), ("step", 2, 0.667), ("step", 3, 1.0), ("end", 3)]


def test_a_callback_can_stop_training():
    class Stop(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if state.global_step == 2:
                control.should_training_stop = True

    torch.manual_seed(0)
    losses = loop.train_batches(Tiny(), iter([_batch(2, 5, s) for s in range(9)]), num_steps=9, callbacks=[Stop()], **HP)
    assert len(losses) == 2


def test_a_non_finite_loss_raises_instead_of_training_on():
    bad = _batch(2, 5)
    m = Tiny()
    with torch.no_grad():
        m.head.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        loop.train_batches(m, iter([bad]), num_steps=1, **HP)


def test_an_unknown_optimizer_name_is_an_error_not_a_default():
    with pytest.raises(ValueError):
        loop.make_optimizer(Tiny(), 1e-3, 0.0, (0.9, 0.999), 1e-8, optim="sgd")


def test_collate_can_force_the_pad_side_regardless_of_the_tokenizer():
    import types
    tok = types.SimpleNamespace(padding_side="left", pad_token_id=0, eos_token_id=1)
    recs = [{"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "labels": [5, 6, 7]},
            {"input_ids": [8], "attention_mask": [1], "labels": [8]}]
    assert loop.collate(recs, tok, pad_side="right")["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert loop.collate(recs, tok)["input_ids"].tolist() == [[5, 6, 7], [0, 0, 8]]
