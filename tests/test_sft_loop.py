"""tests/test_sft_loop.py -- the plain-PyTorch SFT loop: collate, loss, and the step protocol.

Spec: C:/Users/user/.claude/plans/virtual-sauteeing-crown.md Layer 4 task 11.
NO GOVERNING SPEC in the repo's .spec tree; basis is the operator-approved restructure plan.
"""
import types

import pytest
import torch

from rlv2.train.sft import loop


def _tok(side="right", pad=0):
    return types.SimpleNamespace(padding_side=side, pad_token_id=pad, eos_token_id=1)


def _rec(ids, supervised_from):
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * supervised_from + ids[supervised_from:]}


def test_collate_pads_right_to_the_batch_max_and_masks_the_padding():
    b = loop.collate([_rec([5, 6, 7], 1), _rec([8, 9], 1)], _tok("right"))
    assert b["input_ids"].tolist() == [[5, 6, 7], [8, 9, 0]]
    assert b["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert b["labels"].tolist() == [[-100, 6, 7], [-100, 9, -100]]


def test_collate_pads_left_when_the_tokenizer_does():
    b = loop.collate([_rec([5, 6, 7], 1), _rec([8, 9], 1)], _tok("left"))
    assert b["input_ids"].tolist() == [[5, 6, 7], [0, 8, 9]]
    assert b["attention_mask"].tolist() == [[1, 1, 1], [0, 1, 1]]
    assert b["labels"].tolist() == [[-100, 6, 7], [-100, -100, 9]]


def test_collate_keeps_the_padding_a_record_already_carries():
    pre = {"input_ids": [5, 6, 0], "attention_mask": [1, 1, 0], "labels": [-100, 6, -100]}
    b = loop.collate([pre, _rec([8, 9, 4], 1)], _tok("right"))
    assert b["attention_mask"].tolist() == [[1, 1, 0], [1, 1, 1]]


def test_collate_rejects_a_ragged_record():
    bad = {"input_ids": [1, 2, 3], "attention_mask": [1, 1], "labels": [1, 2, 3]}
    with pytest.raises(AssertionError):
        loop.collate([bad], _tok())


class _FixedLogits(torch.nn.Module):
    """Returns the same logits whatever it is fed, so the expected loss is computable by hand."""

    def __init__(self, logits):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self._logits = logits

    def forward(self, input_ids, attention_mask):
        return types.SimpleNamespace(logits=self._logits + self.w)


def test_loss_is_the_mean_cross_entropy_over_supervised_tokens_only():
    vocab = 4
    logits = torch.zeros(1, 3, vocab)
    logits[0, 0, 2] = 5.0                       # position 0 predicts token 2 (the label at position 1)
    batch = {"input_ids": torch.zeros(1, 3, dtype=torch.long),
             "attention_mask": torch.ones(1, 3, dtype=torch.long),
             "labels": torch.tensor([[-100, 2, -100]])}   # only position 1 is supervised
    got = loop.calc_loss_batch(batch, _FixedLogits(logits))
    want = torch.nn.functional.cross_entropy(logits[0, :1], torch.tensor([2]))
    assert got.item() == pytest.approx(want.item(), abs=1e-6)
    assert got.item() < 0.05                    # the model was confident and right


def test_a_batch_with_no_supervised_token_gives_nan_not_a_silent_zero():
    batch = {"input_ids": torch.zeros(1, 2, dtype=torch.long), "attention_mask": torch.ones(1, 2, dtype=torch.long),
             "labels": torch.full((1, 2), -100)}
    assert torch.isnan(loop.calc_loss_batch(batch, _FixedLogits(torch.zeros(1, 2, 4))))


def test_weight_decay_applies_to_weights_only_not_biases_or_norm_weights():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)                 # proj.weight decays, proj.bias does not
            self.ln = torch.nn.LayerNorm(4)                   # inside a LayerNorm: never decays
            self.post_attention_layernorm = torch.nn.Linear(4, 4, bias=False)   # named like a norm
            self.q_norm = torch.nn.Linear(4, 4, bias=False)

    m = Tiny()
    opt = loop.make_optimizer(m, lr=1e-3, weight_decay=0.1, betas=(0.9, 0.999), eps=1e-8)
    decay = {id(p) for g in opt.param_groups if g["weight_decay"] == 0.1 for p in g["params"]}
    nodecay = {id(p) for g in opt.param_groups if g["weight_decay"] == 0.0 for p in g["params"]}
    assert decay == {id(m.proj.weight)}
    assert nodecay == {id(p) for p in (m.proj.bias, m.ln.weight, m.ln.bias,
                                       m.post_attention_layernorm.weight, m.q_norm.weight)}


def test_label_smoothing_raises_the_loss_of_a_confident_correct_prediction():
    logits = torch.zeros(1, 2, 4)
    logits[0, 0, 1] = 8.0
    batch = {"input_ids": torch.zeros(1, 2, dtype=torch.long), "attention_mask": torch.ones(1, 2, dtype=torch.long),
             "labels": torch.tensor([[-100, 1]])}
    plain = loop.calc_loss_batch(batch, _FixedLogits(logits), 0.0).item()
    smooth = loop.calc_loss_batch(batch, _FixedLogits(logits), 0.1).item()
    assert smooth > plain + 0.1
