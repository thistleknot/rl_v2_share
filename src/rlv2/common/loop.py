"""src/rlv2/common/loop.py -- the training loop CPT and SFT share, as plain PyTorch.

Book base: Raschka, *LLMs from Scratch* (Manning)
  ch05/03_bonus_pretraining_on_gutenberg/pretraining_simple.py -- train_model_simple: forward,
        next-token cross entropy, backward, optimizer step, zero_grad, one step per batch
  ch07/01_main-chapter-code/gpt_instruction_finetuning.py -- calc_loss_batch, custom_collate_fn
        (pad to the batch max, pad targets with -100, ignore_index=-100)
  ch05/04_learning_rate_schedulers -- linear warmup, then cosine decay

Notes: docs/PROVENANCE.md (slide -> notes -> code). No governing spec in this repo; basis is the author-approved
restructure of the CPT and SFT loops onto the book loop (2026-10).

OURS (everything below is not in the book; each item says why):
  - A batch is whatever the caller hands in: CPT's token-budgeted quantile batches, SFT's
    pre-packed bins. The batch size varies by design; the token budget keeps the step inside VRAM.
  - Pad side is the caller's choice (SFT follows the tokenizer, CPT pads right).
  - 8-bit AdamW (bitsandbytes), weight decay off for biases and norm weights, embedding tables
    kept in 32-bit state: the optimizer state of a full-parameter 0.8B model does not fit a
    16 GB card in fp32. LoRA runs use plain AdamW.
  - Gradient checkpointing, for the same reason.
  - fp16 autocast with a loss scaler (CPT): fp32 master weights, fp16 activations.
  - Gradient accumulation with the loss normalised by the prediction targets of the WHOLE window:
    each batch's loss is summed and divided by the targets in the window, so variable-size batches
    weigh every token equally and the window gradient is the true mean. The HF Trainer path this
    replaced did two things differently (corrected 2026-10-07, so CPT runs from that date on are
    not strictly comparable with earlier ones): it counted labels before the one-token shift, so for
    CPT each sequence's first token was counted though nothing predicts it (reported loss ~0.3% low),
    and `accelerate`'s `backward` divided the gradient by grad_accum a second time. Pinned by
    tests/test_common_loop.py.
  - Gradient clipping on every step (the book clips only after warmup).
  - The schedule comes from `transformers.get_scheduler`, not the book's hand-written warmup +
    cosine: the HPO drivers sweep nine scheduler types, and "cosine" there is the book's schedule.
  - Label smoothing through `F.cross_entropy` (the book has none).
  - Trainer callbacks (throughput floor, observe, half-epoch eval) keep working: this loop drives
    them through `transformers`' own CallbackHandler instead of re-implementing their protocol.
"""
import math
import re
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def collate(records, tok, pad_side=None):
    """Book custom_collate_fn, for records that carry `input_ids`, `attention_mask` and `labels`.

    Require : equal lengths within a record (-100 in labels = not trained on). Records may
              already be padded to their length block.
    Guarantee: tensors padded to the batch max on `pad_side` (default: the tokenizer's);
               padded labels are -100.
    """
    n = max(len(r["input_ids"]) for r in records)
    left = (pad_side or tok.padding_side) == "left"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids, labels, mask = [], [], []
    for r in records:
        assert len(r["input_ids"]) == len(r["labels"]) == len(r["attention_mask"]), "ragged record"
        pad = n - len(r["input_ids"])
        ids.append(([pad_id] * pad + list(r["input_ids"])) if left else (list(r["input_ids"]) + [pad_id] * pad))
        labels.append(([-100] * pad + list(r["labels"])) if left else (list(r["labels"]) + [-100] * pad))
        mask.append(([0] * pad + list(r["attention_mask"])) if left else (list(r["attention_mask"]) + [0] * pad))
    return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask), "labels": torch.tensor(labels)}


def target_tokens(batch):
    """Next-token targets that carry loss: labels not equal to -100 AFTER the one-token shift."""
    return int((batch["labels"][:, 1:] != -100).sum())


def calc_loss_batch(batch, model, label_smoothing=0.0, n_items=None):
    """Book calc_loss_batch: cross entropy of the next token, padding and prompt ignored.

    Guarantee: fp32. With n_items=None, the mean over the supervised targets of this batch; with
               n_items=N, the SUM over this batch divided by N (accumulation-window weighting).
    """
    dev = next(model.parameters()).device
    logits = model(input_ids=batch["input_ids"].to(dev), attention_mask=batch["attention_mask"].to(dev)).logits
    target = batch["labels"].to(dev)
    loss = F.cross_entropy(logits[:, :-1].float().flatten(0, 1), target[:, 1:].flatten(), ignore_index=-100,
                           label_smoothing=label_smoothing, reduction="mean" if n_items is None else "sum")
    return loss if n_items is None else loss / n_items


_NO_DECAY = re.compile(r"bias|layernorm|rmsnorm|(?:^|\.)norm(?:$|\.)|_norm(?:$|\.)")


def make_optimizer(model, lr, weight_decay, betas, eps, optim="adamw_8bit"):
    """AdamW, 8-bit (`adamw_8bit`) or plain (`adamw_torch`); decay on weights only.

    Guarantee: same parameter groups and same fp32 embedding state as the HF Trainer's optimizer of
    that name, which is what the SFT parity check against the old trainer path pins.
    """
    in_layernorm = {id(p) for m in model.modules() if isinstance(m, torch.nn.LayerNorm) for p in m.parameters()}
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    skip = lambda n, p: id(p) in in_layernorm or _NO_DECAY.search(n.lower())
    groups = [g for g in (
        {"params": [p for n, p in named if not skip(n, p)], "weight_decay": weight_decay},
        {"params": [p for n, p in named if skip(n, p)], "weight_decay": 0.0},
    ) if g["params"]]
    if optim == "adamw_torch":
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)
    if optim != "adamw_8bit":
        raise ValueError(f"unknown optimizer {optim!r}: expected adamw_8bit or adamw_torch")
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, optim_bits=8)
    # Embedding tables keep 32-bit optimizer state: 8-bit state on a sparse, huge-vocab table is the
    # one place quantisation error is not averaged out (the Trainer does the same).
    mgr = bnb.optim.GlobalOptimManager.get_instance()
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            mgr.register_module_override(m, "weight", {"optim_bits": 32})
    return opt


def train_batches(model, batches, *, num_steps, lr, lr_scheduler, warmup_steps, weight_decay, max_grad_norm,
                  betas, eps, label_smoothing=0.0, scheduler_kwargs=None, seed=0, callbacks=(),
                  logging_steps=1, grad_accum=1, amp=False, optim="adamw_8bit", tok=None):
    """Train over `batches`: one optimizer step per `grad_accum` batches, for at most `num_steps` steps.

    Require : `batches` yields collated batch dicts (see collate); the schedule is built for num_steps.
    Guarantee: returns the per-optimizer-step training losses, in step order. A non-finite loss
               raises. Under amp, a step skipped by the loss scaler does not advance the schedule.
    Maintain: callbacks see the Trainer protocol (on_train_begin, on_step_begin, on_step_end,
              on_train_end) with state.global_step / max_steps / epoch.
    """
    assert num_steps > 0, "num_steps must be positive"
    from transformers import get_scheduler
    from transformers.trainer_callback import CallbackHandler, TrainerControl, TrainerState

    torch.manual_seed(seed)
    model.to("cuda" if torch.cuda.is_available() else "cpu")    # the HF Trainer did this for us
    model.gradient_checkpointing_enable()
    model.train()
    use_amp = bool(amp) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    opt = make_optimizer(model, lr, weight_decay, betas, eps, optim)
    sched = get_scheduler(lr_scheduler, opt, num_warmup_steps=warmup_steps, num_training_steps=num_steps,
                          scheduler_specific_kwargs=scheduler_kwargs or None)

    args = SimpleNamespace(max_grad_norm=max_grad_norm, logging_steps=logging_steps)
    state = TrainerState(max_steps=num_steps, global_step=0, epoch=0.0, num_train_epochs=1)
    control = TrainerControl()
    handler = CallbackHandler(list(callbacks), model, tok, opt, sched)
    handler.on_train_begin(args, state, control)

    def windows():
        buf = []
        for b in batches:
            buf.append(b)
            if len(buf) == grad_accum:
                yield buf
                buf = []
        if buf:
            yield buf

    losses = []
    for window in windows():
        handler.on_step_begin(args, state, control)
        n_items = sum(target_tokens(b) for b in window)
        total = 0.0
        for b in window:
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = calc_loss_batch(b, model, label_smoothing, n_items)
            (scaler.scale(loss) if scaler else loss).backward()
            total += float(loss.detach())
        if scaler:
            scaler.unscale_(opt)
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm))
        scale_before = scaler.get_scale() if scaler else None
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        if not scaler or scaler.get_scale() >= scale_before:        # a skipped (inf) step must not move the schedule
            sched.step()
        opt.zero_grad(set_to_none=True)
        losses.append(total)
        state.global_step = len(losses)
        state.epoch = state.global_step / num_steps
        if state.global_step % logging_steps == 0 or state.global_step == num_steps:
            print(f"  [train] step {state.global_step}/{num_steps} loss={total:.4f} grad_norm={gnorm:.3f} "
                  f"lr={sched.get_last_lr()[0]:.3g} batches={len(window)}", flush=True)
        handler.on_step_end(args, state, control)
        if not math.isfinite(total):
            raise FloatingPointError(f"non-finite loss {total} at step {state.global_step}/{num_steps}")
        if control.should_training_stop or state.global_step >= num_steps:
            break
    handler.on_train_end(args, state, control)
    return losses
