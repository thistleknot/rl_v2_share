"""src/rlv2/train/sft/loop.py -- one SFT epoch over pre-packed bins, on the shared training loop.

Book base and the list of what we add to it: src/rlv2/common/loop.py (Raschka, *LLMs from Scratch*
ch07 gpt_instruction_finetuning.py: calc_loss_batch, custom_collate_fn, the train loop).

Notes: docs/PROVENANCE.md (slide -> notes -> code). No governing spec in this repo; basis is the author-approved
restructure of the SFT loop onto the book loop (2026-10).

SFT-specific: a batch is one pre-packed bin from `src/rlv2/train/sft/manifold_sampler.py` (a token budget, so
its record count varies), padded on the tokenizer's side, one optimizer step per bin, fp32 with
8-bit AdamW and no loss scaler.
"""
from rlv2.common.loop import calc_loss_batch, collate, make_optimizer, train_batches  # noqa: F401  (re-exported)


def train_bins(model, bins, tok, **kw):
    """Train one epoch: one optimizer step per bin, schedule spanning all bins.

    Require : bins is a non-empty list of lists of records (see rlv2.common.loop.collate).
    Guarantee: returns the per-step training losses; `kw` is train_batches' hyperparameters.
    """
    assert bins, "bins must be non-empty"
    batches = (collate([{k: v for k, v in r.items() if not k.startswith("_")} for r in recs], tok)
               for recs in bins)
    return train_batches(model, batches, num_steps=len(bins), tok=tok, **kw)
