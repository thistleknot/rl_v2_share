"""Regression tests for four defects found while making the R3 SFT lane run end-to-end.

Spec: playbook.md B-2/B-3 (SFT HPO over the nine optimizer settings). NO GOVERNING SPEC for the
checkpoint-save path -- Basis: operator instruction 2026-08-31 ("solve wrinkles, not raise
them... they better be proper fixes, don't hack a fix"). Promote to a REQ before anything
depends on it.

Each defect below produced a PLAUSIBLE RESULT rather than an error, which is the only reason
they survived as long as they did:

  1. The manifold sampler read `domain`; R3 records carry the same values under `category`.
     Every R3 row mapped to the cell "R3|unknown|Qn", every target ratio was 0.0, and the
     sampler drew ZERO records -- surfacing three frames later as "bins must be non-empty".
  2. Drawing zero was not itself an error, so the diagnosis lived in the wrong module.
  3. FLOOR_WEIGHT was applied and then immediately overwritten by the raw delta, which is
     <= 0 for any at-target cell and is not a valid weight.
  4. The checkpoint save cast the LIVE model with m.half(). Calling .float() does not undo
     that -- it round-trips every parameter through fp16 and the next epoch trains from the
     degraded weights. This repo already recorded that failure once (T0.3).
"""
from __future__ import annotations
import os as _o, sys as _s

import json
import pathlib
import random

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SFT = (ROOT / "src" / "rlv2" / "train" / "sft" / "train.py").read_text(encoding="utf-8")
SAMPLER_SRC = (ROOT / "src" / "rlv2" / "train" / "sft" / "manifold_sampler.py").read_text(encoding="utf-8")


def _code(src: str) -> str:
    """Comments are not code.

    Three tests in the first draft of this file matched their own explanatory prose inside the
    comments that DOCUMENT each fix, and failed against correct code. `test_vibe_sft_knobs.py`
    already carries this lesson; it is repeated here because I made the same mistake again.
    """
    return "\n".join(line.split("#")[0] for line in src.splitlines())


def _r3_pool():
    """Real R3 SFT rows with plausible token lengths.

    Integration-first (Article V): the defect was a field-name mismatch against THIS data, so a
    synthetic fixture would have been written against whichever name I already believed in and
    would not have caught it.
    """
    path = ROOT / "data" / "rl" / "r3_sft.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    rng = random.Random(1)
    for r in rows:
        r["input_ids"] = [0] * rng.choice([94, 106, 122, 162])
    return rows


def _canon_ratios():
    canon = json.loads(
        (ROOT / "data" / "rl" / "cross_hatch_canon.json").read_text(encoding="utf-8"))
    ratios = {k: v for k, v in canon["eval"].items() if not k.startswith("_")}
    total = sum(ratios.values())
    return {k: v / total for k, v in ratios.items()}


def _sampler(n_bins=12, pool=None):
    from rlv2.train.sft.manifold_sampler import ManifoldSampler
    return ManifoldSampler(pool=pool if pool is not None else _r3_pool(),
                           target_ratios=_canon_ratios(), target_tokens=896,
                           n_bins=n_bins, max_len=448)


# --- 1. the category/domain alias -------------------------------------------------------------

def test_r3_records_map_to_real_domain_cells_not_unknown():
    s = _sampler()
    unknown = [c for c in s.populated_cells if "|unknown|" in c]
    assert not unknown, "R3 rows still fall into 'unknown' cells: %s" % sorted(unknown)[:4]


def test_every_populated_cell_has_a_nonzero_target():
    """A cell with target 0.0 is skipped outright, so "every cell at target 0" and "pool
    exhausted" were indistinguishable from inside the draw loop."""
    s = _sampler()
    zero = [c for c in s.populated_cells if s._cell_target(c) <= 0]
    assert not zero, "cells with no canon target: %s" % sorted(zero)[:4]


def test_the_alias_is_read_from_category_when_domain_is_absent():
    from rlv2.train.sft.manifold_sampler import ManifoldSampler
    rec = {"_regime": "R3", "category": "Math", "input_ids": [0] * 100}
    s = _sampler()
    assert ManifoldSampler._cell_of(s, rec).startswith("R3|Math|")


def test_domain_still_wins_when_both_names_are_present():
    """The alias must be a fallback, not a replacement -- older lanes write `domain` and their
    cell keys must not move."""
    from rlv2.train.sft.manifold_sampler import ManifoldSampler
    rec = {"_regime": "R3", "domain": "Code", "category": "Math", "input_ids": [0] * 100}
    s = _sampler()
    assert ManifoldSampler._cell_of(s, rec).startswith("R3|Code|")


def test_sampler_actually_draws_records_for_the_r3_lane():
    """The end-to-end symptom: 0 records, 0 bins, and a ValueError from the trainer."""
    s = _sampler()
    bins = s.sample_epoch(seed=7)
    assert sum(len(b) for b in bins) > 0, "sampler drew nothing for R3"


# --- 2. zero draws must be diagnosed where they happen ----------------------------------------

def test_drawing_zero_raises_naming_the_cells_and_the_targets():
    rows = _r3_pool()[:40]
    for r in rows:
        r.pop("category", None)          # recreate the original mismatch
        r.pop("domain", None)
    s = _sampler(pool=rows)
    with pytest.raises(ValueError) as ei:
        s.sample_epoch(seed=7)
    msg = str(ei.value)
    assert "drew 0 records" in msg
    assert "unknown" in msg, "the error must show the offending cell keys"


# --- 3. the floor weight must survive ---------------------------------------------------------

def test_floor_weight_is_not_overwritten_by_the_raw_delta():
    """weights[cell] = max(delta, FLOOR_WEIGHT) followed by weights[cell] = delta made the
    floor dead and let non-positive weights reach rng.choices."""
    body = _code(SAMPLER_SRC)
    body = body[body.index("def sample_epoch"):]
    body = body[:body.index("\n    def ")]
    assert body.count("weights[cell] = ") == 1, "the floor is assigned then overwritten again"
    assert "max(delta, FLOOR_WEIGHT)" in body


def test_cell_target_is_defined_once():
    """Two definitions existed; the second silently shadowed the first, so the class read as
    though the score axis were ignored."""
    assert _code(SAMPLER_SRC).count("def _cell_target") == 1


# --- 4. the save must not mutate the live model -----------------------------------------------

def test_save_path_does_not_call_half_on_the_live_model():
    """Asserted on the AST, not the text.

    A string search also matches the docstring of `_fp16_state_dict`, which EXPLAINS why
    `model.half()` is wrong -- the same "asserting on prose" trap as `_code` above, one level
    deeper. Only a real call node counts.
    """
    import ast as _ast
    calls = [n for n in _ast.walk(_ast.parse(SFT))
             if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Attribute)
             and n.func.attr == "half"]
    assert not calls, (
        "in-place .half() at line(s) %s degrades the weights the next epoch trains on"
        % [n.lineno for n in calls])


def test_state_dict_is_cast_out_of_place_and_leaves_the_model_in_fp32():
    """The behavioural half of the check above, on a real module rather than by reading source."""
    import torch
    from torch import nn
    m = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 4))
    before = [p.detach().clone() for p in m.parameters()]
    sd = {k: v.detach().to(torch.float16).cpu() for k, v in m.state_dict().items()}
    assert all(v.dtype is torch.float16 for v in sd.values()), "state dict is not fp16"
    assert all(p.dtype is torch.float32 for p in m.parameters()), "live model was cast"
    for b, p in zip(before, m.parameters()):
        assert torch.equal(b, p), "live weights changed during save"


def test_out_is_copied_from_the_final_epoch_not_serialised_again():
    assert "copytree" in _code(SFT), "OUT should be a copy of the last epoch checkpoint"
    assert "OUT = copy of final epoch" in SFT


# --- 5. an epoch must be a meaningful slice of the pool ---------------------------------------

def test_bins_per_epoch_default_covers_about_a_third_of_the_pool():
    """At the old n_bins=4 an "epoch" was ~11% of the data, so three epochs saw a third of the
    pool once and the epoch curve measured sampling noise as much as learning."""
    default = None
    for line in SFT.splitlines():
        if "--bins-per-epoch" in line and "default=" in line:
            default = int(line.split("default=")[1].split(",")[0])
    assert default == 12, "default bins-per-epoch changed; re-measure coverage"
    pool = _r3_pool()
    drawn = sum(len(b) for b in _sampler(n_bins=default, pool=pool).sample_epoch(seed=7))
    frac = drawn / len(pool)
    assert 0.25 <= frac <= 0.42, "an epoch covers %.0f%% of the pool, expected ~1/3" % (100 * frac)


def test_three_epochs_approximate_one_full_pass():
    s = _sampler()
    total = sum(sum(len(b) for b in s.sample_epoch(seed=7 + e)) for e in range(3))
    pool_n = len(_r3_pool())
    assert 0.85 <= total / pool_n <= 1.15, (
        "3-epoch coverage %.0f%% of pool" % (100 * total / pool_n))


# --- 6. no artifacts that nothing reads -------------------------------------------------------

def test_eval_split_files_are_written_only_when_something_reads_them():
    """Six split files were written next to every checkpoint regardless; the `_end` half had no
    reader at all. A run looked like it had scored itself when it had not."""
    code = _code(SFT)
    assert "_eval_end_files" not in code, "the unread _end split is still being produced"
    i = code.index("_eval_mid_files: list[str] = []")
    assert "a.evals_per_epoch > 0" in code[i:i + 200], "mid splits are not gated on the flag"


# --- 7. tied weights must not be written twice -------------------------------------------------

def test_fp16_state_dict_drops_the_tied_lm_head():
    """`save_pretrained` drops tied weights via `_tied_weights_keys`, but ONLY on the path that
    builds the state dict itself. Passing `state_dict=` bypasses it, so the checkpoint carried
    both `lm_head.weight` and the embedding it is tied to -- 248320 x 1024 written twice, 508 MB
    of a 2.01 GB file, on the operation that dominates a short training run.

    Verified end-to-end 2026-08-31: 2.01 GB -> 1.50 GB, lm_head re-tied on load, and greedy
    generation byte-identical to the checkpoint that carried the duplicate.
    """
    import ast as _ast
    import types
    import torch

    fn = [n for n in _ast.parse(SFT).body
          if isinstance(n, _ast.FunctionDef) and n.name == "_fp16_state_dict"]
    assert fn, "_fp16_state_dict is gone; both save sites must share one cast"
    mod = types.ModuleType("h")
    mod.torch = torch
    exec(compile(_ast.Module(body=fn, type_ignores=[]), "h", "exec"), mod.__dict__)

    class _Cfg:
        tie_word_embeddings = True

    class _Fake:
        config = _Cfg()
        _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

        def state_dict(self):
            return {"model.embed_tokens.weight": torch.randn(8, 4),
                    "lm_head.weight": torch.randn(8, 4),
                    "model.layers.0.mlp.weight": torch.randn(4, 4)}

    sd = mod._fp16_state_dict(_Fake())
    assert "lm_head.weight" not in sd, "the tied duplicate is still being written"
    assert "model.embed_tokens.weight" in sd, "the real embedding must survive"
    assert all(v.dtype is torch.float16 for v in sd.values())


def test_untied_models_keep_every_key():
    """The drop must be conditional. An untied model needs its own lm_head."""
    import ast as _ast
    import types
    import torch

    fn = [n for n in _ast.parse(SFT).body
          if isinstance(n, _ast.FunctionDef) and n.name == "_fp16_state_dict"]
    mod = types.ModuleType("h")
    mod.torch = torch
    exec(compile(_ast.Module(body=fn, type_ignores=[]), "h", "exec"), mod.__dict__)

    class _Cfg:
        tie_word_embeddings = False

    class _Fake:
        config = _Cfg()

        def state_dict(self):
            return {"lm_head.weight": torch.randn(8, 4), "embed.weight": torch.randn(8, 4)}

    sd = mod._fp16_state_dict(_Fake())
    assert "lm_head.weight" in sd, "an untied lm_head was dropped -- the checkpoint is incomplete"
