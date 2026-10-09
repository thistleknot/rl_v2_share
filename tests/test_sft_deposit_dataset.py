"""The unseen-splits SFT deposit set is leak-free and trainer-consumable.

NO GOVERNING SPEC row; basis: operator instruction 2026-09-03 ("I have 240 x 3 data...
proceed") reopening PAST 9/16 Branch B after T5's acceptance failure. Provenance for
data/rl/sft_r3_deposit_cpt_rl.jsonl (strict JSONL cannot carry a comment): 480 questions
from r3_cpt.jsonl + r3_rl.jsonl rendered via build_r3_datasets.to_sft_item — the CPT and
RL splits, never seen by the model as supervised QA — plus the 240 rows of
sft_r3_skills.jsonl as replay (standing replay law). See playbook.md "PAST 9/16".

Each property below broke, or nearly broke, a run before it was checked:
- eval leakage would turn every downstream number into a train score;
- an empty `gold` trains on an empty target SILENTLY (vibe_sft reads r.get("gold"));
- a `domain` outside the cross-hatch canon draws ratio 0.0 and kills the sampler
  after the model is already loaded.
"""
from __future__ import annotations

import io
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPOSIT = os.path.join(_ROOT, "data", "rl", "sft_r3_deposit_cpt_rl.jsonl")
EVAL = os.path.join(_ROOT, "data", "rl", "r3_eval.jsonl")
SEEN = os.path.join(_ROOT, "data", "rl", "sft_r3_skills.jsonl")


def _load(path):
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]


def test_deposit_is_720_rows_and_majority_unseen():
    rows = _load(DEPOSIT)
    assert len(rows) == 720, "480 unseen (cpt+rl splits) + 240 replay"
    seen = {r["prompt"].strip() for r in _load(SEEN)}
    unseen = [r for r in rows if r["prompt"].strip() not in seen]
    assert len(unseen) == 480, (
        "the deposit's point is NEW questions; %d unseen found" % len(unseen))


def test_no_eval_question_leaks_into_the_deposit():
    evq = {r["question"].strip().lower() for r in _load(EVAL)}
    leaks = [r["prompt"][:60] for r in _load(DEPOSIT)
             if r["prompt"].strip().lower() in evq]
    assert not leaks, "frozen-eval questions in training data: %r" % leaks


DEPOSIT_HALF = os.path.join(_ROOT, "data", "rl", "sft_r3_deposit_half.jsonl")
RL_HALF = os.path.join(_ROOT, "data", "rl", "rl_mix_half.jsonl")


def test_sft_deposit_and_rl_prompt_halves_are_mutually_exclusive():
    """The SFT-deposit data and the RL-prompt data must be DISJOINT partitions of the
    720-row pool, not the same 720 rows reused at both stages.

    Caught by the operator 2026-09-03: the prior deposit trained on the full 720-row
    mix (data/rl/sft_r3_deposit_cpt_rl.jsonl), and the subsequent RL pass sampled
    prompts from data/rl/rl_mix_all_splits.jsonl — the SAME 720 rows, including the
    r3_rl split the deposit had already trained on. That is one dataset touched twice,
    not two independent experience streams, and it confounds "did the deposit help"
    with "did the RL pass just re-see deposit questions as prompts."
    """
    dep = {r["prompt"].strip() for r in _load(DEPOSIT_HALF)}
    rl = {r["question"].strip() for r in _load(RL_HALF)}
    overlap = dep & rl
    assert not overlap, "SFT deposit and RL prompt pool share %d questions" % len(overlap)


def test_the_two_halves_together_cover_the_original_720_pool():
    all_q = {r["question"].strip() for r in _load(
        os.path.join(_ROOT, "data", "rl", "rl_mix_all_splits.jsonl"))}
    dep = {r["prompt"].strip() for r in _load(DEPOSIT_HALF)}
    rl = {r["question"].strip() for r in _load(RL_HALF)}
    assert dep | rl == all_q, "the split dropped or duplicated rows from the source pool"
    assert 340 <= len(dep) <= 380 and 340 <= len(rl) <= 380, (
        "expected roughly even halves (~360 each), got %d / %d" % (len(dep), len(rl)))


def test_rl_mix_of_all_splits_is_720_unique_and_leak_free():
    """data/rl/rl_mix_all_splits.jsonl — MGPO prompt mix for the post-deposit RL pass.

    All three phase splits concatenated (raw split schema): after the deposit, the
    model's environment is all 720 questions, so MaxEnt must be able to select
    spanning prompts from any of them. Dupes would double-weight a prompt in the
    bootstrap cohort; an eval leak would train on the frozen instrument.
    """
    mix = _load(os.path.join(_ROOT, "data", "rl", "rl_mix_all_splits.jsonl"))
    qs = [r["question"].strip() for r in mix]
    assert len(qs) == 720 and len(set(qs)) == 720, "expected 720 unique questions"
    evq = {r["question"].strip().lower() for r in _load(EVAL)}
    assert not any(q.lower() in evq for q in qs), "frozen-eval question in the RL mix"
    assert all((r.get("gold") or "").strip() for r in mix), "row with empty gold"


def test_every_row_is_trainer_consumable():
    canon_domains = {"Math", "Code", "STEM", "Instruction"}
    for i, r in enumerate(_load(DEPOSIT)):
        assert (r.get("gold") or "").strip(), "row %d: empty gold trains on nothing" % i
        assert (r.get("prompt") or "").strip(), "row %d: empty prompt" % i
        assert r.get("domain") in canon_domains, (
            "row %d: domain %r is not a cross_hatch_canon key; the manifold sampler "
            "draws zero from it" % (i, r.get("domain")))
