"""src/rlv2/train/rl/dpo_pass.py — IPO on judge preference pairs from the harvest sidecars.

Spec: rung R3 of the "break the 9" plan; results in docs/dispositions.md.

Offline preference optimization: no rollouts, no collapse risk. Each question with
both a judge-correct sample and a judge-failed sample becomes one (prompt, chosen,
rejected) pair; TRL DPOTrainer with IPO loss (the small-pair-count regularization),
LoRA adapters with the implicit PEFT reference model, merged and saved full so
score_arms can receipt it like any other arm.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys



def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]


def build_pairs(pool: list[dict], gens: dict[str, list[str]],
                scores: dict[str, list[float]], floor: float) -> list[dict]:
    """chosen >= floor > rejected, per question. Chosen mirrors harvest_sft's
    selection (best score, then shortest, never degenerate); rejected is the
    WORST-scoring nonempty sample — degenerate junk is a legitimate rejected
    (that is exactly the behavior to push probability away from)."""
    from rlv2.train.rl.harvest_sft import select_best, qtext
    from rlv2.train.rl.grpo import _is_degenerate
    pairs = []
    for r in pool:
        rid = str(r.get("id"))
        if rid not in gens or rid not in scores:
            continue
        samples = [{"completion": c, "score": s}
                   for c, s in zip(gens[rid], scores[rid])]
        chosen = select_best(samples, floor)
        if chosen is None:
            continue
        # Rejected = nonempty AND (sub-floor OR degenerate). Judge-fooled spam
        # scores ABOVE the floor, so a score-only filter never admits it — which
        # collapsed the first pair build to 4 pairs. Degenerate junk ranks first
        # as rejected (it is exactly the behavior to push away from), then the
        # worst-scoring honest miss.
        losers = [s for s in samples if s["completion"].strip()
                  and (s["score"] < floor or _is_degenerate(s["completion"]))]
        if not losers:
            continue
        rejected = min(losers,
                       key=lambda s: (not _is_degenerate(s["completion"]), s["score"]))
        pairs.append({"question": qtext(r), "chosen": chosen["completion"],
                      "rejected": rejected["completion"]})
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--pool", default="data/rl/r3_rl.jsonl")
    ap.add_argument("--harvest", required=True,
                    help="harvest out path; .gen.jsonl/.scored.jsonl sidecars read")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # COMPAT SHIM: trl 1.6.0's dpo_trainer imports MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES,
    # which the installed transformers no longer exports. It is only used to detect
    # vision models; ours is text-only, so an empty mapping is correct. Upgrading
    # transformers instead would invalidate the whole validated GRPO/qwen pipeline.
    import transformers.models.auto.modeling_auto as _auto
    if not hasattr(_auto, "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES"):
        _auto.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = {}
    from trl import DPOConfig, DPOTrainer
    from rlv2.eval import cpt_fewshot_lift as FS
    from rlv2.judge.reward import CORRECT_FLOOR

    pool = load_jsonl(args.pool)
    gens = {r["id"]: r["completions"] for r in load_jsonl(args.harvest + ".gen.jsonl")}
    scores = {r["id"]: r["scores"] for r in load_jsonl(args.harvest + ".scored.jsonl")}
    pairs = build_pairs(pool, gens, scores, CORRECT_FLOOR)
    print(f"[dpo] {len(pairs)} preference pairs (floor={CORRECT_FLOOR})", flush=True)
    if len(pairs) < 10:
        raise SystemExit(f"[dpo] only {len(pairs)} pairs — not enough to train")

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    shots = FS._squad_shots()
    rows = [{"prompt": tok.apply_chat_template(
                 FS._fewshot_messages(p["question"], shots), tokenize=False,
                 add_generation_prompt=True, enable_thinking=False),
             "chosen": p["chosen"], "rejected": p["rejected"]} for p in pairs]
    ds = Dataset.from_list(rows)

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.float16, trust_remote_code=True)
    # COMPAT SHIM 2: the trust_remote_code Qwen3_5 class predates the
    # `warnings_issued` dict newer Trainer paths read through PEFT's getattr chain.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    cfg = DPOConfig(
        output_dir=args.out + "_work",
        loss_type=["ipo"],
        beta=args.dpo_beta,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        logging_steps=4,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        max_length=512,
        max_prompt_length=352,
    )
    peft = LoraConfig(r=16, lora_alpha=64, lora_dropout=0.0,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=peft)
    trainer.train()
    print("[dpo] training done; merging adapter", flush=True)
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[dpo] merged model -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
