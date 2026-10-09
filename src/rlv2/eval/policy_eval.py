"""
policy_eval.py — Shared multi-sample generation and scoring.

Used by:
  score_anchor_set.py     G=4 at training temperature → anchor baseline
  generate_rubric.py      4 sampler configs → rubric calibration
  train_outer_loop.py     greedy per domain → outer-loop eval
"""
import random
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlv2.common.config import MODEL_NAME, SYSTEM_PROMPT
from rlv2.eval.half_epoch_eval import _stop_ids


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# A turn boundary that survived decoding. `skip_special_tokens=True` DELETES <|im_end|> and
# <|im_start|> from the text but leaves the role word behind, so an unstopped generation reads
# as "...answer. assistant   <second answer>" with nothing marking the seam.
_TURN_LEAK = re.compile(r"<\|im_(?:end|start)\|>"
                        r"|(?<![\w-])(?:assistant|user|system)(?![\w-])(?:\s{2,}|[\r\n])")


def _cut_at_turn(text: str) -> str:
    """Truncate at the first turn boundary. Belt-and-braces behind the stop-token fix.

    Stop ids end generation at <|im_end|>, which is the real fix. This catches the residue
    when a checkpoint's tokenizer lacks those tokens or a sampler emits the role word as
    plain text -- in either case everything after the seam belongs to a different turn and
    scoring it means scoring a transcript, not an answer.
    """
    m = _TURN_LEAK.search(text)
    return (text[:m.start()] if m else text).strip()


def fmt_prompt(question: str, context: str, tokenizer) -> str:
    user = f"Question: {question}"
    if context:
        user += f"\n\nContext: {context}"
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _generate_one(model, tokenizer, prompt: str, sampler: dict, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # WITHOUT eos_token_id THE MODEL RUNS PAST ITS ANSWER INTO THE NEXT TURN.
    # Measured 2026-08-28: 48/48 rubric generations contained a leaked turn marker and every
    # one ran to the max_new_tokens cap, so the judge was scoring a two-turn transcript. Same
    # defect class already fixed in src/rlv2/eval/half_epoch_eval.py -- that helper is imported, not
    # reimplemented.
    gen_kwargs = dict(max_new_tokens=max_new_tokens, eos_token_id=_stop_ids(tokenizer),
                      pad_token_id=tokenizer.pad_token_id, **sampler)
    if sampler.get("temperature", 0.0) > 0.0:
        gen_kwargs.setdefault("do_sample", True)
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    return _cut_at_turn(_strip_think(
        tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    ))


def generate_greedy_batch(model, tokenizer, prompts: list[str], max_new_tokens: int = 256,
                          batch_size: int = 16) -> list[str]:
    """Greedy generation over many prompts, batched. Greedy is deterministic, so batching is
    behavior-identical to looping _generate_one — just ~13x faster (single-seq gen on this model
    is kernel-launch-bound at ~6 tok/s; batch x16 ~80 tok/s). Left-padding so the generated
    suffix aligns per row. No min_new_tokens (would suppress EOS and corrupt abstain detection)."""
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outs: list[str] = []
    try:
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            inp = tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                                     use_cache=True, eos_token_id=_stop_ids(tokenizer),
                                     pad_token_id=tokenizer.pad_token_id)
            start = inp["input_ids"].shape[1]
            for row in gen:
                outs.append(_cut_at_turn(
                    _strip_think(tokenizer.decode(row[start:], skip_special_tokens=True))))
    finally:
        tokenizer.padding_side = prev_side
    return outs


def load_adapter(adapter_path: str):
    """Load base model + adapter. Returns (model, tokenizer). Caller must del + empty_cache when done."""
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, trust_remote_code=True)  # Turing: native fp16; bf16 emulated/slow
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer


def multi_sample_score(
    records: list[dict],
    adapter_path: str,
    sampler_configs: list[dict],
    judge_model: str,
    max_new_tokens: int = 256,
    seed: int = 0,
    judge_batch_size: int = 1,   # ISOLATED: 1 pair per judge call (batched prompts contaminate scores)
) -> list[dict]:
    """
    For each record generate one completion per sampler config, then score all with judge.

    Returns one result dict per record:
      question, gold, context, domain,
      completions: [str]   — one per sampler, in sampler_configs order
      scores:      [float] — one per sampler, in sampler_configs order
      mean_score:  float
      std_score:   float
    """
    from rlv2.judge.reward import llm_judge_flat

    rng = random.Random(seed)
    model, tokenizer = load_adapter(adapter_path)

    _only = sampler_configs[0] if len(sampler_configs) == 1 else None
    _greedy = (_only is not None and not _only.get("do_sample", False)
               and _only.get("temperature", 0.0) <= 0.0)
    if _greedy:
        # Fast path: a single deterministic (greedy) sampler -> batch ALL records at once (~13x vs
        # serial _generate_one). Used by eval_adapter_by_class every outer-loop round.
        prompts = [fmt_prompt(rec.get("question", ""), rec.get("context", ""), tokenizer) for rec in records]
        comps = generate_greedy_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens, batch_size=16)
        all_completions = [[c] for c in comps]
        print(f"  batched {len(records)} greedy completions")
    else:
        all_completions = []
        for i, rec in enumerate(records):
            prompt = fmt_prompt(rec.get("question", ""), rec.get("context", ""), tokenizer)
            order = list(range(len(sampler_configs)))
            rng.shuffle(order)
            comps = [""] * len(sampler_configs)
            for idx in order:
                comps[idx] = _generate_one(model, tokenizer, prompt, sampler_configs[idx], max_new_tokens)
            all_completions.append(comps)
            print(f"  [{i+1}/{len(records)}] generated {len(sampler_configs)} completions")

    del model
    torch.cuda.empty_cache()
    print("Generation complete. Model freed.")

    # Flatten to (record, sampler) pairs for judge
    flat_pairs, pair_map = [], []
    for rec_idx, (rec, comps) in enumerate(zip(records, all_completions)):
        for s_idx, comp in enumerate(comps):
            flat_pairs.append({
                "question": rec.get("question", ""),
                "gold":     rec.get("answer", ""),
                "context":  rec.get("context", ""),
                "generated": comp,
                "domain":    rec.get("domain", ""),
                "source":    rec.get("source", ""),
            })
            pair_map.append((rec_idx, s_idx))

    print(f"Scoring {len(flat_pairs)} pairs with {judge_model} (2 judge requests in flight)...")
    from rlv2.judge.reward import llm_judge_flat_many
    chunks = [flat_pairs[i:i + judge_batch_size] for i in range(0, len(flat_pairs), judge_batch_size)]
    flat_scores: list[float] = []
    for chunk_scores in llm_judge_flat_many(chunks, model=judge_model):
        flat_scores.extend(chunk_scores)

    # Apply composite reward: verify-gate x continuous quality
    from rlv2.judge.reward import verify_verdict, compose_reward
    composite_scores = []
    for pair in flat_pairs:
        quality_idx = len(composite_scores)
        v = verify_verdict(pair["domain"], pair["generated"], pair["gold"], pair.get("source", ""))
        composite_scores.append(compose_reward(v, flat_scores[quality_idx]))

    score_matrix = [[0.0] * len(sampler_configs) for _ in records]
    for pair_idx, (rec_idx, s_idx) in enumerate(pair_map):
        score_matrix[rec_idx][s_idx] = composite_scores[pair_idx]

    results = []
    for rec, comps, scores in zip(records, all_completions, score_matrix):
        mean_s = sum(scores) / len(scores) if scores else 0.0
        std_s  = (sum((s - mean_s) ** 2 for s in scores) / len(scores)) ** 0.5 if scores else 0.0
        results.append({
            "question":   rec.get("question", ""),
            "gold":       rec.get("answer", ""),
            "context":    rec.get("context", ""),
            "domain":     rec.get("domain", "unknown"),
            "completions": comps,
            "scores":      [round(s, 4) for s in scores],
            "mean_score":  round(mean_s, 4),
            "std_score":   round(std_s, 4),
        })
    return results
