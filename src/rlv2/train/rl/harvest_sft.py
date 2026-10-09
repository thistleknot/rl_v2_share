"""src/rlv2/train/rl/harvest_sft.py — harvest a checkpoint's SAMPLED successes into an SFT dataset.

Spec: rung R2 of the "break the 9" plan; results in docs/dispositions.md.

The ReST/RFT cycle (DeepSeek-R1's SFT<->RL alternation): sample k completions per
training question, judge them, keep the best CORRECT one, fine-tune on those — the
knowledge that exists only in the sampling distribution gets written into the argmax.
Measured motivation: past9_x3 answers 52% under sampling but argmaxes 6/16.

Guards:
- The 16-item eval is NEVER harvested (tripwire raises before any GPU spend).
- Judging goes through reward.llm_judge_flat_many — the empty-answer guard applies.
- Resumable before launch (standing law): generations and scores append to sidecar
  files keyed by question id; rerunning skips completed work.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys


EVAL_FILE = "data/rl/r3_eval_canonical.jsonl"


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]


def qtext(row: dict) -> str:
    return row.get("question") or row.get("prompt") or ""


def assert_no_eval_leakage(pool: list[dict], eval_rows: list[dict]) -> None:
    """Tripwire (fails closed): harvesting an eval question would fine-tune on the
    instrument. Raises before any generation happens."""
    eval_qs = {qtext(r).strip() for r in eval_rows if qtext(r).strip()}
    leaked = [qtext(r)[:80] for r in pool if qtext(r).strip() in eval_qs]
    if leaked:
        raise AssertionError(
            f"EVAL LEAKAGE: {len(leaked)} pool question(s) match the frozen eval; "
            f"first: {leaked[0]!r}")


def select_best(samples: list[dict], floor: float) -> dict | None:
    """Best CORRECT completion for one question: highest judge score wins; ties go
    to the SHORTEST completion (Long2Short — brevity among correct answers only).
    Returns None when nothing clears the floor (the question stays unharvested;
    silence and near-misses are not training data).

    DEGENERATE COMPLETIONS ARE NEVER TRAINING DATA (spot-read 2026-09-05): the
    first harvest's judge passed digit-spam like '1.0.0.1.0.1...' at >=0.70 on 6 of
    10 sampled rows. The RL reward deliberately lets the judge see degenerate text
    (health-signal decision, mgpo reward docstring); the HARVEST must not — junk
    that fools the judge once becomes junk the model is trained to emit. Reuses
    mgpo._is_degenerate (the [degen-alarm] incumbent, Req 38)."""
    from rlv2.train.rl.grpo import _is_degenerate
    correct = [s for s in samples
               if s["score"] >= floor and s["completion"].strip()
               and not _is_degenerate(s["completion"])]
    if not correct:
        return None
    return sorted(correct, key=lambda s: (-s["score"], len(s["completion"])))[0]


def to_sft_row(pool_row: dict, completion: str) -> dict:
    """Deposit-schema row (matches data/rl/sft_r3_deposit_half.jsonl, the format
    vibe_sft trains on): the harvested completion becomes the supervised answer."""
    return {
        "prompt": qtext(pool_row),
        "gold": pool_row.get("gold", ""),
        "answer": completion,
        "context": pool_row.get("context", ""),
        "domain": pool_row.get("category", pool_row.get("domain", "")),
        "skill": pool_row.get("source_skill", pool_row.get("skill", "")),
    }


def _generate(model_path: str, pool: list[dict], k: int, temperature: float,
              gen_path: str) -> dict[str, list[str]]:
    """k sampled completions per question, fewshot chat-template parity with the
    trainer/eval. Appends to gen_path per question; resumes by id."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rlv2.eval import cpt_fewshot_lift as FS

    done: dict[str, list[str]] = {}
    if os.path.exists(gen_path):
        for r in load_jsonl(gen_path):
            done[r["id"]] = r["completions"]
    todo = [r for r in pool if str(r.get("id")) not in done]
    print(f"[harvest-gen] {len(done)} cached, {len(todo)} to generate", flush=True)
    if not todo:
        return done

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, trust_remote_code=True).cuda().eval()
    shots = FS._squad_shots()
    with io.open(gen_path, "a", encoding="utf-8") as out:
        for i, r in enumerate(todo):
            pt = tok.apply_chat_template(
                FS._fewshot_messages(qtext(r), shots), tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            ids = tok(pt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(**ids, max_new_tokens=160, do_sample=True,
                                     temperature=temperature, top_p=0.95,
                                     num_return_sequences=k,
                                     pad_token_id=tok.eos_token_id)
            comps = [tok.decode(g[ids["input_ids"].shape[1]:], skip_special_tokens=True)
                     for g in gen]
            done[str(r.get("id"))] = comps
            out.write(json.dumps({"id": str(r.get("id")), "completions": comps}) + "\n")
            out.flush()
            if (i + 1) % 10 == 0:
                print(f"[harvest-gen] {i + 1}/{len(todo)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return done


def _judge(pool: list[dict], gens: dict[str, list[str]], score_path: str,
           chunk_size: int = 8) -> dict[str, list[float]]:
    """Guarded judging (empty answers score 0.0 without the judge). Appends per
    question to score_path; resumes by id."""
    from rlv2.judge.reward import llm_judge_flat_many

    scored: dict[str, list[float]] = {}
    if os.path.exists(score_path):
        for r in load_jsonl(score_path):
            scored[r["id"]] = r["scores"]
    todo = [r for r in pool if str(r.get("id")) in gens
            and str(r.get("id")) not in scored]
    print(f"[harvest-judge] {len(scored)} cached, {len(todo)} to judge", flush=True)
    with io.open(score_path, "a", encoding="utf-8") as out:
        for i, r in enumerate(todo):
            rid = str(r.get("id"))
            pairs = [{"question": qtext(r), "gold": r.get("gold", ""),
                      "context": r.get("context", ""), "generated": c}
                     for c in gens[rid]]
            chunks = [pairs[j:j + chunk_size] for j in range(0, len(pairs), chunk_size)]
            scores = [s for chunk in llm_judge_flat_many(chunks) for s in chunk]
            scored[rid] = scores
            out.write(json.dumps({"id": rid, "scores": scores}) + "\n")
            out.flush()
            print(f"[harvest-judge] {i + 1}/{len(todo)} done", flush=True)
    return scored


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint to sample from")
    ap.add_argument("--pool", default="data/rl/r3_rl.jsonl")
    ap.add_argument("--out", required=True, help="harvested SFT jsonl")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    from rlv2.judge.reward import CORRECT_FLOOR

    pool = load_jsonl(args.pool)
    assert_no_eval_leakage(pool, load_jsonl(EVAL_FILE))
    print(f"[harvest] pool={len(pool)} | eval tripwire clean | floor={CORRECT_FLOOR}",
          flush=True)

    gen_path = args.out + ".gen.jsonl"
    score_path = args.out + ".scored.jsonl"
    gens = _generate(args.model, pool, args.k, args.temperature, gen_path)
    scores = _judge(pool, gens, score_path)

    rows, n_correct_qs = [], 0
    for r in pool:
        rid = str(r.get("id"))
        if rid not in gens or rid not in scores:
            continue
        samples = [{"completion": c, "score": s}
                   for c, s in zip(gens[rid], scores[rid])]
        best = select_best(samples, CORRECT_FLOOR)
        if best is not None:
            n_correct_qs += 1
            rows.append(to_sft_row(r, best["completion"]))
    with io.open(args.out, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[harvest] {n_correct_qs}/{len(pool)} questions yielded a correct sample "
          f"-> {len(rows)} SFT rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
