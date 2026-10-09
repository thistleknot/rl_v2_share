"""
Stage 5: Synthetic QA generation with regime re-labeling.
For each seed record:
  1. RAG-retrieve fetch_k=144 candidate contexts via ChromaDB
  2. Softmax temperature-sample 1–3 new contexts (diversity sweep across 3 sampler configs)
  3. For ~30% of pairs, deliberately pick a low-similarity context → manufactures R2 examples
  4. Generate a completion for each (question, new_context) pair
  5. Re-label regime based on the *retrieved* context (not the original label)
SQLite checkpoint: a crash mid-run resumes without re-running LLM calls.

Run:  python generate_synthetic_qa.py
"""
import json
import math
import random
import sqlite3
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from rlv2.common.config import (
    DATA_LABELED, DATA_SYNTHETIC, SFT_CHECKPOINT, MODEL_NAME, FETCH_K, SYSTEM_PROMPT,
)
from rlv2.data.index_chroma import query_similar, get_collection, build_index
from rlv2.data.label_regimes import label_regime

random.seed(42)
np.random.seed(42)

DB_PATH = DATA_SYNTHETIC / "progress.db"
OUTPUT_PATH = DATA_SYNTHETIC / "all.jsonl"

# 3 sampler configs for diversity (not the GRPO group)
SAMPLER_CONFIGS = [
    {"temperature": 0.7, "top_p": 0.90, "do_sample": True},
    {"temperature": 1.0, "top_p": 0.95, "do_sample": True},
    {"temperature": 0.4, "top_p": 0.80, "do_sample": True},
]

R2_MANUFACTURE_PROB = 0.30   # fraction of pairs where we pick a mismatched context
PAIRS_PER_SEED = 3           # new (q, c) pairs to derive per seed record
MAX_NEW_TOKENS = 256


def _softmax_sample(similarities: list[float], k: int, temperature: float = 1.0) -> list[int]:
    """Sample k indices from similarity scores using softmax with temperature."""
    sims = np.array(similarities, dtype=np.float32)
    logits = sims / max(temperature, 1e-6)
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    k = min(k, len(probs))
    return list(np.random.choice(len(probs), size=k, replace=False, p=probs))


def _build_prompt(question: str, context: str) -> str:
    user_content = f"Question: {question}"
    if context:
        user_content += f"\n\nContext: {context}"
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _generate(prompt: str, model, tokenizer, sampler_cfg: dict) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            **sampler_cfg,
        )
    # Decode only newly generated tokens
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def _setup_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS generated (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            seed_id INTEGER,
            question TEXT,
            context TEXT,
            answer TEXT,
            regime TEXT,
            sampler_idx INTEGER,
            reward REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_seed ON generated(seed_id)")
    conn.commit()


def _already_done(conn: sqlite3.Connection, seed_id: int) -> bool:
    cur = conn.execute(
        "SELECT COUNT(*) FROM generated WHERE seed_id = ?", (seed_id,)
    )
    return cur.fetchone()[0] >= PAIRS_PER_SEED * len(SAMPLER_CONFIGS)


def _save_records(conn: sqlite3.Connection, records: list[dict]) -> None:
    conn.executemany(
        "INSERT INTO generated (seed_id, question, context, answer, regime, sampler_idx, reward) "
        "VALUES (:seed_id, :question, :context, :answer, :regime, :sampler_idx, :reward)",
        records,
    )
    conn.commit()


def generate_for_seed(
    seed_id: int,
    seed: dict,
    model,
    tokenizer,
    candidates: list[dict],
    n: int = 3,
    pairs_per_seed: int = PAIRS_PER_SEED,
    sampler_cfgs: list = SAMPLER_CONFIGS,
) -> list[dict]:
    """Generate PAIRS_PER_SEED × len(SAMPLER_CONFIGS) synthetic records for one seed."""
    records = []
    # Compute approximate cosine similarities via dot product (embeddings
    # already L2-norm'd); use position in retrieval list as a proxy for similarity rank
    n_candidates = len(candidates)
    similarities = [1.0 - (i / max(n_candidates - 1, 1)) for i in range(n_candidates)]
    # Use position in retrieval list as a proxy for similarity rank
    n = len(candidates)
    similarities = [1.0 - (i / max(n - 1, 1)) for i in range(n)]  # rank-based proxy

    for pair_idx in range(PAIRS_PER_SEED):
        if random.random() < R2_MANUFACTURE_PROB:
            # Intentional mismatch: sample from the bottom quartile → likely R2
            low_k = max(1, n // 4)
            bottom_pool = candidates[n - low_k:]
            cand = random.choice(bottom_pool)
        else:
            # Normal retrieval: softmax sample from top similarities
            temp = SAMPLER_CONFIGS[pair_idx % len(SAMPLER_CONFIGS)]["temperature"]
            idxs = _softmax_sample(similarities, k=1, temperature=temp)
            cand = candidates[idxs[0]]

        new_context = cand.get("context", "")
        question    = seed["question"]

        for s_idx, sampler_cfg in enumerate(SAMPLER_CONFIGS):
            prompt = _build_prompt(question, new_context)
            answer = _generate(prompt, model, tokenizer, sampler_cfg)

            # Re-label regime based on the *retrieved* context
            synthetic_rec = {
                "question": question,
                "context":  new_context,
                "answer":   answer,
                "source":   seed.get("source", ""),
                "domain":   seed.get("domain", ""),
            }
            regime = label_regime(synthetic_rec)

            records.append({
                "seed_id":    seed_id,
                "question":   question,
                "context":    new_context,
                "answer":     answer,
                "regime":     regime,
                "sampler_idx": s_idx,
                "reward":     None,
            })

    return records


def main(n: int = 0) -> None:
    DATA_SYNTHETIC.mkdir(parents=True, exist_ok=True)

    # In test mode: 1 pair × 1 sampler per seed → total records == n
    test_mode = n > 0
    pairs_per_seed = 1 if test_mode else PAIRS_PER_SEED
    sampler_cfgs   = SAMPLER_CONFIGS[:1] if test_mode else SAMPLER_CONFIGS

    # Load seed records
    with open(DATA_LABELED / "all.jsonl", encoding="utf-8") as f:
        seeds = [json.loads(l) for l in f if l.strip()]
    if n:
        seeds = seeds[:n]
    print(f"Loaded {len(seeds)} seed records")

    # Ensure ChromaDB index exists
    try:
        get_collection()
    except Exception:
        print("ChromaDB index not found — building it now...")
        build_index()

    # Load model: base + SFT LoRA adapter
    print(f"Loading policy model from {SFT_CHECKPOINT}...")
    tokenizer = AutoTokenizer.from_pretrained(str(SFT_CHECKPOINT), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(SFT_CHECKPOINT))
    model.eval()

    conn = sqlite3.connect(str(DB_PATH))
    _setup_db(conn)

    skipped = 0
    for seed_id, seed in enumerate(seeds):
        if _already_done(conn, seed_id):
            skipped += 1
            continue

        candidates = query_similar(seed["question"], seed.get("context", ""), k=FETCH_K)
        if not candidates:
            continue

        batch = generate_for_seed(
            seed_id, seed, model, tokenizer, candidates,
            pairs_per_seed=pairs_per_seed, sampler_cfgs=sampler_cfgs,
        )
        _save_records(conn, batch)

        done = seed_id + 1 - skipped
        if done % 50 == 0:
            cur = conn.execute("SELECT COUNT(*) FROM generated")
            print(f"  Seed {seed_id + 1}/{len(seeds)} — {cur.fetchone()[0]} synthetic records")

    # Export to JSONL
    cur = conn.execute(
        "SELECT question, context, answer, regime, sampler_idx FROM generated"
    )
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in cur:
            rec = {
                "question": row[0], "context": row[1], "answer": row[2],
                "regime": row[3], "sampler_config": row[4],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    conn.close()
    print(f"Synthetic QA saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    main(n=args.n)
