import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import boxcox, norm as _norm
from datasets import load_dataset_builder, load_dataset
from huggingface_hub import HfApi

TOTAL_RECORDS = 2000
DOMAIN_PERCENTAGES = {
    "Math": 0.30,
    "Code": 0.18,
    "STEM": 0.12,
    "Instruction": 0.10,
    "Agentic": 0.15,
    "ReadingComp": 0.15,
}

DATASETS_CONFIG = {
    "Math": [
        ("gsm8k", "main"),
        ("qwedsacf/competition_math", "default"),
        ("AI-MO/NuminaMath-CoT", "default"),
    ],
    "Code": [
        ("deepmind/code_contests", "default"),
        ("mbpp", "sanitized"),
        ("openai_humaneval", "openai_humaneval"),
    ],
    "STEM": [
        ("sciq", "default"),
        ("ai2_arc", "ARC-Challenge"),
        ("openbookqa", "main"),
        ("cais/mmlu", "high_school_physics"),
    ],
    "Instruction": [
        ("OpenAssistant/oasst1", "default"),
        ("HuggingFaceH4/databricks_dolly_15k", "default"),
    ],
    "Agentic": [
        ("Salesforce/xlam-function-calling-60k", "default"),
        ("glaiveai/glaive-function-calling-v2", "default"),
    ],
    "ReadingComp": [
        ("rajpurkar/squad_v2", "squad_v2"),
        ("hotpot_qa", "distractor"),
    ],
}

MAX_SIZE_BYTES = 2 * 1024 * 1024 * 1024
OUTPUT_DIR = Path("data/raw")

HF_TOKEN = os.environ.get("HF_READ_TOKEN")


def sample_from_shards(repo_id: str, split: str, n: int, seed: int = 42) -> list[dict]:
    """Fetch the minimum number of Parquet shards needed to collect n rows."""
    api = HfApi(token=HF_TOKEN)
    try:
        all_files = list(api.list_repo_files(repo_id, repo_type="dataset"))
    except Exception as e:
        raise RuntimeError(f"Cannot list files for {repo_id}: {e}")

    shards = [f for f in all_files if f.endswith(".parquet") and split in f]
    if not shards:
        raise RuntimeError(f"No parquet shards found for {repo_id}/{split}")

    rng = random.Random(seed)
    rng.shuffle(shards)

    storage_options = {"headers": {"Authorization": f"Bearer {HF_TOKEN}"}} if HF_TOKEN else {}

    rows: list[dict] = []
    for shard in shards:
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{shard}"
        try:
            df = pd.read_parquet(url, storage_options=storage_options or None)
            shard_rows = df.to_dict("records")
            rng.shuffle(shard_rows)
            rows.extend(shard_rows)
            print(f"    [SHARD] {shard.split('/')[-1]}: +{len(shard_rows)} rows (total {len(rows)})")
        except Exception as e:
            print(f"    [WARN] Shard {shard}: {e}")
            continue
        if len(rows) >= n:
            break

    if not rows:
        raise RuntimeError(f"No rows loaded from shards for {repo_id}")

    return rows[:n]


def normalize_row(ds_name: str, row: dict) -> dict | None:
    """Map a raw row to {question, context, answer}. Return None to skip.

    Code datasets additionally carry `tests` (assert strings) and, for HumanEval,
    `entry_point`. Extra keys flow through to the written record unchanged.
    """
    try:
        match ds_name:
            case "gsm8k":
                return {"question": row["question"], "context": "", "answer": row["answer"]}

            case "qwedsacf/competition_math":
                return {"question": row["problem"], "context": "", "answer": row["solution"]}

            case "AI-MO/NuminaMath-CoT":
                return {"question": row["problem"], "context": "", "answer": row["solution"]}

            case "deepmind/code_contests":
                sols = row.get("solutions")
                sol_list = []
                if isinstance(sols, dict):
                    raw = sols.get("solution")
                    if raw is not None:
                        try:
                            sol_list = list(raw)
                        except (TypeError, ValueError):
                            pass
                answer = sol_list[0] if sol_list else ""
                return {"question": row["description"], "context": "", "answer": answer}

            case "mbpp":
                # tests CARRY THE REWARD -- dropping them is why data/raw/code.jsonl
                # has no way to be scored. Spec: .spec/specs/r3-domain-injection/
                # rlvr-math-code.md (code lane).
                return {"question": row["prompt"], "context": "", "answer": row["code"],
                        "tests": list(row.get("test_list") or []),
                        "test_setup": row.get("test_setup_code") or ""}

            case "openai_humaneval":
                # canonical_solution is a BODY, not a program: it has no def line
                # and is indented. The prompt holds the signature, so the runnable
                # gold is prompt + canonical_solution.
                return {"question": row["prompt"], "context": "",
                        "answer": row["prompt"] + row["canonical_solution"],
                        "tests": [row["test"]] if row.get("test") else [],
                        "entry_point": row.get("entry_point") or ""}

            case "sciq":
                return {"question": row["question"], "context": row.get("support", ""), "answer": row["correct_answer"]}

            case "ai2_arc":
                labels, texts = row["choices"]["label"], row["choices"]["text"]
                answer = texts[labels.index(row["answerKey"])]
                return {"question": row["question"], "context": "", "answer": answer}

            case "openbookqa":
                labels, texts = row["choices"]["label"], row["choices"]["text"]
                answer = texts[labels.index(row["answerKey"])]
                return {"question": row["question_stem"], "context": "", "answer": answer}

            case "cais/mmlu":
                answer = row["choices"][row["answer"]]
                return {"question": row["question"], "context": "", "answer": answer}

            case "OpenAssistant/oasst1":
                # sample_oasst1 returns already-formatted {question, context, answer} dicts
                return {"question": row.get("question", ""), "context": "", "answer": row.get("answer", "")}

            case "HuggingFaceH4/databricks_dolly_15k":
                # HF H4 mirror schema: instruction/input/output (not prompt/context/response)
                return {
                    "question": row.get("instruction", row.get("prompt", "")),
                    "context": row.get("input", row.get("context", "")),
                    "answer": row.get("output", row.get("response", row.get("completion", ""))),
                }

            case "Salesforce/xlam-function-calling-60k":
                tools = row.get("tools", "")
                if isinstance(tools, list):
                    tools = json.dumps(tools)
                return {"question": row["query"], "context": tools, "answer": str(row.get("answers", ""))}

            case "glaiveai/glaive-function-calling-v2":
                system = row.get("system", "")
                chat = row.get("chat", "")
                question, answer = "", ""
                for line in chat.split("\n"):
                    if line.startswith("USER:") and not question:
                        question = line[5:].strip()
                    elif line.startswith("ASSISTANT:") and not answer:
                        answer = line[10:].strip()
                if not question:
                    return None
                return {"question": question, "context": system, "answer": answer}

            case "rajpurkar/squad_v2":
                ans = row.get("answers")
                texts = []
                if isinstance(ans, dict):
                    raw = ans.get("text")
                    if raw is not None:
                        try:
                            texts = list(raw)
                        except (TypeError, ValueError):
                            pass
                return {
                    "question": row["question"],
                    "context": row["context"],
                    "answer": texts[0] if texts else "",
                }

            case "hotpot_qa":
                ctx = row.get("context") or {}
                titles = ctx.get("title", [])
                sentences = ctx.get("sentences", [])
                ctx_text = "\n\n".join(
                    f"{t}:\n{''.join(s)}" for t, s in zip(titles, sentences)
                )
                return {"question": row["question"], "context": ctx_text, "answer": row["answer"]}

            case _:
                return None
    except Exception:
        return None


def sample_oasst1(ds, n: int) -> list[dict]:
    """oasst1 is a message tree; join prompter→assistant pairs before sampling."""
    rows = list(ds)
    by_id = {r["message_id"]: r for r in rows}
    pairs = []
    for r in rows:
        if r.get("role") != "assistant":
            continue
        parent = by_id.get(r.get("parent_id"))
        if parent and parent.get("role") == "prompter":
            pairs.append({"question": parent["text"], "context": "", "answer": r["text"]})
    pairs = pairs[: n * 3]  # over-sample before shuffle
    import random
    random.seed(42)
    random.shuffle(pairs)
    return pairs[:n]




def calculate_allocations(sizes: list[int], target_total: int, lam: float = 0.5) -> list[int]:
    """
    Allocate target_total records across datasets using a two-parameter
    CDF-based weighting over Box-Cox-normalized sizes.

    Pipeline:
      1. np.log  — collapses extreme scale differences
      2. Box-Cox — finds optimal λ to bring the log-distribution to near-normality
      3. Robust z-score — center = γ*mean + (1-γ)*median (γ=0.5); scale by σ
      4. CDF     — norm.cdf(z) maps each dataset to its percentile rank in [0, 1]
      5. Blend   — (1-λ)*uniform + λ*cdf_normalized (λ=0.5)
      6. Scale   — normalize weights to ratios, multiply by target_total

    γ=0.5: robust centering between mean (pulled toward large datasets) and
    median (resistant to outliers). Prevents NuminaMath-scale datasets from
    dominating the z-score distribution.

    λ=0.5: balances CDF-proportional allocation (on-policy distribution) with
    uniform allocation (prevents overfitting to dominant datasets). Both
    parameters become learnable in later stages.
    """
    n = len(sizes)
    if n == 0:
        return []
    if n == 1:
        return [target_total]

    arr = np.array([max(s, 1) for s in sizes], dtype=float)

    # Step 1: log
    log_arr = np.log(arr)

    # Step 2: Box-Cox over log values
    shifted = log_arr - log_arr.min() + 1.0
    # scipy.stats.boxcox raises on constant input. Equal-sized cells already represent
    # the uniform allocation that Box-Cox would converge to — skip the transform.
    if np.all(shifted == shifted[0]):
        bc = shifted  # identity; downstream sigma==0 guard produces uniform allocation
    else:
        bc, _ = boxcox(shifted)

    # Step 3: robust z-score
    gamma = 0.5
    center = gamma * bc.mean() + (1 - gamma) * np.median(bc)
    sigma = bc.std(ddof=0)
    if sigma == 0:
        base = target_total // n
        allocs = [base] * n
        allocs[-1] += target_total - sum(allocs)
        return allocs
    z = (bc - center) / sigma

    # Step 4: CDF — larger dataset → higher percentile → more allocation
    cdf_vals = _norm.cdf(z)

    # Step 5: convex blend of uniform and CDF-proportional
    uniform = np.ones(n) / n
    cdf_norm = cdf_vals / cdf_vals.sum()
    weights = (1 - lam) * uniform + lam * cdf_norm

    # Step 6: scale to target
    ratios = weights / weights.sum()
    allocs = [round(r * target_total) for r in ratios]
    allocs[-1] += target_total - sum(allocs)
    return allocs


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []

    for domain, ds_list in DATASETS_CONFIG.items():
        domain_target = int(TOTAL_RECORDS * DOMAIN_PERCENTAGES[domain])
        print(f"\n--- Processing {domain} | Target: {domain_target} records ---")

        valid_datasets, sizes = [], []

        for ds_name, ds_config in ds_list:
            try:
                builder = load_dataset_builder(ds_name, ds_config, trust_remote_code=True, token=HF_TOKEN)
                info = builder.info
                size_in_bytes = info.dataset_size or 0
                is_large = size_in_bytes > MAX_SIZE_BYTES

                if info.splits is None:
                    split_name = "train"
                    num_examples = 10000
                    is_large = True
                else:
                    splits = list(info.splits.keys())
                    split_name = "train" if "train" in splits else splits[0]
                    num_examples = info.splits[split_name].num_examples

                valid_datasets.append((ds_name, ds_config, split_name, is_large))
                sizes.append(num_examples)
                print(f"  [{'STREAMING' if is_large else 'ACCEPTED'}] {ds_name}: {num_examples} rows.")
            except Exception as builder_err:
                # Builder failed (schema incompatibility or gated dataset) — probe shard list instead
                try:
                    api = HfApi(token=HF_TOKEN)
                    all_files = list(api.list_repo_files(ds_name, repo_type="dataset"))
                    shards = [f for f in all_files if f.endswith(".parquet") and "train" in f]
                    if shards:
                        valid_datasets.append((ds_name, ds_config, "train", True))
                        sizes.append(50000)  # placeholder for log-weight calc
                        print(f"  [SHARD-FALLBACK] {ds_name}: builder failed, found {len(shards)} parquet shards")
                    else:
                        print(f"  [ERROR] {ds_name}: {builder_err}")
                except Exception as e:
                    print(f"  [ERROR] {ds_name}: builder={builder_err}, hub={e}")

        if not valid_datasets:
            print(f"  [SKIP] No valid datasets for {domain}.")
            continue

        allocations = calculate_allocations(sizes, domain_target)
        domain_records: list[dict] = []

        for (ds_name, ds_config, split_name, is_large), n in zip(valid_datasets, allocations):
            try:
                # oasst1: needs full streaming for message-tree joining
                if ds_name == "OpenAssistant/oasst1":
                    ds = load_dataset(ds_name, ds_config, streaming=True, trust_remote_code=True)
                    target_ds = ds[split_name] if isinstance(ds, dict) else ds
                    first_row = next(iter(target_ds))
                    print(f"    [SCHEMA] {ds_name} keys: {list(first_row.keys())}")
                    sampled = sample_oasst1(target_ds, n)
                elif is_large:
                    # Shard-based: download only enough shards to reach n rows
                    sampled = sample_from_shards(ds_name, split_name, n)
                    if sampled:
                        print(f"    [SCHEMA] {ds_name} keys: {list(sampled[0].keys())}")
                else:
                    ds = load_dataset(ds_name, ds_config, trust_remote_code=True)
                    target_ds = ds[split_name] if isinstance(ds, dict) else ds
                    first_row = next(iter(target_ds))
                    print(f"    [SCHEMA] {ds_name} keys: {list(first_row.keys())}")
                    sampled = list(target_ds.shuffle(seed=42).select(range(min(n, len(target_ds)))))

                normalized: list[dict] = []
                for row in sampled:
                    rec = normalize_row(ds_name, row if isinstance(row, dict) else dict(row))
                    if rec is not None:
                        rec["source"] = ds_name
                        rec["domain"] = domain
                        normalized.append(rec)

                print(f"  -> Sampled {len(normalized)} records from {ds_name}")
                domain_records.extend(normalized)
            except Exception as e:
                print(f"  [ERROR] Failed to sample {ds_name}: {e}")

        all_records.extend(domain_records)
        out_path = OUTPUT_DIR / f"{domain.lower()}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in domain_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  => Saved {len(domain_records)} records to {out_path}")

    out_path = OUTPUT_DIR / "all.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n=== Total: {len(all_records)} records saved to {out_path} ===")


if __name__ == "__main__":
    main()
