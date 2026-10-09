"""
Stage 3: Regime Labeling
Adds a 'regime' field (R1/R2/R3/R4) to each record.
Run:  python label_regimes.py
Also importable: label_regime(record) -> str  (used by generate_synthetic_qa.py)
"""
import json
from pathlib import Path
from rlv2.common.config import DATA_RAW, DATA_LABELED

# Domains that have no passage context — assign R3 by provenance
_DOMAIN_REGIME = {
    "Math": "R3",
    "Code": "R3",
    "STEM": "R3",
    "Instruction": "R3",
    "Agentic": "R3",
}

_qa_pipe = None


def _get_qa():
    global _qa_pipe
    if _qa_pipe is None:
        from transformers import pipeline
        _qa_pipe = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=-1,
            handle_impossible_answer=True,
        )
    return _qa_pipe


def _bert_span_present(question: str, context: str, answer: str) -> bool:
    """True if context contains a span that answers the question."""
    # Fast path: verbatim substring
    if answer and answer.lower() in context.lower():
        return True
    try:
        result = _get_qa()(question=question, context=context)
        pred = (result.get("answer") or "").lower().strip()
        score = result.get("score", 0.0)
        gold = answer.lower().strip()
        # Consider it a match if BERT finds a high-confidence span that overlaps with gold
        if score > 0.3 and pred and (pred in gold or gold in pred):
            return True
        # Model returned nothing meaningful → unanswerable
        return False
    except Exception:
        return bool(answer and answer.lower() in context.lower())


def label_regime(record: dict) -> str:
    """
    Assign regime to a single record.
    Returns one of: R1, R2, R3[inferred], R4[inferred]
    """
    context = (record.get("context") or "").strip()
    answer  = (record.get("answer")  or "").strip()
    source  = record.get("source", "")
    domain  = record.get("domain", "")
    question = record.get("question", "")

    # ── No context: provenance-based R3 or R4 ────────────────────────────────
    if not context:
        base = _DOMAIN_REGIME.get(domain, "R3")
        return f"{base}[inferred]"

    # ── Context present: R1 vs R2 ────────────────────────────────────────────

    # SQuAD 2.0: empty answer field == unanswerable by dataset construction
    if source == "rajpurkar/squad_v2":
        return "R1" if answer else "R2"

    # HotpotQA always provides an answer within the retrieved paragraphs
    if source == "hotpot_qa":
        return "R1"

    # General case (synthetic records from Stage 5 or other sources)
    return "R1" if _bert_span_present(question, context, answer) else "R2"


def label_file(input_path: Path, output_path: Path, n: int = 0) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if n:
        records = records[:n]

    print(f"Labeling {len(records)} records from {input_path}…")
    counts: dict[str, int] = {}

    with open(output_path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            rec["regime"] = label_regime(rec)
            counts[rec["regime"]] = counts.get(rec["regime"], 0) + 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(records)} — {counts}")

    print(f"Done. Regime distribution: {counts}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="cap records (0 = no limit)")
    args = ap.parse_args()
    label_file(DATA_RAW / "all.jsonl", DATA_LABELED / "all.jsonl", n=args.n)
