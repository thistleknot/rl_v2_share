"""
Stage 4: Build a ChromaDB vector index over the labeled corpus.
Run:  python index_chroma.py
Also exports: get_collection() and query_similar(question, context, k)
"""
import json
import shutil
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from rlv2.common.config import DATA_LABELED, CHROMA_DIR, CHROMA_COLLECTION, FETCH_K

_BATCH = 512
_ef = DefaultEmbeddingFunction()
_collection: chromadb.Collection | None = None


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(CHROMA_COLLECTION, embedding_function=_ef)
    return _collection


def _embed_text(record: dict) -> str:
    q = record.get("question", "")
    c = record.get("context", "")
    return f"{q} {c}".strip() if c else q


def build_index(input_path: Path = DATA_LABELED / "all.jsonl", n: int = 0) -> None:
    with open(input_path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    if n:
        records = records[:n]

    print(f"Indexing {len(records)} records…")
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=_ef,
        metadata={"hnsw:space": "cosine"},
    )

    for start in range(0, len(records), _BATCH):
        batch = records[start : start + _BATCH]
        texts = [_embed_text(r) for r in batch]
        ids   = [f"rec_{start + i}" for i in range(len(batch))]
        metas = [
            {
                "question": r.get("question", ""),
                "context":  r.get("context", ""),
                "answer":   r.get("answer", ""),
                "regime":   r.get("regime", ""),
                "source":   r.get("source", ""),
                "domain":   r.get("domain", ""),
            }
            for r in batch
        ]
        collection.add(ids=ids, documents=texts, metadatas=metas)
        print(f"  {start + len(batch)}/{len(records)}")

    global _collection
    _collection = collection
    print(f"Index built — {collection.count()} records in '{CHROMA_COLLECTION}'")


def query_similar(question: str, context: str = "", k: int = FETCH_K) -> list[dict]:
    """Return up to k metadata dicts ranked by cosine similarity."""
    text = f"{question} {context}".strip() if context else question
    col = get_collection()
    results = col.query(
        query_texts=[text],
        n_results=min(k, col.count()),
        include=["metadatas", "distances"],
    )
    return results["metadatas"][0]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    build_index(n=args.n)
