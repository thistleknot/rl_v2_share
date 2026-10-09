"""tests/test_prompt_world_model_qa.py — the QA-verifier mode (SURPASS 11 Arm B).

Spec: docs/plans/surpass-11-close-the-untested-lanes.md Arm B.
Task: playbook.md "SURPASS 11" S11-B1.

All tests run without ollama or torch: embed_fn is injected, and qa_rerank's
degeneracy gate is injected, so nothing heavy is imported.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest


from rlv2.train.rl import prompt_world_model as W


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


@pytest.fixture()
def harvest(tmp_path):
    """One harvest stem + pool + eval, ids deliberately MISORDERED between the
    gen and scored sidecars so an index-based join would mispair them."""
    pool = tmp_path / "pool.jsonl"
    _write_jsonl(pool, [{"id": "q1", "question": "What is bagging?"},
                        {"id": "q2", "question": "What is boosting?"}])
    ev = tmp_path / "eval.jsonl"
    _write_jsonl(ev, [{"item_id": "e1", "question": "What is a random forest?"}])
    stem = tmp_path / "h"
    _write_jsonl(Path(str(stem) + ".gen.jsonl"),
                 [{"id": "q1", "completions": ["good answer one", ""]},
                  {"id": "q2", "completions": ["bad answer", "good answer two"]}])
    _write_jsonl(Path(str(stem) + ".scored.jsonl"),
                 [{"id": "q2", "scores": [0.1, 3.0]},      # 3.0 = the clamp bug
                  {"id": "q1", "scores": [0.95, 0.9]}])
    return {"stem": str(stem), "pool": str(pool), "eval": str(ev)}


def test_qa_labels_join_by_id_skip_empty_and_clamp(harvest):
    rows = W.build_qa_labels([harvest["stem"]], pool_path=harvest["pool"],
                             eval_path=harvest["eval"])
    got = {(r["question"], r["answer"]): r["score"] for r in rows}
    # q1's empty second completion is dropped; q1/q2 rows pair with their OWN
    # scores despite the scored file's reversed order; 3.0 clamps to 1.0.
    assert got == {("What is bagging?", "good answer one"): 0.95,
                   ("What is boosting?", "bad answer"): 0.1,
                   ("What is boosting?", "good answer two"): 1.0}


def test_qa_labels_leakage_tripwire_fails_closed(tmp_path, harvest):
    poisoned = tmp_path / "pool2.jsonl"
    _write_jsonl(poisoned, [{"id": "q1", "question": "What is a RANDOM FOREST?"}])
    _write_jsonl(tmp_path / "p.gen.jsonl",
                 [{"id": "q1", "completions": ["an answer"]}])
    _write_jsonl(tmp_path / "p.scored.jsonl", [{"id": "q1", "scores": [0.9]}])
    with pytest.raises(AssertionError, match="EVAL LEAKAGE"):
        W.build_qa_labels([str(tmp_path / "p")], pool_path=str(poisoned),
                          eval_path=harvest["eval"])


def _toy_embed(texts):
    """1-D separable feature: presence of the token 'good'."""
    return np.asarray([[1.0] if "good" in t else [0.0] for t in texts])


def test_qa_ensemble_learns_the_separable_signal(tmp_path):
    labels = ([{"question": "q", "answer": "good %d" % i, "score": 0.95}
               for i in range(8)] +
              [{"question": "q", "answer": "bad %d" % i, "score": 0.1}
               for i in range(8)])
    heads = W.train_qa_ensemble(labels, _toy_embed, k=3,
                                out_path=str(tmp_path / "m.npz"), seed=7)
    held = [{"question": "q", "answer": "good new", "score": 0.9},
            {"question": "q", "answer": "bad new", "score": 0.2}]
    assert W.qa_auc(heads, held, _toy_embed) == 1.0
    # round-trip: load_ensemble reads back what train wrote
    loaded = W.load_ensemble(str(tmp_path / "m.npz"))
    assert len(loaded) == 3


def test_qa_rerank_picks_by_verifier_and_settles_on_recorded_judge(tmp_path):
    labels = ([{"question": "q", "answer": "good %d" % i, "score": 0.95}
               for i in range(8)] +
              [{"question": "q", "answer": "bad %d" % i, "score": 0.1}
               for i in range(8)])
    heads = W.train_qa_ensemble(labels, _toy_embed, k=3,
                                out_path=str(tmp_path / "m.npz"), seed=7)
    ev = tmp_path / "eval.jsonl"
    _write_jsonl(ev, [{"item_id": "e1", "question": "Q one"},
                      {"item_id": "e2", "question": "Q two"}])
    receipt = tmp_path / "k8.json"
    receipt.write_text(json.dumps({"records": [
        {"qid": "e1", "rep": 0, "scores": {"arm": {"score": 0.2}}},
        {"qid": "e1", "rep": 1, "scores": {"arm": {"score": 0.9}}},
        {"qid": "e2", "rep": 0, "scores": {"arm": {"score": 0.95}}},
        # e2 rep1: judge dropped it — no record at all
    ]}), encoding="utf-8")
    (tmp_path / "k8.json.generations.json").write_text(json.dumps({
        # rep-major: gens[rep][qi]
        "arm": [["bad e1r0", "GOODSPAM e2r0"],
                ["good e1r1", "good e2r1"]]}), encoding="utf-8")
    out = W.qa_rerank(str(receipt), "arm", heads, _toy_embed,
                      eval_path=str(ev),
                      is_degenerate=lambda t: "SPAM" in t)
    by_qid = {p["qid"]: p for p in out["picks"]}
    # e1: verifier prefers the 'good' rep 1; its recorded judge score 0.9 -> OK.
    assert by_qid["e1"]["rep"] == 1 and by_qid["e1"]["correct"]
    # e2: rep0 is degenerate (SPAM) so the verifier must pick rep 1 even though
    # rep0 has the high recorded score; rep1 was never judged -> counts WRONG.
    assert by_qid["e2"]["rep"] == 1 and not by_qid["e2"]["correct"]
    assert by_qid["e2"]["judge"] is None
    assert out["correct"] == 1 and out["n"] == 2
