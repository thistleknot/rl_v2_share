"""prompt_world_model — ensemble probe of the prompt->judge environment (W1).

Spec: docs/plans/ensemble-world-model-over-prompts.md · playbook.md "WORLD MODEL".
The leakage tripwire is tested by PLANTING an eval question and confirming the
check goes red (write-probe-must-be-able-to-fail law: a probe that cannot fail
reports confident false health).
"""
from __future__ import annotations
import os as _o
import sys as _s


import io
import json

import numpy as np
import pytest

from rlv2.train.rl import prompt_world_model as wm


def _fake_embed(texts):
    """Deterministic 'embedding': hash-seeded gaussian per text, L2-normalized.
    Distinct texts get distinct, stable vectors — enough structure for the
    ensemble to fit a signal planted along one axis."""
    out = []
    for t in texts:
        rng = np.random.default_rng(abs(hash(t)) % (2**32))
        v = rng.normal(size=32)
        out.append(v / np.linalg.norm(v))
    return np.asarray(out)


def test_leakage_tripwire_fails_closed_on_a_planted_eval_question(tmp_path):
    ev = tmp_path / "eval.jsonl"
    ev.write_text(json.dumps({"question": "What is the Gini impurity?"}) + "\n",
                  encoding="utf-8")
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps({"prompt": "What is the Gini impurity?", "w": 1, "l": 1}) + "\n",
        encoding="utf-8")
    with pytest.raises(AssertionError, match="EVAL LEAKAGE"):
        wm.build_labels(labels_path=str(labels), eval_path=str(ev))


def test_build_labels_merges_ledger_and_sweep_and_skips_legacy_entries(tmp_path):
    ev = tmp_path / "eval.jsonl"
    ev.write_text(json.dumps({"question": "unrelated eval q"}) + "\n", encoding="utf-8")
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"prompt": "q_a", "w": 2, "l": 0}) + "\n",
                      encoding="utf-8")
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({
        "k1": {"prompt": "q_a", "w": 1, "l": 1, "visits": 1},
        "k2": {"prompt": "q_b", "w": 0, "l": 2, "visits": 1},
        "k3": 4,                       # legacy bare visit count: no label value
    }), encoding="utf-8")
    rows = {r["prompt"]: r for r in wm.build_labels(
        ledger_path=str(ledger), labels_path=str(labels), eval_path=str(ev))}
    assert rows["q_a"]["w"] == 3 and rows["q_a"]["l"] == 1, "sweep+ledger must merge"
    assert rows["q_b"]["l"] == 2
    assert len(rows) == 2, "legacy bare-count entries must be skipped, not crash"


def test_ensemble_learns_a_planted_signal_and_agrees_on_trained_data(tmp_path):
    """The Dyna-Q ensemble's core property: the ensemble fits the labels (AUC well above
    chance) and DISAGREES less where data was seen than on novel inputs."""
    rng = np.random.default_rng(3)
    easy = [{"prompt": "easy_%d" % i, "w": 4, "l": 0} for i in range(30)]
    hard = [{"prompt": "hard_%d" % i, "w": 0, "l": 4} for i in range(30)]
    labels = easy + hard

    def embed(texts):
        X = _fake_embed(texts)
        # Plant the signal: easy prompts get +0.8 on axis 0, hard get -0.8.
        for i, t in enumerate(texts):
            X[i, 0] = 0.8 if t.startswith("easy") else -0.8
        return X

    heads = wm.train_ensemble(labels, embed, k=3,
                              out_path=str(tmp_path / "m.npz"), seed=7)
    rep = wm.validate(heads, labels, embed)
    assert rep["auc"] > 0.9, "ensemble failed to fit a linearly-planted signal: %s" % rep

    X_seen = embed([r["prompt"] for r in labels])
    X_novel = _fake_embed(["novel_%d" % i for i in range(30)])  # no planted axis
    _, var_seen = wm.predict(heads, X_seen)
    _, var_novel = wm.predict(heads, X_novel)
    assert var_novel.mean() > var_seen.mean(), (
        "disagreement should be HIGHER off-distribution (part 6 Q2): "
        "seen %.5f vs novel %.5f" % (var_seen.mean(), var_novel.mean()))


def test_save_load_roundtrip_preserves_predictions(tmp_path):
    labels = [{"prompt": "p%d" % i, "w": i % 3, "l": 2 - (i % 3) + 1}
              for i in range(20)]
    path = str(tmp_path / "m.npz")
    heads = wm.train_ensemble(labels, _fake_embed, k=2, out_path=path, seed=7)
    X = _fake_embed(["p1", "p2", "zz"])
    m1, v1 = wm.predict(heads, X)
    m2, v2 = wm.predict(wm.load_ensemble(path), X)
    assert np.allclose(m1, m2) and np.allclose(v1, v2), \
        "loaded ensemble diverges from the trained one"
