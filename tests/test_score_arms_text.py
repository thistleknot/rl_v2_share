"""Every cell written by score_arms carries the answer TEXT, not only its hash.

Spec: .spec/specs/r3-domain-injection/tasks.md #52 (answer text in cells).
Measured 2026-09-27: cells held only a 16-char hash (`ans`), so an eval jump
0.250 -> 0.594 could not be read back and turned out to be the question echoed.
This drives the real `main()` cell-writing path (protocol=single) end to end,
stubbing only the GPU generation (pre-seeded gen cache) and the judge network
call -- no GPU, no live judge.
"""
import hashlib
import json
import os
import sys

from rlv2.eval import score_arms as SA  # noqa: E402
from rlv2.train.rl import instrument as I  # noqa: E402
from rlv2.train.rl import differentiability_ledger as ledger  # noqa: E402
from rlv2.judge import judge as J  # noqa: E402

ANSWER_TEXT = "The capital of France is Paris."
EVAL_PATH = "data/.tmp_test_score_arms_eval.jsonl"
OUT_PATH = "data/.tmp_test_score_arms_out.json"


def _gen_cache_key(arm_path, eval_path, k, temperature):
    state = ledger.phase_state(arm_path)
    raw = "%s|%s|k=%d|t=%.3f|prompt=rl_template" % (
        state, os.path.basename(eval_path), k, temperature)
    return os.path.join("data", "gen_cache",
                        hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16] + ".json")


def test_cell_carries_answer_text(monkeypatch, tmp_path):
    # A tiny frozen eval, one question, one arm, k=1 -- no model, no judge server.
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "q1", "question": "capital of France?",
                            "gold": "Paris"}) + "\n")

    arm_path = "not_a_real_checkpoint_dir"   # not a dir -> skips the config.json guard
    sidecar = OUT_PATH + ".cells.jsonl"
    cache_key = _gen_cache_key(arm_path, EVAL_PATH, 1, 0.0)
    for p in (OUT_PATH, sidecar, cache_key):
        if os.path.exists(p):
            os.remove(p)
    os.makedirs(os.path.dirname(cache_key), exist_ok=True)
    with open(cache_key, "w", encoding="utf-8") as f:
        json.dump([[ANSWER_TEXT]], f)   # k=1 outer list, 1 row inner list

    monkeypatch.setattr(I, "verify", lambda: None)
    monkeypatch.setattr(SA.J, "judge", lambda *a, **k: {"ok": True, "score": 1.0})
    monkeypatch.setattr(
        sys, "argv",
        ["score_arms.py", "--arm", "base=" + arm_path, "--eval", EVAL_PATH,
         "--k", "1", "--protocol", "single", "--out", OUT_PATH])

    try:
        SA.main()

        assert os.path.exists(sidecar)
        cells = [json.loads(l) for l in open(sidecar, encoding="utf-8")]
        assert len(cells) == 1
        cell = cells[0]
        assert cell["text"] == ANSWER_TEXT
        assert cell["ans"] == hashlib.sha1(ANSWER_TEXT.encode("utf-8")).hexdigest()[:16]

        # Main JSON output's per-record shape is unchanged (arm/qid/rep/protocol/scores).
        out = json.load(open(OUT_PATH, encoding="utf-8"))
        rec = out["records"][0]
        assert set(rec) == {"qid", "rep", "protocol", "scores"}
        assert "text" not in rec["scores"]["base"]
    finally:
        for p in (EVAL_PATH, OUT_PATH, sidecar, cache_key):
            if os.path.exists(p):
                os.remove(p)
