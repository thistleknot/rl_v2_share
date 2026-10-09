"""The draw wave judges each finished chunk while the next one generates.

Spec: .spec/specs/r3-domain-injection/tasks.md #45 (judge overlaps generation).
Operator 2026-09-27: "patch the judge to use semaphore first".

The overlap must be invisible to everything downstream: same groups, same order, same
scores as generate-everything-then-judge-everything. What it may change is only the
wall clock, and a judge error must still stop the run rather than vanish in a thread.
"""
import importlib.util
import sys
import time

import pytest

pytest.importorskip("torch")


@pytest.fixture(scope="module")
def gof():
    spec = importlib.util.spec_from_file_location("gof_overlap", "docs/grpo_one_file.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["gof_overlap"] = m
    saved, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(m)
    sys.argv = saved
    return m


def _records(n):
    return [{"question": "q%d" % i, "gold": "g%d" % i} for i in range(n)]


def _fake_gen(delay=0.0, seen=None):
    def gen(model, tokenizer, records, cfg, batch=None, n_fresh=None):
        if seen is not None:
            seen.append((len(records), list(n_fresh) if n_fresh is not None else None))
        time.sleep(delay)
        k = int(cfg["num_rollouts"])
        return [{"answers": ["%s-a%d" % (r["question"], j)
                             for j in range(k if n_fresh is None else n_fresh[i])]}
                for i, r in enumerate(records)]
    return gen


def _fake_judge(delay=0.0, calls=None):
    def judge(groups, model=None):
        if calls is not None:
            calls.append([g["question"] for g in groups])
        time.sleep(delay)
        # score depends on the answer text, so any reordering changes the result
        return [[float(len(a) % 3) for a in g["completions"]] for g in groups]
    return judge


def test_order_and_scores_match_the_sequential_path(gof, monkeypatch):
    cfg = {"num_rollouts": 4, "gen_batch": 8}          # 2 records per chunk
    recs = _records(9)                                  # 5 chunks, the last one short
    monkeypatch.setattr(gof, "generate_groups", _fake_gen())
    calls = []
    gens, raws, _, _, _ = gof.generate_and_judge(None, None, recs, cfg,
                                                 _fake_judge(calls=calls), "jev")
    seq_gens = _fake_gen()(None, None, recs, cfg)
    seq_groups = [{"question": r["question"], "completions": g["answers"]}
                  for r, g in zip(recs, seq_gens)]
    assert [g["answers"] for g in gens] == [g["answers"] for g in seq_gens]
    assert raws == _fake_judge()(seq_groups)
    assert calls == [["q0", "q1"], ["q2", "q3"], ["q4", "q5"], ["q6", "q7"], ["q8"]]


def test_judging_hides_behind_generation(gof, monkeypatch):
    cfg = {"num_rollouts": 4, "gen_batch": 4}           # 1 record per chunk
    monkeypatch.setattr(gof, "generate_groups", _fake_gen(delay=0.2))
    t0 = time.time()
    _, _, t_gen, t_wait, busy = gof.generate_and_judge(
        None, None, _records(4), cfg, _fake_judge(delay=0.2), "jev")
    wall = time.time() - t0
    # sequential would be 4 x (0.2 + 0.2) = 1.6 s; overlapped is ~4 x 0.2 + 0.2 = 1.0 s
    assert busy >= 0.75 and t_gen >= 0.75
    assert t_wait < 0.35, t_wait
    assert wall < 1.3, wall


def test_a_judge_error_is_raised_not_swallowed(gof, monkeypatch):
    monkeypatch.setattr(gof, "generate_groups", _fake_gen())

    def broken(groups, model=None):
        raise RuntimeError("jev server down")
    with pytest.raises(RuntimeError, match="jev server down"):
        gof.generate_and_judge(None, None, _records(3),
                               {"num_rollouts": 4, "gen_batch": 4}, broken, "jev")


def test_replay_fresh_counts_are_sliced_per_chunk(gof, monkeypatch):
    seen = []
    monkeypatch.setattr(gof, "generate_groups", _fake_gen(seen=seen))
    gens, raws, _, _, _ = gof.generate_and_judge(
        None, None, _records(4), {"num_rollouts": 4, "gen_batch": 8},
        _fake_judge(), "jev", n_fresh=[4, 2, 3, 1])
    assert seen == [(2, [4, 2]), (2, [3, 1])]
    assert [len(g["answers"]) for g in gens] == [4, 2, 3, 1]
    assert [len(r) for r in raws] == [4, 2, 3, 1]
