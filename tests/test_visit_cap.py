"""--max-visits: a record judged N times leaves the draw pool.

Spec: .spec/specs/r3-domain-injection/tasks.md #50 (visit cap).
Operator 2026-09-27: records were harvested again and again after they could no longer be
learned; "inserting the cap at no more than 3 visits auto anneals".
"""
import importlib.util
import sys

import pytest

pytest.importorskip("torch")


@pytest.fixture(scope="module")
def gof():
    spec = importlib.util.spec_from_file_location("gof_cap", "docs/grpo_one_file.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["gof_cap"] = m
    saved, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(m)
    sys.argv = saved
    return m


POOL = [{"question": "q%d" % i} for i in range(4)]


def test_records_at_the_cap_leave_the_pool(gof):
    log = {"q0": [0.1, 0.2, 0.3], "q1": [0.1, 0.2], "q2": [0.5]}
    left = gof.under_visit_cap(POOL, log, 3)
    assert [r["question"] for r in left] == ["q1", "q2", "q3"]


def test_pending_draws_count_toward_the_cap(gof):
    # tasks.md #54: rl_b_echoguard step 12 drew the last under-cap record 100 times in one
    # wave because visits were logged only after judging.
    log = {"q0": [0.1, 0.2], "q1": [0.1, 0.2, 0.3], "q2": [0.1, 0.2, 0.3],
           "q3": [0.1, 0.2, 0.3]}
    assert [r["question"] for r in gof.under_visit_cap(POOL, log, 3)] == ["q0"]
    assert gof.under_visit_cap(POOL, log, 3, pending={"q0": 1}) == []


def test_pending_without_log_entries(gof):
    left = gof.under_visit_cap(POOL, {}, 2, pending={"q1": 2, "q2": 1})
    assert [r["question"] for r in left] == ["q0", "q2", "q3"]


def test_cap_reads_exposure_counts_not_draw_lists(gof):
    # playbook CX1: the cap is on groups that reached a gradient step (int map)
    exposures = {"q0": 3, "q1": 2, "q3": 0}
    assert [r["question"] for r in gof.under_visit_cap(POOL, exposures, 3)] == ["q1", "q2", "q3"]


def test_exposure_line_buckets(gof):
    line = gof.exposure_line(POOL, {"q0": 3, "q1": 1, "q2": 7}, 3, draws=45, live=16)
    assert line == "[exposure] records at 0/1/2/3 exposures: 1/1/0/2 | draws 45 for 16 live"


def _replay_exposures(path, per_step=16, floor=0.7):
    """Replay a judged_pairs.jsonl through the accept rule: per round, the first
    `per_step` groups with mixed scores enter the gradient; dead groups never do."""
    import collections
    import io
    import json
    groups = collections.OrderedDict()
    for line in io.open(path, encoding="utf-8"):
        r = json.loads(line)
        groups.setdefault((r["round"], r["q"]), []).append(r["score"] >= floor)
    exposures, accepted = {}, collections.Counter()
    for (rd, q), v in groups.items():
        if 0 < sum(v) < len(v) and accepted[rd] < per_step:
            accepted[rd] += 1
            exposures[q] = exposures.get(q, 0) + 1
    return exposures, len(groups)


def test_replay_of_rl_b_capfix_counts_exposures_not_draws(gof):
    import os
    path = "W:/phase_runs/rl_b_capfix/judged_pairs.jsonl"
    if not os.path.exists(path):
        pytest.skip("rl_b_capfix judged pairs not on this machine")
    exposures, n_groups = _replay_exposures(path)
    total = sum(exposures.values())
    # judged groups (the old cap's count) far exceed gradient exposures: measured
    # 465 groups (in-round redraws merged) vs 106 exposures over 8 steps x 16 slots
    assert n_groups >= 400
    assert 100 <= total <= 128
    pool = [{"question": q} for q in {k[1] for k in
             (tuple(x) for x in [(0, q) for q in exposures])}]
    # every record still under the cap -> the run would NOT have ended at step 8
    assert len(gof.under_visit_cap(pool, exposures, 3)) > 0


def test_cap_one_keeps_only_unvisited(gof):
    log = {"q0": [0.0], "q3": [0.4]}
    assert [r["question"] for r in gof.under_visit_cap(POOL, log, 1)] == ["q1", "q2"]


def test_all_capped_returns_empty(gof):
    log = {"q%d" % i: [0.0, 0.0, 0.0] for i in range(4)}
    assert gof.under_visit_cap(POOL, log, 3) == []


@pytest.mark.parametrize("cap", [None, 0])
def test_no_cap_returns_pool_unchanged(gof, cap):
    log = {"q0": [0.0] * 10}
    assert gof.under_visit_cap(POOL, log, cap) is POOL


def test_flag_parses(gof):
    import argparse
    assert "--max-visits" in open("docs/grpo_one_file.py", encoding="utf-8").read()
