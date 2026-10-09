"""Behavioral contract for the shared-prefix LoRA MGPO selector and reward path."""
from __future__ import annotations
import os as _o, sys as _s

import contextlib
import random

import pytest

from rlv2.train.rl import grpo as mgpo
from rlv2.judge import reward


def _records() -> list[dict]:
    """Return varied regime/domain records with stable prompt identifiers."""
    return [
        {"ptext": "r1-math", "prompt": "q1", "context": "ctx", "gold": "a", "domain": "Math"},
        {"ptext": "r2-math", "prompt": "q2", "context": "ctx", "gold": "", "domain": "Math"},
        {"ptext": "r3-code", "prompt": "q3", "context": "", "gold": "a", "domain": "Code"},
        {"ptext": "r4-code", "prompt": "q4", "context": "", "gold": "", "domain": "Code"},
        {"ptext": "r1-code", "prompt": "q5", "context": "ctx", "gold": "a", "domain": "Code"},
        {"ptext": "r2-code", "prompt": "q6", "context": "ctx", "gold": "", "domain": "Code"},
        {"ptext": "r3-math", "prompt": "q7", "context": "", "gold": "a", "domain": "Math"},
        {"ptext": "r4-math", "prompt": "q8", "context": "", "gold": "", "domain": "Math"},
    ]


def test_bootstrap_cohort_is_deterministic_and_arm_independent() -> None:
    """The physically shared first round must not depend on continuation mode."""
    records = _records()
    first = mgpo.bootstrap_cohort(records, n=8, seed=42)
    replay = mgpo.bootstrap_cohort(records, n=8, seed=42)

    assert [row["ptext"] for row in first] == [row["ptext"] for row in replay]
    assert {row["ptext"] for row in first} == {row["ptext"] for row in records}


def test_continuation_probabilities_separate_uniform_from_maxent() -> None:
    """Uniform stays equal while maxent prioritizes within-group reward variance."""
    cohort = _records()[:3]
    # RETENTION CHANGED 2026-08-31 (threshold-spanning, playbook-replaybuffer-scores.md
    # sec 7): a prompt survives into round 2 only with judge outcomes on BOTH sides of
    # CORRECT_FLOOR. The old fixture's "r3-code" [0.4, 0.5, 0.4, 0.5] had two buckets but
    # zero binary variance -- both outcomes wrong -- which is exactly the case the filter
    # exists to drop, so this test's fixtures now span or die accordingly.
    stats = {
        "r1-math": [0.5, 0.5, 0.5, 0.5],      # never crosses -> dropped (weight 0)
        "r2-math": [0.0, 0.0, 1.0, 1.0],      # spans, maximal spread -> highest weight
        "r3-code": [0.6, 0.75, 0.6, 0.75],    # spans, tighter spread -> lower weight
    }

    uniform = mgpo.continuation_probabilities(cohort, stats, mode="uniform", std_floor=0.01)
    guided = mgpo.continuation_probabilities(cohort, stats, mode="maxent", std_floor=0.01)

    assert list(uniform.values()) == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert guided["r2-math"] > guided["r3-code"] > guided["r1-math"]
    assert guided["r1-math"] == 0.0, "an all-one-side prompt must be dropped, not floored"
    assert sum(guided.values()) == pytest.approx(1.0)
    assert guided != uniform


def test_continuation_draws_with_replacement_and_fails_without_history() -> None:
    """Round two reuses the shared cohort and maxent never silently invents history."""
    cohort = _records()[:3]
    stats = {
        "r1-math": [0.5, 0.5, 0.5, 0.5],
        "r2-math": [0.0, 0.0, 1.0, 1.0],
        "r3-code": [0.4, 0.5, 0.4, 0.5],
    }

    selected = mgpo.select_continuation(
        cohort,
        stats,
        mode="maxent",
        rng=random.Random(7),
        n=20,
        std_floor=0.01,
    )
    assert len(selected) == 20
    assert len({row["ptext"] for row in selected}) < len(selected)
    assert all(row in cohort for row in selected)

    with pytest.raises(ValueError, match="reward history"):
        mgpo.select_continuation(
            cohort,
            {"r1-math": [0.1, 0.2]},
            mode="maxent",
            rng=random.Random(7),
            n=3,
            std_floor=0.01,
        )
    with pytest.raises(ValueError, match="mode"):
        mgpo.continuation_probabilities(cohort, stats, mode="other", std_floor=0.01)


def test_degenerate_completion_still_uses_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fixed LLM judge remains the sole reward authority for repetitive output."""
    prompt = "templated prompt"
    meta = {
        prompt: {
            "prompt": "What is the answer?",
            "gold": "42",
            "context": "",
        }
    }
    stats: dict[str, list[float]] = {prompt: []}
    observed: list[list[dict]] = []

    @contextlib.contextmanager
    def handoff(*, policy):
        yield

    def judge(groups: list[dict]) -> list[list[float]]:
        observed.append(groups)
        return [[0.73]]

    monkeypatch.setattr(reward, "judge_vram_handoff", handoff)
    monkeypatch.setattr(reward, "llm_judge_batch_train", judge)

    score = mgpo.make_reward(meta, stats, offload_models=[])(
        [prompt],
        ["999999999999999999999999999999999999999999999999"],
    )

    assert score == [0.73]
    assert len(observed) == 1
    assert observed[0][0]["completions"][0].startswith("999")
    assert stats[prompt] == [0.73]


def test_all_zero_maxent_diversity_fails_instead_of_becoming_uniform() -> None:
    """MaxEnt with no score diversity cannot silently execute a uniform continuation."""
    cohort = _records()[:3]
    stats = {record["ptext"]: [0.5, 0.5, 0.5, 0.5] for record in cohort}

    with pytest.raises(RuntimeError, match="every bootstrap prompt has zero score diversity"):
        mgpo.continuation_probabilities(cohort, stats, mode="maxent")


def test_training_health_monitor_observes_and_enforces_both_kill_gates(capsys) -> None:
    """Every tenth step is visible; prolonged zero variance and one-class output abort."""
    records = []
    for record in _records()[:4]:
        records.extend([record, record])
    completions = ["substantive answer"] * len(records)
    diverse_scores = [score for _ in range(4) for score in (0.1, 0.9)]
    collapse_monitor = mgpo.TrainingHealthMonitor()

    for _ in range(10):
        collapse_monitor.record(records, completions, diverse_scores)
    output = capsys.readouterr().out
    assert output.count("[observe] step=10") == 3
    assert "| R1 |" in output
    assert "| R2 |" in output
    assert "| R3 |" in output
    with pytest.raises(RuntimeError, match="all completions were class='answer'"):
        collapse_monitor.record(records, completions, diverse_scores)

    zero_monitor = mgpo.TrainingHealthMonitor()
    for _ in range(5):
        zero_monitor.record([records[0]], ["answer"], [0.5])
    with pytest.raises(RuntimeError, match="frac_reward_zero_std=1.0 persisted for 6"):
        zero_monitor.record([records[0]], ["answer"], [0.5])


def test_partition_disjointness_checks_eval_and_rubric(tmp_path) -> None:
    """Runtime preflight rejects either eval leakage or rubric leakage before model load."""
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text('{"prompt": "eval only"}\n', encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        '{"splits": {"rubric": {"questions": ["rubric only"]}}}',
        encoding="utf-8",
    )

    mgpo.assert_partition_disjoint(
        [{"prompt": "train only"}],
        [str(eval_path)],
        str(manifest_path),
    )
    with pytest.raises(AssertionError, match="train/eval leakage"):
        mgpo.assert_partition_disjoint(
            [{"prompt": " EVAL   ONLY "}],
            [str(eval_path)],
            str(manifest_path),
        )
    with pytest.raises(AssertionError, match="train/rubric leakage"):
        mgpo.assert_partition_disjoint(
            [{"prompt": "rubric only"}],
            [str(eval_path)],
            str(manifest_path),
        )
