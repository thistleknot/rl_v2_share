"""tests/test_mgpo_eval_history.py — every patience eval is recorded, not just printed.

Spec: .spec/specs/r3-domain-injection/tasks.md #36 (operator, 2026-09-04: "you could
just record the numbers and stuff at each step"). Per-step eval readings previously
existed only as stdout lines; reconstructing any run's curve meant grepping .tmp logs.
Source-pin style follows tests/test_full_param_rl_gate.py.
"""
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "src" / "rlv2" / "train" / "rl" / "grpo.py").read_text(encoding="utf-8")


def test_eval_history_is_initialized_with_the_other_cross_round_state():
    """It must live OUTSIDE the round loop, beside _best_score/_lift_streak, or a
    two-round run would only keep its last round's evals."""
    init_block = SRC.split("_lift_streak = [0, 0]")[1].split("for rd in range(args.rounds):")[0]
    assert "_eval_history = []" in init_block, \
        "eval history not initialized in the cross-round state block"


def test_r3_patience_eval_records_the_numbers_it_prints():
    """The append sits at the same site as the [patience-eval:r3] print and carries
    the actual readings — correct count, n, rate, mean|correct, KL."""
    r3 = SRC.split("[patience-eval:r3]")[1].split("def ")[0]
    assert "_eval_history.append({" in r3, "r3 eval printed but never recorded"
    for field in ('"round": rd', '"step": int(state.global_step)', '"lane": "r3"',
                  '"correct": len(_corr)', '"n": _n', '"rate": _rate',
                  '"mean_correct": _mc', '"kl": _kl'):
        assert field in r3, f"r3 eval_history entry missing {field}"


def test_vibe_patience_eval_is_recorded_too():
    vibe = SRC.split("[patience-eval] step=")[1].split('for _reg in ["R1", "R2", "R3", "R4"]')[0]
    assert "_eval_history.append({" in vibe, "vibe-lane eval printed but never recorded"
    for field in ('"lane": "vibe"', '"macro_f1": _score', '"f1_answer"', '"f1_abstain"'):
        assert field in vibe, f"vibe eval_history entry missing {field}"


def test_judge_visibility_prints_question_gold_and_per_completion_scores():
    """Task #37 (operator: "I cant tell what a 0/4, 4/4, 0/4 means"): the reward path
    surfaces the step's question, the gold, and each completion with its raw judge
    score, so the N/4 counts are auditable live from the log."""
    block = SRC.split("raw_scores = list(scores)")[1].split('if lane == "r3":')[0]
    assert "[judge] Q:" in block, "question never surfaced beside its judge scores"
    assert "[judge] gold:" in block, "gold never surfaced beside its judge scores"
    assert "_rs:.2f" in block and "_g['completions'][0][:100]" in block, \
        "per-completion judge score + completion excerpt not printed"
    for guard in ('zip(groups, raw_scores)', 'errors="replace"'):
        assert guard in block, f"judge visibility block missing {guard}"


def test_scale_rewards_is_exposed_and_reaches_the_trainer():
    """Task #38 (BREAK THE 9 R1a): a flag that is parsed but never passed is worse
    than no flag — the trial would report Dr. GRPO results while silently running
    the default scaling. Default must stay 'group' so every prior run is preserved."""
    flag = SRC.split('"--scale-rewards"')[1].split("ap.add_argument")[0]
    assert 'choices=["group", "batch", "none"]' in flag
    assert 'default="group"' in flag, "default must preserve prior-run behavior"
    assert "scale_rewards=args.scale_rewards," in SRC, \
        "--scale-rewards parsed but never reaches GRPOConfig"


def test_eval_history_lands_in_train_stats_json():
    dump = SRC.split('"reward_history": reward_history,')[1].split("json.dump")[0]
    assert '"eval_history": _eval_history,' in dump, \
        "eval history collected but never persisted to train_stats.json"
