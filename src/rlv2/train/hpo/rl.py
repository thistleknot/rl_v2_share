"""Three-trial LoRA MGPO learning-rate search on the matched two-round schema.

Require: the proven SFT checkpoint and a five-prompt stratified bootstrap cohort.
Guarantee: each trial runs 5 uniform bootstrap updates followed by 5 reward-STD-guided updates.
Maintain: LoRA r=16/alpha=64, G=4, max_completion=512, beta=0.10, and the isolated 2B judge.
Failure: training/scoring failures are VOID, never encoded as metric zero.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[4]
PYTHON = sys.executable
WORK = ROOT / "harness_runs" / "hpo_rl_lora_mgpo"

BASE = "harness_runs/sft_full_lr2e5_ep3"
MIX = "data/rl/rl_mix_balanced.jsonl"
EVAL_MINI = "data/rl/rl_eval_mini.jsonl"
LEARNING_RATES = (1e-4,)
HPO_PROMPTS = 9
HPO_STEPS = 9  # Per round: 9 regime-balanced prompts (1,3 band over 4 regimes).
MAX_COMPLETION = 512  # LoRA fits at 512 (no VRAM issue without full-param optimizer state)
TRAIN_TIMEOUT_SECONDS = 7200  # 70 steps × ~60s/step + model load/save + scoring overhead
SCORE_TIMEOUT_SECONDS = 180
MIN_TRIALS = len(LEARNING_RATES)  # 1 trial at proven lr


def _void(trial: optuna.Trial, reason: str) -> None:
    """Mark an unmeasured or collapsed trial VOID and stop it without a fake score."""
    trial.set_user_attr("verdict", "VOID")
    trial.set_user_attr("reason", reason)
    raise optuna.TrialPruned(reason)


def objective(trial: optuna.Trial) -> float:
    """Train one 2×9 regime-balanced LoRA MGPO trial and return canonical mini-eval macroF1."""
    learning_rate = trial.suggest_categorical("lr", list(LEARNING_RATES))
    out_dir = WORK / f"trial_{trial.number}_lr{learning_rate:g}"
    out_dir.mkdir(parents=True, exist_ok=True)

    command = [
        PYTHON,
        "-m", "rlv2.train.rl.grpo",
        "--base",
        str(ROOT / BASE),
        "--mix",
        str(ROOT / MIX),
        "--out",
        str(out_dir),
        "--lr",
        str(learning_rate),
        "--rounds",
        "2",
        "--steps",
        str(HPO_STEPS),
        "--target-pool",
        str(HPO_PROMPTS),
        "--arms",
        "maxent",
        "--seed",
        "0",
        "--max-completion",
        str(MAX_COMPLETION),
        "--beta",
        "0.10",
    ]
    print(
        f"\n[trial {trial.number}] lr={learning_rate:g} | prompts={HPO_PROMPTS} | "
        "reward=isolated-2B-judge",
        flush=True,
    )

    environment = os.environ.copy()
    environment["HARNESS_VRAM_FRAC"] = "0.975"
    environment["SKIP_GPU_EXCLUSIVE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    started = time.time()
    try:
        process = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            errors="replace",
            env=environment,
            timeout=TRAIN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        partial_log = (exc.stdout or "") + (exc.stderr or "")
        (out_dir / "train.log").write_text(partial_log, encoding="utf-8")
        _void(trial, f"training exceeded {TRAIN_TIMEOUT_SECONDS}s")

    train_seconds = time.time() - started
    train_log = (process.stdout or "") + (process.stderr or "")
    (out_dir / "train.log").write_text(train_log, encoding="utf-8")
    if process.returncode != 0:
        _void(trial, f"training failed: {train_log[-500:]}")

    model_dir = out_dir / "maxent"
    if not (model_dir / "adapter_model.safetensors").is_file():
        _void(trial, f"LoRA adapter missing: {model_dir / 'adapter_model.safetensors'}")

    tag = f"hpo_rl_judge_t{trial.number}"
    score_command = [
        PYTHON,
        "-m", "rlv2.eval.rl_calibration_eval",
        "--adapter",
        str(model_dir),
        "--model",
        str(ROOT / BASE),
        "--tag",
        tag,
        "--eval-file",
        str(ROOT / EVAL_MINI),
    ]
    try:
        score_process = subprocess.run(
            score_command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=SCORE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        _void(trial, f"canonical scoring exceeded {SCORE_TIMEOUT_SECONDS}s")

    score_log = (score_process.stdout or "") + (score_process.stderr or "")
    (out_dir / "score.log").write_text(score_log, encoding="utf-8")
    if score_process.returncode != 0:
        _void(trial, f"canonical scoring failed: {score_log[-500:]}")

    calibration_path = ROOT / "data" / "rl" / "calibration.json"
    if not calibration_path.is_file():
        _void(trial, f"canonical scorer did not write {calibration_path}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    result = calibration.get(tag)
    if not isinstance(result, dict):
        _void(trial, f"canonical scorer did not write tag {tag}")
    metrics = result.get("macro_f1")
    if not isinstance(metrics, dict) or "macro_f1" not in metrics:
        _void(trial, f"canonical scorer omitted macro_f1 for tag {tag}")

    macro_f1 = float(metrics["macro_f1"])
    f1_abstain = float(metrics.get("f1_abstain", 0.0))

    trial.set_user_attr("verdict", "MEASURED")
    trial.set_user_attr("f1_abstain", f1_abstain)
    trial.set_user_attr("train_seconds", train_seconds)
    print(
        f"  macroF1@11={macro_f1:.4f} | f1_abstain={f1_abstain:.4f} | "
        f"train={train_seconds:.0f}s | MEASURED",
        flush=True,
    )
    return macro_f1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bounded HPO for two-round LoRA MGPO on five prompts"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=MIN_TRIALS,
        help=f"number of trials; minimum {MIN_TRIALS}",
    )
    parser.add_argument("--seed", type=int, default=42, help="Optuna sampler seed")
    args = parser.parse_args()

    trial_count = max(args.n_trials, MIN_TRIALS)
    WORK.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        study_name="hpo_rl_lora_mgpo",
    )
    for learning_rate in LEARNING_RATES:
        study.enqueue_trial({"lr": learning_rate})

    started = time.time()
    study.optimize(objective, n_trials=trial_count, show_progress_bar=False)

    results = [
        {
            "number": completed.number,
            "params": completed.params,
            "value": completed.value,
            "state": completed.state.name,
            "user_attrs": completed.user_attrs,
        }
        for completed in study.trials
    ]
    results_path = WORK / "hpo_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    measured = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not measured:
        raise RuntimeError(
            f"all {trial_count} judge-based MGPO trials were VOID; see {results_path}"
        )
    best = max(measured, key=lambda completed: float(completed.value))
    print("\n" + "=" * 60)
    print(f"BEST: macroF1@11={best.value:.4f}")
    print(f"  params: {best.params}")
    print(f"  measured={len(measured)} void={trial_count - len(measured)}")
    print(f"  wall={(time.time() - started) / 60:.1f} min")
    print(f"  results={results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
