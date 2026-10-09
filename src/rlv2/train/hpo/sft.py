"""hpo_sft_full.py — Optuna TPE sweep over full-param SFT to find the config that learns abstention.

Objective: maximize macroF1@112 (which requires f1_abstain > 0).
Prior: lr=1e-5, epochs=1 (FAILED — f1_abstain=0.000).
Search: lr [5e-6, 1e-5, 2e-5, 5e-5], epochs [1, 2, 3].

Each trial trains full-param SFT, scores on rl_eval.jsonl, reports macroF1.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[4]
PY = sys.executable
WORK = ROOT / "harness_runs" / "hpo_sft_full"
WORK.mkdir(parents=True, exist_ok=True)


def objective(trial):
    lr = trial.suggest_categorical("lr", [5e-6, 1e-5, 2e-5, 5e-5])
    epochs = trial.suggest_int("epochs", 1, 3)

    out_dir = WORK / f"trial_{trial.number}_lr{lr:g}_ep{epochs}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train
    cmd = [
        PY, "-m", "rlv2.train.sft.train",
        "--base", "skills_model",
        "--subset", "data/rl/vibe_sft_subset.jsonl",
        "--out", str(out_dir),
        "--eval", "data/rl/rl_eval.jsonl",
        "--lr", str(lr),
        "--epochs", str(epochs),
        "--full-param",
        "--seed", "0",
    ]
    print(f"\n[trial {trial.number}] lr={lr:g} epochs={epochs}", flush=True)
    t0 = time.time()
    env = os.environ.copy()
    env["HARNESS_VRAM_FRAC"] = "0.975"
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, errors="replace", env=env)
    wall = time.time() - t0

    if proc.returncode != 0:
        err = proc.stderr[-500:] if proc.stderr else proc.stdout[-500:] if proc.stdout else "unknown"
        print(f"  FAILED ({wall:.0f}s): {err[-100:]}", flush=True)
        return 0.0  # failed trial scores zero

    # Score
    score_cmd = [
        PY, "-m", "rlv2.eval.rl_calibration_eval",
        "--model", str(out_dir),
        "--tag", f"hpo_sft_t{trial.number}",
        "--eval-file", "data/rl/rl_eval.jsonl",
    ]
    proc2 = subprocess.run(score_cmd, cwd=ROOT, capture_output=True, text=True, errors="replace")
    if proc2.returncode != 0:
        print(f"  SCORE FAILED: {proc2.stderr[-100:] if proc2.stderr else 'unknown'}", flush=True)
        return 0.0

    # Read result from calibration.json
    cal_path = ROOT / "data" / "rl" / "calibration.json"
    if not cal_path.exists():
        return 0.0
    cal = json.load(open(cal_path, encoding="utf-8"))
    result = cal.get(f"hpo_sft_t{trial.number}", {})
    mf = result.get("macro_f1", {})
    if isinstance(mf, dict):
        macro_f1 = mf.get("macro_f1", 0.0)
        f1_abs = mf.get("f1_abstain", 0.0)
    else:
        macro_f1 = float(mf) if mf else 0.0
        f1_abs = 0.0

    print(f"  macroF1={macro_f1:.4f} f1_abstain={f1_abs:.4f} ({wall:.0f}s)", flush=True)
    return macro_f1


def main():
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="hpo_sft_full",
    )
    # Enqueue the prior (the config that failed — so Optuna starts from known territory)
    study.enqueue_trial({"lr": 1e-5, "epochs": 1})
    # Also enqueue the strongest prior hypothesis (higher lr, more epochs)
    study.enqueue_trial({"lr": 2e-5, "epochs": 3})

    study.optimize(objective, n_trials=8, show_progress_bar=False)

    print("\n" + "=" * 60)
    print(f"BEST: macroF1={study.best_value:.4f}")
    print(f"  params: {study.best_params}")
    print("=" * 60)

    # Save results
    results = [{"number": t.number, "params": t.params, "value": t.value}
               for t in study.trials]
    json.dump(results, open(WORK / "hpo_results.json", "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
