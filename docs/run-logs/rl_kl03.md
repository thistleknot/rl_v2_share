<!-- NO GOVERNING SPEC. Basis: operator 2026-09-24 end-of-run procedure. -->
# Run log: rl_kl03

Written 2026-09-24 20:04. MLflow run `9bf3d03e4122411ca4658614e02b54a8` in experiment `rl_v2` (sqlite:///mlflow.db; view with `mlflow ui --backend-store-uri sqlite:///mlflow.db`).

**Verdict:** TESTED AND LOST on eval (best 5/16 at step 1 = SFT deposit; final 2/16). Stopped by operator at step 33 on blow-up: kl_raw 0.61 -> 2.94 in one step, H 1.90 -> 2.48, gnorm 906, CJK garbage NO-EOS samples. Pool: solved 3 -> 28 (peak steps 25/27) -> 17, straddle 86 -> 77 -> 90, fail 150 -> 131; solved never crossed straddle.

**Note:** Crossing recipe, world model OFF, KL beta 0.3 k3 to SFT ref, bands 0/0.5/1 hq 0.9, grad-clip 1.0, lr 1e-5, 1 inner pass, temperature 0.9, replay buffer. Stable 32 steps, fit the pool (solved 28/240), no transfer to eval, then KL runaway.

## Checkpoint

- run dir: `W:/phase_runs/rl_kl03`  (best: `W:/phase_runs/rl_kl03/best`, latest: `W:/phase_runs/rl_kl03/latest`)
- best eval 5/16 at step 1; final step 33
- trainer commit at logging: `a33f8f5`

## Settings (launcher argv)

```
docs/grpo_one_file.py --base sft_book1e5/sft_full_epoch1 --mix data/rl/r3_rl.jsonl --out W:/phase_runs/rl_kl03 --rounds 100 --steps 1 --grad-accum 16 --lr 1e-5 --rollouts 4 --temperature 0.9 --top-p 0.9 --beta 0.3 --kl-estimator k3 --logprob-norm sum --grad-clip 1.0 --inner-epochs 1 --quality-bands --judge-lane r3 --eval-every 1 --eval-reads 1 --seed 7 --patience 0 --hq-floor 0.9 --lq-reward 0.5 --is-beta 0 --scale-rewards batch --adv-flat 20 --adv-eps 0.01 --straddle-patience 20
```

## Plot

`W:/phase_runs/rl_kl03/final_plot.png`

## Per-step (last-3-visits counters)

| step | eval | solved | straddle | fail | H | reward | len |
|---|---|---|---|---|---|---|---|
| 1 | 5 | 1 | 50 | 98 | 1.71 | 0.234 | 190 |
| 2 | 3 | 1 | 57 | 119 | 1.68 | 0.305 | 149 |
| 3 | 2 | 1 | 64 | 130 | 1.83 | 0.258 | 170 |
| 4 | 2 | 1 | 69 | 137 | 1.67 | 0.422 | 96 |
| 5 | 2 | 1 | 75 | 145 | 1.80 | 0.344 | 131 |
| 6 | 2 | 1 | 81 | 147 | 1.74 | 0.312 | 114 |
| 7 | 2 | 2 | 83 | 150 | 1.76 | 0.422 | 152 |
| 8 | 2 | 2 | 87 | 149 | 1.77 | 0.367 | 109 |
| 9 | 2 | 3 | 88 | 147 | 1.50 | 0.461 | 104 |
| 10 | 2 | 3 | 93 | 143 | 1.70 | 0.430 | 125 |
| 11 | 1 | 6 | 88 | 145 | 1.59 | 0.414 | 104 |
| 12 | 2 | 7 | 89 | 144 | 1.71 | 0.336 | 128 |
| 13 | 2 | 12 | 82 | 145 | 1.69 | 0.383 | 94 |
| 14 | 2 | 13 | 83 | 143 | 1.73 | 0.383 | 112 |
| 15 | 2 | 15 | 86 | 139 | 1.78 | 0.484 | 110 |
| 16 | 1 | 15 | 89 | 136 | 1.78 | 0.414 | 127 |
| 17 | 2 | 17 | 84 | 139 | 1.54 | 0.539 | 113 |
| 18 | 2 | 19 | 82 | 138 | 1.67 | 0.484 | 129 |
| 19 | 2 | 22 | 80 | 137 | 1.64 | 0.406 | 119 |
| 20 | 1 | 23 | 84 | 132 | 1.81 | 0.453 | 129 |
| 21 | 1 | 23 | 86 | 131 | 1.84 | 0.453 | 110 |
| 22 | 1 | 24 | 84 | 132 | 1.75 | 0.438 | 128 |
| 23 | 1 | 27 | 81 | 132 | 1.76 | 0.547 | 117 |
| 24 | 2 | 27 | 78 | 135 | 1.73 | 0.414 | 129 |
| 25 | 1 | 28 | 77 | 135 | 1.82 | 0.469 | 121 |
| 26 | 1 | 27 | 82 | 131 | 1.75 | 0.438 | 141 |
| 27 | 1 | 28 | 79 | 133 | 1.72 | 0.492 | 111 |
| 28 | 1 | 26 | 83 | 131 | 1.56 | 0.438 | 123 |
| 29 | 1 | 24 | 82 | 134 | 1.85 | 0.445 | 134 |
| 30 | 1 | 20 | 88 | 132 | 1.74 | 0.430 | 121 |
| 31 | 2 | 17 | 90 | 133 | 1.68 | 0.406 | 134 |
| 32 | 2 | 17 | 90 | 133 | 1.90 | 0.453 | 160 |
| 33 | 2 | 17 | 89 | 133 | 2.48 | 0.453 | 154 |
