<!-- NO GOVERNING SPEC. Basis: operator 2026-09-24 end-of-run procedure. -->
# Run log: rl_lr3e6

Written 2026-09-25 16:45. MLflow run `5d9c81efdf244c04956b41f669627c8c` in experiment `rl_v2` (sqlite:///mlflow.db; view with `mlflow ui --backend-store-uri sqlite:///mlflow.db`).

**Verdict:** TESTED AND LOST on eval (best 5/16 at steps 34-39 and 61 vs start 4, within the +-1 same-checkpoint wobble; final 1/16). Operator declared collapse at step 66. Pool: solved 0 -> 25 (step 50-55) -> 11, straddle held 95-105 from step 16-60, fail 150 -> 115 (step 53) -> 141; neither straddle>fail nor solved>straddle reached. KL runaway: kl_raw 0.23 (60) -> 0.90 (61) -> 7.34 (63), H 1.7 -> 2.1, mixed-script garbage and NO-EOS samples from step 62.

**Note:** rl_kl03 recipe with lr 3e-6 instead of 1e-5 (KL beta 0.3 k3, bands 0/0.5/1 hq 0.9, grad-clip 1.0, 1 inner pass, temp 0.9, world model OFF). Resumed in place at step 25 (after abandoned replay-sampler detour) and step 46 (exhaustion gate stop; step-46 weights + step-45 optimizer, --adv-flat 0). Stable 60 steps, then KL runaway at 61-63.

## Checkpoint

- run dir: `W:/phase_runs/rl_lr3e6`  (best: `W:/phase_runs/rl_lr3e6/best`, latest: `W:/phase_runs/rl_lr3e6/latest`)
- best eval 5/16 at step 34; final step 66
- trainer commit at logging: `15b4ca3`

## Settings (launcher argv)

```
docs/grpo_one_file.py --base W:/phase_runs/rl_lr3e6/resume_step46 --ref-base sft_book1e5/sft_full_epoch1 --carry-visits W:/phase_runs/rl_lr3e6/visit_log.json --mix data/rl/r3_rl.jsonl --out W:/phase_runs/rl_lr3e6 --rounds 100 --steps 1 --grad-accum 16 --lr 3e-6 --rollouts 4 --temperature 0.9 --top-p 0.9 --beta 0.3 --kl-estimator k3 --logprob-norm sum --grad-clip 1.0 --inner-epochs 1 --quality-bands --judge-lane r3 --eval-every 1 --eval-reads 1 --seed 7 --patience 0 --hq-floor 0.9 --lq-reward 0.5 --is-beta 0 --scale-rewards batch --adv-flat 0 --adv-eps 0.01 --straddle-patience 20
```

## Plot

`W:/phase_runs/rl_lr3e6/final_plot.png`

## Per-step (last-3-visits counters)

| step | eval | solved | straddle | fail | H | reward | len |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 1 | 39 | 85 | 1.63 | 0.320 | 195 |
| 2 | 3 | 2 | 52 | 109 | 1.94 | 0.289 | 213 |
| 3 | 3 | 2 | 62 | 128 | 1.58 | 0.297 | 164 |
| 4 | 3 | 2 | 69 | 131 | 1.80 | 0.297 | 155 |
| 5 | 3 | 0 | 73 | 139 | 1.61 | 0.398 | 132 |
| 6 | 2 | 0 | 82 | 141 | 1.79 | 0.273 | 163 |
| 7 | 3 | 0 | 83 | 144 | 1.70 | 0.445 | 118 |
| 8 | 3 | 0 | 86 | 147 | 1.61 | 0.367 | 143 |
| 9 | 3 | 0 | 87 | 150 | 1.70 | 0.375 | 113 |
| 10 | 3 | 0 | 91 | 148 | 1.62 | 0.461 | 109 |
| 11 | 3 | 0 | 96 | 143 | 1.70 | 0.438 | 116 |
| 12 | 3 | 0 | 95 | 144 | 1.58 | 0.344 | 111 |
| 13 | 3 | 2 | 94 | 144 | 1.92 | 0.406 | 91 |
| 14 | 3 | 2 | 100 | 138 | 1.59 | 0.430 | 118 |
| 15 | 3 | 1 | 103 | 136 | 1.54 | 0.445 | 116 |
| 16 | 3 | 3 | 101 | 136 | 1.81 | 0.445 | 99 |
| 17 | 3 | 5 | 97 | 138 | 1.77 | 0.328 | 129 |
| 18 | 3 | 5 | 97 | 138 | 1.87 | 0.383 | 143 |
| 19 | 3 | 5 | 101 | 134 | 1.90 | 0.328 | 130 |
| 20 | 2 | 5 | 103 | 132 | 1.67 | 0.453 | 120 |
| 21 | 2 | 6 | 101 | 133 | 1.59 | 0.359 | 116 |
| 22 | 2 | 7 | 104 | 129 | 1.80 | 0.367 | 134 |
| 23 | 2 | 9 | 104 | 127 | 1.91 | 0.336 | 115 |
| 24 | 2 | 10 | 103 | 127 | 1.78 | 0.484 | 132 |
| 25 | 2 | 12 | 102 | 126 | 1.78 | 0.398 | 120 |
| 26 | 2 | 11 | 103 | 126 | 1.61 | 0.453 | 110 |
| 27 | 2 | 11 | 103 | 126 | 1.88 | 0.414 | 142 |
| 28 | 2 | 11 | 104 | 125 | 1.56 | 0.438 | 97 |
| 29 | 3 | 11 | 104 | 125 | 1.72 | 0.492 | 95 |
| 30 | 4 | 11 | 100 | 128 | 1.77 | 0.469 | 137 |
| 31 | 4 | 12 | 101 | 126 | 1.52 | 0.367 | 92 |
| 32 | 4 | 14 | 101 | 125 | 1.49 | 0.477 | 81 |
| 33 | 4 | 13 | 99 | 128 | 1.63 | 0.438 | 117 |
| 34 | 5 | 12 | 101 | 127 | 1.75 | 0.484 | 107 |
| 35 | 5 | 12 | 100 | 128 | 1.59 | 0.422 | 105 |
| 36 | 5 | 13 | 101 | 126 | 1.68 | 0.398 | 140 |
| 37 | 5 | 12 | 103 | 125 | 1.82 | 0.477 | 141 |
| 38 | 5 | 13 | 104 | 123 | 1.70 | 0.383 | 116 |
| 39 | 5 | 14 | 103 | 123 | 1.64 | 0.391 | 107 |
| 40 | 3 | 13 | 105 | 122 | 1.67 | 0.539 | 129 |
| 41 | 3 | 14 | 105 | 121 | 1.70 | 0.438 | 121 |
| 42 | 3 | 16 | 103 | 121 | 1.73 | 0.398 | 102 |
| 43 | 3 | 16 | 103 | 121 | 1.78 | 0.391 | 127 |
| 44 | 3 | 18 | 101 | 121 | 1.73 | 0.445 | 127 |
| 45 | 4 | 20 | 101 | 119 | 1.64 | 0.469 | 120 |
| 46 | 4 | 21 | 101 | 118 | 1.72 | 0.469 | 140 |
| 47 | 4 | 22 | 98 | 120 | 1.63 | 0.414 | 116 |
| 48 | 4 | 22 | 99 | 119 | 1.63 | 0.414 | 133 |
| 49 | 4 | 23 | 98 | 119 | 1.67 | 0.438 | 114 |
| 50 | 4 | 25 | 96 | 119 | 1.68 | 0.477 | 121 |
| 51 | 3 | 25 | 96 | 119 | 1.65 | 0.492 | 117 |
| 52 | 2 | 25 | 99 | 116 | 1.86 | 0.305 | 134 |
| 53 | 2 | 25 | 100 | 115 | 1.70 | 0.484 | 121 |
| 54 | 1 | 26 | 99 | 115 | 1.69 | 0.430 | 127 |
| 55 | 2 | 25 | 99 | 116 | 1.71 | 0.453 | 108 |
| 56 | 4 | 24 | 97 | 119 | 1.77 | 0.438 | 121 |
| 57 | 3 | 24 | 95 | 121 | 1.67 | 0.508 | 93 |
| 58 | 2 | 24 | 95 | 121 | 1.72 | 0.391 | 120 |
| 59 | 3 | 24 | 95 | 121 | 1.94 | 0.359 | 146 |
| 60 | 4 | 24 | 95 | 121 | 1.68 | 0.500 | 108 |
| 61 | 5 | 22 | 96 | 122 | 1.92 | 0.492 | 105 |
| 62 | 3 | 19 | 96 | 125 | 2.04 | 0.344 | 109 |
| 63 | 2 | 17 | 91 | 131 | 1.89 | 0.375 | 101 |
| 64 | 2 | 16 | 85 | 138 | 1.95 | 0.484 | 80 |
| 65 | 1 | 12 | 86 | 140 | 2.08 | 0.406 | 83 |
| 66 | 1 | 11 | 88 | 141 | 1.94 | 0.477 | 74 |
