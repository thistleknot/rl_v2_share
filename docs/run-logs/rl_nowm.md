<!-- NO GOVERNING SPEC. Basis: operator 2026-09-24 end-of-run procedure. -->
# Run log: rl_nowm

Written 2026-09-24 10:11. MLflow run `94a6f4a1d6df4c1fbcb8d626c8e13833` in experiment `rl_v2` (sqlite:///mlflow.db; view with `mlflow ui --backend-store-uri sqlite:///mlflow.db`).

**Verdict:** TESTED AND LOST at step 42 (operator stop). Entropy blow-up at step 19 (H 2.94, eval 0/16); pool recovered to solved 13-15 by step 35 then plateaued; fail 192-199 not falling; eval 0-2/16 for 20 steps; H swinging 1.1-4.6. Never crossed.

**Note:** world model OFF, KL OFF (beta 0), replay kept; base sft_book1e5/sft_full_epoch1

## Checkpoint

- run dir: `W:/phase_runs/rl_nowm`  (best: `W:/phase_runs/rl_nowm/best`, latest: `W:/phase_runs/rl_nowm/latest`)
- best eval 6/16 at step 13; final step 43
- trainer commit at logging: `52a47ff`

## Settings (launcher argv)

```
docs/grpo_one_file.py --base sft_book1e5/sft_full_epoch1 --mix data/rl/r3_rl.jsonl --out W:/phase_runs/rl_nowm --rounds 100 --steps 1 --grad-accum 16 --lr 1e-5 --rollouts 4 --temperature 0.8 --top-p 0.9 --beta 0 --logprob-norm mean --grad-clip 25.0 --inner-epochs 2 --raw-reward --judge-lane r3 --eval-every 1 --eval-reads 1 --seed 7 --patience 0 --hq-floor 0.9 --lq-reward 0.5 --is-beta 0 --scale-rewards batch --adv-flat 0 --adv-eps 0.01 --straddle-patience 20
```

## Plot

`W:/phase_runs/rl_nowm/final_plot.png`

## Per-step (last-3-visits counters)

| step | eval | solved | straddle | fail | H | reward | len |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 0 | 28 | 55 | 1.61 | 0.297 | 203 |
| 2 | 2 | 1 | 33 | 70 | 1.46 | 0.406 | 154 |
| 3 | 2 | 1 | 41 | 75 | 1.49 | 0.351 | 139 |
| 4 | 2 | 1 | 44 | 84 | 1.31 | 0.367 | 151 |
| 5 | 2 | 1 | 46 | 101 | 1.19 | 0.379 | 156 |
| 6 | 1 | 1 | 48 | 107 | 0.96 | 0.377 | 125 |
| 7 | 3 | 1 | 53 | 115 | 1.23 | 0.334 | 145 |
| 8 | 1 | 1 | 54 | 128 | 1.05 | 0.335 | 173 |
| 9 | 4 | 1 | 58 | 134 | 1.06 | 0.332 | 170 |
| 10 | 3 | 0 | 62 | 139 | 0.87 | 0.426 | 180 |
| 11 | 3 | 2 | 65 | 141 | 1.06 | 0.446 | 165 |
| 12 | 3 | 4 | 65 | 144 | 0.79 | 0.341 | 147 |
| 13 | 6 | 4 | 68 | 144 | 0.81 | 0.523 | 162 |
| 14 | 5 | 4 | 70 | 146 | 0.66 | 0.309 | 169 |
| 15 | 3 | 4 | 71 | 149 | 0.77 | 0.439 | 187 |
| 16 | 3 | 4 | 71 | 151 | 0.76 | 0.369 | 166 |
| 17 | 1 | 5 | 73 | 154 | 0.92 | 0.385 | 182 |
| 18 | 1 | 5 | 76 | 152 | 1.61 | 0.356 | 208 |
| 19 | 0 | 5 | 75 | 154 | 2.94 | 0.127 | 294 |
| 20 | 0 | 5 | 71 | 159 | 2.98 | 0.168 | 234 |
| 21 | 0 | 6 | 50 | 180 | 3.67 | 0.210 | 216 |
| 22 | 0 | 4 | 44 | 187 | 2.53 | 0.261 | 236 |
| 23 | 0 | 3 | 35 | 194 | 1.84 | 0.282 | 182 |
| 24 | 0 | 3 | 27 | 200 | 3.63 | 0.241 | 252 |
| 25 | 1 | 4 | 25 | 206 | 2.63 | 0.255 | 174 |
| 26 | 1 | 6 | 21 | 212 | 3.37 | 0.288 | 229 |
| 27 | 0 | 7 | 19 | 214 | 2.64 | 0.265 | 172 |
| 28 | 0 | 7 | 16 | 216 | 2.99 | 0.171 | 356 |
| 29 | 0 | 9 | 16 | 214 | 2.25 | 0.279 | 217 |
| 30 | 0 | 8 | 17 | 214 | 3.90 | 0.242 | 375 |
| 31 | 1 | 9 | 21 | 208 | 3.05 | 0.237 | 268 |
| 32 | 0 | 11 | 18 | 209 | 4.77 | 0.316 | 293 |
| 33 | 0 | 12 | 20 | 208 | 2.31 | 0.332 | 244 |
| 34 | 0 | 13 | 23 | 204 | 1.13 | 0.305 | 169 |
| 35 | 2 | 13 | 25 | 200 | 1.22 | 0.417 | 130 |
| 36 | 2 | 14 | 23 | 200 | 1.81 | 0.313 | 225 |
| 37 | 2 | 14 | 28 | 197 | 2.69 | 0.310 | 259 |
| 38 | 1 | 10 | 32 | 196 | 3.43 | 0.285 | 266 |
| 39 | 0 | 11 | 33 | 194 | 2.81 | 0.357 | 153 |
| 40 | 0 | 14 | 32 | 192 | 3.22 | 0.421 | 231 |
| 41 | 1 | 15 | 27 | 197 | 1.09 | 0.418 | 188 |
| 42 | 0 | 13 | 26 | 199 | 4.56 | 0.308 | 213 |
| 43 | 0 | 15 | 26 | 197 | 4.14 | 0.354 | 162 |
