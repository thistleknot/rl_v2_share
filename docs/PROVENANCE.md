# Provenance: slide -> concept -> results -> code

For each slide: which concept it rests on, where the result is recorded, and which module implements it.
Direction of links: slide to concept, results and code, and code to concepts.

**Concept column.** Each slide's concept is named as in the standard RL-theory section names; what was taken from each is in `INSPIRATIONS.md`. Each row is the author's reading of which concept a slide rests on.

**Paths.** `src/rlv2/...` is this repo. `dispositions.md` and `SETTLED.md` are in `docs/`. Run logs are in `docs/run-logs/`.

## Where the cited material lives

| Cited | Where |
|---|---|
| `docs/grpo_one_file.py` | The single-file trainer behind the 60-step run, the replay sampler and the crossing run. `src/rlv2/train/rl/grpo.py` is the earlier trainer. |
| chess program | `src/chess/` (`spec/trivium.spec.md`, `docs/LESSONS.md`, `docs/RL_FINDINGS.md`, code in `chessdq/`). |

Old paths cited on lab-notebook slides, and where they live now:

| Slide cites | Now |
|---|---|
| `src/build_pools_r3.py` | `src/rlv2/data/pools.py` |
| `src/reward.py` | `src/rlv2/judge/reward.py` |
| `src/judge.py` | `src/rlv2/judge/judge.py` |

## The 12-slide talk (`RL_V2_talk_10min_final.pptx`)

| Slide | Concept | Where the result is recorded | Code |
|---|---|---|---|
| 2 Result: base 4, SFT 5, first RL pass 9 | Empirical Comparison Metrics | `dispositions.md`: THE LOCKED MECHANISM AND THE UNIFORM BOARD (247). The exact 9 of 16 line was not located | `src/rlv2/eval/score_arms.py`, `src/rlv2/eval/rl_calibration_eval.py` |
| 3 What was tested | MDP Formulation Checklist; Core Objects of Decision Process | `SETTLED.md`; `dispositions.md`: skills_model PROVENANCE (390) | `src/rlv2/data/pools.py` (748 questions, `COSINE_MAX`); `configs/instrument.json` |
| 4 Verifier reward | Inverse RL / Reward Modeling | `dispositions.md`: JUDGE CHANGE (1424) | `src/rlv2/judge/judge.py`, `src/rlv2/judge/reward.py` |
| 5 Rubric, one question, seven answers | Reward Shaping and PBRS | `dispositions.md`: The reward was a constant for 87% of rollouts (827); `data/synthetic_domain_rubric.md` | `src/rlv2/judge/reward.py` |
| 6 GRPO, modified | Policy Gradients / Actor-Critic; Bias-Variance Tradeoff | `dispositions.md`: six mechanism fixes, zero lift (571) | `src/rlv2/train/rl/grpo.py` (`scale_rewards`, the Dr. GRPO option) |
| 7 SFT scores, then RL steps | Empirical Comparison Metrics | `dispositions.md`: THE SFT THESIS (747); SFT checkpoint MEASURED (792) | `src/rlv2/train/sft/train.py`, `src/rlv2/eval/half_epoch_eval.py` |
| 8 Cost of finding the peak | none | the slide's own table | `src/rlv2/common/cost_meter.py` |
| 9 60-step run | Target Networks, the KL anchor; The Deadly Triad | `dispositions.md`: rl_lr3e6 (1307); THE EVAL PROMPT IS NOT THE TRAINING PROMPT (1328); `docs/run-logs/rl_lr3e6.md` | trainer `docs/grpo_one_file.py`; eval wrapper `src/rlv2/eval/half_epoch_eval.py` (`chat_wrap`) |
| 10 Replay sampler | Replay Buffer; The Deadly Triad | `dispositions.md`: REPLAY SAMPLER (1080); rl_replay2 (1376) | trainer `docs/grpo_one_file.py` |
| 11 Scale | Bias-Variance Tradeoff | `SETTLED.md` (same-checkpoint 6/16 and 7/16); the arithmetic is in the slide notes | none |
| 12 Takeaway | n/a | slides 2 to 11 | n/a |

## The lab notebook (`RL_lab_notebook.pptx`)

| Slide | Concept | Where the result is recorded | Code |
|---|---|---|---|
| 4 Design | MDP Formulation Checklist | `SETTLED.md` | `src/rlv2/data/pools.py` |
| 5 Data | none (data construction) | `dispositions.md` | `src/rlv2/data/pools.py` |
| 6 One training step | Policy Gradients / Actor-Critic | `dispositions.md` (visit-cap entries, 09-27 to 09-29) | `docs/grpo_one_file.py` |
| 7 What the reward sees | Inverse RL / Reward Modeling | `dispositions.md` 09-26, 09-27 | `src/rlv2/judge/reward.py` (`_jev_gate`, `_jev_p_yes`), `src/rlv2/judge/judge.py` (`echoes_question`) |
| 8 Settings | n/a | `dispositions.md` | `docs/grpo_one_file.py` |
| 9 Grounding table | Policy Gradients; Boltzmann / Softmax Exploration; Replay Buffer; Target Networks; The Deadly Triad; Model-Free vs Model-Based; Inverse RL / Reward Modeling; Reward Shaping and PBRS | the table's own rows point to entries 5 to 11 | see the rows below |
| 10 Instrument | Empirical Comparison Metrics | `dispositions.md`: SIGNAL (79), NOISE (114) | `src/rlv2/train/rl/instrument.py` (the frozen instrument stamped into every receipt), `src/rlv2/eval/score_arms.py` |
| 11-12 SFT, where knowledge is written | n/a (supervised learning) | `dispositions.md`: LoRA Chain Approaches, ALL FALSIFIED (3); Proven Path (33) | `src/rlv2/train/sft/train.py` |
| 13 On-policy RL | Policy Gradients / Actor-Critic; On-Policy vs Off-Policy in One Line | `dispositions.md`: SIGNAL (79) | `src/rlv2/train/rl/grpo.py` |
| 14 What gives RL gradient | n/a | `dispositions.md`: THE SFT THESIS (747), SFT checkpoint MEASURED (792) | `src/rlv2/train/sft/train.py` |
| 15 Reward shaping | Reward Shaping and PBRS | `dispositions.md` (827) | `src/rlv2/judge/reward.py` |
| 16 KL penalty and stability | Target Networks | `dispositions.md`: rl_kl03 (1027), rl_lr3e6 (1307) | `docs/grpo_one_file.py` |
| 17 Shape of a good run | Empirical Comparison Metrics | `dispositions.md`: Which run crossed (1252) | none |
| 18 Exploration | Exploration vs Exploitation; Optimistic Initialization in Bandits | `dispositions.md`: RECORD SAMPLER AS RUN (1167) | `docs/grpo_one_file.py` |
| 19 Replay buffer | Replay Buffer | `dispositions.md`: REPLAY SAMPLER (1080), rl_replay2 (1376) | `docs/grpo_one_file.py` |
| 20 Model-based RL | Model-Free vs Model-Based; Nonlinear Learned Dynamics | `dispositions.md`: WORLD-MODEL REWARD ROUTING (925) | `src/rlv2/train/rl/prompt_world_model.py` |
| 21 Reward modeling, Jev | Inverse RL / Reward Modeling | `dispositions.md`: JUDGE CHANGE (1424), JEV GATE adopted (1480) | `src/rlv2/judge/judge.py` |
| 22 Generalization | Function Approximation Pathologies | `dispositions.md`: SWEEP60 VERDICT (1569) | n/a |
| 23 Defects | none | `dispositions.md`: six mechanism fixes (571); THE EVAL PROMPT IS NOT THE TRAINING PROMPT (1328) | `src/rlv2/eval/half_epoch_eval.py` (`chat_wrap`, the eval wrapper) |
| 28-31 Chess | Linear Function Approximation; Feature Design / Basis Functions; Eligibility Traces and TD(lambda); Temporal-Difference Learning; On-Policy vs Off-Policy in One Line; Planning Taxonomy Beyond DP | `src/chess/spec/trivium.spec.md`, `src/chess/docs/LESSONS.md`, `src/chess/docs/RL_FINDINGS.md` | `src/chess/chessdq/` |
| 32 Appendix: 60-step run | Target Networks | `docs/run-logs/rl_lr3e6.md`; `dispositions.md` (1307, 1328) | `docs/grpo_one_file.py` |
| 33 Appendix: replay sampler | Replay Buffer | `dispositions.md` (1376) | `docs/grpo_one_file.py` |

Rows for slides 1 to 3, 24 to 27 and 34 are framing, synthesis and lessons; they cite no module.

## Code to notes

| Module | Notes it rests on |
|---|---|
| `src/rlv2/train/rl/grpo.py` | VibeThinker paper (arXiv:2511.06221) in its docstring; concept Policy Gradients / Actor-Critic; `dispositions.md` (1710 comment: Dr. GRPO) |
| `src/rlv2/common/loop.py` (shared by CPT and SFT) | Raschka, *LLMs from Scratch*: ch05 `pretraining_simple.py`, ch07 `gpt_instruction_finetuning.py`, ch05 learning-rate schedulers (named in its docstring) |
| `src/rlv2/train/sft/train.py` | `dispositions.md`: THE SFT THESIS (747); its docstring (SFT on a class-balanced subset) |
| `src/rlv2/judge/reward.py`, `judge.py` | `dispositions.md`: JUDGE CHANGE (1424), JEV GATE (1480); concept Inverse RL / Reward Modeling |
| `src/rlv2/data/pools.py` | `SETTLED.md`; `dispositions.md` skills_model PROVENANCE (390) |
| `src/rlv2/common/lengths.py` | its own docstring (waste measured on a 400-row SFT subset, 2026-08-05) |

The docstrings of `grpo.py`, `judge.py`, `pools.py`, `half_epoch_eval.py` and `prompt_world_model.py` carry a `Notes:` line
with these pointers. The other modules in the table do not yet; this table carries theirs.
