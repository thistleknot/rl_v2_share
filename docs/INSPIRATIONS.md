# Inspirations

Every idea borrowed from outside, by its short name, with what was taken and where it shows up. Concept names are the standard
RL-theory section names. Results of each trial are in `dispositions.md`; slide numbers are in `PROVENANCE.md`.

## RL theory concepts

| Concept | The idea taken | Where it shows up |
|---|---|---|
| MDP Formulation Checklist, Core Objects of Decision Process | Write the problem down as state, action, reward, transition and terminal rule before choosing an algorithm. Here: a state is the prompt plus the tokens so far, an action is the next token, one judged answer is one trajectory. | `design.md`; RL deck slide 4 |
| Deterministic Transition Simplification | Appending a token is deterministic, so each (state, action) has one successor and no transition matrix is needed. | `design.md` |
| Terminal States and Absorbing-State Convention | End-of-sequence is the absorbing terminal state; all reward is paid at completion. | `design.md`; `src/rlv2/judge/reward.py` |
| Policy Gradients / Actor-Critic | Learn the policy directly and drop the critic. The group of sampled answers supplies the baseline: advantage is reward minus the group mean (no std scaling). | `src/rlv2/train/rl/grpo.py`; RL deck slides 6 and 9 |
| Policy Parametrization / Softmax | The language model's output distribution is the stochastic policy; temperature is its exploration knob. | `design.md`; `src/rlv2/train/rl/grpo.py` |
| Boltzmann / Softmax Exploration | Draw training records with probability proportional to exp(priority / T), priority being the record's reward spread. | `dispositions.md` (record sampler entry); RL deck slide 17 |
| Exploration vs Exploitation, Optimistic Initialization in Bandits | Treat an unseen record as promising: draw unseen records first, then by uncertainty. | RL deck slide 17 |
| Bandits as the Simplest Decision Problem | Each prompt is a one-step decision with a group of answers, a bandit over answer trajectories. | `design.md`; `src/rlv2/train/rl/prompt_world_model.py` |
| Replay Buffer | Reuse stored experience and prioritise by usefulness ("flashcard" priority). The reuse variant, stored judged answers with a clipped ratio, missed its pre-set bar. | RL deck slides 18 and 28; `dispositions.md` (REPLAY SAMPLER, rl_replay2) |
| Target Networks | A slow anchor stabilises learning. Here the anchor is the SFT checkpoint, held with a KL penalty (beta). | RL deck slides 15 and 27 |
| The Deadly Triad | Off-policy data, function approximation and bootstrapping together diverge. Stale replayed answers on a 0.8B network ended in a KL runaway. | RL deck slide 18; RL deck slide 10 |
| Model-Free vs Model-Based; Dyna-Q | Learn a model of the environment from experience and use it for extra updates. Here the model is an ensemble that predicts the judge's verdict. | `src/rlv2/train/rl/prompt_world_model.py`; RL deck slide 19 |
| Nonlinear Learned Dynamics, Function Approximation Pathologies | A learned model can be confidently wrong where it has no data; ensemble disagreement stands in for uncertainty. | `src/rlv2/train/rl/prompt_world_model.py` |
| Inverse RL / Reward Modeling | Infer the reward from a judge instead of writing it by hand. Here: a rubric-driven LLM verifier, later a small yes/no judge (Jev) that needed a guard. | `src/rlv2/judge/`; RL deck slides 7 and 20 |
| Reward Shaping and PBRS | Shaped reward changes what is learned unless it is potential-based. The banded reward gated by correctness paid zero on 87% of answers. | `src/rlv2/judge/reward.py`; RL deck slide 14 |
| Empirical Comparison Metrics, Bias-Variance Tradeoff | Compare methods on one instrument, report the same-checkpoint spread, and treat one noisy rollout as one sample. | `src/rlv2/eval/`; RL deck slides 10 and 30 |
| Truncated Policy Iteration | Partial evaluation then improvement: one epoch per round rather than evaluating to convergence. | `design.md` |
| Linear Function Approximation, Feature Design / Basis Functions | A linear evaluation over hand-built features: 897 features, piece-square planes plus attack coverage. | `src/chess/chessdq/`; chess deck slide 1 |
| Temporal-Difference Learning, Eligibility Traces and TD(lambda) | Bootstrap from the next estimate and blend n-step returns with a trace weight. | `src/chess/spec/trivium.spec.md`; chess deck slide 2 |
| On-Policy vs Off-Policy | Which policy generated the data decides which updates are valid. | RL deck slide 12; chess deck slide 2 |
| Planning Taxonomy Beyond DP | Search (alpha-beta, PUCT) as planning at play time, with a learned value at the leaves. | `src/chess/chessdq/`; chess deck |

## Methods and papers

| Source | The idea taken | Where it shows up |
|---|---|---|
| Raschka, *Build a Large Language Model (From Scratch)*, ch05 and ch07 | The pre-training loop, the instruction-finetuning loop (pad targets with -100), warm-up then cosine learning-rate schedule. | `src/rlv2/common/loop.py`; `src/rlv2/train/sft/loop.py` |
| Raschka, *Build a Reasoning Model (From Scratch)*, ch06 and ch07 | GRPO loss and rollout loop. | `src/rlv2/train/rl/grpo.py` |
| VibeThinker (arXiv:2511.06221) | Split SFT and RL by objective (Spectrum-to-Signal) and weight prompts by closeness to 50% accuracy (MGPO). | `src/rlv2/train/rl/grpo.py`; RL deck slide 6 |
| Dr. GRPO | Remove the per-group std division from GRPO's advantage (mean subtraction only). | `src/rlv2/train/rl/grpo.py`; RL deck slide 6 |
| Prioritized experience replay (Schaul et al., 2015) | Sample experience in proportion to its learning value, with importance weights. | RL deck slide 18 |
| EntiGraph | Synthetic continued pre-training text from entity relations. | `src/rlv2/data/entigraph.py` |

## Chess program

The chess agent's own design lives in `src/chess/`: `spec/trivium.spec.md` (target = lambda-return + search value + outcome,
annealed), `docs/LESSONS.md` and `docs/RL_FINDINGS.md` (what failed and what worked). The concepts it uses are the last four
rows of the first table.
