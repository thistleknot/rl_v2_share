# RL_V2 — GRPO + QLoRA Fine-Tuning: Theory and Design

## Purpose

This is the canonical reference for the RL_V2 project. It grounds every design decision in RL theory from RL theory concepts and in the VibeThinker-3B paper (Sections 2.2.1 and 2.2.2). Both sources are cited by section name throughout.

Use `FAQ.md` for implementation decisions and edge cases.

---

## The harness at a glance

One pipeline, **data -> CPT (optional, with EntiGraph) -> SFT -> RL**, on synthetic data, optionally mixed with RedPajama. One entrypoint:

```
python -m rlv2 data  pull | chunk | entigraph | skills-sft | qa | pools | cpt-mix | rl-mix
python -m rlv2 cpt   --full-param --lr ... --epochs ... --out ...
python -m rlv2 sft   --full-param --base ... --out ...
python -m rlv2 rl    --base ... --arms ... --out ...
python -m rlv2 eval  --model ... --tag ...
python -m rlv2 hpo   sft | rl
```

Each stage keeps the flags it always had (`python -m rlv2 <stage> --help`). `run_synthetic.ps1 -> run_cpt.ps1 -> run_sft.ps1 -> run_rl.ps1` chain them. Everything that is not on this path (probes, one-off builders, the toy `rl_ladder`, the old stage chain) lives in `archive/`, mirrored by its original directory (`archive/src`, `archive/sprawl`, `archive/tools`, `archive/harness`).

| Package | Objective |
|---|---|
| `rlv2/common` | log-band length filter, Box-Cox allocation and token caps, cost and VRAM guards, regime stratification |
| `rlv2/judge` | the judge, the reward, the answer verifiers, the hosted and local LLM clients |
| `rlv2/data` | synthetic data: chunking, EntiGraph, per-skill QA, gated pools, RedPajama pull, CPT and RL mixes |
| `rlv2/cpt` | continued pre-training |
| `rlv2/sft` | supervised fine-tuning (bin-packed batches, manifold sampler) |
| `rlv2/rl` | GRPO / MGPO trainer and the guards around the reward |
| `rlv2/eval` | frozen-eval scoring, half-epoch eval, multi-arm scoring |
| `rlv2/hpo` | Optuna gates for SFT and RL |

### Book code first, our additions marked

Target base for each stage. **CPT and SFT run on one shared plain-PyTorch loop (`rlv2/common/loop.py`, after the two book bases below); RL is the original code, regrouped, not yet rebuilt on its base.**

SFT parity, measured 2026-10-05 (8 steps, same data order, seed 0, lr 1e-4, weight decay 0.1, the old HF-Trainer path from the previous commit against the new loop): all 8 step losses and gradient norms agree to 4 significant figures (step 1: 4.121 / 4.1209; step 8: 2.256 / 2.2556), and the saved weights differ by 0.00000 against 3.15 of total movement from the base. The old path run twice against itself differs by 0.00000 too, so the comparison has no noise floor to hide in. It covers one epoch of 8 steps on one seed, not a full run.

CPT parity, same day (6 optimizer steps on `data/rl/cpt_mix_r3.jsonl`, gradient accumulation 4, fp16 autocast with the loss scaler, 8-bit AdamW, seed 0, lr 1e-4, weight decay 0.1): all 6 step losses and gradient norms agree (3.48 / 3.4798 ... 3.642 / 3.6424; grad norm 9.095 / 9.095, 11.01 / 11.014, 11.69 / 11.687), the loss scaler skips the same first 3 steps in both, and the saved weights differ by 0.00000 against 4.25 of movement from the base. That parity needed two Trainer behaviours reproduced: it counted labelled tokens before the one-token shift (CPT's first token per sequence was counted though nothing predicts it, so the logged loss read ~0.3% low), and `accelerate` divided the gradient by `grad_accum` a second time. **Both were then corrected on 2026-10-07** (CPT is an optional lane we keep, so it should train on the right math): the window loss is now divided by the prediction targets in the window and the gradient is the true window mean. SFT is unaffected (its prompt is masked and it does not accumulate), and the SFT numbers above still hold. **The corrected CPT loop has not been re-run against the old path:** the host's Windows commit limit (about 8 GiB free, 76 python processes) blocked every GPU run that day, so what the correction changes in a CPT run is unmeasured; unit tests pin the corrected math (`tests/test_common_loop.py`). CPT runs from 2026-10-07 on are therefore not strictly comparable with earlier CPT runs. Not parity-tested: the LoRA branches of CPT and SFT, `max_steps` stops, and more than one epoch.

| Stage | Book base (under `docs/`) |
|---|---|
| CPT | *LLMs from Scratch* (Raschka, Manning) ch05 `03_bonus_pretraining_on_gutenberg/pretraining_simple.py` and `04_learning_rate_schedulers` |
| SFT | ch07 `01_main-chapter-code/gpt_instruction_finetuning.py`; *Hands-On LLMs* (O'Reilly) ch12 for the Hugging Face / LoRA variant |
| RL | *Build a Reasoning Model* (Raschka) ch06-07 (`docs/code.zip` `reasoning_from_scratch/ch06.py`, `ch07.py`) is the reference for the GRPO loss, **not adopted**: `rlv2/rl/grpo.py` drives TRL's `GRPOTrainer` directly (advantage, ratio clipping, KL, the DAPO loss), and replacing it would replace library code with a from-scratch copy of it, with no way to check parity short of a live judge. `docs/grpo_one_file.py` is the tagged one-file reading of this stage |
| Synthetic data | *LLMs from Scratch* ch07 `05_dataset-generation` is the reference for generation, **not adopted**: the near-duplicate finder in `02_dataset-utilities` is TF-IDF cosine, while the contamination gate in `rlv2/data/pools.py` is embedding cosine against the eval set, so swapping it would change which records enter the training pools |
| Chunking | no book base worth taking; `rlv2/data/chunking.py` is ours |

What we add on top, and where:

| Addition | Where | What it does |
|---|---|---|
| Log-MAD length filter | `rlv2/common/lengths.py` `log_mad_band`, `filter_outliers` | keeps records whose token length is within median +/- 2 x (scaled MAD), both taken on log lengths (raw lengths are right-skewed); no filtering when the MAD is 0. **SFT only** (`--band-sigmas`); CPT does not call it |
| Box-Cox allocation and token caps | `rlv2/data/pull_datasets.py` `calculate_allocations`; `rlv2/common/token_budget.py` `cap_for` | stratified record counts; per-(regime, domain) max-new-tokens from a Box-Cox fit of gold-answer lengths. The token caps are used by `rl/red_queen_validity`, not by the trainer |
| Rolling latency budget | `rlv2/judge/judge.py` `LatencyBudget` | exponentially weighted mean and variance of judge latency; timeout = min(60, max(15, mean + 4 sd)) seconds. **Judge calls only** |
| CPT and SFT loop additions | `rlv2/common/loop.py` header (`OURS:` list) | variable-size token-budgeted batches, 8-bit AdamW with 32-bit embedding state, gradient checkpointing, fp16 autocast with a loss scaler (CPT), accumulation normalised by the window's prediction targets, clipping every step, the `transformers` scheduler (the HPO sweeps nine types), label smoothing, Trainer callbacks kept through `CallbackHandler` |
| Echo guard | `rlv2/judge/judge.py` `echoes_question` | an answer that opens with the question's own first 6 words scores 0 (RL under the Jev judge learned to echo the question) |

**Not built:** a rolling timeout or rolling max length for *policy generation*. The trainer uses a flat `MAX_COMPLETION = 512` tokens (`rlv2/rl/grpo.py`) and a fixed generation timeout in `rlv2/data/pools.py`.

### Code outline (generated)

Regenerate with `python scripts/ast_outline.py`; `--check` fails if this block is stale.

<!-- AST-OUTLINE:BEGIN (generated by scripts/ast_outline.py; do not edit) -->

**`rlv2/`**

- `__main__.py` (66 lines) the one entrypoint: python -m rlv2 <stage> [stage args].
  - `def main`
  - 1 private helpers
- `config.py` (67 lines) 
- `model_config.py` (84 lines) SINGLE SOURCE OF TRUTH for models + the output-cleanliness CONTRACT (:ModelOutputContract:).
  - `def is_reasoning`
  - `def strip_think`: Remove <think> tags AND their content (residual reasoning bleed).
  - `def assert_clean_text_model`: ENFORCEMENT: a raw reasoning model for FREE TEXT (no schema) bleeds CoT — the contract forbids it.
  - `def clean_generate`: Contract-enforced ollama generation. Free text (schema=None) with a reasoning model RAISES. Always applies

**`rlv2/common/`**

- `common/cost_meter.py` (687 lines) the second ruler: what did this lift COST? (:EfficiencyAccounting:)
  - `class VramCeilingExceeded`: Raised when a stage would cross the hard VRAM wall. Never caught — it must stop the run.
  - `def enforce_vram_ceiling`: Cap the torch allocator to the free window and abort if that is not enough to work in.
  - `def assert_gpu_exclusive`: Refuse to start a GPU stage while ANOTHER training run is alive.
  - `def peak_mib`: Allocator high-water mark in MiB.
  - `class ThroughputCollapsed`: Raised when per-step time breaches the absolute floor. Never caught -- it must stop the run.
  - `def throughput_floor`: A TrainerCallback that kills the run when a single step exceeds a SELF-CALIBRATED time floor.
  - `def current_rung`: The rung this stage should run at, read from disk.
  - `def trip`: Record a VRAM trip and advance the ladder. Returns (next_index, cfg, exhausted).
  - `def is_oom`: True for a real VRAM exhaustion, including our own enforced ceiling.
  - `def run_with_downscale`: Run train_fn(cfg, resume=bool) under the ladder, downscaling on each VRAM trip.
  - `class CostMeter`: Context manager recording wall-clock and peak VRAM for one stage.
  - `def trainable_params`: Parameters that will actually receive gradients. The capacity a method paid for.
  - `def mean_prompt_tokens`: Mean prompt length in tokens — the per-query inference tax (RAG's hidden cost).
  - `def cost_row`: One arm's cost line. Efficiency ratios are suppressed unless the gain is statistically resolved.
  - `def render_cost_table`: Fixed-width cost table. Same column order everywhere so arms are comparable across hypotheses.
  - `def save`: Persist a cost record next to the accuracy JSON. On-disk is primary (repo convention).
  - `def assert_adapter_trained`: Require -- `path` holds a saved PEFT adapter. Guarantee -- raises unless B moved off zero.
  - `def assert_composition`: Require -- `declared` and `actual` map the same ratio names to floats. Guarantee -- raises on
  - `def report_peak_vram`: Emit the ACTUAL peak allocator high-water mark, tagged so log scrapers can find it.
  - 1 private helpers
- `common/lengths.py` (151 lines) log-normal median+2MAD filter, then 4 quantile partitions, to stop paying
  - `def log_mad_band`: Return (lo, hi) token-length band from median +- sigmas*MAD in LOG space.
  - `def filter_outliers`: Split indices into (kept, dropped) by the log-MAD band. DROPS DATA -- report the count.
  - `def quantile_blocks`: Group indices into n_blocks length-alike blocks by quantile. BATCHING ONLY -- drops nothing.
  - `def report`: Print the measured saving. Call before training so the number is on the record, not assumed.
  - 1 private helpers
- `common/loop.py` (191 lines) the training loop CPT and SFT share, as plain PyTorch.
  - `def collate`: Book custom_collate_fn, for records that carry `input_ids`, `attention_mask` and `labels`.
  - `def target_tokens`: Next-token targets that carry loss: labels not equal to -100 AFTER the one-token shift.
  - `def calc_loss_batch`: Book calc_loss_batch: cross entropy of the next token, padding and prompt ignored.
  - `def make_optimizer`: AdamW, 8-bit (`adamw_8bit`) or plain (`adamw_torch`); decay on weights only.
  - `def train_batches`: Train over `batches`: one optimizer step per `grad_accum` batches, for at most `num_steps` steps.
- `common/stratify.py` (305 lines) one joint regime x domain stratification schema, shared by SFT,
  - `def regime_of`: R1/R2/R3/R4 from (context present, answer present).
  - `def is_abstain`
  - `def cell_key`
  - `def cell_counts`
  - `def proportional_allocations`: RAW-proportional allocation: draw_i = round(n * size_i / total), largest-remainder
  - `def stratified_sample`: Draw n records stratified over joint (regime, domain) cells. lam=None -> raw
  - `def allocation_table`: Rows (cell, raw_count, raw_pct, draw_count, draw_pct) for the joint
  - `def print_allocation_table`: Print the joint regime x domain sampling ratios (raw vs drawn). A core,
  - `class Corpus`: Single source of truth: the corpus (data/raw/all.jsonl) + the split manifest +
- `common/token_budget.py` (107 lines) :GenerationBudget: per-(regime x domain) cell max_new_tokens.
  - `def compute`: Fit the per-cell cap from CORPUS gold answers. Returns {"R#|Domain": max_new}.
  - `def load`: Cached per-cell caps; compute+cache on first use.
  - `def cap_for`: Per-cell cap, clamped by the caller's ceiling (gate=low, recall-eval=high).
  - `def filter_truncating`: :TruncationFreeSampling: (data half) — drop any record whose GOLD-answer token count
  - `def batch_completion_cap`: :TruncationFreeSampling: (generation half) — the max_completion_length a batch must use so

**`rlv2/judge/`**

- `judge/judge.py` (782 lines) ONE reasoned score per answer, from one local judge.
  - `def load_rubric`
  - `def rubric_sha`: REQ-J9: the rubric is part of the frozen eval. Its hash goes in the receipt so a
  - `def instrument`: Everything needed to reproduce a score. Print this beside any number it produced.
  - `def build_quality_prompt`: CALL 1. The gold is DELIBERATELY ABSENT -- do not add it back.
  - `def build_score_prompt`: THE call. One number, placed against the rubric's scale, with the gold in hand.
  - `def build_verdict_prompt`: CALL 2. Correctness only -- no quality numbers, so the two axes stay independent.
  - `class LatencyBudget`: Adaptive per-call timeout from an EMA of observed latency (owner design, 2026-08-30).
  - `def clamp_to_band`: Keep the judge's score on the correct side of the 0.70 line its verdict implies.
  - `def combine`: Band from the verdict, position within the band from quality.
  - `def parse`: Merge the two responses. FAILS CLOSED to not_attempted / 0.0 / ok=False.
  - `def should_escalate`: True when the reads disagree about which SIDE of the floor an answer is on.
  - `def parse_score`: One number in, verdict DERIVED from it. FAILS CLOSED to score 0.0 / ok=False.
  - `def never_states_an_answer`: True when the response is a plan or preamble that commits to no claim.
  - `def echoes_question`: True when the answer opens with the question's own first `n` words (tasks.md #51).
  - `def judge`: One graded answer, ONE call. Never raises.
  - `def is_correct`: The binary the CPT/SFT gate reads. `not_attempted` is NOT correct -- and it is also
  - 2 private helpers
- `judge/local_llm.py` (73 lines) ollama-backed drop-in for `nvidia_llm.complete`, same signature.
  - `class QuotaExhausted`: Signature parity with nvidia_llm so callers' except-clauses keep working.
  - `def complete`: Return the assistant message content, or raise after RETRIES attempts.
  - `def stats`
- `judge/nvidia_llm.py` (187 lines) shared inference client for NVIDIA's hosted endpoint.
  - `def client`
  - `class QuotaExhausted`: Raised once the breaker opens. A caller that catches this must STOP, not retry.
  - `def complete`: One chat completion. Returns text, or None after MAX_RETRIES (never a partial substitute).
  - `def stats`
  - 4 private helpers
- `judge/reward.py` (1648 lines) Stage 6: Reward function — regime-branched, returns float in [0, 1].
  - `def set_judge_lane`: Select which judge (model + rubric) the training-reward path uses. Loud on purpose.
  - `def verify_verdict`: Programmatic correctness verdict for verifiable domains. True/False/None.
  - `def compose_reward`: verdict True -> floor + weight*quality; False -> 0.0; None -> quality.
  - `def abstention_verdict`: False when the gate is on and an answerable item is abstained on; else None.
  - `def nli_entailment`: Return probability that premise entails hypothesis (0–1).
  - `def nli_entailment_batch`: Batch version — more efficient for semantic entropy computation.
  - `def reward`: Compute reward for a model answer.
  - `def reward_judge`: F1 + WordNet synset/hypernym coverage, scaled by brevity. Fast, no network calls.
  - `def llm_generate_batch`: Free-form generation, batched, on either backend, with THINKING OFF.
  - `def unload_ollama_model`: Unload one Ollama model without stopping the shared server.
  - `def unload_all_ollama_models`: Unload every resident Ollama model before policy training claims the GPU.
  - `def judge_vram_handoff`: Yield exclusive GPU time for the Ollama judge by offloading ALL models to CPU.
  - `def llm_judge_train`: Score one prompt group with the judge and fail on any malformed result.
  - `def judge_pairs`: Order-aligned scores. Never fabricates. Recursive split breaks the temp=0 glitch;
  - `def llm_judge_flat`
  - `def judge_concurrency`: How many judge calls may be in flight. Depends on WHICH judge is live.
  - `def llm_judge_flat_many`: Score multiple INDEPENDENT pair-chunks with up to `max_workers` Ollama
  - `def llm_judge_batch_train`: Score all prompt groups using flat pair format. Raises on count mismatch.
  - `def llm_judge_eval`: Score all eval classes in one structured call. Returns {cls: score}, {} on failure.
  - 39 private helpers
- `judge/verifiers.py` (522 lines) 
  - `def verify_math`: Programmatic correctness check for math answers.
  - `def score_premises`: Fraction of keyed triplets the answer asserts, in the keyed order.
  - `def verify_entailment`: Does `answer` entail the gold's claims? None when no claim is extractable.
  - `def verify_code`: Run the candidate against the record's asserts. True / False / None.
  - `def code_simplicity`: 1.0 when the candidate is no bigger than the reference, decaying after.
  - `def score_code`: Correctness gate, then simplicity as a tie-breaker. None = not keyable.
  - `def math_conciseness`: 1.0 when the answer is no longer than the reference solution, decaying after.
  - `def score_math`: Correctness gate, then conciseness as a tie-breaker. None = not keyable.
  - `def score_rlvr`: The reward: 0.0-1.0 from a verifiable check, or None to use the judge.
  - `def verify_short_answer`: Short-answer exact match for STEM. True / False / None.
  - `def rlvr_score`: Tiered verifiable reward. Returns (verdict, method).
  - 9 private helpers

**`rlv2/data/`**

- `data/chunking.py` (292 lines) build the CPT corpus: skill notes -> canonical atoms -> packed 512-tok records.
  - `def find_overlap`: Longest suffix of s1 that matches a prefix of s2.
  - `def de_overlap_strings`: Remove overlapping suffixes from consecutive strings. Concatenation reconstructs the source
  - `def self_test`
  - `def mad`
  - `def q`
  - `def sections`: [paragraphs] per section; zero-paragraph sections DROPPED (bare headers carry no content and
  - `def atoms_for`: Canonical units that TILE the source: merge sections below m paragraphs, split above m+2d with
  - `def pack`: Greedy accumulation to `budget` tokens, window stepping by `stride` units.
  - `def main`
- `data/cpt_mix.py` (396 lines) assemble the CPT corpus for the R3 experiment.
  - `def token_len`
  - `def split_to_cap`: Split an over-cap SKILL on paragraph boundaries until every piece fits.
  - `def boxcox_length_band`: (lo, hi) token band from median +- sigmas*MAD*1.4826 in BOX-COX space.
  - `def load`
  - `def render_ntp`
  - `def entigraph_for_skills`: Passages relating two of OUR skills, by metadata or by mention.
  - `def main`
- `data/entigraph.py` (275 lines) EntiGraph over the skills corpus, where the entities are already named.
  - `def load_skills`
  - `def sample_relations`: Within-topic pairs, then triples. Cross-topic relations are mostly vacuous filler.
  - `def main`
  - 3 private helpers
- `data/index_chroma.py` (91 lines) Stage 4: Build a ChromaDB vector index over the labeled corpus.
  - `def get_collection`
  - `def build_index`
  - `def query_similar`: Return up to k metadata dicts ranked by cosine similarity.
  - 1 private helpers
- `data/label_regimes.py` (113 lines) Stage 3: Regime Labeling
  - `def label_regime`: Assign regime to a single record.
  - `def label_file`
  - 2 private helpers
- `data/pools.py` (1878 lines) §6.11 R3-only three-pool generation over the 3 trees-ensembles skills.
  - `def category_pdf`: Box-Cox(lam) of CATEGORY_AVG_WEIGHTS, renormalised. THE allocation basis.
  - `def category_quota`: How many items of each category an n-item pool should hold. Largest remainder.
  - `def resolved_gen_model`: The model that actually generated an item.
  - `def num_predict_for`: Output-token budget for a batch of `batch_n` items.
  - `def gate_reason`: Content-rejection reason for one candidate, or None to accept.
  - `def to_ascii_math`: Rewrite LaTeX / unicode math as ASCII pseudo-code.
  - `class RemoteMiss`: The endpoint returned NOTHING -- distinct from 'the model wrote something unusable'.
  - `def batch_schema`: Strict schema for one batched ask (REQ-G3/G4/G5).
  - `def dedupe_by_fact`: Drop ITEMS that reuse a fact_index. Never drop the batch (REQ-G5).
  - `def embed`
  - `def load_chunks`
  - `def allocate`: Proportional to chunk content, summing EXACTLY to `total`.
  - `def allocate_by_skill`: Split `total` EQUALLY across skills, then across each skill's chunks.
  - `def plan_tasks`: Interleaved A/B work list: [(chunk, pool), ...].
  - `def style_exemplars`: Few-shot exemplars drawn from the real VibeThinker R3 eval, shaped to R3's profile.
  - `def style_exemplars_by_category`: Few-shot exemplars MATCHED to the category being asked for.
  - `def parse_batch`: Pull a LIST of {question, answer} out of model text, tolerantly.
  - `def strip_reasoning`: Remove <think>...</think> traces before any JSON extraction.
  - `def parse_json`
  - `def central_chunks_per_skill`: Pick the most representative chunk for each skill.
  - `def main`
  - 7 private helpers
- `data/pull_datasets.py` (410 lines) 
  - `def sample_from_shards`: Fetch the minimum number of Parquet shards needed to collect n rows.
  - `def normalize_row`: Map a raw row to {question, context, answer}. Return None to skip.
  - `def sample_oasst1`: oasst1 is a message tree; join prompter→assistant pairs before sampling.
  - `def calculate_allocations`: Allocate target_total records across datasets using a two-parameter
  - `def main`
- `data/rl_mix.py` (137 lines) assemble the MGPO training mix: staged RL corpus + data-science skills + abstention.
  - `def main`
  - 1 private helpers
- `data/skills_sft.py` (350 lines) synthetic SFT QA from the stackskills corpus. Stage 1 of the full regimen.
  - `def select`: Stratified by topic, proportional. Keeps every topic represented rather than dropping some.
  - `def split`: Split by SKILL, never by QA pair -- see the module docstring on leakage.
  - `def split_preserving`: Re-derive TRAIN/EVAL from RAW against a PINNED held-out skill list.
  - `def regen_truncated`: Rebuild QA for skills whose source exceeded the OLD generation limit, IN PLACE.
  - `def main`
  - 4 private helpers
- `data/synthetic_qa.py` (260 lines) Stage 5: Synthetic QA generation with regime re-labeling.
  - `def generate_for_seed`: Generate PAIRS_PER_SEED × len(SAMPLER_CONFIGS) synthetic records for one seed.
  - `def main`
  - 6 private helpers

**`rlv2/cpt/`**

- `cpt/train.py` (867 lines) Continued Pretraining with gradual annealing and Box-Cox stratified corpus sampling.
  - `def load_text_corpus`: Load a text file as a single string for chunking.
  - `def load_jsonl_texts`: Load texts from a JSONL file (one JSON object per line with a text field).
  - `def chunk_text`: Tokenize text and split into non-overlapping chunks of chunk_size tokens.
  - `def pack_atoms`: Pack whole records into sequences of at most `budget` tokens, never splitting one.
  - `def derive_token_budget`: Tokens per batch that the cross-entropy logits can afford, from MEASURED VRAM.
  - `def derive_batch_size`: Largest per-device batch whose LOGITS still fit in measured free VRAM.
  - `def build_corpus`: Build the CPT corpus: per-source chunks for Box-Cox allocation.
  - `def allocate_chunks`: Box-Cox stratified allocation across sources.
  - `def sample_epoch`: Sample one epoch's training data with annealing between general and domain.
  - `def main`

**`rlv2/sft/`**

- `sft/loop.py` (25 lines) one SFT epoch over pre-packed bins, on the shared training loop.
  - `def train_bins`: Train one epoch: one optimizer step per bin, schedule spanning all bins.
- `sft/manifold_sampler.py` (375 lines) Copula manifold sampler with delta-ratio correction and token-budget bin packing.
  - `class ManifoldSampler`: Copula manifold sampler with cumulative delta-ratio correction.
  - `def stratified_eval_split`: Split eval set into n stratified groups, each with ≥1 record per populated cell.
- `sft/train.py` (877 lines) SFT on a Box-Cox class-balanced VibeThinker subset. The stage RL needs in front of it.
  - `def build_subset`: Two-axis MMR cross-hatch draw over answer_type x dataset grid.
  - `def fmt`: Prompt exactly as the eval builds it, so SFT teaches the format it is scored on.
  - `def main`
  - 4 private helpers

**`rlv2/rl/`**

- `rl/differentiability_ledger.py` (237 lines) Persist what the replay buffer learns per record, keyed by phase state.
  - `def phase_state`: sha1 of the weights' identity -- content, not path (REQ-L2).
  - `def key`
  - `def load`: Rows for exactly this key, or {} -- never another context's (REQ-L2).
  - `def save`: ONE write, at the end (REQ-L5). Merges into the same key (REQ-L4).
  - `def as_history`: Ledger rows -> the `history` shape grpo_one_file.visit_stats consumes.
  - `def as_last_visit`: Ledger rows -> {question: rewards from the MOST RECENT visit only}.
  - `def prior_history`: THE ONE WAY A TRAINER GETS ITS PRIOR. Call this; do not hand-roll load+as_history.
  - 2 private helpers
- `rl/dpo_pass.py` (139 lines) IPO on judge preference pairs from the harvest sidecars.
  - `def load_jsonl`
  - `def build_pairs`: chosen >= floor > rejected, per question. Chosen mirrors harvest_sft's
  - `def main`
- `rl/entity_validity.py` (192 lines) un-gameable, deterministic "context-entity-precision" correctness signal.
  - `def key_entities`: Tiered key-entities of a text: proper-noun phrases, numbers, and salient content tokens.
  - `def entity_recall`
  - `def entity_precision`: Of the answer's ASSERTED entities (proper nouns + numbers), the fraction grounded in gold+context.
  - `def succinctness`: 1.0 while the answer is no longer than the needed length (max(gold_len, floor)); decays for padding.
  - `def entity_validity`: Drop-in un-gameable correctness in [0,1]. Math delegates to the programmatic verifier.
  - `def classify_answer`: Trichotomy over one answer: correct / neighbor-conflated / hallucinated (Gekhman arXiv:2405.05904).
  - `def trichotomy`: Bucket counts + rates over a batch. `siblings` defaults to the golds themselves (leave-one-out),
  - 4 private helpers
- `rl/exploit_finder.py` (86 lines) find REAL judge exploits: completions the judge scored HIGH but that are
  - `def correctness`: Un-gameable: does the answer actually convey the gold fact? For Math, programmatic verify;
  - `def main`
  - 2 private helpers
- `rl/grpo.py` (2586 lines) VibeThinker's Signal phase (arXiv:2511.06221), two-armed so the mechanism is falsifiable.
  - `class TrainingHealthMonitor`: Enforce the observable-MGPO runtime contract.
  - `def assert_partition_disjoint`: Reject train overlap with every eval and rubric partition before model loading.
  - `def make_reward`: Return the sole MGPO reward path: isolated LLM-judge scores, lane-shaped.
  - `def bootstrap_cohort`: Hierarchical cross-hatch sampling with Box-Cox at each level.
  - `def continuation_probabilities`: Return explicit round-two draw probabilities for the matched control or MGPO arm.
  - `def select_continuation`: Draw round-two prompts with replacement from the fixed bootstrap cohort.
  - `def per_continuation`: PER-style round-2 curriculum: soft priorities + fresh injection (E8).
  - `def coverage_first_cohort`: Operator's curriculum rule (2026-09-07): VISIT COUNT DOMINATES.
  - `def dyna_ipo_step`: Task #44 — Dyna-MGPO's offline update, run at each round boundary.
  - `def probe_disagreement_cohort`: Two-model epistemic cohort selection + visit-count bonus (E1+E2).
  - `def run_arm`
  - `def main`
  - 8 private helpers
- `rl/harvest_sft.py` (193 lines) harvest a checkpoint's SAMPLED successes into an SFT dataset.
  - `def load_jsonl`
  - `def qtext`
  - `def assert_no_eval_leakage`: Tripwire (fails closed): harvesting an eval question would fine-tune on the
  - `def select_best`: Best CORRECT completion for one question: highest judge score wins; ties go
  - `def to_sft_row`: Deposit-schema row (matches data/rl/sft_r3_deposit_half.jsonl, the format
  - `def main`
  - 2 private helpers
- `rl/instrument.py` (160 lines) The frozen instrument, verified at startup and stamped into every receipt.
  - `def load`
  - `def observed`: What the instrument ACTUALLY is right now, read from the live system.
  - `def diff`: Every way the live instrument departs from the pinned one. Empty = clean.
  - `def verify`: Raise unless the live instrument matches the pinned one. Call before scoring.
  - `def stamp`: The block every receipt carries, so a number can never be orphaned from the
  - 2 private helpers
- `rl/prompt_world_model.py` (336 lines) ensemble world model of the prompt->judge environment.
  - `def build_labels`: Harvest (prompt, w, l) rows from the visit ledger and the labeling sweep.
  - `def train_ensemble`: K heads, each on a bootstrap resample (the Dyna-Q EnsembleWorldModel).
  - `def load_ensemble`
  - `def predict`: (p_mean, p_var) per row. p_var is MODEL DISAGREEMENT — part 6: 'best read as
  - `def validate`: The two gates that must PASS before any selection use (part 6: model loss is
  - `def build_qa_labels`: Join each harvest's .gen.jsonl (answer texts) with its .scored.jsonl
  - `def train_qa_ensemble`: Same bootstrap-resampled heads as train_ensemble; target is the judge's
  - `def qa_auc`: Held-out discrimination: does the verifier rank judge-correct answers
  - `def qa_rerank`: Offline pick: for each frozen-eval question, the verifier chooses among the
  - `def main`
  - 4 private helpers
- `rl/red_queen_validity.py` (126 lines) UN-GAMEABLE validity metric for the Red Queen loop.
  - `def measure_validity`: Greedy completions on the holdout, scored programmatically (no judge).
  - 1 private helpers

**`rlv2/eval/`**

- `eval/cpt_fewshot_lift.py` (137 lines) fast isolated CPT knowledge-injection lift, few-shot closed-book (NO re-CPT, NO SFT).
  - `def smoke`: 5-question base+few-shot format check: does the 0.8B answer (not ramble/abstain)?
  - `def main`
  - `def test2`: Rung 2 — + domain-QA SFT (activate latent CPT knowledge). Reuses existing SFT adapters (on disk).
  - 3 private helpers
- `eval/half_epoch_eval.py` (562 lines) F1 on the frozen eval at every half-epoch, in EVERY phase.
  - `def load_eval`
  - `def strip_reasoning`: Remove <think>...</think> traces before the judge sees the answer.
  - `def chat_wrap`: Build the prompt EXACTLY as training wrote it. Think is OFF everywhere.
  - `def banned_ids`: Token ids no phase may ever generate: reasoning blocks and tool calls.
  - `def generate`: Completions from the LIVE model.
  - `def evaluate`: Generate, score, print. Returns the summary, or None if anything went wrong.
  - `class HalfEpochEval`: Fires at every half-epoch boundary, plus once at the end.
  - `def load_policy`: Load a full-param checkpoint or a base+adapter pair.
  - `def main`
  - 2 private helpers
- `eval/policy_eval.py` (208 lines) Shared multi-sample generation and scoring.
  - `def fmt_prompt`
  - `def generate_greedy_batch`: Greedy generation over many prompts, batched. Greedy is deterministic, so batching is
  - `def load_adapter`: Load base model + adapter. Returns (model, tokenizer). Caller must del + empty_cache when done.
  - `def multi_sample_score`: For each record generate one completion per sampler config, then score all with judge.
  - 3 private helpers
- `eval/rl_calibration_eval.py` (733 lines) score RL on the reward it was DESIGNED against: the R1-R4 crosshatch.
  - `def looks_abstain_broad`
  - `def abstain_messages`: Few-shot prompt that PERMITS declining. The default harness forbids it.
  - `def derive_regime`: Regime from (context present, gold present). `rl_eval` carries neither a `regime` field nor a
  - `def verify_scorer`: Assert `reward.reward` responds in BOTH directions before any arm is trusted.
  - `def evaluate`: Evaluate one aligned generation per row and persist complete diagnostics.
  - `def load_model`
  - `def main`: Run one scorer invocation and persist/print its complete diagnostic chain.
  - 3 private helpers
- `eval/score_arms.py` (599 lines) score EVERY arm in ONE invocation, so the instrument cannot drift.
  - `def generate_all`: {arm: [k lists of Q answers]}. One model resident at a time (tech.md: never co-reside).
  - `def score_question`: ONE call ranking every arm's answer to ONE question.
  - `def auc`: P(random correct answer scores above a random wrong one). The MONOTONE separation measure.
  - `def kl_divergence`: KL(correct || incorrect) over binned scores. Laplace-smoothed so an empty bin stays finite.
  - `def bootstrap_ci`: Percentile CI on a proportion. A point estimate at n=16 carries no claim on its own.
  - `def main`
  - 2 private helpers
- `eval/score_r3_categories.py` (174 lines) the R3 objective readout, per CATEGORY, at k samples.
  - `def is_correct`: Did this sample answer the question? The VERDICT decides when there is one.
  - `def load`
  - `def summarise`: records: one row per (item, sample) with keys item_id, category, reward.
  - `def render`: The §6.10 grid, on the axis that carries information.
  - `def main`

**`rlv2/hpo/`**

- `hpo/rl.py` (221 lines) Three-trial LoRA MGPO learning-rate search on the matched two-round schema.
  - `def objective`: Train one 2×9 regime-balanced LoRA MGPO trial and return canonical mini-eval macroF1.
  - `def main`
  - 1 private helpers
- `hpo/sft.py` (110 lines) Optuna TPE sweep over full-param SFT to find the config that learns abstention.
  - `def objective`
  - `def main`

<!-- AST-OUTLINE:END -->

---

## Status (2026-10-02): the RL lane is closed

**Question.** Can a 0.8B model (Qwen3.5-0.8B) hold new domain knowledge (decision trees and ensembles; 748 questions split into 12 rubric examples, a 16-question eval, and 240 each for CPT, SFT and RL) in its weights, and can reinforcement learning put it there?

**Answer so far.** SFT puts it there. RL, as built and tested here, did not carry any gain to unseen questions.

| Test | Result |
|---|---|
| Full-parameter SFT | base 3 → 7 of 16 at epoch 2; the only phase that raised the eval |
| Continued pre-training on the corpus | no lift |
| LoRA adapter deposits, then RL | lost 0.06 to 0.27 on every route; the full-parameter deposit survived |
| Second RL pass on an improved model | flat or lower in six lineages |
| KL penalty (beta) | the leash: beta 0 or 0.001 blew up at step 19; 0.3 held 32 to 60 steps, then ran away |
| Replay of judged answers; world model of the judge | both lost (replay: KL abort at step 24; world model scored the question, not the answer) |
| Jev (small yes/no judge) as the reward | exploitable until guarded (stub, numeric and echo guards); 8% of golds rejected by Jev itself |
| Transfer to unseen questions (final runs, echo guard, exposure-counted visit cap) | eval 32 → 20 of 128, held-out 79 → 73 of 320: no transfer |

The best RL run raised solved training records from near 0 to 57 of 240 and never the held-out score. Three readings of the null are carried forward: mis-specified but not applied, scale (240 records and a 16-question eval), or the wrong tool (RL sharpens, SFT injects). The scale reading was never tested.

**Where to read more.**

- `dispositions.md`: the dated log of every test, with receipts and verdicts (supported, falsified, not tested).
- `SETTLED.md`: decisions not to be re-litigated.
- The lab-notebook deck (12 entries: hypothesis, settings, method, result, verdict) is kept locally and is not in this repository.
- `playbook.md`: the task ledger, including the last fix (visit cap counts gradient exposures, not judged draws).

**What is not in this repository.** The `docs/` folder (papers, notes, figures, decks, and the single-file trainer `docs/grpo_one_file.py`) is left out for now because it holds third-party copyrighted material; links to `docs/...` below point to files that are not here, and tests that load the single-file trainer will not run. Model weights and optimizer states (deleted 2026-09-29 by decision; configs, logs, metrics, judged answers and scoring receipts were kept), the Jev judge model, and the generated training pools under `data/` that are untracked. The base model is the operator's own `skills_model` checkpoint.

The design below is the original theory and pipeline. Where it differs from what was run, `dispositions.md` and `SETTLED.md` govern.

---

## Source hierarchy

### RL theory authority — concept

| Section | Governs |
|---------|---------|
| MDP Formulation Checklist | State / action / reward / transition design |
| Bandits as the Simplest Decision Problem | Why training is bandit-like despite being a formal MDP |
| On-Policy vs Off-Policy in One Line | GRPO on-policy constraint |
| Monte Carlo vs TD | Group rollouts and advantage computation |
| Empirical Return vs Exact Value | Anchor baseline and group scoring rationale |
| Bias-Variance Tradeoff | Why G>1 and why 3-sampler anchor scoring matters |
| Policy Gradients / Actor-Critic (Addendum) | Why we use direct policy optimization, not value-based RL |
| Policy Parametrization / Softmax (Addendum) | The policy IS the LM's softmax distribution |
| Approximate Methods (Addendum) | Why the network approximates the policy, not a Q-table |
| Model-Free vs Model-Based | Why Bellman DP is not applicable |
| Boltzmann / Softmax Exploration | Temperature during generation |
| Exploration vs Exploitation | Why not a bandit domain curriculum |
| Reward Shaping and PBRS | KL penalty and regime-aware reward design |
| Inverse RL / Reward Modeling (Addendum) | LLM-as-judge as learned reward approximator |
| Q-Learning Update Rule | Explains why a stronger judge is required (quality ceiling principle) |

### Algorithm authority — VibeThinker-3B paper

Use Sections 2.2.1 and 2.2.2 as the only direct authority for the RL algorithm:

| Section | Governs |
|---------|---------|
| 2.2.1 Algorithm Backbone | MGPO, grouped rollouts, prompt weighting, on-policy constraint |
| 2.2.2 Multi-domain Reasoning RL | Reward construction, prompt filtering, context strategy, training order |

Supporting context only (do not import into core formulation):

| Section | Use only to |
|---------|-------------|
| 2.1 Supervised Fine-tuning | Justify strong SFT initialization and long-context strategy |
| 3.1 Evaluation Setup | Separate training-time RL from evaluation-time repeated sampling |

Exclude: 2.3 Offline Self-Distillation, 2.4 Instruct RL, 3.2 Evaluation Results.

### Excluded from formulation

Do not cite these as authority for this project:

- Bellman backups, value iteration, policy iteration (tabular) — no state-value table exists
- Q-learning, SARSA, DQN — wrong algorithm family
- Replay buffer, target networks — DQN constructs, not GRPO
- Frame stacking, CNNs — different domain

---

## MDP formulation

*Concepts: What problem am I solving?, MDP Formulation Checklist*

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \rho_0)$$

| Object | Definition |
|--------|-----------|
| ρ₀ | Distribution over (question, context, regime, gold) from `data/raw/all.jsonl` |
| S | prompt + token prefix: `s_t = (x, y_{<t})` |
| A | Qwen3.5-0.8B token vocabulary |
| P | Deterministic: `s_{t+1} = (x, y_{≤t})` — appending one token |
| R | Sparse terminal: `r_T` = composite reward (routed by domain: verify-gate + judge for Math/Code, judge for soft domains); `r_t = 0` for `t < T` |
| Horizon | Episodic; terminates at EOS or `max_completion_length=256` |

*Concepts: Deterministic Transition Simplification* — token appending is deterministic, so the transition matrix collapses to one successor per `(s, a)` pair.

*Concepts: Terminal States and Absorbing-State Convention* — EOS is the absorbing terminal state. All reward is assigned at completion; no reward accrues after termination.

*Concepts: MDP Formulation Checklist* The regime-conditional reward (R1–R4) is a direct consequence of this: collapsing all four regimes into one reward definition would produce a bad formulation.

---

## How the policy is derived

*Concepts: Policy Gradients / Actor-Critic (Addendum), Policy Parametrization / Softmax (Addendum), Approximate Methods (Addendum)*

This is the central design question. The policy is **not** derived from a value function. It is directly optimized by gradient ascent on judge-scored completions.

### What the policy is

The policy π_θ is the language model itself — a parametrized softmax distribution over the next token at every decoding step:

$$\pi_\theta(y_t \mid x, y_{<t}) = \text{softmax}(f_\theta(x, y_{<t}))$$

*Concepts: Policy Parametrization / Softmax* The network f_θ is not approximating a Q-value table. It IS the policy. Every forward pass produces a probability distribution over the vocabulary — that is the policy's action selection at step t.

*Concepts: Approximate Methods* In tabular RL, you would need one table entry per (state, action) pair. Here, state = prompt + all generated tokens so far, and action = one token from a vocabulary of ~150K. The state space is astronomical; no table could enumerate it. The network generalizes across states by learning representations of text.

### The GRPO optimization loop

Starting from the SFT checkpoint (format-primed Qwen3.5-0.8B + QLoRA adapters):

```
for each training step:
    1. sample a batch of (question, context, regime, gold) prompts
    2. generate G=4 completions from current policy π_θ for each prompt
    3. judge scores all completions: r_1 ... r_G  (one structured JSON call, all groups)
    4. compute group-relative advantage: A_i = (r_i − mean(r)) / std(r)
    5. policy gradient: θ ← θ + α ∇_θ E[A_i × log π_θ(y_i | x)]  − β KL(π_θ ‖ π_ref)
```

Step 5 is the key: the gradient **increases log-probability of tokens in high-advantage completions** and **decreases log-probability of tokens in low-advantage completions**. After enough steps, the policy distribution has been sculpted — the model is more likely to generate sequences that resemble high-scoring completions and less likely to generate sequences resembling low-scoring ones.

This is direct policy optimization. The policy is what changes. There is no value table being built and then queried for a greedy action.

### Why this is different from Q-learning

*Concepts: Q-Learning Update Rule*

In Q-learning: learn Q(s,a), then extract policy as π(s) = argmax_a Q(s,a). The policy is derived from the value function.

In GRPO: the policy IS the model. No Q(s,a) is learned. The reward signal goes directly into the policy gradient — no intermediate value table. The advantage A_i plays a similar role to the TD error in Q-learning (it tells the update how much better or worse this completion was than the group mean), but it is applied directly to the policy parameters via log-prob, not to a value estimate.

---

## Why the judge must be stronger than the policy

*Concepts: Q-Learning Update Rule, Inverse RL / Reward Modeling (Addendum)*

In Q-learning, the quality of the learned policy is bounded by the accuracy of the Q-function. If Q(s,a) is biased or has low signal, the policy converges toward those biased estimates. *Concepts: behavior may explore; learning target is greedy* — if the greedy target is wrong, the policy degrades toward it.

The same principle applies here: **the policy is bounded by the ceiling of the judge's discrimination**. If the judge cannot reliably distinguish correct from incorrect answers, GRPO has no coherent signal to optimize toward.

### The quality ceiling argument

| Judge type | Effect |
|-----------|--------|
| Policy self-judges its own outputs | Quality ceiling = policy's current knowledge. Reward hacking likely: model learns to produce outputs that fool its own evaluation, not actually better answers. No upward pressure. |
| Judge is the same capability as policy | Slightly better signal but same ceiling. Cross-validation without transfer. |
| Judge is stronger than policy | Quality ceiling exceeds policy's current knowledge. Judge correctly identifies correct reasoning the policy arrived at by chance vs. by understanding. Creates genuine upward pressure — the policy can improve toward a standard it has not yet reached. |

This project uses `qwen3.5-reasoning-oc:4b` (2B parameters, reasoning model) to judge `Qwen3.5-0.8B` (0.8B parameters, non-reasoning). The judge's knowledge ceiling exceeds the policy's. This is the mechanism by which learning can exceed the policy's initial knowledge state.

*Concepts: Inverse RL / Reward Modeling* The judge is the reward function approximator R(x, y_{1:T}, y*). No symbolic rule can reliably evaluate free-text answers across math, code, STEM, reading comprehension, and agentic tasks simultaneously — so a stronger LM generalizes what "correct" means across all question types. This is the same reason RLHF uses human preferences rather than model self-scoring: the reward signal encodes a quality standard that exceeds the model being trained.

### Two function approximators

This codebase requires two function approximators, not one:

1. **Policy approximator** — Qwen3.5-0.8B + QLoRA. Approximates π_θ(y_t | x, y_{<t}): given state, what token next?
2. **Reward approximator** — qwen3.5-reasoning-oc:4b (judge). Approximates R(x, y_{1:T}, y*): given the completed trajectory and gold answer, how correct was this?

In a game environment, reward is computed deterministically by the game engine. Here, "correctness" is not deterministic from a simple rule — the judge plays the role the game engine plays in Atari.

---

## Why training behaves like a contextual bandit

*Concepts: Bandits as the Simplest Decision Problem*

Concepts defines a bandit as having no state transition model, no long trajectory, no delayed credit assignment, and only action choice under reward uncertainty.

GRPO is formally an episodic MDP, but it is bandit-like in practice because:

- Reward is entirely terminal — no per-token signal exists
- The policy learns which *sequence* was better, not which individual token was decisive
- Credit is assigned uniformly to all tokens in the trajectory based on the terminal judge score

The policy gradient does assign credit to individual tokens via log-probability, but the information content of the reward is sequence-level. Training signal is sparse and episodic, exactly as in a contextual bandit.

---

## Group rollouts and advantage

*Concepts: Monte Carlo vs TD, Empirical Return vs Exact Value, Bias-Variance Tradeoff*

For each prompt x, GRPO samples G=4 completions and scores all of them:

```
G_i  = r_i                          (terminal judge score; episodic, no discounting)
A_i  = (G_i - mean(G)) / std(G)     (group-relative advantage)
```

This is Monte Carlo policy evaluation within the group. *Concepts: n/a* The group baseline subtraction reduces variance. *Concepts: Bias-Variance Tradeoff*

Groups where all G=4 completions score identically collapse to zero advantage — no gradient. This is why `policy_std > 0` must hold across the training pool, and why the anchor baseline must be grounded in group scoring rather than point estimates.

---

## On-policy constraint and KL penalty

*Concepts: On-Policy vs Off-Policy in One Line, Reward Shaping and PBRS*

GRPO trains on rollouts from the policy being updated. *Concepts: on-policy = learn from what you actually did*

The KL penalty β keeps the policy near the SFT reference during updates. Without β, unconstrained gradient steps push the policy off-distribution, creating mismatch between rollout collection and the update target. *Concepts: Reward Shaping and PBRS* The KL term is PBRS-style regularization: grounded in the reference distribution, not an arbitrary bonus.

β=0.174 (from HPO) is the selected balance between learning speed and stability.

*VibeThinker paper (2.2.1):* "we adopt the stabilization strategy and perform all RL stages in an on-policy manner."

---

## Exploration: temperature

*Concepts: Boltzmann / Softmax Exploration, Exploration vs Exploitation*

Completion generation uses temperature sampling. *Concepts: n/a*

- `temperature → 0`: greedy decode — maximum exploitation
- `temperature → ∞`: uniform sampling — maximum exploration

HPO selected temperature=0.921. *Concepts: n/a*

---

## Reward design

*Concepts: MDP Formulation Checklist, Inverse RL / Reward Modeling (Addendum), Reward Shaping and PBRS*

### Regime-conditional reward (R1–R4)

*Concepts: n/a*

Regime labels define what "correct" means per `(question, context)` pair:

| Regime | Condition | Correct behavior |
|--------|-----------|-----------------|
| R1 | Context present, answer in context | Correctness + grounding |
| R2 | Context present, answer not supported | Abstain: "Not supported by context" |
| R3 | No context, answerable from knowledge | Correctness; abstain is wrong here |
| R4 | No context, genuinely unanswerable | Abstain: "Unanswerable" |

R2 and R4 must not be merged: R2 is retrieval failure; R4 is genuine unanswerability. They require distinct gold strings and distinct reward signals.

### Judge rubric as learned reward

The scoring rubric is calibrated from 50 out-of-sample `(question, gold, generated)` triplets via `generate_rubric.py` using `qwen3.5-reasoning-oc:9b`. It generates completions at 4 sampler temperatures (greedy, conservative, balanced, creative) and scores all pairs with the 4B judge, producing `data/rubric_prompt_review.txt` for human review and `data/rubric_variance.json` for cross-sampler variance analysis. The rubric prompt is reviewed before being wired into `reward.py`.

*Concepts: Inverse RL / Reward Modeling* The rubric is a CONTINUOUS [0.0, 1.0] scoring scale (never collapsing to 0/0.5/1.0 anchors) that serves as the correctness signal for soft domains (ReadingComp, Instruction, Agentic, non-numeric STEM) and the quality scorer among answers the verify-gate has already confirmed correct. For Math/Code, the verify-gate is the correctness arbiter; the judge is not.

### Training reward: batch group call

`llm_judge_batch_train` (`reward.py`) scores all G=4 completions across all prompt groups in one structured JSON Ollama call — one GPU forward pass, one HTTP round-trip. The judge sees all completions simultaneously and can calibrate scores relatively within and across groups. Think blocks are stripped before scoring (`_strip_think`). The Ollama judge runs with `"think": false`.

### Eval reward: per-class structured call

`eval_adapter_by_class` (`train_outer_loop.py`) generates one completion per domain class (6 total), then scores all 6 in one structured JSON call. Output: per-class scores. Macro-F1 = mean of 6 domain scores — equal weight per class regardless of dataset proportions.

### Verifiable Rewards (RLVR)

Terminal reward routing is now **composite and domain-conditional**. The verify-gate (`verifiers.py`) runs first for Math and Code domains; soft domains bypass it entirely.

**Math verification:** Extract the final numeric answer from gold and completion — priority: text after the last `####` marker, then `\boxed{…}`, else the LAST number in the string. Normalize ($ stripped, commas stripped) and compare with relative tolerance 1e-6. Numbers inside `####` / `\boxed` regions are treated as a multi-answer SET; bare fallback takes only the last number so intermediate computation numbers never pollute the comparison.

**Code verification:** Execute the completion against MBPP/HumanEval asserts or code_contests stdin/stdout I/O pairs. Language is auto-detected from the code fence. Python/Node run natively; C/C++/Java/Rust run in per-language Docker sandboxes. Returns True/False/None (None when no tests available).

**Composite formula:** Let `v` = verify-gate result and `q` = judge quality in [0,1]. Then:
- `if v is True` → `r_T = 0.7 + 0.3·q` (correct answer with quality bonus)
- `if v is False` → `r_T = 0.0` (programmatically wrong; judge is skipped)
- `if v is None or domain is soft` → `r_T = q` (no deterministic verdict; judge assigns correctness+quality directly)

The 0.7 floor guarantees any verified-correct answer outranks any incorrect one, while the 0.3·q band preserves gradient among correct answers. Soft domains (ReadingComp, Instruction, Agentic, non-numeric STEM) go directly to `r_T = q` — the judge is the sole correctness signal there.

---

## Anchor policy baseline and group scoring

*Concepts: Empirical Return vs Exact Value, Monte Carlo vs TD, Bias-Variance Tradeoff*

`score_anchor_set.py` runs once before the outer loop. It is Monte Carlo policy evaluation of the SFT prior — establishing the distribution of rewards before GRPO begins.

*Concepts: n/a*

**Why point estimates are insufficient:** Scoring a single completion at a fixed temperature gives one noisy draw from the policy's distribution. The judge at temperature=0 is fully deterministic — the same completion always returns the same score. To get a calibrated estimate of what the policy is *expected* to score on a given record, you need multiple samples.

**The correct approach — group scoring:** Generate G completions per record by sampling from the policy at temperature > 0 (the HPO-selected temperature=0.921), then score all G in one judge call. The judge sees the full spread of what the policy produces for that question and calibrates scores relatively — the same mechanism as GRPO's `llm_judge_batch_train`. This produces:

- `group_mean` — empirical per-record expected reward (the actual object GRPO's advantage formula subtracts)
- `group_std` — within-group spread (signal reliability: high std = noisy reward for this record)

**`policy_std > 0` is a pre-condition for GRPO, not a hyperparameter.** If std = 0 across the training pool, all completions score identically, GRPO's advantage collapses to zero, and no gradient exists. The anchor baseline verifies this pre-condition before compute is spent on training.

Output: `data/anchor_scores.json`.

*VibeThinker paper (2.2.2):* "filter out samples yielding accuracy exactly 0.0 or 1.0 as evaluated by the starting checkpoint." The anchor baseline enables this filter: records with `group_mean = 0.0` or `group_mean = 1.0` have no learning value for GRPO (no positive signal or already saturated) and should be excluded from the training pool.

---

## Outer loop as truncated policy iteration

*Concepts: Truncated Policy Iteration, Greedy Policy Improvement, Convergence, sweep behavior, and stopping*

```
repeat (up to N rounds, N = len(grpo_train) // 50):
    # m=1 GRPO epoch — truncated policy evaluation
    train_batch(records, starting_adapter, output_dir)

    # greedy policy improvement check
    macro_f1 = eval_adapter_by_class(...)
    if macro_f1 > best:
        best = macro_f1
        save checkpoint
        patience_wait = 0
    else:
        patience_wait += 1
        if patience_wait >= patience_limit: stop
```

*Concepts: n/a*

*Concepts: Policy iteration outer loop: stop when policy_stable == true* Here `policy_stable` = consecutive rounds without macro-F1 improvement reaching the patience limit.

**Why one epoch per round?** *Concepts: n/a* With a 20-hour budget and N rounds, one GRPO epoch per round maximizes the number of improvement checks against total compute.

---

## Record sampler (the "replay buffer")

Which of the 240 RL records get drawn each step. Every draw generates 4 FRESH rollouts from the
current policy and judges them; no stored answer is reused. (A sampler that reused stored answers
was built, run once, and abandoned -- see "Replay sampler" below.)

### The formula as it runs (rl_lr3e6, all 46+ steps)

```
s_1..s_n   raw judge scores of record i, oldest first (not banded to 0/0.5/1)
a          = 2/(N+1),  N = expected visits per record = 1600/240 ~ 7  ->  a = 0.25
m_t = a*s_t + (1-a)*m_{t-1}      v_t = a*s_t^2 + (1-a)*v_{t-1}      (both / (1-(1-a)^n))
spread_i   = sqrt(v - m^2)                      EMA standard deviation, max 0.5
rank_i     = spread_i / 0.5                     (code writes / 0.25 with T; identical draws at T/2)
             retired -> min(rank, 0.02);  unvisited -> above every visited record;  ties +<=0.001
P(i)       = exp(rank_i / T) / SUM_j exp(rank_j / T)
T          = sqrt(0.15^2 + 12 * mean_i Var[Beta(1+k_i, 1+n_i-k_i)]) / 2,  k = scores >= 0.7
```

Code: `docs/grpo_one_file.py` -- `ema_stats`, `visit_stats`, `boltzmann_weights`, `evidence_temp`.

### Why each piece, and what it is grounded in

| piece | basis |
|---|---|
| spread as priority | a group whose 4 rollouts score alike gives exactly zero GRPO gradient (measured: loss -0.0, grad norm 0.0) |
| Boltzmann `exp(rank/T)` | concept "Boltzmann / Softmax Exploration", pi ~ exp(Q/tau) with Q = the record's spread |
| EMA, N = visits per record | non-stationary policy; span fixed 2026-09-13 from step count to visits/record |
| unvisited ranks first | operator ladder "Maxent > straddle > .5 > 0 > 1"; 2026-09-19 coverage stuck at 232/240 for 5 steps |
| T from evidence, not a step clock | 2026-09-17: identical evidence gave T 0.89 vs 0.15 depending only on max_steps |
| divide by 0.5 | the max of a standard deviation on [0,1]. The code's 0.25 is the max of a VARIANCE -- a units slip that is exactly absorbed by halving T, so the run is unchanged |
| 0.02 retired floor, retire after 3, factor 1.1, the T shape and 0.15 floor | **design choices with no experiment behind the number** |

Full provenance and the offline recomputes: [dispositions.md -- Record sampler as run](dispositions.md#record-sampler-as-run-in-rl_lr3e6-2026-09-25--the-live-formula-its-constants-and-two-defects).

### Raw scores vs banded variance -- a challenger, not a fix

Banding before measuring spread (REQ-R7) would draw half as often from failing records
(draw mass 0.33 -> 0.16) and skip records that come back uniform (665 of 1115 draws in 21 steps
were zero-gradient). It is a challenger arm for a fresh run, judged on how fast fail falls.
Your notebook's Dirichlet `E[Var(r)]` over the three bands has never been on a live path; it was
replaced 2026-09-19 because its score rose with visit count alone (0.1643 at n=4 vs 0.1782 at n=12).

### The shapes: which run crossed, which did not

*(Figure `docs/rl_shapes_solved_straddle_fail.png`: solved / straddle / fail by step for three runs; not included in this repository.)*

| run | lr | KL | base | solved vs straddle |
|---|---|---|---|---|
| crossing run, `archive/rl_full_resume_broken_0921` | 1e-5 | beta 0.02, k1 estimator, priority-weighted (launch command 2026-09-21); also temperature 0.8 and per-token mean log-prob | `sft_arms/sft_full_epoch2`: SFT full run, lr 5.82e-6, 3 epochs cover-once, taken at epoch 2 (input eval 1/16) | **crossed at step 29, 57 vs 48**; step 31 collapse was a broken resume |
| rl_kl03 | 1e-5 | beta 0.3, active (kl_raw 0.13 -> 0.55, runaway 2.94 at step 33) | `sft_book1e5/sft_full_epoch1`: book SFT lr 1e-5 x 3 epochs, taken at epoch 1 | never crossed, peak 28 vs 77 |
| rl_lr3e6 (current) | 3e-6 | beta 0.3, kl_raw <= 0.13 | `sft_book1e5/sft_full_epoch1` (same) | not crossed, 22 vs 98 at step 47 |

Fail falls at about the same rate in all three (~-22 by step 30). What differs is straddle:
the crossing run converted straddle into solved; the current run holds straddle near 100.
The crossing run differs from rl_kl03 in six things: starting checkpoint, KL (0.02 k1 vs 0.3 k3),
temperature (0.8 vs 0.9), log-prob normalisation (mean vs sum), run length (50 vs 100 rounds, so
sampler EMA span 3 vs 7), and nothing else -- lr 1e-5 is shared. Not one-variable. Details: [dispositions.md](dispositions.md), lessons:
[lessons_learned.md](lessons_learned.md).

### Replay sampler (built, abandoned)

`--replay-k N` filled N of K=4 slots with stored best/worst answers (age <= 5 steps), stored score
reused, per-token ratio `rho_t = exp(log pi_now - log pi_gen)` clipped to [0.8, 1.2]. The summed
sequence ratio clipped away 75-94% of replays; per-token drift was 0.004-0.07 nats/token at lr 3e-6.
Run once (rl_replay) from rl_lr3e6 step 25 and abandoned: a visit became 2 fresh answers, so
"solved" meant 2/2 not 4/4 and the census stopped being comparable. If revisited: fresh start only.
Default `--replay-k 0`, parity-verified byte-identical.

### Checkpoints and stopping

`latest/` and `best/` are written atomically with the optimizer every eval (`_atomic_checkpoint`,
`tests/test_atomic_checkpoint.py`): a reboot mid-save leaves the previous checkpoint whole.
The only stop rule is `--straddle-patience 20`; `--adv-flat` must be 0 -- it stopped rl_lr3e6 at
step 46 on 2026-09-25 against the operator's rule.

## Domain sampling: why not a bandit curriculum

*Concepts: Exploration vs Exploitation, Bandits as the Simplest Decision Problem, Empirical Comparison Metrics*

A bandit curriculum would estimate per-class improvement probability and allocate training samples toward domains with the highest expected gain. This is rejected:

**Too few rounds for reliable estimates.** *Concepts: too few episodes → incomplete learning* With 38 rounds across 6 classes, per-class bandit estimates would be noisier than the signal they represent.

**Macro-F1 already encodes the per-class decomposition.** Macro-F1 penalizes any class that fails to improve. A bandit would add noisy redirection on top of the metric that already captures it.

**Stratified equal-weight sampling is the correct curriculum given the metric.** Equal weight per class at training time matches equal weight per class at evaluation time (macro-F1).

6 domain classes: Math, Code, STEM, Instruction, Agentic, ReadingComp. Each round samples an equal number of records per class regardless of dataset proportions.

---

## Hyperparameter search (HPO)

*Concepts: Practical Parameter Intuition for Tabular TD Control, Bias-Variance Tradeoff, Boltzmann / Softmax Exploration*

Optuna TPE, 5 trials. Stratified train/eval sets by (domain, source). 3 epochs per trial, patience=2 at epoch boundaries.

| HPO param | Concept analog | Role |
|-----------|--------------------|------|
| learning_rate | alpha — step size | How fast policy parameters move per gradient step |
| beta | PBRS constraint weight | How strongly the SFT reference anchors updates |
| temperature | tau in Boltzmann exploration | Exploration-exploitation balance during rollout generation |

*Concepts: alpha controls update size*

HPO uses `reward_judge` (fast heuristic, no network calls). *Concepts: alpha and epsilon define the search mechanics* HPO is tuning optimizer mechanics, not the quality of the reward signal — LLM judge is only used in the full outer loop.

Trial 0 seeded at literature priors: `lr=5e-5, β=0.05, temp=0.8`.
Best params: `lr=1.19e-05, β=0.174, temperature=0.921`.

---

## Model

Qwen3.5-0.8B (the `skills_model` checkpoint), trained full-parameter. QLoRA is optional and was falsified as a way to inject knowledge (see Status). The SFT checkpoint serves as the frozen KL reference, pinned for the whole run including resumes, and never updated during GRPO.

*Concepts: Policy Parametrization / Softmax* The policy is the LM's softmax output over the vocabulary at each decoding step.

### Thinking tokens

Training and generation always run with `enable_thinking=False` via `tokenizer.apply_chat_template`. The model's thinking capability is preserved in weights and available at inference — not activated during GRPO. `_strip_think()` in `reward.py` strips any residual `<think>...</think>` blocks before scoring.

Training data has no think-tagged examples. Allowing thinking during rollouts produces think tokens that consume completion budget and receive no targeted reward signal — adding variance with no bias benefit. *Concepts: Bias-Variance Tradeoff* Suppressing thinking reduces variance in Monte Carlo returns.

---

## What this project does NOT do

- **No Bellman bootstrapping.** Reward is terminal; TD targets would require mid-sequence value estimates that don't exist.
- **No DQN-style replay buffer in the production runs.** On-policy GRPO generates fresh rollouts each step. A variant that reuses stored judged answers with a clipped importance ratio was built and tested (`rl_replay2`) and lost: reused answers drifted off-policy within 5 steps and the KL guard aborted the run. The record sampler below prioritizes records, not stored answers. *(Concepts: "Replay Buffer")*
- **No target network.** The SFT reference plays a stabilizing role via KL penalty but is a frozen policy, not a periodically-synced value approximator. *(Concepts: "Target Networks")*
- **No bandit arm selection over domains.** See Domain Sampling section.
- **No sequential domain curriculum** (Math → Code → STEM). VibeThinker uses this; this project trains on all 6 domains simultaneously, with macro-F1 as the honest aggregate signal.
- **MGPO prompt weighting is partial.** VibeThinker weights prompts near intermediate empirical accuracy p(q) ≈ 0.5. This project filters prompts at accuracy exactly 0.0 or 1.0 and draws the rest by uncertainty (the record sampler). The "maxent" weighting lives in `src/mgpo.py`; the single-file trainer `docs/grpo_one_file.py`, which produced the last runs, does not have it.

---

## Summary

| RL concept | This project |
|-----------|-------------|
| MDP | Episodic autoregressive; state = prompt + prefix, action = token, transition = deterministic append |
| Policy | Qwen3.5-0.8B + QLoRA; direct softmax parametrization — policy IS the network |
| Policy derivation | Direct gradient ascent on judge-scored completions; no Q-table |
| Reward | Sparse terminal; judge (qwen3.5-reasoning-oc:4b) approximates R against regime-specific gold |
| Judge strength | Stronger than policy — creates upward quality pressure beyond policy's current ceiling |
| Algorithm | On-policy policy gradient (GRPO); not value-based, not off-policy |
| Monte Carlo | G=4 group rollouts per prompt; advantage = (r − mean) / std |
| Exploration | Boltzmann temperature=0.921 during generation |
| Outer loop | Truncated policy iteration; patience on macro-F1 |
| KL penalty | PBRS-style regularizer; keeps policy near SFT reference |
| Curriculum | Stratified equal-weight sampling; no bandit arm selection |

---

## Run order — full e2e pipeline

### 1. Data acquisition (one-time, no GPU)

```powershell
python pull_datasets.py      # Download & balance 2000 records across 6 domains
                             # Writes: data/raw/all.jsonl
                             # Skip if data/raw/all.jsonl already exists
```

### 2. Dataset partitioning (one-time, no GPU)

```powershell
python partition_data.py     # Create non-overlapping train/eval/rubric/SFT splits
                             # Writes: data/manifest.json
                             # Skip if data/manifest.json already exists
```

Splits (sequential exclusion — no record appears in more than one):

| Split | N | Seed | Purpose |
|-------|---|------|---------|
| `sft` | 400 | 99 | SFT format-priming only |
| `eval_holdout` | 100 | 7 | Outer-loop macro-F1 eval; never in training |
| `rubric` | 50 | 123 | Rubric calibration; never in training |
| `grpo_train` | ~1387 | — | GRPO outer-loop training pool |

### 3. SFT format-priming (one-time, GPU ~20min)

```powershell
python sft_prime.py          # Fine-tune Qwen3.5-0.8B on 400 format-priming examples
                             # Reads:  manifest["sft"]
                             # Writes: checkpoints/sft_prime/
                             # Skip if checkpoints/sft_prime/adapter_model.safetensors exists
```

The SFT checkpoint becomes the frozen KL reference for all GRPO training. It teaches output format (abstention strings, chat template) — not domain knowledge.

### 4. Rubric calibration (one-time, GPU ~40min) ← HUMAN REVIEW REQUIRED

```powershell
python generate_rubric.py    # Generate 50 × 4-sampler completions, score with 4B judge
                             # Reads:  manifest["rubric"], checkpoints/sft_prime/
                             # Writes: data/rubric_prompt_review.txt  ← review this
                             #         data/rubric_variance.json
```

**Stop here.** Open `data/rubric_prompt_review.txt` and verify the (question, gold, generated) triplets look reasonable. This file becomes the rubric that calibrates the judge reward signal for all subsequent training.

### 5. HPO — hyperparameter search (optional, GPU ~30min)

```powershell
python hpo.py                # Optuna TPE: tune learning_rate, beta, temperature
                             # Writes: data/hpo_results.json
                             # Skip to use literature priors (lr=5e-5, beta=0.05, temp=0.8)
```

### 6. Anchor baseline (optional, GPU ~15min)

```powershell
python score_anchor_set.py   # G=4 completions on 50 training records, score with judge
                             # Reads:  manifest["grpo_train"], checkpoints/sft_prime/
                             # Writes: data/anchor_scores.json
```

Verifies `policy_std > 0` — the pre-condition for GRPO to produce non-zero gradients. If >50% of groups show zero variance, investigate before committing to full training.

### 7. GRPO outer loop (GPU ~20hrs)

```powershell
python train_outer_loop.py   # Truncated policy iteration over up to N rounds
                             # Reads:  manifest["grpo_train"] (~1387 records)
                             #         manifest["eval_holdout"] (~108 records)
                             #         data/hpo_results.json (if present)
                             # Writes: checkpoints/grpo/best/        (best adapter, overwritten on improvement)
                             #         data/outer_loop_state.json    (resumable — restart safely)
                             #         logs/outer_loop_*.log
                             #         logs/round_history.jsonl
```

N = `len(grpo_train) // 50` (currently 27). Stops early if `outer_patience = 4` consecutive non-improving rounds.

Resumable: if interrupted, re-running `python train_outer_loop.py` picks up from `data/outer_loop_state.json`.

---

### Re-run options

```powershell
# Resume interrupted training (most common)
python train_outer_loop.py

# Restart GRPO only (keep SFT checkpoint — same seed=99 selection as manifest)
Remove-Item checkpoints\grpo -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item data\outer_loop_state.json -ErrorAction SilentlyContinue
python train_outer_loop.py

# Full clean slate
Remove-Item checkpoints\sft_prime, checkpoints\grpo -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item data\manifest.json, data\outer_loop_state.json -ErrorAction SilentlyContinue
python partition_data.py
python sft_prime.py
python generate_rubric.py
python train_outer_loop.py
```

---

### Deferred — not in active pipeline

| Script | Reason |
|--------|--------|
| `label_regimes.py` | Regime labels (R1–R4) inferred at training time from dataset provenance |
| `index_chroma.py` | ChromaDB index for RAG-based synthetic QA generation |
| `generate_synthetic_qa.py` | Synthetic QA pair generation (ReST feature, not wired in) |
| `score_baseline.py` | Superseded by `score_anchor_set.py` |
| `fold_in.py` | ReST fold-in filter — exists but not called in the active training loop |

---

### Smoke tests

```powershell
python test_pipeline.py
python test_pipeline.py --n 5        # faster
python test_pipeline.py --skip-gpu   # data stages only
```

---

## Regime label design

Regime per `(question, context)` is the spine — reward, abstention, and fold-in all branch on it:

- **R1**: context present, answer in context → reward correctness + grounding
- **R2**: context present, answer NOT supported → reward abstain "Not supported by context"
- **R3**: no context, answerable from knowledge → reward correctness; abstain is wrong here
- **R4**: no context, genuinely unanswerable → reward abstain "Unanswerable"

Do not merge R2 and R4: R2 = retrieval failure; R4 = genuine unanswerability. Distinct gold strings, distinct reward signals.

BERT answer-span identification confirms R1 vs R2 at data-prep time (cached). Do not use span IoU as online RL reward: policy is abstractive, IoU is extractive — a formulation mismatch. Online grounding check for R1 uses NLI entailment (context entails answer). R3 vs R4 cannot be span-labeled (no context) — use dataset provenance, flag as `[inferred]`.

## Data stratification (core): one schema across SFT, rubric, and RL

Every set the policy ever sees — the SFT prime, the rubric calibration set, and each GRPO
training batch — is drawn from **one** joint **regime × domain** stratification schema,
defined once in `stratify.Corpus` (the `CORPUS` singleton) and reused everywhere. This is the
data discipline that makes the abstention signal survive training.

**Why it exists.** A uniform random partition reflects the corpus's natural ~1.7% abstention
rate, so a 50-record rubric set lands ~1 abstention example and a 100-record holdout ~2 — too
few to calibrate the judge or measure abstain-accuracy, and far too few in a 50-record GRPO
batch for the model to keep the "abstain is sometimes correct" behavior. Left uniform, RL
trains the abstention skill out (we observed the swing: 50% SFT abstention → over-abstains;
3.4% → never abstains). Stratification fixes both ends.

**The schema.**
- **Regime** is the free deterministic provenance rule (no model pass):
  `R1 = ctx & ans · R2 = ctx & ¬ans (abstain) · R3 = ¬ctx & ans · R4 = ¬ctx & ¬ans (abstain)`.
- **Joint cell** = `(regime, domain)`. Allocation across the non-empty cells reuses the
  Box-Cox → robust-z → CDF → uniform-blend machinery in `pull_datasets.calculate_allocations`.
**The default is `lam=None` (true raw-proportional: `draw_i = n·size_i/total`, capped at pool size — never repeats).** Because the corpus is now expanded so abstention is naturally ~8.5% from unique data, over-representation is no longer needed; the draw simply preserves the corpus composition. `lam∈[0,1]` remains an optional over-representation knob (the Box-Cox/CDF `calculate_allocations` path) — note that even `lam=1.0` there is CDF-compressed, NOT raw-proportional.

**One schema, three consumers, no drift.** `CORPUS.sample(split, n, lam, seed)` is the single
entry point; SFT, the rubric set, and every RL batch call it on their own split. Because `lam`,
the floors, the regime rule, and the manifest all live in one object, the three sets cannot
disagree — change `lam` once and all of them move together.

**Leakage discipline.** The partition is **stratified-disjoint**: each joint cell is split
across `{sft, eval_holdout, rubric, grpo_train}` by sequential exclusion on the question string,
so no question appears in two splits (asserted in `CORPUS.write_manifest`). Rare abstention cells
are **floored** into rubric (≥2) and holdout (≥3) so they are present for calibration and
measurement. Over-representation only ever **repeats records within a split**, never across — so
nothing that informed the rubric or SFT can leak into RL or eval.

**Inspectable (seeing is believing).** The sampling ratios print as a standing pipeline
artifact: `python stratify.py` dumps the live SFT / RL-batch / corpus tables, `partition_data.py`
prints the per-split regime matrix, and the SFT build + first RL batch print their own draw:

```
  stratification ratios [RL batch draw]  (lam=0.5, n=50):
    cell                   raw    raw%  draw   draw%
    R2|ReadingComp          17    1.3%     3    6.0%   <- abstention boosted ~4x
    R3|Math                424   31.9%     8   16.0%   <- dominant pulled down
    abstain(R2/R4): raw 1.3%  ->  drawn 6.0%

  per-split result of the stratified-disjoint partition (every split carries abstention):
  split            n    R1  R2  R3  R4   abstain%
  sft            400   127   6 267   0      1.5%
  eval_holdout   105    31   3  68   3      5.7%
  rubric          53    16   2  33   2      7.5%
  grpo_train    1330   422  17 891   0      1.3%
```

## Lessons Learned

### Where we are vs what could be (the RL-framing quadrant)

Two orthogonal axes: **on-policy/off-policy** (data source) and **has-a-successor-state / horizon-1**
(problem structure → TD vs MC). A bandit is NOT "not an MDP" — it **is** an MDP, a degenerate one
(single state / horizon-1, no transition). What it lacks is a **successor state**, and the successor is
the entire reason TD exists — the `γV(s′)` bootstrap. Drop the successor and the TD error
`δ = R + γV(s′) − V(s)` collapses to `δ = R − Q(a)`: a sample-mean prediction error, NOT a TD error
("TD" names the bootstrap). We sit in the **horizon-1 row** and never leave it; the only "what could be"
is a horizontal on→off-policy shift:

```
                         on-policy                      off-policy
  horizon-1 / no successor   ◀ WE ARE HERE ▶           teacher-distillation / RWR (deferred → RSI)
  (bandit = degenerate MDP)  GRPO — MC baseline:        train on a STRONGER model's completions;
   → MC, NO bootstrap        group mean = MC analog     off-policy PG needs π_θ/μ importance weights
                             of V(s); advantage = r−mean  that are a product over tokens → explode
  multi-step / has successor  SARSA (TD, needs successor) Q-learning (TD, needs successor)
```

GRPO is gradient-bandit / REINFORCE-family: **no Q-table, no ε-greedy, no bootstrap.** The group mean
is a **Monte-Carlo baseline** (the MC analog of `V(s)`), estimated fresh from the G samples — not
bootstrapped. SARSA/Q-learning need the successor to bootstrap; neither applies. The off-policy cell
(stronger-model generator) turns the experiment into "SFT/distill-then-RL" → deferred to RSI.

**"GRPO variance shrinks over time" is NOT TD-error shrinkage** — there is no TD error in the pipeline.
It's the bandit/MC cousin: as the policy concentrates probability, the G completions homogenize, so
within-group reward spread shrinks, so advantages shrink. Two cautions when logging this:
(1) **raw** within-group reward variance (`r_i`) shrinks, but the **std-normalized advantage** (`A_i`)
is pinned ~unit by the `/std` — plot `r_i` vs `A_i` and you get different stories;
(2) a group going homogeneous (std→0, all-correct or all-wrong) means the **signal dried up** → it
contributes **no gradient** — half "the policy learned," half a **failure mode** (the difficulty bias
Dr. GRPO drops the std to avoid; we mask these via `:ZeroVarMask:`). Distinguish convergence from a
drying well.

A corollary worth stating: with a **static** judge at temp=0 (order-averaged, fixed seeds), the
reward model *is* the ground-truth `q*` **by construction** — `q*` is whatever the reward function
defines. So RMSE/calibration against it is well-defined (e.g. 2B-vs-9B judge agreement). The only
remaining stochasticity is the *generation* (the policy's distribution over completions), which the
G-sample group estimates Monte-Carlo style.

### Hard-won fixes

- **Dead reward signal (0.5-collapse).** The first judge rubric scored 92% of completions exactly
  0.5 → zero within-group variance → zero gradient. Root cause was the **rubric**, not the model
  size (the 9B collapsed too) or the schema. Fix: a **continuous** rubric (full 0.0–1.0, gold is a
  verified reference not a phrasing ceiling, conciseness counts) + a **verify-gate** composite reward
  (programmatic correctness for math/code × continuous judge quality).
- **Judge array short-close.** A bare-array Ollama `format` lets the model close the JSON array
  early and emit the item *count* (`[15]`, `[1]`) instead of N scores, aborting runs. Fix: a
  **required-keyed object schema** (`q0..q{n-1}`, all required) the grammar cannot short-close, plus
  a recursive split that only fails on a single genuinely un-scoreable pair (never fabricates).
- **9B rubric synthesis needs `num_ctx`.** Packing all 50 examples into the synthesis prompt
  overflowed the default context and silently truncated → the 9B produced a generic 3-bin rubric.
  Raising `num_ctx` + truncating per-pair fields fixed it.
- **GPU contention stalls everything.** A background process pinning a 9B in VRAM forced our 0.8B
  onto CPU → a 6-minute job ran 55+ minutes. Always confirm the GPU is free, not just "a model is
  loaded."
- **Corpus hygiene.** 69 duplicate questions (2000 records / 1888 unique). No cross-split leakage,
  but they inflate split counts — dedup to one record per question.
- **Judge batch contamination → score one answer at a time.** The object-schema judge can score many (question, gold, answer) pairs in a single prompt; this CONTAMINATES each score — neighbouring pairs and position bleed in. Proven: an empty-gold fabrication scored 0.00 in isolation (5/5) but 0.95 in a batch; in a greedy SFT-vs-GRPO head-to-head, 8/18 byte-identical completions scored differently purely from batch composition. **Fix: isolated single-pair scoring** (one pair per judge call). At temp 0 the isolated call is deterministic → it IS the canonical score, which also obviates the old "3× randomized-order averaging" denoiser (that only existed to cancel batch-order bias). This contaminated the anchor distribution, the GRPO eval, AND the GRPO training reward — the first apparent GRPO win (+0.29 batched → +0.10 on the isolated head-to-head) was largely a measurement artifact, caught only by reading the actual completions (seeing-is-believing).
- **Judge model: 4B → 2B.** Once scoring is a single isolated call, the judge need not be large — only CONSISTENT and STRONGER than the 0.8B policy (the POC is "a weak model learns to satisfy a stronger model's judgement"). `qwen3.5-oc:2b` applies the rubric correctly (validated: fabrication-on-empty-gold → 0.00, abstention → 0.90) at ~half the latency. Recover throughput with `OLLAMA_NUM_PARALLEL>=2` + the smaller judge, NEVER by re-batching. The 9B is rubric-AUTHOR only; it never scores.
- **Corpus expansion for unique diversity.** A stratified draw over a too-small corpus REPEATS rare cells (6 unique R2 → 25 copies). Expanded 1888 → 2649 (`generate_r4.py` synthetic R4 + `expand_corpus.py` thin-cell top-ups to per-cell floors) so every (regime×domain) cell has enough UNIQUE records; abstention is now ~8.5% from REAL examples, no repetition.
- **Proportional sampling is `lam=None`, not `calculate_allocations(lam=1.0)`.** The Box-Cox/CDF allocator COMPRESSES the distribution even at lam=1.0 (over-draws small cells → repetition). True raw proportional (`draw_i = n·size_i/total`) is a separate path (`stratify.proportional_allocations`, `CORPUS.lam=None`) — that's what eliminates repetition.
- **Two-stage rubric (reproducible).** `generate_rubric.py` authors the BASE quality rubric from the representative 50; `augment_rubric.py` layers the explicit R1–R4 / empty-gold ABSTENTION rules on top (the rubric split is too thin in abstention for from-scratch authoring to learn them). Both are wired into `run_pipeline.py` (steps 4 + 4a) so a from-scratch run reproduces the validated rubric.

### The R-gap: use SFT to match the gold regime prior BEFORE RL

The biggest blocker was **over-abstention**, not capability. The SFT mix was ~50% abstention
(R2+R4) while the **real corpus rate is ~2%** — a 25× over-representation — and **Agentic +
Instruction had zero SFT coverage**. Result: the policy emitted "Unanswerable" on 46% of
generations (Agentic 92%, Math 54%) — on *answerable* questions. With every completion identical
(abstain), there is no reward variance, so GRPO has no gradient. RL cannot fix what it cannot vary.

The fix is to treat the **gold answers' per-domain regime distribution as the target prior** and use
SFT (on-policy, our own model — *not* a teacher) to close the gap between the policy's generations
and that prior:

```
TARGET regime prior (from gold provenance):
  Agentic 100% R1 | Math 100% R3 | Code 99% R3 | STEM 32% R1 / 68% R3
  Instruction 10% R1 / 90% R3 | ReadingComp 91% R1 / 9% R2 | overall 98% attempt / 2% abstain
```

So: **add SFT records for the under-represented R groups per domain** (Agentic R1 function-calls,
Math/Code R3, keep ReadingComp's ~9% R2, a *minimal synthetic* R4 since the corpus has ~0). Once
the policy *attempts* the right regime, its attempts vary in quality (right / partial / wrong) →
reward variance appears → GRPO gets a gradient. **Closing the R gap is the lever; the variance is
the consequence.** Match the prior first, then run RL.
