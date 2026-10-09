# SETTLED — decisions that are NOT to be re-litigated

If something here is wrong, change it here. Do not re-derive it in conversation,
and do not make the operator restate it. Added 2026-09-12 because repeatedly
re-asking settled questions is railroading.

## Models and lanes

- **Judge is the 4B**, `qwen3.5-reasoning-thinking-256k:4b`. The 2B is DELETED
  from the codebase (2026-09-12); every path resolves to the 4B.
- **Policy is Qwen3.5-0.8B**, the operator's own `skills_model` checkpoint.
  Not an HF re-download, not a reconstruction.
- **Lane is r3** — closed-book, no context, every question answerable, NO
  abstention. Rubric: `data/synthetic_domain_rubric.md`.

## Method

- **GRPO with our modifications** is the thing being proven. Dr.GRPO: no
  division by group std.
- **maxent** is mgpo's round-2 prompt selection (retain prompts whose reward
  history SPANS the correctness floor, then Box-Cox reweight). Borrowed where it
  helps; not a replacement for proving GRPO itself.
- **QLoRA is OPTIONAL, never the default.** `--qlora` is off unless asked.
  Adapters for knowledge INJECTION are FALSIFIED (instructions.md 8) — do not
  re-open, do not add LoRA arms to sweeps.
- **Full-parameter** means what each stage's code says: SFT is FFN-only
  (attention frozen, Geva et al.); RL trains all weights with embeddings frozen.

## Reward shaping

Quality is a multiplier GATED BY CORRECTNESS, never an independent axis:

            lq     hq
    C      0.5    1.0
    I      0      0

No partial credit for a wrong answer at any quality. Raw judge score feeds
selection, never the reward directly.

## Protocol

- **3-read majority judging.** Single-read reads ~4 points higher and is noise.
- **Batch shape (operator, 2026-09-12).** Records per optimizer step, each with
  `--num-generations 4`, so the group the advantage is computed over is
  records x generations:

  | stage | records/step | generations/step | pool |
  |---|---|---|---|
  | HPO screen | 4  (`--grad-accum 4`)  | 16 | 25% = 60 of 240 |
  | FULL run   | 16 (`--grad-accum 16`) | 64 | all 240 |

  Only the HPO winner gets the full run.
- **Report the count out of 16 at EVERY step**, not just the endpoint.
- **Judge a stage by its BEST checkpoint**, never its first or last epoch.
- **Every run states --seed.** A seed is NOT determinism: two seeded runs of one
  unchanged model read 6/16 and 7/16.
- **NOT TESTED is a third verdict.** A blocked or invalid run is never a loss.

## Which trainer

- `src/mgpo.py` is the PRODUCTION trainer and the process every slide number came
  from. Use it for anything that must match the board.
- `docs/grpo_one_file.py` is the single-file port, for readability and for the
  book-provenance story. It does NOT have maxent.

## Measured instrument facts (2026-09-12)

- Base = **5/16** under the 4B judge with 3-read majority.
- Think AND tool-call tokens must be suppressed at the logits level; the model
  otherwise burns its whole budget in `<think>` or loops `<tool_call>`.
- Prompt format: `--prompt-format template` when training from the stock base;
  `chat_wrap` costs 7 of 16 questions there.
- A completion with no stop token is a FAILED generation, reward 0.
- Degenerate answers (empty, role-echo) are scored 0 mechanically, never judged.

## Judging: local, and why (measured 2026-09-12)

The hosted endpoint (`sprawl/nvidia_llm.py`, deepseek-v4-flash) grades CORRECTLY
with the same r3 rubric -- 0.95 / 0.95 / 0.0 / 0.1 on a right/right/wrong/wrong
probe -- but it is **slower**, not faster:

    remote, concurrent : 222.7 s for 4 answers = 55.7 s/answer
    local 4B           :  88   s for 4 answers = 22.0 s/answer

So `JUDGE_REMOTE=1` exists to FREE THE GPU (no policy offload, judge runs off-box),
never as a speedup. Reported numbers stay on the local 4B.

Remote judging, when used, is CONCURRENT: one small call per answer through a
thread pool, not one giant object-schema call. The serial shape exists because
ollama serves one request at a time; a hosted endpoint has no such constraint and
a 16-item single call was measured hanging past 10 minutes.

## Latency guard (skill: timeout-guard)

Runaway generations get a ceiling, and the response is a SEED CHANGE, not a longer
wait. Policy, from the skill:

- hard cap 300 s always wins
- thresholds in LOG space (latency is right-tailed; nominal median+MAD misfires)
- kill at exp(log_median + 4*log_mad): MAD ~= 0.6745 sigma, so 4*MAD is Tukey's
  1.5*IQR fence (~2.7 sigma)
- no threshold until 5 completions; MAD == 0 means DO NOT FLAG
- only completions feed the baseline -- never kills, or it anchors to stale "fast"

## ReFT detour (2026-09-12)

`tools/reft_hpo.py` runs completion tuning through the IDENTICAL harness (25%
screen, 4 records/step, same eval, same judge) to prove the e2e cheaply before
spending RL time. pyreft trains 16,392 intervention params against 752M frozen
(0.0022%), so it is minutes not hours. pyvene lacks a Qwen3.5 mapping; it is
registered against Qwen2's, whose decoder stack is identical.

`repeng` is NOT usable: broken upstream on NumPy 2.0 (`np.float_` removed), and it
builds a steering vector by PCA over contrastive pairs -- no completion loss and no
batch shape, so it does not map onto "4 records per step" anyway.

## RLVR verdict (measured 2026-09-12)

- **numeric tier WORKS** — exact match on boxed/#### /gold numbers, 43% of the
  240-record pool, cannot be wrong when it fires.
- **entailment tier FAILED calibration** — 8 of 9 ratios at/below 0.25 with no
  separation; best agreement 4/16 vs the judge's 8/16 at ANY threshold. The signal
  is absent, not mistuned. Do not re-open without changing the question being asked.
- **Use `--reward hybrid`**: numeric first, judge for the rest. `--reward rlvr`
  alone is not a valid instrument on this corpus.
- Generations are cached (`GEN_REUSE=1`, keyed per checkpoint) so any re-scoring is
  free; the tags MUST differ between pre- and post-training evals.
