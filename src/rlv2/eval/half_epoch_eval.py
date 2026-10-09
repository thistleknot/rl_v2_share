"""src/rlv2/eval/half_epoch_eval.py — F1 on the frozen eval at every half-epoch, in EVERY phase.

Notes: `chat_wrap` here is the eval wrapper; docs/dispositions.md "THE EVAL PROMPT IS NOT THE TRAINING PROMPT"
(2026-09-25) records that training used the model's own chat template instead. Slides: talk slides 9 and 10, lab
notebook slide 23 (docs/PROVENANCE.md).

OWNER REQUIREMENT (2026-08-27): measurement is not a per-phase option. CPT, SFT and RL all
report at every half-epoch boundary, whether the run is an HPO trial or a full run, and each
report carries three things:

    1. GENERATED OUTPUTS  -- what the model actually said, not only what it scored
    2. F1 PER QUESTION CATEGORY  -- Math / Code / STEM / Instruction
    3. THE SUPPORT behind each number -- a rate over 4 items and a rate over 40 are not the
       same claim and must never print identically

WHY THIS EXISTS. `cpt.py` had no eval callback at all: a run reported one number at the end
and the SHAPE of the curve -- the instrument the whole experiment reads for lift -- was
invisible. `vibe_sft.py` had one that fired at a single hardcoded step. Neither gave a
per-epoch curve.

IN-PROCESS, NOT A SUBPROCESS. The obvious implementation shells out to
`rl_calibration_eval.py`, but that scores a checkpoint on disk, so every half-epoch would
cost a ~4.3 GB full-param save -- the same writes that filled the drive and killed three
runs. The model is already resident on the GPU; generate from it directly.

Require   - a frozen eval JSONL with `prompt`/`question`, `gold`, `category`, `item_id`.
Guarantee - prints a per-category table with support at every half-epoch; never raises into
            the training loop.
Maintain  - EVAL-ONLY. Never writes to the eval file, never writes a checkpoint, never
            touches optimizer state. Restores the model's training mode on exit.
Assert    - the model is put back in train() mode even when generation raises.
"""
from __future__ import annotations

import json
import os

import torch
from transformers import TrainerCallback

from rlv2.eval import score_r3_categories as sc

EVAL_FILE = "data/rl/r3_eval_canonical.jsonl"

# Decision cells, defined exactly as rl_calibration_eval.py defines them so the two
# instruments cannot disagree:
#
#     should_abstain = (gold is EMPTY)
#     tp  should_abstain AND declined    correctly abstained
#     fn  should_abstain AND answered    confabulated on an unanswerable item
#     fp  should ANSWER  AND declined    over-abstained: refused something it could answer
#     tn  should ANSWER  AND answered    attempted -> split into correct / wrong
#
# ON THE R3 EVAL tp AND fn ARE STRUCTURALLY UNREACHABLE. Every one of the 20 frozen items
# has a non-empty gold, so `should_abstain` is never true and both cells are pinned at 0
# forever. Reading them as "the model never confabulates" would be reading a definition,
# not a result. The cells that MOVE here are fp (refused an answerable question) and tn,
# and confabulation on R3 surfaces as ANSWERED WRONG inside tn -- printed as its own column
# rather than left to be inferred from a rate.
DECLINE_MARKERS = (
    "i don't know", "i do not know", "cannot answer", "can't answer", "unable to answer",
    "not enough information", "insufficient information", "no information",
    "unanswerable", "i'm not sure", "i am not sure",
)
# DERIVED FROM THE GOLDS, NOT PICKED. Operator 2026-09-16, on being shown a hand-set 96:
# a hard cap chosen by hand is the same class of magic number that has bitten this campaign
# repeatedly. Measured the same day over data/rl/r3_eval_canonical.jsonl:
#
#     gold tokens (n=16):  min 4   p50 42   p90 79   max 99
#     cap 96  -> covers 15 of 16 golds
#
# One eval question's own gold is LONGER than the budget, so that question could not be
# answered in full and the no-EOS gate scored the truncation 0. ceil(max_gold * 1.5) = 148
# covers every record with headroom for a model that phrases an answer longer than the gold.
# tests/test_eval_budget.py re-derives this from the live eval file, so a changed eval set
# fails the test instead of silently truncating.
#
# WHY NOT A TIMEOUT INSTEAD (operator's alternative, 2026-09-16): a wall-clock cut depends on
# GPU load, so the same weights get cut at different points on different runs -- which is the
# nondeterminism just removed from this path. And retry-on-timeout cannot converge at
# temperature 0: the retry reproduces the identical runaway. Time budgets belong on TRAINING
# rollouts (sampled, where a stall is the risk); a measurement needs a deterministic bound.
MAX_NEW_TOKENS = 148
# ALL of them, every time. Owner 2026-08-27: "I LITERALLY CAN'T SEE THE OUTPUTS".
# A count table cannot distinguish a wrong answer from a chat-turn echo from a think-trace
# leak, and each needs a different fix -- three separate defects this run were invisible in
# the totals and obvious the moment the text was on screen. 0 = show every item.
N_SHOW = 0


def load_eval(path: str = EVAL_FILE) -> list[dict]:
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# THE SYSTEM PROMPT IS IMPORTED, NOT RESTATED. `build_cpt_mix_r3.NTP_SYSTEM` is what every
# training record was rendered with, and a second hand-written copy here is a train/inference
# skew of exactly the kind that already bit this file twice: once as a missing chat template
# (the model answered by writing the next `user` turn) and once as an injected empty <think>
# block. A drifting system prompt is the same defect wearing a third hat, and it is invisible
# in every metric -- the model simply answers a slightly different question than it was
# trained to answer.
try:
    from rlv2.data.cpt_mix import NTP_SYSTEM as SYSTEM
except Exception:                                            # noqa: BLE001
    SYSTEM = ("You are a helpful assistant. Answer the question directly and concisely. "
              "If the question cannot be answered from available information, say so "
              "clearly.")


def strip_reasoning(s: str) -> str:
    """Remove <think>...</think> traces before the judge sees the answer.

    INCUMBENT REUSED: this is build_pools_r3.strip_reasoning, imported when available so the
    two paths cannot drift, with a local fallback because importing that module pulls in the
    whole generator.

    Currently the chat template pre-fills a CLOSED empty block (`<think>

</think>`) so
    nothing leaks -- measured 0 of 6. That is not a guarantee: CPT trains on raw NTP text and
    can teach the model to open its own `<think>`, and these GGUFs are documented to emit
    reasoning despite think:false (§6.11.1). Grading a reasoning trace as if it were the
    answer would score the model's scratch work.

    An UNCLOSED trace means generation was cut mid-reasoning: there is no answer after it, so
    drop from the opening tag rather than hand the judge scratch work.
    """
    if not s:
        return ""
    try:
        from rlv2.data.pools import strip_reasoning as _incumbent
        return _incumbent(s).strip()
    except Exception:                                        # noqa: BLE001
        import re as _re
        s = _re.sub(r"<think>.*?</think>", "", s, flags=_re.S | _re.I)
        s = _re.sub(r"<think>.*$", "", s, flags=_re.S | _re.I)
        return s.strip()


def chat_wrap(tok, question: str) -> str:
    """Build the prompt EXACTLY as training wrote it. Think is OFF everywhere.

    TRAIN/INFERENCE PARITY IS THE POINT. `build_cpt_mix_r3.render_ntp` writes

        <|im_start|>assistant\n{answer}<|im_end|>

    so CPT teaches the model to begin the answer immediately after `assistant`+newline.
    But `apply_chat_template(..., enable_thinking=False)` still emits a pre-filled empty
    block -- `<think>` newline newline `</think>` -- four tokens the model never saw in
    that position during CPT.

    MEASURED: 0 of 60 QA golds, 0 of 20 eval golds and 0 of 133 mix records contain any
    think marker. The training distribution is think-FREE, so inference must be too.

    The skew gets WORSE the more CPT runs, which matches the observed regression
    (base 2/20 -> CPT 1/20, not_attempted 3 -> 9): the harder the model is trained to
    answer right after the assistant tag, the more an injected block throws it off.

    Built by hand rather than through the template so the two paths cannot drift again.
    If the training renderer changes, change this with it -- they are one decision.
    """
    # THE SYSTEM PROMPT STAYS. Operator 2026-09-12, final: "I think to keep
    # things consistent. We'll just mark abstention wrong since we know all are
    # answerable and keep system prompt."
    #
    # It was briefly removed the same day, which changed the instrument: the SFT
    # checkpoint read 5/16 bare vs 8/16 with the prompt (1-read vs 3-read, so not
    # a clean contrast -- but the wrong direction, and it discarded comparability
    # with every historical number). CONSISTENCY WON. The prompt's abstention
    # clause is handled where it belongs, in SCORING: R3 has zero abstain items
    # (0/16 eval, 0/240 train have an empty gold), so an abstention is marked
    # WRONG mechanically -- see reward.py's abstention guard.
    return ("<|im_start|>system\n%s<|im_end|>\n"
            "<|im_start|>user\n%s<|im_end|>\n"
            "<|im_start|>assistant\n" % (SYSTEM, question))


# THE ONE BANNED LIST, for every phase. Operator 2026-09-15: "Think and tool calling
# needs to be suppressed all phases, we've visited this b4". Resolved through the
# tokenizer rather than hardcoded -- `src/rlv2/train/rl/grpo.py` and `src/rlv2/train/sft/train.py` both carried a
# literal [248068, 248069], which is think-only and silently wrong against any other
# tokenizer. Tool-call markers were suppressed in NO phase at all before this.
BANNED_MARKERS = ("<think>", "</think>",
                  "<tool_call>", "</tool_call>",
                  "<tool_response>", "</tool_response>")


def banned_ids(tok) -> list[int]:
    """Token ids no phase may ever generate: reasoning blocks and tool calls."""
    ids = []
    for t in BANNED_MARKERS:
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            ids.append(i)
    return sorted(set(ids))


def _think_ids(tok) -> list[int]:
    """The token ids that OPEN a reasoning block, so generation cannot emit one.

    WHY THIS IS MECHANICAL AND NOT A PROMPT. `chat_wrap` says "Think is OFF everywhere"
    and enforces it by writing a think-free prompt -- which is a REQUEST. SFT on the R3
    domain set taught the model to open its own `<think>` anyway, and at
    MAX_NEW_TOKENS=96 the block never closes, so `strip_reasoning`'s unclosed-block rule
    deletes the whole completion.

    MEASURED 2026-09-15 on sft_full_epoch3, 16 eval questions, temperature 0.7:

        suppress=False   think-opened 9/16   empty-after-strip 6/16
        suppress=True    think-opened 0/16   empty-after-strip 0/16

    and the base model opens none either way, so this repairs the trained checkpoint
    without moving the baseline it is compared against.
    """
    return banned_ids(tok)


def _stop_ids(tok) -> list[int]:
    """End the answer at the end of the ASSISTANT TURN, not at the context limit.

    Without this the model runs past its answer into the next `<|im_start|>user` turn and
    the eval scores a transcript instead of an answer.
    """
    ids = []
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            ids.append(i)
    if tok.eos_token_id is not None:
        ids.append(tok.eos_token_id)
    return sorted(set(ids)) or None


def generate(model, tok, prompts: list[str], max_new_tokens: int = MAX_NEW_TOKENS,
             batch_size: int = 4, wrap: bool = True, temperature: float = 0.0) -> list[str]:
    """Completions from the LIVE model.

    temperature=0 is GREEDY and is the default for half-epoch readings: a reading is a
    measurement, and sampling noise inside a training curve looks like a trend.

    temperature>0 enables k-sample estimation, which is what Gate A's spec actually asks
    for ("3 items x k~8"). At k=1 on 20 items the standard error is +-1.3 items, so a real
    2-item effect is invisible -- every null measured tonight sits inside that band. Raising
    k is the highest value-per-GPU-minute change available to this experiment.
    """
    if wrap:
        prompts = [chat_wrap(tok, q) for q in prompts]
    _suppress = _think_ids(tok)
    if _suppress and getattr(model, "generation_config", None) is not None:
        model.generation_config.suppress_tokens = _suppress
    outs: list[str] = []
    was_training = model.training
    model.eval()
    # Left padding: with right padding a batched decoder generates from PAD positions and
    # returns truncated or empty strings -- which reads as "the model learned nothing".
    old_side = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                chunk = prompts[i:i + batch_size]
                enc = tok(chunk, return_tensors="pt", padding=True,
                          truncation=True, max_length=512).to(model.device)
                gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                     do_sample=temperature > 0,
                                     temperature=temperature if temperature > 0 else None,
                                     top_p=0.95 if temperature > 0 else None,
                                     eos_token_id=_stop_ids(tok),
                                     # BOTH PLACES. `generation_config` is what a
                                     # checkpoint's own saved config can override, and the
                                     # call argument is what wins for this call; setting
                                     # only one leaves the other free to re-enable think.
                                     suppress_tokens=_suppress or None,
                                     pad_token_id=tok.pad_token_id)
                for j in range(len(chunk)):
                    new = gen[j][enc["input_ids"].shape[1]:]
                    outs.append(strip_reasoning(
                        tok.decode(new, skip_special_tokens=True)))
    finally:
        tok.padding_side = old_side
        if was_training:
            model.train()
    return outs


def evaluate(model, tok, rows: list[dict], tag: str,
             temperature: float = 0.0) -> dict | None:
    """Generate, score, print. Returns the summary, or None if anything went wrong.

    Never raises into the training loop: a measurement that can kill a 30-minute run is
    worse than a missing measurement.
    """
    if not rows:
        print("[eval] %s: eval file empty or missing — SKIPPED" % tag, flush=True)
        return None
    # THE JUDGE IS THE INSTRUMENT (REQ-J1). Token overlap was measured incapable of
    # separating right from wrong on this eval: base "wrong" answers scored 0.06-0.41 while
    # the single correct one scored 0.528, so no threshold divides them. The judge returns a
    # VERDICT, and that verdict -- not a threshold on a score -- is what the gate counts.
    try:
        from rlv2.judge import judge as J
    except Exception as e:                                    # noqa: BLE001
        print("[eval] %s: judge unavailable (%s) — SKIPPED" % (tag, e), flush=True)
        return None

    try:
        # `question` FIRST: `prompt` is a pre-baked raw-text blob, and chat_wrap needs the
        # bare question so it can build the turns itself.
        prompts = [(r.get("question") or r.get("prompt") or "") for r in rows]
        gens = generate(model, tok, prompts, temperature=temperature)
    except Exception as e:                                    # noqa: BLE001
        print("[eval] %s: generation failed (%s) — SKIPPED" % (tag, e), flush=True)
        return None

    # OFFLOAD THE POLICY BEFORE JUDGING. Generation is done; the policy is not needed again
    # in this function, but it is still holding VRAM. ollama then has nowhere to put the 4B
    # judge and falls back to thrashing: measured >120 s per judge call with the policy
    # resident versus ~5 s without it, which turned a 15-item read into a 45-minute stall.
    #
    # The tech steering doc already states the rule -- "Policy and judge never co-reside:
    # unload -> score -> reload" -- and reward.py implements it for the RL path. This is the
    # same rule on the eval path, which never had it.
    #
    # Restored after judging so the training callback can keep going. Never raises: a
    # measurement must not be able to kill a run.
    _dev = None
    try:
        _dev = next(model.parameters()).device
        if _dev.type == "cuda":
            model.to("cpu")
            torch.cuda.empty_cache()
    except Exception:                                        # noqa: BLE001
        _dev = None

    recs, cells = [], {}
    n_judge_fail = 0
    for r, g in zip(rows, gens):
        cat = r.get("category", "?")
        v = J.judge(r.get("question") or r.get("prompt") or "", r.get("gold", ""), g, cat)
        n_judge_fail += not v["ok"]
        # `reward` carries the graded score for RL; the VERDICT carries the decision.
        recs.append({"item_id": r.get("item_id") or r.get("id"),
                     "category": cat, "reward": float(v["score"]),
                     "verdict": v["verdict"], "quality": v["quality"]})

        c = cells.setdefault(cat, {"correct": 0, "wrong": 0, "not_attempted": 0, "n": 0})
        c["n"] += 1
        if v["verdict"] == "correct":
            c["correct"] += 1
        elif v["verdict"] == "incorrect":
            c["wrong"] += 1
        else:
            c["not_attempted"] += 1

    # UNLOAD THE JUDGE. The policy is resident on the GPU for training; ollama loading the
    # 4b judge alongside it pushed the card to 16076 MiB of 16384 and training slowed from
    # 76 s/step to 180 s/step -- a 2.4x tax paid every step AFTER an eval, not just during
    # it, because the judge stays resident until evicted.
    #
    # The tech steering doc already states the rule: "Policy and judge never co-reside:
    # unload -> score -> reload." This is that rule applied to the eval callback.
    #
    # Failure to unload is not fatal -- a slow run still produces a number -- so this never
    # raises into training.
    try:
        from rlv2.judge.reward import unload_ollama_model
        unload_ollama_model(J.JUDGE_MODEL)
    except Exception:                                        # noqa: BLE001
        pass

    # Put the policy back where it was, now that the judge has released the card.
    if _dev is not None and _dev.type == "cuda":
        try:
            model.to(_dev)
        except Exception:                                    # noqa: BLE001
            pass

    summary = sc.summarise(recs, k=1)
    print("\n" + sc.render(summary, tag), flush=True)

    tot = {k: 0 for k in ("n", "correct", "wrong", "not_attempted")}
    inst = J.instrument()
    print("  JUDGE VERDICTS per category (support = n) — %s / rubric %s"
          % (inst["judge_model"], inst["rubric_sha256_16"]), flush=True)
    print("  %-13s %4s %8s %10s %14s"
          % ("category", "n", "correct", "incorrect", "not_attempted"), flush=True)
    for cat in sc.CATEGORY_ORDER:
        c = cells.get(cat)
        if not c:
            print("  %-13s %4d %8s %10s %14s" % (cat, 0, "-", "-", "-"), flush=True)
            continue
        for k in tot:
            tot[k] += c[k]
        print("  %-13s %4d %8d %10d %14d"
              % (cat, c["n"], c["correct"], c["wrong"], c["not_attempted"]), flush=True)
    print("  %-13s %4d %8d %10d %14d"
          % ("TOTAL", tot["n"], tot["correct"], tot["wrong"], tot["not_attempted"]),
          flush=True)
    if n_judge_fail:
        print("  WARNING: %d/%d judge calls FAILED and were counted not_attempted. "
              "Treat this reading as a floor, not a measurement." % (n_judge_fail, len(rows)),
              flush=True)
    summary["cells"] = cells
    summary["judge"] = inst
    # Per-record verdict+reward: what the three projections (binary / mean|correct / KL)
    # are computed FROM. Without these the summary supports only the binary.
    summary["records"] = recs

    # OUTPUTS, not just scores. A table of zeros does not distinguish "wrong answer" from
    # "degenerate loop" from "empty string", and those need different fixes.
    shown = len(gens) if N_SHOW <= 0 else min(N_SHOW, len(gens))
    print("  GENERATED (%d of %d):" % (shown, len(gens)), flush=True)
    for r, g, rec in list(zip(rows, gens, recs))[:shown]:
        print("   [%-11s %-13s] %s"
              % (rec["category"], rec.get("verdict", "?"), (r.get("question") or "")[:110]),
              flush=True)
        print("      gold: %s" % (r.get("gold", "") or "")[:300], flush=True)
        print("      got : %s" % (g or "<EMPTY>")[:300], flush=True)
        print("      score %.3f  quality %.3f"
              % (rec["reward"], rec.get("quality", 0.0)), flush=True)
    return summary


class HalfEpochEval(TrainerCallback):
    """Fires at every half-epoch boundary, plus once at the end.

    `evals_per_epoch` is a knob rather than a hardcoded 2 so a short run (24 steps over 3
    epochs is 8 steps/epoch) can still produce a curve with enough points to read.
    """

    def __init__(self, model, tok, eval_file: str = EVAL_FILE,
                 evals_per_epoch: int = 2, tag: str = "run", epochs: int | None = None):
        self.model = model
        self.tok = tok
        self.rows = load_eval(eval_file)
        self.evals_per_epoch = max(1, evals_per_epoch)
        self.tag = tag
        # EPOCHS MUST BE PASSED IN, not read off TrainingArguments. cpt.py pre-builds every
        # epoch into ONE dataset and therefore sets num_train_epochs=1, so trusting the
        # trainer would treat a 3-epoch run as a single epoch and fire twice for the whole
        # run instead of twice PER EPOCH -- two points is not a curve.
        self.epochs = epochs
        self.history: list[dict] = []
        self._fired: set[int] = set()
        self._every = None

    def on_train_begin(self, args, state, control, **kwargs):
        total = int(state.max_steps or 0)
        epochs = float(self.epochs or args.num_train_epochs or 1)
        epochs = max(1.0, epochs)
        per_epoch = max(1, int(total / epochs))
        self._every = max(1, per_epoch // self.evals_per_epoch)
        print("[eval] half-epoch F1 every %d steps (%d total, ~%d/epoch) on %d frozen items"
              % (self._every, total, per_epoch, len(self.rows)), flush=True)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if not self._every or state.global_step in self._fired:
            return control
        if state.global_step % self._every:
            return control
        self._fired.add(state.global_step)
        s = evaluate(self.model, self.tok, self.rows,
                     "%s step %d (epoch %.2f)" % (self.tag, state.global_step,
                                                  state.epoch or 0.0))
        if s:
            self.history.append({"step": state.global_step, "epoch": state.epoch,
                                 "rate": s["rate"],
                                 "per_category": {c: v["rate"]
                                                  for c, v in s["per_category"].items()}})
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if state.global_step not in self._fired:
            s = evaluate(self.model, self.tok, self.rows, "%s FINAL" % self.tag)
            if s:
                self.history.append({"step": state.global_step, "epoch": state.epoch,
                                     "rate": s["rate"]})
        if len(self.history) > 1:
            print("\n[curve] %s — the SHAPE is the result, not the last number" % self.tag,
                  flush=True)
            print("  %-8s %-8s %s" % ("step", "epoch", "rate"), flush=True)
            for h in self.history:
                print("  %-8d %-8.2f %.3f" % (h["step"], h["epoch"] or 0, h["rate"]),
                      flush=True)
        return control


# ── standalone k-sample readout ───────────────────────────────────────────────────────
# The same instrument the training callback uses, driven from the command line so a
# checkpoint can be scored OUTSIDE a training run. Gate A asks for "k~8"; at k=1 on 20 items
# the standard error is +-1.3 items, which is wider than any effect this experiment has
# produced, so every gate decision needs k>1. The k=8 readings taken on 2026-08-28 were
# produced by an ad-hoc script that was never saved -- this is that driver, kept.
#
# Repeating each row k times is what turns one pass into k draws: `summarise` groups by
# item_id, so the k copies aggregate into a per-item correct/k automatically.

def load_policy(path: str, adapters: str | None = None):
    """Load a full-param checkpoint or a base+adapter pair.

    A full-param dir has no adapter_config.json and must be loaded directly; PeftModel would
    reject it. Branch on what is on disk rather than assuming a shape.
    """
    import torch as _t
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=_t.float16,
                                                 trust_remote_code=True)
    if adapters:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapters)
    model.eval()
    return model.to("cuda" if _t.cuda.is_available() else "cpu"), tok


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="k-sample R3 readout for one checkpoint")
    ap.add_argument("--model", required=True, help="full-param dir, or the base for --adapters")
    ap.add_argument("--adapters", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--eval-file", default=EVAL_FILE)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="k>1 needs sampling or the k draws are identical")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.k > 1 and a.temperature <= 0:
        raise SystemExit("--k > 1 with temperature 0 gives k IDENTICAL greedy draws; "
                         "that is k=1 at k times the cost. Set --temperature > 0.")

    rows = load_eval(a.eval_file)
    if not rows:
        raise SystemExit("eval file %s is empty or missing" % a.eval_file)
    print("[%s] %d items x k=%d = %d observations  (temp %.2f)"
          % (a.tag, len(rows), a.k, len(rows) * a.k, a.temperature), flush=True)

    model, tok = load_policy(a.model, a.adapters)
    # k copies of every row: summarise() groups by item_id, so this is a k-sample estimate.
    expanded = [dict(r) for _ in range(a.k) for r in rows]
    summary = evaluate(model, tok, expanded, a.tag, temperature=a.temperature)
    if summary is None:
        raise SystemExit("evaluation returned nothing -- see the SKIPPED reason above")
    summary["k"] = a.k
    summary["model"] = a.model
    summary["adapters"] = a.adapters
    summary["eval_file"] = a.eval_file

    out = a.out or os.path.join("data", "r3_k%d_%s.json" % (a.k, a.tag))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nTOTAL %d/%d correct  rate %.4f   ->  %s"
          % (summary["correct"], summary["n_obs"], summary["rate"], out), flush=True)


if __name__ == "__main__":
    main()
