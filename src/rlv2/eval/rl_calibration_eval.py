"""src/rlv2/eval/rl_calibration_eval.py — score RL on the reward it was DESIGNED against: the R1-R4 crosshatch.

TWO LAYERED ERRORS THIS FIXES, both mine.

(1) Every RL number in this campaign was `entity_validity` accuracy on the skills QA set -- the SFT
    metric, "can it answer?". That metric has no abstention term, so it returns ~0 no matter what RL
    did. Three "RL contributed nothing" claims rested on it and are withdrawn. A null on the wrong
    readout is not a null.

(2) My first replacement was ALSO wrong: it collapsed abstention into a single class. The design is a
    crosshatch of REGIME x DOMAIN, and the two abstention regimes demand DIFFERENT phrases:

      R1  context, answerable      0.6*F1 + 0.4*NLI_grounding(context, answer)
      R2  context, unanswerable    1.0 iff answer contains "not supported by context"
      R3  no context, answerable   F1; containing EITHER abstain phrase scores 0
      R4  no context, unanswerable 1.0 iff answer contains "unanswerable"

    R2 means *I have context and it does not answer this*. R4 means *I have no context and will not
    confabulate*. Merging them destroys the distinction the reward exists to teach.

NOT EVERY DOMAIN CARRIES EVERY REGIME (rl_eval, 112 items):
    R1 37  Agentic 9 / Instruction 9 / ReadingComp 10 / STEM 9
    R2  4  ReadingComp ONLY          <- 4 items. Report the COUNT, never a bare rate.
    R3 53  Code 13 / Instruction 9 / Math 23 / STEM 8
    R4 18  3 each across all 6 domains

THE FINDING THIS EXPOSES. `mgpo.make_reward` never used the regime system, and `rl_mix.jsonl` does not
carry a `regime` field at all. So GRPO trained on a FLATTENED reward against 4-regime data:

    R1 (791 rows, 31% of the mix)  designed 0.6*F1 + 0.4*grounding  ->  trained on plain F1.
                                   THE GROUNDING TERM WAS DROPPED ENTIRELY.
    R2 (79) and R4 (76)            merged; any of 3 phrases credited, so an R2 item answered
                                   "unanswerable" scored FULL credit when the correct response was
                                   "not supported by context".

The model was never trained to distinguish WHY it declines, and a third of prompts lost the component
that checks the answer is grounded in the context it was given.

SCORED WITH THE INCUMBENT `reward.reward`, not a hand-rolled metric -- it is the function `train.py`
and `score_baseline.py` already use. `regime` is derived from (context present, gold present) because
the field is absent from the data; the derivation is ASSERTED against the known counts above.

    python rl_calibration_eval.py --model <path> [--adapters a,b,c] --tag name
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, ".")
import torch

RL_EVAL = "data/rl/rl_eval_v2.jsonl"
OUT = "data/rl/calibration.json"
REGIME_ORDER = ("R1", "R2", "R3", "R4")
DOMAIN_ORDER = ("Math", "Code", "STEM", "Instruction", "Agentic", "ReadingComp")
# Skills domain topics (extend dynamically from eval data)
SKILLS_TOPICS = (
    "classification", "clustering-dimred", "data-prep", "misc", "model-eval",
    "neural-nets", "optimization", "probability-stats", "regression",
    "reinforcement-learning", "trees-ensembles", "abstain", "domain",
)
# Keep this diagnostic threshold identical to macro_f1_eval.TAU so both incumbent
# evaluators classify answer correctness at the same designed-reward boundary.
ANSWER_CORRECT_THRESHOLD = 0.5

# Expected regime census. Asserted, not trusted: a derivation that silently drifts would reproduce the
# exact flattening this file exists to measure.
EXPECT = {"R1": 37, "R2": 4, "R3": 53, "R4": 18}  # 112-row legacy (rl_eval.jsonl)
EXPECT_V2 = {"R1": 20, "R2": 4, "R3": 20, "R4": 18}  # 62-row balanced (rl_eval_v2.jsonl)

# Semantic abstention, reported as a SECONDARY column. The designed rule is literal phrase matching, so
# a model that says "I don't know" scores 0 under the design while being behaviourally correct. RL is
# TRAINED against the literal detector, so it learns to emit the exact strings while an untrained base
# is scored as confabulating for a correct refusal. The gap between the two columns is the diagnostic:
#   broad rises with literal -> it genuinely learned WHEN to decline
#   only literal rises       -> it learned the reward's VOCABULARY, not the behaviour
ABSTAIN_BROAD = (
    "unanswerable", "not supported by context", "i cannot answer",
    "i don't know", "i do not know", "cannot be determined", "not enough information",
    "insufficient information", "no information", "cannot be answered", "unable to answer",
    "not possible to answer", "does not provide", "isn't specified", "is not specified",
    "not stated", "no answer", "unclear from",
)


def _load(p):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def looks_abstain_broad(text):
    t = (text or "").lower()
    return any(p in t for p in ABSTAIN_BROAD)


def abstain_messages(question, shots, tok=None):
    """Few-shot prompt that PERMITS declining. The default harness forbids it.

    THE CONFOUND THIS FIXES, found 2026-08-03 before the reproduction landed. `cpt_fewshot_lift`
    builds every eval prompt as:

        system: "Answer each question directly and concisely from your own knowledge.
                 Give the specific factual answer only."
        + 4 squad exemplars, ALL answerable, ALL answered, ZERO abstentions

    So the model is instructed never to decline and shown four demonstrations of not declining, and is
    then scored on whether it declines. `broad_abstain_rate = 0.000` in all four arms was therefore
    measuring the PROMPT at least as much as the policy, and no abstention result from that harness
    can separate "no probability mass" from "mass suppressed in context".

    Both styles are reported. The default `cb` style stays byte-identical to every prior number in
    this repo so comparisons remain valid; this style asks the question July's metric was asking.
    **The gap between them is itself the measurement**: a model that abstains under `abstain` but not
    under `cb` HAS the behaviour and is being talked out of it.
    """
    sys_msg = ("You are a knowledgeable assistant. Answer each question directly and concisely from "
               "your own knowledge. Give the specific factual answer only. If the question cannot be "
               "answered -- it asks about the future, private knowledge, or something unknowable -- "
               "reply exactly: unanswerable")
    msgs = [{"role": "system", "content": sys_msg}]
    # Two answerable exemplars, then ONE unanswerable, then one answerable: the abstention exemplar is
    # not last, so the model cannot copy position instead of learning the rule.
    demo = list(shots[:2]) + [("What will be the highest-grossing film of the year 2098?",
                               "unanswerable")] + list(shots[2:3])
    for q, a in demo:
        msgs.append({"role": "user", "content": f"Question: {q}"})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": f"Question: {question}"})
    return msgs


def derive_regime(row):
    """Regime from (context present, gold present). `rl_eval` carries neither a `regime` field nor a
    reliable `answerable` flag, and the reward branches on regime alone."""
    ctx = bool((row.get("context") or "").strip())
    gold = bool((row.get("gold") or "").strip())
    return ("R1" if gold else "R2") if ctx else ("R3" if gold else "R4")


def verify_scorer():
    """Assert `reward.reward` responds in BOTH directions before any arm is trusted.

    A scorer that returns 0.0 for everything would render every arm identical and read as a clean
    null -- indistinguishable from the finding. Checked here, once, loudly.

    Also forces the NLI model to load: R1's 0.4 grounding term silently degrades to F1-only if the
    cross-encoder is unavailable, which would REPRODUCE the exact flattening being measured.
    """
    from rlv2.judge import reward as R
    from rlv2.common.config import ABSTAIN_R2, ABSTAIN_R4
    checks = [
        ("R2 correct phrase", R.reward(f"This is {ABSTAIN_R2}.", "ctx", "R2", ""), 1.0),
        ("R2 given R4 phrase", R.reward(f"This is {ABSTAIN_R4}.", "ctx", "R2", ""), 0.0),
        ("R4 correct phrase", R.reward(f"That is {ABSTAIN_R4}.", "", "R4", ""), 1.0),
        ("R4 given R2 phrase", R.reward(f"That is {ABSTAIN_R2}.", "", "R4", ""), 0.0),
        ("R3 abstain = 0", R.reward(f"{ABSTAIN_R4}", "", "R3", "photosynthesis"), 0.0),
    ]
    for name, got, want in checks:
        assert abs(got - want) < 1e-6, f"scorer broken: {name} -> {got}, expected {want}"
    g = R.nli_entailment("The sky is blue.", "The sky is blue.")
    assert g > 0.5, (f"NLI grounding returned {g:.3f} on a trivial entailment -- the cross-encoder did "
                     f"not load. R1 would silently score on F1 alone, reproducing the flattening this "
                     f"file measures. Refusing to run.")
    print(f"[scorer] verified both directions; NLI live (trivial entailment {g:.3f})", flush=True)


def evaluate(
    model: Any,
    tok: Any,
    rows: list[dict[str, Any]],
    prompt_style: str = "cb",
    eval_file_hint: str = "",
) -> dict[str, Any]:
    """Evaluate one aligned generation per row and persist complete diagnostics.

    Require: every row has prompt, context, gold, and a canonical domain.
    Guarantee: canonical reward and macro-F1 calculations remain unchanged; every item
    has a JSON-safe observable record and belongs to exactly one grid cell.
    Maintain: item order is identical across rows, prompts, generations, items, and
    items_cls. Raise on any alignment or grid-total violation.
    """
    # policy_eval.py lives in sprawl/ after the 1e515fa restructure; src/ modules that
    # import it unqualified die at run time (ModuleNotFoundError). Same defect fixed in
    # src/rlv2/train/sft/train.py -- swept across every src/ importer rather than one at a time.
    import os as _os, sys as _sys
    from rlv2.eval.policy_eval import generate_greedy_batch
    from rlv2.judge import reward as R
    from rlv2.eval import cpt_fewshot_lift as FS
    from rlv2.common.config import ABSTAIN_R2, ABSTAIN_R4

    for index, row in enumerate(rows, start=1):
        missing = [field for field in ("prompt", "context", "gold", "domain") if field not in row]
        if missing:
            raise KeyError(f"eval row {index} missing required fields: {missing}")
        domain = str(row["domain"])
        if domain not in DOMAIN_ORDER and domain not in SKILLS_TOPICS:
            raise ValueError(
                f"eval row {index} has unsupported domain {domain!r}; expected one of {DOMAIN_ORDER + SKILLS_TOPICS}"
            )

    regimes = [derive_regime(r) for r in rows]
    census = collections.Counter(regimes)
    # EXPECT pins the ORIGINAL 112-row heldout (rl_eval.jsonl). EXPECT_V2 pins the 62-row balanced.
    if eval_file_hint == "data/rl/rl_eval.jsonl" and len(rows) == sum(EXPECT.values()):
        assert dict(census) == EXPECT, f"regime census {dict(census)} != expected {EXPECT}"
    elif eval_file_hint == "data/rl/rl_eval_v2.jsonl" and len(rows) == sum(EXPECT_V2.values()):
        assert dict(census) == EXPECT_V2, f"regime census {dict(census)} != expected {EXPECT_V2}"
    else:
        print(f"[census] unpinned eval n={len(rows)}: {dict(census)}", flush=True)

    shots = FS._squad_shots()
    build = (abstain_messages if prompt_style == "abstain"
             else (lambda q, sh: FS._fewshot_messages(q, sh)))
    prompts = [tok.apply_chat_template(build(r["prompt"], shots), tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
               for r in rows]
    print(f"[prompt] style={prompt_style} "
          f"({'PERMITS declining' if prompt_style == 'abstain' else 'FORBIDS declining -- matched to prior numbers'})",
          flush=True)
    model.eval()
    with torch.no_grad():
        gens = generate_greedy_batch(model, tok, prompts, max_new_tokens=128, batch_size=8)

    if not (len(rows) == len(regimes) == len(prompts) == len(gens)):
        raise RuntimeError(
            "eval alignment failed before scoring: "
            f"rows={len(rows)} regimes={len(regimes)} prompts={len(prompts)} gens={len(gens)}"
        )

    per = collections.defaultdict(lambda: collections.defaultdict(list))
    # Cross-regime phrase confusion: rows = true regime, cols = which refusal (if any) was emitted.
    # This is the diagnostic mgpo's flattened reward could not possibly teach -- it credited either.
    confusion = collections.defaultdict(collections.Counter)
    item_records: list[dict[str, Any]] = []

    for index, (g, r, reg, rendered_prompt) in enumerate(
            zip(gens, rows, regimes, prompts), start=1):
        domain = str(r["domain"])

        txt = R._strip_think(g)
        low = txt.lower()
        ctx = str(r["context"] or "")
        gold = str(r["gold"] or "").strip()
        question = str(r["prompt"])
        declined = looks_abstain_broad(txt)

        said_r2 = ABSTAIN_R2 in low
        said_r4 = ABSTAIN_R4 in low and not said_r2   # R2 phrase does not contain the R4 phrase
        confusion[reg]["R2_phrase" if said_r2 else "R4_phrase" if said_r4 else
                       "broad_only" if declined else "answered"] += 1

        rw = R.reward(txt, ctx, reg, gold)
        per[reg]["reward"].append(rw)
        per[reg]["broad_abstain"].append(1.0 if declined else 0.0)
        if reg == "R1":
            # Reported SEPARATELY: grounding is the component mgpo never trained. If it is low while
            # F1 is fine, that is the specific damage, and a blended mean would hide it.
            per[reg]["f1"].append(R._f1(txt, gold))
            per[reg]["grounding"].append(R.nli_entailment(ctx, txt))
        elif reg == "R3":
            per[reg]["f1"].append(R._f1(txt, gold))

        if reg in ("R2", "R4"):
            verdict = "CORRECT_ABSTAIN" if declined else "WRONG_CONFABULATED"
        elif declined:
            verdict = "WRONG_OVER_ABSTAIN"
        elif rw >= ANSWER_CORRECT_THRESHOLD:
            verdict = "CORRECT_ANSWER"
        else:
            verdict = "WRONG_ANSWER"

        record: dict[str, Any] = {
            "index": index,
            "regime": reg,
            "domain": domain,
            "prompt": rendered_prompt,
            "question": question,
            "context": ctx,
            "gold": gold,
            "generation": txt,
            "score": float(rw),
            "score_instrument": "designed_reward",
            "verdict": verdict,
            "should_abstain": reg in ("R2", "R4"),
            "semantic_decline": declined,
            "answer_f1": R._f1(txt, gold) if gold else None,
        }
        item_records.append(record)
        observable = {
            key: record[key] for key in (
                "index", "regime", "domain", "prompt", "question", "context", "gold",
                "generation", "score", "score_instrument", "verdict"
            )
        }
        print(
            f"[eval] {index}/{len(rows)} | "
            f"{json.dumps(observable, ensure_ascii=True, separators=(',', ':'))}",
            flush=True,
        )

    if len(item_records) != len(rows):
        raise RuntimeError(
            f"eval item alignment failed after scoring: items={len(item_records)} rows={len(rows)}"
        )

    # Per-item rewards are REQUIRED, not a nicety: this repo claims no lift without a confidence band
    # excluding zero, and a band needs paired per-item scores. Existing consumers retain `regime` and
    # `reward`; the complete observable payload is additive.
    out = {
        "n": len(rows),
        "census": dict(census),
        "per_regime": {},
        "confusion": {},
        "items": [{**item, "reward": item["score"]} for item in item_records],
    }
    for reg in REGIME_ORDER:
        d = per[reg]
        e = {"n": census[reg], "reward": sum(d["reward"]) / max(1, len(d["reward"])),
             "broad_abstain_rate": sum(d["broad_abstain"]) / max(1, len(d["broad_abstain"]))}
        if d["f1"]:
            e["f1"] = sum(d["f1"]) / len(d["f1"])
        if d["grounding"]:
            e["grounding"] = sum(d["grounding"]) / len(d["grounding"])
        out["per_regime"][reg] = e
        out["confusion"][reg] = dict(confusion[reg])
    # Weighted by regime size = the number GRPO was actually maximising, had it used the design.
    out["designed_reward_mean"] = sum(sum(per[k]["reward"]) for k in per) / max(1, len(rows))

    # JULY-COMPARABLE: preserve the exact incumbent answer/abstain calculations.
    tp = fp = fn = tn_attempted = answered_correctly = answered_wrong = 0
    ans_f1 = []
    for item in item_records:
        declined = bool(item["semantic_decline"])
        gold = str(item["gold"])
        should = not gold
        if should and declined:
            tp += 1                     # correctly abstained
        elif should and not declined:
            fn += 1                     # confabulated on an unanswerable item
        elif not should and declined:
            fp += 1                     # over-abstained: refused something it could answer
            ans_f1.append(0.0)          # a refusal scores zero content, it is not excused
        else:
            tn_attempted += 1
            ans_f1.append(float(item["answer_f1"]))
            if float(item["score"]) >= ANSWER_CORRECT_THRESHOLD:
                answered_correctly += 1
            else:
                answered_wrong += 1

    def _clsf1(p_tp, p_fp, p_fn):
        pr = p_tp / max(1, p_tp + p_fp)
        rc = p_tp / max(1, p_tp + p_fn)
        return (2 * pr * rc / (pr + rc)) if (pr + rc) else 0.0

    f1_abs = _clsf1(tp, fp, fn)
    f1_ans = sum(ans_f1) / max(1, len(ans_f1))
    n_unans, n_ans = tp + fn, fp + tn_attempted
    if answered_correctly + answered_wrong != tn_attempted:
        raise RuntimeError(
            "answer-attempt partition failed: "
            f"correct={answered_correctly} wrong={answered_wrong} attempted={tn_attempted}"
        )

    # Existing bootstrap fields remain present. Additional fields mirror the complete item record so
    # a stored bootstrap sample can be diagnosed without joining against another artifact.
    out["items_cls"] = [
        {
            "regime": item["regime"],
            "domain": item["domain"],
            "should_abstain": bool(not str(item["gold"])),
            "declined": bool(item["semantic_decline"]),
            "ans_f1": item["answer_f1"],
            "index": item["index"],
            "prompt": item["prompt"],
            "question": item["question"],
            "context": item["context"],
            "gold": item["gold"],
            "generation": item["generation"],
            "score": item["score"],
            "score_instrument": item["score_instrument"],
            "verdict": item["verdict"],
        }
        for item in item_records
    ]
    if len(out["items_cls"]) != len(out["items"]):
        raise RuntimeError(
            f"items/items_cls alignment failed: {len(out['items'])} != {len(out['items_cls'])}"
        )
    for item, item_cls in zip(out["items"], out["items_cls"]):
        if (item["index"], item["regime"], item["domain"]) != (
                item_cls["index"], item_cls["regime"], item_cls["domain"]):
            raise RuntimeError(
                f"items/items_cls identity mismatch at item index {item['index']}"
            )

    out["macro_f1"] = {
        "f1_answer": f1_ans,            # token-F1 over answerable items (refusals count as 0)
        "f1_abstain": f1_abs,           # classification F1 on the abstain class
        "macro_f1": (f1_ans + f1_abs) / 2,
        "over_abstain": fp / max(1, n_ans),
        "hallucinated_on_unanswerable": f"{fn}/{n_unans}",
        # Content-correct diagnostics use designed reward >= macro_f1_eval.TAU. The abstain-side
        # counts retain their established decision meaning; attempted answers are split honestly.
        "counts": {"tp_abstained_correctly": tp, "fn_confabulated": fn,
                   "fp_over_abstained": fp, "tn_answered_correctly": answered_correctly,
                   "answered_wrong": answered_wrong},
        "decision_counts": {"tp_abstained_correctly": tp, "fn_confabulated": fn,
                            "fp_over_abstained": fp,
                            "tn_answered_attempted": tn_attempted},
        "answer_correct_threshold": ANSWER_CORRECT_THRESHOLD,
        "answer_correct_score_instrument": "designed_reward",
        "july_reference": {"sft": {"macro_f1": 0.415, "f1_answer": 0.346, "f1_abstain": 0.483,
                                   "halluc": "15/22", "over_abstain": 0.0},
                           "rl": {"macro_f1": 0.505, "f1_answer": 0.344, "f1_abstain": 0.667,
                                  "halluc": "11/22", "over_abstain": 0.0},
                           "commit": "3b10fc8", "delta": +0.091},
    }

    # Retain the incumbent decision cross-hatch separately. Here `tn` means the model attempted an
    # answer; it does NOT claim that answer was content-correct.
    decision_grid: dict[str, dict[str, int]] = {}
    for item in out["items_cls"]:
        cell_key = f"{item['regime']}|{item['domain']}"
        if cell_key not in decision_grid:
            decision_grid[cell_key] = {"n": 0, "tp": 0, "fn": 0, "fp": 0, "tn": 0}
        cell = decision_grid[cell_key]
        cell["n"] += 1
        if item["should_abstain"] and item["declined"]:
            cell["tp"] += 1
        elif item["should_abstain"] and not item["declined"]:
            cell["fn"] += 1
        elif not item["should_abstain"] and item["declined"]:
            cell["fp"] += 1
        else:
            cell["tn"] += 1
    for cell_key, cell in decision_grid.items():
        if cell["tp"] + cell["fn"] + cell["fp"] + cell["tn"] != cell["n"]:
            raise RuntimeError(f"decision grid cell {cell_key} does not partition its support: {cell}")
    if sum(cell["n"] for cell in decision_grid.values()) != len(rows):
        raise RuntimeError(
            "decision grid total mismatch: "
            f"grid={sum(cell['n'] for cell in decision_grid.values())} rows={len(rows)}"
        )
    out["regime_domain_grid"] = decision_grid
    out["regime_domain_grid_labels"] = {
        "tp": "abstained_correctly",
        "fn": "confabulated",
        "fp": "over_abstained",
        "tn": "answered_attempted",
    }

    # Diagnostic correctness grid required by instructions.md: domain rows, regime columns. R1/R3
    # correctness is designed reward >= TAU; R2/R4 correctness is semantic abstention.
    correctness_grid: dict[str, dict[str, dict[str, int]]] = {
        domain: {reg: {"correct": 0, "total": 0} for reg in REGIME_ORDER}
        for domain in DOMAIN_ORDER + SKILLS_TOPICS
    }
    for item in item_records:
        d = str(item["domain"])
        if d not in correctness_grid:
            correctness_grid[d] = {reg: {"correct": 0, "total": 0} for reg in REGIME_ORDER}
        cell = correctness_grid[d][str(item["regime"])]
        cell["total"] += 1
        if str(item["verdict"]).startswith("CORRECT_"):
            cell["correct"] += 1

    total_support = 0
    regime_totals = {reg: {"correct": 0, "total": 0} for reg in REGIME_ORDER}
    all_domains = sorted(correctness_grid.keys())
    for domain in all_domains:
        domain_support = 0
        for reg in REGIME_ORDER:
            cell = correctness_grid[domain][reg]
            if not 0 <= cell["correct"] <= cell["total"]:
                raise RuntimeError(f"invalid correctness grid cell {domain}|{reg}: {cell}")
            domain_support += cell["total"]
            regime_totals[reg]["correct"] += cell["correct"]
            regime_totals[reg]["total"] += cell["total"]
        expected_domain_support = sum(1 for item in item_records if item["domain"] == domain)
        if domain_support != expected_domain_support:
            raise RuntimeError(
                f"correctness grid domain total mismatch for {domain}: "
                f"grid={domain_support} items={expected_domain_support}"
            )
        total_support += domain_support
    if total_support != len(rows):
        raise RuntimeError(
            f"correctness grid total mismatch: grid={total_support} rows={len(rows)}"
        )
    for reg in REGIME_ORDER:
        if regime_totals[reg]["total"] != census[reg]:
            raise RuntimeError(
                f"correctness grid regime total mismatch for {reg}: "
                f"grid={regime_totals[reg]['total']} census={census[reg]}"
            )

    out["domain_regime_correctness_grid"] = correctness_grid
    out["domain_regime_correctness_totals"] = regime_totals
    out["domain_regime_support"] = {
        domain: {reg: correctness_grid[domain][reg]["total"] for reg in REGIME_ORDER}
        for domain in correctness_grid
    }
    out["domain_order"] = list(DOMAIN_ORDER)
    out["regime_order"] = list(REGIME_ORDER)
    # Backward-compatible observed-domain list retained for existing result readers.
    out["domains"] = sorted({str(item["domain"]) for item in item_records})
    return out


def _print_correctness_grid(result: dict[str, Any]) -> None:
    """Print the instructions.md domain-row correctness grid; raise on malformed results."""
    grid = result["domain_regime_correctness_grid"]
    totals = result["domain_regime_correctness_totals"]
    domains = result["domain_order"]
    if not isinstance(grid, dict) or not isinstance(totals, dict) or not isinstance(domains, list):
        raise TypeError("correctness grid result has invalid container types")

    print("\ndomain       | R1       | R2       | R3       | R4       | n (R1/R2/R3/R4)")
    print("-------------|----------|----------|----------|----------|------------------")
    for domain in domains:
        cells = grid[domain]
        rendered = [f"{cells[reg]['correct']}/{cells[reg]['total']}" for reg in REGIME_ORDER]
        support = "/".join(str(cells[reg]["total"]) for reg in REGIME_ORDER)
        print(f"{domain:<13}| {rendered[0]:<9}| {rendered[1]:<9}| {rendered[2]:<9}| "
              f"{rendered[3]:<9}| {support}")
    print("-------------|----------|----------|----------|----------|------------------")
    total_rendered = [f"{totals[reg]['correct']}/{totals[reg]['total']}" for reg in REGIME_ORDER]
    total_support = "/".join(str(totals[reg]["total"]) for reg in REGIME_ORDER)
    print(f"{'TOTAL':<13}| {total_rendered[0]:<9}| {total_rendered[1]:<9}| "
          f"{total_rendered[2]:<9}| {total_rendered[3]:<9}| {total_support}")


def _print_confusion_block(result: dict[str, Any]) -> None:
    """Print complete decision/content diagnostics without relabeling attempts as correct."""
    macro = result["macro_f1"]
    counts = macro["counts"]
    required = (
        "tp_abstained_correctly", "fn_confabulated", "fp_over_abstained",
        "tn_answered_correctly", "answered_wrong",
    )
    missing = [field for field in required if field not in counts]
    if missing:
        raise KeyError(f"confusion block missing required fields: {missing}")
    print(f"tp_abstained_correctly: {counts['tp_abstained_correctly']}")
    print(f"fn_confabulated: {counts['fn_confabulated']}")
    print(f"fp_over_abstained: {counts['fp_over_abstained']}")
    print(f"tn_answered_correctly: {counts['tn_answered_correctly']}")
    print(f"answered_wrong: {counts['answered_wrong']}")
    print(f"f1_answer: {macro['f1_answer']:.3f}")
    print(f"f1_abstain: {macro['f1_abstain']:.3f}")
    print(f"macroF1: {macro['macro_f1']:.3f}")


def load_model(base, adapters):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # Resolve base to absolute path — transformers treats paths with slashes as HF repo IDs
    base = os.path.abspath(base) if os.path.exists(base) else base
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16, trust_remote_code=True)
    if adapters:
        names = [a.strip() for a in adapters.split(",") if a.strip()]
        loaded = []
        for i, a in enumerate(names):
            # Resolve to absolute path — PEFT treats relative paths as HuggingFace repo IDs
            a = os.path.abspath(a)
            if not os.path.isfile(os.path.join(a, "adapter_config.json")):
                raise FileNotFoundError(
                    f"No adapter_config.json at {a!r}. "
                    "Check path (must be absolute or relative to cwd, not a HF repo ID)."
                )
            nm = os.path.basename(a.rstrip("/\\")) or f"a{i}"
            loaded.append(nm)
            m = PeftModel.from_pretrained(m, a, adapter_name=nm) if i == 0 else m
            if i:
                m.load_adapter(a, adapter_name=nm)
        # `set_adapter` on the PeftModel wrapper rejects a list (TypeError: unhashable type). The
        # multi-adapter setter lives on base_model.
        m.base_model.set_adapter(loaded)
        print(f"[model] active adapters: {sorted(m.base_model.active_adapters)}", flush=True)
    return m.cuda(), tok


def main() -> None:
    """Run one scorer invocation and persist/print its complete diagnostic chain.

    Require: model, eval data, and optional adapters pass incumbent gates.
    Guarantee: persisted and printed outputs describe the same aligned item records.
    Maintain: scoring instruments and canonical aggregate calculations are unchanged.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapters", default="")
    ap.add_argument("--tag", required=True)
    # cb = the incumbent harness (forbids declining); abstain = permits it. Both reported:
    # a gap between them means the behaviour EXISTS and the prompt is suppressing it.
    ap.add_argument("--prompt-style", default="cb", choices=["cb", "abstain"])
    ap.add_argument("--eval-file", default=RL_EVAL)
    ap.add_argument(
        "--min-adapter-b",
        type=float,
        default=1e-4,
        help="minimum max absolute LoRA-B value required for a scoreable adapter",
    )
    a = ap.parse_args()
    if a.min_adapter_b <= 0:
        raise ValueError(f"--min-adapter-b must be positive, got {a.min_adapter_b}")

    verify_scorer()
    rows = _load(a.eval_file)
    shafile = a.eval_file.replace(".jsonl", ".sha")
    eval_sha = (io.open(shafile, encoding="utf-8").read().strip()
                if os.path.exists(shafile) else "unstamped")
    print(f"[eval] {a.eval_file}  n={len(rows)}  eval_sha={eval_sha}", flush=True)

    m, tok = load_model(a.model, a.adapters)
    # Gate: refuse to score an untrained adapter (||B|| = 0 means init weights, no learning happened)
    if a.adapters:
        from rlv2.common.cost_meter import assert_adapter_trained
        for adapter_path in [p.strip() for p in a.adapters.split(",") if p.strip()]:
            adapter_path = os.path.abspath(adapter_path)
            assert_adapter_trained(
                adapter_path,
                min_abs_b=a.min_adapter_b,
                label="scorer-gate",
            )
    r = evaluate(m, tok, rows, prompt_style=a.prompt_style, eval_file_hint=a.eval_file)
    r["eval_file"], r["eval_sha"] = a.eval_file, eval_sha

    # Per-arm breakout. sft_heldout and squad_qa measure different things; a single
    # pooled number would hide an arm moving the opposite way from the other.
    # Sliced from the per-item rewards already computed -- NOT a second evaluate() call,
    # which would regenerate every completion for a number we already hold.
    arms = sorted({str(x.get("arm", "all")) for x in rows})
    if len(arms) > 1:
        if len(r["items"]) != len(rows):
            raise RuntimeError(
                f"per-arm slice alignment failed: items={len(r['items'])} rows={len(rows)}"
            )
        r["per_arm"] = {}
        for arm in arms:
            it = [x for x, row in zip(r["items"], rows)
                  if str(row.get("arm", "all")) == arm]
            reg_n = collections.Counter(x["regime"] for x in it)
            r["per_arm"][arm] = {
                "n": len(it),
                "designed_reward_mean": sum(x["reward"] for x in it) / max(1, len(it)),
                "census": dict(reg_n),
            }
            print(f"[arm] {arm:12s} n={len(it):4d} designed_reward "
                  f"{r['per_arm'][arm]['designed_reward_mean']:.4f}  {dict(reg_n)}", flush=True)

    res = json.load(io.open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    r["prompt_style"] = a.prompt_style
    # Provenance: tie every score to a git commit + timestamp for MLflow-like tracking
    import subprocess as _sp
    try:
        _git_sha = _sp.check_output(["git", "rev-parse", "HEAD"], text=True, errors="replace",
                                     cwd=os.path.dirname(os.path.abspath(__file__))).strip()
        _git_dirty = bool(_sp.check_output(["git", "status", "--porcelain"], text=True, errors="replace",
                                            cwd=os.path.dirname(os.path.abspath(__file__))).strip())
    except Exception:
        _git_sha, _git_dirty = "unknown", False
    r["provenance"] = {
        "git_sha": _git_sha,
        "git_dirty": _git_dirty,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": a.model,
        "adapters": a.adapters or None,
        "eval_file": a.eval_file,
        "eval_n": len(rows),
    }
    res[a.tag] = r
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, io.open(OUT, "w", encoding="utf-8"), indent=1)

    print("\n" + "=" * 92)
    print(f"DESIGNED-REWARD CALIBRATION -- {a.tag}   n={r['n']}   "
          f"mean designed reward {r['designed_reward_mean']:.4f}")
    print(f"{'regime':<6}{'n':>4}{'reward':>9}{'F1':>9}{'ground':>9}{'broadAbs':>10}   confusion")
    for reg in REGIME_ORDER:
        e, c = r["per_regime"][reg], r["confusion"][reg]
        f1 = f"{e['f1']:.3f}" if "f1" in e else "-"
        gr = f"{e['grounding']:.3f}" if "grounding" in e else "-"
        print(f"{reg:<6}{e['n']:>4}{e['reward']:>9.3f}{f1:>9}{gr:>9}"
              f"{e['broad_abstain_rate']:>10.3f}   {c}")
    print("\n  R1 grounding is the component mgpo NEVER trained (it optimised plain F1).")
    print("  R2 n=4 -- report the count, never a bare rate, and claim no significance on it.")
    print("  confusion: R2 rows emitting R4_phrase (or vice versa) = the model cannot tell the two")
    print("  refusals apart, which the flattened training reward could not have taught it.")
    print("=" * 92, flush=True)

    _print_correctness_grid(r)
    _print_confusion_block(r)

    # Preserve the incumbent decision cross-hatch as a separate diagnostic. Its `tn` is explicitly
    # an attempted answer, not a content-correct answer; content correctness is in the grid above.
    decision_grid = r["regime_domain_grid"]
    labels = r["regime_domain_grid_labels"]
    print("\nDECISION CROSS-HATCH: regime x domain "
          "(tp=abstained_correctly, fn=confabulated, fp=over_abstained, "
          "tn=answered/attempted)")
    header = f"{'':>6}" + "".join(f"{domain:>14}" for domain in DOMAIN_ORDER)
    print(header)
    print("  " + "-" * (6 + 14 * len(DOMAIN_ORDER)))
    for reg in REGIME_ORDER:
        row_parts = [f"{reg:<6}"]
        for domain in DOMAIN_ORDER:
            cell = decision_grid.get(f"{reg}|{domain}")
            if cell is None:
                row_parts.append(f"{'—':>14}")
            elif reg in ("R2", "R4"):
                row_parts.append(f"{cell['tp']}tp/{cell['fn']}fn".rjust(14))
            else:
                row_parts.append(f"{cell['tn']}tn/{cell['fp']}fp".rjust(14))
        print("".join(row_parts))
    print(f"labels: {json.dumps(labels, sort_keys=True)}")

    print("\nFULL RESULT JSON:")
    print(json.dumps(r, indent=1), flush=True)


if __name__ == "__main__":
    main()
