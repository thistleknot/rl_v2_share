"""src/rlv2/data/rl_mix.py — assemble the MGPO training mix: staged RL corpus + data-science skills + abstention.

WHY A MIX AND NOT JUST MATH. Math is the only domain with a programmatic verifier
(`reward.verify_math` on the GSM8K `#### N` tail), so a Math-only run is the cleanest RL signal and
also the narrowest. Three other sources are folded in deliberately:

  data-science skills   the actual objective of this repo. Training RL only on GSM8K would sharpen
                        reasoning the model will never be asked for. These prompts come from the SFT
                        TRAIN split, so RL refines a distribution the policy has already seen, and
                        the held-out SKILLS (skill-disjoint, never trained) stay clean for readout.
  unanswerable          225 items with EMPTY gold are already staged across all six domains. With
                        `reward.ABSTENTION_GATE` on, an ANSWERABLE item the policy refuses is forced
                        to 0.0. Without these, "I cannot answer" is a free way to dodge every hard
                        prompt -- the exploit the gate exists to close.
  Code / STEM / etc.    non-verifiable, scored by quality. Keeps the policy from collapsing onto the
                        one format that has a hard verifier.

THE MGPO ADAPTATION, STATED PLAINLY. MGPO (arXiv:2511.06221) prioritises prompts by the ENTROPY of
the model's pass rate -- maximal at p = 0.5. That definition assumes a BINARY verifiable reward,
which only Math has here. The generalisation used downstream is to weight by the WITHIN-GROUP REWARD
STANDARD DEVIATION:

    binary reward  ->  std = sqrt(p(1-p)),  maximised at p = 0.5   [identical argmax to entropy]
    continuous     ->  std is still exactly what GRPO's advantage consumes

So reward-std weighting IS MaxEnt on the verifiable slice and extends to the rest without inventing
a second mechanism. It also targets the measured defect directly: the prior GRPO run logged
`frac_reward_zero_std` of 0.2-0.3, meaning a fifth to a third of every batch produced ZERO advantage
and therefore ZERO gradient. Those groups are pure waste, and they are what this selects against.

This is the same principle as T5 on the SFT side (+0.188, p=0.029, from dropping rows the model
already answers): an item the policy always gets right teaches nothing, in SFT or in RL.
"""
import argparse
import io
import json
import os
import random
from collections import Counter

MANIFEST = "data/manifest.json"
RAW_ALL = "data/raw/all.jsonl"
SKILLS_TRAIN = "data/skills/skills_train.jsonl"
OUT_DIR = "data/rl"
OUT_MIX = f"{OUT_DIR}/rl_mix.jsonl"
OUT_EVAL = f"{OUT_DIR}/rl_eval.jsonl"

SEED = 7
SKILLS_N = 500          # ~20% of the mix; enough to steer, not enough to drown the staged corpus
MAX_UNANSWERABLE = 225  # everything staged


def _load_jsonl(p):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills-n", type=int, default=SKILLS_N)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(SEED)

    man = json.load(io.open(MANIFEST, encoding="utf-8"))
    raw = _load_jsonl(RAW_ALL)
    by_q = {r["question"]: r for r in raw}

    def take(split):
        qs = man["splits"][split]["questions"]
        return [by_q[q] for q in qs if q in by_q]

    train_src, eval_src = take("grpo_train"), take("eval_holdout")
    mix = []
    for r in train_src:
        gold = (r.get("answer") or "").strip()
        mix.append({"prompt": r["question"], "gold": gold, "domain": r.get("domain", ""),
                    "source": r.get("source", ""), "context": r.get("context", ""),
                    "answerable": bool(gold),
                    # Math is the only domain with a programmatic verifier; everything else is
                    # scored by quality and contributes reward VARIANCE rather than a hard gate.
                    "verifiable": r.get("domain") == "Math" and bool(gold)})

    # --- data-science skills, from the SFT TRAIN split (never the held-out skills) ---
    n_sk = 0
    if os.path.exists(SKILLS_TRAIN):
        sk = _load_jsonl(SKILLS_TRAIN)
        rng.shuffle(sk)
        # One QA per SKILL, not per pair: five paraphrases of one fact are five near-identical
        # prompts, which would concentrate RL budget on a handful of facts and inflate their weight.
        seen, picked = set(), []
        for r in sk:
            if r["skill"] in seen:
                continue
            seen.add(r["skill"])
            picked.append(r)
            if len(picked) >= a.skills_n:
                break
        for r in picked:
            mix.append({"prompt": r["prompt"], "gold": r["answer"], "domain": "DataScience",
                        "source": f"skills:{r['topic']}", "context": "", "answerable": True,
                        "verifiable": False, "skill": r["skill"]})
        n_sk = len(picked)
    else:
        print(f"WARNING: {SKILLS_TRAIN} missing -- run src/rlv2/data/skills_sft.py first. "
              f"Mix will contain NO data-science items.", flush=True)

    rng.shuffle(mix)
    with io.open(OUT_MIX, "w", encoding="utf-8") as fh:
        for r in mix:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    ev = [{"prompt": r["question"], "gold": (r.get("answer") or "").strip(),
           "domain": r.get("domain", ""), "source": r.get("source", ""),
           "context": r.get("context", ""), "answerable": bool((r.get("answer") or "").strip()),
           "verifiable": r.get("domain") == "Math" and bool((r.get("answer") or "").strip())}
          for r in eval_src]
    with io.open(OUT_EVAL, "w", encoding="utf-8") as fh:
        for r in ev:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    d = Counter(r["domain"] for r in mix)
    print(f"RL MIX -> {OUT_MIX}   {len(mix)} prompts", flush=True)
    for k, v in d.most_common():
        vv = sum(1 for r in mix if r["domain"] == k and r["verifiable"])
        un = sum(1 for r in mix if r["domain"] == k and not r["answerable"])
        print(f"  {k:<14}{v:>5}   verifiable {vv:>4}   unanswerable {un:>4}", flush=True)
    print(f"\n  data-science skills folded in: {n_sk}", flush=True)
    print(f"  unanswerable total: {sum(1 for r in mix if not r['answerable'])} "
          f"(ABSTENTION_GATE must be ON or these are free points)", flush=True)
    print(f"\nRL EVAL -> {OUT_EVAL}   {len(ev)} prompts "
          f"({sum(1 for r in ev if r['verifiable'])} verifiable)", flush=True)
    print("NOTE: held-out SKILLS are scored separately from data/skills/skills_eval.jsonl -- "
          "they are skill-disjoint from both SFT and this RL mix.", flush=True)


if __name__ == "__main__":
    main()
