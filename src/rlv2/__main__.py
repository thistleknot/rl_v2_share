"""rlv2/__main__.py -- the one entrypoint: python -m rlv2 <stage> [stage args].

Notes: README.md "Start here" lists every stage and its inputs and outputs; docs/PROVENANCE.md maps slides to code.
No governing spec in this repo; basis is the author-approved restructure (2026-10).

The harness is data -> CPT (optional) -> SFT -> RL, with eval and HPO beside it. Each stage is a
module with its own argparse; this file only picks the module and hands over argv, so every
stage keeps the exact flags it had when it ran as a script.

    python -m rlv2 data entigraph        # synthetic data: EntiGraph over the skills corpus
    python -m rlv2 cpt --full-param ...  # continued pre-training
    python -m rlv2 sft --full-param ...
    python -m rlv2 rl  --base ...
    python -m rlv2 eval --model ...
    python -m rlv2 hpo sft | rl
"""
import runpy
import sys

STAGES = {
    "data": {
        "pull": "rlv2.data.pull_datasets",          # RedPajama/SlimPajama + GSM8K pull, allocations
        "chunk": "rlv2.data.chunking",              # skill notes -> packed records
        "entigraph": "rlv2.data.entigraph",         # EntiGraph over the skills corpus
        "skills-sft": "rlv2.data.skills_sft",       # per-skill SFT QA
        "qa": "rlv2.data.synthetic_qa",             # RAG + regime-relabelled QA
        "pools": "rlv2.data.pools",                 # gated three-pool generation
        "cpt-mix": "rlv2.data.cpt_mix",             # CPT corpus: chunks + SFT + EntiGraph + replay
        "rl-mix": "rlv2.data.rl_mix",               # RL training mix
    },
    "cpt": "rlv2.train.cpt.train",
    "sft": "rlv2.train.sft.train",
    "rl": "rlv2.train.rl.grpo",
    "eval": "rlv2.eval.rl_calibration_eval",
    "hpo": {"sft": "rlv2.train.hpo.sft", "rl": "rlv2.train.hpo.rl"},
}


def _usage():
    lines = ["usage: python -m rlv2 <stage> [args]", "stages:"]
    for name, target in STAGES.items():
        lines.append(f"  {name:6s}" + (" " + " | ".join(target) if isinstance(target, dict) else ""))
    return "\n".join(lines)


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return 0
    target = STAGES.get(argv[0])
    rest = argv[1:]
    if isinstance(target, dict):
        if not rest or rest[0] not in target:
            print(f"python -m rlv2 {argv[0]} needs one of: {', '.join(target)}", file=sys.stderr)
            return 2
        target, rest = target[rest[0]], rest[1:]
    if target is None:
        print(_usage(), file=sys.stderr)
        return 2
    sys.argv = [target] + rest
    runpy.run_module(target, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
