"""src/rlv2/common/model_config.py — SINGLE SOURCE OF TRUTH for models + the output-cleanliness CONTRACT (:ModelOutputContract:).

WHY: reasoning models (qwen3.5-reasoning-thinking-*) bleed chain-of-thought into their OUTPUT — as `<think>`
tags OR plain "Thinking Process:" prose — which malforms anything that needs clean text/JSON: DSPy demos,
TextGrad's TGD tag-parser, regex-scraped judge scores, ragas-schema rubrics. Falsified paths: DSPy/TextGrad
on the reasoning 4B/9B produced malformed rubrics; the QA-gen judge's #1 false-negative was a regex-scrape of
un-schema'd JSON.

RESOLUTION, by output type (this is the contract every layer must obey):
  - clean JSON  -> reasoning OK, but MUST use think:False + /no_think prefix + format=schema + strip
  - clean TEXT  -> use the NON-reasoning GEN_MODEL; a RAW reasoning model for free text (no schema) is BANNED
  - all rubric PRODUCERS emit the SAME ragas schema the live judge CONSUMES (never drop final_score / bands)

RE-LITIGATION rule: change the live JUDGE_MODEL or the rubric schema -> ripples to EVERYTHING (re-run the
calibration set, the QA-gen judge, and ALL optimizer tests). Change only ONE optimizer's model -> re-run only
that optimizer's test. (Proven contained: swapping the optimizer proposer to GEN_MODEL required re-running only
the rubric bake-off; QA-gen + judge layers were already contract-correct and were left untouched.)

VALIDATED on-contract already (do NOT re-litigate): QA-gen DSPy = GEN_MODEL, 19/20 (qa_dspy_review.md).
"""
import json
import re
import urllib.request

# ── canonical models ──
GEN_MODEL = "qwen3.5:4b"                            # NON-reasoning: clean free-text (QA-gen, rubric rewrite/propose)
JUDGE_MODEL = "qwen3.5-reasoning-thinking-256k:4b"  # THE ONLY JUDGE (2026-09-12).
# The 2B is deleted from this project. It was REJECTED for R3 (3/12 fluent-wrong
                               # decoys accepted); the R3 lane resolves its judge through
                               # reward.set_judge_lane("r3") -> judge.py 4b. See S-12.
JUDGE_HEAVY = "qwen3.5-reasoning-thinking-256k:9b"  # reasoning: rubric AUTHORING / heavy-judge ONLY (+ think-kill + schema)
ENDPOINT = "http://localhost:11434/api/generate"
_REASONING = ("reasoning-thinking", "reasoning-oc")   # substrings that mark a CoT-bleeding reasoning model


def is_reasoning(model: str) -> bool:
    return any(t in (model or "") for t in _REASONING)


def strip_think(text: str) -> str:
    """Remove <think> tags AND their content (residual reasoning bleed)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return re.sub(r"</?think>", "", text).strip()


def assert_clean_text_model(model: str) -> None:
    """ENFORCEMENT: a raw reasoning model for FREE TEXT (no schema) bleeds CoT — the contract forbids it."""
    if is_reasoning(model):
        raise ValueError(
            f":ModelOutputContract: violation — reasoning model {model!r} used for FREE TEXT. It bleeds "
            f"chain-of-thought (<think> or 'Thinking Process:' prose) and malforms the output. Use "
            f"GEN_MODEL={GEN_MODEL!r} for clean text, or route JSON through clean_generate(..., schema=...).")


def clean_generate(prompt, model=GEN_MODEL, schema=None, system=None,
                   max_tokens=2048, temperature=0.0, timeout=900):
    """Contract-enforced ollama generation. Free text (schema=None) with a reasoning model RAISES. Always applies
    BOTH think-kill layers (think:False + /no_think) + strip; structured output when a schema is given."""
    if schema is None:
        assert_clean_text_model(model)                 # reasoning free-text is banned; JSON must pass a schema
    full = (f"{system}\n\n" if system else "") + "/no_think\n" + prompt
    payload = {"model": model, "prompt": full, "stream": False, "think": False,
               "options": {"temperature": temperature, "num_ctx": 12288, "num_predict": max_tokens}}
    if schema:
        payload["format"] = schema
    req = urllib.request.Request(ENDPOINT, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = json.loads(resp.read()).get("response", "") or ""
    out = strip_think(text)
    if schema is None:
        return out
    m = re.search(r"\{.*\}", out, re.S)                 # schema mode: return the parsed object
    return json.loads(m.group(0)) if m else None


if __name__ == "__main__":
    # contract self-check
    print("GEN_MODEL:", GEN_MODEL, "| is_reasoning:", is_reasoning(GEN_MODEL))
    print("JUDGE_HEAVY:", JUDGE_HEAVY, "| is_reasoning:", is_reasoning(JUDGE_HEAVY))
    try:
        assert_clean_text_model(JUDGE_HEAVY)
    except ValueError as e:
        print("ENFORCED:", str(e)[:80], "...")
