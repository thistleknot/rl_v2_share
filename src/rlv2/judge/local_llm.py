"""src/rlv2/judge/local_llm.py — ollama-backed drop-in for `nvidia_llm.complete`, same signature.

WHY THIS EXISTS. `entigraph_skills.py` calls `NV.complete(system, user, max_tokens, temperature)`
against NVIDIA's hosted endpoint. That endpoint is QUOTA-EXHAUSTED (HTTP 429 on every request, at
every spacing up to 10 s gaps -- so it is not a rate limit that backoff can ride out). This module
exposes the identical call so the generation scripts can run locally by swapping one import.

THE CALL PATTERN IS COPIED FROM `build_skills_sft.py`, NOT INVENTED. Two details are load-bearing and
both were learned by getting them wrong:

    /api/chat  (not /api/generate)   -- the messages form these models are tuned for
    "think": False                   -- WITHOUT THIS the reasoning models return an EMPTY content
                                        field and every call silently yields 0 characters. A first
                                        throughput measurement here reported 10.7 s/call of nothing.

MEASURED THROUGHPUT (qwen3.5-oc:4b, this box, 2026-08-02):

    concurrency 1 :  9.9 calls/min
    concurrency 4 : 41.7 calls/min   <- knee; 8 gives 38.4, no better
    concurrency 8 : 38.4 calls/min

So ~1,200 generations costs ~30 min at concurrency 4.

CONTENTION WARNING. ollama runs on the SAME GPU as training. Generation and training must be
SEQUENTIAL, not concurrent -- a 4b model at concurrency 4 holds several GB, and the training stage
derives its VRAM ceiling from what is FREE at launch, so an overlapping generation run silently
starves it.
"""
import json
import os
import time
import urllib.error
import urllib.request

OLLAMA = os.environ.get("OLLAMA_CHAT", "http://127.0.0.1:11434/api/chat")
MODEL = os.environ.get("ENTIGRAPH_GEN_MODEL", os.environ.get("QA_GEN_MODEL", "qwen3.5-oc:4b"))
NUM_CTX = int(os.environ.get("GEN_NUM_CTX", "8192"))
RETRIES = 3


class QuotaExhausted(RuntimeError):
    """Signature parity with nvidia_llm so callers' except-clauses keep working."""


def complete(system, user, max_tokens=1200, temperature=0.8, thinking=False, model=None):
    """Return the assistant message content, or raise after RETRIES attempts.

    Fails LOUD rather than returning "" -- a silent empty string would be written to the corpus as a
    zero-length passage and pollute it invisibly, which is exactly what `think: True` did.
    """
    body = json.dumps({
        "model": model or MODEL, "stream": False, "think": bool(thinking),
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "options": {"temperature": temperature, "num_ctx": NUM_CTX,
                    "num_predict": max_tokens},
    }).encode()
    last = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
            txt = json.loads(urllib.request.urlopen(req, timeout=300).read())["message"]["content"]
            if txt and txt.strip():
                return txt
            last = "empty content (is `think` leaking? this model needs think=False)"
        except Exception as e:                      # noqa: BLE001 - retried, then re-raised
            last = f"{type(e).__name__}: {str(e)[:120]}"
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"ollama complete() failed after {RETRIES} attempts: {last}")


def stats():
    return {"backend": "ollama", "model": MODEL, "endpoint": OLLAMA}
