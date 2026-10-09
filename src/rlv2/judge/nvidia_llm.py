"""src/rlv2/judge/nvidia_llm.py — shared inference client for NVIDIA's hosted endpoint.

WHY THIS EXISTS. Everything in this repo that needs INFERENCE ONLY -- question generation, EntiGraph
relation synthesis, LLM-as-judge -- was running against local ollama, which shares the single 16 GB
GPU with training. That forced the whole pipeline to serialise: generate, THEN train, never both.
Moving inference off-box removes that constraint entirely, so generation now overlaps training.

It is also a better model: deepseek-v4-flash against qwen3.5-oc:4b.

WHAT DOES NOT MOVE. Training, evaluation of OUR checkpoints, and anything measuring OUR model stays
local. This client is for third-party inference only -- a remote model cannot tell us what our
weights learned.

RATE LIMITING IS THE BINDING CONSTRAINT, AND IT IS MEASURED, NOT GUESSED. Concurrency 16 returned
HTTP 429 on 16 of 16 requests; concurrency 8 still lost 3 of 16. A fixed concurrency chosen by hope
would silently drop a fraction of every corpus. So:

  - a module-wide semaphore caps in-flight requests
  - 429 retries with exponential backoff AND jitter (un-jittered backoff re-synchronises the very
    threads that collided, so they collide again on the retry)
  - sustained 429s SHRINK the effective concurrency, and quiet periods let it recover -- the limit
    is discovered at runtime rather than hard-coded, because it is a property of the account and
    the hour, not of this code
  - a failure returns None. Callers RECORD the miss; they never silently substitute a shorter
    corpus, which is how a partial dataset gets mistaken for a complete one.
"""
import os
import random
import threading
import time

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), ".env"))

BASE_URL = "https://integrate.api.nvidia.com/v1"
# The unsuffixed "deepseek-ai/deepseek-v4-flash" was retired from the endpoint; every call
# against it 404s, and complete() turns that into a silent None after 6 retries (~250s each).
# Verified against GET /v1/models on 2026-08-23: only the -0731 build is served.
MODEL = os.environ.get("NVIDIA_MODEL", "deepseek-ai/deepseek-v4-flash-0731")
API_KEY = os.environ.get("NVIDIA_APIKEY") or os.environ.get("NVIDIA_API_KEY")

MAX_CONCURRENCY = 8          # ceiling; the adaptive gate only ever lowers from here
MIN_CONCURRENCY = 2
MAX_RETRIES = 6
BASE_BACKOFF = 4.0           # was 2.0 -- too eager against a throttling endpoint

# GLOBAL COOLDOWN, not per-request. Per-request exponential backoff is necessary and NOT sufficient:
# with 8 workers, one 429 puts ONE worker to sleep while the other seven keep hammering, which is
# how a rate limit becomes a self-inflicted outage. On 2026-08-01 an orphaned 8-way run ground
# against the endpoint for 2.5 h and left every later call 429-ing with zero successes.
# A 429 seen by ANY worker now pauses ALL of them.
COOLDOWN_BASE = 15.0         # seconds, doubling per consecutive 429 burst
COOLDOWN_MAX = 300.0
# CIRCUIT BREAKER: sustained failure with zero successes means the quota is gone, not that the next
# retry will land. Grinding is what burns the quota further and hides the real state.
BREAKER_CONSECUTIVE_FAILS = 12

_client = None
_client_lock = threading.Lock()
_gate = threading.Semaphore(MAX_CONCURRENCY)
_state = {"permits": MAX_CONCURRENCY, "429s": 0, "ok": 0, "fail": 0, "last_shrink": 0.0,
          "cooldown_until": 0.0, "cooldown_len": COOLDOWN_BASE, "consec_fail": 0, "open": False}
_state_lock = threading.Lock()


def client():
    global _client
    with _client_lock:
        if _client is None:
            if not API_KEY:
                raise RuntimeError("NVIDIA_APIKEY not found in .env")
            from openai import OpenAI
            # max_retries=0: retries are handled here, where backoff can also adjust concurrency.
            _client = OpenAI(base_url=BASE_URL, api_key=API_KEY, max_retries=0, timeout=180.0)
        return _client


def _shrink():
    """Take a permit out of circulation. Rate-limited to one shrink per 10 s so a burst of 429s
    from one collision does not collapse concurrency to the floor in a single instant."""
    with _state_lock:
        now = time.time()
        if _state["permits"] <= MIN_CONCURRENCY or now - _state["last_shrink"] < 10:
            return
        if _gate.acquire(blocking=False):
            _state["permits"] -= 1
            _state["last_shrink"] = now


def _grow():
    with _state_lock:
        if _state["permits"] < MAX_CONCURRENCY and _state["ok"] % 40 == 0 and _state["ok"]:
            _gate.release()
            _state["permits"] += 1


class QuotaExhausted(RuntimeError):
    """Raised once the breaker opens. A caller that catches this must STOP, not retry."""


def _enter_cooldown():
    """A 429 seen by ANY worker pauses ALL of them, with a doubling window.

    This is the half that per-request backoff cannot provide: with 8 workers, backing off the one
    that got the 429 leaves seven hammering, and the endpoint never sees the load drop.
    """
    with _state_lock:
        now = time.time()
        if now < _state["cooldown_until"]:
            return _state["cooldown_until"]          # already cooling; do not extend on every hit
        _state["cooldown_len"] = min(_state["cooldown_len"] * 2, COOLDOWN_MAX)
        _state["cooldown_until"] = now + _state["cooldown_len"]
        return _state["cooldown_until"]


def _await_cooldown():
    while True:
        with _state_lock:
            if _state["open"]:
                raise QuotaExhausted(
                    f"circuit breaker OPEN after {_state['consec_fail']} consecutive failures "
                    f"({_state['429s']} 429s, {_state['ok']} ok). The quota is gone; retrying "
                    f"burns it further and hides the real state.")
            until = _state["cooldown_until"]
        wait = until - time.time()
        if wait <= 0:
            return
        time.sleep(min(wait, 5.0) * (0.75 + 0.5 * random.random()))


def complete(system, user, max_tokens=1200, temperature=0.8, thinking=False, model=None):
    """One chat completion. Returns text, or None after MAX_RETRIES (never a partial substitute).

    Raises QuotaExhausted once the breaker opens, so a batch ABORTS instead of grinding.
    """
    from openai import APIError, APITimeoutError, RateLimitError
    body = {"chat_template_kwargs": {"thinking": bool(thinking)}}
    if thinking:
        body["chat_template_kwargs"]["reasoning_effort"] = "high"
    for attempt in range(MAX_RETRIES):
        _await_cooldown()
        with _gate:
            try:
                r = client().chat.completions.create(
                    model=model or MODEL,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=temperature, top_p=0.95, max_tokens=max_tokens,
                    extra_body=body, stream=False)
                with _state_lock:
                    _state["ok"] += 1
                    _state["consec_fail"] = 0
                    _state["cooldown_len"] = COOLDOWN_BASE      # a success resets the window
                _grow()
                return r.choices[0].message.content or ""
            except RateLimitError as e:
                with _state_lock:
                    _state["429s"] += 1
                # Honour Retry-After when the server sends one -- it knows the real window.
                ra = None
                try:
                    ra = float((getattr(e, "response", None).headers or {}).get("retry-after"))
                except Exception:
                    pass
                _shrink()
                until = _enter_cooldown()
                if ra:
                    with _state_lock:
                        _state["cooldown_until"] = max(until, time.time() + ra)
            except (APITimeoutError, APIError, Exception):
                pass
        # Jitter matters: identical backoff re-synchronises the threads that just collided.
        time.sleep(BASE_BACKOFF * (2 ** attempt) * (0.5 + random.random()))
    with _state_lock:
        _state["fail"] += 1
        _state["consec_fail"] += 1
        if _state["consec_fail"] >= BREAKER_CONSECUTIVE_FAILS:
            _state["open"] = True
            print(f"[nvidia_llm] CIRCUIT BREAKER OPEN after {_state['consec_fail']} consecutive "
                  f"failures -- aborting rather than grinding the quota", flush=True)
    return None


def stats():
    with _state_lock:
        return dict(_state)
