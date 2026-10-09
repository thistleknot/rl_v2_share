"""
Behavioural tests for nvidia_llm.py -- the shared remote-inference client.

The failure this file exists to prevent: on 2026-08-23 the endpoint stopped serving the
unsuffixed "deepseek-ai/deepseek-v4-flash" model id. Every call 404'd, complete() dutifully
returned None after 6 retries, and the *caller* coerced None to "" -- so a three-hour pool
generation run produced an empty corpus that was mistaken for early saturation. Two contracts
have to hold for that to be impossible:

  1. the default MODEL is a model the endpoint actually serves;
  2. total failure is reported as None, DISTINCT from a successful empty completion (""), and
     sustained failure opens the breaker instead of grinding.

Every test here runs offline: the OpenAI client is replaced with a fake, and sleeps are
neutralised (complete() would otherwise back off for ~4 minutes per call).
"""
from __future__ import annotations
import os as _o, sys as _s

import sys
import time as _time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rlv2.judge import nvidia_llm as nv  # noqa: E402

RETIRED_MODEL = "deepseek-ai/deepseek-v4-flash"


# --------------------------------------------------------------------------- helpers

class FakeCompletions:
    """Stands in for client().chat.completions. `script` is a list of either exceptions to
    raise or strings to return as message content; the last entry repeats forever."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if isinstance(item, BaseException):
            raise item
        msg = type("M", (), {"content": item})()
        choice = type("C", (), {"message": msg})()
        return type("R", (), {"choices": [choice]})()


def make_client(script):
    fc = FakeCompletions(script)
    chat = type("Chat", (), {"completions": fc})()
    return type("Client", (), {"chat": chat})(), fc


def rate_limit_error(retry_after=None):
    """A genuine openai.RateLimitError -- complete() catches the real class by name."""
    from openai import RateLimitError
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    resp = httpx.Response(429, headers=headers,
                          request=httpx.Request("POST", "http://test/v1/chat/completions"))
    return RateLimitError("rate limited", response=resp, body=None)


class VirtualClock:
    """Sleeps must ADVANCE time, not vanish.

    A no-op sleep looks harmless but turns _await_cooldown() -- `while wait > 0: sleep()` --
    into an infinite spin, because the deadline it is waiting for never arrives. Advancing a
    fake clock instead keeps the real waiting logic under test and makes it run instantly.
    """

    def __init__(self, start):
        self.t = float(start)

    def time(self):
        return self.t

    def sleep(self, seconds):
        self.t += max(0.0, float(seconds))


@pytest.fixture
def clean(monkeypatch):
    """Reset module state, virtualise the clock, and make backoff jitter deterministic."""
    clock = VirtualClock(_time.time())
    monkeypatch.setattr(_time, "time", clock.time)
    monkeypatch.setattr(_time, "sleep", clock.sleep)
    monkeypatch.setattr(nv.random, "random", lambda: 0.5)
    monkeypatch.setattr(nv, "_gate", nv.threading.Semaphore(nv.MAX_CONCURRENCY))
    monkeypatch.setitem(nv._state, "permits", nv.MAX_CONCURRENCY)
    for k, v in (("429s", 0), ("ok", 0), ("fail", 0), ("last_shrink", 0.0),
                 ("cooldown_until", 0.0), ("cooldown_len", nv.COOLDOWN_BASE),
                 ("consec_fail", 0), ("open", False)):
        monkeypatch.setitem(nv._state, k, v)
    yield clock


def install(monkeypatch, script):
    c, fc = make_client(script)
    monkeypatch.setattr(nv, "client", lambda: c)
    return fc


# --------------------------------------------------------------------------- the regression

def test_default_model_is_the_served_0731_build():
    """The unsuffixed id was retired; defaulting to it 404s every call in the whole repo."""
    assert nv.MODEL == "deepseek-ai/deepseek-v4-flash-0731"
    assert nv.MODEL != RETIRED_MODEL


def test_default_model_is_what_gets_sent_on_the_wire(clean, monkeypatch):
    fc = install(monkeypatch, ["ok"])
    nv.complete("sys", "usr")
    assert fc.calls[0]["model"] == "deepseek-ai/deepseek-v4-flash-0731"


def test_explicit_model_argument_overrides_default(clean, monkeypatch):
    fc = install(monkeypatch, ["ok"])
    nv.complete("sys", "usr", model="meta/llama-3.1-8b-instruct")
    assert fc.calls[0]["model"] == "meta/llama-3.1-8b-instruct"


# --------------------------------------------------------------------------- failure is loud

def test_total_failure_returns_none_after_max_retries(clean, monkeypatch):
    fc = install(monkeypatch, [RuntimeError("404 model not found")])
    assert nv.complete("sys", "usr") is None
    assert len(fc.calls) == nv.MAX_RETRIES == 6
    assert nv.stats()["fail"] == 1
    assert nv.stats()["consec_fail"] == 1


def test_successful_empty_completion_is_not_none(clean, monkeypatch):
    """"" means 'the model answered with nothing'; None means 'we never got an answer'.
    Collapsing the two is exactly how the empty-corpus run went unnoticed."""
    install(monkeypatch, [""])
    out = nv.complete("sys", "usr")
    assert out == ""
    assert out is not None
    assert nv.stats()["fail"] == 0
    assert nv.stats()["ok"] == 1


def test_success_returns_content_and_resets_failure_state(clean, monkeypatch):
    nv._state["consec_fail"] = 5
    nv._state["cooldown_len"] = 120.0
    install(monkeypatch, ["the answer"])
    assert nv.complete("sys", "usr") == "the answer"
    assert nv.stats()["consec_fail"] == 0
    assert nv.stats()["cooldown_len"] == nv.COOLDOWN_BASE
    assert nv.stats()["ok"] == 1


def test_transient_failure_is_retried_then_succeeds(clean, monkeypatch):
    fc = install(monkeypatch, [RuntimeError("boom"), "recovered"])
    assert nv.complete("sys", "usr") == "recovered"
    assert len(fc.calls) == 2
    assert nv.stats()["fail"] == 0


# --------------------------------------------------------------------------- breaker

def test_breaker_opens_on_the_nth_consecutive_failure(clean, monkeypatch):
    install(monkeypatch, [RuntimeError("dead")])
    nv._state["consec_fail"] = nv.BREAKER_CONSECUTIVE_FAILS - 1
    assert nv.complete("sys", "usr") is None
    assert nv._state["consec_fail"] == nv.BREAKER_CONSECUTIVE_FAILS == 12
    assert nv._state["open"] is True


def test_breaker_stays_closed_one_failure_short(clean, monkeypatch):
    install(monkeypatch, [RuntimeError("dead")])
    nv._state["consec_fail"] = nv.BREAKER_CONSECUTIVE_FAILS - 2
    assert nv.complete("sys", "usr") is None
    assert nv._state["open"] is False


def test_open_breaker_aborts_immediately_without_calling_the_endpoint(clean, monkeypatch):
    fc = install(monkeypatch, ["never reached"])
    nv._state["open"] = True
    with pytest.raises(nv.QuotaExhausted):
        nv.complete("sys", "usr")
    assert fc.calls == []


# --------------------------------------------------------------------------- rate limiting

def test_rate_limit_counts_429_and_enters_global_cooldown(clean, monkeypatch):
    clock = clean
    install(monkeypatch, [rate_limit_error()])
    started = clock.time()
    assert nv.complete("sys", "usr") is None
    st = nv.stats()
    assert st["429s"] == nv.MAX_RETRIES
    assert st["cooldown_until"] > started


def test_retry_after_header_extends_cooldown_beyond_the_doubling_window(clean, monkeypatch):
    clock = clean
    install(monkeypatch, [rate_limit_error(retry_after=600)])
    monkeypatch.setattr(nv, "_await_cooldown", lambda: None)  # else we'd honour our own wait
    nv.complete("sys", "usr")
    # 600 s from the server beats the local first window (COOLDOWN_BASE * 2 = 30 s).
    assert nv.stats()["cooldown_until"] - clock.time() > 400


def test_cooldown_doubles_per_burst_and_caps(clean):
    assert nv._state["cooldown_len"] == nv.COOLDOWN_BASE
    nv._enter_cooldown()
    assert nv._state["cooldown_len"] == nv.COOLDOWN_BASE * 2
    for _ in range(20):
        nv._state["cooldown_until"] = 0.0      # simulate the window elapsing
        nv._enter_cooldown()
    assert nv._state["cooldown_len"] == nv.COOLDOWN_MAX == 300.0


def test_cooldown_does_not_extend_while_already_cooling(clean):
    first = nv._enter_cooldown()
    length_after_first = nv._state["cooldown_len"]
    second = nv._enter_cooldown()
    assert second == first
    assert nv._state["cooldown_len"] == length_after_first


def test_shrink_lowers_concurrency_but_not_below_floor(clean):
    for _ in range(nv.MAX_CONCURRENCY):
        nv._state["last_shrink"] = 0.0          # defeat the one-per-10s guard
        nv._shrink()
    assert nv._state["permits"] == nv.MIN_CONCURRENCY == 2


def test_shrink_is_rate_limited_to_one_per_ten_seconds(clean):
    nv._shrink()
    after_first = nv._state["permits"]
    nv._shrink()                                 # immediately again: must be ignored
    assert nv._state["permits"] == after_first == nv.MAX_CONCURRENCY - 1


# --------------------------------------------------------------------------- client

def test_client_refuses_to_build_without_an_api_key(monkeypatch):
    monkeypatch.setattr(nv, "API_KEY", None)
    monkeypatch.setattr(nv, "_client", None)
    with pytest.raises(RuntimeError, match="NVIDIA_APIKEY"):
        nv.client()


def test_thinking_flag_is_passed_through_extra_body(clean, monkeypatch):
    fc = install(monkeypatch, ["ok"])
    nv.complete("sys", "usr", thinking=True)
    kw = fc.calls[0]["extra_body"]["chat_template_kwargs"]
    assert kw["thinking"] is True
    assert kw["reasoning_effort"] == "high"


def test_thinking_off_by_default(clean, monkeypatch):
    fc = install(monkeypatch, ["ok"])
    nv.complete("sys", "usr")
    kw = fc.calls[0]["extra_body"]["chat_template_kwargs"]
    assert kw["thinking"] is False
    assert "reasoning_effort" not in kw
