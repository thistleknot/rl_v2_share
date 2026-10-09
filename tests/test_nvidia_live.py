"""
Live smoke test: confirm the NVIDIA endpoint actually returns a completion.

This is deliberately a LIVE test -- it makes a real network call. Its whole
purpose is to answer "can we get a result back from the API, yes or no?",
so mocking the transport would defeat the point.
"""
import os as _o, sys as _s
import pytest

from rlv2.judge import nvidia_llm as nv


def test_api_key_is_loaded():
    """The key must be picked up from .env at import time."""
    assert nv.API_KEY, (
        "No API key resolved. nvidia_llm.load_dotenv() should populate "
        "NVIDIA_APIKEY or NVIDIA_API_KEY from RL_V2/.env"
    )


def test_complete_returns_a_usable_result():
    """A real round-trip must return a non-empty string."""
    if not nv.API_KEY:
        pytest.skip("no API key -- cannot make a live call")

    out = nv.complete(
        "You are terse. Answer with one word.",
        "Reply with exactly: PONG",
        max_tokens=64,
    )

    print("\n--- RAW RESPONSE ---\n%r\n--------------------" % (out,))
    print("model: %s" % nv.MODEL)
    print("stats: %s" % (nv.stats(),))

    # The core assertion: we got something back.
    assert out is not None, "complete() returned None"
    assert isinstance(out, str), "expected str, got %s" % type(out).__name__
    assert out.strip(), "complete() returned an empty/whitespace-only string"
