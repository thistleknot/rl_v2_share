"""How many judge calls may be in flight depends on WHICH judge is live.

Spec: playbook "OPEN — eval overhead" (closed 2026-09-12).

The eval scored 16 answers one-per-call at concurrency 2 -- a cap written for
local Ollama, which serves one request at a time on one GPU. Against OpenRouter
that cap silently throttled the eval to ~21s per read. Measured on the 240-record
pool, zero failures at every level: 1 worker 89.1s, 32 workers 12.1s (7.4x).

The cap must resolve at CALL time. Binding it as a default argument freezes it at
import, so flipping JUDGE_REMOTE after import would have no effect -- which is
exactly how the throttle stayed invisible.
"""
from __future__ import annotations

import os as _o
import sys as _s


import pytest  # noqa: E402

from rlv2.judge import reward as R  # noqa: E402


@pytest.fixture
def clean_env(monkeypatch):
    for k in ("JUDGE_REMOTE", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_local_ollama_keeps_the_queue_of_two(clean_env):
    assert R.judge_concurrency() == 2


def test_openrouter_lifts_the_cap(clean_env):
    clean_env.setenv("OPENROUTER_API_KEY", "sk-or-test")
    clean_env.setenv("JUDGE_REMOTE", "1")
    assert R.judge_concurrency() == 32


def test_a_key_without_the_remote_flag_stays_local(clean_env):
    # having a key is not the same as electing to use it
    clean_env.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert R.judge_concurrency() == 2


def test_the_cap_is_resolved_at_call_time_not_import_time(clean_env):
    before = R.judge_concurrency()
    clean_env.setenv("OPENROUTER_API_KEY", "sk-or-test")
    clean_env.setenv("JUDGE_REMOTE", "1")
    after = R.judge_concurrency()
    assert before == 2 and after == 32, "a default-arg binding would freeze this"


def test_flat_many_defaults_to_the_resolved_cap_not_a_frozen_one():
    import inspect
    sig = inspect.signature(R.llm_judge_flat_many)
    assert sig.parameters["max_workers"].default is None, \
        "max_workers must resolve inside the call, not bind at import"


def test_no_chunks_needs_no_judge_call():
    assert R.llm_judge_flat_many([]) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
