"""The generation cache is keyed by tag — and an untagged reuse must FAIL CLOSED.

Spec: playbook campaign T6, defect found 2026-09-13.

`grpo_one_file` never set `cfg["gen_tag"]`, so every eval fell back to the literal
string "eval". Launched with GEN_REUSE=1, six RL trials x four evals each replayed
ONE cached file:

    [generations] reusing 16/16 from data\generations\eval.jsonl

Every trial returned 6/16 at every step -- including a trial whose generations
were EMPTY (len=0). The model was never generated from after the first eval, so
24 readings measured nothing. ReFT was immune because reft_hpo sets a fresh tag
per step ("never share a cache tag").

The file's own comment already said the tag must move with the checkpoint.
Nothing enforced it, which is why the comment was true and the code was wrong.
"""
from __future__ import annotations

import importlib.util
import os as _o
import sys as _s

for _d in ("docs",):
    _p = _o.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))), _d)
    _p not in _s.path and _s.path.insert(0, _p)

import pytest  # noqa: E402


@pytest.fixture(scope="module")
def gof():
    saved = _s.argv
    _s.argv = ["x"]
    spec = importlib.util.spec_from_file_location(
        "gof_tag", _o.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))),
                                "docs", "grpo_one_file.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    _s.argv = saved
    return m


class _M:
    training = False
    def eval(self): pass
    def train(self, mode=True): pass


def test_reuse_with_the_default_tag_raises(gof, monkeypatch):
    monkeypatch.setenv("GEN_REUSE", "1")
    monkeypatch.delenv("GEN_TAG", raising=False)
    with pytest.raises(RuntimeError, match="DEFAULT gen_tag"):
        gof.evaluate(_M(), None, [], dict(gof.DEFAULTS))


def test_reuse_with_a_real_tag_is_allowed(gof, monkeypatch):
    monkeypatch.setenv("GEN_REUSE", "1")
    cfg = dict(gof.DEFAULTS)
    cfg["gen_tag"] = "run_rd0_step3"
    # no RuntimeError about the tag; it fails later for lack of a real model
    with pytest.raises(Exception) as e:
        gof.evaluate(_M(), None, [], cfg)
    assert "DEFAULT gen_tag" not in str(e.value)


def test_without_reuse_the_default_tag_is_fine(gof, monkeypatch):
    # writing under "eval" is harmless; only READING BACK across models lies
    monkeypatch.delenv("GEN_REUSE", raising=False)
    monkeypatch.delenv("GEN_TAG", raising=False)
    with pytest.raises(Exception) as e:
        gof.evaluate(_M(), None, [], dict(gof.DEFAULTS))
    assert "DEFAULT gen_tag" not in str(e.value)


def test_the_env_tag_also_satisfies_the_guard(gof, monkeypatch):
    monkeypatch.setenv("GEN_REUSE", "1")
    monkeypatch.setenv("GEN_TAG", "some_checkpoint")
    with pytest.raises(Exception) as e:
        gof.evaluate(_M(), None, [], dict(gof.DEFAULTS))
    assert "DEFAULT gen_tag" not in str(e.value)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
