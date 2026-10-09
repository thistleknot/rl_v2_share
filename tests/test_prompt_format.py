"""One prompt format in the RL/pyreft lane, and no way back to two.

Operator 2026-09-14: "We always use the same prompt template" and "gut the other
from any traces so we dont walk back into this."

WHAT IT COST TO HAVE TWO. grpo_one_file carried both and chose at runtime from a
flag defaulting to the hand-built prompt. A base-start RL run left on that default
read 0/16 -- the raw base has never seen that format -- and the run had to be
relaunched twice before the mismatch was spotted.

MEASURED 2026-09-11, stock base, same 16 questions, think and tool-call tokens
banned in both arms:

    chat_wrap (hand-built)  DEGENERATE on 7/16
    model's own template    DEGENERATE on 0/16

src/half_epoch_eval.chat_wrap still exists -- the SFT trainer's own eval calls it
-- but nothing in the RL or pyreft path can reach it.
"""
import io

import pytest

RL = "docs/grpo_one_file.py"
TOOLS = ["archive/tools/reft_hpo.py", "archive/tools/rl_hpo_optuna.py"]


def _src(p):
    return io.open(p, encoding="utf-8").read()


def test_the_rl_trainer_has_no_format_switch():
    s = _src(RL)
    assert "_PROMPT_FORMAT" not in s, "the runtime global is gone"
    assert "cfg_prompt_format" not in s, "and its accessor"
    assert "--prompt-format" not in s, "and the flag that chose between them"


def test_the_rl_trainer_never_imports_the_hand_built_prompt():
    s = _src(RL)
    assert "from rlv2.eval.half_epoch_eval import chat_wrap" not in s
    assert "chat_wrap(tokenizer" not in s


def test_build_prompt_uses_the_models_own_template():
    s = _src(RL)
    i = s.index("def build_prompt(")
    body = s[i:s.index("\ndef ", i + 10)]
    assert "apply_chat_template" in body
    assert "enable_thinking=False" in body, "the pre-filled <think> block is what broke it before"
    assert '"role": "system"' in body, "the system turn stays (operator 2026-09-12)"


@pytest.mark.parametrize("path", TOOLS)
def test_no_tool_reintroduces_a_format_choice(path):
    s = _src(path)
    assert "_PROMPT_FORMAT" not in s, path
    assert "--prompt-format" not in s, path


def test_the_sft_eval_keeps_its_own_prompt_builder():
    """Gutting the RL lane must not break the SFT trainer's eval."""
    s = _src("src/rlv2/eval/half_epoch_eval.py")
    assert "def chat_wrap(" in s
    assert "chat_wrap(tok, q)" in s or "chat_wrap(tok," in s
