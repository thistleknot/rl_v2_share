"""The eval's token budget is derived from the eval's own golds, and covers all of them.

NO GOVERNING SPEC. Basis: operator 2026-09-16 -- pushing back on a hand-set 96. Measured
that day: gold tokens over data/rl/r3_eval_canonical.jsonl ran min 4 / p50 42 / p90 79 /
max 99, so the 96 cap did not even cover one question's OWN GOLD. That answer could not be
produced in full, and the no-EOS gate scored the truncation 0.
"""
import io
import json
import sys


EVAL = "data/rl/r3_eval_canonical.jsonl"
HEADROOM = 1.5


def _gold_tokens():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("skills_model", trust_remote_code=True)
    rows = [json.loads(l) for l in io.open(EVAL, encoding="utf-8") if l.strip()]
    return sorted(len(tok(r["gold"]).input_ids) for r in rows)


def test_the_budget_covers_every_gold():
    """A budget below a gold's own length makes that question unanswerable."""
    from rlv2.eval import half_epoch_eval as he
    assert he.MAX_NEW_TOKENS >= max(_gold_tokens()), (
        "budget %d is below the longest gold %d" % (he.MAX_NEW_TOKENS, max(_gold_tokens())))


def test_the_budget_carries_headroom_over_the_longest_gold():
    """The model may phrase a correct answer longer than the gold does."""
    from rlv2.eval import half_epoch_eval as he
    assert he.MAX_NEW_TOKENS >= max(_gold_tokens()) * HEADROOM - 1


def test_the_budget_is_not_wastefully_large():
    """Headroom is not licence; every extra token is eval time on every arm."""
    from rlv2.eval import half_epoch_eval as he
    assert he.MAX_NEW_TOKENS <= max(_gold_tokens()) * 3


def test_the_budget_is_a_token_count_not_a_clock():
    """A wall-clock cut varies with GPU load and would re-break determinism."""
    src = io.open("src/rlv2/eval/half_epoch_eval.py", encoding="utf-8").read()
    i = src.index("MAX_NEW_TOKENS = ")
    assert src[i + len("MAX_NEW_TOKENS = "):i + 30].split()[0].isdigit()
