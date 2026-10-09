"""Batched rollout generation -- the padding must not change any verdict.

NO GOVERNING SPEC. Basis: operator 2026-09-19 -- "It looks like youre generating
one at a time. Why not batch?"

WHAT CAN GO WRONG HERE IS SILENT. A batched .generate() returns a rectangular
tensor, and every one of these mistakes still produces a plausible-looking
string:

  right padding        pad tokens land BETWEEN the prompt and the answer, so the
                       model conditions on them and sequence_logprob scores them
  a wrong prompt_len   the policy gradient then pays for prompt tokens, which is
                       exactly the completion-only-loss rule being broken
  trailing pads kept   the tail's last token is a pad, so the EOS gate reads
                       "never stopped" and zeroes a perfectly finished answer
  one shared budget    answer_budget is per QUESTION; letting a short record run
                       to a long neighbour's ceiling changes its EOS verdict

So these tests drive a scripted fake model whose output is known exactly, and
assert the RECONSTRUCTION, not the text.
"""
import importlib.util
import sys

import pytest

torch = pytest.importorskip("torch")

PAD, EOS, W = 0, 9, 5      # pad id, stop id, an ordinary word id


@pytest.fixture(scope="module")
def gof():
    spec = importlib.util.spec_from_file_location("gof_bg", "docs/grpo_one_file.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["gof_bg"] = m
    saved, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(m)
    sys.argv = saved
    return m


class _Enc(dict):
    """The two fields .generate() and the slicing need, unpackable as **enc."""

    def __init__(self, ids, mask):
        dict.__init__(self, input_ids=ids, attention_mask=mask)

    input_ids = property(lambda self: self["input_ids"])
    attention_mask = property(lambda self: self["attention_mask"])

    def to(self, _dev):
        return self


class FakeTok:
    """Word-level: a prompt of n characters is n copies of W. Left-pads on demand."""

    pad_token_id = PAD
    eos_token_id = EOS
    padding_side = "right"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False):
        return msgs[-1]["content"]

    def __call__(self, prompts, return_tensors=None, truncation=True, padding=False,
                 max_length=None, add_special_tokens=True):
        single = isinstance(prompts, str)
        if single:
            prompts = [prompts]
        seqs = [[W] * min(len(p), max_length or len(p)) for p in prompts]
        if return_tensors is None:
            # A REAL TOKENIZER RETURNS LISTS HERE, and len() of a list is the token
            # count. Returning a tensor made len() read 1 for every prompt, which
            # silently flattened the prompt half of the sort key.
            return _Enc(seqs[0] if single else seqs, None)
        width = max(len(s) for s in seqs)
        ids, mask = [], []
        for s in seqs:
            gap = width - len(s)
            if self.padding_side == "left":
                ids.append([PAD] * gap + s)
                mask.append([0] * gap + [1] * len(s))
            else:
                ids.append(s + [PAD] * gap)
                mask.append([1] * len(s) + [0] * gap)
        return _Enc(torch.tensor(ids), torch.tensor(mask))

    def convert_tokens_to_ids(self, t):
        return EOS if t == "<|im_end|>" else -1

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


class FakeModel:
    """Emits a scripted tail per row, then pads the rest of the rectangle."""

    device = "cpu"

    def __init__(self, tails):
        self.tails = list(tails)     # consumed in call order, one per row
        self.config = type("C", (), {"use_cache": False})()
        self.is_gradient_checkpointing = False
        self.calls = 0

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=8, **kw):
        self.calls += 1
        rows = []
        for _ in range(input_ids.shape[0]):
            t = self.tails.pop(0)[:max_new_tokens]
            rows.append(list(t) + [PAD] * (max_new_tokens - len(t)))
        return torch.cat([input_ids, torch.tensor(rows)], dim=1)


def _cfg(gof, **kw):
    c = dict(gof.DEFAULTS)
    c.update(max_prompt=16, max_completion=8, gen_timeout_s=0.0, num_rollouts=2)
    c.update(kw)
    return c


def test_left_padding_is_stripped_so_prompt_len_indexes_the_first_answer_token(gof):
    """The short prompt is the one that gets padded; its reconstruction is the test."""
    tok, model = FakeTok(), FakeModel([[W, EOS], [W, W, EOS]])
    rows = gof.sample_responses(model, tok, ["abc", "a"], _cfg(gof),
                                budgets=[8, 8], batch=2)
    full_long, plen_long, _, ok_long = rows[0]
    full_short, plen_short, _, ok_short = rows[1]
    assert plen_long == 3 and plen_short == 1, "prompt_len is the REAL length"
    assert full_short.tolist()[:plen_short] == [W], "no pad survived into the prompt"
    assert full_long.tolist() == [W, W, W, W, EOS], "3 prompt + [W, EOS]"
    assert ok_long and ok_short


def test_a_tail_is_cut_at_its_first_stop_token_not_at_the_rectangle_edge(gof):
    """Trailing pads must never reach the tail -- the EOS gate reads its last token."""
    tok, model = FakeTok(), FakeModel([[W, EOS], [W, W, W, W]])
    rows = gof.sample_responses(model, tok, ["ab", "ab"], _cfg(gof),
                                budgets=[8, 8], batch=2)
    assert rows[0][0].tolist() == [W, W, W, EOS], "cut right after the stop token"
    assert rows[0][3] is True
    assert rows[1][0].tolist() == [W, W, W, W, W, W],         "no stop token: 2 prompt + 4 answer, the rectangle's pads stripped"
    assert rows[1][3] is False, "never stopped -- the EOS gate must see that"


def test_each_record_is_held_to_its_own_budget_not_the_batchs(gof):
    """answer_budget is per QUESTION. A stop token arriving past THIS record's
    budget did not arrive in the serial path either, so it is not a finish."""
    tok, model = FakeTok(), FakeModel([[W, W, W, EOS], [W, W, W, EOS]])
    rows = gof.sample_responses(model, tok, ["ab", "ab"], _cfg(gof),
                                budgets=[8, 2], batch=2)
    assert rows[0][3] is True, "budget 8 -- the stop token at 4 counts"
    assert rows[1][0].tolist() == [W, W, W, W], "budget 2 -> 2 prompt + 2 answer"
    assert rows[1][3] is False, "budget 2 -- the stop token at 4 never arrived"


def test_the_whole_window_rides_one_generate_call(gof):
    """THE POINT OF THE CHANGE. Four sequences, one call -- not four."""
    tok, model = FakeTok(), FakeModel([[W, EOS]] * 4)
    gof.sample_responses(model, tok, ["ab"] * 4, _cfg(gof), budgets=[8] * 4, batch=8)
    assert model.calls == 1, "batch=8 must not split 4 sequences"


def test_the_sub_batch_cap_is_honoured(gof):
    """The KV cache grows linearly with the batch; the cap is what bounds VRAM."""
    tok, model = FakeTok(), FakeModel([[W, EOS]] * 6)
    gof.sample_responses(model, tok, ["ab"] * 6, _cfg(gof), budgets=[8] * 6, batch=2)
    assert model.calls == 3, "6 sequences at cap 2 = 3 calls"


def test_generate_groups_returns_k_rollouts_per_record_in_record_order(gof):
    """The owner mapping is the one thing a flat batch can silently scramble: a
    rollout filed under the wrong record would be judged against the wrong gold."""
    recs = [{"question": "a" * 3, "gold": "g1"}, {"question": "b" * 4, "gold": "g2"}]
    tok, model = FakeTok(), FakeModel([[W, EOS]] * 4)
    cfg = _cfg(gof, num_rollouts=2)
    out = gof.generate_groups(model, tok, recs, cfg, batch=8)
    assert len(out) == 2
    for g, r in zip(out, recs):
        assert len(g["answers"]) == 2 and len(g["token_sets"]) == 2
        assert len(g["finished"]) == 2 and g["budget"] > 0
        plen = g["token_sets"][0][1]
        assert plen == len(gof.build_prompt(r, tok)), "wrong record's prompt length"


def test_generate_groups_returns_the_shape_compute_grpo_loss_hands_back(gof):
    """THE CONTRACT. The probe feeds this straight into compute_grpo_loss(rollouts=),
    so a renamed key blanks the probe rather than crashing it."""
    tok, model = FakeTok(), FakeModel([[W, EOS]] * 2)
    out = gof.generate_groups(model, tok, [{"question": "ab", "gold": "g"}],
                              _cfg(gof, num_rollouts=2), batch=8)
    assert set(out[0]) == {"answers", "token_sets", "finished", "budget"}


def test_short_records_batch_wider_than_long_ones_at_the_same_token_cap(gof):
    """DYNAMIC BATCHING. A fixed record count sizes every batch for the worst case;
    the token cap is what lets six short sequences ride where two long ones would."""
    tok = FakeTok()
    cfg = _cfg(gof, gen_tokens=24, gen_batch=6)
    short = FakeModel([[W, EOS]] * 6)
    gof.sample_responses(short, tok, ["ab"] * 6, cfg, budgets=[2] * 6)
    assert short.calls == 1, "width 4 -> 24//4 = 6 sequences in one call"
    long = FakeModel([[W, EOS]] * 6)
    gof.sample_responses(long, tok, ["abcdefgh"] * 6, cfg, budgets=[8] * 6)
    assert long.calls == 6, "width 16 > the 24-token cap -> one at a time"


def test_the_sort_key_is_prompt_plus_budget_not_budget_alone(gof):
    """A rectangle pays its widest member twice -- once padding the prompt, once in
    the shared max_new_tokens. Sorting on the budget alone leaves the prompt half
    of that unsorted."""
    import inspect
    src = inspect.getsource(gof.sample_responses)
    assert "width = [plen[i] + budgets[i]" in src
    assert "key=lambda i: -width[i]" in src
