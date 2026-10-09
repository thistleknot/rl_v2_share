"""The replay sampler: judged rollouts reused as the world model.

Spec: dispositions.md REPLAY SAMPLER (2026-09-24) · Task: operator 2026-09-24 replay sampler

No GPU, no model load, no judge. A tiny embedding "policy" stands in for the model
so sequence_logprob and the clipped objective run for real; the judged-pairs buffer
is built in tmp_path from synthetic questions, never from the eval set.
"""
import importlib.util
import sys

import pytest

torch = pytest.importorskip("torch")

PAD, EOS, W = 0, 9, 5
V = 12


@pytest.fixture(scope="module")
def gof():
    spec = importlib.util.spec_from_file_location("gof_rs", "docs/grpo_one_file.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["gof_rs"] = m
    saved, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(m)
    sys.argv = saved
    return m


class _Enc(dict):
    def __init__(self, ids, mask):
        dict.__init__(self, input_ids=ids, attention_mask=mask)

    input_ids = property(lambda self: self["input_ids"])
    attention_mask = property(lambda self: self["attention_mask"])

    def to(self, _dev):
        return self


class FakeTok:
    """Word-level: a prompt of n characters is n copies of W."""

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
            return _Enc(seqs[0] if single else seqs, None)
        width = max(len(s) for s in seqs)
        ids, mask = [], []
        for s in seqs:
            gap = width - len(s)
            ids.append([PAD] * gap + s if self.padding_side == "left" else s + [PAD] * gap)
            mask.append([0] * gap + [1] * len(s) if self.padding_side == "left"
                        else [1] * len(s) + [0] * gap)
        return _Enc(torch.tensor(ids), torch.tensor(mask))

    def convert_tokens_to_ids(self, t):
        return EOS if t == "<|im_end|>" else -1

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids if not (skip_special_tokens and int(i) == EOS))


class FakeGen:
    """Scripted .generate(): one tail per row, counted so fresh generation is visible."""

    device = "cpu"

    def __init__(self, tails):
        self.tails = list(tails)
        self.config = type("C", (), {"use_cache": False})()
        self.is_gradient_checkpointing = False
        self.rows = 0

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=8, **kw):
        rows = []
        for _ in range(input_ids.shape[0]):
            self.rows += 1
            t = self.tails.pop(0)[:max_new_tokens]
            rows.append(list(t) + [PAD] * (max_new_tokens - len(t)))
        return torch.cat([input_ids, torch.tensor(rows)], dim=1)


class TinyPolicy(torch.nn.Module):
    """Logits = an embedding row per token: real parameters, real gradients."""

    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = torch.nn.Parameter(torch.randn(V, V, generator=g))

    def forward(self, ids):
        return type("O", (), {"logits": self.emb[ids]})()


def _cfg(gof, **kw):
    c = dict(gof.DEFAULTS)
    c.update(max_prompt=16, max_completion=8, gen_timeout_s=0.0, num_rollouts=4,
             raw_reward=True, logprob_norm="sum")
    c.update(kw)
    return c


def _pairs(gof, tmp_path, q, rows, sidecar=True, step=0):
    """Write judged_pairs.jsonl (+ gen_logp.jsonl) rows for one synthetic question."""
    import json
    with open(tmp_path / "judged_pairs.jsonl", "a", encoding="utf-8") as f:
        for a, sc in rows:
            f.write(json.dumps({"q": q, "gold": "g", "a": a, "score": sc, "step": step,
                                "round": 1, "lane": "stub", "real": True}) + "\n")
    if sidecar:
        gof.append_gen_logp(str(tmp_path), [
            {"key": gof.pair_key(q, a), "logp_sum": -3.0 - i, "n_tok": 2,
             "ans_ids": [W + 1 + i % 2, EOS], "finished": True, "round": 1,
             "opt_step": step} for i, (a, _sc) in enumerate(rows)])


# (a) ------------------------------------------------------------------------
def test_replay_k_zero_leaves_group_formation_unchanged(gof):
    """--replay-k 0: zero slots, K fresh per record, the same four keys back, and the
    loss returns every judge score untouched with no replay rows."""
    assert gof.replay_slots(0, 4) == 0
    assert gof.pick_replays([{"score": 0.1, "key": "a", "opt_step": 0},
                             {"score": 0.9, "key": "b", "opt_step": 0}], 0) == []
    tok, gen = FakeTok(), FakeGen([[W, EOS]] * 8)
    recs = [{"question": "abc", "gold": "g"}, {"question": "de", "gold": "g"}]
    out = gof.generate_groups(gen, tok, recs, _cfg(gof), batch=8)
    assert [len(g["answers"]) for g in out] == [4, 4]
    assert set(out[0]) == {"answers", "token_sets", "finished", "budget"}
    res = gof.compute_grpo_loss(TinyPolicy(), None, tok, recs[0], _cfg(gof),
                                rollouts=out[0], raw=[0.1, 0.9, 0.4, 0.6])
    assert res["raw"] == [0.1, 0.9, 0.4, 0.6]
    assert res["replay_stats"] == [] and res["slot_logp_sum"] is None


# (b) ------------------------------------------------------------------------
def test_eligible_record_gets_best_and_worst_stored_plus_k_minus_n_fresh(gof, tmp_path):
    q = "synthetic question alpha"
    _pairs(gof, tmp_path, q, [("mid", 0.5), ("best", 0.95), ("worst", 0.05), ("low", 0.2)])
    buf = gof.load_replay_buffer(str(tmp_path), {"unrelated eval q"})
    picks = gof.pick_replays(buf[q], gof.replay_slots(2, 4))
    assert [p["a"] for p in picks] == ["best", "worst"]

    tok, gen = FakeTok(), FakeGen([[W, EOS]] * 4)
    rec = {"question": q, "gold": "g"}
    cfg = _cfg(gof)
    gens = gof.generate_groups(gen, tok, [rec], cfg, batch=8,
                               n_fresh=[cfg["num_rollouts"] - len(picks)])
    assert gen.rows == 2, "only K - N fresh rollouts are generated"
    g, raw = gof.attach_replays(gens[0], [0.3, 0.7], rec, picks, tok, cfg, now_step=3)
    assert len(g["answers"]) == 4
    assert g["answers"][2:] == ["best", "worst"]
    assert raw == [0.3, 0.7, 0.95, 0.05]          # stored scores, NOT re-judged
    assert g["replay_logp_gen"][:2] == [None, None]
    assert g["replay_age"][2:] == [3, 3]
    full, plen = g["token_sets"][2]
    assert full[plen:].tolist() == picks[0]["ans_ids"]   # stored ids, not re-encoded
    assert plen == g["token_sets"][0][1], "replayed prompt must match the generated one"


def test_replay_k_above_k_minus_2_is_clamped(gof):
    assert gof.replay_slots(3, 4) == 2
    assert gof.replay_slots(5, 4) == 2
    assert gof.replay_slots(2, 3) == 1


# (c) ------------------------------------------------------------------------
def test_record_without_two_distinct_stored_scores_gets_k_fresh(gof, tmp_path):
    q_flat, q_nolog = "synthetic flat", "synthetic no sidecar"
    _pairs(gof, tmp_path, q_flat, [("a1", 0.0), ("a2", 0.0), ("a3", 0.0)])
    _pairs(gof, tmp_path, q_nolog, [("b1", 0.1), ("b2", 0.9)], sidecar=False)
    buf = gof.load_replay_buffer(str(tmp_path), set())
    assert gof.pick_replays(buf.get(q_flat), 2) == []
    assert q_nolog not in buf, "a pair with no stored pi_gen is not eligible"
    assert gof.pick_replays(buf.get(q_nolog), 2) == []
    tok, gen = FakeTok(), FakeGen([[W, EOS]] * 4)
    out = gof.generate_groups(gen, tok, [{"question": q_flat, "gold": "g"}], _cfg(gof),
                              batch=8, n_fresh=[4 - 0])
    assert len(out[0]["answers"]) == 4 and gen.rows == 4


# (d) ------------------------------------------------------------------------
def test_replayed_scores_stay_out_of_counters_and_beta(gof, tmp_path):
    """Fresh group all-wrong, replayed best is right. If the replayed 1.0 leaked, the
    record would read STRADDLE in recent_* and its Beta would count a success."""
    q = "synthetic question beta"
    _pairs(gof, tmp_path, q, [("right", 1.0), ("wrong", 0.0)])
    buf = gof.load_replay_buffer(str(tmp_path), set())
    picks = gof.pick_replays(buf[q], 2)
    tok, gen = FakeTok(), FakeGen([[W, EOS]] * 2)
    rec = {"question": q, "gold": "g"}
    cfg = _cfg(gof, replay_k=2)
    gens = gof.generate_groups(gen, tok, [rec], cfg, batch=8, n_fresh=[2])
    g, raw = gof.attach_replays(gens[0], [0.0, 0.1], rec, picks, tok, cfg, now_step=1)
    res = gof.compute_grpo_loss(TinyPolicy(), None, tok, rec, cfg, rollouts=g, raw=raw)
    assert res["raw"] == [0.0, 0.1]
    assert res["gated"] == pytest.approx([0.0, 0.1])
    assert len(res["replay_stats"]) == 2
    assert not res["dead"] and res["dead_fresh"] is False

    # the trainer's own book-keeping, fed exactly what the screen hands it
    history, visits = {}, {}
    gof._POOLED.pop(q, None)
    history.setdefault(q, []).extend(res["raw"])
    gof.record_visit(visits, q, res["raw"], floor=0.7)
    assert gof._POOLED[q] == (0, 2), "Beta(w, l) moved on a replayed score"
    sig = gof._straddle_census(history, [rec], 0.7, visits, rollouts=4 - 2)
    assert sig["recent_fail"] == 1 and sig["recent_straddle"] == 0


# (e) ------------------------------------------------------------------------
def _grad_with(gof, logp_gen_offset, alt_ids):
    """Gradient of the loss for a 2 fresh + 1 replayed group, the replayed answer
    carrying token ids `alt_ids` and pi_gen = pi_now + offset."""
    torch.manual_seed(0)
    pol, tok = TinyPolicy(seed=1), FakeTok()
    cfg = _cfg(gof, replay_eps=0.2)
    rec = {"question": "synthetic question eps", "gold": "g"}
    plen = len(tok(gof.build_prompt(rec, tok)).input_ids)
    prompt = [W] * plen
    fresh = [torch.tensor(prompt + [6, EOS]), torch.tensor(prompt + [7, EOS])]
    rep = torch.tensor(prompt + alt_ids)
    now = float(gof.sequence_logprob(pol, rep, plen, dict(cfg)).detach())
    g = {"answers": ["f1", "f2", "r"], "token_sets": [(fresh[0], plen), (fresh[1], plen),
                                                       (rep, plen)],
         "finished": [True, True, True], "budget": 8,
         "replay_logp_gen": [None, None, now - logp_gen_offset],
         "replay_age": [None, None, 2]}
    res = gof.compute_grpo_loss(pol, None, tok, rec, cfg, rollouts=g, raw=[0.2, 0.5, 0.9])
    res["loss_tensor"].backward()
    return pol.emb.grad.clone(), res


def test_replayed_ratio_outside_clip_is_masked_to_zero_gradient(gof):
    """Discriminating form: swap the replayed answer's tokens. If the masked slot
    contributed anything, the gradient would move with its tokens; it does not.
    The in-clip control DOES move, so the test can fail."""
    g1, r1 = _grad_with(gof, logp_gen_offset=-1.0, alt_ids=[8, EOS])   # ratio e^-1
    g2, r2 = _grad_with(gof, logp_gen_offset=-1.0, alt_ids=[10, 11, EOS])
    assert r1["replay_stats"][0]["masked"] and r2["replay_stats"][0]["masked"]
    # per-token: a summed -1.0 over 2 answer tokens is -0.5 per token, still masked
    assert abs(r1["replay_stats"][0]["logratio"] - (-0.5)) < 1e-4
    assert torch.allclose(g1, g2), "a masked replayed sample moved the gradient"

    c1, s1 = _grad_with(gof, logp_gen_offset=-0.05, alt_ids=[8, EOS])  # ratio ~0.95
    c2, s2 = _grad_with(gof, logp_gen_offset=-0.05, alt_ids=[10, 11, EOS])
    assert not s1["replay_stats"][0]["masked"]
    assert not torch.allclose(c1, c2), "control: an in-clip replayed sample must count"


def test_replayed_ratio_is_per_token_so_a_long_answer_is_not_masked(gof):
    """Pre-test 2026-09-24: summed drift masked 75-94% of replays at every age while
    per-token drift sat inside the clip. Summed |logratio| 0.4 over 4 answer tokens is
    0.1 per token: the summed ratio (e^-0.4 = 0.67) would be masked, per-token
    (e^-0.1 = 0.905) is not -- and it still moves the gradient."""
    g1, r1 = _grad_with(gof, logp_gen_offset=-0.4, alt_ids=[6, 7, 8, EOS])
    s = r1["replay_stats"][0]
    assert abs(s["logratio"] - (-0.1)) < 1e-4
    assert abs(s["logratio"] * 4) > 0.18, "the summed log ratio must exceed the clip"
    assert not s["masked"]
    g2, _ = _grad_with(gof, logp_gen_offset=-0.4, alt_ids=[10, 11, 10, EOS])
    assert not torch.allclose(g1, g2), "an in-clip replayed sample must count"


def test_answer_older_than_max_age_is_not_eligible(gof):
    ents = [{"a": "old_best", "score": 0.99, "key": "k1", "opt_step": 4},   # age 6
            {"a": "best", "score": 0.9, "key": "k2", "opt_step": 5},        # age 5
            {"a": "worst", "score": 0.1, "key": "k3", "opt_step": 8}]       # age 2
    picks = gof.pick_replays(ents, 2, now_step=10, max_age=5)
    assert [p["a"] for p in picks] == ["best", "worst"]
    only_old = [ents[0], {"a": "old_worst", "score": 0.0, "key": "k4", "opt_step": 1}]
    assert gof.pick_replays(only_old, 2, now_step=10, max_age=5) == []
    assert [p["a"] for p in gof.pick_replays(only_old, 2)] == ["old_best", "old_worst"]


def test_replay_log_line_reports_counts_and_age(gof):
    line = gof.replay_log_line([{"logratio": 0.1, "masked": False, "age": 1},
                                {"logratio": -0.5, "masked": True, "age": 3},
                                {"logratio": 0.3, "masked": False, "age": 1}])
    assert line == ("[replay] n_replayed=3 n_masked=1 mean_abs_logratio=0.3000 "
                    "by_age={1:2, 3:1}")


# (f) ------------------------------------------------------------------------
def test_eval_overlap_tripwire_fails_closed(gof, tmp_path):
    frozen = "synthetic frozen eval question"          # never a real eval question
    _pairs(gof, tmp_path, "synthetic pool question", [("x", 0.1), ("y", 0.9)])
    _pairs(gof, tmp_path, frozen, [("z", 0.5)])
    with pytest.raises(RuntimeError, match="LEAKAGE TRIPWIRE"):
        gof.load_replay_buffer(str(tmp_path), {frozen})


# (g) ------------------------------------------------------------------------
def test_gen_logp_sidecar_round_trips_by_key(gof, tmp_path):
    q, a = "synthetic question gamma", "an answer\nwith a newline"
    row = {"key": gof.pair_key(q, a), "logp_sum": -12.5, "n_tok": 7,
           "ans_ids": [1, 2, 3, EOS], "finished": False, "round": 2, "opt_step": 5}
    gof.append_gen_logp(str(tmp_path), [row])
    got = gof.load_gen_logp(str(tmp_path))
    assert got[gof.pair_key(q, a)] == row
    assert gof.pair_key(q, a) != gof.pair_key(q, a + " ")


def test_judge_batch_accepts_ragged_groups(monkeypatch):
    """Spec: dispositions.md REPLAY SAMPLER (2026-09-24). With --replay-k an eligible record
    sends K-N fresh answers and an ineligible one sends K, so one judge wave is ragged.
    Measured 2026-09-25: the equal-size assertion crashed rl_replay at step 2."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from rlv2.judge import reward
    # fake judge: score = length of the generated text, one pair per call
    monkeypatch.setattr(reward, "llm_judge_flat_many",
                        lambda chunks, model=None: [[float(len(c[0]["generated"]))] for c in chunks])
    groups = [
        {"question": "q1", "gold": "g", "completions": ["a", "bb", "ccc", "dddd"]},
        {"question": "q2", "gold": "g", "completions": ["ee", "f"]},
        {"question": "q3", "gold": "g", "completions": ["ggg", "hhhh", "i"]},
    ]
    out = reward.llm_judge_batch_train(groups, model="stub")
    assert [len(o) for o in out] == [4, 2, 3]
    assert out == [[1.0, 2.0, 3.0, 4.0], [2.0, 1.0], [3.0, 4.0, 1.0]]
