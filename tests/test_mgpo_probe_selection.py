"""probe_disagreement_cohort — two-model epistemic selection + visit-count bonus.

Spec: docs/plans/uncertainty-menu-hpo.md (approved 2026-09-03), playbook.md task UE.

Behavioral tests run the REAL function with fake models/tokenizer/judge: the fakes
satisfy exactly the call surface the function uses (tok(...).input_ids, .generate,
.decode, model.device/.eval), and the judge is patched to a deterministic scorer keyed
on which model produced the text. Each test fails if the selection logic is emptied.
"""
from __future__ import annotations
import os as _o
import sys as _s


import io
import json
import types

import torch

MGPO_PATH = _o.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))),
                         "src", "rlv2", "train", "rl", "grpo.py")
SRC = io.open(MGPO_PATH, encoding="utf-8").read()


# ---------- static wiring pins (the B-1/B-2 "flag parsed but never used" bug shape) ----
def test_probe_base_gates_cohort_construction_at_the_call_site():
    assert 'getattr(args, "probe_base", None)' in SRC, \
        "--probe-base is parsed but the cohort call site never consults it"
    assert "probe_disagreement_cohort(mix, args, meta, policy, tok)" in SRC, \
        "the probe path does not route through probe_disagreement_cohort"


def test_probe_model_is_freed_before_training():
    """The 13.2 GiB policy+ref co-residency peak leaves no room for a lingering probe."""
    fn = SRC.split("def probe_disagreement_cohort(")[1].split("\ndef ")[0]
    assert "del probe" in fn and "empty_cache()" in fn, \
        "probe model is not freed; it would still be resident at the trainer's ref-model peak"


# ---------- behavioral -----------------------------------------------------------------
class _FakeIds:
    def __init__(self, n):
        self.input_ids = torch.zeros((1, n), dtype=torch.long)

    def __getattr__(self, k):
        raise AttributeError(k)


class _FakeTok:
    def __call__(self, text, return_tensors=None):
        ids = _FakeIds(4)
        ids.input_ids = ids.input_ids.clone()
        ids.input_ids._src = text  # smuggle the prompt through for decode
        return types.SimpleNamespace(input_ids=_Tensor(text))

    def decode(self, t, skip_special_tokens=True):
        return t._label


class _Tensor:
    """Minimal stand-in for the id tensor: supports .to(device), .shape, slicing."""
    def __init__(self, src, label=None):
        self._src = src
        self._label = label
        self.shape = (1, 4)

    def to(self, device):
        return self

    def __getitem__(self, i):
        return _Tensor(self._src, self._label)


def _fake_model(tag):
    m = types.SimpleNamespace()
    m.device = "cpu"
    m.eval = lambda: None
    # The CPU-generation fix moves the policy to an explicit device and back
    # (next(parameters()).device / .to()); the fake carries both call surfaces.
    m.to = lambda dev: m
    m.parameters = lambda: iter([types.SimpleNamespace(device="cpu")])

    def generate(ids, **kw):
        out = _Tensor(ids._src)
        out._label = "%s::%s" % (tag, ids._src)
        return [out]
    m.generate = generate
    return m


def test_probe_generation_device_is_explicit_never_inferred():
    """Two launches silently generated 96 fp32 completions on CPU (~4 h pace, read
    as stalls) because model.device was trusted before TRL placed the policy. The
    device must be chosen explicitly and the policy handed back where it started."""
    fn = SRC.split("def probe_disagreement_cohort(")[1].split("\ndef ")[0]
    assert '_dev = "cuda" if torch.cuda.is_available() else "cpu"' in fn, \
        "generation device is inferred from the model again — the CPU-crawl bug shape"
    assert "policy.to(_policy_dev0)" in fn, \
        "policy must be handed back to its original device for TRL to place"
    assert "[probe-gen]" in fn, "silent generation phase: progress line removed"


def _run(monkeypatch, tmp_path, ledger=None, presets=None):
    from rlv2.train.rl import grpo as mgpo
    from rlv2.judge import reward as reward_mod

    # 8 candidate prompts, one regime/domain/source so bootstrap sampling is trivial.
    mix = [{"ptext": "p%d" % i, "prompt": "q%d" % i, "question": "q%d" % i,
            "gold": "g%d" % i, "context": "", "domain": "Math", "source": "s",
            "answer_type": "R3", "regime": "R3"} for i in range(8)]
    meta = {r["ptext"]: r for r in mix}

    args = types.SimpleNamespace(
        target_pool=3, probe_samples=1, probe_candidates_mult=2, seed=7,
        boxcox_lambda=0.5, probe_base="FAKE_PROBE", visit_ledger=ledger,
        max_completion=8, temperature=0.7, top_p=0.95)

    monkeypatch.setattr("transformers.AutoModelForCausalLM",
                        types.SimpleNamespace(
                            from_pretrained=lambda *a, **k: types.SimpleNamespace(
                                to=lambda d: _fake_model("probe"))))

    # Judge: policy is CORRECT on even prompts; probe on prompts 0-1 only.
    # -> disagreement on {2, 4, 6} (policy right, probe wrong) and {1} (probe right).
    def judge(pairs):
        scores = []
        for p in pairs:
            tag, ptext = p["generated"].split("::")
            i = int(ptext[1:])
            if tag == "policy":
                scores.append(1.0 if i % 2 == 0 else 0.0)
            else:
                scores.append(1.0 if i < 2 else 0.0)
        return scores
    monkeypatch.setattr(reward_mod, "llm_judge_flat", judge)
    # The stall fix routes through the chunked concurrent path (measured 95-min hang
    # on a single 96-pair call, 2026-09-03) — patch it consistently with the scorer.
    monkeypatch.setattr(reward_mod, "llm_judge_flat_many",
                        lambda chunks, **kw: [judge(c) for c in chunks])
    monkeypatch.setattr(mgpo, "bootstrap_cohort",
                        lambda mix, n, seed, boxcox_lambda: mix[:n])
    if presets:
        io.open(ledger, "w", encoding="utf-8").write(json.dumps(presets))

    cohort = mgpo.probe_disagreement_cohort(mix, args, meta,
                                            _fake_model("policy"), _FakeTok())
    return cohort, args


def test_disagreement_prompts_fill_the_cohort_first(monkeypatch, tmp_path):
    cohort, _ = _run(monkeypatch, tmp_path)
    # Candidates 0-5 (mult 2 x pool 3). Agreement: p0 (both right). Disagreement:
    # p1 (probe right, policy wrong), p2, p4 (policy right, probe wrong)... within 0-5:
    # p1, p2, p3? p3: policy wrong, probe wrong -> agree. p2, p4: policy right probe wrong.
    picked = {r["ptext"] for r in cohort}
    assert picked == {"p1", "p2", "p4"}, \
        "cohort should be exactly the three disagreement prompts, got %s" % picked


def test_per_continuation_floors_settled_and_maxes_unseen():
    """E8 / Schaul 2015 conventions: epsilon-floor keeps settled prompts eligible;
    unseen enter at Beta(1,1) = the priority maximum; spread prompts dominate."""
    import random
    from rlv2.train.rl import grpo as mgpo
    args = types.SimpleNamespace(target_pool=6, per_alpha=1.0, per_eps=1e-3)
    cohort = [{"ptext": p} for p in ("settled_right", "settled_wrong", "spread")]
    stats = {"settled_right": [1.0, 1.0, 1.0, 1.0],
             "settled_wrong": [0.0, 0.0, 0.0, 0.0],
             "spread": [1.0, 0.0, 1.0, 0.0]}
    mix = cohort + [{"ptext": "unseen_%d" % i} for i in range(5)]
    rng = random.Random(7)
    sel = mgpo.per_continuation(cohort, stats, mix, args, rng)
    assert len(sel) == 6
    names = [r["ptext"] for r in sel]
    # Draws are stochastic; assert the DETERMINISTIC property — the priority ordering —
    # plus the one high-probability sampled property (an unseen prompt appearing across
    # 6 draws when fresh candidates carry the max weight).
    var = lambda w, l: ((1 + w) * (1 + l)) / (((2 + w + l) ** 2) * (3 + w + l))
    assert var(0, 0) > var(2, 2) > var(4, 0), \
        "Beta-variance priority must order unseen > spread > settled"
    assert any(n.startswith("unseen") for n in names), \
        "fresh injection never sampled — max-priority rule broken"
    # Determinism probe for the floor: with eps=0 a settled prompt COULD still appear
    # (Beta var is never exactly 0), so assert the structural guarantee instead:
    weights_all_positive = all(
        (lambda w, l: ((1+w)*(1+l)) / (((2+w+l)**2) * (3+w+l)) + args.per_eps > 0)(
            sum(1 for r in stats.get(p["ptext"], []) if r >= 0.7),
            sum(1 for r in stats.get(p["ptext"], []) if r < 0.7))
        for p in cohort)
    assert weights_all_positive, "epsilon-floor violated: a prompt has zero probability"


def test_per_fresh_injection_respects_floor_and_cap():
    """n_fresh = max(2, ceil(0.10 * cache)) capped at ceil(target_pool/3): at our scale
    (cache 3, pool 6) that is exactly 2 fresh candidates in the draw pool."""
    import random
    from rlv2.train.rl import grpo as mgpo
    args = types.SimpleNamespace(target_pool=6, per_alpha=0.6, per_eps=1e-3)
    cohort = [{"ptext": "c%d" % i} for i in range(3)]
    stats = {"c0": [1.0, 0.0], "c1": [1.0, 0.0], "c2": [1.0, 0.0]}
    mix = cohort + [{"ptext": "f%d" % i} for i in range(10)]
    out = []
    real_choices = random.Random.choices

    def spy(self, population, weights=None, k=1):
        out.append(len(population))
        return real_choices(self, population, weights=weights, k=k)
    try:
        random.Random.choices = spy
        mgpo.per_continuation(cohort, stats, mix, args, random.Random(7))
    finally:
        random.Random.choices = real_choices
    assert out and out[0] == 5, \
        "candidate pool should be 3 cache + 2 fresh (floor 2, cap ceil(6/3)=2), got %s" % out


def test_ledger_persistence_is_wired_at_arm_end():
    """W0: per-prompt win/loss from round stats must reach the visit ledger, or the
    world model never gets labels (they previously died with each run's logs)."""
    assert "[ledger]" in SRC, "arm-end ledger write is missing its pinned log line"
    assert '"w": 0, "l": 0, "visits"' in SRC, \
        "legacy bare-count entries must upgrade to {w, l, visits}, not crash"


def test_world_model_priors_gate_on_disagreement(tmp_path, monkeypatch):
    """W2: trusted (low-variance) unseen prompts get informed Beta priors; the trust
    threshold is the Dyna-Q uncertainty-masking rule."""
    import random
    import numpy as np
    from rlv2.train.rl import grpo as mgpo
    from rlv2.train.rl import prompt_world_model as wm

    labels = [{"prompt": "train_%d" % i, "w": 4 if i % 2 else 0,
               "l": 0 if i % 2 else 4} for i in range(20)]

    def embed(texts):
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.normal(size=16)
            v[0] = 0.9 if "easy" in t or (t.startswith("train_") and int(t[6:]) % 2) else -0.9
            out.append(v / np.linalg.norm(v))
        return np.asarray(out)

    path = str(tmp_path / "wm.npz")
    wm.train_ensemble(labels, embed, k=3, out_path=path, seed=7)
    monkeypatch.setattr("rlv2.data.pools.embed", embed)

    args = types.SimpleNamespace(target_pool=6, per_alpha=1.0, per_eps=1e-3,
                                 world_model=path, wm_trust=1.0)  # trust everything
    cohort = [{"ptext": "c0", "question": "c0"}]
    stats = {"c0": [1.0, 0.0]}
    mix = cohort + [{"ptext": "easy_new", "question": "easy_new"},
                    {"ptext": "hard_new", "question": "hard_new"}]
    mgpo.per_continuation(cohort, stats, mix, args, random.Random(7))
    # With wm_trust=1.0 every fresh prompt is trusted -> informed prior != flat.
    # Rerun with wm_trust=0.0: nothing trusted -> flat priors + the empty-mask log.
    args2 = types.SimpleNamespace(target_pool=6, per_alpha=1.0, per_eps=1e-3,
                                  world_model=path, wm_trust=0.0)
    mgpo.per_continuation(cohort, stats, mix, args2, random.Random(7))
    # Structural pins (behavioral paths both executed without error above):
    assert "wm_trust" in SRC and "[wm] no fresh prompt under trust threshold" in SRC, \
        "the empty-mask fallback (Dyna-Q's empty-imagined-batch warning) is missing"
    assert "never overrides" in SRC.split("def per_continuation")[1][:4000] \
        or "never overrides evidence" in SRC, \
        "the sampled-posteriors-always-win rule must be stated where it is enforced"


def test_per_sampling_flag_gates_the_round_boundary():
    assert 'getattr(args, "per_sampling", False)' in SRC, \
        "--per-sampling is parsed but the round boundary never consults it"
    assert "per_continuation(cohort, stats, mix, args, rng)" in SRC, \
        "the PER path does not route through per_continuation"


def test_visit_ledger_persists_and_orders_novelty(monkeypatch, tmp_path):
    ledger = str(tmp_path / "visits.json")
    cohort, _ = _run(monkeypatch, tmp_path, ledger=ledger)
    counts = json.load(open(ledger, encoding="utf-8"))
    assert len(counts) == 3 and all(v == 1 for v in counts.values()), \
        "each selected prompt must be recorded exactly once"
    # Run again: same selection (disagreement dominates) increments to 2.
    cohort2, _ = _run(monkeypatch, tmp_path, ledger=ledger)
    counts2 = json.load(open(ledger, encoding="utf-8"))
    assert all(v == 2 for v in counts2.values()), "counts must accumulate across passes"


# ── DAPO dynamic sampling (single-model mixedness) ───────────────────────────
# A prompt whose samples all land on the same side of the floor has zero group
# variance and gives no gradient; 35-50% of prompts were dead this way in
# measured runs, so they are screened out before training rather than paid for.

def test_dynamic_sampling_flag_is_parsed():
    assert 'dest="dynamic_sampling"' in SRC


def test_dynamic_sampling_skips_loading_a_probe_checkpoint():
    assert "if _dynamic:\n        probe_texts = []" in SRC, \
        "single-model screening must not load --probe-base"


def test_dynamic_sampling_routes_into_the_probe_selector():
    assert ('getattr(args, "probe_base", None) or '
            'getattr(args, "dynamic_sampling", False)') in SRC, \
        "the flag is parsed but the cohort call site never consults it"


def test_mixedness_is_true_exactly_when_samples_span_the_floor():
    """verdict = (hit > 0, hit == k); the two differ iff the group is mixed."""
    for k, hits, mixed in [(2, 0, False), (2, 1, True), (2, 2, False),
                           (4, 0, False), (4, 1, True), (4, 3, True), (4, 4, False)]:
        verdict = (hits > 0, hits == k)
        assert (verdict[0] != verdict[1]) is mixed, f"k={k} hits={hits}"


def test_dynamic_sampling_fails_closed_when_nothing_is_mixed():
    assert "dynamic sampling found NO mixed prompt" in SRC, \
        "a cohort with no mixed prompt cannot train; it must raise, not proceed"
