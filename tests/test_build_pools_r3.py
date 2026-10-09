"""Unit tests: src/rlv2/data/pools.py pure logic — allocation, JSON parsing, deixis gate.

These three are the parts that decide dataset *shape* and *validity* before any
model output is trusted:

  1. allocate()   — decides coverage. If it does not sum to the requested total,
                    or starves a chunk, the pools silently under-cover the 24
                    chunks and the stratification claim in §6.11 is false.
  2. parse_json() — the only thing standing between raw model text and an item
                    entering a pool. Must reject truncated/degenerate output.
  3. DEIXIS       — enforces R3 self-containment (§6.11 step 5). A question that
                    says "in the passage above" is unanswerable without context,
                    so it measures nothing at eval time.

The ollama/embedding paths are not exercised here — they are I/O, and the
generation loop is guarded by the receipt assertions in build_pools_r3.main().
"""
from __future__ import annotations
import os as _o, sys as _s

import collections
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlv2.data import pools as bp  # noqa: E402


# --------------------------------------------------------------------------
# allocate()
# --------------------------------------------------------------------------
def _chunks(lengths: list[int]) -> list[dict]:
    return [{"chunk_id": "c%02d" % i, "text": "x" * n} for i, n in enumerate(lengths)]


def test_allocate_sums_exactly_to_total():
    ch = _chunks([1000, 2000, 3000, 500, 1500])
    got = bp.allocate(ch, 60)
    assert sum(got.values()) == 60


def test_allocate_sums_exactly_on_real_shape():
    """24 chunks at the real char counts (9/8/7 split, 34,243 chars) -> exactly 200."""
    lengths = ([13811 // 9] * 9) + ([11125 // 8] * 8) + ([9307 // 7] * 7)
    got = bp.allocate(_chunks(lengths), 200)
    assert len(got) == 24
    assert sum(got.values()) == 200


def test_allocate_enforces_minimum_of_two():
    """A tiny chunk must still get 2 items, never 0 — otherwise it is uncovered."""
    ch = _chunks([10000, 10000, 10000, 5])
    got = bp.allocate(ch, 40)
    assert got["c03"] >= 2
    assert min(got.values()) >= 2


def test_allocate_is_proportional_to_content():
    """A chunk with 4x the content must not receive fewer items than a small one."""
    ch = _chunks([4000, 1000])
    got = bp.allocate(ch, 20)
    assert got["c00"] > got["c01"]


def test_allocate_handles_total_below_minimum_floor():
    """Requesting fewer items than 2*n_chunks must not hang, go negative, or overrun.

    The floor yields to the budget: 3 chunks / total=3 cannot give everyone 2.
    """
    ch = _chunks([100, 100, 100])
    got = bp.allocate(ch, 3)
    assert len(got) == 3
    assert sum(got.values()) == 3
    assert all(v >= 0 for v in got.values())


def test_allocate_never_overruns_the_real_half_budget():
    """REGRESSION: 24 chunks at --n 75 -> half = 37. The old hard floor of 2 forced
    48 and starved the pool built second (A=48+, B=6, one skill)."""
    lengths = ([13811 // 9] * 9) + ([11125 // 8] * 8) + ([9307 // 7] * 7)
    got = bp.allocate(_chunks(lengths), 37)
    assert len(got) == 24
    assert sum(got.values()) == 37, "over-allocated: %d" % sum(got.values())


def test_allocate_covers_every_chunk_when_budget_allows():
    """At a budget that can afford the floor, no chunk is left uncovered."""
    lengths = ([13811 // 9] * 9) + ([11125 // 8] * 8) + ([9307 // 7] * 7)
    got = bp.allocate(_chunks(lengths), 75)
    assert sum(got.values()) == 75
    assert min(got.values()) >= 2


# --------------------------------------------------------------------------
# parse_json()
# --------------------------------------------------------------------------
def test_parse_json_extracts_clean_object():
    s = '{"question": "What does bagging reduce in an ensemble?", "answer": "Variance."}'
    got = bp.parse_json(s)
    assert got == {
        "question": "What does bagging reduce in an ensemble?",
        "answer": "Variance.",
    }


def test_parse_json_strips_surrounding_prose():
    s = 'Sure! Here is the item:\n{"question": "Why does a random forest subsample features?", "answer": "To decorrelate the trees."}\nHope that helps.'
    got = bp.parse_json(s)
    assert got is not None
    assert got["question"].startswith("Why does a random forest")
    assert got["answer"] == "To decorrelate the trees."


def test_parse_json_recovers_from_raw_newlines_in_strings():
    s = '{"question": "What is the role of the learning rate in boosting?",\n "answer": "It shrinks\neach tree contribution."}'
    got = bp.parse_json(s)
    assert got is not None
    assert "learning rate" in got["question"]


def test_parse_json_rejects_non_json():
    assert bp.parse_json("I cannot comply with that request.") is None


def test_parse_json_rejects_truncated_question():
    """<15 chars of question is a degenerate generation, not an exam item."""
    assert bp.parse_json('{"question": "Why?", "answer": "Because it reduces variance."}') is None


def test_parse_json_rejects_empty_answer():
    assert bp.parse_json('{"question": "What does bagging reduce in an ensemble?", "answer": ""}') is None


def test_parse_json_rejects_missing_keys():
    assert bp.parse_json('{"q": "What does bagging reduce?", "a": "Variance."}') is None


# --------------------------------------------------------------------------
# DEIXIS gate
# --------------------------------------------------------------------------
CONTEXT_DEPENDENT = [
    "According to the passage, what does bagging reduce?",
    "In the text above, how are sample weights updated?",
    "What does the author say about out-of-bag error?",
    "Using this dataset, how many trees are grown?",
    "What does the excerpt claim about feature subsampling?",
    "How does the mechanism described above reduce variance?",
    "As described above, what is the meta-learner trained on?",
    "In the following section, which loss is minimised?",
    "Which of the algorithms listed above is most robust to outliers?",
    "How does the procedure outlined earlier avoid overfitting?",
    "What is the purpose of the parameter introduced previously?",
    "Why is the ensemble shown below more accurate than a single tree?",
    "What does the method discussed here assume about base learners?",
]


@pytest.mark.parametrize("q", CONTEXT_DEPENDENT)
def test_deixis_rejects_context_dependent_questions(q):
    assert bp.DEIXIS.search(q) is not None, "should have been rejected: %r" % q


SELF_CONTAINED = [
    "What does bagging reduce in an ensemble, bias or variance?",
    "Why does a random forest sample a subset of features at each split?",
    "How are sample weights updated between rounds of AdaBoost?",
    "What is out-of-bag error and why does it approximate held-out error?",
    "In gradient boosting, what does the learning rate control?",
    "Why must base learners in a stacking ensemble be diverse?",
]


@pytest.mark.parametrize("q", SELF_CONTAINED)
def test_deixis_allows_self_contained_questions(q):
    assert bp.DEIXIS.search(q) is None, "false positive on: %r" % q


# --------------------------------------------------------------------------
# load_chunks() provenance
# --------------------------------------------------------------------------
def test_load_chunks_assigns_stable_per_skill_chunk_ids(tmp_path, monkeypatch):
    rows = [
        {"skill": "trees-ensembles/random-forest", "text": "a"},
        {"skill": "trees-ensembles/random-forest", "text": "b"},
        {"skill": "trees-ensembles/boosted-tree", "text": "c"},
        {"skill": "clustering-dimred/other", "text": "ignored"},
    ]
    p = tmp_path / "chunks.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(bp, "CHUNKS", str(p))

    got = bp.load_chunks()
    assert len(got) == 3, "must drop skills outside TARGET_SKILLS"
    assert [c["chunk_id"] for c in got] == [
        "random-forest#01",
        "random-forest#02",
        "boosted-tree#01",
    ]


def test_pool_c_size_matches_eval_set_size():
    """Eval is one R3 item per skill — no more, no less.

    Phase C emits exactly one item per skill in TARGET_SKILLS, so POOL_C_SIZE
    is a derived quantity, not a free knob. Pinning it to the literal 3 alone
    would let the two drift apart silently; pinning the relationship is what
    actually catches a skill added without a matching eval decision.
    """
    # Pool C tracks EVAL_SKILLS, NOT the training corpus. They were the same list
    # until the training corpus was widened to relieve gate saturation; keying the
    # eval off TARGET_SKILLS would have demanded a new eval item per added skill.
    assert bp.POOL_C_SIZE == len(bp.EVAL_SKILLS)
    assert bp.POOL_C_SIZE == 3


def test_target_skills_stay_within_one_tight_cluster():
    """TRIPWIRE. The TRAINING corpus was widened 3 -> 16 skills on 2026-08-26 because
    the 3-skill corpus saturated against the 0.80 gate (§1.6's sanctioned remedy).
    Two invariants survive that widening and are what this guards: every skill stays
    inside the trees-ensembles family, and MCTS stays out — it is a search/planning
    algorithm, not an ensemble learner, so it would break the "tightly related" claim.
    """
    assert all(s.startswith("trees-ensembles/") for s in bp.TARGET_SKILLS)
    assert "trees-ensembles/monte-carlo-tree-search-mct" not in bp.TARGET_SKILLS
    assert len(set(bp.TARGET_SKILLS)) == len(bp.TARGET_SKILLS), "duplicate skill"


def test_eval_contamination_gate_not_relaxed():
    """TRIPWIRE. The 0.80 cosine gate is what keeps a training item from being a
    paraphrase of an eval question, which would silently inflate measured lift.
    Widening the cluster was the sanctioned way to buy headroom; relaxing this
    is not, and the generator and the converter must agree on the value.
    """
    import importlib.util
    import pathlib

    assert bp.COSINE_MAX == 0.80

    conv = pathlib.Path(__file__).resolve().parents[1] / "archive" / "sprawl" / "build_r3_datasets.py"
    src = conv.read_text(encoding="utf-8")
    assert "COSINE_MAX = 0.80" in src, "converter gate drifted from the generator gate"


def _sk_chunks(spec: list[tuple[str, str, str]]) -> list[dict]:
    return [{"skill": s, "chunk_id": cid, "text": t} for s, cid, t in spec]


def test_central_chunks_picks_chunk_nearest_skill_centroid(monkeypatch):
    """The eval item must probe the skill's core. Given three chunks on an
    arc, the geometric middle wins — not the first or the extremes."""
    import numpy as np

    vecs = {
        "edge_a": [1.0, 0.0],
        "middle": [0.7071067811865476, 0.7071067811865476],
        "edge_b": [0.0, 1.0],
    }
    monkeypatch.setattr(
        bp, "embed", lambda texts: np.array([vecs[t] for t in texts], dtype=float)
    )

    got = bp.central_chunks_per_skill(
        _sk_chunks([
            ("rf", "rf#01", "edge_a"),
            ("rf", "rf#02", "middle"),
            ("rf", "rf#03", "edge_b"),
        ])
    )
    assert set(got) == {"rf"}
    assert got["rf"]["chunk_id"] == "rf#02"


def test_central_chunks_returns_one_per_skill_and_skips_embed_for_singletons(monkeypatch):
    """One eval chunk per skill, and a skill with a single chunk short-circuits
    without an embedding round-trip."""
    import numpy as np

    calls: list[list[str]] = []

    def fake_embed(texts):
        calls.append(list(texts))
        return np.array([[1.0, 0.0]] * len(texts), dtype=float)

    monkeypatch.setattr(bp, "embed", fake_embed)

    got = bp.central_chunks_per_skill(
        _sk_chunks([
            ("rf", "rf#01", "a"),
            ("rf", "rf#02", "b"),
            ("solo", "solo#01", "c"),
        ])
    )
    assert sorted(got) == ["rf", "solo"]
    assert got["solo"]["chunk_id"] == "solo#01"
    assert calls == [["a", "b"]]


# --------------------------------------------------------------------------
# _chat() — endpoint miss vs. empty answer
#
# nvidia_llm.complete() returns None ONLY after exhausting its retries, and ""
# when the model genuinely answered with nothing. _chat used to collapse both to
# "" via `(out or "").strip()`. Because parse_json("") is also None, the caller
# then booked a dead endpoint as a *parse* failure — so a 404 on the model id
# looked like the model emitting garbage, and a three-hour run reported
# saturation over an empty pool. These tests pin the two apart.
# --------------------------------------------------------------------------
def _use_nvidia(monkeypatch, result):
    monkeypatch.setattr(bp, "GEN_BACKEND", "nvidia")

    def fake_complete(system, prompt, **kw):
        return result

    monkeypatch.setattr(bp.nvidia_llm, "complete", fake_complete)


def test_chat_raises_remote_miss_when_endpoint_returns_nothing(monkeypatch):
    _use_nvidia(monkeypatch, None)
    with pytest.raises(bp.RemoteMiss):
        bp._chat("prompt", "system")


def test_chat_returns_empty_string_for_a_successful_empty_answer(monkeypatch):
    """"" is a real answer; None is no answer. Only the latter is a miss."""
    _use_nvidia(monkeypatch, "   \n ")
    assert bp._chat("prompt", "system") == ""


def test_chat_strips_and_returns_content(monkeypatch):
    _use_nvidia(monkeypatch, '  {"question": "q"}  \n')
    assert bp._chat("prompt", "system") == '{"question": "q"}'


def test_a_miss_is_not_confusable_with_a_parse_failure(monkeypatch):
    """The exact collapse that hid the outage: parse_json("") is None, so coercing a
    miss to "" makes it indistinguishable from unusable model output."""
    assert bp.parse_json("") is None
    _use_nvidia(monkeypatch, None)
    with pytest.raises(bp.RemoteMiss):
        bp._chat("prompt", "system")


def test_stats_carries_a_dedicated_miss_counter():
    """The generation loop books misses separately from parse/deixis/cosine."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "src/rlv2/data/pools.py"), encoding="utf-8").read()
    assert '"miss": 0' in src
    assert 'stats["miss"] += 1' in src


# --------------------------------------------------------------------------
# style_exemplars() / _shape_counts() / resolved_gen_model()  (§6.11 steps 1, 6)
# --------------------------------------------------------------------------
def _eval_file(tmp_path, rows: list[dict]) -> str:
    p = tmp_path / "eval_canonical.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(p)


def _row(regime="R3", source="general", domain="Math", q="q", gold="g") -> dict:
    return {"regime": regime, "source": source, "domain": domain,
            "question": q, "prompt": q, "gold": gold}


def test_shape_counts_matches_r3_profile_at_k3():
    """Math/Code/Instruction = 1/1/1. STEM's 0.085 share loses the last remainder,
    so the default 3 exemplars must not spend a slot on the rarest category."""
    assert bp._shape_counts(3, {"Math", "Code", "Instruction", "STEM"}) == {
        "Math": 1, "Code": 1, "Instruction": 1}


def test_shape_counts_allocates_every_slot():
    """Largest-remainder must be exact — a dropped slot silently shrinks the few-shot."""
    for k in range(1, 13):
        assert sum(bp._shape_counts(k, {"Math", "Code", "Instruction", "STEM"}).values()) == k
    assert bp._shape_counts(6, {"Math", "Code", "Instruction", "STEM"}) == {
        "Math": 3, "Code": 1, "Instruction": 1, "STEM": 1}


def test_shape_counts_renormalises_over_present_domains_only():
    """With only Math+STEM in the pool the weights must renormalise over those two
    (0.229 vs 0.085 -> 2/1), not leave slots unfilled because Code is absent."""
    assert bp._shape_counts(3, {"Math", "STEM"}) == {"Math": 2, "STEM": 1}


def test_shape_counts_degenerate_inputs():
    assert bp._shape_counts(0, {"Math"}) == {}
    assert bp._shape_counts(3, set()) == {}


def test_style_exemplars_uses_general_not_domain(tmp_path, monkeypatch):
    """The regression that shaped all 78 prior items: the old selector took
    source=="domain" and its `or r3` fallback never fired, so the genuine
    VibeThinker R3 exemplars were never shown to the generator."""
    rows = [_row(source="domain", domain="Math", q="DOMAIN-MATH"),
            _row(source="general", domain="Math", q="GENERAL-MATH"),
            _row(source="general", domain="Code", q="GENERAL-CODE"),
            _row(source="general", domain="Instruction", q="GENERAL-INSTR")]
    monkeypatch.setattr(bp, "EXEMPLARS", _eval_file(tmp_path, rows))
    out = bp.style_exemplars()
    assert len(out) == 3
    joined = "\n".join(out)
    assert "DOMAIN-MATH" not in joined
    for q in ("GENERAL-MATH", "GENERAL-CODE", "GENERAL-INSTR"):
        assert q in joined


def test_style_exemplars_ignores_other_regimes(tmp_path, monkeypatch):
    rows = [_row(regime="R1", q="R1-Q"), _row(regime="R4", q="R4-Q"),
            _row(regime="R3", q="R3-Q")]
    monkeypatch.setattr(bp, "EXEMPLARS", _eval_file(tmp_path, rows))
    out = bp.style_exemplars()
    assert len(out) == 1
    assert "R3-Q" in out[0] and "R1-Q" not in out[0] and "R4-Q" not in out[0]


def test_style_exemplars_is_deterministic_on_the_real_eval():
    """Fixed seed: exemplar rotation must reproduce byte-for-byte across runs, or the
    generation prompt is unpinned and the pools stop being reproducible (§6.10)."""
    if not os.path.exists(bp.EXEMPLARS):
        pytest.skip("eval_canonical.jsonl not present")
    first = bp.style_exemplars()
    assert len(first) == 3
    assert first == bp.style_exemplars() == bp.style_exemplars()


def test_style_exemplars_carries_gold_and_caps_it(tmp_path, monkeypatch):
    """Real golds are multi-line derivations; uncapped they crowd the chunk out of the
    prompt. Question capped at 200, gold at EXEMPLAR_GOLD_CAP, both single-line."""
    rows = [_row(domain="Math", q="Q" * 400, gold="G" * 5000)]
    monkeypatch.setattr(bp, "EXEMPLARS", _eval_file(tmp_path, rows))
    out = bp.style_exemplars(k=1)
    assert len(out) == 1
    assert out[0] == "Q: %s\n    A: %s" % ("Q" * 200, "G" * bp.EXEMPLAR_GOLD_CAP)


def test_style_exemplars_collapses_whitespace(tmp_path, monkeypatch):
    """Newlines inside a gold would break the '  - ' bullet list in the prompt."""
    rows = [_row(q="a\n\nb   c", gold="x\ny\tz")]
    monkeypatch.setattr(bp, "EXEMPLARS", _eval_file(tmp_path, rows))
    assert bp.style_exemplars(k=1) == ["Q: a b c\n    A: x y z"]


def test_style_exemplars_omits_answer_line_when_gold_empty(tmp_path, monkeypatch):
    rows = [_row(q="only-q", gold="")]
    monkeypatch.setattr(bp, "EXEMPLARS", _eval_file(tmp_path, rows))
    assert bp.style_exemplars(k=1) == ["Q: only-q"]


def test_style_exemplars_tops_up_when_profile_underfills(tmp_path, monkeypatch):
    """A pool with one domain still has to return k items, not just its shape share."""
    rows = [_row(domain="Math", q="m%d" % i) for i in range(5)]
    monkeypatch.setattr(bp, "EXEMPLARS", _eval_file(tmp_path, rows))
    out = bp.style_exemplars(k=3)
    assert len(out) == 3
    assert len({o for o in out}) == 3  # no duplicated exemplar


def test_style_exemplars_empty_when_no_r3_general(tmp_path, monkeypatch):
    rows = [_row(regime="R1"), _row(source="domain")]
    monkeypatch.setattr(bp, "EXEMPLARS", _eval_file(tmp_path, rows))
    assert bp.style_exemplars() == []


def test_style_exemplars_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(bp, "EXEMPLARS", str(tmp_path / "nope.jsonl"))
    assert bp.style_exemplars() == []


def test_resolved_gen_model_reports_the_backend_that_actually_ran(monkeypatch):
    """All 78 prior records were stamped with the ollama fallback id while generation
    ran on the NVIDIA endpoint — the provenance field was decorative (§6.11 step 6)."""
    monkeypatch.setattr(bp, "GEN_BACKEND", "nvidia")
    monkeypatch.setattr(bp.nvidia_llm, "MODEL", "vendor/real-model")
    monkeypatch.setattr(bp, "GEN_MODEL", "qwen3.5-oc:4b")
    assert bp.resolved_gen_model() == "vendor/real-model"

    monkeypatch.setattr(bp, "GEN_BACKEND", "ollama")
    assert bp.resolved_gen_model() == "qwen3.5-oc:4b"


def test_record_and_receipt_use_resolved_gen_model():
    """Guard the write sites so the stamp cannot regress to a bare constant.

    Contract TIGHTENED 2026-08-26: generation now uses a DIFFERENT model per category, so
    a single global stamp is no longer even correct. The original bug this test was written
    for -- 78 items labelled with the ollama fallback id while generation actually ran on
    NVIDIA -- is the same failure this guards, one level more specific: the record must name
    the model that ACTUALLY produced it.
    """
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "src/rlv2/data/pools.py"), encoding="utf-8").read()
    # WIDENED 2026-08-29 for the OpenRouter backend. The stamp is now CONDITIONAL, and both
    # branches are required:
    #   ollama     -> CATEGORY_MODELS routing table IS the truth (a model per category)
    #   openrouter -> one model runs every category, so resolved_gen_model() IS the truth
    # Measured before this fix: two OpenRouter items were stamped with the ollama id, which
    # is the same class of failure this test was written for. What stays forbidden is an
    # UNCONDITIONAL global stamp, not resolved_gen_model() itself.
    assert 'CATEGORY_MODELS.get(category, CATEGORY_MODEL_FALLBACK)' in src, \
        "the per-category routing table must still be the stamp on the local backend"
    assert 'resolved_gen_model() if GEN_BACKEND != "ollama"' in src, \
        "a remote backend must stamp the model that actually ran, not the routing id"
    assert '"gen_model": GEN_MODEL' not in src
    assert '"gen_model": resolved_gen_model(),' not in src, \
        "an UNCONDITIONAL global stamp cannot be right when the local model varies per category"


def test_only_the_two_sanctioned_9b_models_are_used():
    """TRIPWIRE (owner, 2026-08-26): 'use ONLY qwen3.5-reasoning-thinking-256k:9b or
    qwopus-coding-thinking-256k:9b'. A silent downgrade to a smaller model mid-run would
    change what the dataset IS while every count and gate still reported green."""
    allowed = {"qwen3.5-reasoning-thinking-256k:9b", "qwopus-coding-thinking-256k:9b"}
    assert set(bp.CATEGORY_MODELS.values()) <= allowed, \
        "unsanctioned model: %s" % (set(bp.CATEGORY_MODELS.values()) - allowed)
    assert bp.CATEGORY_MODEL_FALLBACK in allowed, "fallback escapes the sanctioned set"
    assert set(bp.CATEGORY_MODELS) == set(bp.CATEGORY_ORDER), "a category has no model"


def test_closed_book_contract_is_in_the_batch_prompt():
    """R3 means closed-book: the question must carry its own setup and must never point at
    a source the reader cannot see. Both halves have to survive in the prompt text."""
    t = bp.BATCH_TEMPLATE
    assert "CLOSED-BOOK" in t
    assert "CARRY ITS OWN SETUP" in t
    for banned in ("the passage", "the author", "as described"):
        assert banned in t, "prompt no longer names %r as banned" % banned


# --------------------------------------------------------------------------
# plan_tasks() — the starvation fix
# --------------------------------------------------------------------------
# Written 2026-08-26. §6.11 claimed this function was "asserted by
# test_plan_tasks_*"; no such test existed, AND main() never called it — it still
# ran the old A-to-exhaustion loop. Both halves of the claim were false, which is
# why run2 produced A=54 / B=6-from-one-skill despite the fix being "in".
def _skill_chunks(per_skill: dict[str, int]) -> list[dict]:
    out = []
    for skill, n in per_skill.items():
        for i in range(n):
            out.append({
                "skill": "trees-ensembles/%s" % skill,
                "chunk_id": "%s#%02d" % (skill, i),
                "text": "x" * 1200,
            })
    return out


_REAL_SHAPE = {"ensemble-method": 9, "random-forest": 8, "boosted-tree": 7}


def test_plan_tasks_interleaves_pools_instead_of_draining_a_first():
    """THE regression. Sequential A-then-B spent the corpus's whole cosine
    headroom on A, so B started against a saturated gate and starved. An
    interleaved plan must never emit a long unbroken run of one pool."""
    plan = bp.plan_tasks(_skill_chunks(_REAL_SHAPE), 70)
    pools = [p for _, p, _ in plan]

    longest = best = 1
    for a, b in zip(pools, pools[1:]):
        best = best + 1 if a == b else 1
        longest = max(longest, best)

    assert longest <= 2, "pool runs of %d — plan is not interleaved" % longest
    assert pools[:4] == ["A", "B", "A", "B"]


def test_plan_tasks_gives_both_pools_every_skill():
    """run2's real damage was not the count but the composition: B's 6 items were
    all one skill, so the SFT set could not train the multi-skill claim at all."""
    plan = bp.plan_tasks(_skill_chunks(_REAL_SHAPE), 70)
    for pool in ("A", "B"):
        skills = {c["skill"] for c, p, _ in plan if p == pool}
        assert len(skills) == 3, "pool %s covers only %d skills" % (pool, len(skills))


def test_plan_tasks_splits_budget_between_pools():
    plan = bp.plan_tasks(_skill_chunks(_REAL_SHAPE), 70)
    counts = collections.Counter(p for _, p, _ in plan)
    assert sum(counts.values()) == 70
    assert abs(counts["A"] - counts["B"]) <= 1


def test_plan_tasks_rotates_category_within_a_chunk():
    """The pair most likely to collide on cosine is two items from the SAME chunk,
    so those must differ in flavour. A global counter would let one chunk draw
    Math twice while another never sees it."""
    plan = bp.plan_tasks(_skill_chunks(_REAL_SHAPE), 70)

    per_chunk = collections.defaultdict(list)
    for c, _, cat in plan:
        per_chunk[c["chunk_id"]].append(cat)

    repeated = [cid for cid, cats in per_chunk.items() if len(cats) > 1]
    assert repeated, "fixture must exercise chunks used more than once"
    for cid in repeated:
        cats = per_chunk[cid]
        assert all(a != b for a, b in zip(cats, cats[1:])), \
            "chunk %s repeats a category back-to-back: %s" % (cid, cats)
        # first four draws on a chunk must be four DIFFERENT flavours
        assert len(set(cats[:4])) == min(4, len(cats))


def test_plan_tasks_uses_every_category():
    plan = bp.plan_tasks(_skill_chunks(_REAL_SHAPE), 70)
    used = {cat for _, _, cat in plan}
    assert used == set(bp.CATEGORY_ORDER), "unused categories: %s" % (
        set(bp.CATEGORY_ORDER) - used)


def test_plan_tasks_categories_are_all_known_prompts():
    """Every category emitted must have prompt text, or emit() raises KeyError
    mid-run — after minutes of paid generation."""
    plan = bp.plan_tasks(_skill_chunks(_REAL_SHAPE), 70)
    for _, _, cat in plan:
        assert cat in bp.R3_CATEGORIES
        assert bp.R3_CATEGORIES[cat].strip()


def test_plan_tasks_handles_zero_budget():
    assert bp.plan_tasks(_skill_chunks(_REAL_SHAPE), 0) == []


def test_template_renders_with_category_slot():
    """TEMPLATE gained a {category} field; a missing key here is a KeyError at
    generation time, not import time."""
    out = bp.TEMPLATE.format(
        chunk="c", style="s", exclusion="e",
        category=bp.R3_CATEGORIES["Math"],
    )
    assert "CATEGORY = MATH" in out
    assert "SELF-CONTAINED" in out


# --------------------------------------------------------------------------
# Generation length cap (owner, 2026-08-26)
# --------------------------------------------------------------------------
# Cap derived from VibeThinker's own R3 gold lengths via a robust prediction
# interval: box-cox(lam=0.5) -> median + 2*MAD*1.4826 -> invert. Without ANY
# output bound, a 256k-context thinking GGUF generated past the 180s HTTP
# timeout on every call, so a 25-minute run wrote ZERO items at 100% GPU.
def test_answer_token_cap_matches_the_documented_derivation():
    """Recompute the cap from eval_canonical.jsonl and confirm the constant.

    This is the test that catches the constant silently drifting away from the
    data it was derived from — if the eval file changes, the cap must be
    re-derived, not assumed.
    """
    import statistics as st

    rows = [json.loads(l) for l in open(bp.EXEMPLARS, encoding="utf-8") if l.strip()]
    r3 = [r for r in rows
          if str(r.get("regime")) == "R3" and str(r.get("source")) == bp.EXEMPLAR_SOURCE]
    lengths = [max(1, round(len(str(r.get("gold") or "")) / 4)) for r in r3]
    assert lengths, "no R3/general golds to derive the cap from"

    lam = 0.5
    y = [(x ** lam - 1) / lam for x in lengths]
    med = st.median(y)
    mad = st.median([abs(v - med) for v in y])
    cap = ((med + 2 * mad * 1.4826) * lam + 1) ** (1 / lam)

    assert abs(bp.ANSWER_TOKEN_CAP - cap) <= 2, (
        "ANSWER_TOKEN_CAP=%s drifted from its derivation (%.1f)" % (bp.ANSWER_TOKEN_CAP, cap))


def test_num_predict_reserves_room_for_the_reasoning_trace():
    """Earned in run9. num_predict bounds TOTAL output, trace included. Budgeting only
    the answer cap let the <think> block eat the whole allowance, so the trace was cut
    mid-tag, strip_reasoning dropped the unterminated block, and nothing was left to
    parse -- Phase C gave up on the skill whose category routes to the chattier model.
    """
    assert bp.THINK_TOKEN_BUDGET >= 1024, "too small to hold a real reasoning trace"
    # a single-item call must fit trace AND answer, not just the answer
    assert bp.num_predict_for(1) >= bp.THINK_TOKEN_BUDGET + bp.ANSWER_TOKEN_CAP
    # the trace allowance is paid ONCE per call, not per item
    per_item = bp.num_predict_for(10) - bp.num_predict_for(9)
    assert per_item < bp.THINK_TOKEN_BUDGET, "trace budget must not scale with batch size"


def test_num_predict_scales_with_batch_and_is_never_unbounded():
    """The bug this guards: no num_predict at all -> unbounded generation -> every
    call exceeds the HTTP timeout -> silent retry loop -> zero items."""
    assert bp.num_predict_for(1) > bp.ANSWER_TOKEN_CAP, "one answer must fit"
    assert bp.num_predict_for(10) > bp.num_predict_for(3) > bp.num_predict_for(1)
    # degenerate inputs must still yield a positive, finite bound
    for bad in (0, -5):
        assert bp.num_predict_for(bad) == bp.num_predict_for(1) > 0


def test_ollama_payload_always_sets_num_predict():
    """Guard the call site itself: the bound has to reach the request body."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "src/rlv2/data/pools.py"), encoding="utf-8").read()
    assert '"num_predict": num_predict' in src
    assert '"num_predict": 4096' not in src, "hard-coded bound replaced the derived one"


def test_post_backoff_is_exponential():
    """Owner: 'if timeouts occur, use exponential decay'. Linear 1.5/3.0/4.5s barely
    backs off when the real cause is a GPU still chewing the previous request."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "src/rlv2/data/pools.py"), encoding="utf-8").read()
    assert "1.5 * (2 ** attempt)" in src
    assert "time.sleep(1.5 * (attempt + 1))" not in src, "reverted to linear backoff"


# --------------------------------------------------------------------------
# parse_batch() — reasoning-trace handling
# --------------------------------------------------------------------------
# These GGUFs emit <think>...</think> despite think:false (§6.11.1). The trace is
# full of braces and quoted fragments, so an unguarded object-scraper mistakes the
# model's scratch work for dataset items.
def test_parse_batch_strips_closed_reasoning_trace():
    raw = (
        "<think>Let me consider {\"question\": \"a draft I rejected\", \"answer\": \"no\"} "
        "and some other options.</think>\n"
        '[{"question": "Why does bagging reduce variance in an ensemble?", '
        '"answer": "Averaging decorrelated errors shrinks variance."}]'
    )
    got = bp.parse_batch(raw)
    assert len(got) == 1, "trace content leaked into the batch"
    assert got[0]["question"].startswith("Why does bagging")


def test_parse_batch_discards_truncated_reasoning_with_no_payload():
    """num_predict cutting the model off mid-<think> must yield NOTHING, not a
    half-formed item scraped out of the reasoning."""
    raw = ('<think>Maybe I will ask {"question": "half-formed idea here", '
           '"answer": "partial thought"} but actually')
    assert bp.parse_batch(raw) == []


def test_parse_batch_still_reads_a_plain_fenced_list():
    raw = ('```json\n[{"question": "In gradient boosting, what does the learning rate '
           'control?", "answer": "It scales each tree contribution."},\n'
           '{"question": "Why does a random forest subsample features at each split?", '
           '"answer": "To decorrelate the trees."}]\n```')
    got = bp.parse_batch(raw)
    assert len(got) == 2
    assert all(len(g["question"]) >= 15 and g["answer"] for g in got)


def test_parse_batch_applies_the_same_quality_bar_as_single_item():
    raw = ('[{"question": "Why?", "answer": "too short a question"},'
           ' {"question": "What does out-of-bag error estimate?", "answer": ""},'
           ' {"question": "What does out-of-bag error estimate in a random forest?",'
           '  "answer": "Held-out generalisation error."}]')
    got = bp.parse_batch(raw)
    assert len(got) == 1, "degenerate items must not enter a pool"
    assert got[0]["answer"] == "Held-out generalisation error."


# --------------------------------------------------------------------------
# strip_reasoning() — shared by BOTH parsers
# --------------------------------------------------------------------------
# Found live in run7: parse_json had no trace-stripping, so its greedy \{.*\}
# swallowed a <think> block containing braces, json.loads failed, all 6 tries
# burned as parse failures, and Phase C GAVE UP on ensemble-method — which would
# have shipped an eval missing a skill while every count still looked plausible.
def test_parse_json_survives_a_reasoning_trace_containing_braces():
    """THE run7 regression. Greedy brace-matching + a brace-laden trace = total
    parse failure, which the caller books as 'model wrote garbage'."""
    raw = (
        '<think>Options: {"question": "draft A"} or maybe {"answer": "draft B"}. '
        'I will go with a weighting question.</think>\n'
        '{"question": "In AdaBoost, how is the weight of a weak learner computed '
        'from its weighted error?", "answer": "alpha = 0.5 * ln((1-err)/err)."}'
    )
    got = bp.parse_json(raw)
    assert got is not None, "trace-wrapped JSON must still parse"
    assert got["question"].startswith("In AdaBoost")
    assert "alpha" in got["answer"]


def test_parse_json_rejects_truncated_reasoning_with_no_payload():
    raw = '<think>Let me draft {"question": "unfinished thought here", "answer": "maybe"}'
    assert bp.parse_json(raw) is None


def test_strip_reasoning_is_shared_by_both_parsers():
    """Both call sites must use the helper — a second inline copy is how the two
    parsers drifted apart in the first place."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "src/rlv2/data/pools.py"), encoding="utf-8").read()
    assert src.count("strip_reasoning(") >= 3, "helper not used by both parsers"
    assert src.count("re.sub(r\"<think>.*?</think>\"") == 1, "trace-stripping duplicated"


@pytest.mark.parametrize("raw,expected", [
    ("plain text, no trace", "plain text, no trace"),
    ("<think>reasoning</think>payload", "payload"),
    ("<THINK>upper</THINK>payload", "payload"),
    ("```json\n{\"a\": 1}\n```", '{"a": 1}'),
])
def test_strip_reasoning_shapes(raw, expected):
    assert bp.strip_reasoning(raw) == expected


def test_module_has_no_undefined_names():
    """Static undefined-name check on the generator.

    Earned 2026-08-26: an orphaned statement left behind when a loop was removed
    referenced a dead loop variable. `ast.parse` passed (it is valid SYNTAX), the
    unit tests passed (they do not exercise main()), and the NameError surfaced
    only at RUNTIME — after Phase C and six generated items, ~10 minutes of GPU
    time in. pyflakes catches it in milliseconds.
    """
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run(
        [sys.executable, "-m", "pyflakes", os.path.join(root, "src", "rlv2", "data", "pools.py")],
        capture_output=True, text=True,
    )
    bad = [ln for ln in (r.stdout + r.stderr).splitlines()
           if "undefined name" in ln or "local variable" in ln]
    assert not bad, "undefined names:\n" + "\n".join(bad)


# --------------------------------------------------------------------------
# DEIXIS — "the <arbitrary noun> above" leak
# --------------------------------------------------------------------------
# LEAKED A REAL ITEM 2026-08-26. The gate matched only a FIXED noun list after
# "the", so "In the boosting pseudocode above, ..." passed: 'pseudocode' is not
# in any list. That question is answerable only with the source in hand — the
# exact failure R3 exists to prevent — and it had already been accepted.
LEAKED_DEIXIS = [
    "In the boosting pseudocode above, what is the exact mathematical operation performed?",
    "In the algorithm above, which term is minimised?",
    "What does the code snippet below return?",
    "In the pseudocode shown above, how is alpha computed?",
    "According to the derivation above, what is the update rule?",
    "In the worked example below, how many trees are grown?",
]


@pytest.mark.parametrize("q", LEAKED_DEIXIS)
def test_deixis_rejects_arbitrary_noun_plus_locator(q):
    assert bp.DEIXIS.search(q) is not None, "deictic reference slipped the gate: %r" % q


# The negative lookahead must not swallow legitimate magnitude comparisons —
# over-blocking silently shrinks the pool and looks identical to saturation.
LEGIT_COMPARISONS = [
    "In AdaBoost, which samples receive weight above the mean after reweighting?",
    "Which samples have residuals below a chosen threshold?",
    "Why are trees with depth above a certain limit prone to overfitting?",
    "What happens to samples whose margin falls below its previous value?",
    "In gradient boosting, what does the learning rate control?",
    "What does out-of-bag error estimate in a random forest?",
]


@pytest.mark.parametrize("q", LEGIT_COMPARISONS)
def test_deixis_allows_magnitude_comparisons(q):
    assert bp.DEIXIS.search(q) is None, "false positive would shrink the pool: %r" % q


# --------------------------------------------------------------------------
# --resume  (durability)
# --------------------------------------------------------------------------
# Resume shipped with TWO runtime bugs on 2026-08-26, both invisible to the
# existing suite because they live in main():
#   1. the Phase C skip edit silently failed to apply, so a resumed run began
#      regenerating eval items — pool C would have reached 6, breaking the
#      frozen eval (§1.4/§1.6);
#   2. the stop condition compared CUMULATIVE accepted against REMAINING
#      budget, so a resume with 142 items and budget 98 exited immediately
#      having generated nothing.
def _src() -> str:
    return open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "src/rlv2/data/pools.py"), encoding="utf-8").read()


def test_resume_never_truncates_the_pool():
    """The generator opened its output "w" unconditionally, so every restart
    destroyed the pool. That cost 142 items once."""
    src = _src()
    assert 'open(args.out, "a" if resumed else "w"' in src, \
        "resume must APPEND; unconditional 'w' truncates the pool"


def test_resume_stop_condition_uses_the_total_not_the_remainder():
    """Unit mismatch regression. `budget` is remaining work; comparing
    cumulative accepted against it exits instantly on any resumed run."""
    src = _src()
    assert "len(accepted) - POOL_C_SIZE >= budget" not in src, \
        "stop condition compares cumulative items against REMAINING budget"
    assert "if len(accepted) >= args.n:" in src

    # demonstrate the arithmetic the bug got wrong
    accepted, target, pool_c = 142, 240, 3
    budget = target - accepted                      # 98 remaining
    assert (accepted - pool_c) >= budget            # the OLD check: fires immediately
    assert not (accepted >= target)                 # the NEW check: correctly continues


def test_pool_c_has_a_hard_cap_not_just_a_skip():
    """A guard that depends on another branch working is not a guard: the skip
    silently failed to apply once, and only a cap would have stopped it."""
    src = _src()
    assert "POOL C SIZE VIOLATION" in src
    assert "assert n_c == POOL_C_SIZE" in src
    assert "PHASE C already complete" in src, "resume must skip a completed Phase C"


def test_resume_refuses_to_append_onto_a_corrupt_file():
    """A zombie writer produced NUL-padded interleaved records once. Resuming
    onto that would bake corruption into the dataset."""
    src = _src()
    assert "REFUSING TO RESUME" in src


def test_resume_subtracts_already_satisfied_quotas():
    """Without this a resumed run re-generates work it already has, blowing
    past --n and unbalancing the per-(chunk, category) allocation."""
    src = _src()
    assert "jobs[key] = max(0, jobs[key] - done.get(key, 0))" in src


def test_generation_timeout_exceeds_the_measured_slow_model():
    """The rule that cost an hour: qwopus-coding-9b was MEASURED at 1451s and
    the timeout was set to 900s, so every Code call was guaranteed to time out,
    retry, and time out again — silently, with no error."""
    assert bp.GEN_TIMEOUT > 1451 * 2, \
        "GEN_TIMEOUT=%s is not comfortably above the measured 1451s" % bp.GEN_TIMEOUT
    assert "retries=1" in _src(), "a bounded generation must not be retried"


def test_receipt_reports_actual_models_not_the_global_fallback():
    """resolved_gen_model() returns ONE global id. With per-category models that is
    simply false, and it is the same provenance lie that mislabelled all 78 items in
    an earlier run. The receipt must count gen_model off the accepted records."""
    src = _src()
    assert 'a.get("gen_model", "?") for a in accepted' in src
    assert "model=%s" not in src, "receipt still stamps a single global model id"


def test_gen_index_is_derived_from_accepted_never_reset():
    """BEHAVIOURAL regression, not a string check.

    `gen_index = 0` sat AFTER the resume block and silently wiped the resumed
    counter, so a resumed run restarted ids at r3-0000 and appended FOUR records
    whose ids collided with existing ones. Ids are the only handle for tracing or
    excising an item, so duplicates corrupt provenance while every count still
    looks correct — which is why the earlier source-string tests missed it.
    """
    src = _src()
    assert "gen_index = (max(int(a.get(\"generation_index\", -1)) for a in accepted) + 1" in src
    assert "\n    gen_index = 0\n" not in src, "a bare reset still follows the resume block"

    # the arithmetic the bug got wrong
    accepted = [{"generation_index": i} for i in range(142)]
    derived = (max(int(a.get("generation_index", -1)) for a in accepted) + 1) if accepted else 0
    assert derived == 142, "resumed ids must continue, not restart"
    assert (0 if not accepted else derived) != 0


def test_pool_ids_are_unique_and_monotonic():
    """Guards the live artifact itself: duplicate ids already reached disk once."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "rl", "pools_r3.jsonl")
    if not os.path.exists(path):
        pytest.skip("no pool yet")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    ids = [r["id"] for r in rows]
    gi = [int(r["generation_index"]) for r in rows]
    assert len(set(ids)) == len(ids), "duplicate ids in the pool"
    assert len(set(gi)) == len(gi), "duplicate generation_index in the pool"
    assert gi == sorted(gi), "generation_index is not monotonic"


# --- deficit-driven generation quota (G-9) -------------------------------------------
# `want` used to come from the plan (~1 per chunk/category) while a batched call asked for
# 16, so ~15 of every 16 candidates were discarded ungated. `want` is now the REMAINING
# GLOBAL CATEGORY QUOTA, derived from `accepted` so a --resume run recomputes it exactly.

def test_category_quota_sums_and_shape():
    from rlv2.data import pools as bp
    assert bp.category_quota(748) == {"Math": 230, "Code": 179,
                                      "STEM": 167, "Instruction": 172}
    for n in (0, 1, 4, 7, 210, 748):
        q = bp.category_quota(n)
        assert sum(q.values()) == n, "quota must sum EXACTLY to n (%d)" % n
        assert set(q) == set(bp.CATEGORY_ORDER)
        assert all(v >= 0 for v in q.values())
    # Math carries the largest average weight, so it must never be under-allocated.
    q = bp.category_quota(748)
    assert q["Math"] == max(q.values())


def test_remaining_quota_derives_from_accepted():
    """The exact arithmetic main() runs; it must be a pure function of `accepted`."""
    from rlv2.data import pools as bp

    def remaining(n, accepted):
        q = bp.category_quota(n)
        have = collections.Counter(a.get("category") for a in accepted)
        return {c: max(0, q[c] - have.get(c, 0)) for c in bp.CATEGORY_ORDER}

    assert remaining(748, []) == {"Math": 230, "Code": 179,
                                  "STEM": 167, "Instruction": 172}
    acc = [{"category": "Math"}] * 230 + [{"category": "Code"}] * 9
    r = remaining(748, acc)
    assert r["Math"] == 0, "a met quota must be skipped, not generated and discarded"
    assert r["Code"] == 170
    assert r["STEM"] == 167 and r["Instruction"] == 172
    # over-shoot clamps at zero rather than going negative
    assert remaining(748, [{"category": "STEM"}] * 500)["STEM"] == 0


def test_deficit_ordering_asks_the_furthest_behind_first():
    """Normalised deficit, the same measure deficit_draw() uses: categories with
    different targets compete on one scale."""
    from rlv2.data import pools as bp
    q = bp.category_quota(748)
    have = {"Math": 200, "Code": 20, "STEM": 167, "Instruction": 100}

    def deficit(c):
        return (q[c] - have[c]) / max(q[c], 1e-9)

    order = sorted(bp.CATEGORY_ORDER, key=lambda c: -deficit(c))
    assert order[0] == "Code", "Code is furthest below quota (20/179)"
    assert order[-1] == "STEM", "STEM is full, so it sorts last and is dropped"
    assert deficit("STEM") == 0.0
    # Math has the biggest raw target but only a 30-item gap, so it must not outrank
    # Instruction's 72-item gap on a smaller target.
    assert deficit("Instruction") > deficit("Math")


# --- skill-spread scheduling (G-4) ----------------------------------------------------
# Category deficit alone leaves every job in a category tied, so file order decided and the
# scheduler drained one skill at a time: a 168-item run put 58 items (35%) through a single
# skill and never called the 29 newest chunks at all. The sort now carries a per-skill
# accepted count, and the tally advances as each job is PICKED so a single wave cannot be
# four chunks of the same skill either.

def _select_waves(jobs, chunk_skill, cat_deficit, model_of, accepted, workers):
    """Mirror of main()'s selection loop (the loop lives inside main(), which the suite
    does not call). Returns the waves in the order the scheduler would run them."""
    pending = list(jobs)
    picked = collections.Counter()
    waves = []
    while pending:
        have = collections.Counter(a.get("source_skill") for a in accepted)
        wave = []
        while pending and len(wave) < workers:
            pending.sort(key=lambda kv: (-cat_deficit(kv[1]),
                                         picked[chunk_skill[kv[0]]],
                                         have[chunk_skill[kv[0]]],
                                         model_of(kv[1])))
            kv = pending.pop(0)
            wave.append(kv)
            picked[chunk_skill[kv[0]]] += 1
        waves.append(wave)
    return waves


def _spread_fixture():
    """Three skills, six chunks each, all one category -- the shape that used to drain."""
    chunk_skill = {}
    jobs = []
    for skill in ("alpha", "beta", "gamma"):
        for i in range(6):
            cid = "%s-%02d" % (skill, i)
            chunk_skill[cid] = skill
            jobs.append((cid, "Math"))
    return jobs, chunk_skill


def test_scheduler_alternates_skills_instead_of_draining_one():
    jobs, chunk_skill = _spread_fixture()
    waves = _select_waves(jobs, chunk_skill, lambda c: 1.0, lambda c: "m",
                          accepted=[], workers=4)
    picked = [chunk_skill[cid] for w in waves for (cid, _) in w]
    assert len(picked) == 18, "no job may be dropped"
    # Round-robin: every block of three consecutive jobs hits all three skills exactly
    # once. The old scheduler produced six alphas before it touched beta at all.
    for i in range(0, 18, 3):
        block = picked[i:i + 3]
        assert sorted(block) == ["alpha", "beta", "gamma"],             "block %d drains a skill: %r (full order %r)" % (i // 3, block, picked)
    # Every skill is reached in the FIRST wave, not after the first one is drained.
    assert set(picked[:3]) == {"alpha", "beta", "gamma"}
    # ... which is exactly what file order, the old key, would NOT have done.
    assert [chunk_skill[cid] for (cid, _) in jobs][:6] == ["alpha"] * 6


def test_skill_spread_respects_category_deficit_first():
    """Spread is a TIE-BREAK. A category further below quota still outranks a fresh skill
    sitting in a category that is nearly full."""
    chunk_skill = {"c-hot": "overused", "c-cold": "untouched"}
    jobs = [("c-hot", "Code"), ("c-cold", "STEM")]
    deficits = {"Code": 0.9, "STEM": 0.05}
    acc = [{"source_skill": "overused"}] * 50
    waves = _select_waves(jobs, chunk_skill, lambda c: deficits[c], lambda c: "m",
                          accepted=acc, workers=1)
    assert waves[0][0][0] == "c-hot", "category deficit must stay the primary key"
    assert waves[1][0][0] == "c-cold"


def test_skill_tally_derives_from_accepted_so_resume_keeps_the_spread():
    """The counts come off `accepted` on every wave, never a cached counter, so a --resume
    run continues the spread it already has rather than restarting it."""
    jobs, chunk_skill = _spread_fixture()
    acc = [{"source_skill": "alpha"}] * 9 + [{"source_skill": "beta"}] * 4
    waves = _select_waves(jobs, chunk_skill, lambda c: 1.0, lambda c: "m",
                          accepted=acc, workers=3)
    first = [chunk_skill[cid] for (cid, _) in waves[0]]
    assert first[0] == "gamma", "the least-used skill leads on resume, not file order"
    assert first[1] == "beta", "then the next-least-used, not the one with 9 items"
    assert set(first) == {"alpha", "beta", "gamma"}, "one wave, three skills"


def test_skill_spread_is_stable_when_one_skill_is_all_that_is_left():
    """Degenerate case: a single skill left in the work list must still be scheduled, not
    starved by its own tally."""
    chunk_skill = {"only-%d" % i: "solo" for i in range(5)}
    jobs = [("only-%d" % i, "Math") for i in range(5)]
    waves = _select_waves(jobs, chunk_skill, lambda c: 1.0, lambda c: "m",
                          accepted=[], workers=2)
    picked = [cid for w in waves for (cid, _) in w]
    assert sorted(picked) == sorted(chunk_skill), "no job may be dropped"


# --------------------------------------------------------------------------
# SCHEMA_LEAK / CODE_DEIXIS — the two holes that passed 206 of 509 items
# --------------------------------------------------------------------------
# MEASURED 2026-08-29 on a real 509-item generation run. Two classes of unusable
# item cleared every existing gate:
#   1. schema leakage — the model answering questions ABOUT the JSON envelope
#      (`distinct_facts`, `fact_index`) instead of the domain;
#   2. code-reference deixis — questions about named locals that exist only in
#      the source passage (`skf`, `clfs`, `dataset_blend_test_j`).
# bp.DEIXIS caught NEITHER: it only knows classic prose deixis.

SCHEMA_LEAKED = [
    "What is the machine learning concept associated with the value 'Gini impurity' "
    "in the distinct_facts list?",
    "What is the year corresponding to the value stored in the first element of the "
    "distinct_facts list?",
    "Consider the string 'elderberry' found at index 4 of distinct_facts. What does it "
    "denote?",
    "Which fact_index corresponds to the bootstrap sampling statement?",
    "In the list of facts, which entry describes out-of-bag error?",
    "What is the value stored at index 2 of the facts array?",
]


@pytest.mark.parametrize("q", SCHEMA_LEAKED)
def test_schema_leak_is_rejected(q):
    assert bp.gate_reason(q, "banana") == "schema", "schema leakage slipped: %r" % q


def test_schema_leak_is_caught_in_the_answer_too():
    """The leak often lands only in the gold — a clean question with 'elderberry' for an
    answer is still an item about the envelope, not the domain."""
    assert bp.gate_reason("What does this term denote?",
                          "the value at index 3 of distinct_facts") == "schema"


CODE_REFERENTIAL = [
    "In the provided Python code, the variable `skf` is instantiated as "
    "StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed). What does it do?",
    "The outer loop iterates using `for j, clf in enumerate(clfs):`. What is its purpose?",
    "The code uses a temporary array dataset_blend_test_j. What does it hold?",
    "In the provided snippet, `clf.fit(X_train, y_train)` creates an ensemble. How many "
    "trees are fit?",
    "The code explicitly sets `rng = np.random.default_rng(0)`. What integer seed is used?",
    "The `GradientBoostingClassifier` in the `clfs` list is configured with specific "
    "hyperparameters. What is the exact value of max_depth?",
    "Consider the line `best_alpha = ccp_alphas[int(np.argmax(cv_scores))]`. Which alpha "
    "is chosen on a tie?",
    "In the evaluation print statements, which score is subject to overfitting bias?",
    "Within the loop, the line `dataset_blend_train[test, j] = clf.predict_proba(X[test])` "
    "assigns a prediction. What determines the row?",
    "Explain the role of the random_state parameter inside the cross-validation loop.",
]


@pytest.mark.parametrize("q", CODE_REFERENTIAL)
def test_code_reference_deixis_is_rejected(q):
    assert bp.gate_reason(q, "an answer") == "codedeixis", \
        "code-reference deixis slipped: %r" % q


@pytest.mark.parametrize("q", CODE_REFERENTIAL)
def test_the_incumbent_deixis_gate_missed_all_of_these(q):
    """Pins WHY this pattern had to be added — the old gate is not merely narrow here,
    it scores zero. If a future edit makes DEIXIS itself cover these, delete this test
    rather than weakening the sibling."""
    assert bp.DEIXIS.search(q) is None


# The discriminator is a DEFINITE reference to a passage-local name, NOT the mere
# presence of code. Over-rejecting silently deletes the whole Code category and looks
# identical to a saturated corpus, so these must keep passing.
LEGIT_CODE_QUESTIONS = [
    "Write a Python function that computes the weighted average of tree predictions.",
    "Write a Python list comprehension that returns the Gini impurity of each split.",
    "What condition must hold for a node to split, and which parameter caps tree depth?",
    "In gradient boosting, what does the learning rate control?",
    "Why does a random forest subsample features at each split?",
    "Write a function `select_features_randomly` that randomly selects `n_features_select` "
    "unique column indices from `n_features_all` features.",
    "Write a Python function `fit_weak_learner` that takes `X`, `y`, and `residuals` and "
    "instantiates a DecisionTreeRegressor.",
    "Suppose we modify `max_features` to 'log2' instead of 'sqrt'. How does this change "
    "the number of candidate features considered at each split?",
    "When instantiating a BaggingRegressor with max_depth=4 and n_estimators=50, what is "
    "the default behavior for bootstrapping?",
    "What is the numerical value of the Gini impurity for a node with 12 owners and 12 "
    "nonowners?",
]


@pytest.mark.parametrize("q", LEGIT_CODE_QUESTIONS)
def test_legitimate_code_questions_still_pass(q):
    assert bp.gate_reason(q, "an answer about trees") is None, \
        "false reject would gut the Code category: %r" % q


@pytest.mark.parametrize("q", SELF_CONTAINED + LEGIT_COMPARISONS)
def test_the_new_patterns_do_not_regress_the_old_accepts(q):
    assert bp.gate_reason(q, "an answer") is None


@pytest.mark.parametrize("q", CONTEXT_DEPENDENT + LEAKED_DEIXIS)
def test_classic_prose_deixis_is_still_booked_as_deixis(q):
    """Distinct stats keys are the point (the receipt must say which hole killed the
    item), so classic deixis must not be re-labelled by the new siblings."""
    assert bp.gate_reason(q, "an answer") == "deixis"


def test_stats_keys_exist_for_every_reason_gate_reason_returns():
    """A reason with no key in the stats dict is a KeyError at generation time, i.e. a
    crash 200 items into a paid run."""
    import inspect
    src = inspect.getsource(bp.main) if hasattr(bp, "main") else inspect.getsource(bp)
    for key in ("codedeixis", "schema"):
        assert '"%s": 0' % key in src, "stats dict never initialises %r" % key
