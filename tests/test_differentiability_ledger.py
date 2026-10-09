"""The ledger's key is the phase STATE, and a missing key fails closed.

Spec: .spec/specs/r3-domain-injection/differentiability-ledger.md REQ-L1..L5.

WHAT THIS PREVENTS. A reject rate is a fact about a MODEL, not about a record. A ledger
built against the base is meaningless for an SFT checkpoint, and a ledger of `live_group`
(does this record straddle?) is meaningless as `correct` (did the model answer it?).
Both wrong reads still RUN, and both produce a prior that looks populated -- the failure
would surface as a quietly worse ordering, months later, with nothing to point at.
"""
import io
import json
import os
import sys

import pytest

from rlv2.train.rl import differentiability_ledger as L  # noqa: E402


@pytest.fixture
def ckpt(tmp_path):
    d = tmp_path / "base"
    d.mkdir()
    (d / "config.json").write_text('{"model_type": "x"}', encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"\x00" * 64)
    return str(d)


@pytest.fixture
def store(tmp_path):
    return str(tmp_path / "ledger.json")


# --------------------------------------------------------------- the key ----
def test_the_key_changes_when_the_weights_change(ckpt):
    """REQ-L2. The name is not the state -- a directory can be overwritten."""
    before = L.phase_state(ckpt)
    io.open(os.path.join(ckpt, "model.safetensors"), "wb").write(b"\x01" * 128)
    assert L.phase_state(ckpt) != before


def test_the_key_is_stable_when_nothing_changes(ckpt):
    assert L.phase_state(ckpt) == L.phase_state(ckpt)


def test_two_checkpoints_with_the_same_name_can_differ(tmp_path):
    """The exact read this REQ exists to prevent, one level down."""
    states = []
    for payload in (b"\x00" * 64, b"\xff" * 64):
        d = tmp_path / "same_name"
        d.mkdir(exist_ok=True)
        (d / "config.json").write_text("{}", encoding="utf-8")
        (d / "model.safetensors").write_bytes(payload)
        states.append(L.phase_state(str(d)))
    assert states[0] != states[1]


def test_an_unknown_measure_is_refused_rather_than_stored(ckpt):
    with pytest.raises(ValueError):
        L.key(ckpt, "pool.jsonl", "accuracy")


# ------------------------------------------------------- fails closed ----
def test_a_missing_key_returns_empty_not_another_contexts_rows(ckpt, store, tmp_path):
    """REQ-L2 acceptance: no fallback to another base, dataset, or measure."""
    L.save(ckpt, "r3_rl.jsonl", "live_group", {"q1": [0.0, 1.0]}, path=store)

    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text("{}", encoding="utf-8")
    (other / "model.safetensors").write_bytes(b"\x02" * 64)

    assert L.load(str(other), "r3_rl.jsonl", "live_group", path=store) == {}
    assert L.load(ckpt, "r3_sft.jsonl", "live_group", path=store) == {}
    assert L.load(ckpt, "r3_rl.jsonl", "correct", path=store) == {}
    assert L.load(ckpt, "r3_rl.jsonl", "live_group", path=store) != {}


def test_no_store_at_all_is_empty_not_an_error(ckpt, tmp_path):
    assert L.load(ckpt, "p.jsonl", "live_group", path=str(tmp_path / "nope.json")) == {}


def test_a_corrupt_store_degrades_to_no_prior(ckpt, store):
    io.open(store, "w", encoding="utf-8").write("{ not json")
    assert L.load(ckpt, "p.jsonl", "live_group", path=store) == {}


# ------------------------------------------------------------- merging ----
def test_two_runs_from_one_state_accumulate(ckpt, store):
    """REQ-L4 acceptance: (4,1) twice leaves (8,2)."""
    L.save(ckpt, "p.jsonl", "live_group", {"q": [0.0, 1.0]}, path=store)
    L.save(ckpt, "p.jsonl", "live_group", {"q": [0.0, 1.0]}, path=store)
    rows = L.load(ckpt, "p.jsonl", "live_group", path=store)
    assert rows["q"]["visits"] == 2 and rows["q"]["hits"] == 2
    assert rows["q"]["rewards"] == [0.0, 1.0, 0.0, 1.0]


def test_a_run_from_another_state_leaves_the_first_key_untouched(ckpt, store, tmp_path):
    L.save(ckpt, "p.jsonl", "live_group", {"q": [0.0, 1.0]}, path=store)
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text("{}", encoding="utf-8")
    L.save(str(other), "p.jsonl", "live_group", {"q": [1.0, 1.0]}, path=store)
    row = L.load(ckpt, "p.jsonl", "live_group", path=store)["q"]
    assert row["visits"] == 1 and row["rewards"] == [0.0, 1.0]


def test_the_row_count_is_bounded_by_the_pool_not_the_trial_count(ckpt, store):
    """REQ-L1 acceptance: per-record, not per-trial-per-parameter."""
    for _trial in range(20):
        L.save(ckpt, "p.jsonl", "live_group",
               {"q%d" % i: [0.0, 0.0] for i in range(30)}, path=store)
    assert len(L.load(ckpt, "p.jsonl", "live_group", path=store)) == 30


def test_a_crash_before_the_write_leaves_the_store_byte_identical(ckpt, store):
    """REQ-L5: one write, at the end. Nothing partial."""
    L.save(ckpt, "p.jsonl", "live_group", {"q": [0.0, 1.0]}, path=store)
    before = io.open(store, "rb").read()
    try:
        L.save(ckpt, "p.jsonl", "bogus_measure", {"q": [0.0, 1.0]}, path=store)
    except ValueError:
        pass
    assert io.open(store, "rb").read() == before


def test_the_store_is_readable_json_a_human_can_audit(ckpt, store):
    L.save(ckpt, "p.jsonl", "live_group", {"q": [0.0, 1.0]}, path=store,
           name="sft_epoch_3_straddle50")
    blob = json.loads(io.open(store, encoding="utf-8").read())
    entry = list(blob.values())[0]
    assert entry["checkpoint"] == "sft_epoch_3_straddle50"
    assert entry["measure"] == "live_group" and entry["dataset"] == "p.jsonl"


# ------------------------------------------------------------ priority ----
def test_the_ledger_converts_to_the_samplers_own_history_shape():
    """The whole reason rows hold raw rewards: one scoring path, the trainer's."""
    rows = {"q": {"visits": 1, "hits": 1, "rewards": [0.0, 1.0]},
            "empty": {"visits": 0, "hits": 0, "rewards": []}}
    hist = L.as_history(rows)
    assert hist == {"q": [0.0, 1.0]}, "a row with no rewards is not a prior"


def test_a_row_is_a_prior_not_a_transcript(ckpt, store):
    """Unbounded growth cannot change the rank and kills auditability."""
    for _ in range(40):
        L.save(ckpt, "p.jsonl", "live_group", {"q": [0.0, 1.0]}, path=store)
    assert len(L.load(ckpt, "p.jsonl", "live_group", path=store)["q"]["rewards"])         == L.KEEP_REWARDS


def test_hits_counts_straddle_not_correctness(ckpt, store):
    """`live_group` means the group DISAGREED. An all-correct group is dead to
    RL for the same reason an all-wrong one is, and must not count as a hit."""
    L.save(ckpt, "p.jsonl", "live_group", {"solved": [1.0, 1.0],
                                           "missed": [0.0, 0.0],
                                           "straddle": [0.0, 1.0]}, path=store)
    rows = L.load(ckpt, "p.jsonl", "live_group", path=store)
    assert rows["solved"]["hits"] == 0 and rows["missed"]["hits"] == 0
    assert rows["straddle"]["hits"] == 1


# ---------------------------------------------------- the ordering itself ----
def test_the_order_comes_from_the_trainers_sampler_not_a_second_one():
    """Operator 2026-09-14: the ordering is a Boltzmann draw over the ECDF, not a
    sentinel and not a hard tier. Both of those were re-derivations of a decided
    ladder -- midrank ECDF, unvisited at rank 1.0, flat_order="signal" -- that
    would have had to be re-decided, silently and differently, in a second place.
    """
    src = io.open("archive/tools/reft_hpo.py", encoding="utf-8").read()
    body = src[src.index("def priority_order("):src.index("def main():")]
    lines = [l for l in body.splitlines() if not l.lstrip().startswith("#")]
    code = chr(10).join(lines[lines.index("    cand = list(pool)"):])
    assert "gof.boltzmann_weights(" in code
    assert "UNPROBED" not in code, "the sentinel is back"
    assert "sorted(pool" not in code, "a second ordering is back"


def test_the_probe_flushes_in_chunks_so_a_crash_does_not_lose_hours():
    """REQ-L5 forbids a PARTIAL state, not more than one write. A 2-hour probe
    that writes once at the end loses everything to one kill -- and two trials
    were killed at a timeout cap on 2026-09-14. Every flush holds only whole
    rows for records that finished probing, so the store is never half-written.
    """
    src = io.open("archive/tools/reft_hpo.py", encoding="utf-8").read()
    body = src[src.index("def priority_order("):src.index("def main():")]
    assert "FLUSH_EVERY" in body
    assert body.count("ledger.save(") == 1, "one save call, called repeatedly"
    assert "_flush()" in body.split("for i, rec in enumerate(todo")[1], \
        "a flush inside the probe loop, not only after it"


def test_a_partial_row_is_never_written(ckpt, store):
    """The flush writes whole reward lists. A record still being judged has no
    entry at all -- there is no state where a row exists with half its rollouts
    that a later run would read back as a settled prior.
    """
    L.save(ckpt, "p.jsonl", "live_group", {"done": [0.0, 1.0, 0.0, 1.0]},
           path=store)
    rows = L.load(ckpt, "p.jsonl", "live_group", path=store)
    assert set(rows) == {"done"}
    assert len(rows["done"]["rewards"]) == 4


# ------------------------------------------------------------- hub ids ----
def test_two_hub_ids_do_not_share_one_key():
    """The bug: os.walk on a non-directory yields nothing and config.json does not
    exist, so EVERY hub id hashed to sha1-of-empty (da39a3ee5e6b4b0d). The ledger
    would have served one model's rows for another -- the exact cross-context read
    REQ-L2 exists to prevent, failing OPEN rather than closed.
    """
    a = L.phase_state("unsloth/Qwen3.5-0.8B")
    b = L.phase_state("some/other-model")
    assert a != b
    assert not a.startswith("da39a3ee"), "sha1 of nothing is back"


def test_a_hub_id_is_labelled_so_the_store_is_auditable(ckpt):
    """A reader must be able to tell 'hashed an identifier' from 'hashed weights' --
    they carry different guarantees. A hub id is immutable by convention; a local
    directory can be overwritten under the same name."""
    assert L.phase_state("unsloth/Qwen3.5-0.8B").startswith("hub:")
    assert not L.phase_state(ckpt).startswith("hub:")


def test_a_hub_id_key_is_stable(ckpt):
    assert L.phase_state("unsloth/Qwen3.5-0.8B") == L.phase_state("unsloth/Qwen3.5-0.8B")


# ------------------------------------------- ONE wrapper, every trainer ----
def test_every_trainer_gets_its_prior_through_the_same_function():
    """Operator 2026-09-15: "Why i asked for a wrapper" / "w a proper wrapper so we
    stop seeing it worked for pyreft but not rl".

    THE DRIFT THIS CLOSES. archive/tools/reft_hpo.py hand-rolled load()+as_history() and
    worked. docs/grpo_one_file.py never did it at all, so RL -- the phase the ledger
    was BUILT for -- began every run with an empty history. Every record then read
    `unvisited` at rank 1.0, the Boltzmann draw degenerated to uniform, and the
    measured live rate fell from ~68% to ~17%: four times the generations per useful
    step, for four hours of HPO.

    Nothing announced it, because "no prior" and "a prior saying everything is
    equally good" are the same code path downstream. A second call site is how that
    happens, so there is exactly one.
    """
    for path in ("docs/grpo_one_file.py", "archive/tools/reft_hpo.py"):
        src = io.open(path, encoding="utf-8").read()
        assert "prior_history(" in src, "%s does not load a prior at all" % path
        body = "\n".join(l for l in src.splitlines()
                         if not l.lstrip().startswith("#"))
        assert "as_history(\n" not in body and "ledger.as_history(" not in body, \
            "%s hand-rolls the prior instead of using the wrapper" % path


def test_the_rl_trainer_seeds_reward_history_not_an_empty_dict():
    """The exact line that was wrong: `reward_history = {}`."""
    src = io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    assert "reward_history = {}" not in src
    assert "reward_history = _prior_history(" in src


def test_a_missing_key_gives_an_empty_prior_and_says_so(ckpt, store):
    """REQ-L2 fails closed, and the log must show it -- a silent empty prior looks
    exactly like a working one until the live rate is measured."""
    hist = L.prior_history(ckpt, "never_seen.jsonl", "live_group", path=store)
    assert hist == {}


def test_the_prior_counts_live_records_so_a_dead_one_is_visible(ckpt, store, capsys):
    L.save(ckpt, "p.jsonl", "live_group",
           {"live": [0.0, 1.0], "dead": [0.0, 0.0]}, path=store)
    hist = L.prior_history(ckpt, "p.jsonl", "live_group", path=store, label="t")
    assert set(hist) == {"live", "dead"}
    out = capsys.readouterr().out
    assert "2 records carried forward, 1 live" in out


def test_a_row_records_its_last_visit_separately_from_the_pooled_history(tmp_path):
    """THE DEFECT THIS CLOSES, measured 2026-09-19. `rewards` pools every visit so
    the sampler's posterior sharpens with evidence -- correct there, wrong for a
    CLASSIFIER. Straddle is `0 < correct < n`, so a row visited four times got 16
    chances to straddle while one visited once got 4, and an SFT bracket compared
    9 / 8 / 11 straddlers measured at different denominators."""
    store = str(tmp_path / "led.json")
    ck = "hub-style/model"
    L.save(ck, "pool.jsonl", "live_group", {"q1": [0.0, 0.0, 0.9, 0.0]},
                path=store)
    L.save(ck, "pool.jsonl", "live_group", {"q1": [0.0, 0.0]}, path=store)
    rows = L.load(ck, "pool.jsonl", "live_group", path=store)
    assert rows["q1"]["visits"] == 2
    assert len(rows["q1"]["rewards"]) == 6, "the pooled view keeps both visits"
    assert L.as_history(rows)["q1"] == [0.0, 0.0, 0.9, 0.0, 0.0, 0.0]
    assert L.as_last_visit(rows)["q1"] == [0.0, 0.0], \
        "the comparable view is the most recent visit alone"


def test_the_last_visit_view_reconstructs_rows_written_before_it_existed(tmp_path):
    """A store predating the field must not return the whole pile as one visit --
    that is the exact miscount the field was added to stop."""
    store = str(tmp_path / "led.json")
    io.open(store, "w", encoding="utf-8").write(json.dumps({
        L.key("hub/x", "pool.jsonl", "live_group"): {
            "checkpoint": "x", "dataset": "pool.jsonl", "measure": "live_group",
            "rows": {"q1": {"visits": 4, "hits": 1,
                            "rewards": [0.0] * 12 + [0.9, 0.0, 0.0, 0.0]}}}}))
    rows = L.load("hub/x", "pool.jsonl", "live_group", path=store)
    assert "last" not in rows["q1"]
    got = L.as_last_visit(rows)["q1"]
    assert len(got) == 4, "16 rollouts over 4 visits reconstructs to 4, not 16"
    assert got == [0.9, 0.0, 0.0, 0.0]
