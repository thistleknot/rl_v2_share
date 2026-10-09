"""Persist what the replay buffer learns per record, keyed by phase state.

Spec: .spec/specs/r3-domain-injection/differentiability-ledger.md REQ-L1..L5.
Task: playbook.md T5.

WHY IT EXISTS, measured 2026-09-14. The pyreft phase wrapper runs 5 HPO arms from ONE
base checkpoint over ONE pool. Every arm ran the identical priority probe -- same seed,
same records, same frozen weights, since rank/layers/lr do not exist at probe time --
at ~32 minutes each. 2.7 hours of judging to compute the same ordering five times.

The probe is a fact about the MODEL, not about the arm. Persist it once.

THE KEY IS THE PHASE STATE, NOT THE CHECKPOINT NAME (REQ-L2). A directory can be
overwritten; two experiments can leave different weights under one name. The key is
sha1 over config.json plus the (name, size, mtime) of every weight shard -- no tensor
is read, and it changes whenever the weights do.

MISSING KEY FAILS CLOSED. No rows means no prior, and the run probes from scratch. It
never falls back to another base, another dataset, or another measure.
"""
from __future__ import annotations

import hashlib
import io
import json
import os

STORE = "data/differentiability_ledger.json"

# A row is a prior, not a transcript. Past this many rollouts the extra history
# cannot change the record's rank, and the store stops being auditable by eye.
KEEP_REWARDS = 16


def _live(rewards, floor=0.7):
    """Did this group straddle -- the only outcome RL can use."""
    above = sum(1 for x in rewards if float(x) >= floor)
    return 0 < above < len(rewards)

# What a row MEANS. RL has rollouts and can ask about within-group spread; SFT has none
# and can only ask about correctness. Sharing one column would average two different
# quantities into a number that looks populated and means nothing (REQ-L2).
MEASURES = ("live_group", "correct")


def phase_state(checkpoint):
    """sha1 of the weights' identity -- content, not path (REQ-L2).

    A HUB ID IS NOT A DIRECTORY, and the naive version returned sha1 of nothing for
    every one of them -- `unsloth/Qwen3.5-0.8B` and `some/other-model` both hashed to
    da39a3ee5e6b4b0d, so the ledger would serve one model's rows for another. That is
    precisely the cross-context read this function exists to make impossible, and it
    fails OPEN rather than closed, which is worse.

    A hub id is immutable by convention (a revision pins content), so hashing the
    identifier is the honest key for it. Local directories still hash their bytes,
    because a directory CAN be overwritten under the same name.
    """
    if not os.path.isdir(checkpoint):
        return "hub:" + hashlib.sha1(
            str(checkpoint).encode("utf-8")).hexdigest()[:12]
    h = hashlib.sha1()
    cfg = os.path.join(checkpoint, "config.json")
    if os.path.exists(cfg):
        h.update(io.open(cfg, "rb").read())
    shards = []
    for root, _dirs, files in os.walk(checkpoint):
        for f in files:
            if f.endswith((".safetensors", ".bin", ".pt")):
                fp = os.path.join(root, f)
                shards.append((os.path.relpath(fp, checkpoint), fp,
                               os.stat(fp).st_size))
    # SIZE AND MTIME ALONE COLLIDE. Two checkpoints written in the same second with
    # the same shard layout -- exactly what a sweep produces -- hash identical, and
    # the ledger then serves one model's rows for another. Read the head and tail of
    # each shard instead: 8 KB per file, no tensor materialised, and the bytes that
    # differ between two trainings of the same architecture are in there.
    EDGE = 4096
    for name, fp, size in sorted(shards):
        h.update(("%s|%d" % (name, size)).encode("utf-8"))
        with io.open(fp, "rb") as fh:
            h.update(fh.read(EDGE))
            if size > EDGE:
                fh.seek(max(EDGE, size - EDGE))
                h.update(fh.read(EDGE))
    return h.hexdigest()[:16]


def key(checkpoint, dataset, measure):
    if measure not in MEASURES:
        raise ValueError("unknown measure %r; one of %s" % (measure, MEASURES))
    k = "%s|%s|%s" % (phase_state(checkpoint), os.path.basename(dataset), measure)
    # THE JUDGE IS PART OF WHAT A ROW MEANS (2026-09-26). The key named the artifact, not
    # the instrument that scored it, so after the switch to Jev every straddle probe would
    # have read the 9B's rewards back as fresh Jev measurements. 9B-era keys are unchanged.
    from rlv2.judge.reward import _jev_enabled
    if _jev_enabled():
        k += "|judge=open-jev-2b+gate1"   # gate1 = reward._jev_gate (2026-09-26); ungated rows keep the old key
    return k


def _load_store(path=STORE):
    if not os.path.exists(path):
        return {}
    try:
        return json.loads(io.open(path, encoding="utf-8").read())
    except (ValueError, OSError):
        return {}


def load(checkpoint, dataset, measure, path=STORE):
    """Rows for exactly this key, or {} -- never another context's (REQ-L2).

    A row is {question: {"visits": n, "hits": k, "rewards": [...]}}.

    THE REWARDS ARE THE POINT. The trainer's own sampler (grpo_one_file
    boltzmann_weights) takes a `history` of raw per-rollout rewards and derives
    spread, mean, the flat-record tiebreak, and the midrank ECDF from it. Storing
    a pre-chewed spread would force a second, divergent scoring path -- exactly
    the duplication that produced a hand-rolled ordering beside the incumbent
    sampler on 2026-09-14.
    """
    store = _load_store(path)
    entry = store.get(key(checkpoint, dataset, measure), {})
    return dict(entry.get("rows", {}))


def save(checkpoint, dataset, measure, rows, path=STORE, name=None):
    """ONE write, at the end (REQ-L5). Merges into the same key (REQ-L4).

    `rows` is {question: [reward, ...]} observed in THIS run; the rewards are
    appended to whatever the key already holds and visits/hits recomputed. A run
    that crashes before this call leaves the store byte-identical.

    KEEP_REWARDS bounds a row: the ledger is a prior, not a transcript, and an
    unbounded list would grow without ever changing the rank it produces.
    """
    store = _load_store(path)
    k = key(checkpoint, dataset, measure)
    entry = store.setdefault(k, {"checkpoint": name or str(checkpoint),
                                 "dataset": os.path.basename(dataset),
                                 "measure": measure, "rows": {}})
    have = entry["rows"]
    for q, observed in rows.items():
        obs = [float(x) for x in observed]
        prev = have.get(q) or {"visits": 0, "hits": 0, "rewards": []}
        kept = (list(prev.get("rewards", [])) + obs)[-KEEP_REWARDS:]
        have[q] = {"visits": int(prev.get("visits", 0)) + 1,
                   "hits": int(prev.get("hits", 0)) + (1 if _live(obs) else 0),
                   "rewards": kept,
                   # THE LAST VISIT, KEPT SEPARATELY. `rewards` accumulates across
                   # visits on purpose -- the sampler's posterior should sharpen with
                   # evidence. A CLASSIFIER must not read it: straddle is
                   # `0 < correct < n`, so a row visited four times gets 16 chances
                   # to straddle while one visited once gets 4, and the counts stop
                   # being comparable. Measured 2026-09-19 across one SFT bracket:
                   # epoch 1 rows sat at 16 rollouts, epoch 3 rows at 4-8, and the
                   # straddle column (9 / 8 / 11) was comparing different denominators.
                   #
                   # This is the visit boundary the flat list cannot express. Readers
                   # that want a comparable classification take `last`; readers that
                   # want the full posterior take `rewards`.
                   "last": obs}
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    io.open(tmp, "w", encoding="utf-8").write(json.dumps(store, indent=1))
    os.replace(tmp, path)
    return len(have)


def as_history(rows):
    """Ledger rows -> the `history` shape grpo_one_file.visit_stats consumes.

    One conversion, here, so the sampler stays the single scoring path.
    """
    return {q: list(r.get("rewards", [])) for q, r in rows.items()
            if r.get("rewards")}


def as_last_visit(rows):
    """Ledger rows -> {question: rewards from the MOST RECENT visit only}.

    THE COMPARABLE VIEW. Use this wherever a row is CLASSIFIED (straddle / solved /
    missed) or scored for available gradient, because both are functions of k out of
    n and n has to be the same across everything being compared.

    Rows written before `last` existed fall back to the tail of `rewards`, sized to
    the visit count -- the best reconstruction available, and it is marked by
    returning fewer than a full visit rather than by guessing a boundary.
    """
    out = {}
    for q, r in rows.items():
        last = r.get("last")
        if not last:
            rw = list(r.get("rewards", []))
            visits = max(1, int(r.get("visits", 1)))
            last = rw[-max(1, len(rw) // visits):] if rw else []
        if last:
            out[q] = [float(x) for x in last]
    return out


def prior_history(checkpoint, dataset, measure="live_group", path=STORE, label=""):
    """THE ONE WAY A TRAINER GETS ITS PRIOR. Call this; do not hand-roll load+as_history.

    Spec: differentiability-ledger.md REQ-L3 -- "Loaded rows seed `reward_history`
    before round 1, so the first Boltzmann draw has posteriors instead of every
    record reading `new`."

    WHY THIS IS A FUNCTION AND NOT TWO CALL SITES. Operator 2026-09-15: "Why i asked
    for a wrapper". `archive/tools/reft_hpo.py` hand-rolled `load()` + `as_history()` and
    worked; `docs/grpo_one_file.py` never did it at all, so RL -- the phase the
    ledger was BUILT for -- started every run with an empty history. Every record
    then read `unvisited` at rank 1.0, the Boltzmann draw degenerated to uniform,
    and the measured live rate fell from ~68% to ~17%: four times the generations
    per useful step. Nothing announced it, because "no prior" and "a prior that
    says everything is equally good" produce the same code path.

    Returns {question: [rewards]}, empty on a missing key (REQ-L2 fails closed).
    Always prints what it found, so a silent empty prior is visible in the log.
    """
    try:
        rows = load(checkpoint, dataset, measure, path=path)
    except Exception as exc:                 # a bad store must never kill a run
        print("[ledger] %sunreadable (%s) -- starting with no prior"
              % (label and label + ": ", exc), flush=True)
        return {}
    hist = as_history(rows)
    live = sum(1 for v in hist.values()
               if len(v) > 1 and max(v) != min(v))
    print("[ledger] %s%d records carried forward, %d live (%s | %s | %s)"
          % (label and label + ": ", len(hist), live,
             os.path.basename(str(checkpoint).rstrip("/\\")),
             os.path.basename(dataset), measure), flush=True)
    return hist
