"""The frozen instrument, verified at startup and stamped into every receipt.

NO GOVERNING SPEC. Basis: operator 2026-09-15 -- "Just give me a unified approach where
we dont have these late gotcha regressions".

WHAT WENT WRONG THAT THIS PREVENTS. In one session the scoring instrument drifted four
separate ways, and every one was discovered by RUNNING INTO IT rather than by a check:

  * `judge.py` had no JUDGE_REMOTE handling, so score_arms scored the eval on the local
    4b while every straddle number came from OpenRouter's 9b. Two judges, one board.
  * `_openrouter_call` accepted a schema and discarded it, so the remote judge was never
    schema-constrained and judge.parse_score threw away every remote reading.
  * The rubric carried an uncommitted 14-line edit; reverting it moved us to a THIRD
    instrument older than either.
  * The banner printed the local judge's name unconditionally, so the log named the wrong
    instrument and was twice read as confirmation.

Each was found hours in. None had to be: they are all one-line comparisons against a
pinned value. That is the whole idea here -- fail at second 1 with a diff.

USAGE, at the top of anything that produces a score:

    from rlv2.train.rl.instrument import verify
    verify()                      # raises with a diff if anything moved
    ...
    receipt["instrument"] = stamp()
"""
from __future__ import annotations

import hashlib
import io
import json
import os

CONFIG = "configs/instrument.json"


def load(path: str = CONFIG) -> dict:
    return json.loads(io.open(path, encoding="utf-8").read())


def _rubric_sha_stripped(path: str) -> str:
    """Stripped, NOT raw. The 2026-09-12 receipt recorded the stripped hash, and a
    byte-for-byte compare against git therefore matched nothing -- which is how the
    campaign's own instrument came to be called unrecoverable, twice."""
    return hashlib.sha256(
        io.open(path, encoding="utf-8").read().strip().encode("utf-8")).hexdigest()[:16]


def observed(cfg: dict | None = None) -> dict:
    """What the instrument ACTUALLY is right now, read from the live system."""
    cfg = cfg or load()
    import sys
    sys.path.insert(0, "src")
    from rlv2.judge import judge as J
    try:
        from rlv2.judge import reward as R
        remote = R._openrouter_enabled()
        judge_model = R.OPENROUTER_JUDGE_MODEL if remote else J.JUDGE_MODEL
        # JUDGE_JEV overrides both paths (reward.llm_judge_batch_train and judge.judge),
        # so the live instrument IS Jev whatever JUDGE_REMOTE says.
        if R._jev_enabled():
            remote, judge_model = False, "open-jev-2b"
    except ImportError:
        remote, judge_model = False, J.JUDGE_MODEL
    return {
        "rubric_sha_stripped": _rubric_sha_stripped(cfg["rubric_path"]),
        "judge_model": judge_model,
        "judge_remote": remote,
        "eval_n": sum(1 for l in io.open(cfg["eval_file"], encoding="utf-8") if l.strip()),
        "gates_present": [
            g for g, ok in (
                ("empty", "not answer.strip()" in _judge_src()),
                ("colon-stub", 'endswith(":")' in _judge_src()),
                ("never_states_an_answer", "never_states_an_answer(" in _judge_src()),
            ) if ok],
    }


def _judge_src() -> str:
    return io.open("src/rlv2/judge/judge.py", encoding="utf-8").read()


def diff(cfg: dict | None = None) -> list[str]:
    """Every way the live instrument departs from the pinned one. Empty = clean."""
    cfg = cfg or load()
    o = observed(cfg)
    out = []
    if o["rubric_sha_stripped"] != cfg["rubric_sha_stripped"]:
        out.append("rubric: pinned %s, live %s (%s)"
                   % (cfg["rubric_sha_stripped"], o["rubric_sha_stripped"],
                      cfg["rubric_path"]))
    if cfg.get("judge_jev_required"):
        want = cfg["judge_model_jev"]
    else:
        if cfg.get("judge_remote_required") and not o["judge_remote"]:
            out.append("judge: JUDGE_REMOTE required but not enabled -- would score on %s "
                       "instead of %s" % (cfg["judge_model_local"], cfg["judge_model_remote"]))
        want = cfg["judge_model_remote"] if cfg.get("judge_remote_required") \
            else cfg["judge_model_local"]
    if o["judge_model"] != want:
        out.append("judge model: pinned %s, live %s" % (want, o["judge_model"]))
    if o["eval_n"] != cfg["eval_n"]:
        out.append("eval: pinned %d questions, live %d (%s)"
                   % (cfg["eval_n"], o["eval_n"], cfg["eval_file"]))
    missing = [g for g in ("empty", "colon-stub", "never_states_an_answer")
               if g not in o["gates_present"]]
    if missing:
        out.append("mechanical gates missing: %s" % ", ".join(missing))
    return out


def verify(cfg_path: str = CONFIG, strict: bool = True) -> dict:
    """Raise unless the live instrument matches the pinned one. Call before scoring.

    `strict=False` reports without raising, for tools that legitimately run off-
    instrument (a resolution study, a historical re-score). Those must say so in
    their own output -- silence is what produced every defect listed in the module
    docstring.
    """
    cfg = load(cfg_path)
    d = diff(cfg)
    if d:
        msg = ("INSTRUMENT DRIFT -- refusing to produce a score.\n  "
               + "\n  ".join(d)
               + "\nPinned in %s (campaign %s). Fix the drift, or change the pin "
                 "deliberately and bump `campaign`." % (cfg_path, cfg["campaign"]))
        if strict:
            raise SystemExit(msg)
        print("[instrument] " + msg, flush=True)
    else:
        print("[instrument] OK | rubric %s | judge %s | eval %d | gates %d"
              % (cfg["rubric_sha_stripped"],
                 cfg["judge_model_jev"] if cfg.get("judge_jev_required")
                 else cfg["judge_model_remote"] if cfg.get("judge_remote_required")
                 else cfg["judge_model_local"],
                 cfg["eval_n"], len(cfg["mechanical_gates"])), flush=True)
    return cfg


def stamp(cfg_path: str = CONFIG) -> dict:
    """The block every receipt carries, so a number can never be orphaned from the
    instrument that produced it -- the failure that made 7/16 unverifiable."""
    cfg = load(cfg_path)
    o = observed(cfg)
    return {"campaign": cfg["campaign"],
            "rubric_sha_stripped": o["rubric_sha_stripped"],
            "judge_model": o["judge_model"], "judge_remote": o["judge_remote"],
            "eval_file": cfg["eval_file"], "eval_n": o["eval_n"],
            "protocol": cfg["protocol"], "reads_per_cell": cfg["reads_per_cell"],
            "k_samples_per_question": cfg["k_samples_per_question"],
            "correct_floor": cfg["correct_floor"],
            "gates": o["gates_present"],
            "drift": diff(cfg) or None}


if __name__ == "__main__":
    import sys
    verify(strict="--report" not in sys.argv)
    print(json.dumps(stamp(), indent=2))
