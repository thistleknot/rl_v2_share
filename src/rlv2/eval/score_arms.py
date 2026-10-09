"""src/rlv2/eval/score_arms.py — score EVERY arm in ONE invocation, so the instrument cannot drift.

NO GOVERNING SPEC. Basis: owner, 2026-08-30. Telos is apples-to-apples comparison.

A comparison is valid iff every arm is measured by the SAME function on the SAME inputs,
differing only in the treatment. Every measurement failure this session was one flaw wearing
different clothes -- arms scored in SEPARATE invocations, letting the instrument drift:

    base scored with the OLD two-call judge   vs   CPT with the NEW one
    Gate B scored INDIVIDUALLY                vs   the batch test COMPARATIVELY
    baseline compared on `reward_mean`        vs   trials on `rate`

Scoring every arm in one call makes the measurement function constant BY CONSTRUCTION: there
is no second invocation for it to differ from. Discipline cannot achieve that; structure can.

THE BATCH IS THE QUESTION. One judge call per eval question holding one answer from every arm,
SHUFFLED. This is the comparative protocol that produced score spread (gold sd 0.0000 -> 0.0757)
with its batch defined by the question rather than arbitrary grouping -- so batch composition
stops being an uncontrolled variable across a frozen eval (S1.4).

Cost falls too: N arms x Q questions individually is N*Q calls; batched by question it is Q.

THREE PROJECTIONS, ALL REPORTED, ALWAYS (owner, settled 2026-08-30):
    binary rate at the fixed 0.70    the ladder, comparable across stages
    mean graded score, correct class RL's target, invisible to the binary
    KL(correct || incorrect)         separation of the two populations
Declared before the run so no stage can pick its headline afterwards -- which is how T9
reported a +0.43 reward gain while accuracy moved +0.021 (p=0.5637).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import statistics as st
import sys
import time
import urllib.request

sys.path.insert(0, "src")
from rlv2.judge import judge as J

EVAL = "data/rl/r3_eval_canonical.jsonl"
# THE 0.70 line lives in judge.py (CORRECT_FLOOR) and nowhere else. This module carried its
# own copy until 2026-08-31; two constants that must agree and are enforced nowhere is how
# the 0.5-threshold drift in rl_calibration_eval went unnoticed.
FLOOR = J.CORRECT_FLOOR
SEED = 20260830
# MEASURED instrument resolution, 2026-08-30: the identical checkpoint scored as 4 arms in
# one invocation spread 0.2093..0.2619 -- max pairwise delta 0.0526. Any per-stage patience
# or lift smaller than this is inside the noise of the instrument and must not count as an
# improvement. Lives here (not in a config) because it is a property of THIS scorer.
# READS PER CELL. Three is the floor: a single read cannot detect its own noise and two
# cannot break a tie. Two more are added ONLY when those three straddle the correctness
# floor -- see the escalation note in _read_cell.
BASE_READS = 3
ESCALATION_READS = 2
RESOLUTION_FLOOR = 0.0526

_SCHEMA = {"type": "object", "properties": {"scores": {"type": "array", "items": {
    "type": "object", "properties": {
        "label": {"type": "integer"},
        "states_gold_claim": {"type": "boolean"},
        "note": {"type": "string"},
        "score": {"type": "number"}},
    "required": ["label", "states_gold_claim", "note", "score"]}}},
    "required": ["scores"]}


# THE RULE LIVES IN judge.py so the trainer's eval and this scorer cannot drift apart --
# they did, by exactly one question, on 2026-09-16.
_should_escalate = J.should_escalate


def _answer_hash(text):
    """Identity of the ANSWER a judgement was made about.

    A resumed cell is valid only for the exact text that was judged. Keying the
    sidecar on (arm, qid, rep) alone let a regenerated answer inherit the verdict
    of the answer it replaced -- silently, and reported as a fresh measurement
    (2026-09-15: "resuming: 32 cells already judged", zero new judgements, the
    old blank-answer verdict reported as the repaired model's score).
    """
    import hashlib
    return hashlib.sha1((text or "").strip().encode("utf-8")).hexdigest()[:16]


def generate_all(arms, rows, k, temperature):
    """{arm: [k lists of Q answers]}. One model resident at a time (tech.md: never co-reside)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rlv2.eval import half_epoch_eval as he

    tok = AutoTokenizer.from_pretrained("skills_model", trust_remote_code=True)
    # THE RL PROMPT, not chat_wrap (2026-09-26). The board picks the checkpoint RL starts
    # from, and RL trains and evals on docs/grpo_one_file.build_prompt: the model's own
    # template, SYSTEM turn, thinking off. chat_wrap made the model open with <|im_end|> on
    # 6-9/16 eval answers (dispositions 2026-09-25), which would rank checkpoints on a
    # prompt RL never shows them. Same template as build_prompt, built here directly.
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": he.SYSTEM},
         {"role": "user", "content": r.get("prompt") or r["question"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
    out = {}
    for name, path in arms.items():
        t = time.time()
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16,
                                                 trust_remote_code=True).cuda().eval()
        out[name] = [he.generate(m, tok, prompts, temperature=temperature, wrap=False) for _ in range(k)]
        del m
        torch.cuda.empty_cache()
        print("  generated %-34s k=%d  %.0fs" % (name, k, time.time() - t), flush=True)
    return out


def score_question(row, entries, rng):
    """ONE call ranking every arm's answer to ONE question.

    SHUFFLED, permutation returned. Position bias is the obvious confound in any ranking
    prompt: with a fixed order, a judge that favours the first slot hands one arm a systematic
    advantage that reads as a real effect.
    """
    order = list(range(len(entries)))
    rng.shuffle(order)
    shown = "\n\n".join("ANSWER %d: %s" % (i + 1, (entries[order[i]][1] or "").strip() or "<EMPTY>")
                        for i in range(len(order)))
    prompt = (
        "%s\n\n---\n"
        "QUESTION: %s\nGOLD: %s\n\n"
        "Below are %d candidate answers to that question, from different models. They are a "
        "MIX: some land the gold's claim, some do not.\n\n"
        # THE RANKING RULE LIVES IN THE RUBRIC, NOT HERE. It used to be duplicated in this
        # prompt, which meant `rescore_rubric.py` and this file read ONE rubric under TWO
        # framings -- a smaller instance of the drift this design exists to remove. The rubric
        # is hashed into every receipt; a rule stated only here is invisible to that hash.
        "For EACH: set `states_gold_claim` first, then score it, applying the scale and the "
        "ranking rule stated in the rubric above.\n"
        "\n%s\n"
        % (J.load_rubric(), row.get("question") or row.get("prompt"), row["gold"],
           len(entries), shown))
    # ROUTE THROUGH judge._call, NOT A BARE urlopen. The first version of this function opened
    # its own connection with a fixed 900 s timeout and no retry, so a single hung judge call
    # raised TimeoutError straight through and killed a run that had already spent 55 MINUTES
    # generating. `_call` carries the adaptive LatencyBudget, catches transport failures, and
    # returns None instead of raising -- which is what lets an evicted call be retried rather
    # than ending the job.
    # RETRIES MUST ESCALATE THE TIMEOUT, or they are decorative.
    #
    # Measured on eval q8, 2026-08-30: the SAME call took 192.3 s cold and 18.3 s warm --
    # identical output (202 tokens, done=stop, parses fine), the difference being ollama's KV
    # cache on a 2808-token prompt. The latency distribution is BIMODAL: warm ~20 s, cold
    # ~190 s. No single EMA band covers both, so a budget learned from warm calls evicts cold
    # ones as if they were hangs.
    #
    # The first loop retried three times at an IDENTICAL timeout against a deterministic cost,
    # which reproduces the identical failure by construction -- three attempts, one guaranteed
    # outcome. Retrying a timeout with the same timeout is not a retry.
    #
    # `num_predict` also scales with the batch: ~90 output tokens PER ANSWER (four reasoning
    # fields each), and at the 256 default a 4-arm call is cut mid-JSON and returns None --
    # which dropped 44% of calls and also looked like a timeout.
    # NO MISSING EVALS (operator, 2026-09-06: "I have 16 eval questions and I
    # fucking expect 16 evaluations"). Two measured leak paths, both closed here:
    # (1) whole-call None after 3 attempts — load-transient (the same r3-0251
    #     call that failed a receipt parsed in 12 s on an idle GPU);
    # (2) PARTIAL parses — the judge returning entries for only some labels was
    #     accepted silently, leaving unscored cells with no log line at all
    #     (surpass11 receipt: r3-0042/r3-0522/r3-0191 asymmetric drops).
    # An incomplete parse now escalates like a timeout, and whatever is still
    # missing afterwards falls back to the SINGLE-ANSWER judge path — a
    # different protocol (absolute, not comparative), so the cell is flagged
    # "fallback" in the record — and main() refuses to call a receipt complete
    # if any cell is still empty after that.
    res = {}
    for attempt, mult in enumerate((1.0, 4.0, 12.0)):
        out = J._call(prompt, _SCHEMA, J.JUDGE_MODEL,
                      timeout=max(60.0, J.BUDGET.timeout() * mult),
                      num_predict=256 + 160 * len(entries))
        for e in (out or {}).get("scores", []):
            li = e.get("label", 0) - 1
            if 0 <= li < len(order):
                # Task #39: clamp to [0,1] like reward.py's judge paths. Measured
                # leak (data/past9_per_receipt.json r3-0007): the judge's note said
                # "Scored 0.98" but the score field carried 3.0, inflating
                # mean|correct to an impossible 1.362.
                res[entries[order[li]][0]] = {
                    "score": max(0.0, min(1.0, float(e["score"]))),
                    "states": bool(e.get("states_gold_claim")),
                    "note": (e.get("note") or "")[:120]}
        if len(res) == len(entries):
            if attempt:
                print("    recovered on attempt %d (timeout x%.0f)" % (attempt + 1, mult),
                      flush=True)
            break
        print("    incomplete: %d/%d answers scored at timeout x%.0f (attempt %d/3), "
              "escalating" % (len(res), len(entries), mult, attempt + 1), flush=True)
    for arm, text in entries:
        if arm in res:
            continue
        r1 = J.judge(row.get("question") or row.get("prompt"), row["gold"], text or "",
                     category=row.get("category"))
        if r1 and r1.get("score") is not None:
            res[arm] = {"score": max(0.0, min(1.0, float(r1["score"]))),
                        "states": None, "fallback": True,
                        "note": "single-answer fallback: " + str(r1.get("verdict", ""))[:90]}
            print("    fallback scored %s (single-answer protocol)" % arm, flush=True)
    missing = [arm for arm, _ in entries if arm not in res]
    return {"per_arm": res, "permutation": [entries[o][0] for o in order],
            "failed": bool(missing), "missing": missing}


def auc(correct, wrong):
    """P(random correct answer scores above a random wrong one). The MONOTONE separation measure.

    0.5 = no separation, 1.0 = perfect. Threshold-free, so it does not depend on the 0.70 line
    at all -- which is why it is the guard against that line being arbitrary or piling up.

    KL is NOT a substitute for this. Measured by hand 2026-08-30: correct at ~0.95 against
    wrong at ~0.72 gives KL 2.58; the same correct against wrong at ~0.05 gives 2.53. Further
    apart, LOWER KL. KL saturates once the supports are disjoint because it asks how badly Q
    fails to cover P, not how far apart they are. Both are reported; they answer different
    questions.
    """
    if not correct or not wrong:
        return float("nan")
    wins = ties = 0
    for c in correct:
        for w in wrong:
            if c > w:
                wins += 1
            elif c == w:
                ties += 1
    return (wins + 0.5 * ties) / (len(correct) * len(wrong))


def kl_divergence(a, b, bins=10):
    """KL(correct || incorrect) over binned scores. Laplace-smoothed so an empty bin stays finite.

    WHAT IT DOES AND DOES NOT MEAN. This is the overlap GRPO actually sees: how much the
    wrong-answer reward distribution fails to explain the right-answer one. It is NOT a
    distance and does NOT grow monotonically with separation -- it SATURATES once the supports
    are disjoint. Use `auc` for "how far apart"; use this for "how much do they overlap".
    """
    if len(a) < 2 or len(b) < 2:
        return float("nan")

    def hist(xs):
        h = [1.0] * bins
        for x in xs:
            h[min(bins - 1, max(0, int(x * bins)))] += 1
        t = sum(h)
        return [v / t for v in h]

    P, Q = hist(a), hist(b)
    return sum(p * math.log(p / q) for p, q in zip(P, Q))


def bootstrap_ci(xs, reps=5000, seed=SEED):
    """Percentile CI on a proportion. A point estimate at n=16 carries no claim on its own."""
    if not xs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(rng.choice(xs) for _ in range(n)) / n for _ in range(reps))
    return (means[int(0.025 * reps)], means[int(0.975 * reps)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH",
                    help="repeatable, e.g. --arm base=skills_model")
    ap.add_argument("--eval", default=EVAL)
    ap.add_argument("--k", type=int, default=8,
                    help="samples per question. Not only statistical power: the per-question "
                         "pass rate it yields IS the measurement of RL's addressable band.")
    # GREEDY, BECAUSE A GRADER MUST NOT RE-ROLL. Measured 2026-09-16 on
    # W:/phase_runs/rl_best8: two invocations at temperature 0.7, k=1, identical
    # weights, agreed on 0 of 16 answers -- one returned an empty string where the
    # other wrote a full answer. That is why `base` read 1, 2, 3, 4 and 5 across this
    # campaign without changing. Sampling belongs in k-sample estimation (k>1, where
    # the spread IS the measurement), never in a single-sample verdict.
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--score-k", type=int, default=0, dest="score_k",
                    help="score only the first N of the k generations. GENERATION is cheap "
                         "and cached; JUDGING is the cost (one call per question per rep). "
                         "k=8 judging is ~85 min, which is an HPO design failure -- "
                         "bounded execution wants tens of minutes. Generate deep once, score "
                         "shallow while iterating, score deep once for the final number.")
    ap.add_argument("--out", default="data/arm_comparison.json")
    ap.add_argument("--protocol", choices=("comparative", "single"), default="comparative",
                    help="'single' = THE LOCKED MECHANISM (operator, 2026-09-06): each "
                         "(question, answer) judged alone 3x, majority of readings is "
                         "the verdict. No comparative prompt (its neighbor-fooling "
                         "cannot be gated); empty/stub answers are gated in judge.judge.")
    a = ap.parse_args()

    arms = dict(x.split("=", 1) for x in a.arm)
    for n, path in arms.items():
        # A HUB ID IS NOT A MISSING CHECKPOINT. `unsloth/Qwen3.5-0.8B` is the BASE
        # model and lives in the HF cache, so the on-disk guard refused to score the
        # one arm every other arm is measured against. Third tool to hit this class
        # of bug in one session -- straddle_probe SKIPPED hub ids, and
        # differentiability_ledger hashed every one of them to the same key.
        #
        # The guard still earns its place for a mistyped LOCAL path, which would
        # otherwise surface as a confusing loader error much later. But it can only
        # judge something it can see: if the path is a directory, it must look like a
        # checkpoint; if it is not, it is a hub id and transformers raises its own
        # clear error.
        if os.path.isdir(path) and not os.path.exists(
                os.path.join(path, "config.json")):
            raise SystemExit("arm %s: %s is a directory but has no config.json"
                             % (n, path))
    rows = [json.loads(l) for l in open(a.eval, encoding="utf-8") if l.strip()]
    # THE INSTRUMENT IS VERIFIED BEFORE ANY SCORE IS PRODUCED. Every drift this
    # session -- local judge vs remote, discarded schema, a rubric that moved --
    # was discovered by running into it hours later. This fails at second 1.
    from rlv2.train.rl.instrument import verify as _verify_instrument, stamp as _instrument_stamp
    _verify_instrument()

    # NAME THE JUDGE THAT ACTUALLY RUNS. This printed J.JUDGE_MODEL unconditionally,
    # so under JUDGE_REMOTE=1 the banner announced the local 4b while every call went
    # to OpenRouter's 9b. A log that names the wrong instrument is worse than one that
    # names none: it was read as confirmation twice before the mismatch was found.
    try:
        from rlv2.judge import reward as _R
        _live_judge = (_R.OPENROUTER_JUDGE_MODEL + " (openrouter)"
                       if _R._openrouter_enabled() else J.JUDGE_MODEL + " (local)")
        if _R._jev_enabled():
            # judge.judge routes to Jev under JUDGE_JEV; the comparative protocol calls the
            # LLM directly and cannot, so refuse it rather than silently mix two judges.
            if a.protocol != "single":
                raise SystemExit("JUDGE_JEV=1 requires --protocol single (comparative calls the LLM judge directly)")
            _live_judge = "open-jev-2b (binary yes/no, %s)" % os.environ.get("JEV_URL", "http://192.168.3.18:8791")
    except ImportError:
        _live_judge = J.JUDGE_MODEL + " (local)"
    print("ONE INVOCATION | %d arms | %d questions | k=%d | judge %s | rubric %s"
          % (len(arms), len(rows), a.k, _live_judge, J.rubric_sha()), flush=True)

    # CACHE THE GENERATIONS PER CHECKPOINT, not per invocation.
    #
    # This used to key on `a.out` and require set(gens) == set(arms), so adding or
    # removing ONE arm threw away every cached generation -- including the base's.
    # Measured 2026-09-15: base was re-rolled in five separate invocations and read
    # 2, 3, 4, 3 and 5 out of 16. At temperature 0.7 with k=1 that is one sampled
    # answer per question, re-drawn every run, and it is the DOMINANT source of
    # movement on this board -- larger than the judge's own variance.
    #
    # The key is what actually determines the answers: the checkpoint's content hash,
    # the eval file, k, and the temperature. Change any of those and it regenerates;
    # change the arm LIST and every unchanged arm is reused.
    import hashlib as _hl
    sys.path.insert(0, "src")
    from rlv2.train.rl.instrument import _rubric_sha_stripped  # noqa: F401  (same hashing idea)
    from rlv2.train.rl import differentiability_ledger as _ledger

    CACHE_DIR = os.path.join("data", "gen_cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    def _gen_key(path):
        """What the answers actually depend on. The checkpoint is hashed by CONTENT
        (differentiability-ledger REQ-L2) so a directory overwritten under the same
        name cannot silently serve another model's answers."""
        state = _ledger.phase_state(path)
        # "prompt=rl_template" is part of the key: the prompt changed from chat_wrap on
        # 2026-09-26, and without it every cached chat_wrap answer would be served back as
        # if it had been generated on the RL prompt.
        raw = "%s|%s|k=%d|t=%.3f|prompt=rl_template" % (state, os.path.basename(a.eval), a.k, a.temperature)
        return os.path.join(CACHE_DIR, _hl.sha1(raw.encode("utf-8")).hexdigest()[:16]
                            + ".json")

    gens, need = {}, {}
    for _name, _path in arms.items():
        ck = _gen_key(_path)
        if os.path.exists(ck):
            try:
                v = json.load(open(ck, encoding="utf-8"))
                if len(v) == a.k:
                    gens[_name] = v
                    print("  reusing generations for %-28s (%s)"
                          % (_name, os.path.basename(ck)), flush=True)
                    continue
            except (ValueError, OSError):
                pass
        need[_name] = _path

    if need:
        fresh = generate_all(need, rows, a.k, a.temperature)
        for _name, v in fresh.items():
            gens[_name] = v
            json.dump(v, open(_gen_key(arms[_name]), "w", encoding="utf-8"))
        print("  cached %d new generation set(s) -> %s" % (len(fresh), CACHE_DIR),
              flush=True)

    score_k = a.score_k if a.score_k > 0 else a.k
    if score_k < a.k:
        print("  SCORING %d of %d generations per question (generation kept at k=%d in the "
              "cache, so a deeper pass costs only judge time)" % (score_k, a.k, a.k), flush=True)

    rng = random.Random(SEED)
    scores = collections.defaultdict(list)
    hits = collections.defaultdict(list)
    perq = collections.defaultdict(lambda: collections.defaultdict(list))
    records = []
    n_failed = [0]
    t0 = time.time()
    missing_cells = []
    if a.protocol == "single":
        # THE LOCKED MECHANISM. Resumable: each judged cell is appended to the
        # sidecar as it lands, so a crashed 800-call sweep resumes, not restarts.
        sidecar = a.out + ".cells.jsonl"
        done = {}
        if os.path.exists(sidecar):
            for line in open(sidecar, encoding="utf-8"):
                c = json.loads(line)
                # A CACHED CELL COUNTS AS DONE ONLY IF IT CAN VOTE (measured
                # 2026-09-07: a cell whose three readings all failed to parse was
                # written as [null,null,null], and every rerun then "resumed" it
                # as complete — so the one missing cell could never be recovered
                # and the receipt stayed permanently incomplete).
                if sum(1 for v in c["reads"] if v is not None) >= 2:
                    done[(c["arm"], c["qid"], c["rep"], c.get("ans"))] = c["reads"]
            print("  resuming: %d cells already judged in %s" % (len(done), sidecar),
                  flush=True)
        side = open(sidecar, "a", encoding="utf-8")
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=3)   # reward.py's proven ollama concurrency

        def _read_cell(row, ans):
            """Three independent readings. An UNPARSEABLE judge reply is NOT a
            zero (measured 2026-09-07: 'The value is 0.5^2 = 0.25.' against gold
            '0.25' read 0.0 three times — judge.FAILED carries score 0.0 with
            ok=False, and this path was recording it as a genuine wrong answer).
            A failed parse is a MISSING reading: retried once, then dropped from
            the vote so the majority is taken over real readings only."""
            def _one():
                for _attempt in range(2):
                    j = J.judge(row.get("question") or row.get("prompt"), row["gold"],
                                ans, category=row.get("category") or "")
                    if j and j.get("ok") and j.get("score") is not None:
                        return min(1.0, max(0.0, float(j["score"])))
                return None

            reads = [_one() for _ in range(BASE_READS)]

            # ESCALATE ONLY ON A SPLIT. Operator 2026-09-16: "Might as well implement the
            # escalate on split as well (for new records)".
            #
            # The judge only flips where an answer sits near the 0.70 floor -- measured the
            # same day, the same checkpoint read 6/16 then 5/16 on ONE such question while
            # the rest were unanimous. Extra reads on a unanimous question buy nothing; the
            # verdict was never in doubt. So spend them only where the reads actually
            # disagree about which SIDE of the floor the answer is on:
            #
            #     every question at 5 reads   16 * 5            = 80 calls
            #     escalate ~20% of questions  16 * 3 + 3 * 2    = 54 calls
            #
            # Same resolution where it matters, a third less judging. A read that failed to
            # parse is MISSING, not a vote, so it cannot trigger an escalation by itself.
            if _should_escalate(reads, FLOOR):
                reads.extend(_one() for _ in range(ESCALATION_READS))
            return reads

        for qi, row in enumerate(rows):
            qid = row.get("item_id") or row.get("id") or str(qi)
            for rep in range(score_k):
                per_arm = {}
                _ans = {name: (gens[name][rep][qi] or "") for name in arms}
                _ah = {name: _answer_hash(_ans[name]) for name in arms}
                futures = {name: pool.submit(_read_cell, row, _ans[name])
                           for name in arms
                           if (name, qid, rep, _ah[name]) not in done}
                for name in arms:
                    reads = done.get((name, qid, rep, _ah[name]))
                    if reads is None:
                        reads = futures[name].result()
                        # Spec: .spec/specs/r3-domain-injection/tasks.md #52 (answer text
                        # in cells). "ans" stays a hash -- resume lookups above key on it --
                        # "text" is the readable answer nobody could recover from a hash,
                        # which hid a reward-hacking failure (echoed questions) for 5 runs.
                        side.write(json.dumps({"arm": name, "qid": qid,
                                               "rep": rep, "ans": _ah[name],
                                               "text": _ans[name],
                                               "reads": reads}) + chr(10))
                        side.flush()
                    votes = [s for s in reads if s is not None]
                    if len(votes) < 2:
                        missing_cells.append((qid, rep, name))
                        n_failed[0] += 1
                        continue
                    correct = sum(1 for s in votes if s >= FLOOR) * 2 > len(votes)
                    # the cell's score = median read, so mean|correct stays meaningful
                    med = sorted(votes)[len(votes) // 2]
                    per_arm[name] = {"score": med, "reads": reads,
                                     "majority_correct": correct}
                    scores[name].append(med)
                    hits[name].append(1 if correct else 0)
                    perq[name][qid].append(1 if correct else 0)
                records.append({"qid": qid, "rep": rep, "protocol": "single-x3-majority",
                                "scores": per_arm})
            print("  q%-3d done (%d arms x %d reps x 3 reads)"
                  % (qi + 1, len(arms), score_k), flush=True)
        pool.shutdown(wait=True)
        side.close()
        print("  scoring %.0fs" % (time.time() - t0))
        if missing_cells:
            print("  INCOMPLETE RECEIPT: %d cells lack a majority of readings:"
                  % len(missing_cells), flush=True)
            for qid, rep, arm in missing_cells:
                print("    MISSING %s rep=%d arm=%s" % (qid, rep, arm), flush=True)
        _finish(a, arms, rows, score_k, scores, hits, perq, records, missing_cells)
        return
    for qi, row in enumerate(rows):
        qid = row.get("item_id") or row.get("id") or str(qi)
        for rep in range(score_k):
            entries = [(name, gens[name][rep][qi]) for name in arms]
            r = score_question(row, entries, rng)
            if r.get("failed"):
                n_failed[0] += 1
                missing_cells += [(qid, rep, arm) for arm in r.get("missing", [])]
            for arm, v in r["per_arm"].items():
                scores[arm].append(v["score"])
                hits[arm].append(1 if v["score"] >= FLOOR else 0)
                perq[arm][qid].append(1 if v["score"] >= FLOOR else 0)
            records.append({"qid": qid, "rep": rep, "permutation": r["permutation"],
                            "scores": r["per_arm"]})
        print("  q%-3d done (%d arms x %d reps)" % (qi + 1, len(arms), score_k), flush=True)
    print("  scoring %.0fs" % (time.time() - t0))
    if missing_cells:
        print("  INCOMPLETE RECEIPT: %d unscored (question, rep, arm) cells remain after "
              "comparative retries AND the single-answer fallback:" % len(missing_cells),
              flush=True)
        for qid, rep, arm in missing_cells:
            print("    MISSING %s rep=%d arm=%s" % (qid, rep, arm), flush=True)

    _finish(a, arms, rows, score_k, scores, hits, perq, records, missing_cells)


def _finish(a, arms, rows, score_k, scores, hits, perq, records, missing_cells):
    """Shared receipt tail: projections table, band table, JSON dump, break-fast.
    Factored out 2026-09-06 so the locked single-protocol path and the comparative
    path write byte-identical receipt shapes."""
    print("\n" + "=" * 76)
    print("THREE PROJECTIONS - all arms, ONE instrument, ONE invocation")
    print("=" * 76)
    print("%-24s %-24s %-18s %-9s %s"
          % ("arm", "binary rate [95% CI]", "mean|correct", "KL(c||i)", "AUC"))
    summary = {}
    for name in arms:
        h, sc = hits[name], scores[name]
        lo, hi = bootstrap_ci(h)
        corr = [x for x in sc if x >= FLOOR]
        wrong = [x for x in sc if x < FLOOR]
        mc = st.mean(corr) if corr else float("nan")
        kl = kl_divergence(corr, wrong)
        au = auc(corr, wrong)
        summary[name] = {"n_obs": len(h), "rate": sum(h) / max(1, len(h)), "ci": [lo, hi],
                         "mean_correct": mc, "kl": kl, "auc": au, "n_correct": len(corr)}
        print("%-24s %.4f [%.3f, %.3f]    %-18s %-9.4f %.4f"
              % (name, summary[name]["rate"], lo, hi,
                 ("%.4f (n=%d)" % (mc, len(corr))) if corr else "-- (n=0)", kl, au))

    print("\n" + "=" * 76)
    print("RL's ADDRESSABLE BAND - per-question pass rate at k=%d" % a.k)
    print("=" * 76)
    print("RL cannot create knowledge. p=0 and p=1 are zero-variance groups giving EXACTLY zero")
    print("gradient; only 0<p<1 can teach. That count is a HARD CEILING on RL accuracy gain.")
    for name in arms:
        pr = {q: sum(v) / len(v) for q, v in perq[name].items()}
        never = sum(1 for v in pr.values() if v == 0)
        always = sum(1 for v in pr.values() if v == 1)
        band = len(pr) - never - always
        summary[name]["band"] = {"never": never, "inconsistent": band, "always": always}
        print("  %-24s never %2d | INCONSISTENT %2d | always %2d   of %d"
              % (name, never, band, always, len(pr)))

    json.dump({"judge": J.instrument(), "k": a.k, "score_k": score_k,
               "temperature": a.temperature,
               "protocol": getattr(a, "protocol", "comparative"),
               "floor": FLOOR, "arms": arms, "summary": summary, "records": records,
               "complete": not missing_cells,
               "missing": [list(c) for c in missing_cells]},
              open(a.out, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % a.out)
    if missing_cells:
        # BREAK FAST (operator, 2026-09-06): a receipt with unscored eval cells is
        # not a result — the run FAILS after persisting its evidence, and no verdict
        # may be read from it. Every question gets its evaluation, or the receipt
        # says so with a nonzero exit.
        print("RECEIPT INCOMPLETE — %d cells unscored. This receipt proves nothing "
              "until the missing cells are scored." % len(missing_cells), flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
