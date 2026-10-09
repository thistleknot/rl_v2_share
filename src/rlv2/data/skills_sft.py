"""src/rlv2/data/skills_sft.py — synthetic SFT QA from the stackskills corpus. Stage 1 of the full regimen.

WHY QA AND NOT THE SKILL TEXT. Training on the .md files directly is continued pretraining, and CPT
is CLOSED here on a clean instrument: at its published learning rate (1e-5) it contributes EXACTLY
+0.000 acquisition through a capable full-parameter readout, while still costing retention
(D_Held 0.792 vs 0.833 without it). Damage falls as lr falls; acquisition never appears at any rate.
So the source text is INPUT to question generation, never a training target.

SCOPE, and why it is not the whole corpus. Measured, not estimated:
  generation   6.67 s/skill at 8-way parallel (ollama is GPU-bound on one model instance)
  training     ~249 tok/s, full-param SFT, 0.8B
  corpus       1467 skills, 1.68M source tokens, median 1086 tok, p95 1695
Full corpus = 163 min generation + 235 min training (3 epochs) = ~6.9 h. Against a 4 h budget the
binding resource is TIME, not memory, so this takes a stratified subset instead of dropping whole
topics. Every topic keeps its proportional share.

WHY 5 QA PER SKILL x 3 EPOCHS. T10 measured phrasing depth as the strongest effect in the battery
(+0.208 held-out, p=0.0075, STILL RISING at 10 phrasings). T7 measured the exposure knee at ~16.
5 phrasings x 3 epochs = 15 exposures per fact, landing on that knee. Cutting phrasings to save time
would sacrifice the single best-evidenced lever; cutting SKILLS costs only coverage. So skills are
the axis that gets trimmed.

HELD-OUT DESIGN. 10% of SKILLS are withheld ENTIRELY -- every QA pair from a held-out skill is
excluded from training. Splitting at the QA level instead would leak: five paraphrases of one fact
would land on both sides, and the eval would measure memorisation of a phrasing rather than
acquisition of a fact. That exact confound was closed in T10 and is not being reopened.

STALL/RESUME. Each skill's pairs are appended the moment they parse, and a completed skill id is
never regenerated. Killing this at any point costs one skill.
"""
import argparse
import glob
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

SKILLS_DIR = r"C:\Users\user\Documents\wiki\data science\stackskills\skills_master"
OUT_DIR = "data/skills"
RAW = f"{OUT_DIR}/skills_qa_raw.jsonl"
TRAIN = f"{OUT_DIR}/skills_train.jsonl"
EVAL = f"{OUT_DIR}/skills_eval.jsonl"
DONE = f"{OUT_DIR}/.done_skills.txt"

GEN_MODEL = os.environ.get("QA_GEN_MODEL", "qwen3.5-oc:4b")
OLLAMA = "http://127.0.0.1:11434/api/chat"
N_SKILLS = 800
QA_PER_SKILL = 5
HELDOUT_FRAC = 0.10
PARALLEL = 8
SEED = 7
# RAISED 2026-08-03 from 6000. At 6000 the generator saw only the head of 99 of the 1,143 generated
# skills (8.7%), discarding text a CPT corpus built from FULL notes does train on -- so QA coverage was
# head-biased against the very knowledge CPT deposits. Measured: max generated note 10,861 chars, and
# at 4.47 chars/token that is ~2,427 tokens, which fits num_ctx 8192 alongside the ~1,200-token reply.
# 12000 covers p100 with headroom. Use --regen-truncated to rebuild the affected skills in place.
MAX_SRC_CHARS = 12000
OLD_MAX_SRC_CHARS = 6000    # the limit the existing RAW rows were generated under

SYSTEM = (
    "You write exam questions from technical notes. Given a skill note, emit EXACTLY 5 "
    "question/answer pairs, one JSON object per line, no other text:\n"
    '{"q": "...", "a": "..."}\n'
    "RULES:\n"
    "1. Each question must be SELF-CONTAINED and name the concept explicitly. A reader who has "
    "never seen the note must be able to answer it. Never write 'this method' or 'the above'.\n"
    "2. Vary the phrasing across the 5: what/why/when/how/compare. Do not restate one question "
    "five ways with synonyms.\n"
    "3. Answers are 1-3 sentences, factual, no preamble, no 'According to the note'.\n"
    "4. Ask about substance -- mechanics, assumptions, tradeoffs, failure modes -- not about the "
    "document's structure."
)

_lock = threading.Lock()
_stats = defaultdict(int)


def _norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def _parse(txt, skill_id, topic, name):
    """Pull {q,a} objects out of the model's reply. Tolerates fences and stray prose."""
    out = []
    for m in re.finditer(r"\{[^{}]*\"q\"\s*:.*?\}", txt, re.S):
        try:
            o = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        q, a = _norm(o.get("q")), _norm(o.get("a"))
        # Reject the failure modes that actually occur: context-dependent questions, and answers
        # so short they carry no fact. A bad pair is worse than a missing one -- it trains noise.
        if len(q) < 15 or len(a) < 15:
            continue
        if re.search(r"\b(this|the above|the note|the document|the passage)\b", q, re.I):
            _stats["ctx_dependent"] += 1
            continue
        out.append({"id": f"{skill_id}::{len(out)}", "skill": skill_id, "topic": topic,
                    "name": name, "prompt": q, "answer": a})
    return out[:QA_PER_SKILL]


def _gen_one(path):
    sid = os.path.splitext(os.path.relpath(path, SKILLS_DIR))[0].replace("\\", "/")
    src = open(path, encoding="utf-8", errors="replace").read()
    name = sid.rsplit("/", 1)[-1]
    m = re.search(r"^name:\s*(.+)$", src, re.M)
    if m:
        name = _norm(m.group(1))
    topic = sid.split("/")[0]
    # BACKEND SWITCH. ollama runs on the SAME GPU as training, so local generation must be strictly
    # sequential with it. The NVIDIA endpoint is OFF-GPU and therefore overlaps training for free --
    # which is the whole reason to have the switch. Set GEN_BACKEND=nvidia to use it.
    use_nv = os.environ.get("GEN_BACKEND", "ollama").lower() == "nvidia"
    body = json.dumps({"model": GEN_MODEL, "stream": False, "think": False,
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": src[:MAX_SRC_CHARS]}],
                       "options": {"temperature": 0.8, "num_ctx": 8192}}).encode()
    for attempt in range(3):
        try:
            if use_nv:
                from rlv2.judge import nvidia_llm as NV
                txt = NV.complete(SYSTEM, src[:MAX_SRC_CHARS], max_tokens=1200, temperature=0.8)
            else:
                req = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
                txt = json.loads(urllib.request.urlopen(req, timeout=300).read())["message"]["content"]
            pairs = _parse(txt, sid, topic, name)
            if pairs:
                with _lock:
                    with open(RAW, "a", encoding="utf-8") as fh:
                        for p in pairs:
                            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
                    with open(DONE, "a", encoding="utf-8") as fh:
                        fh.write(sid + "\n")
                    _stats["skills"] += 1
                    _stats["pairs"] += len(pairs)
                    if _stats["skills"] % 25 == 0:
                        el = time.time() - _stats["t0"]
                        rate = el / _stats["skills"]
                        left = (_stats["total"] - _stats["skills"]) * rate / 60
                        print(f"  {_stats['skills']}/{_stats['total']} skills, "
                              f"{_stats['pairs']} pairs, {rate:.2f}s/skill, ETA {left:.0f} min",
                              flush=True)
                return
            _stats["empty"] += 1
        except (urllib.error.URLError, TimeoutError, OSError, KeyError):
            time.sleep(2 + attempt * 3)
    with _lock:
        _stats["failed"] += 1


def select():
    """Stratified by topic, proportional. Keeps every topic represented rather than dropping some."""
    files = [f for f in glob.glob(os.path.join(SKILLS_DIR, "**", "*.md"), recursive=True)
             if "MASTER_TOC" not in f]
    by_topic = defaultdict(list)
    for f in files:
        by_topic[os.path.relpath(f, SKILLS_DIR).split(os.sep)[0]].append(f)
    rng = random.Random(SEED)
    picked, total = [], len(files)
    for t, fs in sorted(by_topic.items()):
        fs = sorted(fs)
        rng.shuffle(fs)
        k = max(1, round(N_SKILLS * len(fs) / total))
        picked += fs[:k]
        print(f"  {t:<22}{len(fs):>5} available  ->  {min(k, len(fs)):>4} selected", flush=True)
    rng.shuffle(picked)
    return picked[:N_SKILLS]


def split():
    """Split by SKILL, never by QA pair -- see the module docstring on leakage."""
    recs = [json.loads(l) for l in open(RAW, encoding="utf-8") if l.strip()]
    skills = sorted({r["skill"] for r in recs})
    random.Random(SEED).shuffle(skills)
    held = set(skills[:max(1, int(len(skills) * HELDOUT_FRAC))])
    tr = [r for r in recs if r["skill"] not in held]
    ev = [r for r in recs if r["skill"] in held]
    for path, rows in ((TRAIN, tr), (EVAL, ev)):
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    assert not ({r["skill"] for r in tr} & {r["skill"] for r in ev}), "skill leaked across split"
    print(f"\ntrain {len(tr)} pairs / {len(skills)-len(held)} skills"
          f"   held-out {len(ev)} pairs / {len(held)} skills (skill-disjoint)", flush=True)
    return tr, ev


HELD_SKILLS = f"{OUT_DIR}/held_out_skills.json"


def split_preserving(held=None):
    """Re-derive TRAIN/EVAL from RAW against a PINNED held-out skill list.

    WHY `split()` CANNOT BE USED ONCE A SPLIT HAS SHIPPED. `split()` computes the boundary as
    `sorted({skill})` shuffled under SEED, then takes the first `HELDOUT_FRAC`. That is a function of
    the skill SET SIZE, so it is stable only while RAW is frozen. Measured 2026-08-03: the shipped
    eval holds 100 skills, but RAW has since grown to 1,143 and `split()` on today's RAW holds out
    **114 -- overlapping the shipped set by 6.** Re-running it silently relocates 94 of 100 held-out
    skills into train, which would void every number ever measured on that eval (CPT +0.059, the
    -0.118 composition result) without changing a single hyperparameter.

    So the boundary is an ARTIFACT, not a computation. It is pinned to `held_out_skills.json` and
    every later regeneration filters against it. New skills added to RAW join TRAIN by default --
    growing the eval is a deliberate act, never a side effect of the corpus getting bigger.
    """
    if held is None:
        if os.path.exists(HELD_SKILLS):
            held = set(json.load(open(HELD_SKILLS, encoding="utf-8"))["held_out"])
        else:                                   # bootstrap from the shipped eval, then pin it
            held = {json.loads(l)["skill"] for l in open(EVAL, encoding="utf-8") if l.strip()}
            json.dump({"held_out": sorted(held), "pinned": "2026-08-03",
                       "why": "split() is not reproducible once RAW grows; see split_preserving"},
                      open(HELD_SKILLS, "w", encoding="utf-8"), indent=1)
            print(f"  pinned {len(held)} held-out skills -> {HELD_SKILLS}", flush=True)
    recs = [json.loads(l) for l in open(RAW, encoding="utf-8") if l.strip()]
    tr = [r for r in recs if r["skill"] not in held]
    ev = [r for r in recs if r["skill"] in held]
    for path, rows in ((TRAIN, tr), (EVAL, ev)):
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    assert not ({r["skill"] for r in tr} & {r["skill"] for r in ev}), "skill leaked across split"
    print(f"\ntrain {len(tr)} pairs / {len({r['skill'] for r in tr})} skills"
          f"   held-out {len(ev)} pairs / {len({r['skill'] for r in ev})} skills (PINNED)", flush=True)
    return tr, ev


def _src_chars(sid):
    p = os.path.join(SKILLS_DIR, sid.replace("/", os.sep) + ".md")
    return len(open(p, encoding="utf-8", errors="replace").read()) if os.path.exists(p) else -1


def regen_truncated(parallel):
    """Rebuild QA for skills whose source exceeded the OLD generation limit, IN PLACE.

    WHY NOT APPEND. `_gen_one` appends and ids are positional (`{sid}::0..4`), so appending would
    duplicate every id for the affected skills. So the rows are REMOVED first, then regenerated by the
    unchanged `_gen_one`.

    WHY A SNAPSHOT. Removing before generating means a failed skill would lose its QA outright. The
    removed rows are held in memory and restored for any skill that does not come back.

    WHY THE SPLIT SURVIVES. `split()` derives the held-out set from `sorted({skill})` under a fixed
    SEED, so it depends on the skill SET, not on row counts or order. Regenerating existing skills
    leaves that set untouched -- asserted here rather than assumed, because a silently moved eval
    boundary would invalidate every downstream comparison.
    """
    raw = [json.loads(l) for l in open(RAW, encoding="utf-8") if l.strip()]
    before_skills = {r["skill"] for r in raw}
    before_tr = {json.loads(l)["skill"] for l in open(TRAIN, encoding="utf-8") if l.strip()}
    before_ev = {json.loads(l)["skill"] for l in open(EVAL, encoding="utf-8") if l.strip()}

    affected = sorted(s for s in before_skills if _src_chars(s) > OLD_MAX_SRC_CHARS)
    if not affected:
        print("no truncated skills to regenerate")
        return 0
    snap = {s: [r for r in raw if r["skill"] == s] for s in affected}
    keep = [r for r in raw if r["skill"] not in set(affected)]
    print(f"REGEN TRUNCATED  {len(affected)} skills / {sum(len(v) for v in snap.values())} pairs"
          f"   limit {OLD_MAX_SRC_CHARS} -> {MAX_SRC_CHARS}")
    print(f"  in train {len(set(affected) & before_tr)}   in eval {len(set(affected) & before_ev)}",
          flush=True)

    with open(RAW, "w", encoding="utf-8") as fh:
        for r in keep:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    done = {l.strip() for l in open(DONE, encoding="utf-8") if l.strip()} if os.path.exists(DONE) else set()
    with open(DONE, "w", encoding="utf-8") as fh:
        for s in sorted(done - set(affected)):
            fh.write(s + "\n")

    files = [os.path.join(SKILLS_DIR, s.replace("/", os.sep) + ".md") for s in affected]
    _stats["t0"], _stats["total"] = time.time(), len(files)
    with ThreadPoolExecutor(parallel) as ex:
        list(ex.map(_gen_one, files))

    raw2 = [json.loads(l) for l in open(RAW, encoding="utf-8") if l.strip()]
    got = {r["skill"] for r in raw2}
    lost = [s for s in affected if s not in got]
    if lost:
        print(f"  RESTORING {len(lost)} skills that failed to regenerate (snapshot intact)", flush=True)
        with open(RAW, "a", encoding="utf-8") as fh:
            for s in lost:
                for r in snap[s]:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(DONE, "a", encoding="utf-8") as fh:
            for s in lost:
                fh.write(s + "\n")
        raw2 = [json.loads(l) for l in open(RAW, encoding="utf-8") if l.strip()]

    after_skills = {r["skill"] for r in raw2}
    assert after_skills == before_skills, (
        f"skill SET changed: +{sorted(after_skills - before_skills)[:5]} "
        f"-{sorted(before_skills - after_skills)[:5]}")
    ids = [r["id"] for r in raw2]
    assert len(ids) == len(set(ids)), f"duplicate ids after regen ({len(ids)-len(set(ids))})"

    print(f"\n  pairs {len(raw)} -> {len(raw2)}   regenerated {len(affected)-len(lost)}"
          f"   restored {len(lost)}", flush=True)
    # PINNED split, never split() -- see split_preserving's docstring. Re-deriving the boundary here
    # would relocate 94 of the 100 held-out skills and void every number measured on this eval.
    split_preserving(held=before_ev)

    after_tr = {json.loads(l)["skill"] for l in open(TRAIN, encoding="utf-8") if l.strip()}
    after_ev = {json.loads(l)["skill"] for l in open(EVAL, encoding="utf-8") if l.strip()}
    assert after_tr == before_tr and after_ev == before_ev, "TRAIN/EVAL skill split MOVED"
    print(f"  split preserved: train {len(after_tr)} skills, eval {len(after_ev)} skills — identical")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_SKILLS)
    ap.add_argument("--parallel", type=int, default=PARALLEL)
    ap.add_argument("--regen-truncated", action="store_true",
                    help="rebuild QA in place for skills whose source exceeded OLD_MAX_SRC_CHARS")
    a = ap.parse_args()
    if a.regen_truncated:
        return regen_truncated(a.parallel)
    globals()["N_SKILLS"] = a.n
    os.makedirs(OUT_DIR, exist_ok=True)
    done = set()
    if os.path.exists(DONE):
        done = {l.strip() for l in open(DONE, encoding="utf-8") if l.strip()}
    print(f"generating with {GEN_MODEL}, {a.parallel}-way parallel, {QA_PER_SKILL} QA/skill",
          flush=True)
    todo = [f for f in select()
            if os.path.splitext(os.path.relpath(f, SKILLS_DIR))[0].replace("\\", "/") not in done]
    _stats["t0"], _stats["total"] = time.time(), len(todo)
    print(f"{len(done)} skills already done, {len(todo)} to generate\n", flush=True)
    if todo:
        with ThreadPoolExecutor(a.parallel) as ex:
            list(ex.map(_gen_one, todo))
    el = (time.time() - _stats["t0"]) / 60
    print(f"\ngeneration: {_stats['skills']} skills, {_stats['pairs']} pairs, {el:.1f} min "
          f"(failed {_stats['failed']}, empty {_stats['empty']}, "
          f"rejected-context-dependent {_stats['ctx_dependent']})", flush=True)
    split()


if __name__ == "__main__":
    sys.exit(main())
