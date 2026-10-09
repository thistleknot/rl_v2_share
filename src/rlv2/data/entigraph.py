"""src/rlv2/data/entigraph.py — EntiGraph over the skills corpus, where the entities are already named.

WHY THIS IS THE RIGHT CORPUS FOR ENTIGRAPH. EntiGraph (arXiv:2409.07431) works by extracting
entities from a corpus and synthesising text about the RELATIONS between them. On raw book prose the
extraction step is the lossy, expensive part -- entities sit buried under coreference and an LLM has
to guess at them. This corpus hands them over: 1467 curated notes, each with an explicit `name`,
`description` and `topic` in its frontmatter. `entigraph.py` in this repo spends a whole LLM pass on
`extract_entities`; here that pass is SKIPPED because the skill titles ARE the entity set.

WHAT IT ADDS THAT OUR QA DOES NOT. `build_skills_sft.py` generates WITHIN-skill questions -- five
ways of asking about one concept. Nothing in that pipeline ever relates two concepts. A corpus of
1467 isolated notes teaches 1467 isolated facts, and the relational structure between them --
"regularisation trades variance for bias", "bagging and boosting differ in how they weight
errors" -- is exactly what is missing. That structure is what EntiGraph synthesises.

PAIRS ARE SAMPLED WITHIN TOPIC. A relation between "Bayes decision boundary" and "gradient
clipping" is mostly vacuous, and asking an LLM for one produces confident filler. Same-topic pairs
have real relations to find. Triples add the transitive structure pairs cannot express.

ONE GENERATION, TWO DELIVERY CHANNELS. Each call returns a relational PASSAGE and two QA pairs about
that same relation. This is deliberate: it lets the downstream A/B compare CPT-on-passages against
SFT-on-QA with the CONTENT HELD IDENTICAL. Generating them separately would confound the delivery
channel with whatever the second generation pass happened to say.

COVERAGE CONDITION -- LOAD-BEARING. Held-out skills MUST appear in the relation set. The whole test
is whether a fact the model met only through CPT/relational text becomes answerable when its
within-skill QA was never trained. If held-out skills were excluded from generation, CPT would be
asked for facts it never saw and the null would be rigged. This mirrors T0/T14, where measured
coverage was 0.938 and the CPT arm still deposited nothing -- the condition held there and holds here.
"""
import argparse
import glob
import io
import itertools
import json
import os
import random
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# BACKEND SWITCH. NVIDIA's hosted endpoint is the default, but it returns 429 on EVERY request once
# quota is exhausted -- and no backoff rides that out, so a fallback is not optional. Set
# GEN_BACKEND=local to route through ollama with an identical `complete()` signature.
if os.environ.get("GEN_BACKEND", "nvidia").lower() == "local":
    from rlv2.judge import local_llm as NV
else:
    from rlv2.judge import nvidia_llm as NV

SKILLS_DIR = r"C:\Users\user\Documents\wiki\data science\stackskills\skills_master"
# Overridable so a full-corpus run writes to its OWN directory. The 300-skill subset and the full
# corpus draw DIFFERENT relation sets -- pairs are sampled within topic from whichever skill list is
# in play -- so writing both to one file would silently blend two experiments into one corpus.
OUT_DIR = os.environ.get("EG_OUT_DIR", "data/entigraph")
PASSAGES = f"{OUT_DIR}/eg_passages.jsonl"     # CPT channel: continuous relational text
REL_QA = f"{OUT_DIR}/eg_relation_qa.jsonl"    # SFT channel: QA about the same relations
DONE = f"{OUT_DIR}/.done_rels.txt"
SUBSET = f"{OUT_DIR}/subset_skills.json"      # the skill list THIS run is scoped to

# Inference runs on NVIDIA's hosted endpoint, NOT local ollama. Generation no longer contends with
# training for the single 16 GB GPU, so this stage overlaps training instead of blocking it -- which
# is the entire reason it is worth the network dependency. See nvidia_llm.py for the rate-limit gate.
N_RELATIONS = 400
TRIPLE_FRAC = 0.25
PARALLEL = 8
SEED = 7

SYSTEM = (
    "You are writing a reference passage that RELATES several data-science concepts to each other.\n"
    "You are given 2-3 concepts, each with a short description.\n"
    "Return EXACTLY this, nothing else:\n"
    "PASSAGE: <120-200 words relating the concepts: how they interact, when one implies or "
    "constrains the other, where they are confused, which to reach for and why. Name every concept "
    "explicitly. State only what follows from the descriptions given -- do not invent results, "
    "citations, or numbers.>\n"
    'Q1: {"q": "...", "a": "..."}\n'
    'Q2: {"q": "...", "a": "..."}\n'
    "Both questions must be about the RELATION between the concepts, not about one alone, must name "
    "the concepts explicitly so they stand alone, and must have 1-3 sentence factual answers."
)

_lock = threading.Lock()
_stats = defaultdict(int)


def _norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def load_skills(paths):
    out = []
    for p in paths:
        src = io.open(p, encoding="utf-8", errors="replace").read()
        sid = os.path.splitext(os.path.relpath(p, SKILLS_DIR))[0].replace("\\", "/")
        nm = re.search(r"^name:\s*(.+)$", src, re.M)
        ds = re.search(r'^description:\s*"?(.+?)"?\s*$', src, re.M)
        out.append({"id": sid, "topic": sid.split("/")[0],
                    "name": _norm(nm.group(1)) if nm else sid.rsplit("/", 1)[-1],
                    "desc": _norm(ds.group(1)) if ds else ""})
    return out


def sample_relations(skills, n, rng, cov=None):
    """Within-topic pairs, then triples. Cross-topic relations are mostly vacuous filler.

    COVERAGE-AWARE when `cov` is supplied (skill_id -> how many relations already name it).
    Uniform sampling over a corpus that is already partly covered spends most of its budget
    re-describing skills that already have five relations, while others still have none. Since the
    readout asks whether a HELD-OUT skill became answerable, a skill with zero relations is a
    guaranteed miss no matter how good the generator is -- so budget goes to the thin end first.

    Ordering is by total existing coverage ascending, with a random tiebreak so equally-uncovered
    pairs are not drawn in a fixed alphabetical order (which would concentrate on whichever topics
    sort first).
    """
    by_topic = defaultdict(list)
    for s in skills:
        by_topic[s["topic"]].append(s)
    pairs, triples = [], []
    for t, ss in by_topic.items():
        if len(ss) >= 2:
            pairs += list(itertools.combinations(ss, 2))
        if len(ss) >= 3:
            tri = list(itertools.combinations(ss, 3))
            triples += rng.sample(tri, min(len(tri), 400))
    rng.shuffle(pairs)
    rng.shuffle(triples)
    if cov is not None:
        key = lambda grp: (sum(cov.get(s["id"], 0) for s in grp), rng.random())
        pairs.sort(key=key)
        triples.sort(key=key)
    n_tri = int(n * TRIPLE_FRAC)
    return [list(x) for x in pairs[:n - n_tri]] + [list(x) for x in triples[:n_tri]]


def _parse(txt, rel):
    m = re.search(r"PASSAGE:\s*(.+?)(?=\n\s*Q1:|\Z)", txt, re.S)
    passage = _norm(m.group(1)) if m else ""
    qa = []
    for mm in re.finditer(r"\{[^{}]*\"q\"\s*:.*?\}", txt, re.S):
        try:
            o = json.loads(mm.group(0))
        except json.JSONDecodeError:
            continue
        q, a = _norm(o.get("q")), _norm(o.get("a"))
        if len(q) < 20 or len(a) < 20:
            continue
        if re.search(r"\b(this|the above|the passage|the note)\b", q, re.I):
            _stats["ctx_dependent"] += 1
            continue
        qa.append({"prompt": q, "answer": a, "skills": [s["id"] for s in rel],
                   "topic": rel[0]["topic"], "kind": "relation"})
    return passage, qa


def _gen(rel):
    key = "|".join(sorted(s["id"] for s in rel))
    user = "\n".join(f"CONCEPT: {s['name']}\nDESCRIPTION: {s['desc']}" for s in rel)
    txt = NV.complete(SYSTEM, user, max_tokens=900, temperature=0.8)
    if txt is None:
        with _lock:
            _stats["failed"] += 1        # RECORDED, never silently skipped
        return
    passage, qa = _parse(txt, rel)
    if len(passage) <= 200:
        with _lock:
            _stats["short"] += 1
        return
    with _lock:
        with io.open(PASSAGES, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "topic": rel[0]["topic"],
                                 "skills": [s["id"] for s in rel],
                                 "text": passage}, ensure_ascii=False) + "\n")
        with io.open(REL_QA, "a", encoding="utf-8") as fh:
            for r in qa:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        io.open(DONE, "a", encoding="utf-8").write(key + "\n")
        _stats["rels"] += 1
        _stats["qa"] += len(qa)
        _stats["words"] += len(passage.split())
        if _stats["rels"] % 25 == 0:
            el = time.time() - _stats["t0"]
            r_ = el / _stats["rels"]
            print(f"  {_stats['rels']}/{_stats['total']} relations, {_stats['qa']} QA, "
                  f"{r_:.2f}s each, ETA {(_stats['total']-_stats['rels'])*r_/60:.0f} min",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-skills", type=int, default=300)
    ap.add_argument("--n-relations", type=int, default=N_RELATIONS)
    ap.add_argument("--parallel", type=int, default=PARALLEL)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(SEED)

    if os.path.exists(SUBSET):
        sel = json.load(io.open(SUBSET, encoding="utf-8"))
    else:
        # Prefer skills that ALREADY have within-skill QA from build_skills_sft.py. The A/B needs
        # both channels for the same skills, and regenerating QA for a fresh 300 would cost another
        # ~30 min of ollama for no experimental gain.
        have = set()
        if os.path.exists("data/skills/.done_skills.txt"):
            have = {l.strip() for l in io.open("data/skills/.done_skills.txt", encoding="utf-8")
                    if l.strip()}
        files = [f for f in glob.glob(os.path.join(SKILLS_DIR, "**", "*.md"), recursive=True)
                 if "MASTER_TOC" not in f]
        if have:
            files = [f for f in files
                     if os.path.splitext(os.path.relpath(f, SKILLS_DIR))[0].replace("\\", "/")
                     in have] or files
            print(f"restricting to {len(files)} skills that already have within-skill QA",
                  flush=True)
        by_topic = defaultdict(list)
        for f in files:
            by_topic[os.path.relpath(f, SKILLS_DIR).split(os.sep)[0]].append(f)
        picked = []
        for t, fs in sorted(by_topic.items()):
            fs = sorted(fs)
            rng.shuffle(fs)
            picked += fs[:max(2, round(a.n_skills * len(fs) / len(files)))]
        rng.shuffle(picked)
        picked = picked[:a.n_skills]
        skills = load_skills(picked)
        # 100 held out: their WITHIN-skill QA is never trained, but their text IS in the relations.
        rng.shuffle(skills)
        sel = {"held_out": [s["id"] for s in skills[:100]],
               "train": [s["id"] for s in skills[100:]],
               "all": {s["id"]: s for s in skills}}
        json.dump(sel, io.open(SUBSET, "w", encoding="utf-8"), indent=1)
        print(f"subset: {len(sel['train'])} train + {len(sel['held_out'])} held-out skills",
              flush=True)

    skills = list(sel["all"].values())
    done = set()
    if os.path.exists(DONE):
        done = {l.strip() for l in io.open(DONE, encoding="utf-8") if l.strip()}
    # Existing coverage drives where the NEW budget goes. Relations already on disk are counted,
    # so a resumed or seeded run spends on the skills that still have none rather than re-describing
    # the ones that already have five.
    prior = defaultdict(int)
    for k in done:
        for sid in k.split("|"):
            prior[sid] += 1
    if prior:
        thin = sum(1 for s in skills if prior.get(s["id"], 0) == 0)
        print(f"{len(done)} relations already on disk covering {len(skills)-thin}/{len(skills)} "
              f"skills; {thin} still uncovered -> sampling is biased toward them", flush=True)
    rels = sample_relations(skills, a.n_relations, rng, cov=prior if prior else None)
    todo = [r for r in rels if "|".join(sorted(s["id"] for s in r)) not in done]
    # Coverage below is measured over EVERYTHING that will exist when this run finishes -- the
    # relations already on disk plus the ones about to be generated -- not just the new batch.
    final = {s for r in rels for s in [x["id"] for x in r]} | set(prior)
    held_cov = len(final & set(sel["held_out"])) / max(1, len(sel["held_out"]))
    print(f"{len(rels)} relations ({int(a.n_relations*TRIPLE_FRAC)} triples), {len(todo)} to generate",
          flush=True)
    print(f"COVERAGE of held-out skills by the relation set: {held_cov:.3f} "
          f"-- the CPT arm can only be fair if this is high", flush=True)
    _stats["t0"], _stats["total"] = time.time(), len(todo)
    if todo:
        with ThreadPoolExecutor(a.parallel) as ex:
            list(ex.map(_gen, todo))
    el = (time.time() - _stats["t0"]) / 60
    print(f"\nEntiGraph: {_stats['rels']} relations, ~{_stats['words']:,} passage words, "
          f"{_stats['qa']} relational QA, {el:.1f} min "
          f"(failed {_stats['failed']}, short {_stats['short']}, "
          f"ctx-dependent rejected {_stats['ctx_dependent']})", flush=True)


if __name__ == "__main__":
    main()
