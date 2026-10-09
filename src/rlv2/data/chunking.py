"""src/rlv2/data/chunking.py — build the CPT corpus: skill notes -> canonical atoms -> packed 512-tok records.

THE GATE, MEASURED AGAINST THE WINDOW WE ACTUALLY TRAIN AT. `entigraph_ab.py:64` sets BLOCK=512.
Against 512, **1519/1524 skill notes (99.7%) exceed the window** -- chunking is REQUIRED. (Measured
first against 2048, the model's architectural context, which no training script in this repo uses:
0.1% exceed, verdict "do not chunk". Same corpus, opposite answer. Read the window out of the config.)

TWO OVERLAPS, TWO OBJECTS -- the catch-22 and why it is not one.

    chunk + de-overlap  -> ATOMS   canonical units that TILE the source. overlap here = a span stored
                                   twice = a defect. de-overlap is a correctness assert.
    pack into records   -> RECORDS what the trainer consumes. overlap here = a span SEEN twice =
                                   exposure count = a dose you choose.

They annihilate only if conflated. Windowing atoms at size m with stride m-d and THEN de-overlapping
trims exactly the shared span, collapsing every window back to one unit: measured at m=2,d=1 that
degenerated to "emit each paragraph alone" -- 38.8% single-paragraph chunks, 10.5% under 5 words.
So atoms are built with NO overlap (stride = m) and de-overlap must remove ~0 chars; the ROLLING
WINDOW lives in the packing stage instead, where its stride is the exposure knob.

WHY PACK TO A TOKEN BUDGET rather than to a paragraph count. Two sections ~= 110 words ~= 150 tok =
29% of a 512 window. Greedy accumulation to the budget reaches 88-90% fill at every stride AND
self-balances record length, which removes the need to sub-split ragged sections to equalise them.

STRIDE. Not a chunking parameter -- exposure. 1 -> 2.67x, 2 -> 1.72x, 3 -> 1.32x. T7 (the local
exposure-knee experiment) is NOT RUN, so the default comes from the literature prior: canon.md:357
cites Physics of LMs 3.3 at ~1000 exposures/fact for capacity against 4 epochs here, i.e. ~10
effective exposures at stride 1 -- two orders of magnitude below saturation, so the risk is cost,
not over-exposure.

    python semantic_chunk.py [--stride 1] [--budget 512] [--no-show]
"""
import argparse
import glob
import io
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, ".")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SKILLS_DIR = r"C:\Users\user\Documents\wiki\data science\stackskills\skills_master"
EXCLUDE = {"MASTER_TOC.md"}          # a table of contents, not prose: 72,967 words vs p95 958
HDR = re.compile(r"^#{1,6}\s", re.M)
PARA = re.compile(r"\n\s*\n")
SEP = "\n\n"
OUT = "data/cpt_chunks.jsonl"
PNG = "data/chunk_lengths.png"


# ---------------------------------------------------------------- de-overlap (operator's algorithm)
def find_overlap(s1, s2):
    """Longest suffix of s1 that matches a prefix of s2.

    EXACT matching, not difflib.SequenceMatcher: stride-induced overlap is character-identical, so
    exact is both correct and O(n) per pair -- and fuzzy would trim text that merely RESEMBLES the
    neighbour."""
    for i in range(min(len(s1), len(s2)), 0, -1):
        if s1[-i:] == s2[:i]:
            return s1[-i:]
    return ""


def de_overlap_strings(strings):
    """Remove overlapping suffixes from consecutive strings. Concatenation reconstructs the source
    exactly when the chunks tile it."""
    if not strings:
        return []
    result = []
    for i in range(len(strings) - 1):
        s1, s2 = strings[i], strings[i + 1]
        ov = find_overlap(s1, s2)
        if ov:
            s1 = s1[:-len(ov)]
        result.append(s1)
    result.append(strings[-1])
    return result


def self_test():
    s = "abcdefghijklmnopqrstuvwxyz"
    red = de_overlap_strings([s[0:13], s[8:15], s[14:]])
    assert red == ["abcdefgh", "ijklmn", "opqrstuvwxyz"], red
    assert "".join(red) == s, "de-overlap must reconstruct the source exactly"
    return True


# ---------------------------------------------------------------- stats / structure
def mad(xs):
    m = st.median(xs)
    return st.median([abs(x - m) for x in xs]) or 1


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))]


def sections(text):
    """[paragraphs] per section; zero-paragraph sections DROPPED (bare headers carry no content and
    would drag the median toward 1)."""
    out = []
    for part in HDR.split(text):
        part = part.strip()
        if not part:
            continue
        paras = [p.strip() for p in PARA.split(part) if p.strip()]
        if paras:
            out.append(paras)
    return out


def atoms_for(text, m, d):
    """Canonical units that TILE the source: merge sections below m paragraphs, split above m+2d with
    stride = m (NO overlap). De-overlap is then an assert, not a transformation."""
    out, buf = [], []
    for paras in sections(text):
        if len(paras) > m + 2 * d:
            if buf:
                out.append(buf)
                buf = []
            for i in range(0, len(paras), m):
                out.append(paras[i:i + m])
        elif len(paras) < m:
            buf += paras
            if len(buf) >= m:
                out.append(buf)
                buf = []
        else:
            if buf:
                out.append(buf)
                buf = []
            out.append(paras)
    if buf:
        out.append(buf)
    return [SEP.join(p) for p in out]


def pack(units, lens, budget, stride=None):
    """Greedy accumulation to `budget` tokens, window stepping by `stride` units.

    Guarantees full coverage: the window always advances, and the tail is emitted even when the last
    step lands short of a full budget."""
    recs, i, n = [], 0, len(units)
    while i < n:
        acc, tot, j = [], 0, i
        while j < n and tot + lens[j] <= budget:
            acc.append(j)
            tot += lens[j]
            j += 1
        if not acc:                       # a single unit exceeding the budget: emit it, truncation
            acc, j = [i], i + 1           # is the trainer's job, not ours
        recs.append(acc)
        if j >= n:
            break
        # stride=None -> CONTIGUOUS: advance to where this record ended, so every atom is used
        # EXACTLY ONCE. A fixed numeric stride cannot express that: records hold a VARYING number of
        # atoms (budget fills faster when atoms are large), so `i += k` skips whichever atoms did not
        # fit. Measured 2026-08-03: stride 7 against a 7-atom AVERAGE dropped 46% of the corpus, and
        # an "exposure-matched" arm that silently trains on half the text measures corpus size.
        i = j if stride is None else i + stride
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1, help="record-window stride in atoms = exposure knob")
    ap.add_argument("--budget", type=int, default=512, help="tokens per record (archive/sprawl/entigraph_ab.py:64 BLOCK)")
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()
    assert self_test()

    files = [f for f in glob.glob(os.path.join(SKILLS_DIR, "**", "*.md"), recursive=True)
             if os.path.basename(f) not in EXCLUDE]
    texts = {f: io.open(f, encoding="utf-8", errors="replace").read() for f in files}

    from transformers import AutoTokenizer
    from rlv2.common.config import MODEL_NAME
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])

    # ---- gate ----
    doc_t = [ntok(t) for t in texts.values()]
    over = sum(1 for t in doc_t if t > a.budget)
    print(f"CORPUS  {len(files)} skill notes (excluded {sorted(EXCLUDE)})   tokenizer {MODEL_NAME}")
    print(f"  document tokens  p50 {q(doc_t,.5)}  p95 {q(doc_t,.95)}  max {max(doc_t)}")
    print(f"  exceed {a.budget:>4} tok  {over}/{len(doc_t)} ({over/len(doc_t):.1%})"
          f"  -> GATE: {'CHUNK REQUIRED' if over/len(doc_t) > .05 else 'DO NOT CHUNK'}")

    # ---- derive parameters PER DOCUMENT ----
    # The measured win used PER-SKILL derivation (archive/sprawl/chunker_ab.py): each note's own section statistics,
    # not the corpus-wide median. That is also what the method specifies -- "derive chunk size from the
    # document's OWN section statistics". A corpus-wide m/d was what this file originally used, and it
    # produced ~4 atoms per record against per-skill's ~7-9, i.e. a materially different corpus.
    # The +0.150 CPT deposit (interaction p=0.0046) was measured on the PER-SKILL variant, so shipping
    # the corpus-wide one would ship a corpus nothing was measured on.
    per_sec = [len(p) for t in texts.values() for p in sections(t)]
    m, d = int(st.median(per_sec)), int(mad(per_sec))          # reported only, for comparison
    print(f"\nSECTIONS  {len(per_sec)}   paragraphs/section median m={m}  MAD d={d}  upper m+2d={m+2*d}")

    # ---- atoms ----
    # DE-OVERLAP IS NOT APPLIED HERE, and that is the point. Atoms are built with stride = m, so they
    # are disjoint by construction and there is nothing to remove. Running de_overlap_strings over
    # them anyway trimmed 310 chars across 15,871 atoms -- COINCIDENTAL suffix/prefix collisions in
    # markdown (a code fence closing one atom and opening the next, a repeated '---' rule), not
    # structural overlap. On disjoint input every such trim is DATA LOSS wearing dedup's clothes.
    # So disjointness is verified by CONSERVATION instead: every source paragraph appears in exactly
    # one atom. That is the property we actually want, and unlike reconstruction-equality it survives
    # the header text that HDR.split discards.
    per_skill, n_src, n_out = {}, 0, 0
    for f, t in texts.items():
        _ps = [len(p) for p in sections(t)]
        if not _ps:
            continue
        _m, _d = int(st.median(_ps)), int(mad(_ps))            # THIS document's own statistics
        u = atoms_for(t, _m, _d)
        n_src += sum(len(p) for p in sections(t))
        n_out += sum(len([x for x in PARA.split(c) if x.strip()]) for c in u)
        sid = os.path.splitext(os.path.relpath(f, SKILLS_DIR))[0].replace("\\", "/")
        per_skill[sid] = [x for x in u if x.strip()]
    at = [x for v in per_skill.values() for x in v]
    at_t = [ntok(x) for x in at]
    print(f"\nATOMS  {len(at)}  ({len(at)/len(files):.1f}/skill)   "
          f"paragraph conservation {n_out}/{n_src}")
    assert n_out == n_src, f"atoms lost or duplicated paragraphs: {n_out} vs {n_src}"
    print(f"  tokens  p25 {q(at_t,.25)}  p50 {q(at_t,.50)}  p75 {q(at_t,.75)}  p95 {q(at_t,.95)}  max {max(at_t)}")

    # ---- pack ----
    idx, rows, cov = 0, [], 0
    off = {}
    for sid, us in per_skill.items():
        off[sid] = idx
        idx += len(us)
    for sid, us in per_skill.items():
        lens = [ntok(x) for x in us]
        seen = set()
        for acc in pack(us, lens, a.budget, a.stride):
            txt = SEP.join(us[k] for k in acc)
            seen |= set(acc)
            rows.append({"skill": sid, "text": txt, "atoms": len(acc),
                         "tokens": sum(lens[k] for k in acc)})
        cov += len(seen) == len(us)
    rt = [r["tokens"] for r in rows]
    exposure = sum(r["atoms"] for r in rows) / max(1, len(at))
    print(f"\nRECORDS  {len(rows)}   budget {a.budget}   stride {a.stride}")
    print(f"  tokens  p25 {q(rt,.25)}  p50 {q(rt,.50)}  p75 {q(rt,.75)}  p95 {q(rt,.95)}  max {max(rt)}")
    print(f"  fill    p50 {q(rt,.50)/a.budget:.0%}   atoms/record p50 {q([r['atoms'] for r in rows],.5)}")
    print(f"  EXPOSURE {exposure:.2f}x   coverage {cov}/{len(per_skill)} skills complete")
    assert cov == len(per_skill), "a skill lost atoms during packing"
    assert max(rt) <= max(a.budget, max(at_t)), "a record exceeded the budget without cause"

    os.makedirs("data", exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {OUT}")

    # ---- histogram ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.hist(at_t, bins=range(0, a.budget + 16, 8), alpha=.45, color="#999999",
            label=f"atoms (n={len(at)}, p50 {q(at_t,.5)} tok)", edgecolor="none")
    ax.hist(rt, bins=range(0, a.budget + 16, 8), alpha=.80, color="#2b7bba",
            label=f"packed records (n={len(rows)}, p50 {q(rt,.5)} tok)", edgecolor="none")
    ax.axvline(a.budget, color="#d1495b", ls="--", lw=2, label=f"budget {a.budget} tok")
    ax.axvline(q(rt, .5), color="#edae49", ls="--", lw=1.8, label=f"record p50 {q(rt,.5)}")
    ax.set_xlabel(f"tokens   —   atoms m={m} paragraphs (no overlap)  |  records stride={a.stride}, "
                  f"exposure {exposure:.2f}x, fill {q(rt,.5)/a.budget:.0%}")
    ax.set_ylabel("count")
    ax.set_title(f"CPT corpus: {len(at)} atoms packed into {len(rows)} records @ {a.budget} tok "
                 f"({len(files)} skill notes)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(PNG, dpi=132)
    print(f"  -> {PNG}")
    if not a.no_show:
        os.startfile(os.path.abspath(PNG))
    return 0


if __name__ == "__main__":
    sys.exit(main())
