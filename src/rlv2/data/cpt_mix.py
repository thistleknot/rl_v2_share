"""src/rlv2/data/cpt_mix.py — assemble the CPT corpus for the R3 experiment.

§6.3 mix, four components, all of which ALREADY EXIST — nothing is generated here:

    skill chunks     32%   raw domain prose (this is what makes CPT != SFT)
    domain SFT       43%   r3_cpt.jsonl rendered as chat-template NTP text
    EntiGraph        16%   relational passages over THE SAME skills
    RedPajama         9%   the ONLY sidecar; general text, guards against forgetting

ONE PROVENANCE CHAIN: skills -> EntiGraph -> domain SFT. All three components derive from
the same 16 trees-ensembles skills, and the skill chunks are restricted to those that
ACTUALLY produced an accepted pool item -- 25 chunks generated nothing and are excluded.
RedPajama is the only thing in the mix that is not from this domain.

WHY EVAL-SOURCE CHUNKS STAY IN. The chunk set is "produced any accepted item", not "produced
a CPT item". Excluding the chunks the eval questions were written from would mean CPT never
sees that text, which guarantees no deposit for exactly the items being measured -- rigging
the experiment toward failure. The eval QUESTIONS are absent from the corpus (verified
zero-leakage); their SOURCE MATERIAL must be present or there is nothing to learn.

The old mix (`cpt_mix_full.jsonl`, 3,755 records) predates the §3.5.1 scope cut and is
built from book extracts and a 300-skill EntiGraph draw. `src/rlv2/train/cpt/train.py`'s CORPUS_SOURCES has
the same problem — it lists book chapters, none of which are trees-ensembles. This script
builds the mix the CURRENT experiment needs.

CPT on raw chunks ALONE regressed macroF1 0.181 -> 0.124 (more confident, no better
answers), which is why all four components are required rather than just the domain text.

Run:  python -u -m rlv2 data cpt-mix --dry-run
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random

from rlv2.data import pools as bp

# ── LENGTH GATE ────────────────────────────────────────────────────────────────────
# HARD_CAP is not a tuning knob. 512 tokens is the ceiling this GPU trains at, so a record
# above it cannot be learned from -- it can only be silently truncated, which teaches the
# model to predict a sentence that stops mid-clause. Enforce it HERE, at the data layer,
# where the fix is visible in a receipt, rather than in the trainer where it becomes a
# quiet `[:512]`.
#
# Two different treatments above the cap, and the difference is deliberate:
#   SKILLS      are CHUNKED. They are the domain knowledge; dropping one loses coverage
#               of a skill the eval asks about.
#   EVERYTHING  is DROPPED. QA pairs, EntiGraph passages and RedPajama replay are
#   ELSE        interchangeable and plentiful -- an over-long one is not worth splitting.
HARD_CAP = 512
MAD_K = 1.4826          # MAD -> sigma consistency constant under normality
BAND_SIGMAS = 2.0       # median +- 2*MAD*1.4826 in Box-Cox space


def token_len(text: str, tok) -> int:
    return len(tok.encode(text, add_special_tokens=False))


def split_to_cap(text: str, tok, cap: int = HARD_CAP) -> list[str]:
    """Split an over-cap SKILL on paragraph boundaries until every piece fits.

    Paragraph-first, never a blind token slice: these chunks came out of
    `semantic_chunk.py` already aligned to structure, and cutting mid-paragraph to satisfy
    a token count destroys the boundary that made them worth chunking.

    A single paragraph that alone exceeds the cap is emitted truncated -- reported by the
    caller, and the only case where truncation is accepted.
    """
    paras = [p for p in text.split("\n\n") if p.strip()]
    out, cur = [], ""
    for p in paras:
        trial = (cur + "\n\n" + p) if cur else p
        if token_len(trial, tok) <= cap:
            cur = trial
            continue
        if cur:
            out.append(cur)
        cur = p if token_len(p, tok) <= cap else ""
        if not cur:                                  # lone paragraph over the cap
            ids = tok.encode(p, add_special_tokens=False)[:cap]
            out.append(tok.decode(ids))
    if cur:
        out.append(cur)
    return out or [text]


def boxcox_length_band(lengths: list[int], sigmas: float = BAND_SIGMAS):
    """(lo, hi) token band from median +- sigmas*MAD*1.4826 in BOX-COX space.

    Box-Cox rather than a fixed log: it finds the normalising power from the data instead
    of assuming one. Median and MAD rather than mean and sd because length distributions
    are right-skewed -- one long passage moves a mean and does not move a median.

    Returns (lo, hi) in TOKENS, back-transformed.
    """
    from scipy import stats

    vals = [max(1, int(x)) for x in lengths]
    if len(set(vals)) < 3:
        return 0, max(vals)                          # degenerate: nothing to fit
    bc, lam = stats.boxcox(vals)
    srt = sorted(bc)
    med = float(srt[len(srt) // 2])
    mad = float(sorted(abs(x - med) for x in bc)[len(bc) // 2])
    if mad == 0.0:
        return 0, max(vals)
    spread = sigmas * MAD_K * mad

    def inv(y):
        """Back-transform to tokens. The base can go NEGATIVE for a lower bound under a
        negative lambda, and a fractional power of a negative number is NaN -- which then
        crashes int() rather than producing a bound. Clamp to the transform's own domain:
        the floor of the Box-Cox range IS the shortest representable length."""
        import math
        if lam == 0:
            return math.exp(y)
        base = y * lam + 1.0
        if base <= 0:
            return 0.0
        return base ** (1.0 / lam)

    lo_t, hi_t = inv(med - spread), inv(med + spread)
    if not (hi_t == hi_t) or hi_t <= 0:          # NaN or degenerate upper bound
        return 0, max(vals)
    return max(0, int(lo_t)), int(hi_t)

# --qa overrides this. The 180-record variant (r3_cpt_all.jsonl) folds SFT+RL into CPT to
# test ONE thing: does more domain QA produce lift? 133 records regressed (base 2/20 ->
# CPT 0/20 by judge verdict), and the candidate explanations are "too little data" and
# "CPT cannot deposit this at all". Tripling the QA separates them.
QA = "data/rl/r3_cpt.jsonl"
CHUNKS = "data/rl/cpt_skills_chunked_treesx.jsonl"
ENTIGRAPH = "data/entigraph/eg_passages.jsonl"
REDPAJAMA = "data/domain/slimpajama_replay_1773.jsonl"
POOL = "data/rl/pools_r3.jsonl"
OUT = "data/rl/cpt_mix_r3.jsonl"
SEED = 20260826

# §6.3 proportions. QA is the anchor: we spend ALL of r3_cpt (60 items) and size the rest
# around it, rather than truncating the split we deliberately partitioned.
SHARE = {"skill_chunk": 0.32, "domain_sft": 0.43, "entigraph": 0.16, "redpajama": 0.09}

# UNCHANGED, DELIBERATELY. This is what CPT rendered every training record with,
# and every historical number (base, CPT, SFT) was measured under it. Operator
# 2026-09-12: "to keep things consistent ... keep system prompt".
#
# The abstention sentence below is a known hazard -- R3 has ZERO abstain items
# (0/16 eval, 0/240 train have an empty gold), so taking the invitation is always
# wrong. It is NOT removed here, because editing the prompt changes the
# instrument and discards comparability. It is handled in SCORING instead:
# reward.py marks an abstention wrong mechanically on the r3 lane.
NTP_SYSTEM = (
    "You are a helpful assistant. Answer the question directly and concisely. "
    "If the question cannot be answered from available information, say so clearly."
)


def load(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def render_ntp(q: str, a: str) -> str:
    return ("<|im_start|>system\n%s<|im_end|>\n"
            "<|im_start|>user\n%s<|im_end|>\n"
            "<|im_start|>assistant\n%s<|im_end|>" % (NTP_SYSTEM, q, a))


def entigraph_for_skills(rows: list[dict], skills: set[str]) -> list[dict]:
    """Passages relating two of OUR skills, by metadata or by mention.

    The `skills` metadata field only matches 10 rows — it was written against a different
    skill draw. The passage TEXT mentions our concepts far more often, so mention is used
    as a fallback signal rather than trusting stale metadata alone.
    """
    short = {s.split("/")[-1].replace("-", " ") for s in skills}
    keep = []
    for r in rows:
        meta = set(r.get("skills") or [])
        if meta & skills:
            keep.append((r, "metadata"))
            continue
        text = (r.get("text") or "").lower()
        if sum(1 for s in short if s in text) >= 2:   # relates TWO of our concepts
            keep.append((r, "mention"))
    return [{"row": r, "why": why} for r, why in keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=QA,
                    help="domain QA jsonl folded into the mix as NTP text")
    ap.add_argument("--no-length-band", action="store_true",
                    help="apply the HARD 512 CAP only, skipping the Box-Cox outlier band. "
                         "The band exists for the heavy-tailed SKILL CHUNKS; QA records are "
                         "already tight (p50 150, p95 231) so the band only removes "
                         "legitimate training examples -- measured 179 -> 170, a silent 5%% "
                         "loss on an arm whose whole variable is record count")
    ap.add_argument("--components", default=None,
                    help="comma-separated subset of skill_chunk,domain_sft,entigraph,"
                         "redpajama. Omit for the full mix. `--components domain_sft` is "
                         "the PURE sft-as-ntp arm: it replicates the historically proven "
                         "mechanism in isolation, so a lift or its absence is attributable "
                         "to the QA text alone rather than to a four-way blend")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rng = random.Random(SEED)
    skills = set(bp.TARGET_SKILLS)

    qa = load(a.qa)

    # Skill chunks, restricted to those that actually derived a question. chunk_id is
    # ASSIGNED at load time (skill#NN), not stored in the file, so ids must be derived the
    # same way build_pools_r3 did or nothing matches.
    bp.CHUNKS = CHUNKS
    all_chunks = bp.load_chunks()
    produced = {r.get("source_chunk_id") for r in load(POOL)}
    chunks = [c for c in all_chunks
              if c.get("skill") in skills and c["chunk_id"] in produced]
    print("chunks: %d of %d produced an accepted item (%d generated nothing)"
          % (len(chunks), len(all_chunks), len(all_chunks) - len(chunks)))
    eg = entigraph_for_skills(load(ENTIGRAPH), skills)
    redpajama = load(REDPAJAMA)

    # Size everything around the QA split we already partitioned.
    total = round(len(qa) / SHARE["domain_sft"])
    want = {k: round(total * v) for k, v in SHARE.items()}
    want["domain_sft"] = len(qa)

    rng.shuffle(chunks)
    rng.shuffle(eg)
    rng.shuffle(redpajama)
    take_chunks = chunks[: want["skill_chunk"]]
    take_eg = eg[: want["entigraph"]]
    take_redpajama = redpajama[: want["redpajama"]]

    recs: list[dict] = []
    for c in take_chunks:
        recs.append({"text": c["text"], "source": "skill_chunk",
                     "skill": c["skill"], "regime": "NA", "answerable": True})
    for r in qa:
        recs.append({"text": render_ntp(r["question"], r["gold"]), "source": "domain_sft",
                     "skill": r["source_skill"], "category": r.get("category", ""),
                     "regime": "R3", "answerable": True})
    for e in take_eg:
        recs.append({"text": e["row"]["text"], "source": "entigraph",
                     "skill": "|".join(e["row"].get("skills") or []),
                     "matched_by": e["why"], "regime": "NA", "answerable": True})
    for r in take_redpajama:
        recs.append({"text": r.get("text") or r.get("prompt") or "", "source": "redpajama",
                     "regime": "NA", "answerable": True})
    recs = [r for r in recs if (r["text"] or "").strip()]

    if a.components:
        # NOT named `want` -- that name already holds the per-component TARGET COUNTS dict
        # further down, and shadowing it made the receipt crash with a set subscript.
        keep_sources = {c.strip() for c in a.components.split(",") if c.strip()}
        unknown = keep_sources - set(SHARE)
        assert not unknown, "unknown component(s): %s (have %s)" % (
            sorted(unknown), ", ".join(SHARE))
        before = len(recs)
        recs = [r for r in recs if r["source"] in keep_sources]
        print("COMPONENTS RESTRICTED to %s: %d -> %d records"
              % (sorted(keep_sources), before, len(recs)))
        assert recs, "component restriction removed every record"

    # ── LENGTH GATE ────────────────────────────────────────────────────────────────
    # Applied to the ASSEMBLED mix so one band is fitted to the real training
    # distribution, not four separate bands per component.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("skills_model", trust_remote_code=True)

    capped: list[dict] = []
    gate = {"chunked": 0, "pieces": 0, "dropped_over_cap": 0}
    for r in recs:
        nt = token_len(r["text"], tok)
        if nt <= HARD_CAP:
            capped.append(dict(r, n_tokens=nt))
            continue
        if r["source"] == "skill_chunk":
            gate["chunked"] += 1
            for piece in split_to_cap(r["text"], tok):
                capped.append(dict(r, text=piece, n_tokens=token_len(piece, tok)))
                gate["pieces"] += 1
        else:
            gate["dropped_over_cap"] += 1

    # ONE BAND PER COMPONENT, not one band over the mix. The assembled corpus is
    # MULTIMODAL by construction -- skill chunks run p50 389 tokens while QA pairs run
    # p50 150 -- and a single median/MAD fitted across both modes sits between them and
    # calls the long mode an outlier. Measured: a global band [56, 415] threw away 18 of
    # 45 skill chunks, i.e. 40% of the domain knowledge the experiment is trying to
    # deposit, while per-component bands keep all 42 (127 records retained vs 114).
    # Outlier means "unusual FOR ITS OWN KIND", and that is only defined within a
    # component.
    by_source: dict[str, list[dict]] = collections.defaultdict(list)
    for r in capped:
        by_source[r["source"]].append(r)

    kept, bands = [], {}
    for src, group in by_source.items():
        if a.no_length_band:
            bands[src] = (0, HARD_CAP, len(group), len(group))
            kept.extend(group)
            continue
        lo, hi = boxcox_length_band([r["n_tokens"] for r in group])
        keep = [r for r in group if lo <= r["n_tokens"] <= hi]
        bands[src] = (lo, hi, len(group), len(keep))
        kept.extend(keep)
    gate.update(dropped_outlier=len(capped) - len(kept), final=len(kept))

    print()
    print("LENGTH GATE — hard cap %d, then Box-Cox median +- %.0f*MAD*%.4f PER COMPONENT"
          % (HARD_CAP, BAND_SIGMAS, MAD_K))
    print("  over cap: %d skill chunk(s) -> %d pieces | %d non-skill record(s) DROPPED"
          % (gate["chunked"], gate["pieces"], gate["dropped_over_cap"]))
    for src in sorted(bands):
        lo, hi, n_in, n_keep = bands[src]
        print("    %-12s band [%3d, %4d] tok   %3d -> %3d" % (src, lo, hi, n_in, n_keep))
    print("  %d records in -> %d out" % (len(recs), len(kept)))
    # The cap is the binding constraint; the band strips a residual tail. If the band ever
    # starts dropping a large share, the mix composition changed and the receipt should be
    # read before trusting the run.
    assert kept, "length gate removed every record"
    assert max(r["n_tokens"] for r in kept) <= HARD_CAP, (
        "a record above the hard cap survived the gate — the trainer would truncate it "
        "silently, which is the exact failure this gate exists to prevent")

    recs = kept
    rng.shuffle(recs)

    got = collections.Counter(r["source"] for r in recs)
    print("=" * 70)
    print("CPT MIX (R3) — %d records" % len(recs))
    print("=" * 70)
    print("%-14s %7s %7s %8s %8s" % ("component", "want", "got", "share", "target"))
    for k in ("skill_chunk", "domain_sft", "entigraph", "redpajama"):
        print("%-14s %7d %7d %7.1f%% %7.0f%%"
              % (k, want[k], got[k], 100 * got[k] / max(1, len(recs)), 100 * SHARE[k]))
    short = [k for k in want if got[k] < want[k]]
    if short:
        print("\nSHORT: %s" % ", ".join(
            "%s %d/%d" % (k, got[k], want[k]) for k in short))
    if got["entigraph"]:
        by = collections.Counter(r.get("matched_by") for r in recs
                                 if r["source"] == "entigraph")
        print("entigraph matched by: %s" % dict(by))
    # --- LINEAGE ASSERT (SO-1 / §6.3a) -------------------------------------------------
    # The three domain components must reference the SAME skill set. Printing percentages
    # is not enough: a mix can hit every target while EntiGraph describes a different
    # 300-skill draw than the QA — which is exactly what the old cpt_mix_full.jsonl did.
    chunk_skills = {r["skill"] for r in recs if r["source"] == "skill_chunk"}
    qa_skills = {r["skill"] for r in recs if r["source"] == "domain_sft"}
    eg_skills = set()
    for r in recs:
        if r["source"] == "entigraph":
            eg_skills |= {s for s in (r.get("skill") or "").split("|") if s}

    print("\nDERIVATION CHAIN — skills -> chunks -> {entigraph, domain sft}")
    print("  chunk skills           : %d" % len(chunk_skills))
    print("  QA skills              : %d  (also in chunks: %d)"
          % (len(qa_skills), len(qa_skills & chunk_skills)))
    print("  entigraph tagged skills: %d  (in target set: %d)"
          % (len(eg_skills), len(eg_skills & skills)))
    print("  union covered          : %d of %d target skills"
          % (len((chunk_skills | qa_skills) & skills), len(skills)))

    orphan_qa = qa_skills - skills
    assert not orphan_qa, "QA references skills outside the target set: %s" % sorted(orphan_qa)
    orphan_chunks = chunk_skills - skills
    assert not orphan_chunks, "chunks outside the target set: %s" % sorted(orphan_chunks)
    if not a.components:
        assert got["skill_chunk"] and got["domain_sft"], "a domain component is empty"
    else:
        # A restricted arm deliberately drops components; asserting all-present here would
        # block the very experiment the flag exists for.
        assert sum(got.values()), "component restriction left nothing"

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
