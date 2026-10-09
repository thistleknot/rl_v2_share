"""
stratify.py -- one joint regime x domain stratification schema, shared by SFT,
rubric, and RL-batch sampling so all three see the same class mix (incl. abstention).

Regime is the FREE deterministic provenance rule (no model pass):
  R1 ctx&ans | R2 ctx&!ans | R3 !ctx&ans | R4 !ctx&!ans
Allocation across the non-empty (regime, domain) cells reuses the Box-Cox/CDF
machinery in pull_datasets.calculate_allocations; its `lam` uniform-blend knob is the
abstention over-representation lever (lower lam -> rarer cells boosted harder). Rare
cells are drawn WITH replacement (repetition) to meet their allocation.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def regime_of(record: dict) -> str:
    """R1/R2/R3/R4 from (context present, answer present).

    ACCEPTS BOTH `answer` AND `gold`. Earned 2026-08-02: `build_rl_mix.py` renames `answer` -> `gold`
    when it builds `rl_mix.jsonl`, so this function -- reading only `answer` -- would score has_ans
    False on EVERY row and classify the whole RL corpus as R2/R4. That silent incompatibility is a
    large part of why the RL mix was rebuilt WITHOUT this stratifier at all, losing the abstention
    floor that produced the +0.091 macro-F1 lift in July (commit 3b10fc8). A key rename should not be
    able to void a stratification schema.
    """
    has_ctx = bool((record.get("context") or "").strip())
    has_ans = bool((record.get("answer") or record.get("gold") or "").strip())
    if has_ctx and has_ans:
        return "R1"
    if has_ctx:
        return "R2"
    if has_ans:
        return "R3"
    return "R4"


def is_abstain(record: dict) -> bool:
    return regime_of(record) in ("R2", "R4")


def cell_key(record: dict) -> tuple:
    return (regime_of(record), record.get("domain", "unknown"))


def cell_counts(records: list) -> dict:
    counts = defaultdict(int)
    for r in records:
        counts[cell_key(r)] += 1
    return dict(counts)


def proportional_allocations(sizes: list, n: int, min_per_cell: int = 0) -> list:
    """RAW-proportional allocation: draw_i = round(n * size_i / total), largest-remainder
    to sum exactly min(n, total), CAPPED at size_i. Because draw_i <= size_i whenever
    n <= total, the draw never exceeds the unique pool -> NO repetition. This is the
    `lam=None` mode; unlike calculate_allocations (Box-Cox/CDF, which compresses and
    over-weights small cells even at lam=1.0), it preserves the true corpus proportions.

    min_per_cell floors EVERY non-empty cell to >= min(min_per_cell, size) BEFORE the
    proportional remainder, so full category representation is guaranteed at small n (a
    50-record RL batch otherwise rounds the ~0.8% abstention cells to zero). min_per_cell=0
    reproduces the exact original behavior."""
    total = sum(sizes)
    if total <= 0 or n <= 0:
        return [0] * len(sizes)
    # floor: guarantee every non-empty cell is represented (capped at its pool size)
    allocs = [min(min_per_cell, s) for s in sizes]
    if sum(allocs) >= n:                          # floors alone meet/exceed n: trim largest pools, keep breadth
        over = sum(allocs) - n
        order = sorted(range(len(sizes)), key=lambda i: sizes[i], reverse=True)
        g = 0
        while over > 0 and g < len(order) * 50:
            i = order[g % len(order)]
            if allocs[i] > 0:
                allocs[i] -= 1
                over -= 1
            g += 1
        return allocs
    # distribute the remainder proportionally over residual capacity
    remaining = n - sum(allocs)
    residual = [s - a for s, a in zip(sizes, allocs)]
    tot_res = sum(residual) or 1
    exact = [remaining * r / tot_res for r in residual]
    add = [min(int(e), r) for e, r in zip(exact, residual)]
    rem = remaining - sum(add)
    order = sorted(range(len(sizes)), key=lambda i: exact[i] - int(exact[i]), reverse=True)
    guard = 0
    while rem > 0 and guard < len(order) * 20:
        idx = order[guard % len(order)]
        if add[idx] < residual[idx]:
            add[idx] += 1
            rem -= 1
        guard += 1
    return [a + b for a, b in zip(allocs, add)]


def stratified_sample(records: list, n: int, lam=None, seed: int = 0, min_per_cell: int = 0) -> list:
    """Draw n records stratified over joint (regime, domain) cells. lam=None -> raw
    proportional (no over-representation, no repetition); lam in [0,1] -> Box-Cox/CDF
    over-representation (rare cells boosted, sampled WITH replacement). Empty cells excluded.
    min_per_cell floors every non-empty cell to guarantee full category representation at small n
    (the lam=None path only; the Box-Cox path already over-weights rare cells)."""
    if not records or n <= 0:
        return []
    groups = defaultdict(list)
    for r in records:
        groups[cell_key(r)].append(r)
    cells = sorted(groups.keys())
    sizes = [len(groups[c]) for c in cells]
    if lam is None:
        allocs = proportional_allocations(sizes, n, min_per_cell=min_per_cell)
    else:
        from rlv2.data.pull_datasets import calculate_allocations  # lazy: avoid heavy import
        allocs = calculate_allocations(sizes, n, lam=lam)
    rng = random.Random(seed)
    out = []
    for cell, k in zip(cells, allocs):
        pool = groups[cell]
        if k <= 0:
            continue
        if k <= len(pool):
            out.extend(rng.sample(pool, k))
        else:
            out.extend(pool)
            out.extend(rng.choices(pool, k=k - len(pool)))
    rng.shuffle(out)
    return out


def allocation_table(records: list, n: int, lam=None, min_per_cell: int = 0) -> list:
    """Rows (cell, raw_count, raw_pct, draw_count, draw_pct) for the joint
    regime x domain stratification -- the sampling ratios, made inspectable.
    lam=None -> raw proportional; lam in [0,1] -> Box-Cox/CDF over-representation."""
    groups = defaultdict(list)
    for r in records:
        groups[cell_key(r)].append(r)
    cells = sorted(groups.keys())
    sizes = [len(groups[c]) for c in cells]
    total = sum(sizes) or 1
    if lam is None:
        allocs = proportional_allocations(sizes, n, min_per_cell=min_per_cell)
    else:
        from rlv2.data.pull_datasets import calculate_allocations
        allocs = calculate_allocations(sizes, n, lam=lam)
    rows = []
    for (reg, dom), sz, al in zip(cells, sizes, allocs):
        rows.append((f"{reg}|{dom}", sz, 100.0 * sz / total, al, 100.0 * al / max(n, 1)))
    return rows


def print_allocation_table(records: list, n: int, lam: float = 0.5, label: str = "", min_per_cell: int = 0) -> None:
    """Print the joint regime x domain sampling ratios (raw vs drawn). A core,
    always-on pipeline artifact: the over-representation is SHOWN, not asserted."""
    rows = allocation_table(records, n, lam=lam, min_per_cell=min_per_cell)
    tag = f" [{label}]" if label else ""
    print(f"  stratification ratios{tag}  (lam={lam}, n={n}):")
    print(f"    {'cell':<20}{'raw':>6}{'raw%':>8}{'draw':>6}{'draw%':>8}")
    for name, sz, rawp, al, alp in rows:
        print(f"    {name:<20}{sz:>6}{rawp:>7.1f}%{al:>6}{alp:>7.1f}%")
    raw_ab  = sum(sz for nm, sz, rp, al, ap in rows if nm.startswith(("R2|", "R4|")))
    draw_ab = sum(al for nm, sz, rp, al, ap in rows if nm.startswith(("R2|", "R4|")))
    tot = sum(sz for _, sz, _, _, _ in rows) or 1
    print(f"    abstain(R2/R4): raw {100.0*raw_ab/tot:.1f}%  ->  drawn {100.0*draw_ab/max(n,1):.1f}%")


class Corpus:
    """Single source of truth: the corpus (data/raw/all.jsonl) + the split manifest +
    the joint regime x domain stratification schema. Owns lam / abstention floors / set
    sizes so they are defined ONCE; every consumer (partition, SFT, rubric, RL batch,
    measurement) goes through it instead of re-implementing load + filter + sample."""

    def __init__(self, data_path=None, manifest_path=None, lam=None,
                 abstain_floor=None, set_sizes=None, partition_seed=99):
        # lam=None -> RAW-proportional sampling: draw == the cell's slice, so no repetition and
        # the draw matches the (expanded) corpus composition (~8.5% abstention). NOT the same as
        # calculate_allocations(lam=1.0), which CDF-compresses and over-weights small cells.
        # DEFAULT lam=None is the SFT / proportional path (:RegimePrior:); the RL batch passes
        # lam=0.5 EXPLICITLY (Box-Cox/CDF, matches f1_grid scoring) — see sample_batch.
        # abstain_floor: the ~0.8% R4 cells round to 0 in a 50-record rubric split under the
        # natural rate, so the 9B authored the rubric BLIND to R4 (0/6 cells). Floor them into
        # rubric/holdout so every class is present for judge calibration + abstain measurement.
        self.data_path      = Path(data_path) if data_path else _HERE / "data/raw/all.jsonl"
        self.manifest_path  = Path(manifest_path) if manifest_path else _HERE / "data/manifest.json"
        self.lam            = lam
        self.abstain_floor  = abstain_floor if abstain_floor is not None else {}
        self.set_sizes      = set_sizes or {"sft": 400, "eval_holdout": 100, "rubric": 50}
        self.partition_seed = partition_seed
        self._records  = None
        self._manifest = None

    # ---- loading (the ONE copy of load + filter) ----
    def records(self) -> list:
        if self._records is None:
            with open(self.data_path, encoding="utf-8") as f:
                self._records = [json.loads(l) for l in f if l.strip()]
        return self._records

    def manifest(self) -> dict:
        if self._manifest is None:
            if not self.manifest_path.exists():
                raise FileNotFoundError(f"{self.manifest_path} not found -- run archive/sprawl/partition_data.py first")
            self._manifest = json.load(open(self.manifest_path, encoding="utf-8"))
        return self._manifest

    def split(self, name: str) -> list:
        qs = set(self.manifest()["splits"][name]["questions"])
        return [r for r in self.records() if r.get("question", "") in qs]

    def _resolve(self, src):
        return self.split(src) if isinstance(src, str) else src

    # ---- schema passthroughs ----
    def regime(self, rec) -> str:      return regime_of(rec)
    def cell(self, rec) -> tuple:      return cell_key(rec)
    def is_abstain(self, rec) -> bool: return is_abstain(rec)
    def counts(self, recs) -> dict:    return cell_counts(recs)

    # ---- sampling / reporting ----
    def sample(self, src, n, seed=0, lam=None, report=False, label=None, min_per_cell=0) -> list:
        recs  = self._resolve(src)
        lam   = self.lam if lam is None else lam
        label = label if label is not None else (src if isinstance(src, str) else "")
        if report:
            print_allocation_table(recs, n=n, lam=lam, label=label, min_per_cell=min_per_cell)
        return stratified_sample(recs, n=n, lam=lam, seed=seed, min_per_cell=min_per_cell)

    def report(self, src, n, lam=None, label=None, min_per_cell=0) -> None:
        recs  = self._resolve(src)
        label = label if label is not None else (src if isinstance(src, str) else "")
        print_allocation_table(recs, n=n, lam=(self.lam if lam is None else lam), label=label,
                               min_per_cell=min_per_cell)

    # ---- partition (writes the manifest) ----
    def partition(self, records=None) -> dict:
        recs  = self.records() if records is None else records
        total = len(recs) or 1
        groups = defaultdict(list)
        for r in recs:
            groups[cell_key(r)].append(r)
        rng = random.Random(self.partition_seed)
        splits = {"sft": [], "eval_holdout": [], "rubric": [], "grpo_train": []}
        for cell in sorted(groups.keys()):
            pool = groups[cell][:]
            rng.shuffle(pool)
            abstain = is_abstain(pool[0])  # a cell is homogeneous in regime
            counts = {name: round(len(pool) * size / total) for name, size in self.set_sizes.items()}
            if abstain:
                for name, floor in self.abstain_floor.items():
                    counts[name] = max(counts.get(name, 0), floor)
            idx = 0
            for name in ("eval_holdout", "rubric", "sft"):
                take = max(0, min(counts.get(name, 0), len(pool) - idx))
                splits[name].extend(pool[idx:idx + take])
                idx += take
            splits["grpo_train"].extend(pool[idx:])
        return splits

    def write_manifest(self, splits) -> dict:
        qsets = {name: {r["question"] for r in recs} for name, recs in splits.items()}
        names = list(qsets)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ov = qsets[names[i]] & qsets[names[j]]
                assert not ov, f"LEAKAGE: {names[i]} & {names[j]} share {len(ov)} questions"
        manifest = {
            "version": 2,
            "total": sum(len(recs) for recs in splits.values()),
            "method": "stratified_disjoint_regime_x_domain",
            "lam": self.lam,
            "splits": {
                name: {
                    "n": len(recs),
                    "regime_domain_counts": {f"{reg}|{dom}": c
                                             for (reg, dom), c in sorted(cell_counts(recs).items())},
                    "questions": [r["question"] for r in recs],
                }
                for name, recs in splits.items()
            },
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        self._manifest = manifest
        return manifest


CORPUS = Corpus(abstain_floor={"rubric": 2, "eval_holdout": 3})

if __name__ == "__main__":
    # Standalone artifact: `python stratify.py` dumps the live sampling ratios for
    # every consumer of the shared schema -- SFT draw, RL-batch draw, and the corpus raw mix.
    print("=== JOINT regime x domain stratification ratios (shared core schema) ===\n")
    CORPUS.report("sft",        n=400, label="SFT draw")
    print()
    CORPUS.report("grpo_train", n=50,  label="RL batch draw", lam=0.5)   # matches sample_batch (Box-Cox/CDF)
    print()
    print("  corpus raw (regime x domain) cell counts:")
    raw = CORPUS.counts(CORPUS.records())
    tot = sum(raw.values()) or 1
    for (reg, dom), c in sorted(raw.items()):
        print(f"    {reg}|{dom:<14}{c:>5}{100.0*c/tot:>7.1f}%")
