"""src/rlv2/train/sft/manifold_sampler.py — Copula manifold sampler with delta-ratio correction and token-budget bin packing.

One class, used everywhere: SFT HPO, SFT full, RL HPO, RL full.

The manifold has axes:
  - dataset (6): Agentic, Code, Instruction, Math, ReadingComp, STEM
  - answer_type (4): R1, R2, R3, R4
  - token_length_quantile (4): Q1-Q4
  - score_bucket (5, RL only): [0.0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]

Each populated cell has a target proportion (Box-Cox CDF derived). After every draw:
  actual_ratio[cell] = count[cell] / total_drawn
  delta[cell] = target_ratio[cell] - actual_ratio[cell]
  P(next draw from cell) = max(delta, 0) / sum(max(deltas, 0))

Cumulative counts persist across epochs — self-correcting over the full training run.
Randomness breaks ties: weighted random, not deterministic queue.

Token-budget bin packing: records are routed into n_bins bins (each ≤ target_tokens).
Each bin = one training step. n records per epoch is EMERGENT from bin capacity.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from rlv2.common.stratify import regime_of


class ManifoldSampler:
    """Copula manifold sampler with cumulative delta-ratio correction.

    Require: pool records have 'domain', 'input_ids' (tokenized), and optionally score history.
    Guarantee: over many draws, actual proportions converge to target_ratios.
    Maintain: cumulative_counts persist across epochs for cross-epoch correction.
    """

    def __init__(
        self,
        pool: list[dict],
        target_ratios: dict[str, float],
        target_tokens: int = 896,
        n_bins: int = 4,
        max_len: int = 512,
        len_quantiles: int = 4,
        exclude_prompts: set | None = None,
    ):
        """
        Args:
            pool: all available training records (must have 'domain', 'input_ids').
            target_ratios: {cell_key: proportion} for (regime, domain) cells. Sum to 1.
            target_tokens: VRAM budget per bin (token-slots per training step).
            n_bins: number of bins per epoch (= gradient steps per epoch).
            max_len: hard ceiling on sequence length. Records longer are filtered.
            len_quantiles: number of length quantile buckets (default 4).
            exclude_prompts: set of prompt strings to exclude (e.g. eval prompts). Prevents leakage.
        """
        # Filter pool to max_len and exclude eval prompts
        _exclude = exclude_prompts or set()
        self.pool = [r for r in pool
                     if len(r.get("input_ids", [])) <= max_len
                     and r.get("prompt", r.get("ptext", "")) not in _exclude]
        if not self.pool:
            raise ValueError(f"Pool empty after max_len={max_len} + exclusion filter")

        self.target_ratios = target_ratios
        self.target_tokens = target_tokens
        self.n_bins = n_bins
        self.max_len = max_len

        # Compute length quantile boundaries from pool
        lengths = sorted(len(r["input_ids"]) for r in self.pool)
        n = len(lengths)
        self.len_boundaries = [
            lengths[int(n * (q + 1) / len_quantiles) - 1]
            for q in range(len_quantiles - 1)
        ]  # e.g. [p25, p50, p75] — Q4 is everything above p75

        # Index pool by cell
        self._cell_index: dict[str, list[dict]] = defaultdict(list)
        self._cell_pool_indices: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(self.pool):
            cell = self._cell_of(r)
            self._cell_index[cell].append(r)
            self._cell_pool_indices[cell].append(i)

        # Cumulative state — persists across epochs
        self.cumulative_counts: dict[str, int] = defaultdict(int)
        self.cumulative_drawn: int = 0
        # Records consumed so far across ALL epochs of this run. Only consulted when
        # cover_once is set; otherwise every epoch starts from the full pool.
        self._run_used: set[int] = set()
        self.cover_once: bool = False

        # Populated cells (have at least one record)
        self.populated_cells = set(self._cell_index.keys())

    def _len_quantile(self, length: int) -> str:
        """Map a token length to its quantile label."""
        for i, boundary in enumerate(self.len_boundaries):
            if length <= boundary:
                return f"Q{i + 1}"
        return f"Q{len(self.len_boundaries) + 1}"

    def _cell_of(self, record: dict) -> str:
        """Map a record to its manifold cell key: 'R1|Agentic|Q2'.

        Handles both raw records (with gold/context/domain) and tokenized records
        (with _regime/_domain/input_ids).
        """
        # Tokenized records carry _regime and _domain from encoding
        if "_regime" in record:
            regime = record["_regime"]
        else:
            regime = regime_of(record)
        # `category` IS `domain` UNDER ANOTHER NAME. The R3 builder writes `category`
        # ("Math"/"Code"/"STEM"/"Instruction"); everything older writes `domain`. The values are
        # identical and they are exactly the canon's R3|... keys, so reading only `domain` mapped
        # all 240 R3 rows to "unknown", gave every cell target 0.0, and drew ZERO records --
        # surfacing three frames later as a bare "bins must be non-empty".
        domain = record.get("_domain") or record.get("domain") or record.get(
            "category", "unknown")
        length = len(record.get("input_ids", []))
        lq = self._len_quantile(length)
        return f"{regime}|{domain}|{lq}"

    def _base_cell(self, record: dict) -> str:
        """2D cell key (regime|domain) for ratio lookup."""
        return "%s|%s" % (regime_of(record),
                          record.get("domain") or record.get("category", "unknown"))

    def sample_epoch(self, seed: int) -> list[list[dict]]:
        """Sample one epoch: fill n_bins bins using delta-ratio weighted draws.

        Returns: list of bins (each bin = list of records for one training step).
        Records within an epoch are drawn WITHOUT replacement.
        Cumulative counts are updated (persist to next epoch).
        """
        rng = random.Random(seed)
        bins: list[list[dict]] = [[] for _ in range(self.n_bins)]
        bin_max_lens: list[int] = [0] * self.n_bins
        bin_closed: list[bool] = [False] * self.n_bins
        # WHAT "USED" MEANS IS A CHOICE, and it decides total coverage.
        #
        # Per-epoch (the default): the set resets, so each epoch draws independently
        # from the whole pool. Epochs OVERLAP BY CHANCE. Measured 2026-09-14 at 98
        # draws from 237 rows: 237*(1-(1-98/237)^3) = 189 expected unique over three
        # epochs, so ~48 records were never trained on at all while others were seen
        # three times. Lopsided, and invisible unless you compute it.
        #
        # Across-run (`cover_once=True`): the set persists, so epoch 2 draws only
        # what epoch 1 did not. Three epochs partition the pool into disjoint thirds
        # -- every record exactly once, 237 record-steps instead of 189-unique-out-of
        # -711. Operator 2026-09-14: "The intent isnt 100% in one epoch. But 100% in 3".
        epoch_used: set[int] = set(self._run_used) if self.cover_once else set()

        while not all(bin_closed):
            # Compute deltas: target_ratio - actual_ratio for each cell
            # Cells below target get weight = delta (positive). Cells at/above get a floor
            # weight so they can still be drawn (the pool needs to be consumed for bin filling).
            FLOOR_WEIGHT = 0.001  # minimum draw probability for at-target cells
            weights: dict[str, float] = {}
            # SORTED, NOT SET ORDER. `populated_cells` is a set of strings, and
            # CPython randomizes string hashing per process -- so this loop, and
            # therefore `cells` below, came out in a DIFFERENT ORDER every run.
            # rng.choices() then picked a different domain from identical RNG state
            # and --seed controlled nothing. Measured 2026-09-17: the same config
            # run twice partitioned 98/98/41 vs 89/93/55 and scored grad 2.40 vs
            # 2.21 -- noise larger than half the spread across the whole HPO
            # bracket, which made every arm comparison unsafe.
            for cell in sorted(self.populated_cells):
                target_ratio = self._cell_target(cell)
                if target_ratio <= 0:
                    continue
                # Check if cell has available records this epoch
                cell_pool_indices = self._cell_pool_indices.get(cell, [])
                available = [idx for idx in cell_pool_indices if idx not in epoch_used]
                if not available:
                    continue
                actual_ratio = self.cumulative_counts[cell] / max(1, self.cumulative_drawn) if self.cumulative_drawn > 0 else 0.0
                delta = target_ratio - actual_ratio
                # The floor is the point: a cell at or above target must still be drawable,
                # or the pool cannot be consumed. A second `weights[cell] = delta` used to
                # overwrite this line, restoring the raw delta -- which is <= 0 for any
                # at-target cell and is not a valid weight for `rng.choices`.
                weights[cell] = max(delta, FLOOR_WEIGHT)

            if not weights:
                if not any(bins):
                    # NOTHING WAS DRAWN AT ALL. Not "the pool is exhausted" -- the pool was never
                    # eligible. Say which cells were skipped and against which targets, because
                    # the alternative is an empty-bin ValueError three frames downstream that
                    # names neither.
                    # SAY WHICH OF THE TWO CAUSES IT WAS. This message used to blame
                    # the canon unconditionally ("check that records carry a domain the
                    # canon knows") and truncate target_ratios to 6 keys -- so a pool
                    # EXHAUSTED under cover-once printed a canon complaint listing
                    # R1/R2 cells, while the R3 target it needed was present all along,
                    # just past the truncation. Measured 2026-09-20: cost several
                    # diagnostic passes chasing a canon that was never wrong.
                    _no_target = sorted(
                        c for c in self.populated_cells if self._cell_target(c) <= 0)
                    if _no_target:
                        raise ValueError(
                            "manifold sampler drew 0 records: %d of %d populated cells "
                            "have NO target ratio: %s. Known targets: %s. Check that "
                            "records carry a domain the canon knows."
                            % (len(_no_target), len(self.populated_cells),
                               _no_target[:6], sorted(self.target_ratios)))
                    raise ValueError(
                        "manifold sampler drew 0 records: every populated cell has a "
                        "target but NO AVAILABLE RECORDS -- the pool is EXHAUSTED for "
                        "this epoch (cover_once=%s). Populated cells: %s. This means "
                        "an earlier epoch consumed the pool: check --bins-per-epoch "
                        "against the pool size."
                        % (self.cover_once, sorted(self.populated_cells)[:6]))
                break  # all cells at or above target, or pool exhausted for this epoch

            # Weighted random draw
            cells = list(weights.keys())
            w_vals = [weights[c] for c in cells]
            total_w = sum(w_vals)
            probs = [w / total_w for w in w_vals]
            chosen_cell = rng.choices(cells, weights=probs, k=1)[0]

            # Pick a random record from that cell (not yet used this epoch)
            cell_pool_indices = self._cell_pool_indices[chosen_cell]
            available_indices = [idx for idx in cell_pool_indices if idx not in epoch_used]
            if not available_indices:
                continue
            chosen_pool_idx = rng.choice(available_indices)
            record = self.pool[chosen_pool_idx]
            rec_len = len(record["input_ids"])

            # Route to best-fit open bin (smallest new_max after insertion)
            placed = False
            best_bin = -1
            best_waste = float("inf")
            for i in range(self.n_bins):
                if bin_closed[i]:
                    continue
                new_max = max(bin_max_lens[i], rec_len)
                if (len(bins[i]) + 1) * new_max <= self.target_tokens:
                    waste = new_max - rec_len  # how much padding this record adds
                    if waste < best_waste:
                        best_waste = waste
                        best_bin = i

            if best_bin >= 0:
                bins[best_bin].append(record)
                bin_max_lens[best_bin] = max(bin_max_lens[best_bin], rec_len)
                placed = True
            
            if not placed:
                # Can't fit anywhere — skip, deficit persists for next epoch
                epoch_used.add(chosen_pool_idx)  # don't try this record again this epoch
                continue

            # Update state
            epoch_used.add(chosen_pool_idx)
            if self.cover_once:
                self._run_used.add(chosen_pool_idx)
            self.cumulative_counts[chosen_cell] += 1
            self.cumulative_drawn += 1

            # Check bin closure
            for i in range(self.n_bins):
                if not bin_closed[i]:
                    if (len(bins[i]) + 1) * max(bin_max_lens[i], 1) > self.target_tokens:
                        bin_closed[i] = True

        return [b for b in bins if b]

    def _global_idx(self, cell: str, local_idx: int) -> int:
        """Unique index for a record across all cells (for dedup)."""
        return id(self._cell_index[cell][local_idx])

    def add_score_axis(self, reward_history: dict[str, list[float]], n_buckets: int = 5):
        """Add score diversity as a 4th manifold axis (RL only).

        Reindexes the pool into (regime, domain, len_q, score_bucket) cells.
        Score bucket = which 1/n_buckets band the prompt's mean reward falls in.
        Target: uniform across score buckets within each 3D cell.

        Require: reward_history maps prompt_id -> list of reward scores.
        """
        bucket_width = 1.0 / n_buckets
        new_index: dict[str, list[dict]] = defaultdict(list)

        for cell, records in self._cell_index.items():
            for r in records:
                pid = r.get("ptext", r.get("prompt", ""))
                scores = reward_history.get(pid, [])
                if scores:
                    mean_score = sum(scores) / len(scores)
                    bucket_idx = min(int(mean_score / bucket_width), n_buckets - 1)
                else:
                    bucket_idx = n_buckets // 2  # unknown → middle bucket
                new_cell = f"{cell}|S{bucket_idx}"
                new_index[new_cell].append(r)

        self._cell_index = new_index
        self.populated_cells = set(new_index.keys())

        # `_cell_target` already divides by the length quantiles; record the score-bucket count
        # so it divides by that too. (A duplicate `_cell_target` used to be defined earlier in
        # this class WITHOUT the score divisor -- dead, since this one shadowed it, but it made
        # the class read as though the score axis were ignored.)
        self._n_score_buckets = n_buckets

    def _cell_target(self, cell: str) -> float:
        """Target proportion for a manifold cell (handles 3D and 4D)."""
        parts = cell.split("|")
        base_key = f"{parts[0]}|{parts[1]}"
        base_ratio = self.target_ratios.get(base_key, 0.0)
        n_len_q = len(self.len_boundaries) + 1
        n_score = getattr(self, "_n_score_buckets", 1)
        return base_ratio / (n_len_q * n_score)

    def stats(self) -> dict[str, Any]:
        """Report current manifold state for observability."""
        deficits = {}
        for cell in sorted(self.populated_cells):
            target = self._cell_target(cell)
            actual = self.cumulative_counts[cell] / max(1, self.cumulative_drawn)
            deficits[cell] = target - actual

        top_deficit = sorted(deficits.items(), key=lambda x: x[1], reverse=True)[:5]
        top_surplus = sorted(deficits.items(), key=lambda x: x[1])[:3]

        return {
            "total_drawn": self.cumulative_drawn,
            "populated_cells": len(self.populated_cells),
            "top_deficit": top_deficit,
            "top_surplus": top_surplus,
            "max_delta": max(deficits.values()) if deficits else 0,
        }


def stratified_eval_split(
    eval_records: list[dict],
    n_groups: int = 3,
    seed: int = 99,
) -> list[list[dict]]:
    """Split eval set into n stratified groups, each with ≥1 record per populated cell.

    Used to speed up per-epoch eval: score 1/n_groups of the eval per epoch,
    but still cover all cells over the full training run.

    Require: eval_records have '_regime' and '_domain' (or 'domain') fields.
    Guarantee: each group has at least 1 record from every cell that has >= n_groups records.
    """
    rng = random.Random(seed)

    # Group by cell
    cells: dict[str, list[dict]] = defaultdict(list)
    for r in eval_records:
        regime = r.get("_regime", regime_of(r))
        domain = r.get("_domain", r.get("domain", "unknown"))
        cell = f"{regime}|{domain}"
        cells[cell].append(r)

    # Distribute: round-robin across groups ensuring each group gets ≥1 from each cell
    groups: list[list[dict]] = [[] for _ in range(n_groups)]
    for cell, records in cells.items():
        rng.shuffle(records)
        for i, r in enumerate(records):
            groups[i % n_groups].append(r)

    # Shuffle within each group
    for g in groups:
        rng.shuffle(g)

    return groups
