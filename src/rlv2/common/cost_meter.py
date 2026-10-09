"""src/rlv2/common/cost_meter.py — the second ruler: what did this lift COST? (:EfficiencyAccounting:)

WHY THIS EXISTS. Every verdict in this repo is denominated in `entity_validity` accuracy, which answers
"did it learn?" and is silent on "was it worth it?". The operator's objective is BOTH: the most efficient
route to new knowledge inside 16 GB. An arm that buys +0.05 for 3x the GPU-minutes and 4x the labeled
rows is a worse answer than one that buys +0.04 for free, and an accuracy-only table cannot say so.

THE FOUR CURRENCIES, and why each is tracked separately rather than collapsed into one score:

    records      labeled QA pairs consumed. The SCARCEST currency — every row costs generation or human
                 time, and unlike GPU-minutes it does not get cheaper with better hardware.
    train_min    wall-clock GPU minutes. On this thermally-throttled Max-Q this is the axis that decides
                 whether an experiment fits a session at all.
    peak_gb      peak VRAM. A hard 16GB wall, not a gradient: an arm that does not fit is not slower,
                 it is impossible (or, under WDDM, silently paging and 10x slower).
    prompt_tok   mean prompt tokens per query at inference. THE RAG TAX. Retrieval trains nothing, so
                 an accuracy-only table makes it look free; it moves the cost from training to every
                 single query forever, and that only shows up here.

Collapsing these into one number would require exchange rates between GPU-minutes, labeled rows, and
per-query latency that depend on the deployment, not on the experiment. So they are reported side by side
and the verdict names the trade explicitly.

DERIVED. gain_per_record and gain_per_min are (arm_acc - base_acc) / cost. They are reported ONLY when
the gain's confidence band excludes zero — an efficiency ratio computed on noise is a fabricated number,
and the `validation.txt` precedent (+0.030 at p=0.983, later overturned) is exactly that failure.
"""
import json
import os
import time

import torch

# ── HARD VRAM CEILING ────────────────────────────────────────────────────────────────────────────
# Operator law (2026-07-29): NEVER exceed 16 GB, and BREAK if we ever do.
#
# Why this needs code and not discipline: this card is WDDM, which does NOT OOM on overflow — the
# driver pages the excess into system RAM. A run that wants 20 GB does not fail, it silently spills
# and crawls, and the symptom shows up as 64 GB of system RAM rather than as an error. On 2026-07-29
# a probe ran beside an unrelated GPU job, together crossed the wall, and paged.
#
# torch.cuda.set_per_process_memory_fraction is the actual enforcement: it caps the allocator so a
# request past the cap raises torch.cuda.OutOfMemoryError instead of being satisfied by paging. That
# converts a silent 10x slowdown into a loud, immediate, debuggable failure — which is what "break"
# means. A polling check alone cannot do this: by the time a poll observes 16.5 GB the paging has
# already happened.
# These are NOT a static budget. Measured 2026-07-29: with an active display this card routinely has
# ~10 GB already committed by things that have nothing to do with training — Ollama alone holds 9.9 GB
# for a 9B at 256k context (keep_alive keeps it resident AFTER the generation stage finishes), plus
# dwm.exe, explorer.exe, vlc.exe, notepad++.exe, EpicGamesLauncher.exe. Sizing a run against "16 GB
# total" is therefore wrong by ~10 GB, and that is precisely how the wall gets crossed.
#
# So the ceiling is derived from what is ACTUALLY FREE at launch, not from card capacity. The absolute
# cap is a backstop for the case where the card happens to be idle.
# Ceiling is 16384 MiB — the whole card (operator decision 2026-07-29). nvidia-smi speaks MiB, torch
# speaks bytes, and 16384 MiB = 17.18e9 bytes, so the two disagree by 7% if you mix units. Everything
# here is MiB to match what you actually read off nvidia-smi.
VRAM_CEILING_MIB = 16384
VRAM_SAFETY_MIB = 128           # never claim the last slice; the display compositor needs slack
# CAP AGAINST THE CARD, NOT AGAINST FREE (operator, 2026-08-03: "that cap is way too fucking
# aggressive, we should be capping at like 16256"). Deriving the cap from free-now made the ceiling
# ~15007 MiB on an essentially idle card -- 1.1 GB of display/compositor residency was being
# subtracted from OUR budget even though the driver can evict most of it under pressure.
#
# THE TRADE, STATED: the allocator cap can no longer catch cross-process contention, because at
# total-128 it will happily let us request memory another process already holds, and WDDM pages
# rather than failing. The PREFLIGHT ABORT below is now the ONLY guard against that case -- it still
# refuses to start when free VRAM is under the minimum working set, which is the situation that
# actually burned 2026-07-29 (Ollama holding 9.9 GB). Cap = use the card; abort = do not start into
# a card someone else owns.
VRAM_CAP_MIB = 16256            # 16384 - 128
VRAM_MIN_WORKING_SET_MIB = 5120 # below this a full-param 0.8B SFT cannot run at all -- abort, do not try
_MIB = 1024 ** 2


class VramCeilingExceeded(RuntimeError):
    """Raised when a stage would cross the hard VRAM wall. Never caught — it must stop the run."""


def enforce_vram_ceiling(ceiling_mib=VRAM_CEILING_MIB, vram_frac=1.0):
    """Cap the torch allocator to the free window and abort if that is not enough to work in.

    Call this ONCE at the top of every GPU stage, before any model loads.

        ceiling = min(free_now - safety, 16384 MiB)

    Deriving from free-now rather than card capacity is the point: on 2026-07-29 a probe sized itself
    against the full card while Ollama held 9.9 GB, so the two together crossed the wall and WDDM paged
    into system RAM (observed as ~64 GB system RAM, and NO error). Both guards are needed --
      abort   free memory below the minimum working set: a run that cannot fit must not start and hope,
              because WDDM lets it "succeed" 10x slower instead of failing.
      cap     the allocator, so OUR growth raises OutOfMemoryError rather than spilling. This is the
              only mechanism that actually breaks; a poll that observes 16.5 GB is already too late.
    """
    if not torch.cuda.is_available():
        return {"enforced": False, "reason": "no cuda"}
    # RELEASE OUR OWN MEMORY BEFORE MEASURING FREE. `mem_get_info` reports what the DEVICE has free,
    # which includes anything this process is still holding in the caching allocator from a previous
    # stage. Without this the ceiling RATCHETS DOWN stage after stage: measured 2026-08-03, the CPT
    # stage ran with 13.26 GiB allowed, and the SFT stage that followed it was capped at 7.10 GiB
    # while 8.30 GiB was genuinely free -- it OOM'd at step 0 asking for 728 MiB. The cap was
    # protecting the run from memory it had already finished with.
    # Only OUR unreferenced blocks are freed, so a co-resident Ollama still counts against us and the
    # cross-process protection this guard exists for is unchanged.
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    free_b, total_b = torch.cuda.mem_get_info()
    free_mib, total_mib = free_b // _MIB, total_b // _MIB
    # Cap against FREE VRAM (minus safety), not card capacity. The system/display compositor
    # holds ~1.5 GB that we measured and cannot evict. Capping against total lets PyTorch claim
    # memory that will push total usage past 16384 and trigger WDDM paging.
    # Observed: 15.4 GB training + 1.5 GB system = 16.9 GB → silent paging.
    # The cap must respect what is ACTUALLY AVAILABLE, not what the card theoretically holds.
    usable_mib = int(min(free_mib - VRAM_SAFETY_MIB, ceiling_mib, VRAM_CAP_MIB) * vram_frac)
    # ABORT still keys on FREE, not on the cap: the cap decides how much WE may take, the abort
    # decides whether there is room to start at all. Conflating them is what let a contended
    # card page instead of refusing.
    if (free_mib - VRAM_SAFETY_MIB) < VRAM_MIN_WORKING_SET_MIB:
        raise VramCeilingExceeded(
            f"PREFLIGHT ABORT -- {free_mib} MiB free of {total_mib} MiB leaves only {free_mib - VRAM_SAFETY_MIB} MiB "
            f"usable, below the {VRAM_MIN_WORKING_SET_MIB} MiB minimum working set. Something else "
            f"holds the card (check `ollama ps` -- keep_alive keeps a model resident AFTER its stage "
            f"ends, and a 9B at 256k ctx is ~9.9 GB). Free it or wait; starting now would page into "
            f"system RAM instead of failing.")
    frac = min(usable_mib / total_mib, 1.0)
    torch.cuda.set_per_process_memory_fraction(frac)
    print(f"[vram] {free_mib} MiB free of {total_mib} -> ceiling {usable_mib} MiB "
          f"({frac:.0%} of card, rung frac {vram_frac:.3f}) "
          f"ENFORCED. Allocation past it RAISES instead of paging.", flush=True)
    return {"enforced": True, "ceiling_mib": usable_mib, "free_at_start_mib": free_mib}


def assert_gpu_exclusive(raise_on_conflict=True):
    r"""Refuse to start a GPU stage while ANOTHER training run is alive.

    Earned 2026-08-03: a chunker arm was relaunched before the previous run exited, two training
    processes shared the card for ~4 minutes, and step time went 4.94 -> 20.81 s/it. NOTHING caught
    it -- `throughput_floor` trips at 8x and this was 4.2x; `nvidia-smi` shows occupancy either way;
    mtime kept advancing; `enforce_vram_ceiling` passed because there WAS enough free memory for two.
    Contention is neither a stall nor an overflow, so every existing guard is blind to it.

    IDENTITY, NOT MEMORY. A first version thresholded on working set >= 4 GB and returned "exclusive"
    while a trainer sat at 3.69 GB -- working set swings through a run, so a fixed RAM cut is the
    wrong instrument. It also counted PIDs: one launch produces a `py310\Scripts` wrapper AND the
    real interpreter, two PIDs sharing one command line, which reads as a conflict with itself.

    NOT THE GPU EITHER. `nvidia-smi --query-compute-apps` reports `[N/A]` / `[Insufficient
    Permissions]` per process on this WDDM card and `pmon` is unsupported, so the GPU cannot say who
    holds it. Command lines CAN be read, so identity comes from what a process is RUNNING.

    So: enumerate python processes with a `.py` in their command line, drop this process's own tree
    (self + ancestors + descendants), and treat whatever remains as a foreign run. Fails CLOSED --
    a resource gate that expires into "proceed" is worse than no gate.
    """
    import os as _os
    import subprocess
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name LIKE 'python%'\" | "
             "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as ex:
        print(f"[gpu-excl] SKIPPED ({type(ex).__name__}) -- cannot enumerate processes", flush=True)
        return {"checked": False}
    if not out:
        return {"checked": True, "others": []}
    rows = json.loads(out)
    rows = rows if isinstance(rows, list) else [rows]
    by_pid = {r["ProcessId"]: r for r in rows}
    me = _os.getpid()

    # my own tree: walk up to ancestors, then collect anything descending from them
    mine, cur = set(), me
    while cur in by_pid and cur not in mine:
        mine.add(cur)
        cur = by_pid[cur].get("ParentProcessId")
    mine.add(me)
    changed = True
    while changed:                       # descendants, transitively
        changed = False
        for r in rows:
            if r["ProcessId"] not in mine and r.get("ParentProcessId") in mine:
                mine.add(r["ProcessId"])
                changed = True

    # EXCLUDE THE HARNESS, FLAG EVERYTHING ELSE. `".py" in cmdline` alone fired on
    # `~/.claude/hooks/membank.py` and killed a 3-arm run between arms -- fail-closed is the right
    # direction, but stopping the wrong thing is still a defect.
    #
    # A first fix tried to match the REPO PATH in the command line. That cannot work: scripts are
    # invoked relatively (`python chunker_ab.py`), so the repo path never appears, and the fallback
    # heuristic counted backslashes when the interpreter path alone carries five. Both battery cases
    # failed against a genuinely detached process.
    #
    # What actually discriminates is simpler: harness and library code lives in known directories.
    # Anything else running a .py is a candidate conflict, and a false positive costs a wait while a
    # false negative costs a corrupted measurement.
    _EXCLUDE = ("\.claude\\", "/.claude/", "\hooks\\", "/hooks/",
                "site-packages", "\.vscode", "pip", "ensurepip")

    # NARROWED 2026-08-24. "any .py that isn't library code" fires on every always-on MCP
    # server on this box (simple-web-search, noosphere/autogen/todo/consciousness
    # orchestrators, the rrf-graph-rag retriever) -- 30+ PIDs, none of them trainers, none
    # of them ever exiting. A gate that can NEVER pass is not conservative, it is a gate
    # that gets bypassed with SKIP_GPU_EXCLUSIVE, which is precisely the "expires into
    # proceed" failure this module warns about. Measured: of the 32 PIDs it flagged, one
    # (28804, an MCP embedder) had a CUDA runtime loaded and zero were training.
    #
    # The failure mode being guarded is TWO TRAINING RUNS OF THIS PROJECT SHARING THE CARD.
    # So identity needs both halves: running code from THIS repo, and actually on the GPU.
    # Either alone is wrong -- repo-only flags CPU jobs like archive/src/topup_pools_r3.py, and
    # cuda-only flags unrelated MCP embedders that hold a few hundred MB and never train.
    _REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).lower()

    def _has_cuda_loaded(pid):
        """True if the process has the CUDA runtime mapped -- i.e. it is actually on the card.

        nvidia-smi cannot attribute memory per-process on this WDDM card ([N/A] /
        [Insufficient Permissions]), so loaded-module inspection is the only available
        instrument. Fails CLOSED: if the module list cannot be read (access denied on an
        elevated process), assume the process IS on the GPU.
        """
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-Process -Id {pid}).Modules | "
                 "Where-Object {$_.ModuleName -match 'nvcuda|cudart|torch_cuda'} | "
                 "Select-Object -First 1 -ExpandProperty ModuleName"],
                capture_output=True, text=True, timeout=20)
        except Exception:
            return True
        if out.returncode != 0:
            return True
        return bool(out.stdout.strip())

    # Basename match is REQUIRED alongside the repo-path match, not redundant with it: the
    # harness launches `python src/rlv2/train/sft/train.py ...` with cwd=repo, so the command line
    # carries NO absolute repo path and a path-only test would fail OPEN on the exact case
    # this gate exists to catch (two harness trainers on one card).
    _TRAINERS = ("vibe_sft.py", "cpt.py", "mgpo.py", "train_harness.py", "ft_lr_sweep.py",
                 "lora_chain_probe.py", "rl_vibe_ab.py", "rl_calibration_eval.py")

    def _is_foreign_run(cmd):
        c = (cmd or "").lower()
        if ".py" not in c or any(seg in c for seg in _EXCLUDE):
            return False
        return _REPO in c.replace("/", "\\") or any(t in c for t in _TRAINERS)

    candidates = [r for r in rows
                  if r["ProcessId"] not in mine and _is_foreign_run(r.get("CommandLine"))]
    others = [r for r in candidates if _has_cuda_loaded(r["ProcessId"])]
    if candidates and not others:
        print(f"[gpu-excl] {len(candidates)} foreign repo process(es) running but none hold "
              f"CUDA -- not a card conflict: "
              + "; ".join(f"pid {r['ProcessId']}" for r in candidates), flush=True)
    if others:
        desc = "; ".join(f"pid {r['ProcessId']}: {(r.get('CommandLine') or '')[-70:]}"
                         for r in others)
        msg = (f"[gpu-excl] ANOTHER RUN IS ALIVE -- {desc}. Two training runs on one card do not "
               f"fail, they SHARE it, and the only symptom is step time drifting (measured "
               f"4.94 -> 20.81 s/it). Stop it (Stop-Process -Id <pid>) or wait.")
        if raise_on_conflict:
            raise VramCeilingExceeded(msg)
        print(msg, flush=True)
    else:
        print("[gpu-excl] exclusive -- no foreign python run detected", flush=True)
    return {"checked": True, "others": others, "own_tree": sorted(mine)}


def peak_mib():
    """Allocator high-water mark in MiB."""
    return (torch.cuda.max_memory_allocated() // _MIB) if torch.cuda.is_available() else 0


# ── ABSOLUTE THROUGHPUT FLOOR ────────────────────────────────────────────────────────────────────
# Earned 2026-07-30, at a cost of 9.5 hours: the `full` arm did not stall, it CRAWLED. Step time went
# from 1.7s to 10,911s/it -- 9 steps of 112 in 9.5 hours -- while the log kept advancing, so every
# liveness check reported healthy.
#
# WHY THE EWMA WATCHDOG CANNOT CATCH THIS. An EWMA tracks drift by construction: a gradual slowdown
# BECOMES the new baseline, the ceiling rises with it, and the stall test never fires. It detects a
# cliff, not a slide. The fix is not a better EWMA -- it is an ABSOLUTE floor that does not learn.
#
# The floor is derived, not guessed -- but it MUST be derived from the stage it guards.
#
# EARNED 2026-08-02, at a cost of 102 GPU-minutes. MAX_STEP_S was fixed at 30.0s because "healthy arms
# on this box run 1.7-3.2 s/step" -- true at batch 4 x 192 tokens, which is the shape it was written
# for. It was then applied unchanged to a CPT stage running PACKED 1024-token blocks at a steady
# 7.7 s/step. The intended 10x headroom was silently 3.9x, and a single 49.4s hiccup (6.4x) killed a
# run at 775/972 steps.
#
# So the floor now CALIBRATES: it measures this stage's own median step time over CAL_STEPS, then
# FREEZES a multiple of it. Freezing is the whole point -- an EWMA tracks drift by construction, so a
# gradual slowdown becomes the new baseline and the test never fires. It detects a cliff, not a slide.
# Calibrate once, then never learn again.
# A single global CAP cannot do this job -- that was the bug. It is kept only as a pathological
# backstop (no step on this box is ever legitimately 10 minutes), well clear of any real workload.
HARD_CAP_S = 600.0        # never-exceed backstop, deliberately far above every healthy shape
CAL_STEPS = 12            # post-watchdog steps used to measure THIS stage's healthy rate
CAL_MULT = 8.0            # trip point = CAL_MULT x median healthy step
MIN_FLOOR_S = 20.0        # never trip below this, however fast the stage is
MAX_STEP_S = 30.0         # DEPRECATED default for callers that pass no calibration; see throughput_floor
WARMUP_STEPS = 3          # first steps include cudnn autotune + allocator warmup; never judge on them


class ThroughputCollapsed(RuntimeError):
    """Raised when per-step time breaches the absolute floor. Never caught -- it must stop the run."""


def throughput_floor(max_step_s=None, warmup=WARMUP_STEPS, cal_steps=CAL_STEPS, mult=CAL_MULT):
    """A TrainerCallback that kills the run when a single step exceeds a SELF-CALIBRATED time floor.

    Lives INSIDE the training process, because the outer watchdog can only see log growth and a
    crawling run grows its log just fine. This sees the thing that actually matters: seconds per step.

    Require   -- the stage's first `warmup + cal_steps` steps are representative of healthy running.
    Guarantee -- raises ThroughputCollapsed on the first step exceeding the frozen trip point.
    Maintain  -- the trip point NEVER moves after calibration, so a gradual slide cannot raise it.
    Assert    -- the calibrated trip point is printed, so a run's guard is auditable from the log.

    Passing `max_step_s` pins an explicit trip point and skips calibration -- use only when the stage's
    healthy rate is already known. Otherwise the floor is `mult x median(step time over cal_steps)`,
    clamped to [MIN_FLOOR_S, HARD_CAP_S]. A fixed global constant is what killed a CPT run at 80%:
    it had been derived for a workload 4x faster than the one it was guarding.
    """
    import statistics
    from transformers import TrainerCallback

    class _Floor(TrainerCallback):
        def __init__(self):
            self.t = None
            self.n = 0
            self.cal = []
            self.trip = float(max_step_s) if max_step_s else None

        def on_step_begin(self, args, state, control, **kw):
            self.t = time.time()

        def on_step_end(self, args, state, control, **kw):
            if self.t is None:
                return
            dt = time.time() - self.t
            self.n += 1
            if self.n <= warmup:
                return
            if self.trip is None:
                # Calibration window. A collapse HERE is still caught, by the hard backstop only --
                # we cannot know this stage's healthy rate before measuring it.
                self.cal.append(dt)
                if dt > HARD_CAP_S:
                    raise ThroughputCollapsed(
                        f"step {self.n} took {dt:.1f}s during calibration, over the {HARD_CAP_S:.0f}s "
                        f"pathological backstop. No healthy stage on this box is near this.")
                if len(self.cal) >= cal_steps:
                    med = statistics.median(self.cal)
                    self.trip = min(HARD_CAP_S, max(MIN_FLOOR_S, mult * med))
                    print(f"[floor] calibrated: median {med:.2f}s/step over {len(self.cal)} steps "
                          f"-> trip at {self.trip:.1f}s ({mult:.0f}x). Frozen for the run.", flush=True)
                return
            if dt > self.trip:
                med = statistics.median(self.cal) if self.cal else float("nan")
                raise ThroughputCollapsed(
                    f"step {self.n} took {dt:.1f}s, over this stage's calibrated {self.trip:.1f}s trip "
                    f"point (median healthy step {med:.2f}s -> a ~{dt/max(med, 1e-9):.1f}x collapse). "
                    f"Almost certainly the optimizer or allocator spilling to system RAM. Killing "
                    f"now: a crawling run is unusable data AND holds the card for hours while every "
                    f"liveness check reports healthy.")

    return _Floor()


# ── ADAPTIVE DOWNSCALE ON TRIP ───────────────────────────────────────────────────────────────────
# Operator spec (2026-07-29): run at the full 16384 MiB; if we trip it, kill, scale down something
# that affects memory, resume from the last checkpoint, and repeat if we trip again.
#
# WHY NOT batch_size x 0.95, as originally proposed. These runs are ALREADY at
# per_device_train_batch_size=1, so there is nothing to scale. And reducing
# gradient_accumulation_steps would make peak memory WORSE, not better -- accumulation exists to keep
# peak low by processing one sample at a time; fewer accumulation steps means more work held live.
#
# At batch_size=1 the knobs that actually move peak VRAM, strongest first:
#   gradient_checkpointing   recomputes activations in the backward pass instead of storing them.
#                            ~40% off activation memory for ~30% wall-clock. The single biggest win,
#                            so it is rung 1 -- it buys the most memory per unit of damage to the
#                            experiment, and unlike max_length it changes NO data.
#   max_length               activation memory scales with sequence length. Truncates long answers,
#                            which is real damage to a knowledge experiment, so it comes after.
# fp16 and paged_adamw_8bit are already on in every arm and are therefore not rungs.
# SEQUENCE IS A FRACTION, NOT AN ABSOLUTE. Earned 2026-08-03: the rungs held absolute max_length
# values (512 -> 416) written for the CPT stage, where BLOCK=512. SFT's natural length is 192, so
# `rung["max_length"]` of 486 did not downscale it -- it made the sequence 2.5x LONGER, and every
# "downscale" rung increased memory demand until the ladder exhausted at rung 7. A downscale ladder
# that can UPSCALE is broken by construction whenever two stages have different natural sizes.
# A fraction is stage-agnostic and can only ever reduce.
#
# ORDER MATTERS: cheapest-to-the-MEASUREMENT first. `vram_frac` and `grad_ckpt` change how the same
# arm is computed; `max_length` changes WHAT IS MEASURED (truncation drops tokens the arm would
# otherwise train on), so sequence length is stepped only after the free knobs are exhausted.
#
# The fraction ladder is the operator's (2026-08-03): treat the enforced cap as 100% and back off in
# 2.5% steps per OOM. Fragmentation, not true exhaustion, causes most repeat OOMs on this card -- a
# slightly smaller cap gives the caching allocator room to coalesce, and costs nothing measurable.
LADDER = [
    {"vram_frac": 1.000, "seq_frac": 1.00, "grad_ckpt": True, "why": "baseline: full cap (16256 MiB), grad-ckpt on"},
    {"vram_frac": 0.975, "seq_frac": 1.00, "grad_ckpt": True, "why": "OOM 1: cap 97.5% (~15850 MiB)"},
    {"vram_frac": 0.950, "seq_frac": 1.00, "grad_ckpt": True, "why": "OOM 2: cap 95% (~15443 MiB)"},
    {"vram_frac": 0.925, "seq_frac": 1.00, "grad_ckpt": True, "why": "OOM 3: cap 92.5% (~15037 MiB)"},
    {"vram_frac": 0.925, "seq_frac": 0.95, "grad_ckpt": True, "why": "OOM 4: seq x0.95 -- MEASUREMENT CHANGES HERE"},
    {"vram_frac": 0.925, "seq_frac": 0.90, "grad_ckpt": True, "why": "OOM 5: seq x0.90"},
    {"vram_frac": 0.925, "seq_frac": 0.85, "grad_ckpt": True, "why": "OOM 6: seq x0.85"},
    {"vram_frac": 0.925, "seq_frac": 0.80, "grad_ckpt": True, "why": "OOM 7: seq x0.80 -- LAST RUNG"},
]


def _ladder_path(tag):
    return os.path.join("data", "domain", "vram_ladder", f"{tag}.json")


def current_rung(tag):
    """The rung this stage should run at, read from disk.

    Persisted rather than held in memory because the operator spec is *kill* then resume: the retry
    may be a whole new process, so the rung has to survive process death or the backoff resets to
    baseline forever and loops.
    """
    p = _ladder_path(tag)
    i = json.load(open(p, encoding="utf-8"))["rung"] if os.path.exists(p) else 0
    i = max(0, min(i, len(LADDER) - 1))
    return i, dict(LADDER[i])


def trip(tag, err=""):
    """Record a VRAM trip and advance the ladder. Returns (next_index, cfg, exhausted)."""
    i, _ = current_rung(tag)
    nxt = i + 1
    exhausted = nxt >= len(LADDER)
    nxt = min(nxt, len(LADDER) - 1)
    p = _ladder_path(tag)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    hist = json.load(open(p, encoding="utf-8")).get("trips", []) if os.path.exists(p) else []
    hist.append({"at": time.strftime("%Y-%m-%d %H:%M:%S"), "from_rung": i, "err": str(err)[:300]})
    json.dump({"rung": nxt, "trips": hist}, open(p, "w", encoding="utf-8"), indent=1)
    cfg = dict(LADDER[nxt])
    print(f"[vram] TRIP on rung {i} -> downscaling to rung {nxt}: {cfg['why']}"
          + ("  (LADDER EXHAUSTED)" if exhausted else ""), flush=True)
    return nxt, cfg, exhausted


def is_oom(exc):
    """True for a real VRAM exhaustion, including our own enforced ceiling."""
    if isinstance(exc, (torch.cuda.OutOfMemoryError, VramCeilingExceeded)):
        return True
    return "out of memory" in str(exc).lower() or "cuda oom" in str(exc).lower()


def run_with_downscale(tag, train_fn, max_attempts=len(LADDER)):
    """Run train_fn(cfg, resume=bool) under the ladder, downscaling on each VRAM trip.

    train_fn MUST checkpoint mid-run and honour `resume` by continuing from its last checkpoint --
    otherwise every trip restarts from zero and the backoff costs more than it saves. That coupling is
    the whole reason this is safe to be aggressive with: killing is cheap only when resume is free.
    """
    attempt = 0
    while attempt < max_attempts:
        i, cfg = current_rung(tag)
        print(f"[vram] {tag}: attempt {attempt + 1}/{max_attempts} at rung {i} -- {cfg['why']}",
              flush=True)
        try:
            return train_fn(cfg, resume=attempt > 0)
        except BaseException as e:                      # noqa: BLE001 - OOM arrives in several shapes
            if not is_oom(e):
                raise
            # Free the failed graph before retrying, or the next attempt inherits a fragmented heap
            # and trips earlier than its rung warrants.
            torch.cuda.empty_cache()
            _, _, exhausted = trip(tag, e)
            if exhausted:
                raise VramCeilingExceeded(
                    f"{tag}: LADDER EXHAUSTED after {attempt + 1} trips. Last rung "
                    f"{LADDER[-1]['why']} still will not fit. This is a sizing problem, not a "
                    f"scheduling one -- shrink the model, the span, or the eval batch.") from e
            attempt += 1
    raise VramCeilingExceeded(f"{tag}: {max_attempts} attempts exhausted without fitting.")


class CostMeter:
    """Context manager recording wall-clock and peak VRAM for one stage.

    Usage:
        with CostMeter() as m:
            ...train...
        m.as_dict()  -> {"train_min": ..., "peak_gb": ...}

    Peak VRAM is read from torch's allocator high-water mark, reset on entry, so nested or prior stages
    do not contaminate the reading.
    """

    def __init__(self):
        self.t0 = None
        self.minutes = 0.0
        self.peak_gb = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.minutes = (time.time() - self.t0) / 60.0
        if torch.cuda.is_available():
            self.peak_gb = torch.cuda.max_memory_allocated() / 1e9
        return False

    def as_dict(self):
        return {"train_min": round(self.minutes, 2), "peak_gb": round(self.peak_gb, 2)}


def trainable_params(model):
    """Parameters that will actually receive gradients. The capacity a method paid for."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mean_prompt_tokens(tok, prompts):
    """Mean prompt length in tokens — the per-query inference tax (RAG's hidden cost)."""
    if not prompts:
        return 0.0
    return float(sum(len(tok(p)["input_ids"]) for p in prompts) / len(prompts))


def cost_row(tag, cost, acc, base_acc, resolved):
    """One arm's cost line. Efficiency ratios are suppressed unless the gain is statistically resolved.

    `resolved` is the caller's significance decision (band excludes zero). Passing False yields "-" for
    the ratios rather than a number computed on noise.
    """
    gain = acc - base_acc
    rec = cost.get("records", 0) or 0
    mins = cost.get("train_min", 0.0) or 0.0
    per_rec = f"{gain/rec:+.4f}" if (resolved and rec) else "-"
    per_min = f"{gain/mins:+.4f}" if (resolved and mins > 0.01) else "-"
    return {"tag": tag, "acc": acc, "gain": gain, "records": rec,
            "train_min": mins, "peak_gb": cost.get("peak_gb", 0.0),
            "trainable_m": (cost.get("trainable", 0) or 0) / 1e6,
            "prompt_tok": cost.get("prompt_tok", 0.0),
            "gain_per_record": per_rec, "gain_per_min": per_min}


def render_cost_table(rows):
    """Fixed-width cost table. Same column order everywhere so arms are comparable across hypotheses."""
    L = [f"{'arm':<14}{'acc':>7}{'gain':>8}{'recs':>6}{'GPU-min':>9}{'peakGB':>8}"
         f"{'train-M':>9}{'promptTok':>11}{'gain/rec':>10}{'gain/min':>10}"]
    for r in rows:
        L.append(f"{r['tag']:<14}{r['acc']:>7.3f}{r['gain']:>+8.3f}{r['records']:>6}"
                 f"{r['train_min']:>9.1f}{r['peak_gb']:>8.2f}{r['trainable_m']:>9.1f}"
                 f"{r['prompt_tok']:>11.0f}{r['gain_per_record']:>10}{r['gain_per_min']:>10}")
    return L


def save(path, payload):
    """Persist a cost record next to the accuracy JSON. On-disk is primary (repo convention)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prior = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    prior.update(payload)
    json.dump(prior, open(path, "w", encoding="utf-8"), indent=1)
    return prior


# ── DID THE ADAPTER ACTUALLY MOVE? ───────────────────────────────────────────────────────────────
# Earned 2026-08-02. Three "RL contributed nothing" verdicts were produced by evaluating adapters
# whose weights had barely left their initialisation: mgpo's arms ran 2 rounds x 12 steps at lr 1e-6
# (a MECHANISM probe, by design) and were then read as trained policies. max|B| was 8e-6 against SFT's
# 6e-2 -- an effective |dW| of ~1e-6, which fp16 cannot even represent as a change.
#
# LoRA's B initialises to EXACTLY zero, which makes this free to check and impossible to fake: any
# non-trivial max|B| is training that happened. Call it after every LoRA save. A null measured on an
# untrained adapter is not a null about the method.
def assert_adapter_trained(path, min_abs_b=1e-4, label=""):
    """Require -- `path` holds a saved PEFT adapter. Guarantee -- raises unless B moved off zero.

    `min_abs_b` is two orders below a healthy SFT adapter (6e-2) and two above a no-op (8e-6), so it
    separates the two cases without needing a per-stage threshold.
    """
    import os
    from safetensors.torch import load_file
    f = os.path.join(path, "adapter_model.safetensors")
    if not os.path.exists(f):
        # MULTI-ADAPTER LAYOUT. `PeftModel.save_pretrained` writes EACH adapter to its own
        # subdirectory (`<out>/cpt/`, `<out>/rl/`) rather than to the root, so a stacked arm looks
        # like "no adapter" to a root-only check. Measured 2026-08-04: a 48-minute RL arm was
        # reported as FileNotFoundError while both adapters sat one directory down.
        # Check the TRAINABLE adapter -- a frozen one is expected to be unchanged, so averaging it in
        # would dilute exactly the signal this gate exists to read.
        subs = [d for d in sorted(os.listdir(path))
                if os.path.exists(os.path.join(path, d, "adapter_model.safetensors"))]             if os.path.isdir(path) else []
        if not subs:
            raise FileNotFoundError(f"no adapter at {f} (and no per-adapter subdirectory)")
        # prefer a directory whose name is not the known-frozen upstream stage
        pick = next((d for d in subs if d not in ("cpt", "sft")), subs[-1])
        f = os.path.join(path, pick, "adapter_model.safetensors")
        print(f"[trained] multi-adapter layout: checking '{pick}' of {subs}", flush=True)
    sd = load_file(f)
    bs = [v for k, v in sd.items() if "lora_B" in k]
    if not bs:
        raise ValueError(f"{path}: no lora_B tensors -- not a LoRA adapter?")
    mx = max(float(v.abs().max()) for v in bs)
    if mx < min_abs_b:
        raise RuntimeError(
            f"{label or path}: max|lora_B| = {mx:.3e}, below {min_abs_b:.0e}. B inits to ZERO, so this "
            f"adapter barely trained -- effective |dW| ~ {4*mx*1e-2:.1e} is a NO-OP in fp16. Evaluating "
            f"it would produce a null that says nothing about the method. Raise lr or step count.")
    print(f"[trained] {label or path}: max|lora_B| = {mx:.3e} -- adapter moved", flush=True)
    return mx


# ── IS THE CORPUS MIX THE ONE WE DECLARED? ───────────────────────────────────────────────────────
# Earned 2026-08-03. ONE change -- replacing blind 512-token packing with semantic chunking -- moved
# THREE ratios that nobody chose:
#
#     CPT:SFT        0.66 -> 2.35   (3.56x, because 1 skill now yields ~7 records instead of ~2)
#     exposure       1.00 -> 2.94x  (stride-1 rolling window repeats every atom)
#     replay share  10.0% -> 3.6%   (REPLAY_FRAC untouched; the DENOMINATOR tripled)
#
# All three would have surfaced in an A/B as "semantic boundaries help/hurt" -- unattributable. And
# the replay drift is load-bearing: the measured forgetting result (base-correct slice, -0.146,
# p=0.0082) was taken at a 10% share, so a rerun at 3.6% would read as "replay stopped working".
#
# THE RULE: ratios are set against tokens SEEN, not tokens STORED. Exposure multiplies the
# denominator, so every ratio pinned to the numerator drifts when it changes. Declare the mix up
# front and abort on drift -- three separate things to remember is exactly how all three got missed.
def assert_composition(declared, actual, tol=0.05, label="cpt"):
    """Require -- `declared` and `actual` map the same ratio names to floats. Guarantee -- raises on
    any relative drift beyond `tol`. Maintain -- the returned dict is stamped into run_cfg.json, so a
    later diff separates "different data mix" from "different hyperparameters" in one step.

    Ratio names are free-form; the caller owns the vocabulary (anchor, cpt_sft, exposure,
    replay_share, vibethinker). Anything declared MUST be supplied, or the check is silently partial
    -- that is the unread-column failure mode, so a missing key raises rather than being skipped.
    """
    missing = sorted(set(declared) - set(actual))
    if missing:
        raise KeyError(f"[{label}] declared but not measured: {missing} -- a partial composition "
                       f"check reports health it did not verify")
    drift, lines = [], []
    for k in sorted(declared):
        d, a = float(declared[k]), float(actual[k])
        rel = abs(a - d) / abs(d) if d else (0.0 if a == 0 else float("inf"))
        ok = rel <= tol
        lines.append(f"    {k:<14}declared {d:>10.4f}   actual {a:>10.4f}   {rel:>6.1%}  "
                     f"{'ok' if ok else 'DRIFT'}")
        if not ok:
            drift.append(f"{k}: declared {d:.4f}, actual {a:.4f} ({rel:.1%} off)")
    print(f"[composition] {label}", flush=True)
    for ln in lines:
        print(ln, flush=True)
    if drift:
        raise RuntimeError(f"[{label}] composition drift beyond {tol:.0%}: " + "; ".join(drift) +
                           " -- the mix is a declared parameter, not a side effect of the chunker. "
                           "Fix the ratio or update the declaration deliberately.")
    return {"declared": dict(declared), "actual": {k: float(actual[k]) for k in declared},
            "tol": tol}


def report_peak_vram(label=""):
    r"""Emit the ACTUAL peak allocator high-water mark, tagged so log scrapers can find it.

    Why this exists: the only VRAM line the trainers emitted was
    `enforce_vram_ceiling`'s `[vram] <free> MiB free of <total>` startup banner.
    `archive/harness/train_harness.py:peak_vram_mb` regexed `(\d+) MiB` and took the max,
    so every row in results/harness_runs.csv recorded FREE-AT-START as if it were
    peak usage. lwr_sft_full_twin's "peak 15263 MB" was free VRAM, not consumption,
    and a memory gate derived from it blocked a run that may fit fine.

    Guarantee: prints exactly one `[vram-peak] <MiB> MiB` line, which is the tag the
    harness now matches. Returns the value in MiB, or None if CUDA is absent.
    Failure mode: torch resets its high-water mark on `reset_peak_memory_stats`;
    call this before any such reset or the number understates.
    """
    import torch
    if not torch.cuda.is_available():
        return None
    peak_mib = int(torch.cuda.max_memory_allocated() / (1024 ** 2))
    resv_mib = int(torch.cuda.max_memory_reserved() / (1024 ** 2))
    print(f"[vram-peak] {peak_mib} MiB allocated / {resv_mib} MiB reserved {label}".rstrip(), flush=True)
    return peak_mib
