from pathlib import Path

MODEL_NAME = "unsloth/Qwen3.5-0.8B"  # smoke-test model (cached); swap to Qwen/Qwen2.5-2B after `hf auth login`

SYSTEM_PROMPT = (
    "You are a helpful assistant. When context is provided, answer based only on that context. "
    "If the context does not support an answer, respond with exactly: Not supported by context\n"
    "If the question cannot be answered by anyone, respond with exactly: Unanswerable"
)

ABSTAIN_R2 = "not supported by context"
ABSTAIN_R4 = "unanswerable"

# Data paths
DATA_RAW       = Path("data/raw")
DATA_LABELED   = Path("data/labeled")
DATA_SYNTHETIC = Path("data/synthetic")
DATA_SCORED    = Path("data/scored")
DATA_FILTERED  = Path("data/filtered")
CHROMA_DIR     = Path("data/chroma")

# Model checkpoints
CHECKPOINT_DIR  = Path("checkpoints")
SFT_CHECKPOINT  = CHECKPOINT_DIR / "sft_prime"
GRPO_CHECKPOINT = CHECKPOINT_DIR / "grpo"

# Data manifest
MANIFEST_PATH   = Path("data/manifest.json")

# GRPO / ReST hyper-parameters
GRPO_GROUP_SIZE   = 4      # G: completions per (q,c) step — middle of 8–16 range

# :BandedCostAllocation: RL batch — per-cell [min,max] GRPO-group band, N solved (not fixed 50).
# min/max are the per-cell group band; BAND_BETA is cost-awareness (1=cheap-cells->max, Math/Code->min).
# (1,3) N≈22 FAST Red-Queen proof · (2,3) N≈33 balanced · (3,5) N≈52 production quality.
USE_BANDED_BATCH  = False   # lift-measurement control: match the plateau's fixed-50 λ=0.5 batch (banded 1,3 was the wallclock scaffold, not the plateau config)
BAND_MIN, BAND_MAX, BAND_BETA = 1, 3, 1.0
# RL completion ceiling — sets BOTH the eligibility filter (SELECT answers <= ceiling, per :Truncation-
# FreeSampling:) AND the rollout cap, matched so NOTHING is ever truncated. 256 = fast :AtomicProofBatch:
# (selects short-answer records within every domain; drops long Math/Code CoT — clean + fast, disposable
# policy); 1024 = production (per-cell budget on the filtered set). NEVER truncate — always select.
RL_COMPLETION_CEILING = 256

# :ThermalGovernor: — pause between RL rounds when the Max-Q GPU thermal-throttles (co-resident judge
# keeps it at 100% util so it never cools on its own). Thresholds below the observed 70C throttle point.
THERMAL_GOVERNOR       = True
THERMAL_HOT_C          = 68     # pause trigger (or the SW-thermal throttle bit)
THERMAL_RESUME_C       = 60     # resume once cooled to here
THERMAL_POLL_S         = 5      # cooldown poll interval
THERMAL_MAX_COOLDOWN_S = 180    # hard cap on a single pause
THERMAL_EMA_ALPHA      = 0.3    # temp-EMA smoothing (control input for the future momentum controller)
GRPO_STEPS_PER_REST_ROUND = 200   # inner GRPO steps before each ReST data refresh
REST_ROUNDS       = 1       # outer ReST iterations
KL_THRESHOLD      = 0.1
HIGH_REWARD_THRESHOLD = 0.7
LOW_ENTROPY_THRESHOLD = 0.6

# Embedding model used in ChromaDB index
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION = "rl_v2_corpus"
FETCH_K = 144

# Sampler configs for SFT-variance generation
SAMPLER_CONSERVATIVE = {"temperature": 0.4, "top_p": 0.80}
SAMPLER_BALANCED     = {"temperature": 0.7, "top_p": 0.90}
SAMPLER_CREATIVE     = {"temperature": 1.0, "top_p": 0.95}
SAMPLER_GREEDY       = {"temperature": 0.0, "do_sample": False}  # temp 0 baseline
