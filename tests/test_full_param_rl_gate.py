"""Full-parameter RL is gated on the OPTIMIZER, not closed as a lane.

NO GOVERNING SPEC. Basis: operator instruction 2026-09-02 ("if we could do full rl w
deepspeed or some other means like gradient checkpoints (i.e. no lora) we'd likely have
better results. you able to do this for me"). Promote to a REQ before depending on it.

WHY THIS FILE EXISTS. `.spec/steering/product.md` records full-parameter RL as NOT_RUN --
"does not fit in 16 GB alongside rollouts and a judge" -- and mgpo.py enforced that with an
unconditional raise. But the disposition is arithmetic about fp32 Adam state, not about
full-parameter training as such:

    752M params, fp16 weights 1.5 + fp16 grads 1.5 + Adam m/v 6.0 + fp32 master 3.0
      = ~12.0 GB static, against ~13.4 GB usable -> ~1.4 GB left for G=4 rollouts. Dead.

    Same model with 8-bit Adam: m/v 6.0 -> 1.5, so ~7.5 GB static, ~6 GB headroom. Viable.

So the guard must stay for the configuration that was actually measured to fail, and must
NOT block the configuration that changes the term the measurement was about. These tests
pin both halves, because deleting the guard outright is the tempting wrong fix and would
re-open a lane that genuinely OOMs.
"""
from __future__ import annotations
import os as _o
import sys as _s


import ast
import io

MGPO = _o.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))),
                    "src", "rlv2", "train", "rl", "grpo.py")
SRC = io.open(MGPO, encoding="utf-8").read()


def test_the_optim_flag_offers_the_8bit_path_and_defaults_to_the_safe_one():
    """Default must stay adamw_torch: every prior LoRA result was produced with it, and
    silently switching the optimizer under existing runs would make old and new numbers
    incomparable."""
    tree = ast.parse(SRC)
    optim_call = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
            continue
        if node.args and getattr(node.args[0], "value", None) == "--optim":
            optim_call = node
    assert optim_call is not None, "--optim flag is missing"
    kw = {k.arg: k.value for k in optim_call.keywords}
    assert getattr(kw["default"], "value", None) == "adamw_torch", \
        "default optimizer must remain adamw_torch so prior results stay comparable"
    choices = [e.value for e in kw["choices"].elts]
    assert "adamw_bnb_8bit" in choices, "the 8-bit path that makes full-param fit is missing"


def test_full_param_is_refused_regardless_of_optimizer():
    """CORRECTED 2026-09-02. An earlier version of this test asserted the guard conditioned
    on the optimizer, encoding my claim that 8-bit Adam was the way through. It is not.

    Measured with the FULL lever set (adamw_bnb_8bit + fp32 master + gradient checkpointing):
    OOM at step 3 of 6, 13.06 GiB allocated with 0 free, plus grad_norm=nan on 2 of the first
    3 steps. So the refusal must NOT be escapable by choosing an optimizer -- only by the
    explicit hardware override."""
    assert "args.full_param and not args.allow_full_param_oom" in SRC, \
        "the guard is escapable by optimizer choice; 8-bit Adam was measured to OOM too"
    assert 'args.optim == "adamw_torch"' not in SRC, \
        "the retracted optimizer-conditional guard is back"


def test_the_guard_carries_the_measurement_that_closed_the_lane():
    """A bare 'dispositioned' verdict is unreviewable. The next person must be able to see
    WHAT was tried and HOW it failed without going digging, or they will retry it."""
    for needle in ("13.06 GiB",              # the OOM
                   "grad_norm=nan",          # the second, independent failure
                   "expandable_segments",    # why paper headroom is unreachable on Windows
                   "adamw_bnb_8bit"):        # the lever that was tried and did not save it
        assert needle in SRC, "guard does not carry the evidence: %s" % needle


def test_the_hardware_override_exists_and_is_named_for_its_cost():
    """The lane is closed by THIS card, not by the method. A bigger GPU should not need a
    source edit -- but the flag must not read like a tuning option either."""
    assert "--allow-full-param-oom" in SRC, "no override for larger hardware"
    assert "allow_full_param_oom" in SRC


def test_deepspeed_is_explicitly_ruled_out_with_a_reason():
    """The operator proposed DeepSpeed. It is the wrong tool HERE -- ZeRO 1/2 shard across
    devices and this is one GPU -- and that reasoning must be recorded, or it gets proposed
    again every time the lane comes up."""
    assert "DeepSpeed" in SRC, "no note on why DeepSpeed is not the lever"
    assert "single GPU" in SRC or "single-GPU" in SRC


def test_full_param_loads_fp32_master_weights():
    """Measured 2026-09-02: the full-param probe died at the first clip_grad_norm_ with
    `ValueError: Attempting to unscale FP16 gradients.`

    LoRA gets away with an fp16 load because the base is frozen and PEFT makes the adapter
    params fp32, so the only gradients GradScaler sees are already fp32. Full-param has no
    frozen half -- an fp16 load means fp16 gradients, which GradScaler cannot unscale.

    This is invisible in a dry read of the code and only appears one optimizer step in, so
    it is pinned here rather than left to the next person to rediscover on GPU time."""
    assert "_policy_dtype" in SRC, "policy dtype is no longer chosen per parameterisation"
    assert 'torch.float32 if getattr(args, "full_param", False) else torch.float16' in SRC, \
        "full-param must load fp32 master weights; fp16 makes GradScaler raise"
    assert "dtype=_policy_dtype" in SRC, "the computed dtype is not used at the load site"
    assert "Attempting to unscale FP16 gradients" in SRC, \
        "the measured failure is not recorded where the fix lives"


def test_mid_eval_treats_a_full_param_checkpoint_as_a_model_not_an_adapter():
    """Measured 2026-09-02: both lr-sweep trials trained round 1 successfully and then died
    at the round boundary with

        FileNotFoundError: No adapter_config.json at .../_mid_eval_rd0

    because the between-rounds eval passed the saved directory as `--adapters`. That is
    correct for LoRA and wrong for full-param, where `save_pretrained` writes a plain model
    and no adapter config exists.

    The failure is late (after ~9 minutes of training) and looks like an eval bug rather than
    a parameterisation bug, so it is pinned here."""
    assert "_fp = bool(getattr(args, \"full_param\", False))" in SRC, \
        "mid-eval no longer branches on the parameterisation"
    assert '([] if _fp else ["--adapters", mid_eval_dir])' in SRC, \
        "full-param must NOT pass its checkpoint as --adapters"
    assert "mid_eval_dir if _fp else args.base" in SRC, \
        "full-param must evaluate the saved checkpoint as --model"
    assert "No adapter_config.json" in SRC, \
        "the measured failure is not recorded where the fix lives"


def test_best_checkpoint_reload_handles_full_param_and_lives_in_ONE_place():
    """THIRD occurrence of the same bug, which is why it is now a single helper.

    `save_pretrained` on a full-param policy writes a plain model with no
    adapter_config.json, so `PeftModel.from_pretrained` on it raises

        ValueError: Can't find 'adapter_config.json' at 'mgpo/maxent/_best_rd1'

    It was fixed at the mid-eval call, then reappeared in TWO near-identical
    best-checkpoint reload blocks and killed a full-param run AFTER it had trained to 7/16.
    Two copies of a rule is how a third divergence happens."""
    assert "def _reload_best(" in SRC, "the shared reload helper is gone"
    # Exactly one adapter reload inside the helper, plus the SFT-adapter continuation path.
    # Any growth here means a copy has been reintroduced.
    # Count CALLS, not prose: the helper's own docstring names the API it replaced.
    assert SRC.count("PeftModel.from_pretrained(") <= 2, \
        "best-checkpoint reload has been duplicated again instead of using _reload_best"
    # Assignment sites only -- `def _reload_best(args,` matches the bare name too.
    assert SRC.count("policy = _reload_best(args,") == 2, \
        "both reload sites must route through the helper"
    assert "Can't find 'adapter_config.json'" in SRC, \
        "the measured failure is not recorded where the fix lives"


def test_best_dirs_survive_until_arm_end_because_trl_rebuilds_the_ref_from_their_path():
    """FOURTH manifestation of the full-param checkpoint bug, measured 2026-09-02 on
    fpv2_hpo_lr1e-5 -- and the subtlest: the reload itself SUCCEEDED, and the crash came one
    step later. `_reload_best` leaves config._name_or_path pointing at the best dir, and
    TRL's next GRPOTrainer(...) constructs its REFERENCE model from that exact path string.
    rmtree-after-load therefore made round-2 trainer construction die with

        HFValidationError: Repo id must be in the form ... '<...>/_best_rd0'

    (isdir failed on the freshly-deleted dir, so transformers fell through to the hub).
    Round 1 never hit it because its _name_or_path was args.base. Best dirs are now cleaned
    exactly once, at arm end, after the final save."""
    assert "shutil.rmtree(_best_dir[0]" not in SRC, \
        "a mid-arm delete of the best dir is back; TRL's ref-model load will crash round 2"
    assert '_glob.glob(os.path.join(work, "_best_rd*"))' in SRC, \
        "the deferred arm-end cleanup is gone; best dirs will accumulate 3 GB each"
    assert "config._name_or_path" in SRC, \
        "the WHY is not recorded where the fix lives; the delete will look safe to restore"
    # Companion spike fix, same site: reloading best while the old trainer was alive put
    # old policy + new policy + next ref co-resident (measured 13.22 GiB vs 9.40 without a
    # reload -- a Memory-gate fail that survived by luck). The old trainer is freed first.
    assert "tr = None\n            torch.cuda.empty_cache()" in SRC, \
        "the old trainer is no longer freed before best-reload; boundary peak returns"
    # And it must be a REBIND, not a del: the round summary evaluates `... if tr else []`,
    # and `del tr` turned that into UnboundLocalError (measured, T4 first attempt).
    assert "\n            del tr\n" not in SRC, \
        "del unbinds the name the round summary still evaluates"


def test_full_param_reload_uses_the_training_dtype():
    """A checkpoint reloaded at a different dtype than it trained at is silently NOT the
    model that produced best_score. Full-param trains with fp32 master weights, so its reload
    must be fp32; LoRA stays fp16."""
    helper = SRC.split("def _reload_best(")[1].split("\ndef ")[0]
    assert "dtype=torch.float32" in helper, "full-param reload must use fp32 master weights"
    assert "dtype=torch.float16" in helper, "the LoRA reload path must stay fp16"


def test_the_run_reports_true_peak_memory_not_a_sample():
    """The Memory gate (free >= peak x 1.3) is unevaluable from an nvidia-smi poll, and
    reading one as a peak is what caused a false 'full-param fits' retraction."""
    assert "torch.cuda.max_memory_allocated()" in SRC, \
        "no allocator-reported peak; §5's Memory gate cannot be evaluated"
    assert "[vram-peak]" in SRC, "peak is computed but never surfaced in the log"


def test_embeddings_can_be_frozen_and_the_reason_is_recorded():
    """vocab 248320 x hidden 1024 with tied weights = 254M of 752M params (34%) in ONE
    tensor. Training it costs ~1.5 GB of grad + optimizer state that an RL fine-tune has
    little use for, and embedding rows are the classic fp16 overflow source."""
    assert "--freeze-embeddings" in SRC
    assert "embed_tokens" in SRC and "lm_head" in SRC, \
        "the freeze does not name the tied tensors it holds out"
    assert "254M" in SRC or "254.3M" in SRC or "34%" in SRC, \
        "the size argument that justifies the flag is not recorded"


def test_optim_is_actually_passed_to_the_trainer_not_just_parsed():
    """A flag that is accepted and ignored is worse than no flag: the run would report
    full-param success while silently using fp32 Adam and OOMing or thrashing."""
    assert 'optim=getattr(args, "optim", "adamw_torch")' in SRC, \
        "--optim is parsed but never reaches the trainer config"
