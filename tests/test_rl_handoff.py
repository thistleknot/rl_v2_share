"""Unit test: RL training state survives the GPU handoff to Ollama judge.

Tests the EXACT sequence that happens per GRPO step:
1. Model on CUDA, optimizer has state (simulates mid-training)
2. Offload model to CPU, free GPU entirely
3. Call Ollama judge with NO contention (must be < 45s for G=4)
4. Restore model to CUDA
5. Do a backward pass, confirm parameters updated

If any phase fails or judge exceeds 45s, the handoff pattern is broken.
"""
from __future__ import annotations
import os as _o, sys as _s

import gc
import json
import io
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT = "harness_runs/sft_full_lr2e5_ep3"
MIX_FILE = "data/rl/rl_mix_balanced.jsonl"
JUDGE_TIMEOUT = 70.0  # seconds — measured floor for G=4 with 1555-token rubric+prompt on 2B @ RTX5000


def test_handoff():
    print("=" * 60)
    print("UNIT TEST: RL state preservation across judge handoff")
    print("=" * 60)

    # ── Phase 1: Load model, create optimizer, do a dummy forward ──
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, torch_dtype=torch.float16, trust_remote_code=True
    ).cuda()
    # Freeze attention (same as mgpo.py full-param FFN-only)
    for name, param in model.named_parameters():
        if any(attn in name for attn in ("q_proj", "k_proj", "v_proj", "o_proj",
                                          "in_proj_qkv", "in_proj_a", "in_proj_b",
                                          "in_proj_z", "out_proj")):
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-6
    )

    # Dummy forward to create optimizer state
    dummy_input = tok("Hello world", return_tensors="pt").to("cuda")
    with torch.amp.autocast("cuda", dtype=torch.float16):
        loss = model(**dummy_input, labels=dummy_input["input_ids"]).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Snapshot a parameter to compare after round-trip
    target_param = next(p for p in model.parameters() if p.requires_grad)
    param_before = target_param.data.clone().cpu()
    opt_state_before = optimizer.state[target_param]["exp_avg"].clone().cpu()

    load_seconds = time.perf_counter() - t0
    gpu_used = torch.cuda.memory_allocated() // (1024 * 1024)
    print(f"\n[phase 1] Model loaded + optimizer stepped | {load_seconds:.1f}s | GPU={gpu_used} MiB")
    print(f"  param sample: {target_param.data.flatten()[:5].tolist()}")
    print(f"  opt exp_avg:  {opt_state_before.flatten()[:5].tolist()}")

    # ── Phase 2: Offload to CPU, free GPU ──
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    model.to("cpu")
    # Also move optimizer state tensors to CPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                state[k] = v.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    offload_seconds = time.perf_counter() - t1

    free_mib = torch.cuda.mem_get_info()[0] // (1024 * 1024)
    print(f"\n[phase 2] Offloaded to CPU | {offload_seconds:.1f}s | GPU free={free_mib} MiB")
    assert free_mib > 14000, f"GPU not properly freed: {free_mib} MiB free (expected >14000)"
    print(f"  PASS: GPU is clear ({free_mib} MiB free)")

    # ── Phase 3: Call Ollama judge with exclusive GPU ──
    # Pre-warm the judge so we measure inference, not cold-load
    import urllib.request
    warm_body = json.dumps({"model": "qwen3.5-oc:2b", "prompt": "hello", "stream": False}).encode()
    warm_req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=warm_body, headers={"Content-Type": "application/json"}, method="POST"
    )
    print("\n[phase 3] Pre-warming judge...", end=" ", flush=True)
    t_warm = time.perf_counter()
    with urllib.request.urlopen(warm_req, timeout=120) as resp:
        resp.read()
    print(f"warm in {time.perf_counter() - t_warm:.1f}s")

    mix = [json.loads(l) for l in io.open(MIX_FILE, encoding="utf-8") if l.strip()][:1]
    r = mix[0]
    from rlv2.judge.reward import llm_judge_batch_train
    groups = [{
        "question": r.get("prompt", ""),
        "gold": r.get("gold", ""),
        "context": r.get("context", ""),
        "completions": [
            "I do not know.",
            "The answer is 42.",
            "Based on the context, the answer is unclear.",
            "This follows directly from the passage.",
        ],
    }]

    t2 = time.perf_counter()
    scores = llm_judge_batch_train(groups)
    judge_seconds = time.perf_counter() - t2

    print(f"\n[phase 3] Judge scored G=4 | {judge_seconds:.1f}s | scores={scores}")
    assert judge_seconds < JUDGE_TIMEOUT, (
        f"FAIL: judge took {judge_seconds:.1f}s, contract is <{JUDGE_TIMEOUT}s — contention exists"
    )
    assert len(scores) == 1 and len(scores[0]) == 4, f"unexpected shape: {scores}"
    print(f"  PASS: judge completed in {judge_seconds:.1f}s < {JUDGE_TIMEOUT}s")

    # ── Phase 4: Unload judge, restore model to CUDA ──
    from rlv2.judge.reward import unload_ollama_model
    t3 = time.perf_counter()
    unload_ollama_model()
    model.to("cuda")
    # Restore optimizer state tensors to CUDA
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and not v.is_cuda:
                state[k] = v.cuda()
    restore_seconds = time.perf_counter() - t3

    print(f"\n[phase 4] Restored to CUDA | {restore_seconds:.1f}s")

    # Verify parameter identity preserved
    param_after_restore = target_param.data.clone().cpu()
    opt_state_after = optimizer.state[target_param]["exp_avg"].clone().cpu()
    assert torch.equal(param_before, param_after_restore), "FAIL: parameter values changed during offload!"
    assert torch.equal(opt_state_before, opt_state_after), "FAIL: optimizer state changed during offload!"
    print(f"  PASS: parameter values identical after round-trip")
    print(f"  PASS: optimizer exp_avg identical after round-trip")

    # ── Phase 5: Backward pass works after restore ──
    t4 = time.perf_counter()
    optimizer.zero_grad()
    dummy_input2 = tok("Test backward pass", return_tensors="pt").to("cuda")
    with torch.amp.autocast("cuda", dtype=torch.float16):
        loss2 = model(**dummy_input2, labels=dummy_input2["input_ids"]).loss
    loss2.backward()
    optimizer.step()
    backward_seconds = time.perf_counter() - t4

    param_after_backward = target_param.data.clone().cpu()
    param_changed = not torch.equal(param_before, param_after_backward)
    print(f"\n[phase 5] Backward + optimizer step | {backward_seconds:.1f}s | loss={loss2.item():.4f}")
    print(f"  param changed after step: {param_changed}")
    assert param_changed, "FAIL: optimizer step didn't update parameters!"
    print(f"  PASS: parameters updated by optimizer after restore")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"ALL PHASES PASSED")
    print(f"  load:     {load_seconds:.1f}s")
    print(f"  offload:  {offload_seconds:.1f}s")
    print(f"  judge:    {judge_seconds:.1f}s (contract: <{JUDGE_TIMEOUT}s)")
    print(f"  restore:  {restore_seconds:.1f}s")
    print(f"  backward: {backward_seconds:.1f}s")
    print(f"  TOTAL handoff overhead: {offload_seconds + restore_seconds:.1f}s")
    print(f"{'=' * 60}")

    del model, optimizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    test_handoff()
