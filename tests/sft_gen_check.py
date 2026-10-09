"""Check what the SFT checkpoint generates at temp=1.0 (GRPO sampling config).

If the SFT base already produces garbage at this temperature, GRPO will amplify it.
"""
from __future__ import annotations
import os as _o, sys as _s

import json
import io
import sys
import torch

sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from rlv2.eval import cpt_fewshot_lift as FS

CHECKPOINT = "harness_runs/sft_full_lr2e5_ep3"
MIX = "data/rl/rl_mix_balanced.jsonl"
THINK_ID = 248068
THINK_CLOSE_ID = 248069

tok = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=torch.float16, trust_remote_code=True
).cuda().eval()

gen_cfg = GenerationConfig(
    max_new_tokens=256,
    do_sample=True,
    temperature=1.0,
    top_p=0.95,
    suppress_tokens=[THINK_ID, THINK_CLOSE_ID],
    pad_token_id=tok.pad_token_id,
)

shots = FS._squad_shots()
mix = [json.loads(l) for l in io.open(MIX, encoding="utf-8") if l.strip()][:5]

print("=" * 70)
print("SFT CHECKPOINT GENERATION CHECK @ temp=1.0, top_p=0.95, suppress_think")
print("=" * 70)

for idx, r in enumerate(mix):
    ptext = tok.apply_chat_template(
        FS._fewshot_messages(r["prompt"], shots),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tok(ptext, return_tensors="pt").to("cuda")
    prompt_len = inputs["input_ids"].shape[1]
    out = model.generate(**inputs, generation_config=gen_cfg, num_return_sequences=4)

    print(f"\n[prompt {idx}] q={r['prompt'][:80]}")
    print(f"  gold={r.get('gold', '')[:80]}")
    for i, o in enumerate(out):
        gen_ids = o[prompt_len:].tolist()
        text = tok.decode(gen_ids, skip_special_tokens=True)
        print(f"  [{i}] len={len(gen_ids):3d} | {text[:150]}")

print("\n" + "=" * 70)
print("If completions above are mostly garbage/repetitive, SFT is not clean.")
print("=" * 70)
