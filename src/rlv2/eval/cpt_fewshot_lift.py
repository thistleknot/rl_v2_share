"""src/rlv2/eval/cpt_fewshot_lift.py — fast isolated CPT knowledge-injection lift, few-shot closed-book (NO re-CPT, NO SFT).

Isolates CPT's knowledge injection with the minimum confound: K original-squad Q/A few-shot exemplars (general
topics -> ZERO ML-domain leakage) elicit the answer format; CLOSED-BOOK (no context) domain-QA eval tests
INTERNAL parametric knowledge; CPT is the ONLY A/B variable. Reuses the no-replay GaLore CPT base already on disk.
  A (base)     = MODEL_NAME             + K-shot squad -> closed-book span_qa_eval
  B (CPT base) = cpt_probe_galore/base  + same K-shot -> closed-book span_qa_eval   (no-replay full-rank FFN GaLore)
lift = B_recall - A_recall (entity_validity = key-entity F1 x succinctness). Smoke-gates few-shot format first.
Writes data/domain/probe/cpt_fewshot_lift.txt.
"""
import json
import random
import sys
sys.path.insert(0, ".")
import torch
from rlv2.common.config import MODEL_NAME

PROBE = "data/domain/probe"
CPT_BASE = "cpt_probe_galore/base"      # no-replay full-rank FFN GaLore CPT base (already on disk)
SQUAD = "data/raw/readingcomp.jsonl"
K = 4
# CLOSED-BOOK system prompt — NOT the repo's context-grounded SYSTEM_PROMPT (whose abstain-if-no-context protocol
# would make a closed-book model abstain on everything, killing the knowledge test).
CB_SYSTEM = ("You are a knowledgeable assistant. Answer each question directly and concisely from your own "
             "knowledge. Give the specific factual answer only.")


def _squad_shots(k=K, seed=0):
    """K original-squad answerable short-span Q/A exemplars (general topics, no ML leakage)."""
    rows = [json.loads(l) for l in open(SQUAD, encoding="utf-8") if l.strip()]
    ans = [r for r in rows if str(r.get("source", "")).startswith("rajpurkar/squad") and (r.get("answer") or "").strip()]
    random.Random(seed).shuffle(ans)
    return [(r["question"], r["answer"]) for r in ans[:k]]


def _fewshot_messages(question, shots):
    """CB system + K squad (user/assistant) turns + the CLOSED-BOOK domain question (no context)."""
    msgs = [{"role": "system", "content": CB_SYSTEM}]
    for q, a in shots:
        msgs.append({"role": "user", "content": f"Question: {q}"})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": f"Question: {question}"})
    return msgs


def _eval_base(base_model, recs, shots, tag, sft_path=None):
    """Load base_model (optionally + SFT adapter for Test 2), K-shot closed-book greedy gen, entity_validity recall."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from rlv2.eval.policy_eval import generate_greedy_batch
    from rlv2.train.rl.red_queen_validity import _looks_abstain
    from rlv2.train.rl.entity_validity import entity_validity
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb, device_map={"": 0},
                                                 trust_remote_code=True)
    if sft_path is not None:                              # Test 2: + domain-QA SFT adapter (activate latent CPT knowledge)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, sft_path)
    prompts = [tok.apply_chat_template(_fewshot_messages(r["prompt"], shots), tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) for r in recs]
    gens = generate_greedy_batch(model, tok, prompts, max_new_tokens=128, batch_size=16)
    del model
    torch.cuda.empty_cache()
    recalls, ab_n = [], 0
    for r, g in zip(recs, gens):
        ab_n += int(_looks_abstain(g))
        recalls.append(entity_validity(g, r["answer"], "", ""))
    return {"tag": tag, "recall": sum(recalls) / len(recalls) if recalls else 0.0,
            "answer_rate": 1 - ab_n / len(recs) if recs else 0.0, "abstain": ab_n, "gens": gens[:3]}


def smoke(shots):
    """5-question base+few-shot format check: does the 0.8B answer (not ramble/abstain)?"""
    recs = [json.loads(l) for l in open(f"{PROBE}/span_qa_eval.jsonl", encoding="utf-8") if l.strip()][:5]
    m = _eval_base(MODEL_NAME, recs, shots, "smoke")
    print(f"[smoke] base+{K}-shot: answer_rate={m['answer_rate']:.2f} recall={m['recall']:.3f}", flush=True)
    for i, g in enumerate(m["gens"]):
        print(f"  ex{i}: {g[:140]!r}", flush=True)
    ok = m["answer_rate"] >= 0.5
    print(f"[smoke] {'PASS - few-shot elicits format' if ok else 'FAIL - base rambles/abstains; SFT-teacher fallback'}",
          flush=True)
    return ok


def main():
    shots = _squad_shots()
    print(f"[fewshot-lift] {K} squad shots: {[q[:40] for q, _ in shots]}", flush=True)
    if not smoke(shots):
        print("[fewshot-lift] SMOKE FAILED — base won't few-shot; the SFT-teacher fallback is the next step "
              "(not auto-run).", flush=True)
        return
    recs = [json.loads(l) for l in open(f"{PROBE}/span_qa_eval.jsonl", encoding="utf-8") if l.strip()]
    print(f"[fewshot-lift] Test 1 (few-shot only) on {len(recs)} closed-book domain QA...", flush=True)
    a = _eval_base(MODEL_NAME, recs, shots, "base")
    print(f"[fewshot-lift] base: recall={a['recall']:.3f} answer_rate={a['answer_rate']:.3f}", flush=True)
    b = _eval_base(CPT_BASE, recs, shots, "cpt (no-replay galore)")
    print(f"[fewshot-lift] cpt : recall={b['recall']:.3f} answer_rate={b['answer_rate']:.3f}", flush=True)
    lift = b["recall"] - a["recall"]
    L = ["=" * 80, " CPT few-shot CLOSED-BOOK lift — knowledge injection isolated (no SFT, no replay)", "=" * 80,
         f"{K}-shot original-squad exemplars | closed-book | {len(recs)} domain-QA | entity_validity recall", "",
         f"{'arm':<26}{'recall':>8}{'answer_rate':>13}{'abstain':>9}"]
    for m in (a, b):
        L.append(f"{m['tag']:<26}{m['recall']:>8.3f}{m['answer_rate']:>13.3f}{m['abstain']:>9}")
    L += ["-" * 80, f"LIFT (cpt - base) = {lift:+.3f}",
          ("CPT INJECTS knowledge (few-shot-retrievable) — real lift under clean isolation."
           if lift > 0.02 else
           "FLAT/DOWN — CPT does not inject even few-shot closed-book. Next rung: + domain-QA SFT (Test 2).")]
    open(f"{PROBE}/cpt_fewshot_lift.txt", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L), flush=True)


def test2():
    """Rung 2 — + domain-QA SFT (activate latent CPT knowledge). Reuses existing SFT adapters (on disk)."""
    shots = _squad_shots()
    recs = [json.loads(l) for l in open(f"{PROBE}/span_qa_eval.jsonl", encoding="utf-8") if l.strip()]
    print(f"[test2] + domain-QA SFT, {K}-shot, closed-book on {len(recs)} domain QA...", flush=True)
    a = _eval_base(MODEL_NAME, recs, shots, "base+SFT", sft_path="sft_probe_nocpt/sft_adapter")
    print(f"[test2] base+SFT: recall={a['recall']:.3f}", flush=True)
    b = _eval_base(CPT_BASE, recs, shots, "cpt+SFT", sft_path="sft_probe_galore/sft_adapter")
    print(f"[test2] cpt+SFT : recall={b['recall']:.3f}", flush=True)
    lift = b["recall"] - a["recall"]
    L = ["=" * 80, " CPT Test 2 — + domain-QA SFT, few-shot closed-book (activate latent CPT knowledge)", "=" * 80,
         f"{'arm':<16}{'recall':>8}{'answer_rate':>13}{'abstain':>9}"]
    for m in (a, b):
        L.append(f"{m['tag']:<16}{m['recall']:>8.3f}{m['answer_rate']:>13.3f}{m['abstain']:>9}")
    L += ["-" * 80, f"LIFT (cpt - base) = {lift:+.3f}",
          ("CPT knowledge ACTIVATES under domain-QA SFT — real lift." if lift > 0.02 else
           "FLAT/DOWN — CPT does not inject even with domain-QA SFT. CPT closed on recall (all rungs negative).")]
    open(f"{PROBE}/cpt_fewshot_test2.txt", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    test2() if "test2" in sys.argv else main()
