"""
src/rlv2/train/rl/entity_validity.py — un-gameable, deterministic "context-entity-precision" correctness signal.

The old yardstick (gold-token RECALL over ALL gold tokens) punished terse-correct answers:
  Q "who proposed the Gini coefficient?"  gold "...proposed by Corrado Gini as a measure of..."
  answer "Corrado Gini"  -> token-recall 0.13 (covers 2 of ~12 gold tokens) even though it is PERFECT.
That noise made the Red Queen's validity curve unreadable and manufactured phantom "exploits".

This metric scores against the gold's KEY ENTITIES (proper-noun phrases, numbers, salient content
terms) instead of every token, and rewards efficiency:
  entity_recall    = gold key-entities the answer covers            -> comprehensive
  entity_precision = answer's asserted entities that are in gold+ctx -> on-point (punishes wrong/extra padding)
  succinctness     = length efficiency vs the needed length          -> succinct
  validity = F1(precision, recall) * succinctness

"Corrado Gini" -> ~1.0 ; a 200-token ramble that names Max Lorenz instead scores low. Deterministic,
CPU-only, no external deps — so it is un-gameable and cheap enough to run inside the training loop.
"""
import re

# Small closed stopword list — deterministic, no nltk download needed.
_STOP = frozenset("""
a an the this that these those of in on at to from by for with about into over after before
and or but nor so yet as if then than because while is are was were be been being am do does did
has have had having will would shall should can could may might must not no nor it its it's they
them their he she his her him you your we our us i me my mine ours yours theirs which who whom whose
what when where why how all any both each few more most other some such only own same s t can
""".split())

_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_NUM = re.compile(r"\d[\d,]*\.?\d*%?")
# Maximal run of Capitalized words (a proper-noun phrase), allowing lowercase connectors of/the/and/de.
_PROPER = re.compile(r"\b([A-Z][\w'\-]*(?:\s+(?:of|the|and|de|von|van|di|da|la|le)\s+|\s+)?){0,4}[A-Z][\w'\-]*\b")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (text or "").lower())).strip()


def _content_tokens(text: str) -> set:
    return {w for w in _norm(text).split() if w and w not in _STOP and len(w) > 1}


def key_entities(text: str) -> dict:
    """Tiered key-entities of a text: proper-noun phrases, numbers, and salient content tokens.
    Returns normalized strings so matching is case/punctuation-insensitive."""
    text = text or ""
    propers = set()
    for m in _PROPER.finditer(text):
        toks = m.group(0).strip().split()
        while toks and toks[0].lower() in _STOP:   # strip leading stopwords (sentence-initial "The Gini" -> "Gini")
            toks = toks[1:]
        while toks and toks[-1].lower() in _STOP:  # strip trailing stopwords
            toks = toks[:-1]
        if toks:
            propers.add(_norm(" ".join(toks)))
            if len(toks) > 1:                      # also expose the head token ("Gini" from "Corrado Gini")
                propers.add(_norm(toks[-1]))
    propers = {p for p in propers if p}
    numbers = {_norm(m.group(0)) for m in _NUM.finditer(text)}
    numbers = {n for n in numbers if n}
    content = _content_tokens(text)
    return {"proper": propers, "number": numbers, "content": content}


def _contains(hay: str, needle: str) -> bool:
    if not needle:
        return False
    return re.search(r"\b" + re.escape(needle) + r"\b", hay) is not None


def _gold_key_set(ge: dict) -> set:
    """Gold's answer-bearing keys: proper nouns + numbers if any exist, else salient content tokens."""
    keys = ge["proper"] | ge["number"]
    return keys if keys else ge["content"]


def entity_recall(answer: str, gold: str, context: str = "") -> float:
    keys = _gold_key_set(key_entities(gold))
    if not keys:
        return 0.0
    ans = _norm(answer)
    return sum(_contains(ans, k) for k in keys) / len(keys)


def entity_precision(answer: str, gold: str, context: str = "") -> float:
    """Of the answer's ASSERTED entities (proper nouns + numbers), the fraction grounded in gold+context.
    Empty asserted set -> 1.0 (nothing false claimed); recall then carries correctness."""
    ae = key_entities(answer)
    asserted = ae["proper"] | ae["number"]
    if not asserted:
        return 1.0
    ground = _norm(gold + " " + (context or ""))
    return sum(_contains(ground, k) for k in asserted) / len(asserted)


def succinctness(answer: str, gold: str, floor: int = 24) -> float:
    """1.0 while the answer is no longer than the needed length (max(gold_len, floor)); decays for padding."""
    a = len(_norm(answer).split())
    if a == 0:
        return 0.0
    target = max(len(_norm(gold).split()), floor)
    return min(1.0, target / a)


def entity_validity(completion: str, gold: str, domain: str = "", context: str = "") -> float:
    """Drop-in un-gameable correctness in [0,1]. Math delegates to the programmatic verifier."""
    completion = completion or ""
    gold = (gold or "").strip()
    if not gold:
        return 0.0
    if domain == "Math":
        try:
            from rlv2.judge.verifiers import verify_math
            v = verify_math(completion, gold)
            if v is not None:
                return 1.0 if v else 0.0
        except Exception:
            pass
    # Short/exact-answer golds (multiple-choice options, single terms like "feature", "gas exchange")
    # are under-served by proper-noun/content extraction: use exact-token recall OR a semantic (NLI)
    # backoff so an answer that conveys the concept without the literal gold word is not scored 0.
    if len(_content_tokens(gold)) <= 4:
        exact = entity_recall(completion, gold, context)
        from rlv2.judge.reward import nli_entailment
        sem = float(nli_entailment(completion, gold))   # fail-fast: a broken NLI model must surface, not silently score 0
        return max(exact, sem) * succinctness(completion, gold)
    p = entity_precision(completion, gold, context)
    r = entity_recall(completion, gold, context)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return f1 * succinctness(completion, gold)


CORRECT, NEIGHBOR, HALLUCINATED = "correct", "neighbor", "hallucinated"


def classify_answer(completion: str, gold: str, siblings=()) -> str:
    """Trichotomy over one answer: correct / neighbor-conflated / hallucinated (Gekhman arXiv:2405.05904).

    Thesis: a scalar recall cannot distinguish "knows nothing" from "knows the neighbourhood but not the
    fact" — yet those imply opposite next moves (more data vs. contrastive separation). The confusable
    cluster is supplied as `siblings`: the gold answers of the OTHER questions drawn from the SAME probe
    span, so a "neighbor" verdict means the model named a real entity from the right region of the corpus
    and attached it to the wrong question. That is the bucket promptv2 wants driven to zero.

    Decision rule (no magic thresholds — membership, not degree):
      correct      -> the answer covers >=1 of gold's key entities.
      neighbor     -> covers NO gold key entity, but >=1 key entity of some sibling gold.
      hallucinated -> covers neither. Abstentions land here too (they assert no key entity).

    Invariant: exactly one label per answer, so the three buckets partition n.
    Failure mode: an empty `siblings` collapses the neighbor bucket into hallucinated by construction —
    the caller must pass the cluster, or the trichotomy degenerates to a binary silently.
    """
    ans = _norm(completion or "")
    if not ans:
        return HALLUCINATED
    if any(_contains(ans, k) for k in _gold_key_set(key_entities(gold or ""))):
        return CORRECT
    for sib in siblings:
        if sib is None or _norm(sib) == _norm(gold or ""):
            continue
        if any(_contains(ans, k) for k in _gold_key_set(key_entities(sib))):
            return NEIGHBOR
    return HALLUCINATED


def trichotomy(completions, golds, siblings=None) -> dict:
    """Bucket counts + rates over a batch. `siblings` defaults to the golds themselves (leave-one-out),
    which is the intended use: the eval set's own gold answers ARE the confusable cluster of the span."""
    golds = list(golds)
    pool = list(golds if siblings is None else siblings)
    labels = [classify_answer(c, g, pool) for c, g in zip(completions, golds)]
    n = len(labels) or 1
    counts = {k: labels.count(k) for k in (CORRECT, NEIGHBOR, HALLUCINATED)}
    assert sum(counts.values()) == len(labels), "trichotomy must partition the batch"
    return {"counts": counts, "rates": {k: v / n for k, v in counts.items()}, "labels": labels, "n": len(labels)}


if __name__ == "__main__":
    gold = "The Gini coefficient was proposed by Corrado Gini as a measure of inequality of income or wealth."
    cases = [
        ("terse-correct",      "Corrado Gini"),
        ("verbose-correct",    "The Gini coefficient was proposed by the Italian statistician Corrado Gini."),
        ("verbose-WRONG",      "The Gini coefficient is a measure developed by the economist Max Lorenz in the early 1900s to quantify wealth distribution across populations of interest."),
        ("padded-plausible",   "This is a classic question about inequality metrics and economics; the coefficient is famous and widely used in many countries for policy analysis and research."),
        ("abstain",            "I cannot answer this from the context."),
    ]
    from rlv2.train.rl.exploit_finder import correctness as old_correctness
    print(f"{'case':<18} {'entity':>7} {'old(recall/nli)':>16}")
    for name, ans in cases:
        print(f"{name:<18} {entity_validity(ans, gold):>7.3f} {old_correctness(ans, gold):>16.3f}")
