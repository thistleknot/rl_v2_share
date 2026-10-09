import re


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)


def _parse_numbers(text: str) -> list:
    """
    Parse all numbers from text in order of appearance, handling $, comma, % etc.
    Returns a list of floats (order preserved; duplicates kept).
    """
    numbers = []
    pattern = r'[\$]?[\-]?\d+(?:[,]\d+)*(?:\.\d+)?[%]?'
    for match in re.finditer(pattern, text):
        num_str = match.group(0)
        try:
            num_str = num_str.replace('$', '').replace('%', '').replace(',', '')
            numbers.append(float(num_str))
        except ValueError:
            continue
    return numbers


def _extract_numbers_from_text(text: str) -> set:
    """
    Extract final answer number(s) from text.
    Priority:
    1. Text after last #### marker  -> ALL numbers in that region (multi-answer set)
    2. Contents of \\boxed{...}      -> ALL numbers in that region (multi-answer set)
    3. Fallback: the LAST number in the string only.

    The #### / boxed regions are explicit final-answer regions, so every number
    there is a real answer (e.g. quadratic roots "#### 2, -3" -> {2, -3}). The
    bare fallback must NOT collect every number, or intermediate computation
    numbers in a worked solution ("3*6=18 ... total 26") would pollute the set
    and reject a correct final answer. The last number is the gsm8k convention
    for the final answer. (Limitation: multi-answer expressed without a #### /
    boxed marker is not captured -- acceptable for the gsm8k POC corpus.)
    """
    # Priority 1: #### marker (gsm8k style) -- region may hold multiple answers
    if '####' in text:
        after_hash = text[text.rfind('####') + 4:].strip()
        nums = _parse_numbers(after_hash)
        if nums:
            return set(nums)

    # Priority 2: \boxed{...} -- region may hold multiple answers
    boxed_match = re.search(r'\\boxed\{([^}]*)\}', text)
    if boxed_match:
        nums = _parse_numbers(boxed_match.group(1))
        if nums:
            return set(nums)

    # Priority 3: bare text -> last number only (robust to worked-solution noise)
    nums = _parse_numbers(text)
    if not nums:
        return set()
    return {nums[-1]}


def _sets_equal_with_tolerance(set1: set, set2: set, tolerance: float = 1e-6) -> bool:
    """
    Compare two sets of floats with tolerance.
    abs(a - b) <= tolerance * max(1, abs(a), abs(b))
    """
    if len(set1) != len(set2):
        return False

    used = set()
    for num1 in set1:
        found = False
        for num2 in set2:
            if num2 not in used:
                max_val = max(1, abs(num1), abs(num2))
                if abs(num1 - num2) <= tolerance * max_val:
                    used.add(num2)
                    found = True
                    break
        if not found:
            return False

    return True


def verify_math(answer: str, gold: str):
    """
    Programmatic correctness check for math answers.
    Returns True/False if a numeric final answer is extractable from BOTH
    answer and gold; returns None if gold has no extractable numeric answer
    (caller falls back to the LLM judge).
    """
    try:
        answer_cleaned = _strip_think_blocks(answer)
        gold_numbers = _extract_numbers_from_text(gold)

        if not gold_numbers:
            return None

        candidate_numbers = _extract_numbers_from_text(answer_cleaned)

        if not candidate_numbers:
            return False

        return _sets_equal_with_tolerance(candidate_numbers, gold_numbers)
    except Exception:
        return False


if __name__ == "__main__":
    # single-answer
    assert verify_math("The total is 36", "...\n#### 36") is True
    assert verify_math("#### 19", "#### 19") is True
    assert verify_math("the answer is 40", "#### 36") is False
    # THE key real case: worked solution, no #### in candidate -> last number wins
    assert verify_math("White: 3*6=18, Blue: 2*4=8, total 18+8=26", "...\n#### 26") is True
    assert verify_math("He has 2+4+3+10 = 19 pets.", "#### 19") is True
    # multi-answer expressed in the #### / boxed final-answer region (order-independent)
    assert verify_math("#### 2, -3", "#### -3, 2") is True
    assert verify_math("the roots are \\boxed{2, -3}", "#### -3, 2") is True
    # non-numeric candidate, and non-numeric gold (fallback to judge)
    assert verify_math("forty-two", "#### 42") is False
    assert verify_math("anything", "the answer is blue") is None
    print("verify_math self-test PASS")


# ═══════════════════════════════════════════════════════════════════════════
# RLVR TIER 2 — entailment against the gold's own claims
#
# Basis: operator 2026-09-12. The LLM judge is the harness bottleneck: a 16-answer
# 3-read eval costs ~30 min locally, which dwarfs ReFT training (16k params,
# seconds) and dominates every RL step. A verifiable reward removes it.
#
# WHY ENTAILMENT AND NOT TERM MATCHING. Measured on 60 real golds: triplet
# extraction fires on 88%, but the triplets are uneven -- clean
# ("Gini impurity index | is defined for | nodes containing m classes") next to
# degenerate ("they | are | equal") and subject-swallowed-the-sentence. Scoring on
# term overlap would reward parser noise and, worse, teach keyword stuffing: a
# model that names "convex hull" without the reasoning. Entailment asks the right
# question -- does the answer SUPPORT this claim -- and degrades gracefully when a
# triplet is noisy, where exact matching does not.
#
# Tiering is deliberate: exact numbers where they exist (free, deterministic),
# entailment otherwise, judge only where both abstain.
# ═══════════════════════════════════════════════════════════════════════════
_TRIPLET_EX = None


def _triplet_spans(text: str) -> list:
    """SPO triplets as (subject, predicate, object) tuples. [] when none found.

    Spec: .spec/specs/r3-domain-injection/rlvr-premise-spec.md (pass 1).

    The slot boundaries are the POINT here -- the premise key checks each slot
    separately. Flattening to one string and re-splitting by thirds produces
    slots like subject="the", which is the defect recorded in docs/README-rlvr.md.
    """
    global _TRIPLET_EX
    try:
        if _TRIPLET_EX is None:
            import spacy, triplet_extract as _te
            _TRIPLET_EX = _te.OpenIEExtractor(nlp=spacy.load("en_core_web_sm"))
        out = []
        for t in _TRIPLET_EX.extract_triplet_objects(text) or []:
            s_, r_, o_ = (getattr(t, "subject", ""), getattr(t, "relation", ""),
                          getattr(t, "object", ""))
            if s_ and r_ and o_:
                out.append((str(s_).strip(), str(r_).strip(), str(o_).strip()))
        return out
    except Exception:
        return []


def _triplets(text: str) -> list:
    """SPO triplets from a gold answer, flattened. [] when none found."""
    return [" ".join(t).strip() for t in _triplet_spans(text)]


def _tokens(text):
    """Content tokens: lowercase, punctuation to whitespace, empties dropped.

    Spec: .spec/specs/r3-domain-injection/rlvr-premise-spec.md (scoring).
    `= 1/n` and `= 1 n` must tokenize alike -- spaCy rewrites the punctuation in
    its slot text, so matching on the raw phrase fails against the gold itself.
    """
    return [t for t in re.sub(r"[^0-9a-z]+", " ", str(text or "").lower()).split() if t]


def score_premises(answer: str, premises: list):
    """Fraction of keyed triplets the answer asserts, in the keyed order.

    Spec: .spec/specs/r3-domain-injection/rlvr-premise-spec.md (scoring).

    Returns None when there is no key -- "not keyable, use the judge". None is
    NOT 0.0: a key we could not build is not evidence the answer is wrong
    (constitution Article X, no false endings).
    """
    if not premises:
        return None
    hay = _tokens(answer)
    if not hay:
        return 0.0
    first = {}
    for i, w in enumerate(hay):
        first.setdefault(w, i)

    positions = []
    for p in premises:
        where = []
        for slot in ("subject", "predicate", "object"):
            terms = ((p.get(slot) or {}).get("accept")) or []
            hit = -1
            for term in terms:
                toks = _tokens(term)
                # CONTENT TOKENS, not the exact phrase. spaCy's slot text is a
                # lemmatized reconstruction, not a verbatim span of the gold --
                # `generates hull of` for a gold reading "generates the convex
                # hull of". Exact matching scored 9 of 12 golds BELOW 1.0
                # against their own key (mean 0.286), i.e. it punished perfect
                # answers. Measured 2026-09-12.
                if toks and all(t in first for t in toks):
                    hit = max(first[t] for t in toks)
                    break
            where.append(hit)
        # satisfied only when every slot of the triplet is present
        # anchor at the LAST slot to appear: the point by which the answer has
        # asserted the whole triplet. Anchoring at the first slot instead lets a
        # term shared between two triplets pin both to the same early position,
        # which scored a fully REVERSED answer as perfectly ordered.
        positions.append(max(where) if all(w >= 0 for w in where) else None)

    hits = [w for w in positions if w is not None]
    if not hits:
        return 0.0
    # longest non-decreasing run: how many keyed premises the answer asserts IN
    # ORDER. One concept, no invented constant -- order costs a premise rather
    # than multiplying the whole score to zero, so an answer that states both
    # facts backwards still scores what it actually asserted.
    best = [1] * len(hits)
    for i in range(len(hits)):
        for j in range(i):
            if hits[j] <= hits[i] and best[j] + 1 > best[i]:
                best[i] = best[j] + 1
    return round(max(best) / float(len(premises)), 4)


def verify_entailment(answer: str, gold: str, threshold: float = 0.45,
                      min_ceiling: float = 0.30):
    """Does `answer` entail the gold's claims? None when no claim is extractable.

    Premise is the model's answer, hypothesis is each gold triplet: we are asking
    whether the answer SUPPORTS the gold claim, not whether the strings match.
    Scores the mean entailment over the gold's triplets; falls back to the whole
    gold as one hypothesis when extraction yields nothing usable.
    """
    a = _strip_think_blocks(answer).strip()
    if not a:
        return False
    claims = _triplets(gold) or ([gold.strip()] if gold and gold.strip() else [])
    if not claims:
        return None
    try:
        from rlv2.judge.reward import nli_entailment
    except Exception:
        return None
    scores = [nli_entailment(a, c) for c in claims]
    scores = [x for x in scores if x is not None]
    if not scores:
        return None

    # SCORE RELATIVE TO THE GOLD'S OWN CEILING, not against an absolute bar.
    # Measured over 240 golds (2026-09-12): self-entailment runs 0.007 to 0.996,
    # median 0.915 but p05 only 0.179 -- the bottom 5% of golds barely entail
    # their OWN claims because their triplets are junk. An absolute threshold
    # would mark a good answer wrong on exactly those records. So:
    #
    #   ratio = mean_entailment(answer) / mean_entailment(gold)
    #
    # and a record whose ceiling is below `min_ceiling` is NOT entailment-
    # verifiable at all -- return None so the caller falls back to the judge
    # rather than scoring against a broken yardstick.
    ceil = [nli_entailment(gold, c) for c in claims]
    ceil = [x for x in ceil if x is not None]
    ceiling = (sum(ceil) / len(ceil)) if ceil else 0.0
    if ceiling < min_ceiling:
        return None
    return ((sum(scores) / len(scores)) / ceiling) >= threshold


# ═══════════════════════════════════════════════════════════════════════════
# RLVR CODE — execute against the dataset's own asserts
#
# Spec: .spec/specs/r3-domain-injection/rlvr-math-code.md
#
# WHY NOT COMPILE. compile() passes on code that returns the wrong answer, which
# is exactly the case a reward must separate. A probe that cannot go red on a
# wrong input turns an unknown into a confident wrong signal. So compile is a
# GATE (reject before running) and execution is the SCORE.
#
# The asserts are not ours to invent: MBPP ships `test_list`, HumanEval ships a
# `check(candidate)` function. normalize_row was dropping both.
# ═══════════════════════════════════════════════════════════════════════════
CODE_TIMEOUT_S = 10
SHORT_ANSWER_MAX_WORDS = 10
CODE_PREAMBLE = "import math, cmath, re, sys, itertools, functools, collections, heapq, bisect, string"


def _extract_code(answer: str) -> str:
    """Strip think blocks and unwrap a ```python fence if the model used one."""
    text = _strip_think_blocks(str(answer or ""))
    fences = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if fences:
        return max(fences, key=len).strip()
    return text.strip()


def verify_code(answer: str, record: dict):
    """Run the candidate against the record's asserts. True / False / None.

    Returns None ONLY when the record carries no tests -- not keyable, so the
    caller falls back to the judge. An absent test is not evidence the code is
    wrong (constitution Article X), so it is never False.
    """
    import json as _json
    import subprocess
    import sys as _sys
    import tempfile

    tests = list((record or {}).get("tests") or [])
    if not tests:
        return None

    code = _extract_code(answer)
    if not code:
        return False
    try:                                   # GATE, not the score
        compile(code, "<candidate>", "exec")
    except SyntaxError:
        return False

    entry = (record or {}).get("entry_point") or ""
    # MBPP asserts call math.isclose / cmath directly and the dataset does NOT
    # ship the import. Without this preamble those records die on
    # `NameError: name 'math' is not defined` and a CORRECT gold scores False --
    # measured 2026-09-12: 10 of 421 golds, every one of them float math.
    parts = [CODE_PREAMBLE, code, (record or {}).get("test_setup") or ""]
    parts.extend(tests)
    # HumanEval ships `check(candidate)` but never calls it.
    if entry and any("def check" in t for t in tests):
        parts.append("check(%s)" % entry)
    program = "\n\n".join(p for p in parts if p)

    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(program)
            path = fh.name
        proc = subprocess.run([_sys.executable, path], capture_output=True,
                              timeout=CODE_TIMEOUT_S)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False                        # a hang is a failure, never a block
    except Exception:
        return False
    finally:
        if path:
            try:
                import os as _os
                _os.unlink(path)
            except OSError:
                pass


def _ast_nodes(code):
    """AST node count: one number for size AND branching. None if unparseable."""
    import ast
    try:
        return sum(1 for _ in ast.walk(ast.parse(code)))
    except SyntaxError:
        return None


def code_simplicity(answer: str, gold: str) -> float:
    """1.0 when the candidate is no bigger than the reference, decaying after.

    Spec: .spec/specs/r3-domain-injection/rlvr-math-code.md (complexity).

    Measured against the dataset's OWN canonical solution, never against
    "shortest possible", and CAPPED AT 1.0. That cap is the whole design: with
    no cap, the optimum is code golf -- nested one-liners with the error
    handling stripped out -- because shrinking below the reference would keep
    paying. Capped, matching a human-written solution already earns full credit
    and there is nothing left to win by compressing further.

    Returns 1.0 (no signal, no penalty) when either side will not parse.
    """
    cand = _ast_nodes(_extract_code(answer))
    ref = _ast_nodes(str(gold or ""))
    if not cand or not ref:
        return 1.0
    return round(min(1.0, float(ref) / float(cand)), 4)


def score_code(answer: str, record: dict):
    """Correctness gate, then simplicity as a tie-breaker. None = not keyable.

    reward = 0.0                      when any assert fails
           = 0.8 + 0.2 * simplicity   when every assert passes

    Correctness is a GATE, not a term: elegance can never buy back a wrong
    answer, or the model learns to write beautiful broken code. Simplicity is
    capped at 20% of the reward so it breaks ties without driving the gradient.
    """
    ok = verify_code(answer, record)
    if ok is None:
        return None
    if not ok:
        return 0.0
    return round(0.8 + 0.2 * code_simplicity(answer, (record or {}).get("answer", "")), 4)


def math_conciseness(answer: str, gold: str) -> float:
    """1.0 when the answer is no longer than the reference solution, decaying after.

    Spec: .spec/specs/r3-domain-injection/rlvr-math-code.md (quality).

    Same shape as code_simplicity and for the same reason: the dataset gives ONE
    worked solution per record, and a ratio capped at 1.0 turns that single
    reference into a quality gradient -- no graded answer pairs, no preference
    data. Matching the reference already earns full credit, so there is nothing
    to win by truncating further.

    WEAKER THAN THE CODE AXIS, stated plainly: token count is a rambling filter,
    not a measure of whether a derivation is elegant. The correctness gate is
    what keeps it safe -- an answer that skips its reasoning to be short gets the
    number wrong and scores 0.
    """
    cand = len(_tokens(_strip_think_blocks(str(answer or ""))))
    ref = len(_tokens(str(gold or "")))
    if not cand or not ref:
        return 1.0
    return round(min(1.0, float(ref) / float(cand)), 4)


def score_math(answer: str, gold: str):
    """Correctness gate, then conciseness as a tie-breaker. None = not keyable."""
    ok = verify_math(answer, gold)
    if ok is None:
        return None
    if not ok:
        return 0.0
    return round(0.8 + 0.2 * math_conciseness(answer, gold), 4)


def score_rlvr(answer: str, gold: str, record: dict = None):
    """The reward: 0.0-1.0 from a verifiable check, or None to use the judge.

    Spec: .spec/specs/r3-domain-injection/rlvr-math-code.md
    Returns (reward, method) so a run reports what fraction it actually verified
    rather than assuming.
    """
    if record and (record.get("tests") or []):
        r = score_code(answer, record)
        if r is not None:
            return r, "code"
    r = score_math(answer, gold)
    if r is not None:
        return r, "numeric"
    return None, "judge"


def verify_short_answer(answer: str, gold: str):
    """Short-answer exact match for STEM. True / False / None.

    Spec: .spec/lessons_learned.md section 4 (verifier policy by domain).

    STEM golds are 2-word spans (sciq/arc/openbookqa/mmlu): 'deductive',
    'air pollution'. Containment on content tokens, so "the answer is deductive"
    matches gold "deductive" without the model being forced onto one phrasing.

    Returns None when the gold is too long to be an answer KEY rather than an
    explanation -- measured: STEM median 2 words / p90 8, vs R3 median 28 where
    token matching scored AUC 0.580 (coin flip). Above the cutoff this instrument
    does not work, and None sends the record to the judge instead of guessing.
    """
    g = _tokens(gold)
    if not g or len(g) > SHORT_ANSWER_MAX_WORDS:
        return None
    a = _tokens(_strip_think_blocks(str(answer or "")))
    if not a:
        return False
    # contiguous span match: the gold's tokens must appear in order, adjacent
    return any(a[i:i + len(g)] == g for i in range(len(a) - len(g) + 1))


def rlvr_score(answer: str, gold: str, record: dict = None):
    """Tiered verifiable reward. Returns (verdict, method).

    verdict is True/False, or None when nothing verifiable applies and the caller
    must fall back to the LLM judge. method names which tier decided, so a run can
    report what fraction it actually verified rather than assuming.
    """
    # ROUTE BY WHAT THE RECORD DECLARES, never by sniffing the gold's shape.
    # Length alone cannot tell a 2-word STEM key from a 2-word prose answer, and
    # guessing wrong silently scores an unverifiable record as WRONG. The pools
    # carry `verifier` for exactly this.
    declared = (record or {}).get("verifier")
    if declared == "judge":
        return None, "judge"
    if declared == "code" or (record or {}).get("tests"):
        v = verify_code(answer, record)
        return (v, "code") if v is not None else (None, "judge")
    if declared == "short_answer":
        v = verify_short_answer(answer, gold)
        return (v, "short_answer") if v is not None else (None, "judge")

    v = verify_math(answer, gold)
    if v is not None:
        return v, "numeric"
    return None, "judge"
