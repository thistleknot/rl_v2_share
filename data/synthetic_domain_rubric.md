<!-- Spec: .spec/specs/r3-domain-injection/judge-rubric.md — REQ-J4, J5, J6, J7, J10, J11, J15
     Task: playbook.md R-1 (rubric integrity) · .spec/specs/r3-domain-injection/tasks.md #29
     REQ-J15 (verifier-style per-category correctness) added 2026-08-31 from
     playbook-replaybuffer-scores.md §3/§4/§5/§6, operator instruction. -->

# Judge rubric — R3 closed-book domain QA (decision trees & ensembles)

<!-- LANE: synthetic domain R3. ONE OF TWO RUBRICS, BY DESIGN (REQ-J10).
     The other is `reward._VIBETHINKER_RUBRIC`, inline in src/reward.py, serving the
     VibeThinker lane: 6 domains, context-dependent, abstention a first-class verdict scoring
     >0.9 on a true unanswerable. THIS file is closed-book: 4 categories, no context, NO
     abstention. They are NOT duplicates and must not be merged — A rewards abstention, B
     forbids it.

     EDITING THIS FILE CHANGES ONLY THE R3 LANE. Do not carry an edit across to
     reward._VIBETHINKER_RUBRIC; that happened 2026-08-29 and silently altered the RL reward.

     EXEMPLAR PROVENANCE (REQ-J11). Every worked example below is drawn from the 12-item
     RUBRIC SPLIT and from nowhere else, and each one carries its item id in brackets so the
     claim is checkable rather than asserted. `calibrate_rubric.check_exemplar_provenance`
     re-derives those ids from this text and RAISES if any of them is sitting in r3_cpt,
     r3_sft, r3_rl or the frozen eval. That guard exists because the 2026-08-29 repartition
     silently moved the old exemplars into the training splits and nothing re-checked. -->

Every question here IS answerable from knowledge. There is no abstention and no context to
be missing. An answer either lands the gold's claim or it does not.

STEP 1 - CORRECTNESS. Gold is one verified-correct reference, not a target to copy
and not a ceiling. Different wording is fine; extra correct detail is fine.
  correct   - contains the gold's essential claim, contradicts nothing in it
  incorrect - anything else

Decide correctness the way a task-specific VERIFIER would, by the question's category:

  Math        - check the FINAL ANSWER: the number, ratio, or comparison must match the
                gold's. A clean method with a wrong result is incorrect; method prose never
                rescues a wrong final answer.
  Code        - check what the described code actually COMPUTES or does: behaviour, output,
                parameter semantics must match the gold's. Judge as if you executed it;
                plausible-sounding API talk with the wrong behaviour is incorrect.
  STEM        - check the NAMED FACT or mechanism against the gold's, as answer matching:
                the right concept under the wrong name, or the wrong concept fluently
                explained, is incorrect.
  Instruction - check EVERY explicit part of the ask (a computation AND a verification, an
                ordering, a count). Satisfying some parts is incorrect; constraint
                satisfaction is all-or-nothing.

If the gold makes more than one essential claim, correct requires ALL of them. An answer
that lands one claim of two is incorrect — score it in the wrong bands below, usually
0.4-0.6 if the claim it does make is committed and false, 0.3-0.4 if it simply omits.

If the QUESTION itself cannot be assessed — incomplete, self-contradictory, or with no
determinable correct answer — still return the JSON, but start `why` with "INVALID QUERY:".
That flags a data defect upstream; never silently score an unassessable item as if it
were sound.

STEP 2 - SCORE. Where a wrong answer lands depends on HOW wrong it is.

  0.0        "I don't know", empty, degenerate, or completely off-topic.
             Not plausible as an answer to this question at all.

             ALSO 0.0: A RESPONSE THAT NEVER STATES AN ANSWER. Planning,
             restating the question, listing what it is about to do, or
             reasoning that runs out of room before committing - "Thinking
             Process: 1. Analyze the Request..." with no conclusion - is NOT an
             answer, no matter how on-topic it reads or how much of the right
             vocabulary it contains. Ask: can you point to the sentence that
             answers the question? If not, it is a 0.0, not a 0.4.

             This band exists because of a measured failure (2026-09-11): all
             sixteen eval responses were reasoning preambles cut off by the
             token budget, and the judge scored every one of them at or above
             0.70. On-topic analysis is not a claim, and only a claim can be
             correct.
  0.1-0.3    Off-topic-ish. Answers something related to the domain, but not the
             question that was asked.
  0.4-0.6    PLAUSIBLE AND WRONG. On-topic, coherent, offers supporting context,
             reads like a real answer - and the claim is wrong. A confident wrong
             number, an invented parameter name, or a flipped yes/no lands here.
             This is the MOST COMMON wrong answer. It is NOT a 0.
  0.7-0.9    Correct, but verbose, padded, or indirect.
  0.9-1.0    Correct, precise, concise. May beat gold.

0.7 is the line: at or above is correct, below is wrong.
Never round to 0/0.5/1 - interpolate.
SHORTER IS BETTER among correct answers. Two answers that make the same true claim are not
equal: the one that gets there without preamble scores higher. Padding is not a tie-break in
the answer's favour, it is what separates 0.95 from 0.72. Conciseness never crosses the 0.7
line though - a terse wrong answer is still wrong, and a wordy right answer is still right.
Fluency is not correctness: a beautifully written wrong answer is still 0.4-0.6.
An answer that ATTEMPTS the question and misses is wrong, never "no attempt".

THE GOLD BAND IS 0.9-1.0, NOT 1.0 (REQ-J4, REQ-J5). Golds are verified correct but they
are not all equally tight — several are wordy, and one is a bare sentence. Scoring every
gold 1.000 means you are not measuring anything: the spread across golds is what proves the
instrument still has resolution above "correct". Reserve 1.0 for an answer that is correct,
complete AND could not be said more briefly.

CALIBRATION - every wrong answer below is fluent and on-topic. Exemplars are held-out
rubric-split items, never training or eval items.

EVERY CORRECT ANSWER BELOW SCORES DIFFERENTLY. That is the point of this block, not an
accident of the examples chosen. They are separated by how tightly they answer, and the
score moves across 0.72-0.98 accordingly. If you find yourself giving the same number to two
correct answers of visibly different quality, you are not using the scale.

BEING VERBOSE IS NOT THE SAME AS BEING VAGUE. A low-band correct answer is one that STATES
THE GOLD'S CLAIM and takes too long doing it. An answer that never commits to the claim -
that agrees, defers, restates the question, or says the approach is standard - is INCORRECT
no matter how confident it sounds. Length moves an answer within its band; commitment is
what decides which band it is in. The last example below exists only to make that line
visible, because widening the score range must not widen what counts as correct.

  Math [r3-0313] "In AdaBoost, how do the weight ratios w'/w compare for correctly vs
                  incorrectly classified samples?"
    0.98 correct "correct: w'/w < 1; incorrect: w'/w > 1"
                 exact, complete, could not be shorter -- the top of the band
    0.50 wrong   "both ratios exceed 1; every weight grows each round"     (flipped)
    0.25 wrong   "the weights are renormalised to sum to 1"                (true, not asked)

  Code [r3-0847] "With BaggingRegressor over DecisionTreeRegressor bases, which statistical
                  property does averaging primarily reduce?"
    0.72 correct "Averaging is a technique used across many ensemble methods, and in this
                  particular setting what it is acting on, statistically speaking, is the
                  variance of the individual base learners."
                 CORRECT but padded and indirect -- it arrives at variance through a
                 preamble. Right answer, low band.
    0.95 correct "variance - averaging high-variance trees"
                 same claim, said directly
    0.45 wrong   "bias, since averaging corrects each tree's underfitting" (wrong term)

  STEM [r3-0346] "How does a new weak learner in a boosting ensemble differ from its
                  predecessors?"
    0.85 correct "it is trained on what the current ensemble still gets wrong"
                 CORRECT and concise, but names the idea rather than the mechanism --
                 middle of the band, not the top
    0.50 wrong   "it fits a bootstrap resample of the original target"     (that is bagging)
    0.15 wrong   "each learner is deeper than the last"                    (off-topic-ish)

  Instruction [r3-0152] "A branch has 6 of class A and 2 of class B. Gini impurity, and
                         does 2*p_A*p_B agree?"
    0.92 correct "0.375 by 1-sum(p^2); 2*(3/4)*(1/4)=0.375, agrees"
                 complete and verified both ways, marginally wordy
    0.55 wrong   "0.25, and the two-class formula confirms it", clean method (arithmetic wrong)
    0.25 wrong   "Gini is the chance of misclassifying a random sample"    (true, not asked)

THE WRONG SIDE HAS THREE BANDS AND YOU MUST USE ALL OF THEM. 0.4-0.6 is for an answer that
COMMITS to a specific claim and the claim is false — a named-but-wrong mechanism, a confident
wrong number, a flipped comparison. It is the most demanding kind of wrong, and it is the
only kind that earns 0.4-0.6.

An answer that commits to nothing has not earned that band:

  0.30-0.40  answers a real but DIFFERENT question about the domain
  0.10-0.25  on-topic noise: names concepts without making a claim about this question
  0.00-0.10  content-free: agrees, defers, restates the question, calls the approach
             standard, or is empty. It asserted nothing, so there is nothing to be wrong
             about, and it must not score alongside an answer that actually tried.

Worked, on the bagging-variance question above:
    0.45  "bias, since averaging corrects each tree's underfitting"
          committed to a named mechanism; the mechanism is wrong
    0.32  "it reduces total training time, since the trees fit in parallel"
          a true statement about bagging, answering a different question
    0.18  "ensembles combine multiple estimators to improve performance"
          on-topic, asserts nothing about variance
    0.05  "Yes, that is the standard approach and nothing here prevents it"
          content-free deferral

The five correct answers above score 0.98, 0.95, 0.92, 0.85 and 0.72. The wrong ones score
0.55, 0.50, 0.50, 0.45, 0.32, 0.25, 0.25, 0.18, 0.15 and 0.05. Neither list is one repeated
number, and yours should not be either. If every wrong answer you score lands near 0.45, you
are treating "wrong" as one category when it is three.

SCORE THEM AGAINST EACH OTHER, NOT ONE AT A TIME.

You will usually be shown SEVERAL answers together. That is deliberate, and it is how this
scale is meant to be applied. Shown one answer alone the only question you can really answer
is "is this correct?" — and if it is, it gets the same number every time. Measured three
separate times: twelve verified-correct answers scored 0.980, every one, spread 0.0000.

Shown them together you can see that one states the claim in a single clause while another
takes three sentences to reach it. That is the difference the 0.7-1.0 band exists to record.

  - Rank the answers against each other WITHIN whichever side of 0.70 they fall on.
  - The tightest correct answer scores highest; the most content-free wrong answer lowest.
  - Use DISTINCT values where answers genuinely differ. Give two answers the same number only
    when they are genuinely equivalent.
  - If you find yourself giving many answers one number, you are not ranking them.
  - Ranking NEVER moves an answer across 0.70. Which side it lands on is decided by whether
    it states the gold's claim, and nothing else. Order sorts within a side; it does not
    promote or demote across the line.

The answers come from different models and are shown in random order. Nothing about their
order tells you anything about their quality.

Return: {"verdict": "correct|incorrect", "score": 0.0-1.0, "why": "one clause"}
