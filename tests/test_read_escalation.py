"""Extra judge reads are spent only where the reads disagree about the floor.

NO GOVERNING SPEC. Basis: operator 2026-09-16 -- "Might as well implement the escalate on
split as well (for new records)".

WHY NOT JUST RAISE EVERY QUESTION TO 5 READS. The judge only flips where an answer sits
near the 0.70 floor: measured the same day, the same checkpoint read 6/16 then 5/16 on ONE
such question while the other fifteen were unanimous. Extra reads on a unanimous question
buy nothing -- the verdict was never in doubt.

    every question at 5 reads    16 * 5           = 80 calls
    escalate ~20% of questions   16 * 3 + 3 * 2   = 54 calls
"""
import sys


FLOOR = 0.70


def _esc(reads):
    from rlv2.eval import score_arms as S
    return S._should_escalate(reads, FLOOR)


def test_unanimously_correct_does_not_escalate():
    assert _esc([0.95, 0.88, 0.75]) is False


def test_unanimously_incorrect_does_not_escalate():
    assert _esc([0.10, 0.42, 0.69]) is False


def test_a_split_across_the_floor_escalates():
    """Two say correct, one says not -- the verdict turns on one read."""
    assert _esc([0.72, 0.91, 0.55]) is True


def test_the_boundary_is_the_floor_itself_not_near_it():
    """0.70 is correct; 0.699 is not. A cluster just above the line is unanimous."""
    assert _esc([0.70, 0.71, 0.99]) is False
    assert _esc([0.70, 0.69]) is True


def test_a_failed_read_is_missing_not_a_vote():
    """An unparseable reply must not trigger an escalation by itself -- it is an outage,
    not a disagreement."""
    assert _esc([0.95, 0.91, None]) is False
    assert _esc([0.95, None, None]) is False
    assert _esc([None, None, None]) is False


def test_a_split_still_escalates_when_one_read_failed():
    assert _esc([0.95, 0.20, None]) is True


def test_the_reader_uses_the_predicate():
    import io
    src = io.open("src/rlv2/eval/score_arms.py", encoding="utf-8").read()
    assert "if _should_escalate(reads, FLOOR):" in src
    assert "reads.extend(_one() for _ in range(ESCALATION_READS))" in src


# ------------------------------------------------- one rule, both scorers ----
def test_the_rule_has_exactly_one_definition():
    """Operator 2026-09-16: "This sounds like a bug of change we still haven't resolved."
    It was. Escalation was added to score_arms and not to the trainer's eval, so the same
    checkpoint read 6/16 in one and 5/16 in the other -- a question beside the floor
    resolved by five reads in one path and three in the other. The rule now lives in
    judge.py and both callers import it."""
    import io

    from rlv2.judge import judge as J
    from rlv2.eval import score_arms as S
    assert S._should_escalate is J.should_escalate, "score_arms redefined the rule"
    gof = io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    assert "from rlv2.judge.judge import should_escalate as _should_escalate" in gof, \
        "the trainer's eval does not use the shared rule"


def test_the_trainer_takes_the_majority_over_the_READS_IT_TOOK():
    """An escalated question has 5 reads and an unescalated one has 3, so a fixed
    denominator would mis-count exactly the questions escalation exists to fix."""
    import io
    gof = io.open("docs/grpo_one_file.py", encoding="utf-8").read()
    assert "sum(1 for x in _v if x >= floor) * 2 > len(_v)" in gof
    assert "votes * 2 > n_reads" not in gof, "fixed denominator is back"


def test_the_default_floor_is_the_judges_own():
    from rlv2.judge import judge as J
    assert J.should_escalate([J.CORRECT_FLOOR, J.CORRECT_FLOOR - 0.01]) is True
    assert J.should_escalate([J.CORRECT_FLOOR, J.CORRECT_FLOOR + 0.01]) is False
