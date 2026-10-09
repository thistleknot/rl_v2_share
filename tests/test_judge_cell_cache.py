"""A cached judgement is valid ONLY for the exact answer text that was judged.

NO GOVERNING SPEC. Basis: 2026-09-15. After the think-suppression fix regenerated every
answer, a re-score printed "resuming: 32 cells already judged", made zero judge calls,
and reported the OLD verdict on blank answers as the repaired model's score (0/16). The
sidecar key was (arm, qid, rep) -- it did not record WHAT was judged.
"""
import sys



def test_hash_separates_different_answers():
    from rlv2.eval import score_arms as sa
    assert sa._answer_hash("the ensemble averages the trees") != sa._answer_hash("")


def test_hash_ignores_surrounding_whitespace_only():
    from rlv2.eval import score_arms as sa
    assert sa._answer_hash("  x\n") == sa._answer_hash("x")


def test_blank_and_missing_answer_share_one_identity():
    from rlv2.eval import score_arms as sa
    assert sa._answer_hash(None) == sa._answer_hash("")


def test_resume_key_carries_the_answer_hash():
    """The sidecar writer and the resume reader must agree on a 4-part key."""
    import io
    src = io.open("src/rlv2/eval/score_arms.py", encoding="utf-8").read()
    assert 'done[(c["arm"], c["qid"], c["rep"], c.get("ans"))]' in src, \
        "resume reader dropped the answer hash from the key"
    assert '(name, qid, rep, _ah[name]) not in done' in src, \
        "cache-hit check dropped the answer hash"
    assert '"ans": _ah[name]' in src, "sidecar writer stopped recording the answer hash"


def test_a_stale_cell_does_not_satisfy_a_new_answer():
    """Simulate the exact 2026-09-15 defect on the key logic alone."""
    from rlv2.eval import score_arms as sa
    done = {("sft", "q1", 0, sa._answer_hash("")): [0.0, 0.0, 0.0]}
    fresh = sa._answer_hash("A random forest averages many decorrelated trees.")
    assert ("sft", "q1", 0, fresh) not in done, \
        "regenerated answer inherited the blank answer's verdict"
