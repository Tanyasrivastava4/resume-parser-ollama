"""
tests/test_milestone_v6_1_evaluate_pilot.py

Targeted tests for the new pure-function evaluation helper in
scripts/_milestone_v6_1_evaluate_pilot.py (Milestone V6.1 Phase 2).
`evaluate()` itself (the fuzzy/normalized metric) is REUSED UNCHANGED
from scripts/_milestone_v5_8_offline_evaluation.py, per the frozen
V6.0 protocol, and is not re-tested here (it is not new code). Only
`exact_match_score`, written new for this milestone, is tested.

No LLM calls.
"""
from scripts._milestone_v6_1_evaluate_pilot import exact_match_score


class TestExactMatchScore:
    def test_perfect_match(self):
        r = exact_match_score(["Python", "SQL"], ["Python", "SQL"])
        assert r["exact_recall"] == 1.0
        assert r["exact_precision"] == 1.0
        assert r["exact_matched"] == 2

    def test_partial_recall(self):
        r = exact_match_score(["Python"], ["Python", "SQL"])
        assert r["exact_recall"] == 0.5
        assert r["exact_precision"] == 1.0

    def test_case_and_whitespace_sensitive_like_production_comparator(self):
        # Matches validators/json_comparator.py's own exact-match
        # philosophy: "python" != "Python", no normalization.
        r = exact_match_score(["python"], ["Python"])
        assert r["exact_recall"] == 0.0

    def test_extra_items_reduce_precision_not_recall(self):
        r = exact_match_score(["Python", "SQL", "Hallucinated"], ["Python", "SQL"])
        assert r["exact_recall"] == 1.0
        assert r["exact_precision"] == round(2 / 3, 4)

    def test_empty_extraction(self):
        r = exact_match_score([], ["Python"])
        assert r["exact_recall"] == 0.0
        assert r["exact_precision"] is None  # undefined, not zero

    def test_empty_gt_returns_none(self):
        r = exact_match_score(["Python"], [])
        assert r["exact_recall"] is None
        assert r["exact_precision"] is None
