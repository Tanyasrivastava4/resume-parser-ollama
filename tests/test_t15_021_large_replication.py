"""
tests/test_t15_021_large_replication.py

Milestone T15/Phase 2C -- focused, narrowly-scoped tests for the
research-only T15 analysis code (scripts/_milestone_t15_analysis.py).
New diagnostic tests, distinct from the project's production
regression suite -- test only T15's own deterministic entity-matching/
classification helpers, never invoking Ollama, never touching
production behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import scripts._milestone_t15_analysis as analysis  # noqa: E402


def _match(rid, jobs, gt_jobs):
    analysis.GT_JOBS = {rid: gt_jobs}
    return analysis._match_jobs_to_gt(rid, jobs)


class TestEntityMatching:
    def test_perfect_match(self):
        gt = [{"company": "Acme Corp"}, {"company": "Widget Inc"}]
        jobs = [{"company": "Acme Corp"}, {"company": "Widget Inc"}]
        matched, extra, missing, split, merge, idx = _match("x", jobs, gt)
        assert len(matched) == 2
        assert idx == {0, 1}

    def test_missing_entity_tracked_by_index(self):
        gt = [{"company": "Acme Corp"}, {"company": "Old Co"}]
        jobs = [{"company": "Acme Corp"}]
        matched, extra, missing, split, merge, idx = _match("x", jobs, gt)
        assert idx == {0}
        assert len(missing) == 1
        assert missing[0]["company"] == "Old Co"


class TestRecallCalculation:
    def test_recall_is_matched_over_gt_count(self):
        gt = [{"company": "A"}, {"company": "B"}, {"company": "C"}, {"company": "D"}]
        jobs = [{"company": "A"}, {"company": "B"}]
        matched, extra, missing, split, merge, idx = _match("x", jobs, gt)
        recall = len(matched) / len(gt)
        assert recall == 0.5


class TestPerEntityTracking:
    def test_matched_indices_identify_which_gt_entities_recovered(self):
        gt = [{"company": "A"}, {"company": "B"}, {"company": "C"}]
        jobs = [{"company": "C"}, {"company": "A"}]  # out-of-order, B missing
        matched, extra, missing, split, merge, idx = _match("x", jobs, gt)
        assert idx == {0, 2}  # A and C matched, B (index 1) missing


class TestTailEntityClassification:
    def test_all_or_nothing_pattern_detectable_from_indices(self):
        gt = [{"company": f"Job{i}"} for i in range(8)]
        # Simulates a T15-observed "all tail entities recovered together" run.
        jobs = [{"company": f"Job{i}"} for i in range(8)]
        matched, extra, missing, split, merge, idx = _match("x", jobs, gt)
        tail_indices = {5, 6, 7}
        assert tail_indices.issubset(idx)


class TestStageAFailureClassification:
    def test_stage_a_parse_failure_classified_correctly(self):
        d = {"status": "stage_a_parse_failed"}
        result = analysis._classify_run(d, [], set(), [], [], None)
        assert result == "stage_a_structural_failure"

    def test_stage_b_parse_failure_classified_correctly(self):
        d = {"status": "stage_b_parse_failed"}
        result = analysis._classify_run(d, [], set(), [], [], None)
        assert result == "stage_b_structural_failure"

    def test_schema_failure_classified_correctly(self):
        d = {"status": "schema_failed"}
        result = analysis._classify_run(d, [{"company": "A"}], set(), [], [], 1.0)
        assert result == "schema_failure"

    def test_split_takes_precedence_over_fabrication(self):
        d = {"status": "ok"}
        result = analysis._classify_run(
            d, [{"company": "A"}], {"A"}, [], [{"company": "Extra"}], 1.0
        )
        assert result == "entity_split"

    def test_successful_complete_structure(self):
        d = {"status": "ok"}
        result = analysis._classify_run(d, [{"company": "A"}], set(), [], [], 1.0)
        assert result == "successful_complete_entity_structure"

    def test_partial_recovery(self):
        d = {"status": "ok"}
        result = analysis._classify_run(d, [{"company": "A"}], set(), [], [], 0.5)
        assert result == "partial_entity_recovery"


class TestStageBFixationReused:
    """T15 reuses T13's own, already-fixed check_entity_fixation_violation
    unmodified -- confirmed importable and behaviorally correct here."""

    def test_no_violation_for_identical_lists(self):
        from scripts._milestone_t13_entity_first_pilot import check_entity_fixation_violation
        stage_a = [{"ordinal": 1, "company": "Acme Corp"}]
        returned = [{"company": "Acme Corp"}]
        assert check_entity_fixation_violation(stage_a, returned) is False
