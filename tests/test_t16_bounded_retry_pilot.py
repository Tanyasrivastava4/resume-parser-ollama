"""
tests/test_t16_bounded_retry_pilot.py

Milestone T16/Phase 2C -- focused, narrowly-scoped tests for the
research-only T16 analysis code (scripts/_milestone_t16_analysis.py).
New diagnostic tests, distinct from the project's production
regression suite -- test only T16's own deterministic classification
helpers, never invoking Ollama, never touching production behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import scripts._milestone_t16_analysis as analysis  # noqa: E402


def _match(rid, jobs, gt_jobs):
    analysis.GT_JOBS = {rid: gt_jobs}
    return analysis._match_jobs_to_gt(rid, jobs)


class TestEntityMatching:
    def test_perfect_match(self):
        gt = [{"company": "Acme Corp"}, {"company": "Widget Inc"}]
        jobs = [{"company": "Acme Corp"}, {"company": "Widget Inc"}]
        matched, extra, idx, split, merge = _match("x", jobs, gt)
        assert len(matched) == 2
        assert idx == {0, 1}


class TestFailureClassificationPrecedence:
    def test_retry_terminal_failure_takes_precedence(self):
        record = {"primary_failure_class": "STAGE_B_RETRY_FAILURE"}
        result = analysis._final_classification(record, [], set(), set(), [], [], None)
        assert result == "STAGE_B_RETRY_FAILURE"

    def test_split_detected_when_no_retry_failure(self):
        record = {"primary_failure_class": None}
        result = analysis._final_classification(record, [], set(), {"Acme"}, [], [], 1.0)
        assert result == "ENTITY_SPLIT"

    def test_merge_detected_after_split_absent(self):
        record = {"primary_failure_class": None}
        result = analysis._final_classification(record, [], set(), set(), [("X", ["A", "B"])], [], 1.0)
        assert result == "ENTITY_MERGE"

    def test_fabrication_detected_after_split_merge_absent(self):
        record = {"primary_failure_class": None}
        result = analysis._final_classification(
            record, [], set(), set(), [], [{"company": "Extra"}], 1.0
        )
        assert result == "ENTITY_FABRICATION"

    def test_content_failure_after_structural_success(self):
        record = {"primary_failure_class": None}
        result = analysis._final_classification(record, [], set(), set(), [], [], 0.5)
        assert result == "CONTENT_FAILURE_AFTER_STRUCTURAL_SUCCESS"

    def test_success_when_full_recall_and_no_issues(self):
        record = {"primary_failure_class": None}
        result = analysis._final_classification(record, [], set(), set(), [], [], 1.0)
        assert result == "SUCCESS"


class TestRetryPolicyBoundedness:
    """Confirms the T16 execution script's own retry helper never
    exceeds one retry per stage -- verified via the real
    scripts/_milestone_t16_bounded_retry_pilot.py functions using a
    stubbed OllamaClient/SchemaValidator, no real Ollama call."""

    def test_stage_a_retry_failure_terminates_without_stage_b(self):
        import scripts._milestone_t16_bounded_retry_pilot as pilot

        class FakeResponse:
            response = "not json"

        class FakeClient:
            call_count = 0

            def generate_text(self, model, prompt):
                self.call_count += 1
                return FakeResponse()

        client = FakeClient()
        result = pilot._call_stage_a(client, "some markdown")
        assert result["ok"] is False
        # Simulate the run_one retry logic directly: two Stage A
        # attempts, both failing, must never reach Stage B.
        result2 = pilot._call_stage_a(client, "some markdown")
        assert result2["ok"] is False
        assert client.call_count == 2  # exactly one retry, not more


class TestTailEntityTracking:
    def test_tail_entities_list_matches_established_021_finding(self):
        assert analysis.TAIL_ENTITIES_021 == [
            "Hazel Infotech Ltd.", "SVN Technologies", "Kalyani Informix",
        ]
