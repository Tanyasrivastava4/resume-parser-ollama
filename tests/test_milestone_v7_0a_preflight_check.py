"""
tests/test_milestone_v7_0a_preflight_check.py

Milestone V7.0A -- focused tests for the RESEARCH-ONLY, zero-LLM-call
pre-flight script that supersedes V7.0's own script after protocol
closure (022 removed; source-file hash-lock now has a real comparison
target).
"""
from __future__ import annotations

import pytest

from pathlib import Path

from scripts._milestone_v7_0a_preflight_check import (
    CONDITIONS,
    HARD_CALL_CAP,
    PLANNED_CALLS,
    REQUIRED_RESUMES,
    RUNS_PER_CONDITION,
    check_call_budget,
    check_output_directory_safety,
    check_source_artifacts_hash_locked,
    sha256_of,
)


class TestFinalFrozenPopulation:
    def test_required_resumes_are_exactly_016_and_005(self):
        assert set(REQUIRED_RESUMES.keys()) == {"016", "005"}

    def test_022_is_not_in_the_final_population(self):
        assert "022" not in REQUIRED_RESUMES

    def test_005_still_uses_its_own_v2_4h_benchmark_lineage(self):
        assert REQUIRED_RESUMES["005"]["lineage"] == "output/v2_4h_benchmark/stage1/markdown"


class TestCallBudgetArithmetic:
    def test_planned_calls_is_20(self):
        assert PLANNED_CALLS == len(REQUIRED_RESUMES) * CONDITIONS * RUNS_PER_CONDITION
        assert PLANNED_CALLS == 20

    def test_planned_calls_dropped_from_v7_0s_30(self):
        # V7.0's own planned-with-optional figure was 30; V7.0A's
        # closure must show a real, disclosed reduction, not silently
        # keep the old figure.
        assert PLANNED_CALLS < 30

    def test_hard_cap_exceeds_planned(self):
        assert HARD_CALL_CAP >= PLANNED_CALLS

    def test_check_call_budget_passes(self):
        assert check_call_budget() is True


@pytest.mark.local_data('any_corpus_markdown')
class TestHashLockRealComparison:
    def test_hash_lock_check_passes_against_real_repository_state(self):
        # Real check: the frozen hash file must exist and match the
        # actual current on-disk files -- this is what closes the gap
        # V7.0's own script had (a hash with nothing to compare against).
        assert check_source_artifacts_hash_locked() is True

    def test_frozen_hash_file_has_a_real_comparison_target(self):
        import csv
        p = Path("benchmark/baselines/v7_0a_frozen_source_file_hashes.csv")
        assert p.exists()
        with p.open() as f:
            rows = list(csv.DictReader(f))
        assert {r["resume_id"] for r in rows} == {"016", "005"}
        for r in rows:
            assert len(r["sha256"]) == 64  # a real sha256 hex digest, not a placeholder


class TestOutputDirectorySafety:
    def test_output_directory_check_runs_without_error(self):
        assert check_output_directory_safety() is True


class TestSha256Helper:
    def test_sha256_of_is_deterministic(self):
        p = Path("settings.py")
        assert sha256_of(p) == sha256_of(p)
