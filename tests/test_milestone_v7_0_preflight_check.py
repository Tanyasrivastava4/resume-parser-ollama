"""
tests/test_milestone_v7_0_preflight_check.py

Milestone V7.0 -- focused tests for the RESEARCH-ONLY, zero-LLM-call
pre-flight script (scripts/_milestone_v7_0_preflight_check.py) that
gates the future, not-yet-authorized regression-corpus-extension
execution. These tests exercise the deterministic logic only -- no
network/subprocess calls are made real in the test environment beyond
what the script itself already restricts to read-only metadata queries
(`ollama list`), consistent with the project's own established
convention for testing V6.0's equivalent script.
"""
from __future__ import annotations

import pytest

from pathlib import Path

from scripts._milestone_v7_0_preflight_check import (
    CONDITIONS,
    HARD_CALL_CAP,
    OPTIONAL_RESUMES,
    PLANNED_CALLS_REQUIRED_ONLY,
    PLANNED_CALLS_WITH_OPTIONAL,
    REQUIRED_RESUMES,
    RUNS_PER_CONDITION,
    check_call_budget,
    check_output_directory_safety,
    check_source_artifacts_available,
    sha256_of,
)


class TestFrozenPopulationShape:
    def test_required_resumes_are_exactly_016_and_005(self):
        assert set(REQUIRED_RESUMES.keys()) == {"016", "005"}

    def test_optional_resumes_are_exactly_022(self):
        assert set(OPTIONAL_RESUMES.keys()) == {"022"}

    def test_005_uses_its_own_v2_4h_benchmark_lineage(self):
        assert REQUIRED_RESUMES["005"]["lineage"] == "output/v2_4h_benchmark/stage1/markdown"

    def test_016_and_022_use_output_markdown_lineage(self):
        assert REQUIRED_RESUMES["016"]["lineage"] == "output/markdown"
        assert OPTIONAL_RESUMES["022"]["lineage"] == "output/markdown"


class TestCallBudgetArithmetic:
    def test_planned_calls_required_only_is_20(self):
        assert PLANNED_CALLS_REQUIRED_ONLY == len(REQUIRED_RESUMES) * CONDITIONS * RUNS_PER_CONDITION
        assert PLANNED_CALLS_REQUIRED_ONLY == 20

    def test_planned_calls_with_optional_is_30(self):
        assert PLANNED_CALLS_WITH_OPTIONAL == 30

    def test_hard_cap_exceeds_planned_with_optional(self):
        assert HARD_CALL_CAP >= PLANNED_CALLS_WITH_OPTIONAL

    def test_check_call_budget_passes_under_frozen_shape(self):
        assert check_call_budget() is True


@pytest.mark.local_data('any_corpus_markdown')
class TestSourceArtifactAvailability:
    def test_all_frozen_population_source_files_exist(self):
        # Real check against the actual repository state -- if this
        # fails, the frozen design's own source artifacts have moved.
        assert check_source_artifacts_available() is True


class TestOutputDirectorySafety:
    def test_output_directory_check_runs_without_error(self):
        assert check_output_directory_safety() is True


class TestSha256Helper:
    def test_sha256_of_is_deterministic(self):
        p = Path("settings.py")
        assert sha256_of(p) == sha256_of(p)

    def test_sha256_of_differs_for_different_files(self):
        assert sha256_of(Path("settings.py")) != sha256_of(Path("config.py"))
