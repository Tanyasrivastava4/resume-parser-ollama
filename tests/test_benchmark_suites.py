"""
tests/test_benchmark_suites.py

Tests for scripts/benchmark_suites.py (Milestone 22.4, Fast Benchmark
Infrastructure).

Every test builds its own temporary suites YAML and mapping.csv (via
tmp_path) — nothing here reads the real project's
configs/benchmark_suites.yaml or benchmark/ground_truth/json/mapping.csv,
so these tests are independent of that data ever changing.
"""

import csv

import pytest
import yaml

from scripts.benchmark_suites import (
    FULL_SUITE_NAME,
    MappingFileNotFoundError,
    SuiteFileNotFoundError,
    SuiteNotFoundError,
    SuiteResumeNotFoundError,
    is_full_suite,
    load_suites,
    resolve_suite,
)


def _write_mapping_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["ResumeID", "PDF", "Markdown", "JSON"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@pytest.fixture
def ground_truth_dir(tmp_path):
    """A ground_truth_dir with a 3-row mapping.csv, mirroring the real
    project's benchmark/ground_truth/json/ layout."""
    directory = tmp_path / "ground_truth"
    directory.mkdir()
    _write_mapping_csv(
        directory / "mapping.csv",
        [
            {"ResumeID": "001", "PDF": "alice.pdf", "Markdown": "alice.md", "JSON": "resume001.json"},
            {"ResumeID": "002", "PDF": "bob.pdf", "Markdown": "bob.md", "JSON": "resume002.json"},
            {"ResumeID": "003", "PDF": "carol.pdf", "Markdown": "carol.md", "JSON": "resume003.json"},
        ],
    )
    return directory


@pytest.fixture
def suites_path(tmp_path):
    """A suites YAML with a real subset, a single-resume subset, a
    "full" wildcard suite, and an empty suite."""
    path = tmp_path / "benchmark_suites.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "smoke": ["alice", "bob"],
                "single": ["carol"],
                "full": ["*"],
                "empty": [],
            }
        ),
        encoding="utf-8",
    )
    return path


class TestLoadSuites:
    def test_loads_every_suite(self, suites_path):
        suites = load_suites(suites_path)

        assert set(suites.keys()) == {"smoke", "single", "full", "empty"}
        assert suites["smoke"] == ["alice", "bob"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(SuiteFileNotFoundError):
            load_suites(tmp_path / "does_not_exist.yaml")

    def test_empty_file_returns_empty_dict(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")

        assert load_suites(path) == {}


class TestResolveSuite:
    def test_resolves_named_subset_in_mapping_csv_order(self, suites_path, ground_truth_dir):
        rows = resolve_suite("smoke", suites_path=suites_path, ground_truth_dir=ground_truth_dir)

        assert [row["Markdown"] for row in rows] == ["alice.md", "bob.md"]

    def test_resolves_single_resume_suite(self, suites_path, ground_truth_dir):
        rows = resolve_suite("single", suites_path=suites_path, ground_truth_dir=ground_truth_dir)

        assert len(rows) == 1
        assert rows[0]["Markdown"] == "carol.md"
        assert rows[0]["JSON"] == "resume003.json"

    def test_full_wildcard_resolves_every_row(self, suites_path, ground_truth_dir):
        rows = resolve_suite("full", suites_path=suites_path, ground_truth_dir=ground_truth_dir)

        assert len(rows) == 3
        assert {row["Markdown"] for row in rows} == {"alice.md", "bob.md", "carol.md"}

    def test_empty_suite_resolves_to_no_rows(self, suites_path, ground_truth_dir):
        rows = resolve_suite("empty", suites_path=suites_path, ground_truth_dir=ground_truth_dir)

        assert rows == []

    def test_unknown_suite_name_raises_with_defined_suites_listed(
        self, suites_path, ground_truth_dir
    ):
        with pytest.raises(SuiteNotFoundError) as exc_info:
            resolve_suite("doesnotexist", suites_path=suites_path, ground_truth_dir=ground_truth_dir)

        message = str(exc_info.value)
        assert "doesnotexist" in message
        assert "smoke" in message  # the error surfaces what IS defined

    def test_suite_member_not_in_mapping_csv_raises(self, tmp_path, ground_truth_dir):
        bad_suites_path = tmp_path / "bad_suites.yaml"
        bad_suites_path.write_text(
            yaml.safe_dump({"broken": ["alice", "not_a_real_resume"]}), encoding="utf-8"
        )

        with pytest.raises(SuiteResumeNotFoundError) as exc_info:
            resolve_suite("broken", suites_path=bad_suites_path, ground_truth_dir=ground_truth_dir)

        assert "not_a_real_resume" in str(exc_info.value)

    def test_missing_mapping_csv_raises(self, suites_path, tmp_path):
        empty_ground_truth_dir = tmp_path / "no_mapping_here"
        empty_ground_truth_dir.mkdir()

        with pytest.raises(MappingFileNotFoundError):
            resolve_suite("smoke", suites_path=suites_path, ground_truth_dir=empty_ground_truth_dir)

    def test_duplicate_name_in_suite_is_deduplicated(self, tmp_path, ground_truth_dir):
        dup_suites_path = tmp_path / "dup_suites.yaml"
        dup_suites_path.write_text(
            yaml.safe_dump({"dup": ["alice", "alice", "bob"]}), encoding="utf-8"
        )

        rows = resolve_suite("dup", suites_path=dup_suites_path, ground_truth_dir=ground_truth_dir)

        assert [row["Markdown"] for row in rows] == ["alice.md", "bob.md"]


class TestIsFullSuite:
    def test_none_is_full(self):
        assert is_full_suite(None) is True

    def test_full_name_is_full(self):
        assert is_full_suite(FULL_SUITE_NAME) is True
        assert is_full_suite("full") is True

    def test_named_subset_is_not_full(self):
        assert is_full_suite("smoke") is False
        assert is_full_suite("regression") is False
