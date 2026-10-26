"""
tests/test_parser_evaluator_finalization.py

Tests for Milestone 14.10 (Evaluation Framework Finalization) changes to
evaluation/parser_evaluator.py and scripts/run_parser_evaluation.py:

    Problem 1 - the real project's mapping.csv now points at real
        markdown filenames (previously Aman.md/Anandu.md/Ankush.md,
        which didn't exist under output/markdown/).
    Problem 2 - ground-truth files are validated against
        schemas/resume_schema_v1.json before use; invalid ones are
        skipped with a clear reason, and evaluation continues with the
        rest.
    Problem 3 - overall_accuracy is never a misleading 100% when zero
        resumes succeeded.
    Problem 4 - failures carry full context (resume, reason, ground
        truth file, markdown file), surfaced via
        ParserEvaluationResult.failure_details, benchmark_report.md,
        and the CLI's printed output.

ParserPipeline is mocked via constructor injection throughout (as in
tests/test_parser_evaluator.py) — no real Ollama server is required.
Every test builds its own temporary ground_truth_dir/markdown_dir/
output_dir, except TestRealMappingCsvPointsAtRealMarkdownFiles, which
deliberately reads the real project's mapping.csv/output/markdown to
confirm Problem 1's data fix (read-only; nothing is written).
"""

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from config import GROUND_TRUTH_DIR, MARKDOWN_DIR
from evaluation.parser_evaluator import (
    EvaluationFailure,
    ParserEvaluationResult,
    ParserEvaluator,
)
from pipeline.parser_pipeline import ParserPipeline
from scripts.run_parser_evaluation import print_evaluation_result
from validators.json_comparator import JsonComparator

VALID_GROUND_TRUTH = {
    "personal_information": {
        "full_name": "Alice Example",
        "first_name": "Alice",
        "last_name": "Example",
        "current_location": None,
    },
    "contact_information": {
        "email": ["alice@example.com"],
        "phone": [],
        "linkedin": None,
        "github": None,
        "portfolio": None,
    },
    "professional_summary": "Engineer.",
    "skills": ["Python"],
    "languages": ["English"],
    "work_experience": [],
    "education": [],
}

# Valid JSON, but missing every required top-level property -> fails
# schema validation without failing JSON parsing.
SCHEMA_INVALID_GROUND_TRUTH = {"skills": ["SQL"]}


def _write_mapping_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["ResumeID", "PDF", "Markdown", "JSON"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@pytest.fixture
def mock_parser_pipeline():
    pipeline = MagicMock(spec=ParserPipeline)
    pipeline.run.return_value = SimpleNamespace(
        draft_json={}, module_results={}, merged_json=VALID_GROUND_TRUTH, processing_time=0.01
    )
    return pipeline


@pytest.mark.local_data('any_corpus_markdown')
class TestRealMappingCsvPointsAtRealMarkdownFiles:
    """Problem 1: checks the real project's mapping.csv data file
    directly (read-only) rather than a synthetic copy, since the bug
    was in that data, not in general evaluator logic."""

    def test_every_mapped_markdown_file_exists_on_disk(self):
        mapping_path = GROUND_TRUTH_DIR / "mapping.csv"
        with mapping_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 23
        missing = [
            row["Markdown"] for row in rows if not (MARKDOWN_DIR / row["Markdown"]).is_file()
        ]
        assert missing == []

    def test_first_three_rows_no_longer_use_the_old_broken_short_names(self):
        mapping_path = GROUND_TRUTH_DIR / "mapping.csv"
        with mapping_path.open(newline="", encoding="utf-8") as f:
            rows = {row["ResumeID"]: row["Markdown"] for row in csv.DictReader(f)}

        assert rows["001"] != "Aman.md"
        assert rows["002"] != "Anandu.md"
        assert rows["003"] != "Ankush.md"


class TestGroundTruthSchemaValidation:
    """Problem 2."""

    def test_schema_invalid_ground_truth_is_skipped_others_continue(
        self, tmp_path, mock_parser_pipeline
    ):
        ground_truth_dir = tmp_path / "ground_truth"
        ground_truth_dir.mkdir()
        markdown_dir = tmp_path / "markdown"
        markdown_dir.mkdir()

        (markdown_dir / "alice.md").write_text("# Alice", encoding="utf-8")
        (markdown_dir / "bob.md").write_text("# Bob", encoding="utf-8")

        (ground_truth_dir / "resume001.json").write_text(
            json.dumps(VALID_GROUND_TRUTH), encoding="utf-8"
        )
        (ground_truth_dir / "resume002.json").write_text(
            json.dumps(SCHEMA_INVALID_GROUND_TRUTH), encoding="utf-8"
        )

        _write_mapping_csv(
            ground_truth_dir / "mapping.csv",
            [
                {
                    "ResumeID": "001",
                    "PDF": "alice.pdf",
                    "Markdown": "alice.md",
                    "JSON": "resume001.json",
                },
                {
                    "ResumeID": "002",
                    "PDF": "bob.pdf",
                    "Markdown": "bob.md",
                    "JSON": "resume002.json",
                },
            ],
        )

        evaluator = ParserEvaluator(
            parser_pipeline=mock_parser_pipeline,
            json_comparator=JsonComparator(),
            ground_truth_dir=ground_truth_dir,
            markdown_dir=markdown_dir,
            output_dir=tmp_path / "comparison",
        )

        result = evaluator.evaluate()

        assert result.total_resumes == 2
        assert result.successful_resumes == 1
        assert result.failed_resumes == 1
        assert "bob" in result.failures
        assert "schema" in result.failures["bob"].lower()
        # The pipeline must never even be invoked for the skipped resume.
        called_stems = {
            Path(call.args[0]).stem for call in mock_parser_pipeline.run.call_args_list
        }
        assert called_stems == {"alice"}

    def test_invalid_json_is_still_reported_distinctly_from_schema_failure(
        self, tmp_path, mock_parser_pipeline
    ):
        ground_truth_dir = tmp_path / "ground_truth"
        ground_truth_dir.mkdir()
        markdown_dir = tmp_path / "markdown"
        markdown_dir.mkdir()

        (markdown_dir / "bob.md").write_text("# Bob", encoding="utf-8")
        (ground_truth_dir / "resume002.json").write_text(
            "{not valid json", encoding="utf-8"
        )
        _write_mapping_csv(
            ground_truth_dir / "mapping.csv",
            [
                {
                    "ResumeID": "002",
                    "PDF": "bob.pdf",
                    "Markdown": "bob.md",
                    "JSON": "resume002.json",
                }
            ],
        )

        evaluator = ParserEvaluator(
            parser_pipeline=mock_parser_pipeline,
            json_comparator=JsonComparator(),
            ground_truth_dir=ground_truth_dir,
            markdown_dir=markdown_dir,
            output_dir=tmp_path / "comparison",
        )

        result = evaluator.evaluate()

        assert result.failed_resumes == 1
        assert "not valid json" in result.failures["bob"].lower()
        assert "schema" not in result.failures["bob"].lower()

    def test_schema_valid_ground_truth_passes_and_is_evaluated(
        self, tmp_path, mock_parser_pipeline
    ):
        ground_truth_dir = tmp_path / "ground_truth"
        ground_truth_dir.mkdir()
        markdown_dir = tmp_path / "markdown"
        markdown_dir.mkdir()

        (markdown_dir / "alice.md").write_text("# Alice", encoding="utf-8")
        (ground_truth_dir / "resume001.json").write_text(
            json.dumps(VALID_GROUND_TRUTH), encoding="utf-8"
        )
        _write_mapping_csv(
            ground_truth_dir / "mapping.csv",
            [
                {
                    "ResumeID": "001",
                    "PDF": "alice.pdf",
                    "Markdown": "alice.md",
                    "JSON": "resume001.json",
                }
            ],
        )

        evaluator = ParserEvaluator(
            parser_pipeline=mock_parser_pipeline,
            json_comparator=JsonComparator(),
            ground_truth_dir=ground_truth_dir,
            markdown_dir=markdown_dir,
            output_dir=tmp_path / "comparison",
        )

        result = evaluator.evaluate()

        assert result.successful_resumes == 1
        assert result.failed_resumes == 0


class TestOverallAccuracyNeverMisleading:
    """Problem 3."""

    def test_all_resumes_failing_reports_zero_not_full_accuracy(
        self, tmp_path, mock_parser_pipeline
    ):
        ground_truth_dir = tmp_path / "ground_truth"
        ground_truth_dir.mkdir()
        markdown_dir = tmp_path / "markdown"
        markdown_dir.mkdir()
        # No markdown files created at all -> every row fails at
        # "markdown not found", so comparisons stays empty.

        _write_mapping_csv(
            ground_truth_dir / "mapping.csv",
            [
                {"ResumeID": "001", "PDF": "a.pdf", "Markdown": "a.md", "JSON": "resume001.json"},
                {"ResumeID": "002", "PDF": "b.pdf", "Markdown": "b.md", "JSON": "resume002.json"},
            ],
        )

        evaluator = ParserEvaluator(
            parser_pipeline=mock_parser_pipeline,
            json_comparator=JsonComparator(),
            ground_truth_dir=ground_truth_dir,
            markdown_dir=markdown_dir,
            output_dir=tmp_path / "comparison",
        )

        result = evaluator.evaluate()

        assert result.successful_resumes == 0
        assert result.failed_resumes == 2
        assert result.overall_accuracy == 0.0

    def test_benchmark_report_shows_zero_percent_not_100_percent(
        self, tmp_path, mock_parser_pipeline
    ):
        ground_truth_dir = tmp_path / "ground_truth"
        ground_truth_dir.mkdir()
        markdown_dir = tmp_path / "markdown"
        markdown_dir.mkdir()

        _write_mapping_csv(
            ground_truth_dir / "mapping.csv",
            [{"ResumeID": "001", "PDF": "a.pdf", "Markdown": "a.md", "JSON": "resume001.json"}],
        )

        evaluator = ParserEvaluator(
            parser_pipeline=mock_parser_pipeline,
            json_comparator=JsonComparator(),
            ground_truth_dir=ground_truth_dir,
            markdown_dir=markdown_dir,
            output_dir=tmp_path / "comparison",
        )

        evaluator.evaluate()

        report_text = (tmp_path / "comparison" / "benchmark_report.md").read_text(
            encoding="utf-8"
        )
        overall_accuracy_line = next(
            line for line in report_text.splitlines() if line.startswith("| Overall accuracy")
        )
        assert "0.00%" in overall_accuracy_line
        assert "100.00%" not in overall_accuracy_line


class TestFailureDetails:
    """Problem 4."""

    def test_failure_details_populated_with_full_context(self, tmp_path, mock_parser_pipeline):
        ground_truth_dir = tmp_path / "ground_truth"
        ground_truth_dir.mkdir()
        markdown_dir = tmp_path / "markdown"
        markdown_dir.mkdir()

        _write_mapping_csv(
            ground_truth_dir / "mapping.csv",
            [
                {
                    "ResumeID": "999",
                    "PDF": "missing.pdf",
                    "Markdown": "missing.md",
                    "JSON": "resume999.json",
                }
            ],
        )

        evaluator = ParserEvaluator(
            parser_pipeline=mock_parser_pipeline,
            json_comparator=JsonComparator(),
            ground_truth_dir=ground_truth_dir,
            markdown_dir=markdown_dir,
            output_dir=tmp_path / "comparison",
        )

        result = evaluator.evaluate()

        assert "missing" in result.failure_details
        detail = result.failure_details["missing"]
        assert isinstance(detail, EvaluationFailure)
        assert detail.resume == "missing"
        assert "not found" in detail.reason.lower()
        assert detail.markdown_file == str(markdown_dir / "missing.md")
        assert detail.ground_truth_file == str(ground_truth_dir / "resume999.json")
        # The plain-string failures dict must still carry the same reason.
        assert result.failures["missing"] == detail.reason

    def test_benchmark_report_failures_table_has_all_four_columns(
        self, tmp_path, mock_parser_pipeline
    ):
        ground_truth_dir = tmp_path / "ground_truth"
        ground_truth_dir.mkdir()
        markdown_dir = tmp_path / "markdown"
        markdown_dir.mkdir()

        _write_mapping_csv(
            ground_truth_dir / "mapping.csv",
            [
                {
                    "ResumeID": "999",
                    "PDF": "missing.pdf",
                    "Markdown": "missing.md",
                    "JSON": "resume999.json",
                }
            ],
        )

        evaluator = ParserEvaluator(
            parser_pipeline=mock_parser_pipeline,
            json_comparator=JsonComparator(),
            ground_truth_dir=ground_truth_dir,
            markdown_dir=markdown_dir,
            output_dir=tmp_path / "comparison",
        )

        evaluator.evaluate()

        report_text = (tmp_path / "comparison" / "benchmark_report.md").read_text(
            encoding="utf-8"
        )
        assert "| Resume | Reason | Ground Truth File | Markdown File |" in report_text
        assert str(markdown_dir / "missing.md") in report_text
        assert str(ground_truth_dir / "resume999.json") in report_text

    def test_cli_prints_resume_reason_ground_truth_file_and_markdown_file(self, capsys):
        result = ParserEvaluationResult(
            total_resumes=1,
            successful_resumes=0,
            failed_resumes=1,
            overall_accuracy=0.0,
            failures={"bob": "Ground truth file is not valid JSON: /x/resume002.json"},
            failure_details={
                "bob": EvaluationFailure(
                    resume="bob",
                    reason="Ground truth file is not valid JSON: /x/resume002.json",
                    ground_truth_file="/x/resume002.json",
                    markdown_file="/y/bob.md",
                )
            },
        )

        print_evaluation_result(result, Path("/fake/comparison"))

        captured = capsys.readouterr()
        assert "Resume:             bob" in captured.out
        assert (
            "Reason:             Ground truth file is not valid JSON: /x/resume002.json"
            in captured.out
        )
        assert "Ground truth file:  /x/resume002.json" in captured.out
        assert "Markdown file:      /y/bob.md" in captured.out

    def test_cli_falls_back_to_plain_reason_when_no_failure_details(self, capsys):
        """Backward compatibility: a ParserEvaluationResult built without
        failure_details (e.g. constructed directly by older code/tests)
        must still print something useful, not crash."""
        result = ParserEvaluationResult(
            total_resumes=1,
            successful_resumes=0,
            failed_resumes=1,
            overall_accuracy=0.0,
            failures={"bob": "Markdown file not found: bob.md"},
        )

        print_evaluation_result(result, Path("/fake/comparison"))

        captured = capsys.readouterr()
        assert "bob: Markdown file not found: bob.md" in captured.out
