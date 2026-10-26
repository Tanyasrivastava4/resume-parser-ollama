"""
tests/test_parser_evaluator.py

Tests for evaluation/parser_evaluator.py (Milestone 14.8).

ParserPipeline is mocked via constructor injection in the unit tests
(it's the one dependency that would otherwise need a real Ollama
server); JsonComparator runs for real (pure, fast, deterministic logic).
Every test builds its own temporary ground_truth_dir/markdown_dir/
output_dir — including a mapping.csv matching the real project's
benchmark/ground_truth/json/mapping.csv layout (ResumeID, PDF, Markdown,
JSON columns) — so nothing here touches the project's real benchmark/ or
output/ directories except the one integration test, which cleans up
after itself.

One integration test at the bottom runs the real pipeline (real module
extractors, real ParserMerger, real SchemaValidator) against a real
local Ollama server. It skips itself if Ollama or the required model
isn't available.
"""

import csv
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from config import OUTPUT_DIR
from evaluation.parser_evaluator import (
    MappingFileNotFoundError,
    ParserEvaluationResult,
    ParserEvaluator,
)
from pipeline.parser_pipeline import ParserPipeline
from services.ollama_client import OllamaClient
from validators.json_comparator import JsonComparator

INTEGRATION_TEXT_MODEL = "qwen3:1.7b"

ALICE_GROUND_TRUTH = {
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

BOB_GROUND_TRUTH = {
    "personal_information": {
        "full_name": "Bob Example",
        "first_name": "Bob",
        "last_name": "Example",
        "current_location": None,
    },
    "contact_information": {
        "email": [],
        "phone": [],
        "linkedin": None,
        "github": None,
        "portfolio": None,
    },
    "professional_summary": None,
    "skills": ["SQL"],
    "languages": [],
    "work_experience": [],
    "education": [],
}


def _write_mapping_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["ResumeID", "PDF", "Markdown", "JSON"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@pytest.fixture
def dataset(tmp_path):
    """A small ground-truth + markdown dataset with a mapping.csv,
    mirroring the real project's benchmark/ground_truth/json/ layout."""
    ground_truth_dir = tmp_path / "ground_truth"
    ground_truth_dir.mkdir()
    markdown_dir = tmp_path / "markdown"
    markdown_dir.mkdir()

    (markdown_dir / "alice.md").write_text("# Alice Example", encoding="utf-8")
    (markdown_dir / "bob.md").write_text("# Bob Example", encoding="utf-8")

    (ground_truth_dir / "resume001.json").write_text(
        json.dumps(ALICE_GROUND_TRUTH), encoding="utf-8"
    )
    (ground_truth_dir / "resume002.json").write_text(
        json.dumps(BOB_GROUND_TRUTH), encoding="utf-8"
    )

    _write_mapping_csv(
        ground_truth_dir / "mapping.csv",
        [
            {"ResumeID": "001", "PDF": "alice.pdf", "Markdown": "alice.md", "JSON": "resume001.json"},
            {"ResumeID": "002", "PDF": "bob.pdf", "Markdown": "bob.md", "JSON": "resume002.json"},
        ],
    )

    return SimpleNamespace(ground_truth_dir=ground_truth_dir, markdown_dir=markdown_dir)


@pytest.fixture
def mock_parser_pipeline():
    pipeline = MagicMock(spec=ParserPipeline)

    def _run(markdown_path):
        stem = Path(markdown_path).stem
        merged = {"alice": ALICE_GROUND_TRUTH, "bob": BOB_GROUND_TRUTH}.get(stem, {})
        return SimpleNamespace(
            draft_json={}, module_results={}, merged_json=merged, processing_time=0.01
        )

    pipeline.run.side_effect = _run
    return pipeline


@pytest.fixture
def evaluator(dataset, mock_parser_pipeline, tmp_path):
    return ParserEvaluator(
        parser_pipeline=mock_parser_pipeline,
        json_comparator=JsonComparator(),
        ground_truth_dir=dataset.ground_truth_dir,
        markdown_dir=dataset.markdown_dir,
        output_dir=tmp_path / "comparison",
    )


class TestHappyPath:
    def test_returns_expected_result(self, evaluator):
        result = evaluator.evaluate()

        assert isinstance(result, ParserEvaluationResult)
        assert result.total_resumes == 2
        assert result.successful_resumes == 2
        assert result.failed_resumes == 0
        assert result.overall_accuracy == 1.0
        assert result.resume_accuracy == {"alice": 1.0, "bob": 1.0}
        assert result.failures == {}
        assert result.processing_time >= 0

    def test_module_accuracy_covers_all_seven_modules(self, evaluator):
        result = evaluator.evaluate()

        assert set(result.module_accuracy.keys()) == {
            "personal_information",
            "contact_information",
            "professional_summary",
            "skills",
            "languages",
            "work_experience",
            "education",
        }
        assert all(accuracy == 1.0 for accuracy in result.module_accuracy.values())

    def test_uses_parser_pipeline_not_individual_services(
        self, evaluator, mock_parser_pipeline
    ):
        evaluator.evaluate()

        assert mock_parser_pipeline.run.call_count == 2


class TestMissingMarkdown:
    def test_missing_markdown_recorded_as_failure_others_continue(
        self, dataset, mock_parser_pipeline, tmp_path
    ):
        _write_mapping_csv(
            dataset.ground_truth_dir / "mapping.csv",
            [
                {"ResumeID": "001", "PDF": "alice.pdf", "Markdown": "alice.md", "JSON": "resume001.json"},
                {"ResumeID": "999", "PDF": "missing.pdf", "Markdown": "missing.md", "JSON": "resume001.json"},
            ],
        )
        evaluator = ParserEvaluator(
            parser_pipeline=mock_parser_pipeline,
            json_comparator=JsonComparator(),
            ground_truth_dir=dataset.ground_truth_dir,
            markdown_dir=dataset.markdown_dir,
            output_dir=tmp_path / "comparison",
        )

        result = evaluator.evaluate()

        assert result.total_resumes == 2
        assert result.successful_resumes == 1
        assert result.failed_resumes == 1
        assert "missing" in result.failures
        assert "not found" in result.failures["missing"].lower()


class TestMissingGroundTruth:
    def test_missing_ground_truth_recorded_as_failure(
        self, dataset, mock_parser_pipeline, tmp_path
    ):
        _write_mapping_csv(
            dataset.ground_truth_dir / "mapping.csv",
            [
                {"ResumeID": "001", "PDF": "alice.pdf", "Markdown": "alice.md", "JSON": "resume001.json"},
                {"ResumeID": "002", "PDF": "bob.pdf", "Markdown": "bob.md", "JSON": "resume_does_not_exist.json"},
            ],
        )
        evaluator = ParserEvaluator(
            parser_pipeline=mock_parser_pipeline,
            json_comparator=JsonComparator(),
            ground_truth_dir=dataset.ground_truth_dir,
            markdown_dir=dataset.markdown_dir,
            output_dir=tmp_path / "comparison",
        )

        result = evaluator.evaluate()

        assert result.successful_resumes == 1
        assert result.failed_resumes == 1
        assert "bob" in result.failures
        assert "ground truth" in result.failures["bob"].lower()

    def test_missing_mapping_csv_raises(self, tmp_path, mock_parser_pipeline):
        empty_ground_truth_dir = tmp_path / "no_mapping_here"
        empty_ground_truth_dir.mkdir()
        evaluator = ParserEvaluator(
            parser_pipeline=mock_parser_pipeline,
            ground_truth_dir=empty_ground_truth_dir,
            markdown_dir=tmp_path / "markdown",
            output_dir=tmp_path / "comparison",
        )

        with pytest.raises(MappingFileNotFoundError):
            evaluator.evaluate()


class TestParserFailure:
    def test_pipeline_exception_recorded_as_failure_others_continue(
        self, evaluator, mock_parser_pipeline
    ):
        def _run(markdown_path):
            if Path(markdown_path).stem == "bob":
                raise RuntimeError("model exploded")
            return SimpleNamespace(
                draft_json={}, module_results={}, merged_json=ALICE_GROUND_TRUTH, processing_time=0.01
            )

        mock_parser_pipeline.run.side_effect = _run

        result = evaluator.evaluate()

        assert result.successful_resumes == 1
        assert result.failed_resumes == 1
        assert result.failures["bob"] == "model exploded"


class TestComparisonFailure:
    def test_invalid_ground_truth_json_recorded_as_failure(
        self, dataset, mock_parser_pipeline, tmp_path
    ):
        (dataset.ground_truth_dir / "resume002.json").write_text(
            "{this is not valid json", encoding="utf-8"
        )
        evaluator = ParserEvaluator(
            parser_pipeline=mock_parser_pipeline,
            json_comparator=JsonComparator(),
            ground_truth_dir=dataset.ground_truth_dir,
            markdown_dir=dataset.markdown_dir,
            output_dir=tmp_path / "comparison",
        )

        result = evaluator.evaluate()

        assert result.successful_resumes == 1
        assert result.failed_resumes == 1
        assert "not valid json" in result.failures["bob"].lower()


class TestReportGeneration:
    def test_all_four_reports_are_written(self, evaluator, tmp_path):
        evaluator.evaluate()

        output_dir = tmp_path / "comparison"
        for filename in (
            "field_accuracy.csv",
            "resume_accuracy.csv",
            "module_accuracy.csv",
            "benchmark_report.md",
        ):
            assert (output_dir / filename).is_file(), filename

    def test_module_accuracy_csv_content(self, evaluator, tmp_path):
        evaluator.evaluate()

        with (tmp_path / "comparison" / "module_accuracy.csv").open(
            newline="", encoding="utf-8"
        ) as csv_file:
            rows = {row["module"]: row for row in csv.DictReader(csv_file)}

        assert rows["skills"]["accuracy"] == "1.0000"
        assert rows["skills"]["total"] == "2"

    def test_benchmark_report_contains_summary_and_failures_section(
        self, evaluator, tmp_path
    ):
        evaluator.evaluate()

        report_text = (tmp_path / "comparison" / "benchmark_report.md").read_text(
            encoding="utf-8"
        )

        assert "# Parser Evaluation Report" in report_text
        assert "Total resumes" in report_text
        assert "## Module Accuracy" in report_text
        assert "## Failures" in report_text
        assert "None." in report_text  # no failures in the happy-path dataset


class TestDiagnosticMode:
    """Milestone 20.3: diagnostic mode is opt-in (diagnostic=True) and
    additive only -- it must never change evaluate()'s existing return
    value or files when left off (the default), and even when on, must
    never change the strict numbers, only add new information alongside
    them."""

    def test_default_evaluator_has_no_diagnostics(self, evaluator):
        result = evaluator.evaluate()

        assert result.diagnostics == {}

    def test_diagnostic_true_populates_diagnostics_without_changing_strict_result(
        self, dataset, mock_parser_pipeline, tmp_path
    ):
        plain_evaluator = ParserEvaluator(
            parser_pipeline=mock_parser_pipeline,
            json_comparator=JsonComparator(),
            ground_truth_dir=dataset.ground_truth_dir,
            markdown_dir=dataset.markdown_dir,
            output_dir=tmp_path / "comparison_plain",
        )
        plain_result = plain_evaluator.evaluate()

        diagnostic_evaluator = ParserEvaluator(
            parser_pipeline=mock_parser_pipeline,
            json_comparator=JsonComparator(),
            ground_truth_dir=dataset.ground_truth_dir,
            markdown_dir=dataset.markdown_dir,
            output_dir=tmp_path / "comparison_diagnostic",
            diagnostic=True,
        )
        diagnostic_result = diagnostic_evaluator.evaluate()

        # Every strict number is identical either way.
        assert diagnostic_result.overall_accuracy == plain_result.overall_accuracy
        assert diagnostic_result.resume_accuracy == plain_result.resume_accuracy
        assert diagnostic_result.module_accuracy == plain_result.module_accuracy
        assert diagnostic_result.successful_resumes == plain_result.successful_resumes

        # ... but diagnostic mode additionally populated diagnostics.
        assert set(diagnostic_result.diagnostics.keys()) == {"alice", "bob"}
        for diag in diagnostic_result.diagnostics.values():
            assert diag.strict_accuracy == 1.0  # happy-path fixture: identical GT/actual

    def test_diagnostic_true_writes_additional_reports_without_touching_existing_ones(
        self, dataset, mock_parser_pipeline, tmp_path
    ):
        output_dir = tmp_path / "comparison"
        evaluator = ParserEvaluator(
            parser_pipeline=mock_parser_pipeline,
            json_comparator=JsonComparator(),
            ground_truth_dir=dataset.ground_truth_dir,
            markdown_dir=dataset.markdown_dir,
            output_dir=output_dir,
            diagnostic=True,
        )

        evaluator.evaluate()

        # The four reports evaluate() has always written are still there...
        for filename in (
            "field_accuracy.csv",
            "resume_accuracy.csv",
            "module_accuracy.csv",
            "benchmark_report.md",
        ):
            assert (output_dir / filename).is_file(), filename
        # ...alongside the three new diagnostic-mode reports.
        for filename in (
            "diagnostic_summary.csv",
            "diagnostic_mismatch_by_field.csv",
            "diagnostic_report.md",
        ):
            assert (output_dir / filename).is_file(), filename

    def test_diagnostic_false_writes_no_diagnostic_reports(self, evaluator, tmp_path):
        evaluator.evaluate()

        output_dir = tmp_path / "comparison"
        for filename in (
            "diagnostic_summary.csv",
            "diagnostic_mismatch_by_field.csv",
            "diagnostic_report.md",
        ):
            assert not (output_dir / filename).exists(), filename

    def test_strict_report_files_byte_identical_with_diagnostic_on_or_off(
        self, dataset, mock_parser_pipeline, tmp_path, monkeypatch
    ):
        """The strongest form of requirement 1: not just equal numbers,
        but byte-identical report files, whether or not diagnostic mode
        ran alongside them.

        benchmark_report.md embeds the run's wall-clock processing time
        (`| Processing time | 0.04s |`), which legitimately differs
        between the two evaluate() calls below and made this test
        timing-flaky (observed 2026-10-17 under CPU load: 0.04s vs 0.05s).
        Pin the evaluator's clock so the comparison tests report content,
        not scheduling luck. Nothing else about the two runs is patched."""
        import evaluation.parser_evaluator as parser_evaluator_module

        monkeypatch.setattr(
            parser_evaluator_module,
            "time",
            SimpleNamespace(perf_counter=lambda: 0.0),
        )
        plain_dir = tmp_path / "plain"
        ParserEvaluator(
            parser_pipeline=mock_parser_pipeline,
            json_comparator=JsonComparator(),
            ground_truth_dir=dataset.ground_truth_dir,
            markdown_dir=dataset.markdown_dir,
            output_dir=plain_dir,
        ).evaluate()

        diagnostic_dir = tmp_path / "diagnostic"
        ParserEvaluator(
            parser_pipeline=mock_parser_pipeline,
            json_comparator=JsonComparator(),
            ground_truth_dir=dataset.ground_truth_dir,
            markdown_dir=dataset.markdown_dir,
            output_dir=diagnostic_dir,
            diagnostic=True,
        ).evaluate()

        for filename in (
            "field_accuracy.csv",
            "resume_accuracy.csv",
            "module_accuracy.csv",
            "benchmark_report.md",
        ):
            assert (plain_dir / filename).read_text(encoding="utf-8") == (
                diagnostic_dir / filename
            ).read_text(encoding="utf-8"), filename

    def test_a_resume_that_fails_evaluation_is_not_added_to_diagnostics(
        self, dataset, mock_parser_pipeline, tmp_path
    ):
        def _run(markdown_path):
            if Path(markdown_path).stem == "bob":
                raise RuntimeError("model exploded")
            return SimpleNamespace(
                draft_json={}, module_results={}, merged_json=ALICE_GROUND_TRUTH, processing_time=0.01
            )

        mock_parser_pipeline.run.side_effect = _run
        evaluator = ParserEvaluator(
            parser_pipeline=mock_parser_pipeline,
            json_comparator=JsonComparator(),
            ground_truth_dir=dataset.ground_truth_dir,
            markdown_dir=dataset.markdown_dir,
            output_dir=tmp_path / "comparison",
            diagnostic=True,
        )

        result = evaluator.evaluate()

        assert result.failed_resumes == 1
        assert "bob" not in result.diagnostics
        assert "alice" in result.diagnostics


class TestOutputCleanupHelper:
    """Sanity check that the CSVs/report use the resume names, not the
    numeric ResumeID from mapping.csv."""

    def test_resume_accuracy_csv_uses_markdown_stem_as_name(self, evaluator, tmp_path):
        evaluator.evaluate()

        with (tmp_path / "comparison" / "resume_accuracy.csv").open(
            newline="", encoding="utf-8"
        ) as csv_file:
            names = {row["resume_name"] for row in csv.DictReader(csv_file)}

        assert names == {"alice", "bob"}


@pytest.mark.integration
def test_real_evaluator_end_to_end_integration(tmp_path):
    """Optional integration test running the real pipeline (real module
    extractors, real ParserMerger, real SchemaValidator) against a real
    local Ollama server. Skips itself if Ollama or the required model
    isn't available."""
    ollama_client = OllamaClient()

    if not ollama_client.is_running():
        pytest.skip("Ollama is not running locally; skipping integration test.")
    if not ollama_client.model_exists(INTEGRATION_TEXT_MODEL):
        pytest.skip(f"Model '{INTEGRATION_TEXT_MODEL}' is not installed; skipping.")

    resume_name = "test_evaluator_integration_resume"
    ground_truth_dir = tmp_path / "ground_truth"
    ground_truth_dir.mkdir()
    markdown_dir = tmp_path / "markdown"
    markdown_dir.mkdir()
    output_dir = tmp_path / "comparison"

    (markdown_dir / f"{resume_name}.md").write_text(
        "# Ada Lovelace\n\n## Summary\n\nMathematician and writer.\n\n"
        "## Contact\n\nEmail: ada@example.com\n\n"
        "## Skills\n\n- Mathematics\n\n## Languages\n\n- English\n",
        encoding="utf-8",
    )
    ground_truth = {
        "personal_information": {
            "full_name": "Ada Lovelace",
            "first_name": "Ada",
            "last_name": "Lovelace",
            "current_location": None,
        },
        "contact_information": {
            "email": ["ada@example.com"],
            "phone": [],
            "linkedin": None,
            "github": None,
            "portfolio": None,
        },
        "professional_summary": "Mathematician and writer.",
        "skills": ["Mathematics"],
        "languages": ["English"],
        "work_experience": [],
        "education": [],
    }
    (ground_truth_dir / "resume001.json").write_text(json.dumps(ground_truth), encoding="utf-8")
    _write_mapping_csv(
        ground_truth_dir / "mapping.csv",
        [
            {
                "ResumeID": "001",
                "PDF": f"{resume_name}.pdf",
                "Markdown": f"{resume_name}.md",
                "JSON": "resume001.json",
            }
        ],
    )

    # ParserPipeline (used internally, for real, by ParserEvaluator here)
    # always saves its own artifacts under the project's real output/
    # tree — evaluator's ground_truth_dir/markdown_dir/output_dir only
    # control the evaluator's own inputs and reports, so those real
    # pipeline artifacts are cleaned up in `finally` below.
    evaluator = ParserEvaluator(
        ground_truth_dir=ground_truth_dir,
        markdown_dir=markdown_dir,
        output_dir=output_dir,
    )

    try:
        result = evaluator.evaluate()

        assert isinstance(result, ParserEvaluationResult)
        assert result.total_resumes == 1
        assert result.successful_resumes + result.failed_resumes == 1
        assert (output_dir / "benchmark_report.md").is_file()
        assert (output_dir / "field_accuracy.csv").is_file()
        assert (output_dir / "resume_accuracy.csv").is_file()
        assert (output_dir / "module_accuracy.csv").is_file()
    finally:
        shutil.rmtree(OUTPUT_DIR / "modules" / resume_name, ignore_errors=True)
        (OUTPUT_DIR / "json_final" / f"{resume_name}.json").unlink(missing_ok=True)
        (OUTPUT_DIR / "json_draft" / f"{resume_name}.json").unlink(missing_ok=True)
