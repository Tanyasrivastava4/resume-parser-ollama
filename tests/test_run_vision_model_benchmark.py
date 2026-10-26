"""
tests/test_run_vision_model_benchmark.py

Tests for scripts/run_vision_model_benchmark.py (Milestone 16.2).

BatchProcessor and OllamaClient are always mocked — no test here talks to
a real Ollama server or renders a real PDF, matching
tests/test_run_markdown_benchmark.py's convention for the analogous
existing script. MarkdownBenchmark itself is deliberately NOT mocked:
it's cheap, deterministic, pure-text-diff logic (no Ollama dependency),
so tests exercise the real, unmodified class against small real
fixture files under tmp_path — this both tests this script's plumbing
end-to-end and double-checks the "reuse it, don't duplicate it"
requirement is actually true, not just claimed.
"""

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from benchmark.markdown_benchmark import MarkdownBenchmarkResult
from config import BASE_DIR, OUTPUT_DIR
from services.batch_processor import BatchFailure, BatchProcessingResult, BatchProcessor, DocumentBatchRecord
from services.ollama_client import OllamaClient
from scripts.run_vision_model_benchmark import (
    ModelBenchmarkOutcome,
    archive_generated_markdown,
    backup_markdown_dir,
    benchmark_one_model,
    build_summary_report,
    composite_accuracy,
    parse_models_arg,
    restore_markdown_dir,
    run_vision_model_benchmark_cli,
    _parse_args,
)


def _make_batch_result(records) -> BatchProcessingResult:
    successes = [r for r in records if r.success]
    failures = [
        BatchFailure(pdf_path=r.pdf_path, error=r.error) for r in records if not r.success
    ]
    return BatchProcessingResult(
        total_documents=len(records),
        successful_documents=len(successes),
        failed_documents=len(failures),
        total_processing_time=12.5,
        average_processing_time=12.5 / len(records) if records else 0.0,
        list_of_failures=failures,
        results=records,
    )


def _make_benchmark_result(**overrides) -> MarkdownBenchmarkResult:
    defaults = dict(
        total_resumes=1,
        successful_resumes=1,
        failed_resumes=0,
        overall_character_accuracy=0.9,
        overall_word_accuracy=0.9,
        overall_line_accuracy=0.9,
        overall_heading_accuracy=0.9,
        total_missing_lines=1,
        total_additional_lines=2,
        total_differences=3,
    )
    defaults.update(overrides)
    return MarkdownBenchmarkResult(**defaults)


class TestParseModelsArg:
    def test_splits_comma_separated_list(self):
        assert parse_models_arg("qwen2.5vl:3b,qwen2.5vl:7b") == [
            "qwen2.5vl:3b",
            "qwen2.5vl:7b",
        ]

    def test_trims_whitespace_around_each_model(self):
        assert parse_models_arg(" qwen2.5vl:3b , qwen2.5vl:7b ") == [
            "qwen2.5vl:3b",
            "qwen2.5vl:7b",
        ]

    def test_single_model_is_valid(self):
        assert parse_models_arg("qwen2.5vl:3b") == ["qwen2.5vl:3b"]

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_models_arg("")

    def test_only_commas_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_models_arg(" , , ")


class TestParseArgs:
    def test_requires_models(self):
        with pytest.raises(SystemExit) as exc_info:
            _parse_args([])
        assert exc_info.value.code != 0

    def test_accepts_models_argument(self):
        args = _parse_args(["--models", "qwen2.5vl:3b,qwen2.5vl:7b"])
        assert args.models == "qwen2.5vl:3b,qwen2.5vl:7b"

    def test_help_exits_cleanly(self):
        with pytest.raises(SystemExit) as exc_info:
            _parse_args(["--help"])
        assert exc_info.value.code == 0


class TestArchiveGeneratedMarkdown:
    """record.markdown_path is always relative to the project root (the
    same contract DocumentExtractionResult.markdown_path establishes),
    so these tests create real files under OUTPUT_DIR, matching the
    convention tests/test_document_extractor.py already uses for the
    same reason, and clean them up afterward."""

    @pytest.fixture
    def real_markdown_file(self):
        document_name = "test_vmb_archive_resume"
        relative_path = f"output/markdown/{document_name}.md"
        absolute_path = BASE_DIR / relative_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        absolute_path.write_text("# Generated content\n", encoding="utf-8")
        yield document_name, relative_path
        absolute_path.unlink(missing_ok=True)

    def test_copies_only_successful_records(self, tmp_path, real_markdown_file):
        document_name, relative_path = real_markdown_file
        records = [
            DocumentBatchRecord(
                pdf_path="input/resumes/a.pdf",
                success=True,
                processing_time=1.0,
                page_count=1,
                markdown_path=relative_path,
            ),
            DocumentBatchRecord(
                pdf_path="input/resumes/b.pdf",
                success=False,
                processing_time=0.5,
                error="model exploded",
            ),
        ]
        batch_result = _make_batch_result(records)
        destination_dir = tmp_path / "archive"

        copied = archive_generated_markdown(batch_result, destination_dir)

        assert copied == 1
        archived_file = destination_dir / f"{document_name}.md"
        assert archived_file.is_file()
        assert archived_file.read_text(encoding="utf-8") == "# Generated content\n"
        assert len(list(destination_dir.iterdir())) == 1

    def test_creates_destination_dir_if_missing(self, tmp_path, real_markdown_file):
        _, relative_path = real_markdown_file
        records = [
            DocumentBatchRecord(
                pdf_path="input/resumes/a.pdf",
                success=True,
                processing_time=1.0,
                page_count=1,
                markdown_path=relative_path,
            )
        ]
        destination_dir = tmp_path / "does" / "not" / "exist" / "yet"

        archive_generated_markdown(_make_batch_result(records), destination_dir)

        assert destination_dir.is_dir()

    def test_returns_zero_for_all_failed_batch(self, tmp_path):
        records = [
            DocumentBatchRecord(
                pdf_path="input/resumes/a.pdf", success=False, processing_time=0.5, error="boom"
            )
        ]
        destination_dir = tmp_path / "archive"

        copied = archive_generated_markdown(_make_batch_result(records), destination_dir)

        assert copied == 0
        assert list(destination_dir.iterdir()) == []


class TestBackupRestoreMarkdownDir:
    def test_backup_returns_none_when_source_missing(self, tmp_path):
        missing_dir = tmp_path / "does_not_exist"
        assert backup_markdown_dir(missing_dir) is None

    def test_restore_is_noop_when_backup_is_none(self, tmp_path):
        markdown_dir = tmp_path / "markdown"
        markdown_dir.mkdir()
        (markdown_dir / "keep.md").write_text("original", encoding="utf-8")

        restore_markdown_dir(markdown_dir, None)

        assert (markdown_dir / "keep.md").read_text(encoding="utf-8") == "original"

    def test_backup_then_restore_recovers_original_content(self, tmp_path):
        markdown_dir = tmp_path / "markdown"
        markdown_dir.mkdir()
        (markdown_dir / "resume_a.md").write_text("original a", encoding="utf-8")
        (markdown_dir / "resume_b.md").write_text("original b", encoding="utf-8")

        backup_dir = backup_markdown_dir(markdown_dir)

        # Simulate a model run overwriting the shared markdown directory.
        shutil.rmtree(markdown_dir)
        markdown_dir.mkdir()
        (markdown_dir / "resume_a.md").write_text("overwritten by another model", encoding="utf-8")

        restore_markdown_dir(markdown_dir, backup_dir)

        assert (markdown_dir / "resume_a.md").read_text(encoding="utf-8") == "original a"
        assert (markdown_dir / "resume_b.md").read_text(encoding="utf-8") == "original b"
        assert not backup_dir.exists()


class TestCompositeAccuracy:
    def test_averages_the_four_metrics(self):
        result = _make_benchmark_result(
            overall_character_accuracy=1.0,
            overall_word_accuracy=0.8,
            overall_line_accuracy=0.6,
            overall_heading_accuracy=0.4,
        )
        assert composite_accuracy(result) == pytest.approx(0.7)


class TestBenchmarkOneModel:
    @pytest.fixture
    def reference_and_markdown_dirs(self, tmp_path):
        reference_dir = tmp_path / "reference"
        reference_dir.mkdir()
        (reference_dir / "resume_one.md").write_text("# Resume One\nLine A\n", encoding="utf-8")
        return reference_dir

    def test_skips_model_not_installed(self, tmp_path):
        ollama_client = MagicMock(spec=OllamaClient)
        ollama_client.model_exists.return_value = False
        batch_processor = MagicMock(spec=BatchProcessor)

        outcome = benchmark_one_model(
            "not-a-real-model:1b",
            batch_processor=batch_processor,
            ollama_client=ollama_client,
            reference_dir=tmp_path / "reference",
            benchmark_models_dir=tmp_path / "benchmark_models",
        )

        assert outcome.succeeded is False
        assert "not installed" in outcome.skip_reason
        assert outcome.batch_result is None
        assert outcome.benchmark_result is None
        batch_processor.process_batch.assert_not_called()

    def test_installed_model_runs_batch_and_benchmark(
        self, tmp_path, reference_and_markdown_dirs
    ):
        reference_dir = reference_and_markdown_dirs
        document_name = "test_vmb_one_model_resume"
        relative_path = f"output/markdown/{document_name}.md"
        absolute_path = BASE_DIR / relative_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        absolute_path.write_text("# Resume One\nLine A\n", encoding="utf-8")
        (reference_dir / f"{document_name}.md").write_text(
            "# Resume One\nLine A\n", encoding="utf-8"
        )

        try:
            ollama_client = MagicMock(spec=OllamaClient)
            ollama_client.model_exists.return_value = True
            batch_processor = MagicMock(spec=BatchProcessor)
            batch_processor.process_batch.return_value = _make_batch_result(
                [
                    DocumentBatchRecord(
                        pdf_path="input/resumes/x.pdf",
                        success=True,
                        processing_time=3.0,
                        page_count=1,
                        markdown_path=relative_path,
                    )
                ]
            )
            benchmark_models_dir = tmp_path / "benchmark_models"

            outcome = benchmark_one_model(
                "qwen2.5vl:3b",
                batch_processor=batch_processor,
                ollama_client=ollama_client,
                reference_dir=reference_dir,
                benchmark_models_dir=benchmark_models_dir,
            )

            assert outcome.succeeded is True
            batch_processor.process_batch.assert_called_once_with(
                model_name="qwen2.5vl:3b", prompt_id="resume/extract_markdown"
            )
            # Archived a real copy of the generated markdown.
            archived_file = benchmark_models_dir / "qwen2.5vl:3b" / "markdown" / f"{document_name}.md"
            assert archived_file.is_file()
            # Scored with the real, unmodified MarkdownBenchmark — perfect
            # match between identical reference and generated text.
            assert outcome.benchmark_result.overall_character_accuracy == 1.0
            assert outcome.benchmark_result.successful_resumes == 1
        finally:
            absolute_path.unlink(missing_ok=True)


class TestBuildSummaryReport:
    def test_ranks_models_by_composite_accuracy_descending(self):
        better = ModelBenchmarkOutcome(
            model_name="qwen2.5vl:7b",
            succeeded=True,
            batch_result=_make_batch_result(
                [
                    DocumentBatchRecord(
                        pdf_path="a.pdf", success=True, processing_time=10.0, page_count=1
                    )
                ]
            ),
            benchmark_result=_make_benchmark_result(
                overall_character_accuracy=0.99,
                overall_word_accuracy=0.99,
                overall_line_accuracy=0.99,
                overall_heading_accuracy=0.99,
            ),
        )
        worse = ModelBenchmarkOutcome(
            model_name="qwen2.5vl:3b",
            succeeded=True,
            batch_result=_make_batch_result(
                [
                    DocumentBatchRecord(
                        pdf_path="a.pdf", success=True, processing_time=5.0, page_count=1
                    )
                ]
            ),
            benchmark_result=_make_benchmark_result(
                overall_character_accuracy=0.80,
                overall_word_accuracy=0.80,
                overall_line_accuracy=0.80,
                overall_heading_accuracy=0.80,
            ),
        )

        report = build_summary_report([worse, better])

        better_index = report.index("qwen2.5vl:7b")
        worse_index = report.index("qwen2.5vl:3b")
        assert better_index < worse_index
        assert "| qwen2.5vl:7b |" in report
        assert "99.00%" in report

    def test_includes_all_required_columns(self):
        outcome = ModelBenchmarkOutcome(
            model_name="qwen2.5vl:3b",
            succeeded=True,
            batch_result=_make_batch_result(
                [DocumentBatchRecord(pdf_path="a.pdf", success=True, processing_time=1.0, page_count=1)]
            ),
            benchmark_result=_make_benchmark_result(),
        )

        report = build_summary_report([outcome])

        for column in (
            "Model",
            "Processing Time",
            "Character",
            "Word",
            "Line",
            "Heading",
            "Missing Lines",
            "Additional Lines",
            "Rank",
        ):
            assert column in report

    def test_skipped_models_listed_separately_and_not_ranked(self):
        ran = ModelBenchmarkOutcome(
            model_name="qwen2.5vl:3b",
            succeeded=True,
            batch_result=_make_batch_result(
                [DocumentBatchRecord(pdf_path="a.pdf", success=True, processing_time=1.0, page_count=1)]
            ),
            benchmark_result=_make_benchmark_result(),
        )
        skipped = ModelBenchmarkOutcome(
            model_name="qwen2.5vl:99b",
            succeeded=False,
            skip_reason="Model 'qwen2.5vl:99b' is not installed in Ollama",
        )

        report = build_summary_report([ran, skipped])

        assert "Skipped Models" in report
        assert "qwen2.5vl:99b" in report
        assert "not installed" in report
        # Rank 1 belongs to the one model that actually ran.
        assert "| qwen2.5vl:3b | 12.50s | 90.00% | 90.00% | 90.00% | 90.00% | 1 | 2 | 1 |" in report

    def test_empty_outcomes_list_produces_report_with_no_rows(self):
        report = build_summary_report([])
        assert "# Vision Model Benchmark Summary" in report


class TestRunVisionModelBenchmarkCli:
    def test_restores_markdown_dir_after_run(self, tmp_path):
        markdown_dir = tmp_path / "markdown"
        markdown_dir.mkdir()
        (markdown_dir / "existing.md").write_text("pre-existing content", encoding="utf-8")

        ollama_client = MagicMock(spec=OllamaClient)
        ollama_client.model_exists.return_value = False  # skip immediately, no real work
        batch_processor = MagicMock(spec=BatchProcessor)

        run_vision_model_benchmark_cli(
            models=["some-model:1b"],
            batch_processor=batch_processor,
            ollama_client=ollama_client,
            markdown_dir=markdown_dir,
            reference_dir=tmp_path / "reference",
            benchmark_models_dir=tmp_path / "benchmark_models",
            summary_report_path=tmp_path / "benchmark_models" / "summary.md",
        )

        assert (markdown_dir / "existing.md").read_text(encoding="utf-8") == "pre-existing content"

    def test_restores_markdown_dir_even_if_a_model_run_raises(self, tmp_path):
        markdown_dir = tmp_path / "markdown"
        markdown_dir.mkdir()
        (markdown_dir / "existing.md").write_text("pre-existing content", encoding="utf-8")

        ollama_client = MagicMock(spec=OllamaClient)
        ollama_client.model_exists.return_value = True
        batch_processor = MagicMock(spec=BatchProcessor)
        batch_processor.process_batch.side_effect = RuntimeError("Ollama connection lost")

        with pytest.raises(RuntimeError):
            run_vision_model_benchmark_cli(
                models=["some-model:1b"],
                batch_processor=batch_processor,
                ollama_client=ollama_client,
                markdown_dir=markdown_dir,
                reference_dir=tmp_path / "reference",
                benchmark_models_dir=tmp_path / "benchmark_models",
                summary_report_path=tmp_path / "benchmark_models" / "summary.md",
            )

        assert (markdown_dir / "existing.md").read_text(encoding="utf-8") == "pre-existing content"

    def test_writes_summary_report_file(self, tmp_path):
        ollama_client = MagicMock(spec=OllamaClient)
        ollama_client.model_exists.return_value = False
        batch_processor = MagicMock(spec=BatchProcessor)
        summary_path = tmp_path / "benchmark_models" / "summary.md"

        run_vision_model_benchmark_cli(
            models=["some-model:1b"],
            batch_processor=batch_processor,
            ollama_client=ollama_client,
            markdown_dir=tmp_path / "markdown",
            reference_dir=tmp_path / "reference",
            benchmark_models_dir=tmp_path / "benchmark_models",
            summary_report_path=summary_path,
        )

        assert summary_path.is_file()
        assert "Vision Model Benchmark Summary" in summary_path.read_text(encoding="utf-8")

    def test_returns_one_outcome_per_model(self, tmp_path):
        ollama_client = MagicMock(spec=OllamaClient)
        ollama_client.model_exists.return_value = False
        batch_processor = MagicMock(spec=BatchProcessor)

        outcomes = run_vision_model_benchmark_cli(
            models=["model-a:1b", "model-b:1b"],
            batch_processor=batch_processor,
            ollama_client=ollama_client,
            markdown_dir=tmp_path / "markdown",
            reference_dir=tmp_path / "reference",
            benchmark_models_dir=tmp_path / "benchmark_models",
            summary_report_path=tmp_path / "benchmark_models" / "summary.md",
        )

        assert [outcome.model_name for outcome in outcomes] == ["model-a:1b", "model-b:1b"]
