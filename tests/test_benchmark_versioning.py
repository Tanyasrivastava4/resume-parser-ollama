"""
tests/test_benchmark_versioning.py

Tests for benchmark/versioning.py (Milestone 12.6).

Every test points BenchmarkVersionManager at a temporary directory tree
(via constructor injection), so none of this touches the project's real
benchmark/ folder or its existing history. DocumentExtractionResult-style
records are built directly rather than running a real batch, since
versioning only needs BatchProcessingResult's shape, not real extraction.
"""

import json

import pytest

from benchmark.versioning import (
    BenchmarkRun,
    BenchmarkVersionManager,
    OBSERVATIONS_TEMPLATE,
    REVIEW_CSV_HEADER,
)
from services.batch_processor import BatchProcessingResult, DocumentBatchRecord
from services.prompt_manager import PromptManager


@pytest.fixture
def prompts_dir(tmp_path):
    root = tmp_path / "prompts"
    resume_dir = root / "resume"
    resume_dir.mkdir(parents=True)
    (resume_dir / "extract_markdown.md").write_text(
        "Extract this resume as markdown.", encoding="utf-8"
    )
    return root


@pytest.fixture
def resumes_dir(tmp_path):
    root = tmp_path / "input_resumes"
    root.mkdir()
    (root / "alice.pdf").touch()
    (root / "bob.pdf").touch()
    return root


@pytest.fixture
def manager(tmp_path, prompts_dir, resumes_dir):
    return BenchmarkVersionManager(
        markdown_history_dir=tmp_path / "history" / "markdown",
        latest_dir=tmp_path / "latest",
        resumes_dir=resumes_dir,
        prompt_manager=PromptManager(prompts_dir=prompts_dir),
    )


def _fake_batch_result(markdown_dir) -> BatchProcessingResult:
    """Build a BatchProcessingResult with one success and one failure,
    including a real markdown file on disk for the success case so
    collect_markdown_outputs has something to copy."""
    markdown_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = markdown_dir / "alice.md"
    markdown_path.write_text("# Alice\n\nExtracted content.", encoding="utf-8")

    records = [
        DocumentBatchRecord(
            pdf_path="input/resumes/alice.pdf",
            success=True,
            processing_time=1.23,
            page_count=1,
            markdown_path=str(markdown_path),
        ),
        DocumentBatchRecord(
            pdf_path="input/resumes/bob.pdf",
            success=False,
            processing_time=0.05,
            error="boom",
        ),
    ]
    return BatchProcessingResult(
        total_documents=2,
        successful_documents=1,
        failed_documents=1,
        total_processing_time=1.28,
        average_processing_time=0.64,
        list_of_failures=[],
        results=records,
    )


class TestBenchmarkIdGeneration:
    def test_first_run_is_v1_001(self, manager):
        run = manager.prepare_benchmark_run("qwen2.5vl:3b", "resume/extract_markdown")

        assert run.benchmark_id == "MD_V1_001"
        assert run.version == 1
        assert run.sequence == 1

    def test_second_run_with_same_prompt_increments_sequence(self, manager):
        first = manager.prepare_benchmark_run("qwen2.5vl:3b", "resume/extract_markdown")
        second = manager.prepare_benchmark_run("qwen2.5vl:3b", "resume/extract_markdown")

        assert first.benchmark_id == "MD_V1_001"
        assert second.benchmark_id == "MD_V1_002"

    def test_third_run_continues_the_sequence(self, manager):
        manager.prepare_benchmark_run("qwen2.5vl:3b", "resume/extract_markdown")
        manager.prepare_benchmark_run("qwen2.5vl:3b", "resume/extract_markdown")
        third = manager.prepare_benchmark_run("qwen2.5vl:3b", "resume/extract_markdown")

        assert third.benchmark_id == "MD_V1_003"

    def test_prompt_change_bumps_version_and_resets_sequence(
        self, manager, prompts_dir
    ):
        manager.prepare_benchmark_run("qwen2.5vl:3b", "resume/extract_markdown")
        manager.prepare_benchmark_run("qwen2.5vl:3b", "resume/extract_markdown")

        (prompts_dir / "resume" / "extract_markdown.md").write_text(
            "A completely different prompt.", encoding="utf-8"
        )
        manager.prompt_manager.clear_cache()

        third = manager.prepare_benchmark_run("qwen2.5vl:3b", "resume/extract_markdown")

        assert third.benchmark_id == "MD_V2_001"

    def test_never_overwrites_an_existing_benchmark_folder(self, manager):
        run = manager.prepare_benchmark_run("qwen2.5vl:3b", "resume/extract_markdown")

        with pytest.raises(FileExistsError):
            run.folder.mkdir(parents=True, exist_ok=False)


class TestBenchmarkFolderCreation:
    def test_folder_contains_all_required_files(self, manager):
        run = manager.prepare_benchmark_run("qwen2.5vl:3b", "resume/extract_markdown")

        assert run.folder.is_dir()
        for filename in (
            "benchmark_config.json",
            "prompt.md",
            "settings_snapshot.json",
            "review.csv",
            "observations.md",
        ):
            assert (run.folder / filename).is_file(), filename
        assert (run.folder / "markdown").is_dir()

    def test_files_exist_before_any_batch_result_is_known(self, manager):
        """Robustness requirement: these files must exist even if the
        benchmark crashes right after this call, before results.csv or
        summary.json (written later by run_benchmark()) ever appear."""
        run = manager.prepare_benchmark_run("qwen2.5vl:3b", "resume/extract_markdown")

        assert not (run.folder / "results.csv").exists()
        assert not (run.folder / "summary.json").exists()
        assert (run.folder / "benchmark_config.json").is_file()
        assert (run.folder / "prompt.md").is_file()
        assert (run.folder / "settings_snapshot.json").is_file()


class TestBenchmarkConfig:
    def test_contains_required_fields(self, manager):
        run = manager.prepare_benchmark_run("qwen2.5vl:3b", "resume/extract_markdown")
        config = json.loads((run.folder / "benchmark_config.json").read_text())

        assert config["benchmark_id"] == "MD_V1_001"
        assert config["benchmark_version"] == "V1"
        assert "timestamp" in config
        assert config["dataset"]["size"] == 2
        assert config["model"]["name"] == "qwen2.5vl:3b"
        assert config["model"]["context"] == 8192
        assert config["prompt"]["id"] == "resume/extract_markdown"
        assert config["prompt"]["version"] == "V1"
        assert config["pdf"]["dpi"] == 200
        assert config["ollama"]["timeout"] == 900.0
        assert config["ollama"]["max_retries"] == 3
        assert config["ollama"]["retry_backoff_seconds"] == 5.0
        machine = config["machine"]
        for key in (
            "hostname",
            "operating_system",
            "processor",
            "architecture",
            "python_version",
        ):
            assert machine[key]


class TestPromptCopy:
    def test_prompt_md_contains_the_exact_prompt_text(self, manager, prompts_dir):
        run = manager.prepare_benchmark_run("qwen2.5vl:3b", "resume/extract_markdown")

        expected = (prompts_dir / "resume" / "extract_markdown.md").read_text()
        assert (run.folder / "prompt.md").read_text() == expected

    def test_later_prompt_edits_do_not_change_historical_copies(
        self, manager, prompts_dir
    ):
        first = manager.prepare_benchmark_run("qwen2.5vl:3b", "resume/extract_markdown")
        original_text = (first.folder / "prompt.md").read_text()

        (prompts_dir / "resume" / "extract_markdown.md").write_text(
            "Edited after the fact.", encoding="utf-8"
        )

        assert (first.folder / "prompt.md").read_text() == original_text


class TestSettingsSnapshot:
    def test_contains_benchmark_related_settings_only(self, manager):
        run = manager.prepare_benchmark_run("qwen2.5vl:3b", "resume/extract_markdown")
        snapshot = json.loads((run.folder / "settings_snapshot.json").read_text())

        assert snapshot == {
            "PDF_RENDER_DPI": 200,
            "OLLAMA_HOST": "http://localhost:11434",
            "OLLAMA_TIMEOUT": 900.0,
            "OLLAMA_MAX_RETRIES": 3,
            "OLLAMA_RETRY_BACKOFF_SECONDS": 5.0,
            "OLLAMA_NUM_CTX": 8192,
        }


class TestReviewTemplate:
    def test_review_csv_has_header_only(self, manager):
        run = manager.prepare_benchmark_run("qwen2.5vl:3b", "resume/extract_markdown")

        assert (run.folder / "review.csv").read_text() == REVIEW_CSV_HEADER


class TestObservationsTemplate:
    def test_observations_md_has_empty_sections(self, manager):
        run = manager.prepare_benchmark_run("qwen2.5vl:3b", "resume/extract_markdown")

        assert (run.folder / "observations.md").read_text() == OBSERVATIONS_TEMPLATE
        for heading in (
            "## Summary",
            "## Resume Reviews",
            "## General Findings",
            "## Conclusions",
        ):
            assert heading in OBSERVATIONS_TEMPLATE


class TestMarkdownCopy:
    def test_copies_only_successful_markdown_files(self, manager, tmp_path):
        run = manager.prepare_benchmark_run("qwen2.5vl:3b", "resume/extract_markdown")
        result = _fake_batch_result(tmp_path / "generated_output" / "markdown")

        manager.collect_markdown_outputs(run, result)

        copied = sorted(p.name for p in (run.folder / "markdown").iterdir())
        assert copied == ["alice.md"]
        assert (run.folder / "markdown" / "alice.md").read_text() == (
            "# Alice\n\nExtracted content."
        )


class TestLatestFolder:
    def test_sync_latest_mirrors_the_run_folder(self, manager, tmp_path):
        run = manager.prepare_benchmark_run("qwen2.5vl:3b", "resume/extract_markdown")
        result = _fake_batch_result(tmp_path / "generated_output" / "markdown")
        manager.collect_markdown_outputs(run, result)
        (run.folder / "results.csv").write_text("resume_name\nalice\n", encoding="utf-8")
        (run.folder / "summary.json").write_text('{"total": 2}', encoding="utf-8")

        manager.sync_latest(run)

        for filename in (
            "benchmark_config.json",
            "prompt.md",
            "settings_snapshot.json",
            "review.csv",
            "observations.md",
            "results.csv",
            "summary.json",
        ):
            assert (manager.latest_dir / filename).read_text() == (
                run.folder / filename
            ).read_text()
        assert (manager.latest_dir / "markdown" / "alice.md").is_file()

    def test_sync_latest_replaces_previous_contents(self, manager, tmp_path):
        first = manager.prepare_benchmark_run("qwen2.5vl:3b", "resume/extract_markdown")
        manager.sync_latest(first)
        (manager.latest_dir / "stale_file_from_previous_run.txt").write_text("old")

        second = manager.prepare_benchmark_run("qwen2.5vl:3b", "resume/extract_markdown")
        manager.sync_latest(second)

        assert not (manager.latest_dir / "stale_file_from_previous_run.txt").exists()
        assert json.loads((manager.latest_dir / "benchmark_config.json").read_text())[
            "benchmark_id"
        ] == "MD_V1_002"


class TestBackwardCompatibility:
    def test_benchmark_run_dataclass_fields_are_stable(self):
        run = BenchmarkRun(benchmark_id="MD_V1_001", version=1, sequence=1, folder=None)

        assert run.benchmark_id == "MD_V1_001"
        assert run.version_label == "V1"

    def test_batch_processing_result_without_results_field_still_works(self, manager):
        """BatchProcessingResult.results defaults to [] (Milestone 12
        added it after BatchProcessor already existed), so versioning
        code must not assume it's always populated."""
        run = manager.prepare_benchmark_run("qwen2.5vl:3b", "resume/extract_markdown")
        empty_result = BatchProcessingResult(
            total_documents=0,
            successful_documents=0,
            failed_documents=0,
            total_processing_time=0.0,
            average_processing_time=0.0,
        )

        manager.collect_markdown_outputs(run, empty_result)

        assert list((run.folder / "markdown").iterdir()) == []
