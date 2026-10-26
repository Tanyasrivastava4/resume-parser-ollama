"""
tests/test_batch_processor.py

Tests for services/batch_processor.py (Milestone 11).

DocumentExtractor is mocked in every unit test, since batch processing
logic (finding PDFs, continuing past failures, aggregating results) is
independent of what the extractor actually does. One integration test at
the bottom runs a real DocumentExtractor (real PDFLoader/PromptManager,
real Ollama) over a small real batch, and skips itself if Ollama or the
expected model isn't available.
"""

import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock

import fitz
import pytest

from config import BASE_DIR
from services.batch_processor import BatchFailure, BatchProcessor, DocumentBatchRecord
from services.document_extractor import DocumentExtractionResult, DocumentExtractor
from services.ollama_client import OllamaClient
from services.prompt_manager import PromptManager

INTEGRATION_VISION_MODEL = "qwen2.5vl:3b"


def _make_pdf(path, page_count: int = 1) -> None:
    document = fitz.open()
    for _ in range(page_count):
        document.new_page()
    document.save(path)
    document.close()


def _make_pdf_with_text(path, page_count: int = 1) -> None:
    """Like _make_pdf, but with real (non-blank) text content. Needed
    wherever a real DocumentExtractor (not a mock) processes the PDF:
    since Milestone 15B.5, a genuinely blank page (what _make_pdf
    produces) is detected and skipped before reaching the vision
    model, which is the wrong fixture for a test whose premise is a
    normal, successful extraction. Twenty lines is deliberately
    generous — see tests/test_document_extractor.py's
    _make_pdf_with_text docstring for why a single short line measured
    below the blank-detection threshold at this project's real render
    DPI and isn't safe to rely on."""
    document = fitz.open()
    for page_number in range(page_count):
        page = document.new_page()
        lines = [
            f"Sample resume content, page {page_number + 1}, line {line_number}."
            for line_number in range(20)
        ]
        page.insert_text((72, 72), "\n".join(lines), fontsize=11)
    document.save(path)
    document.close()


def _fake_result(document_name: str) -> DocumentExtractionResult:
    return DocumentExtractionResult(
        markdown_path=f"output/markdown/{document_name}.md",
        response=SimpleNamespace(response="# fake"),
        processing_time=0.01,
        page_count=1,
        image_paths=[f"output/images/{document_name}/page_001.png"],
    )


@pytest.fixture
def mock_extractor():
    return MagicMock(spec=DocumentExtractor)


class TestFindAndProcess:
    def test_empty_input_dir_returns_zero_results(self, tmp_path, mock_extractor):
        processor = BatchProcessor(document_extractor=mock_extractor, input_dir=tmp_path)

        result = processor.process_batch(model_name="qwen2.5vl:3b", prompt_id="describe")

        assert result.total_documents == 0
        assert result.successful_documents == 0
        assert result.failed_documents == 0
        assert result.average_processing_time == 0.0
        assert result.list_of_failures == []
        mock_extractor.extract.assert_not_called()

    def test_nonexistent_input_dir_returns_zero_results(self, tmp_path, mock_extractor):
        processor = BatchProcessor(
            document_extractor=mock_extractor, input_dir=tmp_path / "does_not_exist"
        )

        result = processor.process_batch(model_name="qwen2.5vl:3b", prompt_id="describe")

        assert result.total_documents == 0
        assert result.list_of_failures == []

    def test_finds_pdfs_in_nested_folders(self, tmp_path, mock_extractor):
        (tmp_path / "a.pdf").touch()
        nested = tmp_path / "batch_2"
        nested.mkdir()
        (nested / "b.pdf").touch()
        (tmp_path / "not_a_pdf.txt").touch()
        mock_extractor.extract.return_value = _fake_result("x")

        processor = BatchProcessor(document_extractor=mock_extractor, input_dir=tmp_path)
        result = processor.process_batch(model_name="qwen2.5vl:3b", prompt_id="describe")

        assert result.total_documents == 2
        assert mock_extractor.extract.call_count == 2

    def test_all_documents_succeed(self, tmp_path, mock_extractor):
        pdf_paths = [tmp_path / f"{name}.pdf" for name in ("a", "b", "c")]
        for pdf_path in pdf_paths:
            pdf_path.touch()
        mock_extractor.extract.return_value = _fake_result("shared")

        processor = BatchProcessor(document_extractor=mock_extractor, input_dir=tmp_path)
        result = processor.process_batch(model_name="qwen2.5vl:3b", prompt_id="describe")

        assert result.total_documents == 3
        assert result.successful_documents == 3
        assert result.failed_documents == 0
        assert result.list_of_failures == []
        assert result.average_processing_time == pytest.approx(
            result.total_processing_time / 3
        )
        assert len(result.results) == 3
        assert all(record.success for record in result.results)
        assert [record.page_count for record in result.results] == [1, 1, 1]
        assert [record.markdown_path for record in result.results] == [
            "output/markdown/shared.md"
        ] * 3

    def test_continues_and_records_failure_after_one_document_fails(
        self, tmp_path, mock_extractor
    ):
        for name in ("a", "b", "c"):
            (tmp_path / f"{name}.pdf").touch()
        mock_extractor.extract.side_effect = [
            RuntimeError("boom"),
            _fake_result("b"),
            _fake_result("c"),
        ]

        processor = BatchProcessor(document_extractor=mock_extractor, input_dir=tmp_path)
        result = processor.process_batch(model_name="qwen2.5vl:3b", prompt_id="describe")

        assert result.total_documents == 3
        assert result.successful_documents == 2
        assert result.failed_documents == 1
        assert len(result.list_of_failures) == 1
        assert result.list_of_failures[0] == BatchFailure(
            pdf_path=str(tmp_path / "a.pdf"), error="boom"
        )
        assert mock_extractor.extract.call_count == 3

        assert len(result.results) == 3
        failed_record, *successful_records = result.results
        assert failed_record == DocumentBatchRecord(
            pdf_path=str(tmp_path / "a.pdf"),
            success=False,
            processing_time=failed_record.processing_time,
            error="boom",
        )
        assert all(record.success for record in successful_records)

    def test_calls_extractor_with_correct_arguments_per_pdf(self, tmp_path, mock_extractor):
        pdf_path = tmp_path / "only.pdf"
        pdf_path.touch()
        mock_extractor.extract.return_value = _fake_result("only")

        processor = BatchProcessor(document_extractor=mock_extractor, input_dir=tmp_path)
        processor.process_batch(model_name="qwen2.5vl:3b", prompt_id="resume/extract_markdown")

        mock_extractor.extract.assert_called_once_with(
            pdf_path, model_name="qwen2.5vl:3b", prompt_id="resume/extract_markdown"
        )


@pytest.mark.integration
def test_real_batch_processing_integration(tmp_path):
    """Optional integration test with a real DocumentExtractor (real
    PDFLoader, PromptManager, and Ollama). Skips itself if Ollama or the
    expected vision model isn't available."""
    ollama_client = OllamaClient()

    if not ollama_client.is_running():
        pytest.skip("Ollama is not running locally; skipping integration test.")
    if not ollama_client.model_exists(INTEGRATION_VISION_MODEL):
        pytest.skip(f"Model '{INTEGRATION_VISION_MODEL}' is not installed; skipping.")

    prompts_root = tmp_path / "prompts"
    prompts_root.mkdir()
    (prompts_root / "describe.md").write_text(
        "Describe these pages as markdown.", encoding="utf-8"
    )

    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir()
    _make_pdf_with_text(resumes_dir / "batch_resume_one.pdf")
    _make_pdf_with_text(resumes_dir / "batch_resume_two.pdf")

    extractor = DocumentExtractor(
        ollama_client=ollama_client,
        prompt_manager=PromptManager(prompts_dir=prompts_root),
    )
    processor = BatchProcessor(document_extractor=extractor, input_dir=resumes_dir)

    try:
        result = processor.process_batch(
            model_name=INTEGRATION_VISION_MODEL, prompt_id="describe"
        )

        assert result.total_documents == 2
        assert result.successful_documents == 2
        assert result.failed_documents == 0
        for name in ("batch_resume_one", "batch_resume_two"):
            assert (BASE_DIR / "output" / "markdown" / f"{name}.md").is_file()
    finally:
        for name in ("batch_resume_one", "batch_resume_two"):
            (BASE_DIR / "output" / "markdown" / f"{name}.md").unlink(missing_ok=True)
            shutil.rmtree(BASE_DIR / "output" / "images" / name, ignore_errors=True)
