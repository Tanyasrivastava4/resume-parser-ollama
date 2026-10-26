"""
tests/test_document_extractor.py

Tests for services/document_extractor.py (Milestone 10, refactored to send
one Ollama request per page instead of one request for the whole PDF —
see the production incident: sending every page in a single request
exceeded the model's context window on multi-page resumes; blank-page
detection added in Milestone 15B.5).

PDFLoader and PromptManager run for real against temporary files (they're
cheap, deterministic, local operations), so these tests exercise the real
page-rendering and prompt-loading paths. Only OllamaClient is mocked in
the unit tests, since it depends on a running local model server.

Milestone 15B.5 note: `_make_pdf()` below creates genuinely blank pages
(fitz's `new_page()` with no content) — before this milestone that
didn't matter, since every page was sent to the (mocked) model
regardless of its actual content. Now that blank pages are detected
and skipped for real, any test whose premise requires `generate_vision`
to actually be called must use `_make_pdf_with_text()` instead, so its
rendered pages contain real ink and aren't classified Blank. `_make_pdf()`
itself is kept for the tests that specifically exercise blank-page
behavior.

One integration test at the bottom talks to a real local Ollama server
with a real vision model, over a real multi-page PDF, to confirm the
per-page request / merge behavior end to end. It skips itself (rather
than failing) if Ollama or the expected model isn't available.
"""

import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import fitz
import ollama
import pytest

from config import BASE_DIR, OUTPUT_DIR
from services.document_extractor import (
    AllPagesBlankError,
    DocumentExtractionResult,
    DocumentExtractor,
    UnsupportedOCRStrategyError,
)
from services.ollama_client import ModelNotFoundError, OllamaClient
from services.pdf_loader import InvalidPDFError, PDFNotFoundError
from services.prompt_manager import PromptManager, PromptNotFoundError

INTEGRATION_VISION_MODEL = "qwen2.5vl:3b"


def _make_pdf(path, page_count: int) -> None:
    """Create a minimal valid PDF with `page_count` blank pages at `path`.
    Pages rendered from this PDF are classified Blank by
    services/page_quality_classifier.py (Milestone 15B.5) — use this
    helper only for tests that specifically want that."""
    document = fitz.open()
    for _ in range(page_count):
        document.new_page()
    document.save(path)
    document.close()


def _make_pdf_with_text(path, page_count: int) -> None:
    """Create a minimal valid PDF with `page_count` pages, each
    containing real, non-blank text content, so pages rendered from it
    are NOT classified Blank by services/page_quality_classifier.py
    (Milestone 15B.5) and reach generate_vision() like every page did
    before that milestone.

    A single short line was tried and rejected here: at this project's
    real render DPI (services/pdf_loader.py, settings.PDF_RENDER_DPI),
    one line of inserted text produced an ink ratio of 0.00097 —
    *below* BLANK_INK_RATIO_THRESHOLD (0.001), and so was itself
    misclassified Blank. Twenty lines reliably measures ~0.04 (40x the
    threshold), comfortably inside the "content-bearing" range this
    project's own real sparse/normal pages occupy (0.009-0.28, per
    docs/HALLUCINATION_INVESTIGATION.md's calibration data)."""
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


def _content_text(label: str) -> str:
    """Twenty lines of text — see _make_pdf_with_text's docstring for
    why a single short line isn't enough to reliably read as
    non-blank at this project's real render DPI."""
    return "\n".join(f"{label}, line {line_number}." for line_number in range(20))


def _fake_page_response(**kwargs) -> SimpleNamespace:
    """Build a stand-in per-page response whose text identifies which
    image it was generated for, so merge order can be verified."""
    image_path = Path(kwargs["image_paths"][0])
    return SimpleNamespace(
        model=kwargs["model"],
        response=f"Content for {image_path.stem}",
        done=True,
    )


@pytest.fixture
def prompts_dir(tmp_path):
    """A temporary prompts/ tree with one variable-free prompt, so tests
    never touch the project's real prompts/ directory."""
    prompts_root = tmp_path / "prompts"
    prompts_root.mkdir()
    (prompts_root / "describe.md").write_text(
        "Describe these pages as markdown.", encoding="utf-8"
    )
    return prompts_root


@pytest.fixture
def output_cleanup():
    """Track document names whose output/images/, output/markdown/, and
    output/temp/ artifacts should be removed after the test."""
    document_names = []
    yield document_names
    for document_name in document_names:
        shutil.rmtree(OUTPUT_DIR / "images" / document_name, ignore_errors=True)
        shutil.rmtree(OUTPUT_DIR / "temp" / document_name, ignore_errors=True)
        (OUTPUT_DIR / "markdown" / f"{document_name}.md").unlink(missing_ok=True)


@pytest.fixture
def mock_ollama_client():
    client = MagicMock(spec=OllamaClient)
    client.generate_vision.side_effect = _fake_page_response
    return client


@pytest.fixture
def extractor(mock_ollama_client, prompts_dir):
    return DocumentExtractor(
        ollama_client=mock_ollama_client,
        prompt_manager=PromptManager(prompts_dir=prompts_dir),
    )


class TestExtract:
    def test_returns_extraction_result_with_expected_fields(
        self, tmp_path, extractor, output_cleanup
    ):
        document_name = "test_resume_multi_page"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=2)

        result = extractor.extract(
            pdf_path, model_name="qwen2.5vl:3b", prompt_id="describe"
        )

        assert isinstance(result, DocumentExtractionResult)
        assert result.markdown_path == f"output/markdown/{document_name}.md"
        assert result.page_count == 2
        assert result.processing_time >= 0
        assert result.image_paths == [
            f"output/images/{document_name}/page_001.png",
            f"output/images/{document_name}/page_002.png",
        ]

    def test_response_contains_full_merged_text_like_before_the_refactor(
        self, tmp_path, extractor, output_cleanup
    ):
        """result.response.response must still equal the complete
        document's markdown, so callers that read it don't need to
        change even though it's now built from several page requests."""
        document_name = "test_resume_response_contract"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=2)

        result = extractor.extract(
            pdf_path, model_name="qwen2.5vl:3b", prompt_id="describe"
        )

        saved_text = (BASE_DIR / result.markdown_path).read_text(encoding="utf-8")
        assert result.response.response == saved_text
        assert "page_001" in result.response.response
        assert "page_002" in result.response.response

    def test_merges_pages_in_order(self, tmp_path, extractor, output_cleanup):
        document_name = "test_resume_merge_order"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=3)

        result = extractor.extract(
            pdf_path, model_name="qwen2.5vl:3b", prompt_id="describe"
        )

        saved_text = (BASE_DIR / result.markdown_path).read_text(encoding="utf-8")
        assert saved_text == (
            "Content for page_001\n\nContent for page_002\n\nContent for page_003"
        )

    def test_sends_exactly_one_page_image_per_request(
        self, tmp_path, extractor, mock_ollama_client, output_cleanup
    ):
        document_name = "test_resume_one_request_per_page"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=3)

        extractor.extract(pdf_path, model_name="qwen2.5vl:3b", prompt_id="describe")

        assert mock_ollama_client.generate_vision.call_count == 3
        for call in mock_ollama_client.generate_vision.call_args_list:
            assert len(call.kwargs["image_paths"]) == 1
            assert call.kwargs["image_paths"][0].is_file()

    def test_first_page_prompt_is_the_base_prompt_unchanged(
        self, tmp_path, extractor, mock_ollama_client, output_cleanup
    ):
        """Milestone 23.13 changed this test's original assertion (that
        every page shares one identical prompt) — that stopped being
        true by design once cross-page context exists for page 2
        onward (see TestCrossPageContext below). What must still hold,
        and is asserted here, is that the FIRST page's prompt is
        exactly the base template, with no context block appended,
        since there is no preceding page for it."""
        document_name = "test_resume_same_prompt"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=3)

        extractor.extract(pdf_path, model_name="qwen2.5vl:3b", prompt_id="describe")

        first_call_prompt = mock_ollama_client.generate_vision.call_args_list[0].kwargs["prompt"]
        assert first_call_prompt == "Describe these pages as markdown."

    def test_temp_page_files_are_cleaned_up(self, tmp_path, extractor, output_cleanup):
        document_name = "test_resume_temp_cleanup"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=2)

        extractor.extract(pdf_path, model_name="qwen2.5vl:3b", prompt_id="describe")

        assert not (OUTPUT_DIR / "temp" / document_name).exists()

    def test_temp_page_files_are_cleaned_up_even_on_failure(
        self, tmp_path, extractor, mock_ollama_client, output_cleanup
    ):
        document_name = "test_resume_temp_cleanup_on_failure"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=2)
        mock_ollama_client.generate_vision.side_effect = [
            _fake_page_response(model="qwen2.5vl:3b", image_paths=[Path("page_001.png")]),
            RuntimeError("model exploded on page 2"),
        ]

        with pytest.raises(RuntimeError):
            extractor.extract(pdf_path, model_name="qwen2.5vl:3b", prompt_id="describe")

        assert not (OUTPUT_DIR / "temp" / document_name).exists()

    def test_missing_pdf_raises_pdf_not_found(self, tmp_path, extractor):
        missing_path = tmp_path / "does_not_exist.pdf"

        with pytest.raises(PDFNotFoundError):
            extractor.extract(missing_path, model_name="qwen2.5vl:3b", prompt_id="describe")

    def test_missing_prompt_raises_prompt_not_found(
        self, tmp_path, extractor, output_cleanup
    ):
        document_name = "test_resume_missing_prompt"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf(pdf_path, page_count=1)

        with pytest.raises(PromptNotFoundError):
            extractor.extract(
                pdf_path, model_name="qwen2.5vl:3b", prompt_id="does_not_exist"
            )

    def test_model_not_found_propagates(
        self, tmp_path, extractor, mock_ollama_client, output_cleanup
    ):
        document_name = "test_resume_bad_model"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=1)
        mock_ollama_client.generate_vision.side_effect = ModelNotFoundError("nope")

        with pytest.raises(ModelNotFoundError):
            extractor.extract(
                pdf_path, model_name="does-not-exist:1b", prompt_id="describe"
            )

    def test_invalid_pdf_is_not_retried(self, mock_ollama_client, prompts_dir):
        """Milestone 12.5: retry logic is scoped to OllamaConnectionError
        only. A PDFLoader failure must propagate immediately, with the
        loader called exactly once."""
        pdf_loader_calls = []

        def failing_pdf_loader(path):
            pdf_loader_calls.append(path)
            raise InvalidPDFError("not a real pdf")

        extractor = DocumentExtractor(
            ollama_client=mock_ollama_client,
            prompt_manager=PromptManager(prompts_dir=prompts_dir),
            pdf_loader=failing_pdf_loader,
        )

        with pytest.raises(InvalidPDFError):
            extractor.extract("whatever.pdf", model_name="qwen2.5vl:3b", prompt_id="describe")

        assert len(pdf_loader_calls) == 1
        mock_ollama_client.generate_vision.assert_not_called()

    def test_retry_count_defaults_to_zero_when_ollama_client_does_not_track_it(
        self, tmp_path, extractor, output_cleanup
    ):
        """OllamaClient exposes `last_retry_count` as a plain attribute,
        not part of its formal public API, so DocumentExtractor must
        tolerate a client that doesn't have it (e.g. a minimal test
        double) by defaulting retry_count to 0."""
        document_name = "test_resume_retry_count_default"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=1)

        result = extractor.extract(
            pdf_path, model_name="qwen2.5vl:3b", prompt_id="describe"
        )

        assert result.retry_count == 0

    def test_retry_count_sums_across_pages(
        self, tmp_path, mock_ollama_client, prompts_dir, output_cleanup
    ):
        document_name = "test_resume_retry_count_summed"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=2)

        # Simulate OllamaClient.last_retry_count changing per call, as the
        # real client would after retrying a transient failure.
        retry_counts = iter([2, 1])

        def generate_with_retry_count(**kwargs):
            mock_ollama_client.last_retry_count = next(retry_counts)
            return _fake_page_response(**kwargs)

        mock_ollama_client.generate_vision.side_effect = generate_with_retry_count

        extractor = DocumentExtractor(
            ollama_client=mock_ollama_client,
            prompt_manager=PromptManager(prompts_dir=prompts_dir),
        )
        result = extractor.extract(
            pdf_path, model_name="qwen2.5vl:3b", prompt_id="describe"
        )

        assert result.retry_count == 3


class TestBlankPageHandling:
    """Milestone 15B.5: a page classified Blank by
    services/page_quality_classifier.py must never reach
    generate_vision(), and must contribute an empty string to that
    page's position in the merged markdown. Every non-blank page's
    handling must be completely unaffected."""

    def test_single_blank_page_is_skipped_and_contributes_empty_string(
        self, tmp_path, extractor, mock_ollama_client, output_cleanup
    ):
        document_name = "test_resume_single_blank_page"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf(pdf_path, page_count=1)  # genuinely blank page

        with pytest.raises(AllPagesBlankError):
            extractor.extract(pdf_path, model_name="qwen2.5vl:3b", prompt_id="describe")

        mock_ollama_client.generate_vision.assert_not_called()

    def test_mixed_blank_and_non_blank_pages_only_blank_pages_skipped(
        self, tmp_path, extractor, mock_ollama_client, output_cleanup
    ):
        document_name = "test_resume_mixed_blank_pages"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"

        document = fitz.open()
        page1 = document.new_page()
        page1.insert_text((72, 72), _content_text("Real content on page one"), fontsize=11)
        document.new_page()  # page two: genuinely blank
        page3 = document.new_page()
        page3.insert_text((72, 72), _content_text("Real content on page three"), fontsize=11)
        document.save(pdf_path)
        document.close()

        result = extractor.extract(
            pdf_path, model_name="qwen2.5vl:3b", prompt_id="describe"
        )

        # Only the two non-blank pages ever reached the model.
        assert mock_ollama_client.generate_vision.call_count == 2
        called_images = {
            Path(call.kwargs["image_paths"][0]).stem
            for call in mock_ollama_client.generate_vision.call_args_list
        }
        assert called_images == {"page_001", "page_003"}

        # The blank page's position contributes an empty segment, so
        # the merge shows two adjacent blank-line separators where
        # page two's content would otherwise be.
        saved_text = (BASE_DIR / result.markdown_path).read_text(encoding="utf-8")
        assert saved_text == "Content for page_001\n\n\n\nContent for page_003"

    def test_all_pages_blank_raises_all_pages_blank_error(
        self, tmp_path, extractor, mock_ollama_client, output_cleanup
    ):
        document_name = "test_resume_all_blank_pages"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf(pdf_path, page_count=3)

        with pytest.raises(AllPagesBlankError):
            extractor.extract(pdf_path, model_name="qwen2.5vl:3b", prompt_id="describe")

        mock_ollama_client.generate_vision.assert_not_called()

    def test_retry_count_only_counts_non_blank_pages(
        self, tmp_path, mock_ollama_client, prompts_dir, output_cleanup
    ):
        document_name = "test_resume_retry_count_with_blank_page"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"

        document = fitz.open()
        page1 = document.new_page()
        page1.insert_text((72, 72), _content_text("Real content on page one"), fontsize=11)
        document.new_page()  # blank
        document.save(pdf_path)
        document.close()

        mock_ollama_client.last_retry_count = 2
        extractor = DocumentExtractor(
            ollama_client=mock_ollama_client,
            prompt_manager=PromptManager(prompts_dir=prompts_dir),
        )

        result = extractor.extract(
            pdf_path, model_name="qwen2.5vl:3b", prompt_id="describe"
        )

        assert mock_ollama_client.generate_vision.call_count == 1
        assert result.retry_count == 2

    def test_logs_blank_page_detection_and_ocr_skip(
        self, tmp_path, extractor, output_cleanup, caplog
    ):
        document_name = "test_resume_blank_page_logging"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"

        document = fitz.open()
        page1 = document.new_page()
        page1.insert_text((72, 72), _content_text("Real content on page one"), fontsize=11)
        document.new_page()  # blank
        document.save(pdf_path)
        document.close()

        with caplog.at_level("INFO"):
            extractor.extract(pdf_path, model_name="qwen2.5vl:3b", prompt_id="describe")

        messages = [record.message for record in caplog.records]
        assert any("Detected blank page" in message for message in messages)
        assert any("Skipping OCR" in message for message in messages)

    def test_non_blank_pages_completely_unaffected_by_blank_page_check(
        self, tmp_path, extractor, mock_ollama_client, output_cleanup
    ):
        """Regression: a document with no blank pages at all must
        behave exactly as it did before Milestone 15B.5 — every page
        sent to the model, in order, with no change to call shape."""
        document_name = "test_resume_no_blank_pages_regression"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=3)

        result = extractor.extract(
            pdf_path, model_name="qwen2.5vl:3b", prompt_id="describe"
        )

        assert mock_ollama_client.generate_vision.call_count == 3
        saved_text = (BASE_DIR / result.markdown_path).read_text(encoding="utf-8")
        assert saved_text == (
            "Content for page_001\n\nContent for page_002\n\nContent for page_003"
        )
        assert result.page_count == 3


class TestSplitColumnsStrategy:
    """Milestone 15C.3: the opt-in "split_columns" OCR strategy. Default
    behavior (no strategy passed, or strategy="single_page") must stay
    byte-for-byte identical to every test above this class — these tests
    only exercise the new, explicitly-selected code path."""

    def test_unsupported_strategy_raises(self, tmp_path, extractor, output_cleanup):
        document_name = "test_resume_bad_strategy"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=1)

        with pytest.raises(UnsupportedOCRStrategyError):
            extractor.extract(
                pdf_path,
                model_name="qwen2.5vl:3b",
                prompt_id="describe",
                strategy="quadrants",
            )

    def test_sends_two_requests_per_page_left_then_right(
        self, tmp_path, extractor, mock_ollama_client, output_cleanup
    ):
        document_name = "test_resume_split_two_requests"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=2)

        extractor.extract(
            pdf_path,
            model_name="qwen2.5vl:3b",
            prompt_id="describe",
            strategy="split_columns",
        )

        assert mock_ollama_client.generate_vision.call_count == 4
        called_stems = [
            Path(call.kwargs["image_paths"][0]).stem
            for call in mock_ollama_client.generate_vision.call_args_list
        ]
        assert called_stems == [
            "page001_left",
            "page001_right",
            "page002_left",
            "page002_right",
        ]

    def test_merges_left_before_right_within_a_page(
        self, tmp_path, extractor, output_cleanup
    ):
        document_name = "test_resume_split_merge_order"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=1)

        result = extractor.extract(
            pdf_path,
            model_name="qwen2.5vl:3b",
            prompt_id="describe",
            strategy="split_columns",
        )

        saved_text = (BASE_DIR / result.markdown_path).read_text(encoding="utf-8")
        assert saved_text == "Content for page001_left\n\nContent for page001_right"

    def test_merges_pages_and_columns_in_order(
        self, tmp_path, extractor, output_cleanup
    ):
        document_name = "test_resume_split_multi_page_order"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=2)

        result = extractor.extract(
            pdf_path,
            model_name="qwen2.5vl:3b",
            prompt_id="describe",
            strategy="split_columns",
        )

        saved_text = (BASE_DIR / result.markdown_path).read_text(encoding="utf-8")
        assert saved_text == (
            "Content for page001_left\n\nContent for page001_right"
            "\n\nContent for page002_left\n\nContent for page002_right"
        )

    def test_crops_are_half_width_full_height_not_quadrants(
        self, tmp_path, extractor, mock_ollama_client, output_cleanup
    ):
        """Crop dimensions must be captured *during* the call, since the
        crop files live under output/temp/ and are deleted once
        extract() finishes (matching every other temp-file test in this
        module)."""
        document_name = "test_resume_split_crop_dimensions"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=1)

        crop_dimensions = []

        def capture_crop_dimensions(**kwargs):
            crop_path = kwargs["image_paths"][0]
            pixmap = fitz.Pixmap(str(crop_path))
            crop_dimensions.append((pixmap.width, pixmap.height))
            return _fake_page_response(**kwargs)

        mock_ollama_client.generate_vision.side_effect = capture_crop_dimensions

        extractor.extract(
            pdf_path,
            model_name="qwen2.5vl:3b",
            prompt_id="describe",
            strategy="split_columns",
        )

        original_image = fitz.Pixmap(
            str(BASE_DIR / f"output/images/{document_name}/page_001.png")
        )
        (left_width, left_height), (right_width, right_height) = crop_dimensions

        assert left_height == original_image.height
        assert right_height == original_image.height
        assert left_width == original_image.width // 2
        assert left_width + right_width == original_image.width

    def test_retry_count_sums_both_halves(
        self, tmp_path, mock_ollama_client, prompts_dir, output_cleanup
    ):
        document_name = "test_resume_split_retry_count"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=1)

        retry_counts = iter([2, 1])  # left, right

        def generate_with_retry_count(**kwargs):
            mock_ollama_client.last_retry_count = next(retry_counts)
            return _fake_page_response(**kwargs)

        mock_ollama_client.generate_vision.side_effect = generate_with_retry_count

        extractor = DocumentExtractor(
            ollama_client=mock_ollama_client,
            prompt_manager=PromptManager(prompts_dir=prompts_dir),
        )
        result = extractor.extract(
            pdf_path,
            model_name="qwen2.5vl:3b",
            prompt_id="describe",
            strategy="split_columns",
        )

        assert result.retry_count == 3

    def test_blank_pages_still_skipped_under_split_columns(
        self, tmp_path, extractor, mock_ollama_client, output_cleanup
    ):
        """A page classified Blank never reaches the model under either
        strategy — split_columns must not bypass the Milestone 15B.5
        blank-page check."""
        document_name = "test_resume_split_blank_page_skip"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"

        document = fitz.open()
        page1 = document.new_page()
        page1.insert_text((72, 72), _content_text("Real content on page one"), fontsize=11)
        document.new_page()  # blank
        document.save(pdf_path)
        document.close()

        result = extractor.extract(
            pdf_path,
            model_name="qwen2.5vl:3b",
            prompt_id="describe",
            strategy="split_columns",
        )

        # Only page one's two halves reach the model; page two is skipped whole.
        assert mock_ollama_client.generate_vision.call_count == 2
        saved_text = (BASE_DIR / result.markdown_path).read_text(encoding="utf-8")
        assert saved_text == "Content for page001_left\n\nContent for page001_right\n\n"

    def test_default_strategy_unaffected_when_split_columns_not_requested(
        self, tmp_path, extractor, mock_ollama_client, output_cleanup
    ):
        """Regression: not passing `strategy` at all must still behave
        exactly like the pre-15C.3 single request per page."""
        document_name = "test_resume_split_default_unaffected"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=2)

        extractor.extract(pdf_path, model_name="qwen2.5vl:3b", prompt_id="describe")

        assert mock_ollama_client.generate_vision.call_count == 2
        called_stems = [
            Path(call.kwargs["image_paths"][0]).stem
            for call in mock_ollama_client.generate_vision.call_args_list
        ]
        assert called_stems == ["page_001", "page_002"]

    def test_selectable_via_settings_default(
        self, tmp_path, extractor, mock_ollama_client, output_cleanup, monkeypatch
    ):
        """Milestone 15C.3 requirement: the strategy must be selectable
        through configuration (settings.OCR_STRATEGY), not only through
        the explicit `strategy` argument."""
        monkeypatch.setattr(
            "services.document_extractor.OCR_STRATEGY", "split_columns"
        )
        document_name = "test_resume_split_via_settings"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=1)

        extractor.extract(pdf_path, model_name="qwen2.5vl:3b", prompt_id="describe")

        assert mock_ollama_client.generate_vision.call_count == 2
        called_stems = [
            Path(call.kwargs["image_paths"][0]).stem
            for call in mock_ollama_client.generate_vision.call_args_list
        ]
        assert called_stems == ["page001_left", "page001_right"]


class TestCrossPageContext:
    """Milestone 23.13: bounded cross-page tail-context injection, for
    the confirmed deterministic failure where a section heading ends
    one page and its content begins the next with no heading of its
    own (docs/MILESTONE_23_11_WORK_EXPERIENCE_ROOT_CAUSE_INVESTIGATION.md,
    docs/MILESTONE_23_12_WORK_EXPERIENCE_MODEL_TRIAL_AND_OCR_ARCHITECTURE_INVESTIGATION.md).

    All tests here are unit-level, against a mocked OllamaClient — they
    prove the context-building/prompt-injection mechanism itself, NOT
    that the real vision model recovers the real "Earlier Experience"
    case. That is verified separately, live, against the real PDF, and
    reported in docs/MILESTONE_23_13_CROSS_PAGE_CONTEXT_RESULTS.md —
    not claimed here.
    """

    def test_page_two_receives_bounded_tail_context_from_page_one(
        self, tmp_path, extractor, mock_ollama_client, output_cleanup
    ):
        document_name = "test_resume_context_page_two"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=2)

        extractor.extract(pdf_path, model_name="qwen2.5vl:3b", prompt_id="describe")

        calls = mock_ollama_client.generate_vision.call_args_list
        second_call_prompt = calls[1].kwargs["prompt"]
        assert "Content for page_001" in second_call_prompt
        assert "Preceding Page Context" in second_call_prompt
        assert second_call_prompt.startswith("Describe these pages as markdown.")

    def test_first_page_receives_no_previous_page_context(
        self, tmp_path, extractor, mock_ollama_client, output_cleanup
    ):
        document_name = "test_resume_context_first_page"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=2)

        extractor.extract(pdf_path, model_name="qwen2.5vl:3b", prompt_id="describe")

        first_call_prompt = mock_ollama_client.generate_vision.call_args_list[0].kwargs["prompt"]
        assert "Preceding Page Context" not in first_call_prompt
        assert first_call_prompt == "Describe these pages as markdown."

    def test_context_is_truncated_to_the_configured_character_limit(
        self, tmp_path, mock_ollama_client, prompts_dir, output_cleanup
    ):
        from services.document_extractor import MAX_CROSS_PAGE_CONTEXT_CHARS

        document_name = "test_resume_context_truncated"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=2)

        long_text = "X" * (MAX_CROSS_PAGE_CONTEXT_CHARS * 3)

        def responses(**kwargs):
            image_path = Path(kwargs["image_paths"][0])
            if image_path.stem == "page_001":
                return SimpleNamespace(model=kwargs["model"], response=long_text, done=True)
            return _fake_page_response(**kwargs)

        mock_ollama_client.generate_vision.side_effect = responses
        extractor = DocumentExtractor(
            ollama_client=mock_ollama_client,
            prompt_manager=PromptManager(prompts_dir=prompts_dir),
        )

        extractor.extract(pdf_path, model_name="qwen2.5vl:3b", prompt_id="describe")

        second_call_prompt = mock_ollama_client.generate_vision.call_args_list[1].kwargs["prompt"]
        # The full long_text must NOT appear verbatim (it would if
        # unbounded); only a slice up to the configured limit may.
        assert long_text not in second_call_prompt
        injected_context = "X" * MAX_CROSS_PAGE_CONTEXT_CHARS
        assert injected_context in second_call_prompt
        # And not one character more than the limit.
        assert ("X" * (MAX_CROSS_PAGE_CONTEXT_CHARS + 1)) not in second_call_prompt

    def test_page_image_content_unaffected_by_context_injection(
        self, tmp_path, extractor, mock_ollama_client, output_cleanup
    ):
        """The context feature must only change the PROMPT text — the
        image(s) sent for each page must remain exactly what they were
        before this milestone: one image per page under single_page."""
        document_name = "test_resume_context_image_unaffected"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=2)

        extractor.extract(pdf_path, model_name="qwen2.5vl:3b", prompt_id="describe")

        for call in mock_ollama_client.generate_vision.call_args_list:
            assert len(call.kwargs["image_paths"]) == 1
            assert call.kwargs["image_paths"][0].is_file()

    def test_context_does_not_carry_across_more_than_one_page(
        self, tmp_path, extractor, mock_ollama_client, output_cleanup
    ):
        document_name = "test_resume_context_not_carried_further"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=3)

        extractor.extract(pdf_path, model_name="qwen2.5vl:3b", prompt_id="describe")

        calls = mock_ollama_client.generate_vision.call_args_list
        third_call_prompt = calls[2].kwargs["prompt"]
        # Page 3 must see page 2's tail...
        assert "Content for page_002" in third_call_prompt
        # ...but never page 1's, even though page 1 is two pages back.
        assert "Content for page_001" not in third_call_prompt

    def test_prompt_file_contains_cross_page_continuation_instruction(self):
        """Requirement 6: the real, on-disk OCR prompt (not a test
        fixture) must contain the new rule explaining how to use
        injected preceding-page context."""
        from config import PROMPTS_DIR

        prompt_text = (PROMPTS_DIR / "resume" / "extract_markdown.md").read_text(
            encoding="utf-8"
        )
        assert "Preceding Page Context" in prompt_text
        assert "NOT part of the image" in prompt_text or "not part of the image" in prompt_text.lower()

    def test_single_page_document_behavior_unchanged(
        self, tmp_path, extractor, mock_ollama_client, output_cleanup
    ):
        """Requirement 7: with no preceding page at all (a one-page
        document), the single request's prompt must be byte-for-byte
        identical to pre-Milestone-23.13 behavior."""
        document_name = "test_resume_context_single_page_unchanged"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=1)

        extractor.extract(pdf_path, model_name="qwen2.5vl:3b", prompt_id="describe")

        only_call_prompt = mock_ollama_client.generate_vision.call_args_list[0].kwargs["prompt"]
        assert only_call_prompt == "Describe these pages as markdown."

    def test_page_after_a_blank_page_receives_no_context(
        self, tmp_path, extractor, mock_ollama_client, output_cleanup
    ):
        """A blank page (Milestone 15B.5) contributes "" as its text —
        the page after it must therefore see no context block, not an
        empty/broken one, exactly like a document's first page."""
        document_name = "test_resume_context_after_blank_page"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"

        document = fitz.open()
        page1 = document.new_page()
        page1.insert_text((72, 72), _content_text("Real content on page one"), fontsize=11)
        document.new_page()  # blank
        page3 = document.new_page()
        page3.insert_text((72, 72), _content_text("Real content on page three"), fontsize=11)
        document.save(pdf_path)
        document.close()

        extractor.extract(pdf_path, model_name="qwen2.5vl:3b", prompt_id="describe")

        # Only pages 1 and 3 reach the model (page 2 is blank, skipped).
        calls = mock_ollama_client.generate_vision.call_args_list
        assert len(calls) == 2
        page_three_prompt = calls[1].kwargs["prompt"]
        assert "Preceding Page Context" not in page_three_prompt
        assert page_three_prompt == "Describe these pages as markdown."

    def test_earlier_experience_regression_case_page_two_receives_heading_context(
        self, tmp_path, mock_ollama_client, prompts_dir, output_cleanup
    ):
        """Unit-level regression test for the known case: page 1 ends
        with an open section heading and no content under it; page 2
        begins with orphaned bullets and no heading. Proves ONLY that
        page 2's request is built with the required preceding context
        — NOT that a real vision model would correctly use it. Live
        verification against the real Rima_Karmakar_CV.pdf is separate
        (docs/MILESTONE_23_13_CROSS_PAGE_CONTEXT_RESULTS.md)."""
        document_name = "test_resume_earlier_experience_regression"
        output_cleanup.append(document_name)
        pdf_path = tmp_path / f"{document_name}.pdf"
        _make_pdf_with_text(pdf_path, page_count=2)

        page_one_markdown = (
            "## Professional Experience\n\n"
            "### Some Company – Some Role (2021 – 2024)\n"
            "- Did some things.\n\n"
            "## Earlier Experience\n"
        )

        def responses(**kwargs):
            image_path = Path(kwargs["image_paths"][0])
            if image_path.stem == "page_001":
                return SimpleNamespace(
                    model=kwargs["model"], response=page_one_markdown, done=True
                )
            return _fake_page_response(**kwargs)

        mock_ollama_client.generate_vision.side_effect = responses
        extractor = DocumentExtractor(
            ollama_client=mock_ollama_client,
            prompt_manager=PromptManager(prompts_dir=prompts_dir),
        )

        extractor.extract(pdf_path, model_name="qwen2.5vl:3b", prompt_id="describe")

        second_call_prompt = mock_ollama_client.generate_vision.call_args_list[1].kwargs["prompt"]
        assert "Earlier Experience" in second_call_prompt
        assert "Preceding Page Context" in second_call_prompt


@pytest.mark.integration
def test_real_ollama_extract_integration(tmp_path, prompts_dir, output_cleanup):
    """Optional integration test against a real local Ollama server and
    real vision model, over a real multi-page PDF, so it actually
    exercises the one-request-per-page and merge behavior end to end.
    Skips itself if Ollama or the expected model isn't available."""
    ollama_client = OllamaClient()

    if not ollama_client.is_running():
        pytest.skip("Ollama is not running locally; skipping integration test.")
    if not ollama_client.model_exists(INTEGRATION_VISION_MODEL):
        pytest.skip(f"Model '{INTEGRATION_VISION_MODEL}' is not installed; skipping.")

    document_name = "test_resume_integration"
    output_cleanup.append(document_name)
    pdf_path = tmp_path / f"{document_name}.pdf"
    _make_pdf_with_text(pdf_path, page_count=2)

    extractor = DocumentExtractor(
        ollama_client=ollama_client,
        prompt_manager=PromptManager(prompts_dir=prompts_dir),
    )

    result = extractor.extract(
        pdf_path, model_name=INTEGRATION_VISION_MODEL, prompt_id="describe"
    )

    assert isinstance(result.response, ollama.GenerateResponse)
    assert result.page_count == 2
    assert (BASE_DIR / result.markdown_path).is_file()
    assert not (OUTPUT_DIR / "temp" / document_name).exists()
