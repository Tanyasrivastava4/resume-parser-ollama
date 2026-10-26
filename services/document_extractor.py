"""
services/document_extractor.py

Generic Document Extractor (Milestone 10, refactored for per-page
requests; blank-page detection added in Milestone 15B.5).

Wires together four existing services to turn a PDF into raw markdown
text using a vision model:

    PDFLoader -> page images -> PromptManager -> prompt text
                                       |
                                       v
                    PageQualityClassifier.is_blank_page() PER PAGE
                                       |
                        blank? --yes--> contribute "" (no model call)
                                       |
                                       no
                                       v
                  one OllamaClient.generate_vision call PER PAGE
                                       |
                                       v
        each page's raw markdown saved to output/temp/<document>/
                                       |
                                       v
      merged in page order and saved to output/markdown/<document>.md

Milestone 15B.5's blank-page check exists because
docs/HALLUCINATION_INVESTIGATION.md confirmed a completely blank page
sent to the vision model reliably produces a fabricated resume
template rather than empty output. Detection
(services/page_quality_classifier.py) is deterministic pixel
statistics only — no AI, no vision model — so a page classified Blank
never reaches generate_vision() at all, and simply contributes an
empty string to that page's position in the merged markdown. Every
non-blank page's handling is completely unchanged from before this
milestone.

Earlier versions sent every page image in a single Ollama request, which
exceeded the model's context window for multi-page resumes. Each page is
now sent as its own request, reusing the same prompt every time, and the
per-page responses are merged afterwards — the public API and returned
DocumentExtractionResult are unchanged, so existing callers (including
BatchProcessor and the benchmark runner) need no changes.

This is deliberately generic — it works for any PDF, prompt, and vision
model. It does not know it will eventually be used for resumes, does not
parse or clean the model's output, and does not extract or validate JSON.
The markdown produced here is exactly what the model returned per page,
concatenated in order.

Milestone 15C.3 adds a second, opt-in OCR strategy: "split_columns".
docs/SIDEBAR_RECOGNITION_INVESTIGATION.md found that a full-page image
of a multi-column resume can cause the vision model to drop an entire
column (confirmed on Naukri_NileshShivramMagare[6y_6m]), while the same
model reads that column correctly when it's given as its own, isolated
image. "split_columns" acts on that finding: each non-blank page is
additionally cropped into left-half and right-half full-height images
(never quadrants — a prior experiment showed quadrant cuts can drop
content of their own), each half is sent through the same OCR prompt as
its own request, and the two responses are merged left-then-right into
that page's markdown. The default strategy is, and remains, "single_page"
(settings.OCR_STRATEGY) — every existing caller that does not explicitly
opt in sees identical behavior to before this milestone.

Milestone 23.13 adds bounded cross-page tail-context (both strategies):
docs/MILESTONE_23_11_WORK_EXPERIENCE_ROOT_CAUSE_INVESTIGATION.md and
docs/MILESTONE_23_12_WORK_EXPERIENCE_MODEL_TRIAL_AND_OCR_ARCHITECTURE_INVESTIGATION.md
confirmed a different, deterministic content-loss pattern from
split_columns' — not a dropped column, but a dropped SECTION when its
heading falls at the very bottom of one page and its actual content
(e.g. bullet points) begins the next page with no heading of its own.
Because each page is still sent as its own fully independent request
(unchanged — this milestone does not revert to whole-document context,
which the module's own history above shows was already tried and
rejected for exceeding the model's context window), the vision model
has no way to know a page's headerless leading content might continue
something from the page before it, and — confirmed via direct evidence,
not assumed — simply omits it. `_extract_pages()` now carries forward a
small, bounded (MAX_CROSS_PAGE_CONTEXT_CHARS) tail fragment of each
page's own extracted text into the next page's prompt only (never
further back — see `_build_cross_page_context()`), so the model has
just enough evidence to recognize a continuation without ever being
handed the whole previous page. `prompts/resume/extract_markdown.md`'s
rule 14 tells the model exactly how this context may and may not be
used. A page with no preceding page (the first) or whose preceding page
contributed no text (e.g. blank) sees its prompt completely unchanged
from before this milestone.
"""

import copy
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import fitz  # PyMuPDF
import ollama

from config import BASE_DIR, OUTPUT_DIR
from services.ollama_client import OllamaClient
from services.page_quality_classifier import is_blank_page
from services.pdf_loader import load_pdf_pages
from services.prompt_manager import PromptManager
from settings import OCR_STRATEGY

logger = logging.getLogger(__name__)

# Root directory for per-page markdown scratch files, cleaned up once each
# document's pages have been merged into its final markdown file.
TEMP_DIR = OUTPUT_DIR / "temp"

# Supported values for the `strategy` argument to DocumentExtractor.extract().
# "single_page" (Milestone 10) sends the whole page in one request, unchanged.
# "split_columns" (Milestone 15C.3) additionally splits each non-blank page
# into left/right halves — see the module docstring.
OCR_STRATEGY_SINGLE_PAGE = "single_page"
OCR_STRATEGY_SPLIT_COLUMNS = "split_columns"
SUPPORTED_OCR_STRATEGIES = (OCR_STRATEGY_SINGLE_PAGE, OCR_STRATEGY_SPLIT_COLUMNS)

# Milestone 23.13: maximum number of characters of the immediately
# preceding page's own extracted text carried forward as context for the
# next page's request — see _build_cross_page_context()'s docstring for
# why this exists and docs/MILESTONE_23_13_CROSS_PAGE_CONTEXT_RESULTS.md
# for the evidence this bound is based on. Deliberately small: this is
# context to help the model recognize a section continuing onto this
# page, not a second copy of the previous page's content.
MAX_CROSS_PAGE_CONTEXT_CHARS = 300

_CROSS_PAGE_CONTEXT_HEADER = (
    "--- Preceding Page Context (tail of the previous page's own "
    "transcription, for continuity only — this text is NOT part of the "
    "image below) ---"
)
_CROSS_PAGE_CONTEXT_FOOTER = "--- End Preceding Page Context ---"


class DocumentExtractorError(Exception):
    """Base exception for all Document Extractor failures."""


class AllPagesBlankError(DocumentExtractorError):
    """Raised when every page of a document was classified Blank, so no
    vision-model response exists to build a DocumentExtractionResult
    from. A genuinely all-blank source document is treated as an error
    condition rather than silently producing an empty result."""


class UnsupportedOCRStrategyError(DocumentExtractorError):
    """Raised when `strategy` (or settings.OCR_STRATEGY) is not one of
    SUPPORTED_OCR_STRATEGIES."""


@dataclass
class DocumentExtractionResult:
    """Result of extracting one PDF document into markdown.

    Attributes:
        markdown_path: Path to the saved markdown file, as a POSIX-style
            string relative to the project root, e.g.
            "output/markdown/resume1.md".
        response: An `ollama.GenerateResponse`-shaped object whose
            `.response` text is the full merged markdown (the same text
            saved to `markdown_path`), so callers reading
            `result.response.response` see the complete document exactly
            as they did before pages were sent one request at a time.
            Its other fields (model, timing, etc.) come from the last
            page's request.
        processing_time: Wall-clock time, in seconds, for the whole
            extraction (loading pages, prompting every page, merging, and
            saving output).
        page_count: Number of pages the PDF was rendered into.
        image_paths: Paths to each rendered page image, as POSIX-style
            strings relative to the project root, in page order.
        retry_count: Total number of transient-connection retries used
            across every page's request (see OllamaClient's retry
            logic). Stored for future milestones (e.g. benchmarking) to
            expose if needed; defaults to 0 so existing code that builds
            a DocumentExtractionResult without it keeps working.
    """

    markdown_path: str
    response: ollama.GenerateResponse
    processing_time: float
    page_count: int
    image_paths: List[str]
    retry_count: int = 0


class DocumentExtractor:
    """Converts a PDF into markdown using a vision model, one page per request.

    Composes PDFLoader (page rasterization), PromptManager (prompt
    loading), and OllamaClient (model inference). Dependencies are
    injectable so callers — and tests — can supply their own instances,
    e.g. a mocked OllamaClient.
    """

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        prompt_manager: Optional[PromptManager] = None,
        pdf_loader: Callable[[Union[str, Path]], List[str]] = load_pdf_pages,
    ) -> None:
        """Initialize the extractor.

        Args:
            ollama_client: OllamaClient to use for inference. Defaults to
                a new OllamaClient() built from settings.py.
            prompt_manager: PromptManager to use for loading prompts.
                Defaults to a new PromptManager() rooted at
                config.PROMPTS_DIR.
            pdf_loader: Callable that renders a PDF to page images and
                returns their paths. Defaults to
                services.pdf_loader.load_pdf_pages.
        """
        self.ollama_client = ollama_client or OllamaClient()
        self.prompt_manager = prompt_manager or PromptManager()
        self._load_pdf_pages = pdf_loader

    def extract(
        self,
        pdf_path: Union[str, Path],
        model_name: str,
        prompt_id: str,
        strategy: Optional[str] = None,
    ) -> DocumentExtractionResult:
        """Extract a PDF's contents as markdown using a vision model.

        Every page of `pdf_path` is rendered to an image, and each image
        is sent to `model_name` as its own request, one page at a time,
        alongside the same prompt loaded from `prompt_id`. Each page's
        raw response is saved to a temporary per-page markdown file, then
        all pages are merged in order and saved verbatim to
        output/markdown/<document_name>.md.

        This one-request-per-page approach avoids exceeding the model's
        context window on multi-page documents, which sending every page
        in a single request could do.

        Args:
            pdf_path: Path to the source PDF file.
            model_name: Name of an installed, vision-capable Ollama
                model, e.g. "qwen2.5vl:3b".
            prompt_id: Prompt identifier to load via PromptManager, e.g.
                "resume/extract_markdown".
            strategy: OCR strategy to use — "single_page" (default,
                unchanged since Milestone 10) or "split_columns"
                (Milestone 15C.3, experimental — see module docstring).
                Defaults to `None`, which resolves to
                settings.OCR_STRATEGY (itself "single_page" unless
                overridden), so callers that don't pass this see
                identical behavior to before this milestone.

        Returns:
            A DocumentExtractionResult describing what was produced.

        Raises:
            PDFNotFoundError: If `pdf_path` does not exist or is not a
                file (from services.pdf_loader).
            InvalidPDFError: If `pdf_path` cannot be read as a PDF, or
                has no pages (from services.pdf_loader).
            AllPagesBlankError: If every page of `pdf_path` was
                classified Blank (Milestone 15B.5) — there is no
                vision-model response to build a result from.
            UnsupportedOCRStrategyError: If `strategy` (after resolving
                the default) is not one of SUPPORTED_OCR_STRATEGIES.
            PromptNotFoundError: If `prompt_id` has no matching prompt
                file (from services.prompt_manager).
            ModelNotFoundError: If `model_name` is not installed in
                Ollama (from services.ollama_client).
            OllamaNotRunningError: If the Ollama server cannot be reached
                (from services.ollama_client).
            OllamaConnectionError: If a different connection-level error
                occurs while calling Ollama (from services.ollama_client).
        """
        start_time = time.perf_counter()
        pdf_path = Path(pdf_path)
        document_name = pdf_path.stem
        strategy = strategy or OCR_STRATEGY
        if strategy not in SUPPORTED_OCR_STRATEGIES:
            raise UnsupportedOCRStrategyError(
                f"Unsupported OCR strategy {strategy!r}; expected one of "
                f"{SUPPORTED_OCR_STRATEGIES}"
            )

        logger.info(
            "Extracting document %s with model=%s prompt=%s strategy=%s",
            pdf_path,
            model_name,
            prompt_id,
            strategy,
        )

        image_paths = self._load_pdf_pages(pdf_path)
        absolute_image_paths = [BASE_DIR / path for path in image_paths]
        prompt_text = self.prompt_manager.load(prompt_id)

        temp_dir = TEMP_DIR / document_name
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            page_responses, temp_paths, retry_count = self._extract_pages(
                absolute_image_paths, model_name, prompt_text, temp_dir, strategy
            )
            merged_markdown = self._merge_markdown(temp_paths)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        markdown_path = self._save_markdown(document_name, merged_markdown)
        response = self._build_combined_response(page_responses, merged_markdown)
        processing_time = time.perf_counter() - start_time

        logger.info(
            "Extracted %d page(s) from %s in %.2fs (%d retr(y/ies)) -> %s",
            len(image_paths),
            pdf_path,
            processing_time,
            retry_count,
            markdown_path,
        )

        return DocumentExtractionResult(
            markdown_path=markdown_path.relative_to(BASE_DIR).as_posix(),
            response=response,
            processing_time=processing_time,
            page_count=len(image_paths),
            image_paths=image_paths,
            retry_count=retry_count,
        )

    def _extract_pages(
        self,
        absolute_image_paths: List[Path],
        model_name: str,
        prompt_text: str,
        temp_dir: Path,
        strategy: str,
    ) -> Tuple[List[ollama.GenerateResponse], List[Path], int]:
        """Send each non-blank page image to the model as its own
        request (or two requests, for "split_columns" — see below),
        saving every page's raw response (or, for a page classified
        Blank, an empty string) to a temporary markdown file.

        A page classified Blank by is_blank_page() (Milestone 15B.5)
        never reaches the model at all — see the module docstring for
        why. This check runs against the whole, unsplit page image
        regardless of strategy, since a genuinely blank page is blank
        in both halves too. Every non-blank page's handling under
        "single_page" is unchanged from before Milestone 15C.3.

        Returns:
            A (page_responses, temp_paths, retry_count) tuple.
            temp_paths is in page order and has one entry per page,
            including blank ones. page_responses contains only the
            real responses from non-blank pages (one per page for
            "single_page", two per page — left then right — for
            "split_columns"), in the order they were received.
            retry_count is the sum of OllamaClient's per-call retry
            counts across every non-blank page (both halves, for
            "split_columns").
        """
        page_responses: List[ollama.GenerateResponse] = []
        temp_paths: List[Path] = []
        total_pages = len(absolute_image_paths)
        retry_count = 0
        # Milestone 23.13: the immediately preceding page's own extracted
        # text (possibly ""), or None before the first page — see
        # _build_cross_page_context()'s docstring. Reassigned after every
        # page, blank or not, so context is always exactly one page deep,
        # never carried further back.
        previous_page_text: Optional[str] = None

        for page_number, image_path in enumerate(absolute_image_paths, start=1):
            logger.info("Extracting page %d/%d (%s)", page_number, total_pages, image_path)
            page_prompt_text = self._append_cross_page_context(prompt_text, previous_page_text)

            if is_blank_page(image_path):
                logger.info(
                    "Skipping OCR for page %d/%d (%s) — blank page",
                    page_number,
                    total_pages,
                    image_path,
                )
                page_text = ""
            elif strategy == OCR_STRATEGY_SPLIT_COLUMNS:
                page_text, responses, page_retry_count = self._extract_page_split_columns(
                    image_path, model_name, page_prompt_text, temp_dir, page_number
                )
                page_responses.extend(responses)
                retry_count += page_retry_count
            else:
                page_text, responses, page_retry_count = self._extract_page_single(
                    image_path, model_name, page_prompt_text
                )
                page_responses.extend(responses)
                retry_count += page_retry_count

            temp_path = temp_dir / f"page{page_number:03d}.md"
            temp_path.write_text(page_text, encoding="utf-8")
            temp_paths.append(temp_path)
            previous_page_text = page_text

        return page_responses, temp_paths, retry_count

    def _extract_page_single(
        self, image_path: Path, model_name: str, prompt_text: str
    ) -> Tuple[str, List[ollama.GenerateResponse], int]:
        """The "single_page" strategy (Milestone 10, unchanged): send
        the whole page image as one request.

        Returns a (page_text, responses, retry_count) tuple, matching
        the shape _extract_page_split_columns returns so both can be
        used interchangeably by _extract_pages.
        """
        response = self.ollama_client.generate_vision(
            model=model_name,
            prompt=prompt_text,
            image_paths=[image_path],
        )
        retry_count = getattr(self.ollama_client, "last_retry_count", 0)
        return response.response, [response], retry_count

    def _extract_page_split_columns(
        self,
        image_path: Path,
        model_name: str,
        prompt_text: str,
        temp_dir: Path,
        page_number: int,
    ) -> Tuple[str, List[ollama.GenerateResponse], int]:
        """The "split_columns" strategy (Milestone 15C.3): crop the page
        into left-half and right-half full-height images, send each
        through the same OCR prompt as its own request, and merge the
        two responses left-then-right.

        Returns a (page_text, responses, retry_count) tuple: page_text
        is the two halves' responses joined with a blank line, in
        left-then-right order; responses is [left_response,
        right_response]; retry_count is the sum of both calls' retries.
        """
        left_path, right_path = self._split_page_left_right(image_path, temp_dir, page_number)

        left_response = self.ollama_client.generate_vision(
            model=model_name,
            prompt=prompt_text,
            image_paths=[left_path],
        )
        left_retry_count = getattr(self.ollama_client, "last_retry_count", 0)

        right_response = self.ollama_client.generate_vision(
            model=model_name,
            prompt=prompt_text,
            image_paths=[right_path],
        )
        right_retry_count = getattr(self.ollama_client, "last_retry_count", 0)

        page_text = "\n\n".join([left_response.response, right_response.response])
        return page_text, [left_response, right_response], left_retry_count + right_retry_count

    @staticmethod
    def _split_page_left_right(
        image_path: Path, temp_dir: Path, page_number: int
    ) -> Tuple[Path, Path]:
        """Crop a rendered page image into left-half and right-half
        full-height PNGs (never quadrants — a prior experiment,
        docs/SIDEBAR_RECOGNITION_INVESTIGATION.md, found that quartering
        a page can drop content that a half-page crop preserves), and
        save them under `temp_dir`.

        Uses PyMuPDF directly (matching services/page_quality_classifier.py's
        approach to reading rendered PNGs), rather than adding a new
        imaging dependency.
        """
        source_pixmap = fitz.Pixmap(str(image_path))
        width, height = source_pixmap.width, source_pixmap.height
        half_width = width // 2

        left_clip = fitz.IRect(0, 0, half_width, height)
        right_clip = fitz.IRect(half_width, 0, width, height)
        left_pixmap = fitz.Pixmap(source_pixmap, width, height, left_clip)
        right_pixmap = fitz.Pixmap(source_pixmap, width, height, right_clip)

        left_path = temp_dir / f"page{page_number:03d}_left.png"
        right_path = temp_dir / f"page{page_number:03d}_right.png"
        left_pixmap.save(left_path)
        right_pixmap.save(right_path)
        return left_path, right_path

    @staticmethod
    def _merge_markdown(temp_paths: List[Path]) -> str:
        """Concatenate each page's temporary markdown file, in page order."""
        return "\n\n".join(path.read_text(encoding="utf-8") for path in temp_paths)

    @staticmethod
    def _build_cross_page_context(previous_page_text: Optional[str]) -> Optional[str]:
        """Milestone 23.13: bounded tail fragment of the immediately
        preceding page's own extracted text, for the confirmed
        deterministic failure where a section heading ends one page and
        its content begins the next with no heading of its own — see
        docs/MILESTONE_23_12_WORK_EXPERIENCE_MODEL_TRIAL_AND_OCR_ARCHITECTURE_INVESTIGATION.md
        and docs/MILESTONE_23_13_CROSS_PAGE_CONTEXT_RESULTS.md.

        Deliberately narrow: this only ever looks at the ONE page
        immediately before the page currently being processed — never
        further back — and only ever returns up to
        MAX_CROSS_PAGE_CONTEXT_CHARS characters, taken from the END of
        that page's text (the part closest to the page break, and so
        the most relevant to whether the next page continues it).

        Args:
            previous_page_text: The text the immediately preceding page
                contributed (possibly ""), or None if there is no
                preceding page (this is the first page) or its result
                isn't being tracked.

        Returns:
            The bounded tail, or None if there is no preceding page or
            it contributed no text (e.g. it was blank) — in either
            case, no context block should be added to the prompt at
            all.
        """
        if not previous_page_text:
            return None
        tail = previous_page_text.strip()
        if not tail:
            return None
        if len(tail) > MAX_CROSS_PAGE_CONTEXT_CHARS:
            tail = tail[-MAX_CROSS_PAGE_CONTEXT_CHARS:]
        return tail

    @staticmethod
    def _append_cross_page_context(prompt_text: str, previous_page_text: Optional[str]) -> str:
        """Milestone 23.13: return `prompt_text` unchanged if there is no
        usable preceding-page context, otherwise `prompt_text` with a
        clearly delimited context block appended — see
        _build_cross_page_context() for what that block contains and
        prompts/resume/extract_markdown.md (rule 14) for how the model
        is instructed to use it (evidence only, never to be copied into
        the output, never a license to invent content not actually
        present on the current page's own image).
        """
        context = DocumentExtractor._build_cross_page_context(previous_page_text)
        if context is None:
            return prompt_text
        return (
            f"{prompt_text}\n\n{_CROSS_PAGE_CONTEXT_HEADER}\n"
            f"{context}\n{_CROSS_PAGE_CONTEXT_FOOTER}"
        )

    @staticmethod
    def _build_combined_response(
        page_responses: List[ollama.GenerateResponse], merged_markdown: str
    ) -> ollama.GenerateResponse:
        """Build the DocumentExtractionResult.response value: a copy of
        the last non-blank page's response with `.response` replaced by
        the full merged markdown, so `result.response.response` still
        yields the complete document, matching the pre-refactor
        single-request behavior.

        Raises:
            AllPagesBlankError: If `page_responses` is empty — every
                page of the document was classified Blank, so there is
                no real model response to build a result from.
        """
        if not page_responses:
            raise AllPagesBlankError(
                "Every page was classified blank; no vision-model response "
                "is available to build a result from."
            )
        combined_response = copy.copy(page_responses[-1])
        combined_response.response = merged_markdown
        return combined_response

    def _save_markdown(self, document_name: str, markdown_text: str) -> Path:
        """Save raw markdown text to output/markdown/<document_name>.md."""
        markdown_dir = OUTPUT_DIR / "markdown"
        markdown_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = markdown_dir / f"{document_name}.md"
        markdown_path.write_text(markdown_text, encoding="utf-8")
        return markdown_path
