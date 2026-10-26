"""
services/batch_processor.py

Batch Processing (Milestone 11).

Scans input/resumes/ for every PDF and runs each one through
DocumentExtractor, one at a time. A failure on any single document is
logged and recorded, not raised — the batch keeps going so one bad PDF
can't stop an entire run.

This module does no JSON extraction, resume parsing, or benchmarking of
its own. Markdown output is already saved by DocumentExtractor; this
module's job is purely to find PDFs, call the extractor for each, and
collect timing/success/failure results across the batch.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from config import BASE_DIR, RESUMES_INPUT_DIR
from services.document_extractor import DocumentExtractor

logger = logging.getLogger(__name__)


@dataclass
class BatchFailure:
    """One document's failure within a batch run.

    Attributes:
        pdf_path: Path to the PDF that failed, as a POSIX-style string
            relative to the project root when possible.
        error: The failure's error message.
    """

    pdf_path: str
    error: str


@dataclass
class DocumentBatchRecord:
    """Per-document outcome from a batch run, covering both successes and
    failures — the detail callers like the benchmark runner need to build
    a per-resume report, which the failure-only `list_of_failures` can't
    provide on its own.

    Attributes:
        pdf_path: Path to the source PDF, as a POSIX-style string relative
            to the project root when possible.
        success: Whether extraction completed without error.
        processing_time: Seconds spent on this document (time to success,
            or time elapsed before it failed).
        error: The failure's error message, or None on success.
        page_count: Number of pages rendered, or None on failure.
        markdown_path: Path to the saved markdown file, or None on
            failure.
    """

    pdf_path: str
    success: bool
    processing_time: float
    error: Optional[str] = None
    page_count: Optional[int] = None
    markdown_path: Optional[str] = None


@dataclass
class BatchProcessingResult:
    """Aggregate result of processing a batch of PDFs.

    Attributes:
        total_documents: Number of PDFs found and attempted.
        successful_documents: Number that extracted without error.
        failed_documents: Number that raised an error during extraction.
        total_processing_time: Wall-clock time, in seconds, for the whole
            batch run (finding files and processing every document).
        average_processing_time: total_processing_time divided by
            total_documents (0.0 if no documents were found).
        list_of_failures: One BatchFailure per document that failed, in
            the order they were processed.
        results: One DocumentBatchRecord per document attempted (success
            or failure), in the order they were processed.
    """

    total_documents: int
    successful_documents: int
    failed_documents: int
    total_processing_time: float
    average_processing_time: float
    list_of_failures: List[BatchFailure] = field(default_factory=list)
    results: List[DocumentBatchRecord] = field(default_factory=list)


class BatchProcessor:
    """Finds every PDF under an input directory and extracts each with
    DocumentExtractor, continuing past individual failures.
    """

    def __init__(
        self,
        document_extractor: Optional[DocumentExtractor] = None,
        input_dir: Path = RESUMES_INPUT_DIR,
    ) -> None:
        """Initialize the batch processor.

        Args:
            document_extractor: DocumentExtractor to use for each PDF.
                Defaults to a new DocumentExtractor() built from its own
                default dependencies.
            input_dir: Directory to scan for PDFs. Defaults to
                config.RESUMES_INPUT_DIR (input/resumes/).
        """
        self.document_extractor = document_extractor or DocumentExtractor()
        self.input_dir = Path(input_dir)

    def process_batch(self, model_name: str, prompt_id: str) -> BatchProcessingResult:
        """Extract every PDF found under `input_dir`.

        Each PDF is passed to `DocumentExtractor.extract`, one at a time.
        If a document raises any exception, it's logged and recorded as
        a failure in the returned result, and processing continues with
        the next PDF.

        Args:
            model_name: Name of the vision model to pass through to
                DocumentExtractor for every document.
            prompt_id: Prompt identifier to pass through to
                DocumentExtractor for every document.

        Returns:
            A BatchProcessingResult summarizing the whole run.
        """
        pdf_paths = self._find_pdfs()
        total_documents = len(pdf_paths)
        failures: List[BatchFailure] = []
        records: List[DocumentBatchRecord] = []
        successful_documents = 0

        logger.info("Found %d PDF(s) under %s", total_documents, self.input_dir)

        start_time = time.perf_counter()
        for pdf_path in tqdm(pdf_paths, desc="Processing resumes", unit="pdf"):
            relative_path = self._to_relative_str(pdf_path)
            document_start = time.perf_counter()
            try:
                extraction_result = self.document_extractor.extract(
                    pdf_path, model_name=model_name, prompt_id=prompt_id
                )
                successful_documents += 1
                records.append(
                    DocumentBatchRecord(
                        pdf_path=relative_path,
                        success=True,
                        processing_time=extraction_result.processing_time,
                        page_count=extraction_result.page_count,
                        markdown_path=extraction_result.markdown_path,
                    )
                )
                logger.info("Processed %s", pdf_path)
            except Exception as exc:  # noqa: BLE001 - one bad PDF must not stop the batch
                document_processing_time = time.perf_counter() - document_start
                error_message = str(exc)
                logger.error("Failed to process %s: %s", pdf_path, error_message)
                failures.append(BatchFailure(pdf_path=relative_path, error=error_message))
                records.append(
                    DocumentBatchRecord(
                        pdf_path=relative_path,
                        success=False,
                        processing_time=document_processing_time,
                        error=error_message,
                    )
                )
        total_processing_time = time.perf_counter() - start_time

        average_processing_time = (
            total_processing_time / total_documents if total_documents else 0.0
        )

        logger.info(
            "Batch complete: %d/%d succeeded in %.2fs (avg %.2fs/doc)",
            successful_documents,
            total_documents,
            total_processing_time,
            average_processing_time,
        )

        return BatchProcessingResult(
            total_documents=total_documents,
            successful_documents=successful_documents,
            failed_documents=len(failures),
            total_processing_time=total_processing_time,
            average_processing_time=average_processing_time,
            list_of_failures=failures,
            results=records,
        )

    def _find_pdfs(self) -> List[Path]:
        """Return every PDF under `input_dir`, sorted for a stable order."""
        if not self.input_dir.is_dir():
            logger.warning("Input directory does not exist: %s", self.input_dir)
            return []
        return sorted(
            path
            for path in self.input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        )

    @staticmethod
    def _to_relative_str(path: Path) -> str:
        """Render `path` relative to the project root when possible, for
        consistency with the path strings other services return."""
        try:
            return path.relative_to(BASE_DIR).as_posix()
        except ValueError:
            return str(path)
