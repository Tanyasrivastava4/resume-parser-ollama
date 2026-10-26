"""
services/column_aware_stage1_extractor.py

Milestone V2.4H — production-readiness wrapper around the column-aware
linearization algorithm proven in Milestones V2.4D/E/F/G. This is a
NEW file; it is the ONLY place this milestone adds file-I/O around
that algorithm. `services/experimental_column_layout.py` itself is
NOT modified by this milestone — every function it exports
(`analyze_page_layout`, `linearize_page`, `_coverage_gap`,
`_merge_same_row_blocks`, `_split_into_sublines`) is called here
exactly as committed, unchanged.

Mirrors `services/native_stage1_extractor.py`'s own shape (a small
class with one `extract(pdf_path, output_dir) -> Result` method) so
`services/stage1_router.py` and its caller can treat this extractor
interchangeably with `NativeStage1Extractor` at the call site — see
`docs/MILESTONE_V2_4H_TWO_COLUMN_ROUTE_INTEGRATION.md` §2 for why this
shape was chosen (reuse of an already-proven interface, not a new
design).

Selectivity is enforced entirely by the caller
(`services/stage1_router.py`) — this class does not decide whether it
should run; it is only invoked for documents `Stage1Router` has
already classified as requiring column-aware handling.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from services.experimental_column_layout import analyze_page_layout, linearize_page

logger = logging.getLogger(__name__)


class ColumnAwareStage1ExtractorError(Exception):
    """Base exception for all column-aware Stage-1 extraction failures."""


@dataclass
class ColumnAwareExtractionResult:
    """Result of running ColumnAwareStage1Extractor.extract() on one PDF.

    Shape deliberately mirrors NativeExtractionResult
    (services/native_stage1_extractor.py) where the concept is
    analogous, for the same reason that module's own docstring gives —
    easy comparison, not a claim of interchangeability the caller must
    still decide.
    """

    pdf_path: str
    markdown_path: str
    text: str
    method: str = "column_aware_linearization"
    page_count: int = 0
    pages_with_column_gap: int = 0
    char_count: int = 0
    processing_time: float = 0.0


class ColumnAwareStage1Extractor:
    """Extracts a PDF's text using the column-aware linearization
    algorithm (services/experimental_column_layout.py, unmodified),
    writing a Stage-2-compatible text file.

    Import of `fitz` is deferred into `extract()`, matching
    NativeStage1Extractor's own defensive posture.
    """

    def extract(self, pdf_path: Path, output_dir: Path) -> ColumnAwareExtractionResult:
        """Extract `pdf_path` via the column-aware linearizer and write
        the result to `output_dir/<pdf_stem>.md`.

        Every page is processed independently through
        `analyze_page_layout` (unmodified) — a page with no detected
        column gap is linearized in its own natural top-to-bottom
        block order (the algorithm's own documented no-op behavior for
        single-column pages), so calling this on a page that turns out
        not to need column handling is safe, not just for the
        documents this milestone's router sends here.

        Raises:
            ColumnAwareStage1ExtractorError: If PyMuPDF cannot open
                the PDF.
        """
        start_time = time.perf_counter()
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        import fitz  # PyMuPDF — deferred import, matches NativeStage1Extractor

        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            raise ColumnAwareStage1ExtractorError(
                f"PyMuPDF could not open {pdf_path}: {exc}"
            ) from exc

        try:
            page_count = doc.page_count
            page_texts: List[str] = []
            pages_with_gap = 0
            for page_index in range(page_count):
                page = doc.load_page(page_index)
                blocks = page.get_text("blocks")
                records, gap = analyze_page_layout(
                    blocks, page_index + 1, page.rect.width, page.rect.height
                )
                if gap is not None:
                    pages_with_gap += 1
                page_texts.append(linearize_page(records))
        finally:
            doc.close()

        full_text = "\n\n".join(page_texts)
        markdown_path = output_dir / f"{pdf_path.stem}.md"
        markdown_path.write_text(full_text, encoding="utf-8")

        result = ColumnAwareExtractionResult(
            pdf_path=str(pdf_path),
            markdown_path=str(markdown_path),
            text=full_text,
            page_count=page_count,
            pages_with_column_gap=pages_with_gap,
            char_count=len(full_text),
            processing_time=time.perf_counter() - start_time,
        )
        logger.info(
            "Column-aware-extracted %s in %.4fs (%d page(s), %d with detected column gap, %d chars)",
            pdf_path,
            result.processing_time,
            page_count,
            pages_with_gap,
            result.char_count,
        )
        return result
