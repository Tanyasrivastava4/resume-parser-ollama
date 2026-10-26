"""
services/native_stage1_extractor.py

Native Stage-1 Extraction Adapter (Milestone V2.3 — prototype, opt-in,
NOT wired into the default production path).

Produces a Stage-2-compatible text artifact directly from a PDF's native
text layer, using only tooling already confirmed available and already
used read-only during Milestone V2.1
(docs/MILESTONE_V2_1_NATIVE_PDF_VS_OCR_INVESTIGATION.md):

    - System Poppler `pdftotext -layout` (primary extraction; V2.1 §11
      found this reconstructs tables and preserves two-column layout
      better than PyMuPDF's own sort mode for the one confirmed
      multi-column case).
    - PyMuPDF (`fitz`, available only under `.venv/bin/python` — see
      config note below) for cross-check text, embedded-font count, and
      hyperlink-annotation recovery (V2.1 §12 — directly reproduced the
      ADJ-034 Ritik Gupta finding).

No package is installed by this module. No production module is
imported or modified. This adapter's output is written wherever the
caller tells it to (never a hardcoded production path), and its own
default caller (scripts/run_v2_3_stage1_prototype.py) always points it
at an experimental directory, never output/markdown/.

Milestone V2.2 (docs/MILESTONE_V2_2_HYBRID_ROUTING_AND_CONFIDENCE_ARCHITECTURE.md,
§4/§7) explicitly deferred calibrating a corruption-ratio threshold and a
cross-tool-agreement threshold to a future milestone, citing only 1-2
data points as evidence. This module computes both signals (they are
cheap, and future calibration needs the data), but — matching that
document's own instruction not to invent thresholds — neither signal is
used here to alter what gets written or how; they are diagnostic-only,
carried into the provenance sidecar (services/stage1_provenance.py) for
a future, dedicated calibration milestone (V2.2 §12, Phase V2.5).
"""

import logging
import re
import subprocess
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class NativeStage1ExtractorError(Exception):
    """Base exception for all native Stage-1 extraction failures."""


class PdftotextNotAvailableError(NativeStage1ExtractorError):
    """Raised when the `pdftotext` binary cannot be invoked at all (not
    installed, not on PATH). Distinct from a `pdftotext` call that runs
    but produces no usable output, which is not an error — see
    NativeExtractionResult.char_count instead."""


# Characters treated as "expected" and excluded from the corruption-ratio
# numerator: printable ASCII, common accented Latin ranges, and the
# smart-punctuation / bullet glyphs V2.1 (§20) found on multiple
# resumes and confirmed were decorative, not corruption (e.g. ,
# the Wingdings/Symbol bullet glyph).
_SAFE_NON_ASCII_RANGES = (
    (0x00A0, 0x02FF),  # Latin-1 supplement + Latin Extended-A/B (accents)
    (0x2010, 0x2027),  # general punctuation: dashes, quotes, ellipsis
    (0x2030, 0x205E),  # further general punctuation
    (0xF0B7, 0xF0B7),  # confirmed decorative bullet glyph (V2.1 §20)
)


def _is_safe_non_ascii(ch: str) -> bool:
    code_point = ord(ch)
    for lo, hi in _SAFE_NON_ASCII_RANGES:
        if lo <= code_point <= hi:
            return True
    return False


def _corruption_ratio(text: str) -> float:
    """Ratio of non-ASCII characters outside the curated "expected" set
    (see _SAFE_NON_ASCII_RANGES) to total non-whitespace characters.

    DIAGNOSTIC SIGNAL ONLY (see module docstring) — reproduces the
    broadened-scan methodology V2.1 §20 used to separate resume010's
    genuine corruption (1.56%) from every other resume's decorative-icon
    noise (<=0.75%). That document explicitly labeled the specific cut
    point a calibration requirement, not a validated threshold; this
    function returns the raw ratio only, and nothing in this codebase
    acts on it yet.
    """
    non_whitespace = [ch for ch in text if not ch.isspace()]
    if not non_whitespace:
        return 0.0
    suspicious = sum(
        1 for ch in non_whitespace if ord(ch) > 127 and not _is_safe_non_ascii(ch)
    )
    return suspicious / len(non_whitespace)


_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set:
    return set(_WORD_RE.findall(text.lower()))


def _jaccard_similarity(text_a: str, text_b: str) -> Optional[float]:
    """Word-level Jaccard overlap between two extractions of the same
    document. DIAGNOSTIC SIGNAL ONLY — the same general method V2.1 §8
    used to compare native extraction against OCR, applied here between
    the two native tools themselves (pdftotext vs PyMuPDF) as a
    cross-signal-confidence candidate (V2.2 §7.4). Returns None if both
    texts are empty (undefined overlap, not zero agreement).
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a and not tokens_b:
        return None
    union = tokens_a | tokens_b
    if not union:
        return None
    return len(tokens_a & tokens_b) / len(union)


@dataclass
class HyperlinkAnnotation:
    """One embedded URI annotation found via PyMuPDF's page.get_links(),
    filtered to kind == fitz.LINK_URI — see V2.1 §12."""

    page_number: int  # 1-indexed
    uri: str
    visible_in_text: bool  # True if `uri` (or its host) appears in the
    # extracted text already — i.e. whether this link would have been
    # discoverable without hyperlink-annotation access at all.


@dataclass
class NativeExtractionResult:
    """Result of running NativeStage1Extractor.extract() on one PDF.

    Attributes mirror DocumentExtractionResult
    (services/document_extractor.py) where the concept is analogous, so
    the two are easy to compare, but this is a distinct type — this
    adapter's output has never been treated as interchangeable with
    OCR's by anything in this codebase; that decision belongs to the
    caller (services/stage1_router.py), not this class.
    """

    pdf_path: str
    markdown_path: str
    text: str
    method: str  # "pdftotext_layout" (primary) — the only method this
    # prototype writes as the chosen Stage-2 artifact; PyMuPDF text is
    # computed only for cross-checking, never written as the primary
    # artifact (see module docstring: V2.1 found pdftotext -layout
    # outperforms PyMuPDF's own text order on the one confirmed
    # multi-column case).
    page_count: int
    pdftotext_char_count: int
    pymupdf_char_count: int
    embedded_font_count: int
    hyperlinks: List[HyperlinkAnnotation] = field(default_factory=list)
    corruption_ratio: float = 0.0  # diagnostic only, see module docstring
    cross_tool_jaccard: Optional[float] = None  # diagnostic only
    processing_time: float = 0.0


class NativeStage1Extractor:
    """Extracts a PDF's native text layer into a Stage-2-compatible text
    file, using `pdftotext -layout` as the primary source and PyMuPDF
    for cross-checking and hyperlink recovery.

    Import of `fitz` is deferred into `extract()` (not module-level)
    so that importing this module under an interpreter without PyMuPDF
    installed (e.g. base `python3`, per V2.1's own finding that `fitz`
    exists only under `.venv/bin/python`) does not fail merely by being
    imported — the same defensive posture services/pdf_loader.py and
    services/document_extractor.py do NOT need, since production always
    runs under an environment with PyMuPDF present; this module is new
    and may reasonably be imported for inspection under either
    interpreter.
    """

    def extract(self, pdf_path: Path, output_dir: Path) -> NativeExtractionResult:
        """Extract `pdf_path`'s native text and write it to
        `output_dir/<pdf_stem>.md`.

        Args:
            pdf_path: Source PDF.
            output_dir: Directory the Stage-2-compatible text artifact
                is written to. Never defaulted to output/markdown/ by
                this method — the caller must pass an explicit,
                experimental directory (see
                scripts/run_v2_3_stage1_prototype.py).

        Returns:
            A NativeExtractionResult with the extracted text and every
            diagnostic signal computed.

        Raises:
            PdftotextNotAvailableError: If the `pdftotext` binary cannot
                be invoked at all.
            NativeStage1ExtractorError: If PyMuPDF cannot open the PDF.
        """
        start_time = time.perf_counter()
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pdftotext_text = self._run_pdftotext(pdf_path)

        import fitz  # PyMuPDF — see class docstring for why this import is deferred

        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            raise NativeStage1ExtractorError(
                f"PyMuPDF could not open {pdf_path}: {exc}"
            ) from exc

        try:
            page_count = doc.page_count
            pymupdf_text_parts: List[str] = []
            font_names = set()
            hyperlinks: List[HyperlinkAnnotation] = []

            for page_index in range(page_count):
                page = doc.load_page(page_index)
                page_text = page.get_text()
                pymupdf_text_parts.append(page_text)

                for font_info in page.get_fonts():
                    font_names.add(font_info[3])

                for link in page.get_links():
                    if link.get("kind") == fitz.LINK_URI:
                        uri = link.get("uri", "")
                        if not uri:
                            continue
                        visible = uri in pdftotext_text or uri in page_text
                        hyperlinks.append(
                            HyperlinkAnnotation(
                                page_number=page_index + 1,
                                uri=uri,
                                visible_in_text=visible,
                            )
                        )
        finally:
            doc.close()

        pymupdf_text = "\n\n".join(pymupdf_text_parts)

        chosen_text = pdftotext_text
        markdown_path = self._write_output(pdf_path, output_dir, chosen_text, hyperlinks)

        result = NativeExtractionResult(
            pdf_path=str(pdf_path),
            markdown_path=str(markdown_path),
            text=chosen_text,
            method="pdftotext_layout",
            page_count=page_count,
            pdftotext_char_count=len(pdftotext_text),
            pymupdf_char_count=len(pymupdf_text),
            embedded_font_count=len(font_names),
            hyperlinks=hyperlinks,
            corruption_ratio=_corruption_ratio(pdftotext_text),
            cross_tool_jaccard=_jaccard_similarity(pdftotext_text, pymupdf_text),
            processing_time=time.perf_counter() - start_time,
        )
        logger.info(
            "Native-extracted %s in %.4fs (pdftotext=%d chars, pymupdf=%d chars, "
            "fonts=%d, links=%d, corruption_ratio=%.4f, cross_tool_jaccard=%s)",
            pdf_path,
            result.processing_time,
            result.pdftotext_char_count,
            result.pymupdf_char_count,
            result.embedded_font_count,
            len(hyperlinks),
            result.corruption_ratio,
            f"{result.cross_tool_jaccard:.4f}" if result.cross_tool_jaccard is not None else "n/a",
        )
        return result

    @staticmethod
    def _run_pdftotext(pdf_path: Path) -> str:
        """Run `pdftotext -layout <pdf> -` (write to stdout) and return
        its output as text. `-layout` is used, not plain mode, per
        V2.1 §11's finding that it correctly preserves table row/column
        alignment and (§6/§9) two-column reading order better than the
        alternative tested there.
        """
        try:
            proc = subprocess.run(
                ["pdftotext", "-layout", str(pdf_path), "-"],
                capture_output=True,
                timeout=30,
                check=False,
            )
        except FileNotFoundError as exc:
            raise PdftotextNotAvailableError(
                "pdftotext binary not found on PATH"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise NativeStage1ExtractorError(
                f"pdftotext timed out on {pdf_path}"
            ) from exc

        if proc.returncode != 0:
            logger.warning(
                "pdftotext returned nonzero exit (%d) for %s: %s",
                proc.returncode,
                pdf_path,
                proc.stderr.decode("utf-8", errors="replace").strip(),
            )
        return proc.stdout.decode("utf-8", errors="replace")

    @staticmethod
    def _write_output(
        pdf_path: Path,
        output_dir: Path,
        text: str,
        hyperlinks: List[HyperlinkAnnotation],
    ) -> Path:
        """Write the chosen text to `output_dir/<pdf_stem>.md`, UTF-8,
        matching the exact contract pipeline/parser_pipeline.py's
        `run()` reads via `markdown_path.read_text(encoding="utf-8")`
        (confirmed directly this milestone — see
        docs/MILESTONE_V2_3_NATIVE_STAGE1_PROTOTYPE.md §7).

        If any embedded hyperlink annotation is not already visible in
        the extracted text, its URI is appended in a clearly delimited
        section — this is the direct, minimal mechanism that recovers
        cases like Ritik Gupta's LinkedIn URL (V2.1 §12) without
        inventing any new structure Stage 2's prompts don't already
        know how to read (plain text).
        """
        undiscovered = [h for h in hyperlinks if not h.visible_in_text]
        content = text
        if undiscovered:
            lines = [
                "",
                "--- Embedded Hyperlink Annotations (not visible as plain text; "
                "recovered from the PDF's own link annotations) ---",
            ]
            for h in undiscovered:
                lines.append(f"Page {h.page_number}: {h.uri}")
            content = text + "\n" + "\n".join(lines) + "\n"

        markdown_path = output_dir / f"{pdf_path.stem}.md"
        markdown_path.write_text(content, encoding="utf-8")
        return markdown_path
