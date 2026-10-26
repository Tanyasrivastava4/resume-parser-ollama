"""
services/stage1_router.py

Document-level Stage-1 routing (Milestone V2.3 — prototype, opt-in, NOT
wired into the default production path; Milestone V2.4H — extended
with a third, selective route, still opt-in at the call-site level —
see that milestone's own report for why this does not change
scripts/run_parser_pipeline.py's or services/batch_processor.py's
default behavior).

Implements THREE routing rules:

    native_char_count == 0 AND embedded_font_count == 0
        -> route "ocr_fallback"
    otherwise, if any page shows a genuine multi-column gap
    (services/experimental_column_layout.py's own, unmodified
    `analyze_page_layout` — the SAME detector Milestones V2.4D/E/F/G
    already validated, not a duplicate)
        -> route "column_aware_native"
    otherwise
        -> route "native"

The ocr_fallback/native split is the original V2.3 rule, chosen
because it is the only rule this project has directly, freshly
measured across the entire 22-PDF frozen baseline (re-measured this
milestone, not reused from a cached number — see
docs/MILESTONE_V2_3_NATIVE_STAGE1_PROTOTYPE.md §2) as a clean,
non-arbitrary separation rather than a fuzzy cutoff:
Naukri_ShubhangiBagla[4y_0m].pdf (resume018, the corpus's one confirmed
IMAGE_ONLY case per V2.1) measured 0 characters and 0 embedded fonts.
Every other one of the 22 PDFs measured at least 2,354 characters and at
least 1 embedded font — a gap of multiple orders of magnitude, not a
threshold requiring interpolation. This module therefore treats
"0 and 0" as the rule, not an arbitrary cutoff like "<50 chars" chosen
without a data point to justify it.

The column_aware_native route (Milestone V2.4H) reuses
`analyze_page_layout`'s own gap-detection criteria exactly as already
validated: a ≥10pt gap, away from the page margins, splitting the page
into two ≥3-block, ≥10%-page-height-spanning regions. Milestones
V2.4D/E/F/G measured this detector at 1/22 true-positive, 0/21
false-positive across the entire frozen corpus, across three
independent re-checks (V2.4D §7, V2.4E's own corpus scan, V2.4F §3,
V2.4G §3) — this router does not recompute or duplicate that
detection logic; it imports and calls the exact same function.

Milestone V2.2 (§4/§5) identified several additional classification
signals — corruption ratio, cross-tool agreement, table presence — and
explicitly labeled every one of them a CALIBRATION REQUIREMENT, not a
validated threshold (as few as 1-2 data points each). This router does
NOT use any of them to decide a route. It still computes and surfaces
them (via NativeStage1Extractor's diagnostics, carried through by the
caller into the provenance sidecar) so a future, dedicated calibration
milestone (V2.2 §12, Phase V2.5) has real data to calibrate against —
but using them to change routing behavior now would be inventing an
unsupported threshold, which this milestone's own instructions
explicitly forbid.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

ROUTE_NATIVE = "native"
ROUTE_OCR_FALLBACK = "ocr_fallback"
ROUTE_COLUMN_AWARE_NATIVE = "column_aware_native"


class Stage1RouterError(Exception):
    """Base exception for all Stage1Router failures."""


@dataclass
class RouteDecision:
    """The routing outcome for one PDF.

    Attributes:
        pdf_path: The PDF this decision was made for.
        route: ROUTE_NATIVE or ROUTE_OCR_FALLBACK.
        reason: Human-readable explanation of why this route was chosen.
        signals: Every signal this router computed, diagnostic-only
            beyond the two (char_count, font_count) that actually
            decided the route — see module docstring.
    """

    pdf_path: str
    route: str
    reason: str
    signals: Dict[str, object] = field(default_factory=dict)


class Stage1Router:
    """Decides, per document, whether native extraction or OCR should be
    the primary Stage-1 path — see module docstring for the single rule
    this applies.
    """

    def classify(self, pdf_path: Path) -> RouteDecision:
        """Classify `pdf_path` and return a RouteDecision.

        Raises:
            Stage1RouterError: If PyMuPDF cannot open the PDF at all —
                per V2.2 §5 Route E ("Exceptional/Unsupported"), this is
                surfaced as an error rather than guessed at; the caller
                (scripts/run_v2_3_stage1_prototype.py) treats this the
                same as an OCR-fallback case, since OCR requires no
                assumption about the document's own native structure
                (the safest available fallback per V2.2 §5).
        """
        pdf_path = Path(pdf_path)
        import fitz  # PyMuPDF — deferred import, see NativeStage1Extractor's docstring

        # Milestone V2.4H: reuses the exact, unmodified detector
        # services/experimental_column_layout.py already validated —
        # imported here, not reimplemented, per that milestone's own
        # "avoid duplicate detectors" instruction.
        from services.experimental_column_layout import analyze_page_layout

        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            raise Stage1RouterError(f"PyMuPDF could not open {pdf_path}: {exc}") from exc

        try:
            char_count = 0
            font_names = set()
            pages_with_column_gap = 0
            for page in doc:
                char_count += len(page.get_text())
                for font_info in page.get_fonts():
                    font_names.add(font_info[3])
                # Only the gap-detection call is new here; character/font
                # counting above is byte-for-byte the same as before this
                # milestone. Column detection is checked on every page
                # regardless of char/font outcome so the signal is always
                # available in the sidecar, even though it only changes
                # the route when the document isn't already going to
                # ocr_fallback (checked below, not here).
                blocks = page.get_text("blocks")
                _, gap = analyze_page_layout(blocks, page.number + 1, page.rect.width, page.rect.height)
                if gap is not None:
                    pages_with_column_gap += 1
            page_count = doc.page_count
        finally:
            doc.close()

        font_count = len(font_names)
        signals = {
            "char_count": char_count,
            "embedded_font_count": font_count,
            "page_count": page_count,
            "pages_with_column_gap": pages_with_column_gap,
        }

        if char_count == 0 and font_count == 0:
            decision = RouteDecision(
                pdf_path=str(pdf_path),
                route=ROUTE_OCR_FALLBACK,
                reason=(
                    "Zero native characters and zero embedded fonts — the one "
                    "signal combination this project has directly measured as "
                    "a clean, non-arbitrary separator of image-only documents "
                    "(see module docstring)."
                ),
                signals=signals,
            )
        elif pages_with_column_gap > 0:
            decision = RouteDecision(
                pdf_path=str(pdf_path),
                route=ROUTE_COLUMN_AWARE_NATIVE,
                reason=(
                    f"Native text layer present ({char_count} chars, {font_count} "
                    f"embedded font(s)); {pages_with_column_gap} of {page_count} "
                    "page(s) show a genuine multi-column gap (same detector "
                    "validated in Milestones V2.4D/E/F/G at 1/22 true-positive, "
                    "0/21 false-positive) — routing to the column-aware "
                    "linearizer instead of plain native extraction."
                ),
                signals=signals,
            )
        else:
            decision = RouteDecision(
                pdf_path=str(pdf_path),
                route=ROUTE_NATIVE,
                reason=(
                    f"Native text layer present ({char_count} chars, "
                    f"{font_count} embedded font(s)); no multi-column gap detected."
                ),
                signals=signals,
            )

        logger.info(
            "Stage1Router: %s -> %s (%s)", pdf_path, decision.route, decision.reason
        )
        return decision
