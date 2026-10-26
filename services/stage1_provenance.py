"""
services/stage1_provenance.py

Provenance sidecar writer (Milestone V2.3 — prototype, opt-in).

Implements the "(B) separate sidecar/debug artifact" recommendation from
docs/MILESTONE_V2_2_HYBRID_ROUTING_AND_CONFIDENCE_ARCHITECTURE.md §9:
non-breaking (no existing consumer of output/json_final/ or
output/markdown/ is touched), additive, and directly modeled on the
project's own existing sidecar precedent
(output/modules/<resume>/<module>.json, written by
pipeline/parser_pipeline.py's `_save_module_json`).

Writes ONE JSON file per document, at
<output_dir>/<pdf_stem>.json, never inside output/json_final/ or any
frozen-baseline directory. Does not modify, and is never merged into,
the final resume schema.
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROVENANCE_SCHEMA_VERSION = "v2.3-prototype-1"


def write_provenance_sidecar(
    output_dir: Path,
    pdf_stem: str,
    route_decision: "RouteDecision",  # noqa: F821 - services.stage1_router.RouteDecision
    extraction_method: str,
    extraction_diagnostics: Optional[Dict[str, Any]] = None,
    fallback_events: Optional[List[str]] = None,
    stage2_compatibility: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a provenance sidecar JSON file for one document.

    Args:
        output_dir: Directory to write into (an experimental
            directory — never output/json_final/ or a frozen-baseline
            directory).
        pdf_stem: The document's filename stem, used as the sidecar's
            own filename.
        route_decision: The Stage1Router.RouteDecision made for this
            document.
        extraction_method: "native" or "ocr_fallback" — which path was
            actually used to produce the Stage-2-compatible artifact
            (may differ from route_decision.route only if a fallback
            event occurred — see `fallback_events`).
        extraction_diagnostics: Free-form diagnostic signals from
            whichever extractor actually ran (e.g.
            NativeExtractionResult's corruption_ratio, cross_tool_jaccard,
            hyperlink count) — carried through verbatim, not
            reinterpreted, so a future calibration milestone has the raw
            numbers.
        fallback_events: Human-readable notes on any fallback that
            occurred during this document's processing (e.g. "native
            extraction attempted but pdftotext unavailable; used OCR").
            Empty list if none.
        stage2_compatibility: Optional result of a controlled Stage-2
            compatibility check (see
            scripts/run_v2_3_stage1_prototype.py's --stage2-check flag)
            — never the real ParserPipeline.run() output, only a
            pass/fail + module name from a direct, non-persisting
            BaseModuleExtractor.extract() call.

    Returns:
        Path to the written sidecar file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "document_identifier": pdf_stem,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "routing": {
            "selected_route": route_decision.route,
            "reason": route_decision.reason,
            "classification_signals": route_decision.signals,
        },
        "extraction": {
            "method_used": extraction_method,
            "diagnostics": extraction_diagnostics or {},
        },
        "fallback_events": fallback_events or [],
        "stage2_compatibility": stage2_compatibility,
    }

    sidecar_path = output_dir / f"{pdf_stem}.json"
    sidecar_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote provenance sidecar: %s", sidecar_path)
    return sidecar_path
