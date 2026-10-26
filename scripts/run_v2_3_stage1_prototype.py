"""
scripts/run_v2_3_stage1_prototype.py

Milestone V2.3 — Native Stage-1 / Routing Prototype CLI (opt-in only).

This is a NEW, separate entry point. It does not replace, call, or
modify scripts/run_parser_pipeline.py or services/batch_processor.py,
and running it changes nothing about what those scripts do when invoked
on their own — the production OCR-first default path is untouched.

Usage:
    python scripts/run_v2_3_stage1_prototype.py --mode routed
    python scripts/run_v2_3_stage1_prototype.py --mode native-only
    python scripts/run_v2_3_stage1_prototype.py --mode ocr-only --pdf AMAN_CV_BCA.pdf
    python scripts/run_v2_3_stage1_prototype.py --mode routed --stage2-check

--mode (Step 6's required "clearly documented way to invoke either
path"):
    routed       (default) Stage1Router decides native vs. OCR per
                 document — this is the actual V2 hybrid prototype.
    native-only  Force every document through NativeStage1Extractor,
                 regardless of what the router would decide. Useful for
                 inspecting native output on every document, including
                 ones the router would send to OCR.
    ocr-only     Force every document through the existing, UNMODIFIED
                 DocumentExtractor (real vision-model OCR calls) —
                 reproduces today's production Stage-1 behavior exactly,
                 except output is written to --output-dir instead of
                 output/markdown/ (see _ExperimentalDocumentExtractor
                 below), so a live run never overwrites production or
                 frozen-baseline markdown.

All output — extracted text and provenance sidecars — is written under
--output-dir (default: output/v2_3_prototype/), never under
output/markdown/, output/json_final/, output/modules/, or any
benchmark/baselines/ or benchmark/ground_truth/ directory. This script
never writes to any of those paths.

--stage2-check, if given, additionally proves the produced text is
Stage-2-compatible by calling ONE real, unmodified module extractor
(PersonalInformationExtractor.extract(), from
services/personal_information_extractor.py) directly against the
produced file. BaseModuleExtractor.extract() (confirmed by direct read
this milestone — see docs/MILESTONE_V2_3_NATIVE_STAGE1_PROTOTYPE.md §7)
writes nothing to disk itself, so this check calls Ollama and returns
an in-memory result only; it never touches output/json_final/ or
output/modules/, and never calls pipeline.ParserPipeline.run() (whose
save paths are hardcoded and not caller-injectable — see the same
report section for why that path is deliberately avoided here).
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import BASE_DIR, OUTPUT_DIR, RESUMES_INPUT_DIR  # noqa: E402
from services.column_aware_stage1_extractor import (  # noqa: E402
    ColumnAwareStage1Extractor,
    ColumnAwareStage1ExtractorError,
)
from services.document_extractor import DocumentExtractor  # noqa: E402
from services.native_stage1_extractor import (  # noqa: E402
    NativeStage1Extractor,
    NativeStage1ExtractorError,
)
from services.stage1_provenance import write_provenance_sidecar  # noqa: E402
from services.stage1_router import (  # noqa: E402
    ROUTE_COLUMN_AWARE_NATIVE,
    ROUTE_NATIVE,
    ROUTE_OCR_FALLBACK,
    RouteDecision,
    Stage1Router,
    Stage1RouterError,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "v2_3_prototype"
OCR_MODEL_NAME = "qwen2.5vl:3b"  # confirmed via scripts/run_vision_model_benchmark.py,
# the same model name production's own Stage-1 OCR path uses today.
OCR_PROMPT_ID = "resume/extract_markdown"  # same confirmation.

MODE_ROUTED = "routed"
MODE_NATIVE_ONLY = "native-only"
MODE_OCR_ONLY = "ocr-only"
SUPPORTED_MODES = (MODE_ROUTED, MODE_NATIVE_ONLY, MODE_OCR_ONLY)


class _ExperimentalDocumentExtractor(DocumentExtractor):
    """DocumentExtractor subclass used ONLY by this prototype script.

    Overrides exactly one method — `_save_markdown` — to redirect the
    output file into an experimental directory instead of the
    hardcoded `output/markdown/` every other caller of DocumentExtractor
    still writes to unchanged. Every other method (page rasterization,
    blank-page detection, per-page vision calls, cross-page context,
    split-columns strategy, retry counting) is inherited completely
    unmodified from services/document_extractor.py — this subclass adds
    no new extraction behavior, it only changes where the result lands
    on disk, so it counts as running "the existing OCR path... exactly
    as before" per this milestone's Step 4/Step 9 requirements.
    """

    def __init__(self, *args, experimental_output_dir: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self._experimental_output_dir = Path(experimental_output_dir)

    def _save_markdown(self, document_name: str, markdown_text: str) -> Path:
        self._experimental_output_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = self._experimental_output_dir / f"{document_name}.md"
        markdown_path.write_text(markdown_text, encoding="utf-8")
        return markdown_path


def _find_pdfs(input_dir: Path, only: Optional[List[str]]) -> List[Path]:
    if only:
        return [input_dir / name for name in only]
    if not input_dir.is_dir():
        return []
    return sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"
    )


def _run_native(pdf_path: Path, markdown_dir: Path) -> dict:
    extractor = NativeStage1Extractor()
    result = extractor.extract(pdf_path, markdown_dir)
    return {
        "extraction_method": "native",
        "markdown_path": result.markdown_path,
        "diagnostics": {
            "page_count": result.page_count,
            "pdftotext_char_count": result.pdftotext_char_count,
            "pymupdf_char_count": result.pymupdf_char_count,
            "embedded_font_count": result.embedded_font_count,
            "hyperlink_count": len(result.hyperlinks),
            "hyperlinks_recovered_only_via_annotation": sum(
                1 for h in result.hyperlinks if not h.visible_in_text
            ),
            "corruption_ratio": result.corruption_ratio,
            "cross_tool_jaccard": result.cross_tool_jaccard,
            "processing_time_seconds": result.processing_time,
        },
    }


def _run_column_aware(pdf_path: Path, markdown_dir: Path) -> dict:
    extractor = ColumnAwareStage1Extractor()
    result = extractor.extract(pdf_path, markdown_dir)
    return {
        "extraction_method": "column_aware_native",
        "markdown_path": result.markdown_path,
        "diagnostics": {
            "page_count": result.page_count,
            "pages_with_column_gap": result.pages_with_column_gap,
            "char_count": result.char_count,
            "processing_time_seconds": result.processing_time,
        },
    }


def _run_ocr(pdf_path: Path, markdown_dir: Path) -> dict:
    extractor = _ExperimentalDocumentExtractor(experimental_output_dir=markdown_dir)
    result = extractor.extract(pdf_path, model_name=OCR_MODEL_NAME, prompt_id=OCR_PROMPT_ID)
    return {
        "extraction_method": "ocr_fallback",
        "markdown_path": result.markdown_path,
        "diagnostics": {
            "page_count": result.page_count,
            "processing_time_seconds": result.processing_time,
            "retry_count": result.retry_count,
        },
    }


def _stage2_check(markdown_path: Path) -> dict:
    """Call PersonalInformationExtractor.extract() directly (no
    ParserPipeline, no disk writes) to prove the produced text is
    Stage-2-compatible. See module docstring for why."""
    from services.personal_information_extractor import PersonalInformationExtractor

    extractor = PersonalInformationExtractor()
    try:
        module_result = extractor.extract(markdown_path)
        return {
            "checked_module": "PersonalInformationExtractor",
            "passed": True,
            "processing_time_seconds": module_result.processing_time,
        }
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return {
            "checked_module": "PersonalInformationExtractor",
            "passed": False,
            "error": str(exc),
        }


def run(
    input_dir: Path,
    output_dir: Path,
    mode: str,
    only: Optional[List[str]] = None,
    stage2_check: bool = False,
) -> dict:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode {mode!r}; expected one of {SUPPORTED_MODES}")

    markdown_dir = output_dir / "markdown"
    provenance_dir = output_dir / "provenance"

    pdf_paths = _find_pdfs(input_dir, only)
    logger.info("Found %d PDF(s) under %s (mode=%s)", len(pdf_paths), input_dir, mode)

    router = Stage1Router()
    summary = {
        "native": 0,
        "ocr_fallback": 0,
        "column_aware_native": 0,
        "errors": 0,
        "total": len(pdf_paths),
    }

    for pdf_path in pdf_paths:
        pdf_stem = pdf_path.stem
        fallback_events: List[str] = []
        start = time.perf_counter()

        try:
            decision = router.classify(pdf_path)
        except Stage1RouterError as exc:
            logger.error("Classification failed for %s: %s", pdf_path, exc)
            decision = RouteDecision(
                pdf_path=str(pdf_path),
                route=ROUTE_OCR_FALLBACK,
                reason=f"Classification failed ({exc}); routing to OCR fallback per "
                "V2.2 Route E safe-failure policy.",
                signals={},
            )
            fallback_events.append(f"classification_error: {exc}")

        if mode == MODE_NATIVE_ONLY:
            effective_route = ROUTE_NATIVE
        elif mode == MODE_OCR_ONLY:
            effective_route = ROUTE_OCR_FALLBACK
        else:
            effective_route = decision.route

        try:
            if effective_route == ROUTE_NATIVE:
                extraction = _run_native(pdf_path, markdown_dir)
                summary["native"] += 1
            elif effective_route == ROUTE_COLUMN_AWARE_NATIVE:
                extraction = _run_column_aware(pdf_path, markdown_dir)
                summary["column_aware_native"] += 1
            else:
                extraction = _run_ocr(pdf_path, markdown_dir)
                summary["ocr_fallback"] += 1
        except (NativeStage1ExtractorError, ColumnAwareStage1ExtractorError) as exc:
            logger.warning(
                "%s extraction failed for %s (%s); falling back to OCR.",
                effective_route, pdf_path, exc,
            )
            fallback_events.append(f"{effective_route}_extraction_failed: {exc}")
            extraction = _run_ocr(pdf_path, markdown_dir)
            summary["ocr_fallback"] += 1

        stage2_result = None
        if stage2_check:
            stage2_result = _stage2_check(Path(extraction["markdown_path"]))

        write_provenance_sidecar(
            output_dir=provenance_dir,
            pdf_stem=pdf_stem,
            route_decision=decision,
            extraction_method=extraction["extraction_method"],
            extraction_diagnostics=extraction["diagnostics"],
            fallback_events=fallback_events,
            stage2_compatibility=stage2_result,
        )

        elapsed = time.perf_counter() - start
        logger.info(
            "%s: route=%s effective=%s elapsed=%.3fs",
            pdf_stem,
            decision.route,
            effective_route,
            elapsed,
        )

    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone V2.3 native Stage-1 / routing prototype (opt-in only)."
    )
    parser.add_argument("--input-dir", type=Path, default=RESUMES_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", type=str, default=MODE_ROUTED, choices=SUPPORTED_MODES)
    parser.add_argument(
        "--pdf",
        action="append",
        default=None,
        help="Restrict to specific PDF filename(s) under --input-dir. Repeatable.",
    )
    parser.add_argument(
        "--stage2-check",
        action="store_true",
        help="Additionally verify Stage-2 compatibility via a direct, "
        "non-persisting PersonalInformationExtractor.extract() call.",
    )
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    summary = run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        only=args.pdf,
        stage2_check=args.stage2_check,
    )
    print("\nV2.3 Stage-1 Prototype Summary")
    print("-------------------------------")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
