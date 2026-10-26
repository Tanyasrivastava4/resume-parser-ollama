"""
evaluation/parser_evaluator.py

Parser Evaluator (Milestone 14.8; finalized in Milestone 14.10 —
ground-truth schema validation, corrected overall-accuracy aggregation,
and richer per-failure reporting).

Evaluates the complete ParserPipeline against a ground-truth dataset:

    benchmark/ground_truth/json/mapping.csv
        (ResumeID, PDF, Markdown, JSON columns — pairs each markdown
        file with its ground-truth JSON file)
                |
                v
    for every mapped resume:
        output/markdown/<Markdown>
                |
                v
        ParserPipeline.run()  (NOT individual services)
                |
                v
        merged JSON
                |
                v
        JsonComparator.compare(ground_truth, merged)
                |
                v
        collect statistics

This module does not render markdown and does not run DocumentExtractor —
it only reads markdown that already exists on disk. A single resume's
failure (missing markdown, missing/invalid ground truth, a pipeline
error) is recorded and evaluation continues with the next resume, the
same resilience philosophy as BatchProcessor.
"""

import csv
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import COMPARISON_DIR, GROUND_TRUTH_DIR, MARKDOWN_DIR
from pipeline.parser_pipeline import ParserPipeline
import hashlib

from validators.json_comparator import (
    COMPARATOR_VERSION,
    ComparisonResult,
    DiagnosticResult,
    JsonComparator,
    load_optional_top_level_keys,
    strict_excluding_optional_keys,
)
from validators.schema_validator import SchemaValidationError, SchemaValidator

logger = logging.getLogger(__name__)

MAPPING_CSV_FILENAME = "mapping.csv"


class ParserEvaluatorError(Exception):
    """Base exception for all Parser Evaluator failures."""


class MappingFileNotFoundError(ParserEvaluatorError):
    """Raised when ground_truth_dir has no mapping.csv — a hard stop,
    since without it there is nothing to evaluate."""


@dataclass
class EvaluationFailure:
    """Full context for why a single resume could not be evaluated
    (Milestone 14.10, Problem 4).

    Attributes:
        resume: The resume name (markdown file stem).
        reason: Human-readable reason evaluation failed.
        ground_truth_file: Path to the ground-truth JSON file that was
            (or would have been) used.
        markdown_file: Path to the markdown file that was (or would
            have been) used.
    """

    resume: str
    reason: str
    ground_truth_file: str
    markdown_file: str


@dataclass
class ParserEvaluationResult:
    """Result of evaluating ParserPipeline against the ground-truth dataset.

    Attributes:
        total_resumes: Number of resumes listed in mapping.csv that
            evaluation was attempted for.
        successful_resumes: Number that were parsed and compared
            without error.
        failed_resumes: Number that could not be evaluated (missing
            markdown, missing/invalid ground truth, or a pipeline
            error).
        overall_accuracy: matched fields / total fields, summed across
            every successful resume (0.0, never 1.0/100%, if there
            were no successful resumes to compare — Milestone 14.10,
            Problem 3).
        module_accuracy: Mapping of module name (e.g.
            "personal_information") to its accuracy, aggregated across
            every successful resume.
        field_accuracy: Mapping of dot-notation field path to its
            accuracy, aggregated across every successful resume.
        resume_accuracy: Mapping of resume name to that resume's own
            accuracy, for successful resumes only.
        processing_time: Wall-clock time, in seconds, for the whole
            evaluation run.
        failures: Mapping of resume name to why it failed, for resumes
            that could not be evaluated. Kept as plain strings for
            backward compatibility.
        failure_details: Mapping of resume name to an EvaluationFailure
            carrying the reason plus the ground-truth and markdown
            file paths involved (Milestone 14.10, Problem 4).
        diagnostics: Mapping of resume name to its DiagnosticResult
            (Milestone 20.3). Always empty unless the evaluator was
            constructed with diagnostic=True — existing callers that
            never opt in see the same empty dict this field defaults to
            today, so nothing about ParserEvaluationResult's existing
            shape or values changes for them.
    """

    total_resumes: int
    successful_resumes: int
    failed_resumes: int
    overall_accuracy: float
    module_accuracy: Dict[str, float] = field(default_factory=dict)
    field_accuracy: Dict[str, float] = field(default_factory=dict)
    resume_accuracy: Dict[str, float] = field(default_factory=dict)
    processing_time: float = 0.0
    failures: Dict[str, str] = field(default_factory=dict)
    failure_details: Dict[str, EvaluationFailure] = field(default_factory=dict)
    diagnostics: Dict[str, DiagnosticResult] = field(default_factory=dict)
    # Decision D8 (2026-10-17) -- additive, all defaulted so every existing
    # constructor call and every pickled/compared result keeps working.
    # The raw strict figures above are untouched; these report the SAME
    # outcomes with optional top-level schema keys (today: certifications,
    # projects) left out, as a separately named diagnostic.
    overall_accuracy_excluding_optional_keys: float = 0.0
    resume_accuracy_excluding_optional_keys: Dict[str, float] = field(default_factory=dict)
    optional_keys_excluded: List[str] = field(default_factory=list)
    # Run identification (D8): which scoring rules and which ground truth.
    comparator_version: str = COMPARATOR_VERSION
    ground_truth_dir: str = ""
    ground_truth_row_count: int = 0
    ground_truth_mapping_sha256: str = ""


class ParserEvaluator:
    """Evaluates ParserPipeline's output against ground-truth JSON.

    Uses ParserPipeline as a whole (not its individual services) to
    parse each resume, and JsonComparator to score the result against
    its ground truth. Every dependency is constructor-injected; this
    class holds no global state.
    """

    def __init__(
        self,
        parser_pipeline: Optional[ParserPipeline] = None,
        json_comparator: Optional[JsonComparator] = None,
        schema_validator: Optional[SchemaValidator] = None,
        ground_truth_dir: Path = GROUND_TRUTH_DIR,
        markdown_dir: Path = MARKDOWN_DIR,
        output_dir: Path = COMPARISON_DIR,
        diagnostic: bool = False,
    ) -> None:
        """Initialize the evaluator.

        Args:
            parser_pipeline: ParserPipeline used to parse every resume.
                Defaults to a new ParserPipeline() built from its own
                default dependencies.
            json_comparator: JsonComparator used to score each result
                against its ground truth. Defaults to a new
                JsonComparator().
            schema_validator: SchemaValidator used to validate each
                ground-truth JSON file before it's used (Milestone
                14.10, Problem 2). Defaults to a new SchemaValidator()
                (schemas/resume_schema_v1.json).
            ground_truth_dir: Directory containing mapping.csv and the
                ground-truth JSON files. Defaults to
                config.GROUND_TRUTH_DIR (benchmark/ground_truth/json/).
            markdown_dir: Directory containing the markdown files listed
                in mapping.csv. Defaults to config.MARKDOWN_DIR
                (output/markdown/).
            output_dir: Directory to write the comparison reports into.
                Defaults to config.COMPARISON_DIR
                (benchmark/comparison/).
            diagnostic: Opt-in diagnostic mode (Milestone 20.3). Defaults
                to False, in which case evaluate()'s behavior, return
                value, and every file it writes are identical to before
                this parameter existed — diagnostic scoring is computed,
                and generate_diagnostic_reports() is called, only when
                this is True. See docs/JSON_COMPARATOR_DIAGNOSTIC.md.
        """
        self.parser_pipeline = parser_pipeline or ParserPipeline()
        self.json_comparator = json_comparator or JsonComparator()
        self.schema_validator = schema_validator or SchemaValidator()
        self.ground_truth_dir = Path(ground_truth_dir)
        self.markdown_dir = Path(markdown_dir)
        self.output_dir = Path(output_dir)
        self.diagnostic = diagnostic

    def evaluate(self) -> ParserEvaluationResult:
        """Run the full evaluation over every resume in mapping.csv.

        Returns:
            A ParserEvaluationResult summarizing the whole run. Also
            writes field_accuracy.csv, resume_accuracy.csv,
            module_accuracy.csv, and benchmark_report.md into
            output_dir.

        Raises:
            MappingFileNotFoundError: If ground_truth_dir has no
                mapping.csv.
        """
        start_time = time.perf_counter()
        mapping_rows = self._load_mapping()

        comparisons: Dict[str, ComparisonResult] = {}
        diagnostics: Dict[str, DiagnosticResult] = {}
        failures: Dict[str, str] = {}
        failure_details: Dict[str, EvaluationFailure] = {}

        for row in mapping_rows:
            resume_name = Path(row["Markdown"]).stem
            markdown_path = self.markdown_dir / row["Markdown"]
            ground_truth_path = self.ground_truth_dir / row["JSON"]
            try:
                comparison, diagnostic_result = self._evaluate_one(markdown_path, ground_truth_path)
                comparisons[resume_name] = comparison
                if diagnostic_result is not None:
                    diagnostics[resume_name] = diagnostic_result
                logger.info("Evaluated %s", resume_name)
            except Exception as exc:  # noqa: BLE001 - one bad resume must not stop evaluation
                reason = str(exc)
                logger.error(
                    "Failed to evaluate %s: %s (ground truth: %s, markdown: %s)",
                    resume_name,
                    reason,
                    ground_truth_path,
                    markdown_path,
                )
                failures[resume_name] = reason
                failure_details[resume_name] = EvaluationFailure(
                    resume=resume_name,
                    reason=reason,
                    ground_truth_file=str(ground_truth_path),
                    markdown_file=str(markdown_path),
                )

        processing_time = time.perf_counter() - start_time

        result = self._build_result(
            comparisons, failures, failure_details, processing_time, diagnostics
        )
        self._attach_run_identification(result)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.json_comparator.generate_reports(comparisons, output_dir=self.output_dir)
        self._write_module_accuracy_csv(comparisons, self.output_dir / "module_accuracy.csv")
        self._write_benchmark_report(result, self.output_dir / "benchmark_report.md")
        self._write_run_identification(result, self.output_dir / "run_identification.json")
        if self.diagnostic:
            # Additive only: generate_reports() above has already written
            # every file evaluate() has always written, unchanged. This
            # writes new files alongside them (diagnostic_summary.csv,
            # diagnostic_mismatch_by_field.csv, diagnostic_report.md) and
            # touches nothing already on disk.
            self.json_comparator.generate_diagnostic_reports(diagnostics, output_dir=self.output_dir)

        logger.info(
            "Evaluation complete: %d/%d succeeded in %.2fs (overall accuracy %.2f%%)",
            result.successful_resumes,
            result.total_resumes,
            processing_time,
            result.overall_accuracy * 100,
        )

        return result

    def _evaluate_one(
        self, markdown_path: Path, ground_truth_path: Path
    ) -> Tuple[ComparisonResult, Optional[DiagnosticResult]]:
        """Parse and compare a single mapped resume.

        Returns a (ComparisonResult, DiagnosticResult) pair. The
        DiagnosticResult is None unless the evaluator was constructed
        with diagnostic=True (Milestone 20.3) — computing it costs one
        extra, purely local comparison pass (no extra ParserPipeline.run()
        call, no extra I/O), but it's still skipped by default so a
        caller who never asked for diagnostic mode pays nothing for it.

        Raises:
            FileNotFoundError: If the markdown or ground-truth file is
                missing.
            ValueError: If the ground-truth file is not valid JSON, or
                does not conform to the resume schema (Milestone 14.10,
                Problem 2).
            Any exception ParserPipeline.run() raises (markdown not
                found, invalid JSON, schema validation failure, a merge
                conflict, or an Ollama-related error).
        """
        if not markdown_path.is_file():
            raise FileNotFoundError(f"Markdown file not found: {markdown_path}")
        if not ground_truth_path.is_file():
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")

        try:
            ground_truth_json = json.loads(ground_truth_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Ground truth file is not valid JSON: {ground_truth_path}") from exc

        try:
            self.schema_validator.validate(ground_truth_json)
        except SchemaValidationError as exc:
            raise ValueError(
                f"Ground truth file does not conform to schema: {ground_truth_path} ({exc})"
            ) from exc

        pipeline_result = self.parser_pipeline.run(markdown_path)

        if self.diagnostic:
            diagnostic_result = self.json_comparator.compare_diagnostic(
                ground_truth_json, pipeline_result.merged_json
            )
            # diagnostic_result.strict_result is compare()'s own
            # unmodified output for this exact pair — reusing it instead
            # of calling compare() a second time guarantees the strict
            # ComparisonResult returned here is identical either way.
            return diagnostic_result.strict_result, diagnostic_result

        return self.json_comparator.compare(ground_truth_json, pipeline_result.merged_json), None

    def _load_mapping(self) -> list:
        """Read mapping.csv's rows (ResumeID, PDF, Markdown, JSON columns)."""
        mapping_path = self.ground_truth_dir / MAPPING_CSV_FILENAME
        if not mapping_path.is_file():
            raise MappingFileNotFoundError(f"mapping.csv not found: {mapping_path}")

        with mapping_path.open(newline="", encoding="utf-8") as mapping_file:
            return list(csv.DictReader(mapping_file))

    def _attach_run_identification(self, result: ParserEvaluationResult) -> None:
        """Decision D8: record which ground truth this run scored against
        (directory, mapping.csv row count and sha256) on the result, so
        every report produced from it is self-describing. The comparator
        version is already set by the dataclass default."""
        result.ground_truth_dir = str(self.ground_truth_dir)
        mapping_path = self.ground_truth_dir / MAPPING_CSV_FILENAME
        try:
            raw = mapping_path.read_bytes()
        except OSError:
            return
        result.ground_truth_mapping_sha256 = hashlib.sha256(raw).hexdigest()
        with mapping_path.open(newline="", encoding="utf-8") as mapping_file:
            result.ground_truth_row_count = sum(1 for _ in csv.DictReader(mapping_file))

    @staticmethod
    def _write_run_identification(result: ParserEvaluationResult, path: Path) -> None:
        """Decision D8: machine-readable twin of the report's
        "Run identification" section."""
        payload = {
            "comparator_version": result.comparator_version,
            "ground_truth_dir": result.ground_truth_dir,
            "ground_truth_row_count": result.ground_truth_row_count,
            "ground_truth_mapping_sha256": result.ground_truth_mapping_sha256,
            "optional_keys_excluded": list(result.optional_keys_excluded),
            "strict_accuracy_raw": result.overall_accuracy,
            "strict_accuracy_excluding_optional_keys": (
                result.overall_accuracy_excluding_optional_keys
            ),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _build_result(
        comparisons: Dict[str, ComparisonResult],
        failures: Dict[str, str],
        failure_details: Dict[str, EvaluationFailure],
        processing_time: float,
        diagnostics: Optional[Dict[str, DiagnosticResult]] = None,
    ) -> ParserEvaluationResult:
        """Aggregate per-resume comparisons into a ParserEvaluationResult."""
        total_matched = sum(result.matched_fields for result in comparisons.values())
        total_fields = sum(result.total_fields for result in comparisons.values())
        # 0.0 (not 1.0) when nothing was successfully compared — an empty
        # numerator/denominator is "nothing evaluated", never "100%
        # accurate" (Milestone 14.10, Problem 3).
        overall_accuracy = (total_matched / total_fields) if total_fields else 0.0

        # Decision D8: same outcomes, optional top-level keys excluded.
        optional_keys = load_optional_top_level_keys()
        exclusions = {
            name: strict_excluding_optional_keys(result, optional_keys)
            for name, result in comparisons.items()
        }
        ex_matched = sum(ex.matched_fields for ex in exclusions.values())
        ex_total = sum(ex.total_fields for ex in exclusions.values())
        overall_accuracy_excluding_optional_keys = (ex_matched / ex_total) if ex_total else 0.0

        resume_accuracy = {name: result.accuracy for name, result in comparisons.items()}
        field_accuracy = ParserEvaluator._aggregate_field_accuracy(comparisons)
        module_accuracy = {
            module: counts["matched"] / counts["total"] if counts["total"] else 0.0
            for module, counts in ParserEvaluator._aggregate_module_counts(comparisons).items()
        }

        return ParserEvaluationResult(
            total_resumes=len(comparisons) + len(failures),
            successful_resumes=len(comparisons),
            failed_resumes=len(failures),
            overall_accuracy=overall_accuracy,
            module_accuracy=module_accuracy,
            field_accuracy=field_accuracy,
            resume_accuracy=resume_accuracy,
            processing_time=processing_time,
            failures=failures,
            failure_details=failure_details,
            diagnostics=diagnostics or {},
            overall_accuracy_excluding_optional_keys=overall_accuracy_excluding_optional_keys,
            resume_accuracy_excluding_optional_keys={
                name: ex.accuracy for name, ex in exclusions.items()
            },
            optional_keys_excluded=optional_keys,
        )

    @staticmethod
    def _aggregate_field_accuracy(comparisons: Dict[str, ComparisonResult]) -> Dict[str, float]:
        """Per-field-path accuracy aggregated across every resume."""
        totals: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "matched": 0})
        for result in comparisons.values():
            for field_result in result.field_results:
                counts = totals[field_result.field_path]
                counts["total"] += 1
                if field_result.status == "PASS":
                    counts["matched"] += 1
        return {
            field_path: (counts["matched"] / counts["total"]) if counts["total"] else 0.0
            for field_path, counts in totals.items()
        }

    @staticmethod
    def _aggregate_module_counts(
        comparisons: Dict[str, ComparisonResult],
    ) -> Dict[str, Dict[str, int]]:
        """Aggregate field results by top-level module (the first
        segment of each dot-notation field path) across every resume."""
        totals: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "matched": 0, "missing": 0, "extra": 0}
        )
        for result in comparisons.values():
            for field_result in result.field_results:
                module_name = field_result.field_path.split(".", 1)[0]
                counts = totals[module_name]
                counts["total"] += 1
                if field_result.status == "PASS":
                    counts["matched"] += 1
                elif field_result.status == "MISSING":
                    counts["missing"] += 1
                elif field_result.status == "EXTRA":
                    counts["extra"] += 1
        return totals

    @staticmethod
    def _write_module_accuracy_csv(
        comparisons: Dict[str, ComparisonResult], csv_path: Path
    ) -> None:
        """One row per module, aggregated across every resume."""
        totals = ParserEvaluator._aggregate_module_counts(comparisons)

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["module", "total", "matched", "missing", "extra", "accuracy"])
            for module_name in sorted(totals):
                counts = totals[module_name]
                accuracy = (counts["matched"] / counts["total"]) if counts["total"] else 0.0
                writer.writerow(
                    [
                        module_name,
                        counts["total"],
                        counts["matched"],
                        counts["missing"],
                        counts["extra"],
                        f"{accuracy:.4f}",
                    ]
                )

    @staticmethod
    def _write_benchmark_report(result: ParserEvaluationResult, report_path: Path) -> None:
        """Write a human-readable Markdown summary of the evaluation."""
        lines = [
            "# Parser Evaluation Report",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Total resumes | {result.total_resumes} |",
            f"| Successful | {result.successful_resumes} |",
            f"| Failed | {result.failed_resumes} |",
            f"| Overall accuracy | {result.overall_accuracy:.2%} |",
            (
                "| Strict accuracy excluding optional keys (diagnostic, D8) | "
                f"{result.overall_accuracy_excluding_optional_keys:.2%} |"
            ),
            f"| Processing time | {result.processing_time:.2f}s |",
            "",
            "## Run identification",
            "",
            "| Item | Value |",
            "|---|---|",
            f"| Comparator version | {result.comparator_version} |",
            f"| Ground truth directory | {result.ground_truth_dir or '(not recorded)'} |",
            f"| Ground truth mapping rows | {result.ground_truth_row_count or '(not recorded)'} |",
            f"| Ground truth mapping sha256 | {result.ground_truth_mapping_sha256 or '(not recorded)'} |",
            (
                "| Optional keys excluded from the diagnostic metric | "
                f"{', '.join(result.optional_keys_excluded) or '(none)'} |"
            ),
            "",
            "Overall accuracy is the raw strict figure (comparator rules unchanged, "
            "comparable with every historical report). The diagnostic figure "
            "re-counts the same outcomes with optional top-level schema keys left "
            "out (decision D8, 2026-10-17); it is reported beside, never instead of, "
            "the raw figure.",
            "",
            "## Module Accuracy",
            "",
            "| Module | Accuracy |",
            "|---|---|",
        ]
        for module_name in sorted(result.module_accuracy):
            lines.append(f"| {module_name} | {result.module_accuracy[module_name]:.2%} |")

        lines += [
            "",
            "## Resume Accuracy",
            "",
            "| Resume | Accuracy | Excluding optional keys (D8) |",
            "|---|---|---|",
        ]
        for resume_name in sorted(result.resume_accuracy):
            excluding = result.resume_accuracy_excluding_optional_keys.get(
                resume_name, result.resume_accuracy[resume_name]
            )
            lines.append(
                f"| {resume_name} | {result.resume_accuracy[resume_name]:.2%} | {excluding:.2%} |"
            )

        lines += ["", "## Failures", ""]
        if result.failures:
            lines += [
                "| Resume | Reason | Ground Truth File | Markdown File |",
                "|---|---|---|---|",
            ]
            for resume_name in sorted(result.failures):
                detail = result.failure_details.get(resume_name)
                reason = detail.reason if detail else result.failures[resume_name]
                ground_truth_file = detail.ground_truth_file if detail else ""
                markdown_file = detail.markdown_file if detail else ""
                lines.append(
                    f"| {resume_name} | {reason} | {ground_truth_file} | {markdown_file} |"
                )
        else:
            lines.append("None.")

        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
