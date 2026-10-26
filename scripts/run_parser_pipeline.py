"""
scripts/run_parser_pipeline.py

CLI entry point for ParserPipeline (Milestone 14.8.1; batching added in
Milestone 14.9.4; ResumeRetryQueue integration added in Milestone 18.7).

Finds every *.md file under --input-dir (recursively) and runs
ParserPipeline.run() on each, continuing past any single resume's
failure. This is a thin CLI wrapper — all real parsing logic lives in
pipeline/parser_pipeline.py and pipeline/resume_retry_queue.py, neither
of which this milestone modifies.

Usage:
    python scripts/run_parser_pipeline.py [--input-dir DIR] [--output-dir DIR] [--batch-size N] [--suite NAME]
    python -m scripts.run_parser_pipeline [--input-dir DIR] [--output-dir DIR] [--batch-size N] [--suite NAME]

--batch-size splits the markdown files into batches of that size,
completing one batch fully (including its own printed statistics) before
starting the next — useful for checking progress on a long run without
waiting for every resume to finish. Without it, every markdown file is
processed as a single batch, exactly as before Milestone 14.9.4.

--suite (Milestone 22.4, Fast Benchmark Infrastructure) restricts the
run to a named subset of resumes defined in
configs/benchmark_suites.yaml (e.g. "smoke", "regression"), resolved
against benchmark/ground_truth/json/mapping.csv via
scripts/benchmark_suites.py — never a hardcoded filename. Without
--suite, or with --suite full, behavior is byte-for-byte identical to
every prior milestone: every *.md file under --input-dir is processed,
exactly as before this milestone existed.

Note on --output-dir: historically ParserPipeline.run() always saved its
own output to output/json_final/<resume>.json, a destination fixed inside
pipeline/parser_pipeline.py; --output-dir was therefore only ever used to
describe where results land in the final summary, and passing a different
value never changed where anything was saved (a warning is logged if it
doesn't match). That remains true of --output-dir.

--output-root DIR (remediation R-08, 2026-10-17) is the real redirect:
ParserPipeline is built with output_root=DIR, so module JSON lands in
DIR/modules/<resume>/ and final JSON in DIR/json_final/<resume>.json, and
--persist-review-reports writes under DIR/review_reports/. Omitting it
(every pre-R-08 invocation) keeps the fixed output/ locations exactly.
When --output-root is given, the summary reports DIR/json_final.

Milestone V5.3 — optional review-report persistence:
--persist-review-reports (default off, fully backward compatible) additionally
writes evaluation.review_result_contract.write_review_result_contract()'s
output to output/review_reports/<resume>.json for every resume that
succeeds (initial pass or retry queue) -- a purely additive, opt-in
artifact built from the SAME ParserPipelineResult pipeline.run() already
returned, never a second pipeline.run() call and never an LLM call.
This never touches output/json_final/<resume>.json or
output/modules/<resume>/*.json -- those are written exactly as before,
by ParserPipeline itself, completely unaware this flag exists. A
persistence failure for one resume (e.g. a filesystem error) is caught,
logged, and does not affect that resume's own success/failure status --
the resume already succeeded at parsing; only the separate, optional
review-artifact write failed. Without this flag, this script's behavior
is byte-for-byte identical to every prior milestone.

Milestone 18.7 — ResumeRetryQueue integration:
docs/PARSER_RELIABILITY_DESIGN.md (Milestone 18.6) found that
ResumeRetryQueue — a whole-resume retry layer built and tested since
Milestone 14.9.3, on top of ParserPipeline's own per-module retries —
was never actually wired into this, the real production entry point
(it was only used in benchmark/markdown_benchmark.py). This script now
runs in two passes: every markdown file is first attempted once via the
existing per-batch loop (unchanged — this already includes
ParserPipeline's own per-module retries); any resume that still fails
is queued, not retried immediately, so the rest of the initial batch
keeps moving. Once every batch has been attempted, the queued resumes
are handed to a single ResumeRetryQueue (reusing the exact same
ParserPipeline instance, and ResumeRetryQueue's own unmodified retry
logic — see pipeline/resume_retry_queue.py). The final printed report
distinguishes resumes that succeeded on the initial pass, resumes that
only succeeded once queued for retry, and resumes that still failed
after both layers of retry are exhausted.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

# Run directly (`python scripts/run_parser_pipeline.py`), the script's
# own directory ends up on sys.path instead of the project root, so
# `config` and `pipeline` wouldn't otherwise be importable. Add the
# project root explicitly so this works both that way and as `python -m
# scripts.run_parser_pipeline` from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tqdm import tqdm  # noqa: E402

from config import OUTPUT_DIR  # noqa: E402
from evaluation.review_result_contract import write_review_result_contract  # noqa: E402
from pipeline.parser_pipeline import ParserPipeline  # noqa: E402
from pipeline.resume_retry_queue import ResumeRetryQueue  # noqa: E402
from scripts.benchmark_suites import is_full_suite, resolve_suite  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = OUTPUT_DIR / "markdown"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "json_final"


def find_markdown_files(input_dir: Path) -> List[Path]:
    """Return every *.md file under `input_dir`, recursively, sorted for
    a stable order.

    Args:
        input_dir: Directory to scan.

    Returns:
        A sorted list of markdown file paths. Empty if `input_dir`
        doesn't exist.
    """
    if not input_dir.is_dir():
        logger.warning("Input directory does not exist: %s", input_dir)
        return []
    return sorted(input_dir.rglob("*.md"))


def find_suite_markdown_files(input_dir: Path, suite: str) -> List[Path]:
    """Resolve `suite` (Milestone 22.4) to the markdown files it covers.

    Args:
        input_dir: Directory the resolved filenames are relative to —
            the same directory `find_markdown_files` would otherwise
            scan.
        suite: A suite name from configs/benchmark_suites.yaml,
            resolved against benchmark/ground_truth/json/mapping.csv
            (see scripts/benchmark_suites.py) — never a hardcoded
            filename.

    Returns:
        A sorted list of markdown file paths, one per resume the suite
        names. A path is included even if the file doesn't actually
        exist under `input_dir`; ParserPipeline.run() raising
        MarkdownNotFoundError for it is handled by the existing
        per-resume failure handling in `_process_batch`, the same as
        any other missing-file failure.

    Raises:
        Any exception scripts.benchmark_suites.resolve_suite raises
        (unknown suite name, missing configs/benchmark_suites.yaml,
        missing mapping.csv, or a suite member mapping.csv has no row
        for).
    """
    rows = resolve_suite(suite)
    return sorted(input_dir / row["Markdown"] for row in rows)


def _batchify(markdown_paths: List[Path], batch_size: Optional[int]) -> List[List[Path]]:
    """Split `markdown_paths` into batches of `batch_size`.

    Args:
        markdown_paths: Files to split, in the order they'll be
            processed.
        batch_size: Files per batch. `None` or a non-positive value
            means "one batch containing everything" (i.e. unbatched).

    Returns:
        A list of batches (each a list of paths), in order. Empty if
        `markdown_paths` is empty.
    """
    if not markdown_paths:
        return []
    if not batch_size or batch_size <= 0:
        return [markdown_paths]
    return [
        markdown_paths[start : start + batch_size]
        for start in range(0, len(markdown_paths), batch_size)
    ]


def _persist_review_report_if_enabled(
    markdown_path: Path,
    result: Any,
    persist_review_reports: bool,
    output_root: Path = OUTPUT_DIR,
) -> None:
    """Milestone V5.3: if `persist_review_reports`, write this resume's
    review-result contract (built from the SAME already-completed
    `result` — no second pipeline.run() call, no LLM call) to
    output/review_reports/<stem>.json.

    A failure here (e.g. a filesystem error) is caught and logged, not
    raised — the resume's own parsing already succeeded by the time
    this is called; only this separate, optional artifact write is at
    risk, and it must never turn an already-successful resume into a
    reported failure.

    No-op entirely when `persist_review_reports` is False — the exact
    same "do nothing unless explicitly asked" behavior every other
    optional flag in this script (--batch-size, --suite) already has.
    """
    if not persist_review_reports:
        return
    stem = markdown_path.stem
    try:
        path = write_review_result_contract(stem, stem, result, output_root)
        logger.info("Persisted review report for %s -> %s", markdown_path, path)
    except OSError as exc:
        logger.error("Failed to persist review report for %s: %s", markdown_path, exc)


def _process_batch(
    batch: List[Path],
    pipeline: ParserPipeline,
    batch_index: int,
    total_batches: int,
    persist_review_reports: bool = False,
    review_root: Path = OUTPUT_DIR,
) -> Tuple[int, List[Path], List[float]]:
    """Run `pipeline` over one batch, continuing past any single
    resume's failure.

    Args:
        review_root: Remediation R-08 -- root directory review reports are
            written under (default config.OUTPUT_DIR, the pre-R-08 behavior).
        persist_review_reports: Milestone V5.3, default False (fully
            backward compatible). See _persist_review_report_if_enabled().

    Returns:
        (successful, failed_paths, processing_times) for this batch.
        `failed_paths` lists every resume whose `pipeline.run()` call
        raised, in the order encountered — Milestone 18.7 queues these
        for a second pass via ResumeRetryQueue once every batch has
        been attempted (see run_parser_pipeline_cli), rather than only
        counting failures as before Milestone 18.7.
    """
    successful = 0
    failed_paths: List[Path] = []
    processing_times: List[float] = []

    desc = f"Batch {batch_index}/{total_batches}"
    for markdown_path in tqdm(batch, desc=desc, unit="resume"):
        try:
            result = pipeline.run(markdown_path)
            processing_times.append(result.processing_time)
            successful += 1
            logger.info("Parsed %s", markdown_path)
            _persist_review_report_if_enabled(
                markdown_path, result, persist_review_reports, review_root
            )
        except Exception as exc:  # noqa: BLE001 - one bad resume must not stop the run
            logger.error("Failed to parse %s: %s", markdown_path, exc)
            failed_paths.append(markdown_path)

    return successful, failed_paths, processing_times


def _process_retry_queue(
    failed_paths: List[Path],
    retry_queue: ResumeRetryQueue,
    persist_review_reports: bool = False,
    review_root: Path = OUTPUT_DIR,
) -> Tuple[int, List[Tuple[Path, str]], List[float]]:
    """Run `retry_queue` over every resume that failed its initial
    attempt, continuing past any single resume's failure — the same
    continue-past-failures policy `_process_batch` already uses for the
    initial pass.

    `retry_queue` is used exactly as designed (see
    pipeline/resume_retry_queue.py, not modified by this milestone): a
    resume queued here gets one fresh whole-resume attempt, retried
    once more internally by ResumeRetryQueue itself if that attempt
    fails with a retryable error. This function does not add any
    further retry logic of its own.

    Args:
        failed_paths: Markdown paths that failed their initial
            `ParserPipeline.run()` attempt (across every batch).
        retry_queue: A ResumeRetryQueue wrapping the same
            ParserPipeline instance the initial pass used.
        persist_review_reports: Milestone V5.3, default False. See
            _persist_review_report_if_enabled().

    Returns:
        (retried_successful, permanent_failures, processing_times).
        `permanent_failures` is a list of (markdown_path, reason)
        pairs — resumes still failing after ResumeRetryQueue's own
        retry was exhausted, or that failed with a non-retryable error
        ResumeRetryQueue passed straight through.
    """
    retried_successful = 0
    permanent_failures: List[Tuple[Path, str]] = []
    processing_times: List[float] = []

    for markdown_path in tqdm(failed_paths, desc="Retry queue", unit="resume"):
        try:
            result = retry_queue.run(markdown_path)
            processing_times.append(result.processing_time)
            retried_successful += 1
            logger.info("Retry queue succeeded for %s", markdown_path)
            _persist_review_report_if_enabled(
                markdown_path, result, persist_review_reports, review_root
            )
        except Exception as exc:  # noqa: BLE001 - one bad resume must not stop the run
            logger.error("Retry queue failed for %s: %s", markdown_path, exc)
            permanent_failures.append((markdown_path, str(exc)))

    return retried_successful, permanent_failures, processing_times


def _print_summary(
    heading: str, total: int, successful: int, failed: int, processing_times: List[float]
) -> None:
    """Print a Total/Success/Failed/Average time block under `heading`."""
    average_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
    print(f"\n{heading}")
    print("-" * len(heading))
    print(f"Total:            {total}")
    print(f"Success:          {successful}")
    print(f"Failed:           {failed}")
    print(f"Average time:     {average_time:.4f}s")


def _print_final_report(
    initial_successful: int,
    retried_successful: int,
    permanent_failures: List[Tuple[Path, str]],
) -> None:
    """Print the final report distinguishing initial successes, resumes
    that only succeeded once queued for retry, and resumes that failed
    permanently — still failing after both ParserPipeline's own
    per-module retries and ResumeRetryQueue's whole-resume retry."""
    total = initial_successful + retried_successful + len(permanent_failures)
    heading = "Final Report"
    print(f"\n{heading}")
    print("-" * len(heading))
    print(f"Total resumes:        {total}")
    print(f"Initial successes:    {initial_successful}")
    print(f"Retried successes:    {retried_successful}")
    print(f"Permanent failures:   {len(permanent_failures)}")
    if permanent_failures:
        print("\nPermanently failed resumes:")
        for markdown_path, reason in permanent_failures:
            print(f"  - {markdown_path}: {reason}")


def run_parser_pipeline_cli(
    input_dir: Path,
    output_dir: Path,
    pipeline: Optional[ParserPipeline] = None,
    batch_size: Optional[int] = None,
    suite: Optional[str] = None,
    persist_review_reports: bool = False,
    output_root: Optional[Path] = None,
) -> None:
    """Run ParserPipeline over every markdown file under `input_dir`,
    continuing past any single resume's failure, then hand any resume
    that still failed to a ResumeRetryQueue once every batch has been
    attempted, and print a final report.

    Args:
        input_dir: Directory to scan for *.md files, recursively.
        output_dir: Directory to report as where results were saved
            (see the module docstring's note on --output-dir).
        pipeline: ParserPipeline to use. Defaults to a new
            ParserPipeline() built from its own default dependencies.
            The same instance is reused for the ResumeRetryQueue phase.
        batch_size: If given, files are processed in batches of this
            size — one batch is completed (and its own statistics
            printed) before the next one starts. Defaults to `None`,
            meaning every file is processed as a single batch.
        suite: Milestone 22.4. If given and not "full", restricts the
            run to the named subset from configs/benchmark_suites.yaml,
            resolved via scripts/benchmark_suites.py against
            mapping.csv, instead of scanning `input_dir` for every
            *.md file. `None` (the default) or "full" preserve the
            original, pre-Milestone-22.4 behavior exactly — a full
            directory scan.
        persist_review_reports: Milestone V5.3, default False (fully
            backward compatible). If True, additionally writes a
            review-result contract (evaluation.review_result_contract)
            to output/review_reports/<resume>.json for every resume
            that succeeds, built from the SAME ParserPipelineResult
            pipeline.run() already returned — no second pipeline.run()
            call, no LLM call. Never affects output/json_final/ or
            output/modules/, and a persistence failure for one resume
            never turns that resume into a reported failure (see
            _persist_review_report_if_enabled()).
        output_root: Remediation R-08. If given (and `pipeline` is not
            injected), the ParserPipeline is built with this
            output_root, so everything it saves -- and the review
            reports -- land under it instead of the fixed output/
            locations, and `output_dir` is reported as
            <output_root>/json_final. None (the default) is the exact
            pre-R-08 behavior.
    """
    if output_root is not None:
        output_root = Path(output_root)
        output_dir = output_root / "json_final"
    elif output_dir != DEFAULT_OUTPUT_DIR:
        logger.warning(
            "ParserPipeline always saves its own output to %s; --output-dir "
            "only affects what this summary reports, not where results are "
            "actually saved (use --output-root to redirect).",
            DEFAULT_OUTPUT_DIR,
        )
    review_root = OUTPUT_DIR if output_root is None else output_root

    if pipeline is None:
        pipeline = ParserPipeline() if output_root is None else ParserPipeline(output_root=output_root)
    if is_full_suite(suite):
        markdown_paths = find_markdown_files(input_dir)
        logger.info("Found %d markdown file(s) under %s", len(markdown_paths), input_dir)
    else:
        markdown_paths = find_suite_markdown_files(input_dir, suite)
        logger.info(
            "Resolved %d markdown file(s) for suite '%s' under %s",
            len(markdown_paths),
            suite,
            input_dir,
        )

    batches = _batchify(markdown_paths, batch_size)
    total_batches = len(batches)

    overall_successful = 0
    overall_failed_paths: List[Path] = []
    overall_processing_times: List[float] = []

    for batch_index, batch in enumerate(batches, start=1):
        print(f"\nBatch {batch_index}/{total_batches}")
        successful, failed_paths, processing_times = _process_batch(
            batch, pipeline, batch_index, total_batches, persist_review_reports, review_root
        )

        overall_successful += successful
        overall_failed_paths.extend(failed_paths)
        overall_processing_times.extend(processing_times)

        _print_summary(
            f"Batch {batch_index}/{total_batches} summary",
            len(batch),
            successful,
            len(failed_paths),
            processing_times,
        )

    _print_summary(
        "Parser pipeline summary",
        len(markdown_paths),
        overall_successful,
        len(overall_failed_paths),
        overall_processing_times,
    )

    # Milestone 18.7: every resume that failed its initial attempt,
    # across every batch, is queued here and handed to a
    # ResumeRetryQueue — reusing `pipeline` (the exact same instance
    # the initial pass used) and ResumeRetryQueue's own, unmodified
    # retry logic — only after the entire initial pass has finished,
    # so one resume's retry never blocks the rest of the batch.
    retried_successful = 0
    permanent_failures: List[Tuple[Path, str]] = []
    if overall_failed_paths:
        print(f"\nRetry Queue ({len(overall_failed_paths)} resume(s))")
        retry_queue = ResumeRetryQueue(parser_pipeline=pipeline)
        retried_successful, permanent_failures, retry_processing_times = _process_retry_queue(
            overall_failed_paths, retry_queue, persist_review_reports, review_root
        )
        _print_summary(
            "Retry queue summary",
            len(overall_failed_paths),
            retried_successful,
            len(permanent_failures),
            retry_processing_times,
        )

    _print_final_report(overall_successful, retried_successful, permanent_failures)
    print(f"Output directory: {output_dir}")
    if persist_review_reports:
        print(f"Review reports directory: {review_root / 'review_reports'}")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the parser pipeline CLI."""
    parser = argparse.ArgumentParser(
        description="Run ParserPipeline over every markdown file in a directory."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory to scan for *.md files, recursively. Defaults to {DEFAULT_INPUT_DIR}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory to report as where ParserPipeline saved its output. "
            f"Defaults to {DEFAULT_OUTPUT_DIR}, which is also the only place "
            "ParserPipeline actually saves to (see the module docstring)."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Remediation R-08: write ALL pipeline output under this directory "
            "(<root>/modules/<resume>/, <root>/json_final/<resume>.json and, with "
            "--persist-review-reports, <root>/review_reports/) instead of the "
            f"fixed {OUTPUT_DIR}. Omit to keep the fixed locations exactly as before."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Process markdown files in batches of this size, completing "
            "one batch (and printing its own statistics) before starting "
            "the next. Defaults to processing every file as one batch."
        ),
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help=(
            "Milestone 22.4: restrict the run to a named subset from "
            "configs/benchmark_suites.yaml (e.g. smoke, regression), "
            "resolved against mapping.csv. Omit, or pass 'full', to "
            "process every *.md file under --input-dir, exactly as "
            "before this option existed."
        ),
    )
    parser.add_argument(
        "--persist-review-reports",
        action="store_true",
        default=False,
        help=(
            "Milestone V5.3: additionally write a review-result contract "
            "(evaluation.review_result_contract) to "
            "output/review_reports/<resume>.json for every resume that "
            "succeeds, built from the same already-completed parse -- no "
            "extra pipeline run, no LLM call. Off by default; omitting "
            "this flag preserves this script's exact pre-V5.3 behavior."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    run_parser_pipeline_cli(
        args.input_dir,
        args.output_dir,
        batch_size=args.batch_size,
        suite=args.suite,
        persist_review_reports=args.persist_review_reports,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
