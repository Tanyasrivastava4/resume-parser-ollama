"""
scripts/run_markdown_benchmark.py

CLI entry point for MarkdownBenchmark (Milestone 15A.3).

Instantiates MarkdownBenchmark with its own defaults and calls run(),
then prints the results. This is a thin CLI wrapper — all real
benchmarking logic lives in benchmark/markdown_benchmark.py, which this
script does not modify.

Usage:
    python scripts/run_markdown_benchmark.py
    python -m scripts.run_markdown_benchmark

No arguments are required; MarkdownBenchmark's own defaults (reference
markdown under benchmark/ground_truth/markdown/, generated markdown
under output/markdown/, report under output/benchmark/markdown/) are
used as-is.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

# Run directly (`python scripts/run_markdown_benchmark.py`), the
# script's own directory ends up on sys.path instead of the project
# root, so `benchmark` wouldn't otherwise be importable. Add the
# project root explicitly so this works both that way and as `python
# -m scripts.run_markdown_benchmark` from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmark.markdown_benchmark import (  # noqa: E402
    MarkdownBenchmark,
    MarkdownBenchmarkResult,
)

logger = logging.getLogger(__name__)


def print_benchmark_result(result: MarkdownBenchmarkResult, output_dir: Path) -> None:
    """Print a human-readable summary of a MarkdownBenchmarkResult.

    Args:
        result: The benchmark result to print.
        output_dir: Directory the benchmark saved its report into.
    """
    print("\nMarkdown benchmark summary")
    print("---------------------------")
    print(f"Total resumes:    {result.total_resumes}")
    print(f"Successful:       {result.successful_resumes}")
    print(f"Failed:           {result.failed_resumes}")
    print(f"Processing time:  {result.processing_time:.2f}s")

    print("\nOverall accuracy")
    print("----------------")
    print(f"  Character: {result.overall_character_accuracy:.2%}")
    print(f"  Word:      {result.overall_word_accuracy:.2%}")
    print(f"  Line:      {result.overall_line_accuracy:.2%}")
    print(f"  Heading:   {result.overall_heading_accuracy:.2%}")

    print("\nLine differences")
    print("----------------")
    print(f"  Missing lines:    {result.total_missing_lines}")
    print(f"  Additional lines: {result.total_additional_lines}")
    print(f"  Total differences: {result.total_differences}")

    if result.comparisons:
        print("\nPer resume results")
        print("-------------------")
        for resume_name in sorted(result.comparisons):
            comparison = result.comparisons[resume_name]
            print(
                f"  {resume_name}: char={comparison.character_accuracy:.2%} "
                f"word={comparison.word_accuracy:.2%} "
                f"line={comparison.line_accuracy:.2%} "
                f"heading={comparison.heading_accuracy:.2%}"
            )

    if result.failures:
        print("\nFailures")
        print("--------")
        for resume_name in sorted(result.failures):
            print(f"  {resume_name}: {result.failures[resume_name]}")

    print(f"\nReport saved to: {output_dir}")


def run_markdown_benchmark_cli(
    benchmark: Optional[MarkdownBenchmark] = None,
) -> MarkdownBenchmarkResult:
    """Run MarkdownBenchmark.run() and print the result.

    Args:
        benchmark: MarkdownBenchmark to use. Defaults to a new
            MarkdownBenchmark() built from its own default
            dependencies.

    Returns:
        The MarkdownBenchmarkResult from run().
    """
    benchmark = benchmark or MarkdownBenchmark()
    result = benchmark.run()
    print_benchmark_result(result, benchmark.output_dir)
    return result


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the markdown benchmark CLI.

    Takes no real arguments (MarkdownBenchmark's own defaults are
    always used) — this only exists so `-h`/`--help` works and an
    unexpected argument fails fast with a clear error, instead of
    being silently ignored.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run MarkdownBenchmark against the manually verified reference "
            "markdown dataset. Takes no arguments; all paths come from "
            "MarkdownBenchmark's own defaults."
        )
    )
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    _parse_args()
    run_markdown_benchmark_cli()


if __name__ == "__main__":
    main()
