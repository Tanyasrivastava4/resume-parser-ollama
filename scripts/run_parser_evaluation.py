"""
scripts/run_parser_evaluation.py

CLI entry point for ParserEvaluator (Milestone 14.8.1).

Instantiates ParserEvaluator with its own defaults and calls evaluate(),
then prints the results. This is a thin CLI wrapper — all real
evaluation logic lives in evaluation/parser_evaluator.py, which this
milestone does not modify.

Usage:
    python scripts/run_parser_evaluation.py
    python -m scripts.run_parser_evaluation
    python scripts/run_parser_evaluation.py --suite smoke

No arguments are required; ParserEvaluator's own defaults (mapping.csv
under benchmark/ground_truth/json/, markdown under output/markdown/,
reports under benchmark/comparison/) are used as-is.

--suite (Milestone 22.4, Fast Benchmark Infrastructure) restricts the
run to a named subset of resumes defined in
configs/benchmark_suites.yaml (e.g. "smoke", "regression"), resolved
against mapping.csv via scripts/benchmark_suites.py — never a
hardcoded filename, and no change to ParserEvaluator itself. This
works by pointing a ParserEvaluator at a temporary ground_truth_dir
containing only the suite's mapping.csv rows (plus a symlink to each
row's real ground-truth JSON file) — ParserEvaluator's own existing,
unmodified ground_truth_dir constructor override already does the rest.
A suite run's reports are written to benchmark/comparison/<suite>/, not
the default benchmark/comparison/, so a quick suite run never overwrites
a full run's canonical reports. Without --suite, or with --suite full,
behavior is byte-for-byte identical to every prior milestone.

--ground-truth / --ground-truth-dir (remediation R-07, 2026-10-17) select
which ground-truth set to score against. Before R-07 this CLI could only
use config.GROUND_TRUTH_DIR (the legacy benchmark/ground_truth/json/, 23
rows including the resume012 job description), while the project's
canonical frozen baseline (Milestone 23.67) scores against the
human-adjudicated benchmark/ground_truth/adjudicated_v1/ (22 rows) via a
one-off runner script. Now:

    --ground-truth legacy           (default) exactly the pre-R-07 behavior
    --ground-truth adjudicated_v1   benchmark/ground_truth/adjudicated_v1/
    --ground-truth-dir <path>       any directory holding mapping.csv + JSON

A non-default selection writes its reports under
benchmark/comparison/<selection>/[<suite>/] so the legacy reports are
never overwritten, and the printed summary always names the ground-truth
directory and its mapping.csv row count so a report is self-describing.
"""

import argparse
import csv
import logging
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple

# Run directly (`python scripts/run_parser_evaluation.py`), the script's
# own directory ends up on sys.path instead of the project root, so
# `evaluation` wouldn't otherwise be importable. Add the project root
# explicitly so this works both that way and as `python -m
# scripts.run_parser_evaluation` from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (  # noqa: E402
    COMPARISON_DIR,
    DEFAULT_GROUND_TRUTH_NAME,
    GROUND_TRUTH_DIR,
    GROUND_TRUTH_DIRS,
)
from evaluation.parser_evaluator import (  # noqa: E402
    ParserEvaluationResult,
    ParserEvaluator,
)
from scripts.benchmark_suites import is_full_suite, resolve_suite  # noqa: E402

logger = logging.getLogger(__name__)

MAPPING_CSV_FIELDNAMES = ["ResumeID", "PDF", "Markdown", "JSON"]


def _count_mapping_rows(ground_truth_dir: Path) -> Optional[int]:
    """Number of rows in ground_truth_dir/mapping.csv, or None if there is
    no readable mapping.csv there (printing must never fail a run)."""
    mapping_path = Path(ground_truth_dir) / "mapping.csv"
    try:
        with mapping_path.open(newline="", encoding="utf-8") as mapping_file:
            return sum(1 for _ in csv.DictReader(mapping_file))
    except OSError:
        return None


def print_evaluation_result(
    result: ParserEvaluationResult,
    output_dir: Path,
    ground_truth_dir: Optional[Path] = None,
) -> None:
    """Print a human-readable summary of a ParserEvaluationResult.

    Args:
        result: The evaluation result to print.
        output_dir: Directory the evaluator saved its reports into.
        ground_truth_dir: Remediation R-07 -- if given, the ground-truth
            directory the run scored against is printed, with its
            mapping.csv row count, so the summary states which GT it
            means. Optional so every pre-R-07 caller (the milestone
            runner scripts) is unaffected.
    """
    print("\nParser evaluation summary")
    print("--------------------------")
    if ground_truth_dir is not None:
        row_count = _count_mapping_rows(ground_truth_dir)
        rows_text = "mapping.csv unreadable" if row_count is None else f"{row_count} resumes in mapping.csv"
        print(f"Ground truth:     {ground_truth_dir} ({rows_text})")
    print(f"Total resumes:    {result.total_resumes}")
    print(f"Successful:       {result.successful_resumes}")
    print(f"Failed:           {result.failed_resumes}")
    print(f"Overall accuracy: {result.overall_accuracy:.2%}")
    excluded = ", ".join(getattr(result, "optional_keys_excluded", []) or []) or "none"
    print(
        "Strict accuracy excluding optional keys (diagnostic, D8): "
        f"{getattr(result, 'overall_accuracy_excluding_optional_keys', 0.0):.2%} "
        f"[excluded: {excluded}]"
    )
    print(f"Comparator version: {getattr(result, 'comparator_version', '(not recorded)')}")
    print(f"Processing time:  {result.processing_time:.2f}s")

    print("\nModule accuracy")
    print("---------------")
    for module_name in sorted(result.module_accuracy):
        print(f"  {module_name}: {result.module_accuracy[module_name]:.2%}")

    print("\nField accuracy")
    print("--------------")
    for field_path in sorted(result.field_accuracy):
        print(f"  {field_path}: {result.field_accuracy[field_path]:.2%}")

    print("\nResume accuracy")
    print("---------------")
    for resume_name in sorted(result.resume_accuracy):
        print(f"  {resume_name}: {result.resume_accuracy[resume_name]:.2%}")

    if result.failures:
        print("\nFailures")
        print("--------")
        for resume_name in sorted(result.failures):
            detail = result.failure_details.get(resume_name)
            if detail:
                print(f"  Resume:             {detail.resume}")
                print(f"  Reason:             {detail.reason}")
                print(f"  Ground truth file:  {detail.ground_truth_file}")
                print(f"  Markdown file:      {detail.markdown_file}")
                print()
            else:
                print(f"  {resume_name}: {result.failures[resume_name]}")

    print(f"\nReports saved to: {output_dir}")


def resolve_ground_truth_dir(
    ground_truth: Optional[str] = None,
    ground_truth_dir: Optional[Path] = None,
) -> Tuple[Path, str]:
    """Remediation R-07: turn the CLI's ground-truth selection into a
    concrete directory plus a short label used to scope report output.

    Args:
        ground_truth: A key of config.GROUND_TRUTH_DIRS ("legacy",
            "adjudicated_v1"), or None for the default ("legacy").
        ground_truth_dir: An explicit directory containing mapping.csv
            and the ground-truth JSON files. Mutually exclusive with
            `ground_truth`.

    Returns:
        (directory, label). label is the selection name, or "custom" for
        an explicit directory. For the default selection the directory
        is exactly config.GROUND_TRUTH_DIR -- callers use that to keep
        the pre-R-07 code path byte-for-byte unchanged.

    Raises:
        ValueError: both arguments given, or an unknown selection name.
    """
    if ground_truth is not None and ground_truth_dir is not None:
        raise ValueError("Pass either ground_truth or ground_truth_dir, not both.")
    if ground_truth_dir is not None:
        return Path(ground_truth_dir), "custom"
    name = DEFAULT_GROUND_TRUTH_NAME if ground_truth is None else ground_truth
    if name not in GROUND_TRUTH_DIRS:
        raise ValueError(
            f"Unknown ground-truth selection {name!r}; known: {', '.join(GROUND_TRUTH_DIRS)}."
        )
    return GROUND_TRUTH_DIRS[name], name


def _is_default_ground_truth(ground_truth_dir: Path) -> bool:
    return Path(ground_truth_dir) == Path(GROUND_TRUTH_DIR)


@contextmanager
def _suite_scoped_evaluator(
    suite: str,
    ground_truth_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Iterator[ParserEvaluator]:
    """Build a ParserEvaluator scoped to `suite` (Milestone 22.4).

    Resolves `suite` via scripts.benchmark_suites.resolve_suite (which
    reads mapping.csv — no filename is ever hardcoded here), then
    writes a temporary mapping.csv containing only those rows, plus a
    symlink to each row's real ground-truth JSON file so
    SchemaValidator/JsonComparator read the exact same files a full run
    would. ParserEvaluator itself is used exactly as-is, via its own,
    already-existing ground_truth_dir/output_dir constructor overrides
    — nothing about evaluation/parser_evaluator.py changes.

    Args:
        suite: A suite name from configs/benchmark_suites.yaml, not
            "full" (callers should use a plain `ParserEvaluator()` for
            "full" / no suite — see run_parser_evaluation_cli).
        ground_truth_dir: Remediation R-07 -- the ground-truth directory
            whose mapping.csv the suite is resolved against and whose JSON
            files are symlinked. None (the default) means config.GROUND_TRUTH_DIR
            -- resolved at call time, not definition time -- the pre-R-07
            behavior.
        output_dir: Remediation R-07 -- where the scoped evaluator writes
            its reports. Defaults to benchmark/comparison/<suite>/, the
            pre-R-07 behavior.

    Yields:
        A ParserEvaluator whose ground_truth_dir is the temporary,
        suite-scoped directory described above, and whose output_dir is
        benchmark/comparison/<suite>/ — so a suite run's reports never
        overwrite the full run's canonical benchmark/comparison/
        reports. The temporary ground_truth_dir is deleted when the
        context exits.

    Raises:
        Any exception scripts.benchmark_suites.resolve_suite raises
        (unknown suite name, missing configs/benchmark_suites.yaml,
        missing mapping.csv, or a suite member mapping.csv has no row
        for).
    """
    ground_truth_dir = Path(GROUND_TRUTH_DIR if ground_truth_dir is None else ground_truth_dir)
    if output_dir is None:
        output_dir = COMPARISON_DIR / suite
    if _is_default_ground_truth(ground_truth_dir):
        rows = resolve_suite(suite)
    else:
        rows = resolve_suite(suite, ground_truth_dir=ground_truth_dir)
    with tempfile.TemporaryDirectory(prefix=f"benchmark_suite_{suite}_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        with (temp_dir / "mapping.csv").open("w", newline="", encoding="utf-8") as mapping_file:
            writer = csv.DictWriter(mapping_file, fieldnames=MAPPING_CSV_FIELDNAMES)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
                (temp_dir / row["JSON"]).symlink_to((ground_truth_dir / row["JSON"]).resolve())

        logger.info(
            "Resolved %d resume(s) for suite '%s' against %s; reports will be written to %s",
            len(rows),
            suite,
            ground_truth_dir,
            output_dir,
        )
        yield ParserEvaluator(ground_truth_dir=temp_dir, output_dir=output_dir)


def run_parser_evaluation_cli(
    evaluator: Optional[ParserEvaluator] = None,
    suite: Optional[str] = None,
    ground_truth: Optional[str] = None,
    ground_truth_dir: Optional[Path] = None,
) -> ParserEvaluationResult:
    """Run ParserEvaluator.evaluate() and print the result.

    Args:
        evaluator: ParserEvaluator to use. Defaults to a new
            ParserEvaluator() built from its own default dependencies.
            When given explicitly, `suite` is ignored — an
            explicitly-injected evaluator (e.g. in tests) always wins.
            Its evaluation behaviour is unchanged by R-07; the only
            difference is that the printed summary now also names the
            evaluator's ground_truth_dir when it has one (output only).
        suite: Milestone 22.4. If given and not "full", restricts the
            run to the named subset from configs/benchmark_suites.yaml
            (see _suite_scoped_evaluator). `None` (the default) or
            "full" preserve the original, pre-Milestone-22.4 behavior
            exactly — a plain `ParserEvaluator()` over every resume in
            mapping.csv.
        ground_truth / ground_truth_dir: Remediation R-07 -- which
            ground-truth set to score against (see
            resolve_ground_truth_dir). Both None (the default) means the
            legacy set and the exact pre-R-07 code path: a plain
            `ParserEvaluator()` / the original suite scoping. Any other
            selection builds the evaluator with that ground_truth_dir and
            writes reports under benchmark/comparison/<label>/[<suite>/]
            so the legacy reports are never overwritten.

    Returns:
        The ParserEvaluationResult from evaluate().
    """
    if evaluator is not None:
        result = evaluator.evaluate()
        print_evaluation_result(
            result, evaluator.output_dir, getattr(evaluator, "ground_truth_dir", None)
        )
        return result

    selected_dir, label = resolve_ground_truth_dir(ground_truth, ground_truth_dir)
    default_selection = _is_default_ground_truth(selected_dir)

    if is_full_suite(suite):
        if default_selection:
            evaluator = ParserEvaluator()
        else:
            evaluator = ParserEvaluator(
                ground_truth_dir=selected_dir, output_dir=COMPARISON_DIR / label
            )
        result = evaluator.evaluate()
        print_evaluation_result(result, evaluator.output_dir, selected_dir)
        return result

    if default_selection:
        scoped = _suite_scoped_evaluator(suite)
    else:
        scoped = _suite_scoped_evaluator(
            suite, ground_truth_dir=selected_dir, output_dir=COMPARISON_DIR / label / suite
        )
    with scoped as scoped_evaluator:
        result = scoped_evaluator.evaluate()
        print_evaluation_result(result, scoped_evaluator.output_dir, selected_dir)
        return result


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the parser evaluation CLI.

    Only --suite is a real argument (Milestone 22.4); every other path
    still comes from ParserEvaluator's own defaults, or from the
    suite's resolved, temporary ground_truth_dir when --suite is given.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run ParserEvaluator against the ground-truth dataset (or a named "
            "subset of it via --suite). All paths otherwise come from "
            "ParserEvaluator's own defaults."
        )
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help=(
            "Milestone 22.4: restrict the run to a named subset from "
            "configs/benchmark_suites.yaml (e.g. smoke, regression), "
            "resolved against mapping.csv. Omit, or pass 'full', to "
            "evaluate every resume in mapping.csv, exactly as before "
            "this option existed."
        ),
    )
    ground_truth_group = parser.add_mutually_exclusive_group()
    ground_truth_group.add_argument(
        "--ground-truth",
        type=str,
        choices=sorted(GROUND_TRUTH_DIRS),
        default=None,
        help=(
            "Remediation R-07: which ground-truth set to score against. "
            f"Default: '{DEFAULT_GROUND_TRUTH_NAME}' ({GROUND_TRUTH_DIR}), exactly as "
            "before this option existed. 'adjudicated_v1' is the human-adjudicated "
            "set the Milestone 23.67 frozen baseline uses. A non-default selection "
            "writes reports under benchmark/comparison/<selection>/."
        ),
    )
    ground_truth_group.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=None,
        help=(
            "Remediation R-07: an explicit directory containing mapping.csv and "
            "the ground-truth JSON files (reports go to benchmark/comparison/custom/). "
            "Mutually exclusive with --ground-truth."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    run_parser_evaluation_cli(
        suite=args.suite,
        ground_truth=args.ground_truth,
        ground_truth_dir=args.ground_truth_dir,
    )


if __name__ == "__main__":
    main()
