"""
scripts/_milestone_23_22_batch_run.py

Batched full-benchmark runner for Milestone 23.22 (Reliability
Hardening) -- same resilient pattern established in Milestones
23.18/23.20/23.21 (see project memory's operational note), parameterized
by a run number so this can be invoked 5 separate times (post-fix
validation runs, mirroring Milestone 23.21's own 5-run methodology
exactly, so the two are directly comparable) without one run's output
overwriting another's.

This milestone is a RELIABILITY milestone, not an accuracy milestone --
these runs exist to measure success/failure rate and retry/schema-failure
frequency, not benchmark score. The script itself is accuracy-agnostic
(diagnostic=True is still passed only because ParserEvaluator requires
some mode; the analysis phase reads success/failure and retry data, not
the accuracy numbers).

Usage: pass the run number (1-5) as the first argument. Re-run the exact
same command to continue a run that was interrupted -- only resumes
without a checkpoint for that run are re-evaluated:

    python scripts/_milestone_23_22_batch_run.py 1
    python scripts/_milestone_23_22_batch_run.py 2
    ...

Each run's checkpoints live in /tmp/milestone_23_22/run<N>/batch_checkpoints/,
its final comparison reports in /tmp/milestone_23_22/run<N>/comparison/
(a dedicated output_dir, never benchmark/comparison/), and its full
per-entry diagnostic dump at /tmp/milestone_23_22/run<N>/diagnostic_full.txt.
"""

import logging
import pickle
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.parser_evaluator import EvaluationFailure, ParserEvaluator  # noqa: E402
from scripts.run_parser_evaluation import print_evaluation_result  # noqa: E402
from validators.json_comparator import ENTRY_STATUS_FULL_MATCH  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("milestone_23_22_batch_run")

BATCH_SIZE = 10


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"1", "2", "3", "4", "5"}:
        print("Usage: python scripts/_milestone_23_22_batch_run.py <run_number 1-5>")
        sys.exit(1)
    run_id = sys.argv[1]

    run_dir = Path(f"/tmp/milestone_23_22/run{run_id}")
    checkpoint_dir = run_dir / "batch_checkpoints"
    comparison_dir = run_dir / "comparison"
    diagnostic_out_path = run_dir / "diagnostic_full.txt"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    evaluator = ParserEvaluator(diagnostic=True, output_dir=comparison_dir)
    mapping_rows = evaluator._load_mapping()

    remaining_rows = [
        row
        for row in mapping_rows
        if not (checkpoint_dir / f"{Path(row['Markdown']).stem}.pkl").exists()
    ]
    already_done = len(mapping_rows) - len(remaining_rows)

    if not remaining_rows:
        logger.info(
            "Run %s: all %d resumes already checkpointed -- writing final merged reports.",
            run_id,
            len(mapping_rows),
        )
        _write_final_reports(evaluator, mapping_rows, checkpoint_dir, comparison_dir, diagnostic_out_path)
        return

    batch_rows = remaining_rows[:BATCH_SIZE]
    logger.info(
        "Run %s batch: processing %d resume(s) (%d/%d already checkpointed): %s",
        run_id,
        len(batch_rows),
        already_done,
        len(mapping_rows),
        ", ".join(Path(row["Markdown"]).stem for row in batch_rows),
    )

    for row in batch_rows:
        resume_name = Path(row["Markdown"]).stem
        markdown_path = evaluator.markdown_dir / row["Markdown"]
        ground_truth_path = evaluator.ground_truth_dir / row["JSON"]
        checkpoint_path = checkpoint_dir / f"{resume_name}.pkl"

        start = time.perf_counter()
        try:
            comparison, diagnostic_result = evaluator._evaluate_one(markdown_path, ground_truth_path)
            elapsed = time.perf_counter() - start
            with checkpoint_path.open("wb") as f:
                pickle.dump(
                    {
                        "status": "ok",
                        "comparison": comparison,
                        "diagnostic": diagnostic_result,
                        "elapsed": elapsed,
                    },
                    f,
                )
            logger.info("Run %s: checkpointed %s (ok, %.2fs)", run_id, resume_name, elapsed)
        except Exception as exc:  # noqa: BLE001 - one bad resume must not stop the batch
            elapsed = time.perf_counter() - start
            reason = str(exc)
            logger.error("Run %s: failed to evaluate %s: %s", run_id, resume_name, reason)
            with checkpoint_path.open("wb") as f:
                pickle.dump(
                    {
                        "status": "failed",
                        "reason": reason,
                        "ground_truth_file": str(ground_truth_path),
                        "markdown_file": str(markdown_path),
                        "elapsed": elapsed,
                    },
                    f,
                )

    remaining_after = len(remaining_rows) - len(batch_rows)
    if remaining_after == 0:
        logger.info("Run %s: that was the last batch -- writing final merged reports now.", run_id)
        _write_final_reports(evaluator, mapping_rows, checkpoint_dir, comparison_dir, diagnostic_out_path)
    else:
        logger.info(
            "Run %s batch complete. %d resume(s) still remaining. Re-run this exact script (same run number) to continue.",
            run_id,
            remaining_after,
        )


def _write_final_reports(evaluator, mapping_rows, checkpoint_dir, comparison_dir, diagnostic_out_path) -> None:
    comparisons = {}
    diagnostics = {}
    failures = {}
    failure_details = {}
    total_elapsed = 0.0

    for row in mapping_rows:
        resume_name = Path(row["Markdown"]).stem
        checkpoint_path = checkpoint_dir / f"{resume_name}.pkl"
        with checkpoint_path.open("rb") as f:
            data = pickle.load(f)
        total_elapsed += data.get("elapsed", 0.0)
        if data["status"] == "ok":
            comparisons[resume_name] = data["comparison"]
            if data["diagnostic"] is not None:
                diagnostics[resume_name] = data["diagnostic"]
        else:
            failures[resume_name] = data["reason"]
            failure_details[resume_name] = EvaluationFailure(
                resume=resume_name,
                reason=data["reason"],
                ground_truth_file=data["ground_truth_file"],
                markdown_file=data["markdown_file"],
            )

    result = evaluator._build_result(
        comparisons, failures, failure_details, total_elapsed, diagnostics
    )

    comparison_dir.mkdir(parents=True, exist_ok=True)
    evaluator.json_comparator.generate_reports(comparisons, output_dir=comparison_dir)
    evaluator._write_module_accuracy_csv(comparisons, comparison_dir / "module_accuracy.csv")
    evaluator._write_benchmark_report(result, comparison_dir / "benchmark_report.md")
    evaluator.json_comparator.generate_diagnostic_reports(diagnostics, output_dir=comparison_dir)

    print_evaluation_result(result, comparison_dir)

    diagnostic_out_path.parent.mkdir(parents=True, exist_ok=True)
    with diagnostic_out_path.open("w", encoding="utf-8") as f:
        for resume_name in sorted(diagnostics):
            diag = diagnostics[resume_name]
            for array_name, we_diag in diag.array_diagnostics.items():
                f.write(f"=== {resume_name} :: {array_name} ===\n")
                for i, entry in enumerate(we_diag.entries):
                    f.write(f"  entry {i}: status={entry.status}\n")
                    if entry.status == ENTRY_STATUS_FULL_MATCH:
                        continue
                    for fc in entry.field_comparisons:
                        if fc.matched:
                            continue
                        f.write(f"    field={fc.field_name} expected={fc.expected!r} actual={fc.actual!r}\n")
        for resume_name in sorted(failures):
            f.write(f"=== {resume_name} === EXCLUDED (failed in this run): {failures[resume_name]}\n")
    logger.info("Diagnostic detail written to: %s", diagnostic_out_path)


if __name__ == "__main__":
    main()
