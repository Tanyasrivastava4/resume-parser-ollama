"""
scripts/_milestone_23_18_batch_run.py

Batched variant of scripts/_milestone_23_18_diagnostic_full_run.py, so
a crash/stop only loses the *current* batch's progress (at most
BATCH_SIZE resumes' pipeline runs), not the whole 23-resume run.

Usage: run this exact command repeatedly (once per batch):

    python scripts/_milestone_23_18_batch_run.py

Each invocation:
  1. Looks at /tmp/milestone_23_18/batch_checkpoints/ to see which
     resumes (from benchmark/ground_truth/json/mapping.csv) already
     have a checkpoint from a *previous* invocation.
  2. Runs the full ParserPipeline + comparison for up to BATCH_SIZE of
     the resumes that don't yet have one, pickling each one's
     ComparisonResult/DiagnosticResult to its own checkpoint file the
     moment it finishes -- so if this process is killed mid-batch, any
     resume that already got a checkpoint written is never re-run.
  3. Once every resume in mapping.csv has a checkpoint, the next
     invocation instead loads all of them, merges them into one
     ParserEvaluationResult exactly the way
     ParserEvaluator.evaluate() would have if it had run the whole
     dataset in one uninterrupted pass, and writes the same reports
     scripts/_milestone_23_18_diagnostic_full_run.py does
     (benchmark_report.md, diagnostic_report.md, etc. in
     benchmark/comparison/, plus the plain-text work_experience
     diagnostic dump at /tmp/milestone_23_18/diagnostic_full_after.txt).

To start a fresh run from scratch, delete
/tmp/milestone_23_18/batch_checkpoints/ first.

Not a permanent addition to scripts/ -- a scratch script for this
milestone's one measurement run, kept only for the paper trail.
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
logger = logging.getLogger("milestone_23_18_batch_run")

BATCH_SIZE = 5
CHECKPOINT_DIR = Path("/tmp/milestone_23_18/batch_checkpoints")
DIAGNOSTIC_OUT_PATH = Path("/tmp/milestone_23_18/diagnostic_full_after.txt")


def main() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    evaluator = ParserEvaluator(diagnostic=True)
    mapping_rows = evaluator._load_mapping()

    remaining_rows = [
        row
        for row in mapping_rows
        if not (CHECKPOINT_DIR / f"{Path(row['Markdown']).stem}.pkl").exists()
    ]
    already_done = len(mapping_rows) - len(remaining_rows)

    if not remaining_rows:
        logger.info(
            "All %d resumes already checkpointed -- writing final merged reports.",
            len(mapping_rows),
        )
        _write_final_reports(evaluator, mapping_rows)
        return

    batch_rows = remaining_rows[:BATCH_SIZE]
    logger.info(
        "Batch: processing %d resume(s) (%d/%d already checkpointed): %s",
        len(batch_rows),
        already_done,
        len(mapping_rows),
        ", ".join(Path(row["Markdown"]).stem for row in batch_rows),
    )

    for row in batch_rows:
        resume_name = Path(row["Markdown"]).stem
        markdown_path = evaluator.markdown_dir / row["Markdown"]
        ground_truth_path = evaluator.ground_truth_dir / row["JSON"]
        checkpoint_path = CHECKPOINT_DIR / f"{resume_name}.pkl"

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
            logger.info("Checkpointed %s (ok, %.2fs)", resume_name, elapsed)
        except Exception as exc:  # noqa: BLE001 - one bad resume must not stop the batch
            elapsed = time.perf_counter() - start
            reason = str(exc)
            logger.error("Failed to evaluate %s: %s", resume_name, reason)
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
        logger.info("That was the last batch -- writing final merged reports now.")
        _write_final_reports(evaluator, mapping_rows)
    else:
        logger.info(
            "Batch complete. %d resume(s) still remaining. Re-run this exact script to continue.",
            remaining_after,
        )


def _write_final_reports(evaluator: ParserEvaluator, mapping_rows: list) -> None:
    comparisons = {}
    diagnostics = {}
    failures = {}
    failure_details = {}
    total_elapsed = 0.0

    for row in mapping_rows:
        resume_name = Path(row["Markdown"]).stem
        checkpoint_path = CHECKPOINT_DIR / f"{resume_name}.pkl"
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

    evaluator.output_dir.mkdir(parents=True, exist_ok=True)
    evaluator.json_comparator.generate_reports(comparisons, output_dir=evaluator.output_dir)
    evaluator._write_module_accuracy_csv(comparisons, evaluator.output_dir / "module_accuracy.csv")
    evaluator._write_benchmark_report(result, evaluator.output_dir / "benchmark_report.md")
    evaluator.json_comparator.generate_diagnostic_reports(diagnostics, output_dir=evaluator.output_dir)

    print_evaluation_result(result, evaluator.output_dir)

    DIAGNOSTIC_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DIAGNOSTIC_OUT_PATH.open("w", encoding="utf-8") as f:
        for resume_name in sorted(diagnostics):
            diag = diagnostics[resume_name]
            we_diag = diag.array_diagnostics.get("work_experience")
            f.write(f"=== {resume_name} ===\n")
            if we_diag is None:
                f.write("  (no work_experience array_diagnostic)\n")
                continue
            for i, entry in enumerate(we_diag.entries):
                f.write(f"  entry {i}: status={entry.status}\n")
                if entry.status == ENTRY_STATUS_FULL_MATCH:
                    continue
                for fc in entry.field_comparisons:
                    if fc.matched:
                        continue
                    f.write(f"    field={fc.field_name} expected={fc.expected!r} actual={fc.actual!r}\n")
        for resume_name in sorted(failures):
            f.write(f"=== {resume_name} === EXCLUDED (failed in this benchmark run)\n")
    logger.info("Diagnostic detail written to: %s", DIAGNOSTIC_OUT_PATH)


if __name__ == "__main__":
    main()
