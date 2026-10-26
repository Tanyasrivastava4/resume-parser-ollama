"""
scripts/_milestone_23_67_frozen_baseline_run.py

Frozen-parser baseline run for Milestone 23.67 -- evaluates the CURRENT,
unmodified ParserPipeline against the human-adjudicated GT
(benchmark/ground_truth/adjudicated_v1/), for the 22 genuine legacy
resumes only (resume012 permanently excluded).

Same resilient checkpoint-per-resume pattern established in
scripts/_milestone_23_22_batch_run.py: each resume's evaluation result
(or failure) is pickled to its own checkpoint file as soon as it
completes, so an interrupted run can be resumed by simply re-running this
script -- only resumes without a checkpoint are re-evaluated. Re-running
this script does NOT re-evaluate already-checkpointed resumes and does
NOT constitute "manually retrying a failed resume outside the pipeline"
(a completed FAILURE checkpoint is still a checkpoint; it is not retried
by this script).

Nothing here modifies parser/prompt/schema/validator/comparator source.
This script only orchestrates existing, unmodified components.

Usage: python scripts/_milestone_23_67_frozen_baseline_run.py
"""

import logging
import pickle
import time
from pathlib import Path

import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.parser_evaluator import EvaluationFailure, ParserEvaluator  # noqa: E402
from scripts.run_parser_evaluation import print_evaluation_result  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("milestone_23_67_frozen_baseline")

RUN_DIR = Path("benchmark/baselines/milestone_23_67")
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
COMPARISON_DIR = RUN_DIR / "comparison"
LOG_PATH = RUN_DIR / "run.log"
STATUS_PATH = RUN_DIR / "status.txt"
GT_DIR = Path("benchmark/ground_truth/adjudicated_v1")


def main() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    evaluator = ParserEvaluator(
        ground_truth_dir=GT_DIR,
        output_dir=COMPARISON_DIR,
        diagnostic=True,
    )
    mapping_rows = evaluator._load_mapping()
    logger.info("Loaded %d mapping row(s) from %s/mapping.csv", len(mapping_rows), GT_DIR)

    STATUS_PATH.write_text(f"IN_PROGRESS 0/{len(mapping_rows)}\n")

    for i, row in enumerate(mapping_rows, start=1):
        resume_name = Path(row["Markdown"]).stem
        checkpoint_path = CHECKPOINT_DIR / f"{resume_name}.pkl"
        if checkpoint_path.exists():
            logger.info("[%d/%d] %s -- already checkpointed, skipping", i, len(mapping_rows), resume_name)
            continue

        markdown_path = evaluator.markdown_dir / row["Markdown"]
        ground_truth_path = evaluator.ground_truth_dir / row["JSON"]

        logger.info("[%d/%d] Evaluating %s ...", i, len(mapping_rows), resume_name)
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
            logger.info("[%d/%d] %s -- OK (%.1fs, accuracy %.1f%%)", i, len(mapping_rows), resume_name, elapsed, comparison.accuracy * 100)
        except Exception as exc:  # noqa: BLE001 -- one bad resume must not stop the run
            elapsed = time.perf_counter() - start
            reason = f"{type(exc).__name__}: {exc}"
            logger.error("[%d/%d] %s -- FAILED (%.1fs): %s", i, len(mapping_rows), resume_name, elapsed, reason)
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
        STATUS_PATH.write_text(f"IN_PROGRESS {i}/{len(mapping_rows)}\n")

    # All resumes now checkpointed -- write final merged reports.
    logger.info("All resumes checkpointed. Writing final merged reports.")
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

    result = evaluator._build_result(comparisons, failures, failure_details, total_elapsed, diagnostics)

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    evaluator.json_comparator.generate_reports(comparisons, output_dir=COMPARISON_DIR)
    evaluator._write_module_accuracy_csv(comparisons, COMPARISON_DIR / "module_accuracy.csv")
    evaluator._write_benchmark_report(result, COMPARISON_DIR / "benchmark_report.md")
    evaluator.json_comparator.generate_diagnostic_reports(diagnostics, output_dir=COMPARISON_DIR)

    print_evaluation_result(result, COMPARISON_DIR)

    STATUS_PATH.write_text(
        f"COMPLETE {len(mapping_rows)}/{len(mapping_rows)} "
        f"(succeeded={len(comparisons)} failed={len(failures)})\n"
    )
    logger.info(
        "DONE. succeeded=%d failed=%d total=%d",
        len(comparisons), len(failures), len(mapping_rows),
    )


if __name__ == "__main__":
    main()
