"""
scripts/_milestone_23_40_delivery_run.py

Delivery runner for Milestone 23.40 (Proven Fix Delivery Run) -- same
resilient, checkpointed pattern established in Milestones 23.15/23.18/
23.20/23.21/23.22 (see project memory's operational note), adapted for
this milestone's own explicit requirement to persist checkpoints outside
/tmp (this machine has previously shut down mid-run) and to run only
ONCE (not 5 rounds) with the express purpose of *delivering* already-
proven-safe fixes into output/json_final and output/modules, not
measuring reliability or accuracy in its own right.

No production code is touched by this script. It only calls
ParserEvaluator._evaluate_one() (unmodified, existing code), which in
turn calls the unmodified ParserPipeline.run() -- the exact same call
path every full-benchmark run in this project already uses. This
script's only job is to make that call resiliently, once per resume,
with a persistent checkpoint so a machine interruption never loses
already-completed work and never silently re-does or discards it.

Usage: re-run this exact command to continue an interrupted run -- only
resumes without a checkpoint are re-evaluated:

    python scripts/_milestone_23_40_delivery_run.py

Checkpoints: benchmark/delivery/milestone_23_40/checkpoints/<resume>.pkl
Comparison reports (diagnostic mode): benchmark/delivery/milestone_23_40/comparison/
Per-resume log: benchmark/delivery/milestone_23_40/logs/run.log

output/json_final/<resume>.json and output/modules/<resume>/*.json are
overwritten by ParserPipeline.run() itself, as they are on every prior
benchmark run in this project -- this script does not write to those
paths directly. A full pre-run backup of both was already taken to
benchmark/delivery/milestone_23_40/before/ prior to this script's first
invocation, per this milestone's own explicit before/after preservation
requirement.
"""

import logging
import pickle
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.parser_evaluator import ParserEvaluator  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(_PROJECT_ROOT / "benchmark/delivery/milestone_23_40/logs/run.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("milestone_23_40_delivery_run")

RUN_DIR = _PROJECT_ROOT / "benchmark/delivery/milestone_23_40"
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
COMPARISON_DIR = RUN_DIR / "comparison"
STATUS_PATH = RUN_DIR / "run_status.json"


def main() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    evaluator = ParserEvaluator(diagnostic=True, output_dir=COMPARISON_DIR)
    mapping_rows = evaluator._load_mapping()

    remaining_rows = [
        row
        for row in mapping_rows
        if not (CHECKPOINT_DIR / f"{Path(row['Markdown']).stem}.pkl").exists()
    ]
    already_done = len(mapping_rows) - len(remaining_rows)
    logger.info(
        "Delivery run: %d/%d resumes already checkpointed. %d remaining this invocation.",
        already_done,
        len(mapping_rows),
        len(remaining_rows),
    )

    for row in remaining_rows:
        resume_name = Path(row["Markdown"]).stem
        markdown_path = evaluator.markdown_dir / row["Markdown"]
        ground_truth_path = evaluator.ground_truth_dir / row["JSON"]
        checkpoint_path = CHECKPOINT_DIR / f"{resume_name}.pkl"

        start_wall = time.strftime("%Y-%m-%dT%H:%M:%S")
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
                        "start_wall": start_wall,
                        "end_wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
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
                        "exception_type": type(exc).__name__,
                        "ground_truth_file": str(ground_truth_path),
                        "markdown_file": str(markdown_path),
                        "elapsed": elapsed,
                        "start_wall": start_wall,
                        "end_wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                    f,
                )

    all_checkpointed = all(
        (CHECKPOINT_DIR / f"{Path(row['Markdown']).stem}.pkl").exists() for row in mapping_rows
    )
    if all_checkpointed:
        logger.info("All %d resumes checkpointed. Delivery run complete.", len(mapping_rows))
        ok_count = 0
        failed_count = 0
        for row in mapping_rows:
            resume_name = Path(row["Markdown"]).stem
            with (CHECKPOINT_DIR / f"{resume_name}.pkl").open("rb") as f:
                data = pickle.load(f)
            if data["status"] == "ok":
                ok_count += 1
            else:
                failed_count += 1
        logger.info("Final tally: %d ok, %d failed, of %d total.", ok_count, failed_count, len(mapping_rows))
        STATUS_PATH.write_text(
            f'{{"total": {len(mapping_rows)}, "ok": {ok_count}, "failed": {failed_count}, "complete": true}}\n'
        )
    else:
        logger.info("Run incomplete. Re-run this exact script to continue.")
        STATUS_PATH.write_text(
            f'{{"total": {len(mapping_rows)}, "checkpointed": {already_done + len(remaining_rows)}, "complete": false}}\n'
        )


if __name__ == "__main__":
    main()
