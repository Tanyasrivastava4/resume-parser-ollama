"""
scripts/_milestone_23_22_reliability_round_run.py

Batched full-benchmark runner for Milestone 23.22's 5-round reliability
validation experiment — a persistent-storage variant of
scripts/_milestone_23_22_batch_run.py, changed in exactly one respect:
checkpoints, comparison reports, and logs are written to a persistent,
non-/tmp directory so a repeat of the reboot that destroyed the first
attempt's /tmp/milestone_23_22/ artifacts cannot destroy this attempt's
too. No model, prompt, schema, validator, batch size, or evaluation
logic is different from the prior script in any way.

Usage: pass the round number (1-5) as the first argument. Re-run the
exact same command to continue a round that was interrupted -- only
resumes without a checkpoint for that round are re-evaluated:

    python scripts/_milestone_23_22_reliability_round_run.py 1
    python scripts/_milestone_23_22_reliability_round_run.py 1   # continues round 1
    python scripts/_milestone_23_22_reliability_round_run.py 2   # starts round 2
    ...

Each round's checkpoints live in
benchmark/reliability/milestone_23_22/round<N>/checkpoints/, its final
comparison reports in benchmark/reliability/milestone_23_22/round<N>/comparison/
(a dedicated output_dir, never benchmark/comparison/ -- this experiment
does not touch that shared location at all), its full per-entry
diagnostic dump at
benchmark/reliability/milestone_23_22/round<N>/diagnostic_full.txt, and
a full INFO+ log (including every retry/failure line, per extractor,
per exception type) at
benchmark/reliability/milestone_23_22/round<N>/logs/round<N>.log.

round_status.json in the round's own root directory tracks start_time
(set once, on the round's first invocation) and end_time + status
(set only when that round's final batch completes and merged reports
are written) -- so a round's own start/complete state survives a crash
independent of anything in /tmp.

BATCH_SIZE is unchanged from the prior script (10) -- not touched here,
per the experiment's own rule against changing batch size without the
runner requiring it.
"""

import json
import logging
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.parser_evaluator import EvaluationFailure, ParserEvaluator  # noqa: E402
from scripts.run_parser_evaluation import print_evaluation_result  # noqa: E402
from validators.json_comparator import ENTRY_STATUS_FULL_MATCH  # noqa: E402

logger = logging.getLogger("milestone_23_22_reliability_round_run")
logger.setLevel(logging.INFO)

BATCH_SIZE = 10

_EXPERIMENT_ROOT = _PROJECT_ROOT / "benchmark" / "reliability" / "milestone_23_22"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_status(status_path: Path) -> dict:
    if status_path.is_file():
        return json.loads(status_path.read_text(encoding="utf-8"))
    return {}


def _write_status(status_path: Path, status: dict) -> None:
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"1", "2", "3", "4", "5"}:
        print("Usage: python scripts/_milestone_23_22_reliability_round_run.py <round_number 1-5>")
        sys.exit(1)
    run_id = sys.argv[1]

    run_dir = _EXPERIMENT_ROOT / f"round{run_id}"
    checkpoint_dir = run_dir / "checkpoints"
    comparison_dir = run_dir / "comparison"
    logs_dir = run_dir / "logs"
    diagnostic_out_path = run_dir / "diagnostic_full.txt"
    status_path = run_dir / "round_status.json"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # File handler on the root logger so every module's logger (pipeline,
    # evaluator, extractors) -- all of which propagate to root -- gets
    # captured, including the per-extractor retry/failure lines from
    # pipeline/parser_pipeline.py's _run_with_retry.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not any(
        isinstance(h, logging.FileHandler) and getattr(h, "_milestone_23_22_round", None) == run_id
        for h in root_logger.handlers
    ):
        file_handler = logging.FileHandler(logs_dir / f"round{run_id}.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        file_handler._milestone_23_22_round = run_id  # type: ignore[attr-defined]
        root_logger.addHandler(file_handler)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root_logger.addHandler(stream_handler)

    status = _read_status(status_path)
    if "start_time" not in status:
        status["start_time"] = _now_iso()
        status["round"] = int(run_id)
        status["status"] = "in_progress"
        status["total_resumes"] = None
        status["end_time"] = None
        _write_status(status_path, status)

    evaluator = ParserEvaluator(diagnostic=True, output_dir=comparison_dir)
    mapping_rows = evaluator._load_mapping()

    if status.get("total_resumes") is None:
        status["total_resumes"] = len(mapping_rows)
        _write_status(status_path, status)

    remaining_rows = [
        row
        for row in mapping_rows
        if not (checkpoint_dir / f"{Path(row['Markdown']).stem}.pkl").exists()
    ]
    already_done = len(mapping_rows) - len(remaining_rows)

    if not remaining_rows:
        logger.info(
            "Round %s: all %d resumes already checkpointed -- writing final merged reports.",
            run_id,
            len(mapping_rows),
        )
        _write_final_reports(evaluator, mapping_rows, checkpoint_dir, comparison_dir, diagnostic_out_path)
        status["status"] = "complete"
        status["end_time"] = _now_iso()
        _write_status(status_path, status)
        return

    batch_rows = remaining_rows[:BATCH_SIZE]
    logger.info(
        "Round %s batch: processing %d resume(s) (%d/%d already checkpointed): %s",
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
            logger.info("Round %s: checkpointed %s (ok, %.2fs)", run_id, resume_name, elapsed)
        except Exception as exc:  # noqa: BLE001 - one bad resume must not stop the batch
            elapsed = time.perf_counter() - start
            reason = str(exc)
            logger.error("Round %s: failed to evaluate %s: %s", run_id, resume_name, reason)
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
        logger.info("Round %s: that was the last batch -- writing final merged reports now.", run_id)
        _write_final_reports(evaluator, mapping_rows, checkpoint_dir, comparison_dir, diagnostic_out_path)
        status["status"] = "complete"
        status["end_time"] = _now_iso()
        _write_status(status_path, status)
    else:
        logger.info(
            "Round %s batch complete. %d resume(s) still remaining. Re-run this exact script (same round number) to continue.",
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
            array_field_names = set(diag.array_diagnostics.keys())
            f.write(f"=== {resume_name} :: scalar ===\n")
            for fr in diag.strict_result.field_results:
                top = fr.field_path.split(".")[0]
                if top in array_field_names:
                    continue
                f.write(f"  field={fr.field_path} status={fr.status} expected={fr.expected!r} actual={fr.actual!r}\n")
        for resume_name in sorted(failures):
            f.write(f"=== {resume_name} === EXCLUDED (failed in this run): {failures[resume_name]}\n")
    logger.info("Diagnostic detail written to: %s", diagnostic_out_path)

    # Round-level completed/failed summary, alongside the standard reports.
    summary_path = diagnostic_out_path.parent / "round_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "total_resumes": len(mapping_rows),
                "succeeded": sorted(comparisons.keys()),
                "failed": {name: failures[name] for name in sorted(failures)},
                "overall_accuracy": result.overall_accuracy,
                "total_elapsed_seconds": total_elapsed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
