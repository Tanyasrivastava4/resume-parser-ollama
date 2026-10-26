"""
scripts/_milestone_v2_4_ab_run.py

Milestone V2.4 -- controlled A/B run of the EXISTING, unmodified
ParserPipeline (7 modules, WorkExperienceValidator, EducationValidator,
ParserMerger, SchemaValidator -- none touched this milestone) against
two different Stage-1 markdown sources, scored against
benchmark/ground_truth/adjudicated_v1/:

    --config native_only  ->  output/v2_4_benchmark/native_only/stage1/markdown/
                              (V2.3 prototype, --mode native-only, run as-is;
                               resume018 forced through native extraction too)
    --config routed       ->  output/v2_3_prototype/markdown/
                              (V2.3 prototype, --mode routed, already produced
                               last milestone -- reused, not regenerated, so
                               the one real OCR call for resume018 is not
                               repeated)

Same resilient checkpoint-per-resume pattern established in
scripts/_milestone_23_67_frozen_baseline_run.py: each resume's evaluation
result (or failure) is pickled to its own checkpoint file as soon as it
completes, so an interrupted run resumes exactly where it left off.

Uses services/isolated_parser_pipeline.py's IsolatedParserPipeline so
ParserPipeline's own hardcoded output/modules/ and output/json_final/
writes never fire -- this run's own JSON output lands only under
output/v2_4_benchmark/<config>/{modules,json_final}/, never colliding
with production output or the frozen Milestone 23.67 baseline.

Nothing here modifies parser/prompt/schema/validator/comparator source,
and the V2.3 prototype's own routing logic is not touched -- both
markdown sources were produced by calling that script exactly as it
already exists.

Usage:
    python scripts/_milestone_v2_4_ab_run.py --config native_only
    python scripts/_milestone_v2_4_ab_run.py --config routed
"""

import argparse
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
from services.isolated_parser_pipeline import IsolatedParserPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("milestone_v2_4_ab_run")

GT_DIR = Path("benchmark/ground_truth/adjudicated_v1")

CONFIGS = {
    "native_only": {
        "markdown_dir": Path("output/v2_4_benchmark/native_only/stage1/markdown"),
        "run_dir": Path("output/v2_4_benchmark/native_only"),
    },
    "routed": {
        "markdown_dir": Path("output/v2_3_prototype/markdown"),
        "run_dir": Path("output/v2_4_benchmark/routed"),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    run_dir = cfg["run_dir"]
    checkpoint_dir = run_dir / "checkpoints"
    comparison_dir = run_dir / "comparison"
    status_path = run_dir / "status.txt"
    pipeline_output_root = run_dir / "pipeline_output"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    isolated_pipeline = IsolatedParserPipeline(output_root=pipeline_output_root)
    evaluator = ParserEvaluator(
        parser_pipeline=isolated_pipeline,
        ground_truth_dir=GT_DIR,
        markdown_dir=cfg["markdown_dir"],
        output_dir=comparison_dir,
        diagnostic=True,
    )
    mapping_rows = evaluator._load_mapping()
    logger.info(
        "[%s] Loaded %d mapping row(s); markdown_dir=%s",
        args.config, len(mapping_rows), cfg["markdown_dir"],
    )

    status_path.write_text(f"IN_PROGRESS 0/{len(mapping_rows)}\n")

    for i, row in enumerate(mapping_rows, start=1):
        resume_name = Path(row["Markdown"]).stem
        checkpoint_path = checkpoint_dir / f"{resume_name}.pkl"
        if checkpoint_path.exists():
            logger.info("[%s][%d/%d] %s -- already checkpointed, skipping", args.config, i, len(mapping_rows), resume_name)
            continue

        markdown_path = evaluator.markdown_dir / row["Markdown"]
        ground_truth_path = evaluator.ground_truth_dir / row["JSON"]

        logger.info("[%s][%d/%d] Evaluating %s ...", args.config, i, len(mapping_rows), resume_name)
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
            logger.info(
                "[%s][%d/%d] %s -- OK (%.1fs, strict accuracy %.1f%%)",
                args.config, i, len(mapping_rows), resume_name, elapsed, comparison.accuracy * 100,
            )
        except Exception as exc:  # noqa: BLE001 -- one bad resume must not stop the run
            elapsed = time.perf_counter() - start
            reason = f"{type(exc).__name__}: {exc}"
            logger.error("[%s][%d/%d] %s -- FAILED (%.1fs): %s", args.config, i, len(mapping_rows), resume_name, elapsed, reason)
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
        status_path.write_text(f"IN_PROGRESS {i}/{len(mapping_rows)}\n")

    logger.info("[%s] All resumes checkpointed. Writing final merged reports.", args.config)
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

    result = evaluator._build_result(comparisons, failures, failure_details, total_elapsed, diagnostics)

    comparison_dir.mkdir(parents=True, exist_ok=True)
    evaluator.json_comparator.generate_reports(comparisons, output_dir=comparison_dir)
    evaluator._write_module_accuracy_csv(comparisons, comparison_dir / "module_accuracy.csv")
    evaluator._write_benchmark_report(result, comparison_dir / "benchmark_report.md")
    evaluator.json_comparator.generate_diagnostic_reports(diagnostics, output_dir=comparison_dir)

    print_evaluation_result(result, comparison_dir)

    status_path.write_text(
        f"COMPLETE {len(mapping_rows)}/{len(mapping_rows)} "
        f"(succeeded={len(comparisons)} failed={len(failures)})\n"
    )
    logger.info(
        "[%s] DONE. succeeded=%d failed=%d total=%d",
        args.config, len(comparisons), len(failures), len(mapping_rows),
    )


if __name__ == "__main__":
    main()
