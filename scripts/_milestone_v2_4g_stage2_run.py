"""
scripts/_milestone_v2_4g_stage2_run.py

Milestone V2.4G -- controlled, single-resume (resume017) Stage-2
execution across three Stage-1 representations (A: OCR baseline,
B: native extraction, C: experimental column-aware linearization),
2 runs each, using the real, UNMODIFIED ParserPipeline (same prompts,
schemas, validators, model configuration, retry behavior as
production) via services/isolated_parser_pipeline.py's
IsolatedParserPipeline, so output never touches output/json_final/,
output/modules/, or any prior milestone's benchmark output.

Not a benchmark script -- processes exactly 6 markdown files (one
resume, 3 representations, 2 runs each), not the 22-resume corpus.
"""
import logging
import pickle
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from services.isolated_parser_pipeline import IsolatedParserPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("milestone_v2_4g_stage2_run")

INPUT_DIR = Path("output/v2_4g_experimental/stage1_runs")
OUTPUT_ROOT = Path("output/v2_4g_experimental/stage2_output")
CHECKPOINT_DIR = Path("output/v2_4g_experimental/checkpoints")

CASES = [
    "resume017_A_run1", "resume017_A_run2",
    "resume017_B_run1", "resume017_B_run2",
    "resume017_C_run1", "resume017_C_run2",
]


def main() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = IsolatedParserPipeline(output_root=OUTPUT_ROOT)

    for case in CASES:
        checkpoint_path = CHECKPOINT_DIR / f"{case}.pkl"
        if checkpoint_path.exists():
            logger.info("%s -- already checkpointed, skipping", case)
            continue
        markdown_path = INPUT_DIR / f"{case}.md"
        logger.info("Running Stage-2 for %s ...", case)
        try:
            result = pipeline.run(markdown_path)
            with checkpoint_path.open("wb") as f:
                pickle.dump({"status": "ok", "merged_json": result.merged_json,
                             "processing_time": result.processing_time}, f)
            logger.info("%s -- OK (%.1fs)", case, result.processing_time)
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
            with checkpoint_path.open("wb") as f:
                pickle.dump({"status": "failed", "reason": reason}, f)
            logger.error("%s -- FAILED: %s", case, reason)

    logger.info("All 6 cases attempted.")


if __name__ == "__main__":
    main()
