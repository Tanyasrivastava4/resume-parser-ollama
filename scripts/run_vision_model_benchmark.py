"""
scripts/run_vision_model_benchmark.py

Vision Model Benchmark Runner (Milestone 16.2).

Implements the framework designed in docs/VISION_MODEL_BENCHMARK_PLAN.md
(Milestone 16.1): benchmark several OCR vision models against the same
resume dataset, reusing BatchProcessor (extraction) and MarkdownBenchmark
(scoring) completely unmodified. This script contains no extraction logic
and no markdown-comparison logic of its own — it only orchestrates the
two existing services and formats their results.

For every model in --models:
    1. BatchProcessor.process_batch(model_name, PROMPT_ID) generates
       markdown for every resume under input/resumes/, via the existing,
       unmodified DocumentExtractor -> output/markdown/<resume>.md.
    2. Every *successfully generated* file (per BatchProcessingResult;
       a resume a model failed on is never copied, so a stale file from
       a previous model's run can never masquerade as this model's
       output) is copied to
       output/benchmark_models/<model_name>/markdown/<resume>.md.
    3. MarkdownBenchmark(generated_dir=.../markdown/, output_dir=.../benchmark/).run()
       scores that directory against the existing, unmodified reference
       markdown under benchmark/ground_truth/markdown/ — the same class,
       called with different constructor arguments; nothing about it is
       reimplemented here.
    4. The per-model BatchProcessingResult and MarkdownBenchmarkResult
       are combined into one summary row.

output/markdown/ is the shared "current" tree every other benchmark and
test in this project reads (and that Milestone 15C.3 was careful to
leave untouched). Each model's run overwrites it in turn, so this script
backs it up before the first model runs and restores it after the last
one (success or failure) — running this script has no side effect on it.

Usage:
    python scripts/run_vision_model_benchmark.py --models qwen2.5vl:3b,qwen2.5vl:7b
"""

import argparse
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

# Run directly (`python scripts/run_vision_model_benchmark.py`), the
# script's own directory ends up on sys.path instead of the project
# root — add the project root explicitly, matching every other script
# under scripts/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmark.markdown_benchmark import (  # noqa: E402
    MarkdownBenchmark,
    MarkdownBenchmarkResult,
)
from config import MARKDOWN_DIR, MARKDOWN_GROUND_TRUTH_DIR, OUTPUT_DIR  # noqa: E402
from services.batch_processor import BatchProcessingResult, BatchProcessor  # noqa: E402
from services.ollama_client import OllamaClient  # noqa: E402

logger = logging.getLogger(__name__)

# Prompt every model is benchmarked with — the project's one standard OCR
# prompt (unchanged), so model choice is the only variable across runs.
PROMPT_ID = "resume/extract_markdown"

BENCHMARK_MODELS_DIR = OUTPUT_DIR / "benchmark_models"
SUMMARY_REPORT_PATH = BENCHMARK_MODELS_DIR / "summary.md"


@dataclass
class ModelBenchmarkOutcome:
    """Everything collected for one model's benchmark run.

    Attributes:
        model_name: The Ollama model name benchmarked, e.g. "qwen2.5vl:3b".
        succeeded: False if the model was skipped entirely (e.g. not
            installed) — a model that ran but scored badly is still
            `succeeded=True`; only a run that never produced results is
            `succeeded=False`.
        skip_reason: Why the model was skipped, or None if it ran.
        batch_result: The BatchProcessingResult from extracting every
            resume with this model, or None if skipped.
        benchmark_result: The MarkdownBenchmarkResult from scoring this
            model's output, or None if skipped.
    """

    model_name: str
    succeeded: bool
    skip_reason: Optional[str] = None
    batch_result: Optional[BatchProcessingResult] = None
    benchmark_result: Optional[MarkdownBenchmarkResult] = None


def parse_models_arg(raw: str) -> List[str]:
    """Split a comma-separated --models value into a list of model names.

    Args:
        raw: The raw --models string, e.g. "qwen2.5vl:3b,qwen2.5vl:7b".

    Returns:
        A list of trimmed, non-empty model names, in the order given.

    Raises:
        ValueError: If `raw` contains no usable model name.
    """
    models = [name.strip() for name in raw.split(",") if name.strip()]
    if not models:
        raise ValueError(f"--models contained no usable model name: {raw!r}")
    return models


def archive_generated_markdown(
    batch_result: BatchProcessingResult, destination_dir: Path
) -> int:
    """Copy every successfully generated markdown file from this batch
    run into `destination_dir`.

    Only files this specific run actually produced are copied (via
    `batch_result.results`, not a directory listing) — a resume this
    model failed on has `markdown_path=None` and is skipped, so a stale
    file left over from a previous model's run in the shared
    output/markdown/ tree can never be mistaken for this model's output.

    Args:
        batch_result: The BatchProcessingResult to archive.
        destination_dir: Directory to copy each markdown file into.
            Created if it does not already exist.

    Returns:
        The number of files copied.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for record in batch_result.results:
        if not record.success or not record.markdown_path:
            continue
        source_path = _PROJECT_ROOT / record.markdown_path
        shutil.copy2(source_path, destination_dir / source_path.name)
        copied += 1
    return copied


def backup_markdown_dir(markdown_dir: Path) -> Optional[Path]:
    """Copy `markdown_dir` to a sibling backup directory before any model
    run overwrites it.

    Returns:
        The backup directory's path, or None if `markdown_dir` did not
        exist (nothing to back up).
    """
    if not markdown_dir.exists():
        return None
    backup_dir = markdown_dir.parent / f"{markdown_dir.name}_backup_vision_model_benchmark"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(markdown_dir, backup_dir)
    return backup_dir


def restore_markdown_dir(markdown_dir: Path, backup_dir: Optional[Path]) -> None:
    """Restore `markdown_dir` from `backup_dir`, saved by
    backup_markdown_dir. A no-op if `backup_dir` is None."""
    if backup_dir is None:
        return
    if markdown_dir.exists():
        shutil.rmtree(markdown_dir)
    shutil.move(str(backup_dir), str(markdown_dir))


def benchmark_one_model(
    model_name: str,
    batch_processor: BatchProcessor,
    ollama_client: OllamaClient,
    reference_dir: Path = MARKDOWN_GROUND_TRUTH_DIR,
    benchmark_models_dir: Path = BENCHMARK_MODELS_DIR,
    prompt_id: str = PROMPT_ID,
) -> ModelBenchmarkOutcome:
    """Extract every resume with `model_name`, archive the results, and
    score them with MarkdownBenchmark.

    Args:
        model_name: Ollama vision model to benchmark, e.g. "qwen2.5vl:3b".
        batch_processor: BatchProcessor to extract every resume with.
            DocumentExtractor always saves generated markdown under the
            fixed config.MARKDOWN_DIR regardless of model — this
            function doesn't take a markdown_dir argument because there
            is nowhere else for it to come from; each successfully
            generated file's real location comes from
            batch_result.results[i].markdown_path instead (see
            archive_generated_markdown).
        ollama_client: OllamaClient used only to check `model_name` is
            installed before attempting a full batch run.
        reference_dir: Reference markdown directory passed through to
            MarkdownBenchmark unchanged.
        benchmark_models_dir: Root directory this model's archived
            markdown and benchmark report are written under.
        prompt_id: Prompt identifier passed through to BatchProcessor.

    Returns:
        A ModelBenchmarkOutcome. `succeeded=False` (with `skip_reason`
        set) if `model_name` is not installed in Ollama — no batch run
        or benchmark is attempted in that case.
    """
    logger.info("=== Benchmarking model: %s ===", model_name)

    if not ollama_client.model_exists(model_name):
        skip_reason = f"Model '{model_name}' is not installed in Ollama"
        logger.warning("%s — skipping.", skip_reason)
        return ModelBenchmarkOutcome(
            model_name=model_name, succeeded=False, skip_reason=skip_reason
        )

    batch_result = batch_processor.process_batch(model_name=model_name, prompt_id=prompt_id)

    model_dir = benchmark_models_dir / model_name
    model_markdown_dir = model_dir / "markdown"
    model_benchmark_dir = model_dir / "benchmark"

    copied = archive_generated_markdown(batch_result, model_markdown_dir)
    logger.info("Archived %d markdown file(s) for %s -> %s", copied, model_name, model_markdown_dir)

    benchmark = MarkdownBenchmark(
        reference_dir=reference_dir,
        generated_dir=model_markdown_dir,
        output_dir=model_benchmark_dir,
    )
    benchmark_result = benchmark.run()

    return ModelBenchmarkOutcome(
        model_name=model_name,
        succeeded=True,
        batch_result=batch_result,
        benchmark_result=benchmark_result,
    )


def composite_accuracy(benchmark_result: MarkdownBenchmarkResult) -> float:
    """Mean of the four overall accuracy metrics, used only to rank
    models in the summary report. Not part of MarkdownBenchmark's own
    API (it computes an analogous per-resume figure internally, but has
    nothing that aggregates to the per-model figure this report needs),
    so this is new, minimal reporting logic — not a duplicate of any
    existing scoring logic."""
    return (
        benchmark_result.overall_character_accuracy
        + benchmark_result.overall_word_accuracy
        + benchmark_result.overall_line_accuracy
        + benchmark_result.overall_heading_accuracy
    ) / 4


def build_summary_report(outcomes: List[ModelBenchmarkOutcome]) -> str:
    """Build the output/benchmark_models/summary.md content.

    Models that ran are ranked by composite_accuracy, highest first.
    Models that were skipped (e.g. not installed) are listed separately
    and excluded from ranking.

    Args:
        outcomes: One ModelBenchmarkOutcome per model requested.

    Returns:
        The summary report's full Markdown text.
    """
    ran = [outcome for outcome in outcomes if outcome.succeeded]
    skipped = [outcome for outcome in outcomes if not outcome.succeeded]
    ranked = sorted(ran, key=lambda outcome: composite_accuracy(outcome.benchmark_result), reverse=True)

    lines = ["# Vision Model Benchmark Summary", ""]

    lines += [
        "| Model | Processing Time | Character | Word | Line | Heading | "
        "Missing Lines | Additional Lines | Rank |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, outcome in enumerate(ranked, start=1):
        benchmark_result = outcome.benchmark_result
        processing_time = outcome.batch_result.total_processing_time
        lines.append(
            f"| {outcome.model_name} | {processing_time:.2f}s | "
            f"{benchmark_result.overall_character_accuracy:.2%} | "
            f"{benchmark_result.overall_word_accuracy:.2%} | "
            f"{benchmark_result.overall_line_accuracy:.2%} | "
            f"{benchmark_result.overall_heading_accuracy:.2%} | "
            f"{benchmark_result.total_missing_lines} | "
            f"{benchmark_result.total_additional_lines} | "
            f"{rank} |"
        )

    if skipped:
        lines += ["", "## Skipped Models", "", "| Model | Reason |", "|---|---|"]
        for outcome in skipped:
            lines.append(f"| {outcome.model_name} | {outcome.skip_reason} |")

    return "\n".join(lines) + "\n"


def run_vision_model_benchmark_cli(
    models: List[str],
    batch_processor: Optional[BatchProcessor] = None,
    ollama_client: Optional[OllamaClient] = None,
    markdown_dir: Path = MARKDOWN_DIR,
    reference_dir: Path = MARKDOWN_GROUND_TRUTH_DIR,
    benchmark_models_dir: Path = BENCHMARK_MODELS_DIR,
    summary_report_path: Path = SUMMARY_REPORT_PATH,
) -> List[ModelBenchmarkOutcome]:
    """Benchmark every model in `models` and write the summary report.

    output/markdown/ (`markdown_dir`) is backed up before the first
    model runs and restored after the last one, so this call has no net
    effect on it regardless of how many models are benchmarked or
    whether any of them fail.

    Args:
        models: Model names to benchmark, in the order given.
        batch_processor: BatchProcessor to use for every model. Defaults
            to a new BatchProcessor() built from its own defaults.
        ollama_client: OllamaClient used to check model availability.
            Defaults to a new OllamaClient() built from settings.py.
        markdown_dir: See benchmark_one_model.
        reference_dir: See benchmark_one_model.
        benchmark_models_dir: See benchmark_one_model.
        summary_report_path: Where to write the summary Markdown report.

    Returns:
        One ModelBenchmarkOutcome per model in `models`, in order.
    """
    batch_processor = batch_processor or BatchProcessor()
    ollama_client = ollama_client or OllamaClient()

    backup_dir = backup_markdown_dir(markdown_dir)
    try:
        outcomes = [
            benchmark_one_model(
                model_name,
                batch_processor=batch_processor,
                ollama_client=ollama_client,
                reference_dir=reference_dir,
                benchmark_models_dir=benchmark_models_dir,
            )
            for model_name in models
        ]
    finally:
        restore_markdown_dir(markdown_dir, backup_dir)

    summary_report_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = build_summary_report(outcomes)
    summary_report_path.write_text(report_text, encoding="utf-8")
    logger.info("Wrote summary report to %s", summary_report_path)
    print(report_text)

    return outcomes


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the vision model benchmark CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark multiple OCR vision models against the same resume "
            "dataset, using the existing MarkdownBenchmark for scoring."
        )
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated list of installed Ollama vision model names, "
        "e.g. qwen2.5vl:3b,qwen2.5vl:7b",
    )
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    models = parse_models_arg(args.models)
    run_vision_model_benchmark_cli(models)


if __name__ == "__main__":
    main()
