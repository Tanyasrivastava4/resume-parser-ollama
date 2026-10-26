"""
scripts/_milestone_23_16_diagnostic_full_run.py

One-off script for Milestone 23.16's post-implementation full 23-resume
benchmark measurement. Mirrors scripts/run_parser_evaluation.py's
default (no --suite) full-dataset path exactly (this IS "the existing
23-resume benchmark" the milestone asks for), with diagnostic=True
added so ParserEvaluator additionally writes diagnostic_summary.csv,
diagnostic_mismatch_by_field.csv, and diagnostic_report.md to
benchmark/comparison/ alongside its normal reports -- the same
diagnostic-mode mechanism Milestone 23.15 used, so this run's
work_experience.technologies/entry/field accuracy figures are directly
comparable to Milestone 23.15's own diagnostic figures using the same
method.

Not a permanent addition to scripts/ -- a scratch script for this
milestone's one measurement run, kept only for the paper trail (deleted
or left in place per the milestone's own reporting convention).
"""

import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.parser_evaluator import ParserEvaluator  # noqa: E402
from scripts.run_parser_evaluation import print_evaluation_result  # noqa: E402
from validators.json_comparator import ENTRY_STATUS_FULL_MATCH  # noqa: E402

logging.basicConfig(level=logging.INFO)

evaluator = ParserEvaluator(diagnostic=True)
result = evaluator.evaluate()
print_evaluation_result(result, evaluator.output_dir)

# Same plain-text per-job diagnostic dump format Milestone 23.14/23.15
# used, restricted to work_experience, for a direct grep-tally
# comparison if needed.
out_path = Path("/tmp/milestone_23_16/diagnostic_full_after.txt")
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    for resume_name in sorted(result.diagnostics):
        diag = result.diagnostics[resume_name]
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
    for resume_name in sorted(result.failures):
        f.write(f"=== {resume_name} === EXCLUDED (failed in this benchmark run)\n")
print(f"\nDiagnostic detail written to: {out_path}")
