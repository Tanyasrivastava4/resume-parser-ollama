"""
scripts/_milestone_23_15_diagnostic_full_run.py

One-off script for Milestone 23.15's post-implementation full 23-resume
benchmark measurement. Mirrors scripts/run_parser_evaluation.py's
default (no --suite) full-dataset path exactly, with diagnostic=True
added so ParserEvaluator additionally writes diagnostic_summary.csv,
diagnostic_mismatch_by_field.csv, and diagnostic_report.md to
benchmark/comparison/ alongside its normal reports — the same
diagnostic-mode mechanism Milestone 23.14 used to build
/tmp/milestone_23_14/diagnostic_full.txt, so this run's output is
directly comparable to that "before" baseline using the same tallying
method.

Not a permanent addition to scripts/ — a scratch script for this
milestone's one measurement run, kept only for the paper trail.
"""

import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.parser_evaluator import ParserEvaluator  # noqa: E402
from scripts.run_parser_evaluation import print_evaluation_result  # noqa: E402
from services.work_experience_validator import WorkExperienceValidator  # noqa: E402
from validators.json_comparator import ENTRY_STATUS_FULL_MATCH  # noqa: E402

logging.basicConfig(level=logging.INFO)

# Non-invasive instrumentation (no shipped file is modified): wrap
# WorkExperienceValidator.validate so every ValidationReport it produces
# during this run is captured, keyed by call order, purely so this
# script can report "number of values nulled by WEV-016/017/018"
# per-resume after the fact. ParserEvaluator/ParserPipeline themselves
# already discard each resume's ValidationReport once merged_json is
# built; this only observes what already happens, it changes nothing
# about validate()'s real behavior or return value.
_captured_reports = []
_original_validate = WorkExperienceValidator.validate


def _instrumented_validate(self, markdown, jobs):
    validated, report = _original_validate(self, markdown, jobs)
    _captured_reports.append(report)
    return validated, report


WorkExperienceValidator.validate = _instrumented_validate

evaluator = ParserEvaluator(diagnostic=True)
result = evaluator.evaluate()
WorkExperienceValidator.validate = _original_validate
print_evaluation_result(result, evaluator.output_dir)

# Tally WEV-016/017/018 activity across every captured report (one per
# resume the WorkExperienceExtractor+validator successfully ran for —
# resumes that failed before reaching the validator, e.g. Arjun
# Sharma's extractor-level JSON error, have no report here).
provenance_rule_ids = {"WEV-016", "WEV-017", "WEV-018"}
nulled_by_rule = {"WEV-016": 0, "WEV-017": 0, "WEV-018": 0}
unmatched_warnings = 0
for report in _captured_reports:
    for issue in report.issues:
        if issue.rule_id in provenance_rule_ids:
            nulled_by_rule[issue.rule_id] += 1
        elif issue.rule_id == "WEV-016/017/018":
            unmatched_warnings += 1

print("\nWEV-016/017/018 activity across this run")
print("------------------------------------------")
print(f"Resumes with a captured ValidationReport: {len(_captured_reports)}")
print(f"employment_type nulled (WEV-016): {nulled_by_rule['WEV-016']}")
print(f"location nulled (WEV-017): {nulled_by_rule['WEV-017']}")
print(f"duration nulled (WEV-018): {nulled_by_rule['WEV-018']}")
print(f"Total nulled: {sum(nulled_by_rule.values())}")
print(f"'No confident span match' warnings (WEV-016/017/018): {unmatched_warnings}")

# Also dump a diagnostic_full.txt in the same plain-text per-job format
# Milestone 23.14 used (only work_experience, only mismatched
# sub-fields within PARTIAL_MATCH/MISSING/EXTRA entries), for a direct
# grep-tally comparison against /tmp/milestone_23_14/diagnostic_full.txt.
out_path = Path("/tmp/milestone_23_15/diagnostic_full_after.txt")
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
