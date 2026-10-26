"""
scripts/_milestone_t12_boundary_marker_pilot.py

Milestone T12/Phase 2C -- controlled repeated calls, per the frozen
protocol in docs/T12_PHASE2C_BOUNDARY_MARKER_PRE_REGISTRATION.md.

Research-only. Reuses T9's own, unmodified span-extraction helper
(scripts/_milestone_t9_phase2b_span_extraction.py, itself calling only
WorkExperienceLocator's existing, unmodified private boundary methods)
and the real, unmodified WorkExperienceExtractor for all three
conditions. Does not modify or import into pipeline/parser_pipeline.py.
Does not modify WorkExperienceLocator.

Runs N=5 BASELINE + N=5 SPAN + N=5 MARKER real qwen3:1.7b calls per
resume, for exactly 001, 013, 016 (006 excluded entirely -- locator-
safety control only, per the frozen protocol). All calls are FRESH --
none of T10/T11's own checkpoints are reused as T12 measurements.

MARKER = SPAN's exact text + "\n```" (one line, three backticks,
appended with no blank line) -- the single, fixed, resume-agnostic
marker defined in the pre-registration.

Writes:
  benchmark/baselines/t12_pilot_results.csv (one row per call)
  <scratchpad>/t12_checkpoints/<resume_id>__<condition>_<run>.pkl
"""
from __future__ import annotations

import csv
import pickle
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_t9_phase2b_span_extraction import extract_section_span  # noqa: E402
from services.work_experience_extractor import WorkExperienceExtractor  # noqa: E402
from services.base_module_extractor import InvalidJSONError  # noqa: E402
from validators.schema_validator import SchemaValidationError  # noqa: E402

CKPT_DIR = Path(
    "/private/tmp/claude-501/-Users-ruchigupta-Documents-ResumeAIBenchmark/"
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/t12_checkpoints"
)
# NOTE (remediation CI gate fix, 2026-10-17): CKPT_DIR is this script's original,
# machine-specific scratch location, kept verbatim so the recorded milestone
# stays reproducible on the machine it ran on. It used to be created HERE, at
# import time, which made merely importing this module fail on any other
# machine (PermissionError on /private on the Linux CI runner). It is now
# created lazily, immediately before the first checkpoint path is used.
# Nothing else about the script, its results or its tests changes.

MARKDOWN_DIR = Path("output/markdown")

# Frozen population -- 006 excluded entirely, per the frozen protocol.
POPULATION = {
    "001": "AMAN_CV_BCA",
    "013": "Kishan Yadav_Data Scientist_Intileo",
    "016": "Mukul Kanojia_Full Stack Developer_Intileo",
}

N_REPEATS = 5
MAX_ATTEMPTS = 3
MARKER_LINE = "```"


def _run_with_retry(label, func):
    last_exc = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            start = time.perf_counter()
            result = func()
            elapsed = time.perf_counter() - start
            return {"status": "ok", "result": result, "attempts": attempt, "elapsed_s": elapsed, "error": None}
        except (InvalidJSONError, SchemaValidationError) as e:
            last_exc = e
            print(f"  [{label}] attempt {attempt} failed: {e}", flush=True)
    return {"status": "failed", "result": None, "attempts": MAX_ATTEMPTS, "elapsed_s": None, "error": str(last_exc)}


def main() -> None:
    extractor = WorkExperienceExtractor()
    rows = []

    for rid, stem in POPULATION.items():
        md_path = MARKDOWN_DIR / f"{stem}.md"
        full_text = md_path.read_text(encoding="utf-8")

        span_result = extract_section_span(full_text)
        assert span_result.found, f"{rid}: expected a found span"
        span_text = span_result.section_text
        marker_text = span_text + "\n" + MARKER_LINE

        CKPT_DIR.mkdir(parents=True, exist_ok=True)

        span_path = CKPT_DIR / f"{rid}__span_input.md"
        span_path.write_text(span_text, encoding="utf-8")
        marker_path = CKPT_DIR / f"{rid}__marker_input.md"
        marker_path.write_text(marker_text, encoding="utf-8")

        conditions = [
            ("baseline", md_path, len(full_text)),
            ("span", span_path, len(span_text)),
            ("marker", marker_path, len(marker_text)),
        ]

        for condition, path, input_chars in conditions:
            for run in range(1, N_REPEATS + 1):
                print(f"\n=== {rid} {condition} run {run}/{N_REPEATS} ===", flush=True)
                outcome = _run_with_retry(
                    f"{rid}-{condition}-{run}", lambda p=path: extractor.extract(p)
                )
                ckpt_path = CKPT_DIR / f"{rid}__{condition}_{run}.pkl"
                with ckpt_path.open("wb") as f:
                    pickle.dump({
                        "resume_id": rid, "stem": stem, "condition": condition, "run": run,
                        "input_chars": input_chars, "status": outcome["status"],
                        "attempts": outcome["attempts"], "elapsed_s": outcome["elapsed_s"],
                        "error": outcome["error"],
                        "json_object": (outcome["result"].json_object if outcome["result"] else None),
                    }, f)
                jobs = (
                    outcome["result"].json_object.get("work_experience", [])
                    if outcome["result"] else []
                )
                print(f"  {condition} run {run}: status={outcome['status']} "
                      f"attempts={outcome['attempts']} elapsed={outcome['elapsed_s']} "
                      f"n_jobs={len(jobs)}", flush=True)
                rows.append({
                    "resume_id": rid, "stem": stem, "condition": condition, "run": run,
                    "status": outcome["status"], "attempts": outcome["attempts"],
                    "elapsed_s": outcome["elapsed_s"], "input_chars": input_chars,
                    "entity_count": len(jobs),
                })

    out = Path("benchmark/baselines/t12_pilot_results.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)", flush=True)
    print("\nT12 PILOT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
