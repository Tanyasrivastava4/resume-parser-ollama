"""
scripts/_milestone_t9_phase2b_span_constrained_pilot.py

Milestone T9/Phase 2B -- Phase E controlled single-call comparison.

Research-only. Does NOT modify or import into pipeline/parser_pipeline.py.
Uses the real, unmodified WorkExperienceExtractor (same class, same
prompt, same model, same schema, same code) for BOTH conditions --
the only difference is which markdown file is passed as input:

  BASELINE:  the real, full, already-committed output/markdown/<stem>.md
  CANDIDATE: a temp file containing ONLY the deterministic Work
             Experience SECTION span (via
             scripts/_milestone_t9_phase2b_span_extraction.py, which
             itself only calls WorkExperienceLocator's own existing,
             unmodified private boundary methods)

Only SAFE-SPAN resumes (per this milestone's own Phase D classification,
frozen before this script runs) get a candidate call. UNSAFE-SPAN
resumes are not passed to this script at all.

Writes: benchmark/baselines/t9_phase2b_pilot_results.csv
        <scratchpad>/t9_phase2b_checkpoints/<resume_id>__{baseline,candidate}.pkl
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
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/t9_phase2b_checkpoints"
)
# NOTE (remediation CI gate fix, 2026-10-17): CKPT_DIR is this script's original,
# machine-specific scratch location, kept verbatim so the recorded milestone
# stays reproducible on the machine it ran on. It used to be created HERE, at
# import time, which made merely importing this module fail on any other
# machine (PermissionError on /private on the Linux CI runner). It is now
# created lazily, immediately before the first checkpoint path is used.
# Nothing else about the script, its results or its tests changes.

MARKDOWN_DIR = Path("output/markdown")

# Frozen population -- SAFE-SPAN resumes only. Set by this milestone's
# own Phase D classification, run before this script.
SAFE_SPAN_POPULATION = {
    "001": "AMAN_CV_BCA",
    "013": "Kishan Yadav_Data Scientist_Intileo",
    "016": "Mukul Kanojia_Full Stack Developer_Intileo",
}

MAX_ATTEMPTS = 3  # mirrors pipeline/parser_pipeline.py's own retry policy


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
            print(f"  [{label}] attempt {attempt} failed: {e}")
    return {"status": "failed", "result": None, "attempts": MAX_ATTEMPTS, "elapsed_s": None, "error": str(last_exc)}


def main() -> None:
    extractor = WorkExperienceExtractor()
    rows = []

    for rid, stem in SAFE_SPAN_POPULATION.items():
        print(f"\n=== {rid} ({stem}) ===")
        md_path = MARKDOWN_DIR / f"{stem}.md"
        full_text = md_path.read_text(encoding="utf-8")

        # BASELINE: real, unmodified extract() on the real, full,
        # already-committed markdown file -- exactly the production call.
        baseline_outcome = _run_with_retry(
            f"{rid}-baseline", lambda: extractor.extract(md_path)
        )
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        with (CKPT_DIR / f"{rid}__baseline.pkl").open("wb") as f:
            pickle.dump({
                "resume_id": rid, "stem": stem, "condition": "baseline",
                "input_chars": len(full_text),
                "status": baseline_outcome["status"],
                "attempts": baseline_outcome["attempts"],
                "elapsed_s": baseline_outcome["elapsed_s"],
                "error": baseline_outcome["error"],
                "json_object": (
                    baseline_outcome["result"].json_object if baseline_outcome["result"] else None
                ),
            }, f)
        print(f"  baseline: status={baseline_outcome['status']} attempts={baseline_outcome['attempts']} "
              f"elapsed={baseline_outcome['elapsed_s']}")

        # CANDIDATE: same extractor, same prompt, same model, same
        # schema -- only the input markdown differs (constrained to the
        # deterministic Work Experience section span, written to a temp
        # file so the real, unmodified extract() can be called exactly
        # as-is).
        span_result = extract_section_span(full_text)
        assert span_result.found, f"{rid}: expected a SAFE, found span -- population frozen incorrectly"
        candidate_md_path = CKPT_DIR / f"{rid}__candidate_input.md"
        candidate_md_path.write_text(span_result.section_text, encoding="utf-8")

        candidate_outcome = _run_with_retry(
            f"{rid}-candidate", lambda: extractor.extract(candidate_md_path)
        )
        with (CKPT_DIR / f"{rid}__candidate.pkl").open("wb") as f:
            pickle.dump({
                "resume_id": rid, "stem": stem, "condition": "candidate",
                "input_chars": len(span_result.section_text),
                "span_start_line": span_result.start_line_0indexed,
                "span_end_line": span_result.end_line_0indexed,
                "span_path": span_result.path,
                "status": candidate_outcome["status"],
                "attempts": candidate_outcome["attempts"],
                "elapsed_s": candidate_outcome["elapsed_s"],
                "error": candidate_outcome["error"],
                "json_object": (
                    candidate_outcome["result"].json_object if candidate_outcome["result"] else None
                ),
            }, f)
        print(f"  candidate: status={candidate_outcome['status']} attempts={candidate_outcome['attempts']} "
              f"elapsed={candidate_outcome['elapsed_s']} input_chars={len(span_result.section_text)} "
              f"(full={len(full_text)}, reduction={1 - len(span_result.section_text)/len(full_text):.1%})")

        rows.append({
            "resume_id": rid, "stem": stem,
            "baseline_status": baseline_outcome["status"],
            "baseline_attempts": baseline_outcome["attempts"],
            "baseline_elapsed_s": baseline_outcome["elapsed_s"],
            "baseline_job_count": (
                len(baseline_outcome["result"].json_object.get("work_experience", []))
                if baseline_outcome["result"] else None
            ),
            "candidate_status": candidate_outcome["status"],
            "candidate_attempts": candidate_outcome["attempts"],
            "candidate_elapsed_s": candidate_outcome["elapsed_s"],
            "candidate_job_count": (
                len(candidate_outcome["result"].json_object.get("work_experience", []))
                if candidate_outcome["result"] else None
            ),
            "full_input_chars": len(full_text),
            "span_input_chars": len(span_result.section_text),
        })

    out = Path("benchmark/baselines/t9_phase2b_pilot_results.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
