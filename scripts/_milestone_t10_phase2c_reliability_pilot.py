"""
scripts/_milestone_t10_phase2c_reliability_pilot.py

Milestone T10/Phase 2C -- Phase C controlled repeated calls, per the
frozen pre-registration in docs/T10_PHASE2C_PRE_REGISTRATION.md.

Research-only. Reuses T9's own, unmodified span-extraction helper
(scripts/_milestone_t9_phase2b_span_extraction.py, itself calling only
WorkExperienceLocator's existing, unmodified private boundary methods)
and the real, unmodified WorkExperienceExtractor for both conditions.
Does not modify or import into pipeline/parser_pipeline.py.

Runs N=5 baseline + N=5 candidate real qwen3:1.7b calls per resume,
for exactly 001, 013, 016 (006 excluded from the candidate condition
per the frozen pre-registration -- unsafe span, T9).

Writes:
  benchmark/baselines/t10_phase2c_pilot_results.csv (one row per call)
  <scratchpad>/t10_phase2c_checkpoints/<resume_id>__{baseline,candidate}_<run>.pkl
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
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/t10_phase2c_checkpoints"
)
# NOTE (remediation CI gate fix, 2026-10-17): CKPT_DIR is this script's original,
# machine-specific scratch location, kept verbatim so the recorded milestone
# stays reproducible on the machine it ran on. It used to be created HERE, at
# import time, which made merely importing this module fail on any other
# machine (PermissionError on /private on the Linux CI runner). It is now
# created lazily, immediately before the first checkpoint path is used.
# Nothing else about the script, its results or its tests changes.

MARKDOWN_DIR = Path("output/markdown")

# Frozen population -- SAFE-SPAN resumes only (006 excluded, per
# pre-registration). Order and membership fixed before any call.
POPULATION = {
    "001": "AMAN_CV_BCA",
    "013": "Kishan Yadav_Data Scientist_Intileo",
    "016": "Mukul Kanojia_Full Stack Developer_Intileo",
}

N_REPEATS = 5  # frozen, per pre-registration -- not changed after seeing results
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
            print(f"  [{label}] attempt {attempt} failed: {e}", flush=True)
    return {"status": "failed", "result": None, "attempts": MAX_ATTEMPTS, "elapsed_s": None, "error": str(last_exc)}


def main() -> None:
    extractor = WorkExperienceExtractor()
    rows = []

    for rid, stem in POPULATION.items():
        md_path = MARKDOWN_DIR / f"{stem}.md"
        full_text = md_path.read_text(encoding="utf-8")

        span_result = extract_section_span(full_text)
        assert span_result.found, f"{rid}: expected a SAFE, found span"
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        candidate_md_path = CKPT_DIR / f"{rid}__candidate_input.md"
        candidate_md_path.write_text(span_result.section_text, encoding="utf-8")

        for run in range(1, N_REPEATS + 1):
            print(f"\n=== {rid} ({stem}) run {run}/{N_REPEATS} ===", flush=True)

            # BASELINE
            outcome = _run_with_retry(f"{rid}-baseline-{run}", lambda: extractor.extract(md_path))
            ckpt_path = CKPT_DIR / f"{rid}__baseline_{run}.pkl"
            with ckpt_path.open("wb") as f:
                pickle.dump({
                    "resume_id": rid, "stem": stem, "condition": "baseline", "run": run,
                    "input_chars": len(full_text), "status": outcome["status"],
                    "attempts": outcome["attempts"], "elapsed_s": outcome["elapsed_s"],
                    "error": outcome["error"],
                    "json_object": (outcome["result"].json_object if outcome["result"] else None),
                }, f)
            jc = len(outcome["result"].json_object.get("work_experience", [])) if outcome["result"] else None
            print(f"  baseline run {run}: status={outcome['status']} attempts={outcome['attempts']} "
                  f"elapsed={outcome['elapsed_s']} job_count={jc}", flush=True)
            rows.append({
                "resume_id": rid, "stem": stem, "condition": "baseline", "run": run,
                "status": outcome["status"], "attempts": outcome["attempts"],
                "elapsed_s": outcome["elapsed_s"], "job_count": jc,
                "input_chars": len(full_text),
            })

            # CANDIDATE
            outcome_c = _run_with_retry(
                f"{rid}-candidate-{run}", lambda: extractor.extract(candidate_md_path)
            )
            ckpt_path_c = CKPT_DIR / f"{rid}__candidate_{run}.pkl"
            with ckpt_path_c.open("wb") as f:
                pickle.dump({
                    "resume_id": rid, "stem": stem, "condition": "candidate", "run": run,
                    "input_chars": len(span_result.section_text), "status": outcome_c["status"],
                    "attempts": outcome_c["attempts"], "elapsed_s": outcome_c["elapsed_s"],
                    "error": outcome_c["error"],
                    "json_object": (outcome_c["result"].json_object if outcome_c["result"] else None),
                }, f)
            jc_c = len(outcome_c["result"].json_object.get("work_experience", [])) if outcome_c["result"] else None
            print(f"  candidate run {run}: status={outcome_c['status']} attempts={outcome_c['attempts']} "
                  f"elapsed={outcome_c['elapsed_s']} job_count={jc_c}", flush=True)
            rows.append({
                "resume_id": rid, "stem": stem, "condition": "candidate", "run": run,
                "status": outcome_c["status"], "attempts": outcome_c["attempts"],
                "elapsed_s": outcome_c["elapsed_s"], "job_count": jc_c,
                "input_chars": len(span_result.section_text),
            })

    out = Path("benchmark/baselines/t10_phase2c_pilot_results.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
