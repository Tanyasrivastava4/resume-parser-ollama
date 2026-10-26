"""
scripts/_milestone_t10_phase2c_resume_remaining.py

Milestone T10/Phase 2C -- completes the exact pre-registered 30-call
plan after the original run (scripts/_milestone_t10_phase2c_reliability_pilot.py)
was interrupted mid-execution by an infrastructure event (background
process killed, not a model/schema failure) after 28/30 calls. This
script runs ONLY the specific remaining pre-registered calls -- it
does not add, remove, or re-select any resume/condition/run beyond
what the frozen pre-registration already specified:

  016 baseline run 5/5
  016 candidate run 4/5
  016 candidate run 5/5

Reuses the exact same, already-written candidate input file
(016__candidate_input.md) and the same extractor/model/settings as the
original run -- no change to experimental conditions.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from services.work_experience_extractor import WorkExperienceExtractor  # noqa: E402
from services.base_module_extractor import InvalidJSONError  # noqa: E402
from validators.schema_validator import SchemaValidationError  # noqa: E402

CKPT_DIR = Path(
    "/private/tmp/claude-501/-Users-ruchigupta-Documents-ResumeAIBenchmark/"
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/t10_phase2c_checkpoints"
)
MARKDOWN_DIR = Path("output/markdown")
MAX_ATTEMPTS = 3

RID = "016"
STEM = "Mukul Kanojia_Full Stack Developer_Intileo"


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
    md_path = MARKDOWN_DIR / f"{STEM}.md"
    full_text = md_path.read_text(encoding="utf-8")
    candidate_md_path = CKPT_DIR / f"{RID}__candidate_input.md"
    span_text = candidate_md_path.read_text(encoding="utf-8")

    remaining = [
        ("baseline", 5, md_path, len(full_text)),
        ("candidate", 4, candidate_md_path, len(span_text)),
        ("candidate", 5, candidate_md_path, len(span_text)),
    ]

    for condition, run, path, input_chars in remaining:
        print(f"\n=== {RID} {condition} run {run}/5 (resumed) ===", flush=True)
        outcome = _run_with_retry(f"{RID}-{condition}-{run}", lambda p=path: extractor.extract(p))
        ckpt_path = CKPT_DIR / f"{RID}__{condition}_{run}.pkl"
        with ckpt_path.open("wb") as f:
            pickle.dump({
                "resume_id": RID, "stem": STEM, "condition": condition, "run": run,
                "input_chars": input_chars, "status": outcome["status"],
                "attempts": outcome["attempts"], "elapsed_s": outcome["elapsed_s"],
                "error": outcome["error"],
                "json_object": (outcome["result"].json_object if outcome["result"] else None),
            }, f)
        jc = len(outcome["result"].json_object.get("work_experience", [])) if outcome["result"] else None
        print(f"  {condition} run {run}: status={outcome['status']} attempts={outcome['attempts']} "
              f"elapsed={outcome['elapsed_s']} job_count={jc}", flush=True)

    print("\nresume-remaining complete", flush=True)


if __name__ == "__main__":
    main()
