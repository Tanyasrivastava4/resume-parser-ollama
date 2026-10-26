"""
scripts/_milestone_t11_condition_c_diagnostic.py

Milestone T11 -- controlled diagnostic experiment, per the frozen
pre-registration in docs/T11_PRE_REGISTRATION.md.

Research-only. Reuses the real, unmodified WorkExperienceExtractor.
Does not modify or import into pipeline/parser_pipeline.py. Does not
modify WorkExperienceLocator or any production file.

Runs N=5 real qwen3:1.7b calls against Condition C's input (T10's own
candidate span + exactly one appended line, "## Projects", nothing
else) for resume 001 only. Condition A is NOT re-run -- T10's own 5
already-collected candidate runs are reused as-is.
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

T11_CKPT_DIR = Path(
    "/private/tmp/claude-501/-Users-ruchigupta-Documents-ResumeAIBenchmark/"
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/t11_checkpoints"
)
CONDITION_C_PATH = T11_CKPT_DIR / "001__conditionC_input.md"
N_REPEATS = 5  # frozen, per pre-registration
MAX_ATTEMPTS = 3


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
    input_chars = len(CONDITION_C_PATH.read_text(encoding="utf-8"))

    for run in range(1, N_REPEATS + 1):
        print(f"\n=== Condition C run {run}/{N_REPEATS} ===", flush=True)
        outcome = _run_with_retry(f"conditionC-{run}", lambda: extractor.extract(CONDITION_C_PATH))
        ckpt_path = T11_CKPT_DIR / f"001__conditionC_{run}.pkl"
        with ckpt_path.open("wb") as f:
            pickle.dump({
                "resume_id": "001", "condition": "C", "run": run,
                "input_chars": input_chars, "status": outcome["status"],
                "attempts": outcome["attempts"], "elapsed_s": outcome["elapsed_s"],
                "error": outcome["error"],
                "json_object": (outcome["result"].json_object if outcome["result"] else None),
            }, f)
        jobs = (outcome["result"].json_object.get("work_experience", []) if outcome["result"] else [])
        print(f"  run {run}: status={outcome['status']} attempts={outcome['attempts']} "
              f"elapsed={outcome['elapsed_s']} jobs={jobs}", flush=True)

    print("\nCondition C complete", flush=True)


if __name__ == "__main__":
    main()
