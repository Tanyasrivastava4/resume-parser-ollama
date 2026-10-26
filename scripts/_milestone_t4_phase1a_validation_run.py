"""
scripts/_milestone_t4_phase1a_validation_run.py

Milestone T4 / Phase 1A -- bounded real-Ollama validation run for the
T3-built CertificationExtractor and ProjectExtractor. RESEARCH-ONLY,
isolated -- direct calls to services/certification_extractor.py and
services/project_extractor.py exactly as T3 built them (no modification),
NEVER through pipeline/parser_pipeline.py (which does not reference
either extractor at all, confirmed by direct grep before this run).

Frozen population: benchmark/baselines/t4_phase1a_frozen_validation_population.csv
(8 resumes, selected and rationale-documented BEFORE this script ran).
Model: qwen3:1.7b (current production baseline, settings.DEFAULT_TEXT_MODEL,
unchanged). 1 run per (resume, module) slot -- 16 planned calls.

Retry policy: ONE retry per slot on InvalidJSONError/SchemaValidationError
(the same production-contract failures the base extractor itself can
raise), matching this project's own established V3.4/V5.6/V6.1 policy.
The original failed attempt is preserved and disclosed, never silently
discarded. A slot failing both attempts is recorded as a genuine
failure, not retried further -- no open-ended retries, no prompt
iteration during this run.

Checkpointed per (resume, module) slot, resumable.
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

from services.base_module_extractor import InvalidJSONError  # noqa: E402
from services.certification_extractor import CertificationExtractor  # noqa: E402
from services.ollama_client import OllamaClient  # noqa: E402
from services.project_extractor import ProjectExtractor  # noqa: E402
from validators.schema_validator import SchemaValidationError  # noqa: E402

MODEL = "qwen3:1.7b"
MARKDOWN_DIR = Path("output/markdown")
CKPT_DIR = Path(
    "/private/tmp/claude-501/-Users-ruchigupta-Documents-ResumeAIBenchmark/"
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/t4_phase1a_checkpoints"
)
# NOTE (remediation CI gate fix, 2026-10-17): CKPT_DIR is this script's original,
# machine-specific scratch location, kept verbatim so the recorded milestone
# stays reproducible on the machine it ran on. It used to be created HERE, at
# import time, which made merely importing this module fail on any other
# machine (PermissionError on /private on the Linux CI runner). It is now
# created lazily, immediately before the first checkpoint path is used.
# Nothing else about the script, its results or its tests changes.


def _load_population() -> list[dict]:
    with open("benchmark/baselines/t4_phase1a_frozen_validation_population.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_slot(resume_id: str, stem: str, module: str) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{resume_id}__{module}.pkl"
    if ckpt_path.exists():
        print(f"SKIP (already checkpointed): {ckpt_path.name}")
        return

    markdown_path = MARKDOWN_DIR / f"{stem}.md"
    client = OllamaClient()
    extractor_cls = CertificationExtractor if module == "certifications" else ProjectExtractor
    extractor = extractor_cls(ollama_client=client)

    print(f"RUNNING: {resume_id} {module} ...")
    n_attempts = 0
    attempts_log = []
    result_record = None

    for attempt in (1, 2):  # exactly one predeclared retry, no more
        n_attempts += 1
        t0 = time.perf_counter()
        try:
            result = extractor.extract(markdown_path, model_name=MODEL)
            elapsed = time.perf_counter() - t0
            result_record = {
                "status": "ok", "json_object": result.json_object,
                "elapsed_seconds": elapsed, "n_attempts": n_attempts,
            }
            attempts_log.append({"attempt": attempt, "status": "ok", "elapsed_seconds": elapsed})
            break
        except (InvalidJSONError, SchemaValidationError) as e:
            elapsed = time.perf_counter() - t0
            attempts_log.append({
                "attempt": attempt, "status": "failed",
                "error_type": type(e).__name__, "error": str(e), "elapsed_seconds": elapsed,
            })
            if attempt == 2:
                result_record = {
                    "status": "failed", "error_type": type(e).__name__, "error": str(e),
                    "elapsed_seconds": elapsed, "n_attempts": n_attempts,
                }

    with ckpt_path.open("wb") as f:
        pickle.dump({
            "resume_id": resume_id, "stem": stem, "module": module,
            "result": result_record, "attempts_log": attempts_log,
        }, f)
    print(f"DONE: {resume_id} {module} -> status={result_record['status']} "
          f"n_attempts={result_record['n_attempts']} elapsed={result_record['elapsed_seconds']:.1f}s")


def main() -> None:
    population = _load_population()
    for row in population:
        for module in ("certifications", "projects"):
            run_slot(row["resume_id"], row["stem"], module)


if __name__ == "__main__":
    main()
