"""
scripts/_milestone_t15_021_large_replication.py

Milestone T15/Phase 2C -- controlled repeated calls, per the frozen
protocol in docs/T15_PHASE2C_ENTITY_FIRST_021_LARGE_REPLICATION_PRE_REGISTRATION.md.

Research-only. Imports T13's own prompts/functions DIRECTLY (not
re-typed), the same objects T14 already imported, to guarantee
byte-identical replication across T13->T14->T15:
_STAGE_A_PROMPT, _STAGE_B_PROMPT_TEMPLATE, _extract_json,
check_entity_fixation_violation, MODEL, MAX_ATTEMPTS_BASELINE.

NO RETRY POLICY is added to either Entity-First stage -- this is the
single most important rule this milestone follows. A Stage A or Stage
B parse/schema failure is recorded once, with zero retries.

Runs N=10 BASELINE + N=10 ENTITY-FIRST for resume 021 (primary), and
N=3 + N=3 for resumes 001 and 013 (safety controls). 006 is excluded
entirely.

Writes:
  benchmark/baselines/t15_pilot_results.csv
  <scratchpad>/t15_checkpoints/<resume_id>__<condition>_<run>.pkl
"""
from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_t13_entity_first_pilot import (  # noqa: E402
    MODEL,
    MAX_ATTEMPTS_BASELINE,
    _STAGE_A_PROMPT,
    _STAGE_B_PROMPT_TEMPLATE,
    _extract_json,
    check_entity_fixation_violation,
)
from services.ollama_client import OllamaClient  # noqa: E402
from services.work_experience_extractor import WorkExperienceExtractor  # noqa: E402
from services.base_module_extractor import InvalidJSONError  # noqa: E402
from validators.schema_validator import SchemaValidator, SchemaValidationError  # noqa: E402
from config import SCHEMAS_DIR  # noqa: E402

SCHEMA_PATH = SCHEMAS_DIR / "modules" / "work_experience_schema.json"
MARKDOWN_DIR = Path("output/markdown")
CKPT_DIR = Path(
    "/private/tmp/claude-501/-Users-ruchigupta-Documents-ResumeAIBenchmark/"
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/t15_checkpoints"
)
# NOTE (remediation CI gate fix, 2026-10-17): CKPT_DIR is this script's original,
# machine-specific scratch location, kept verbatim so the recorded milestone
# stays reproducible on the machine it ran on. It used to be created HERE, at
# import time, which made merely importing this module fail on any other
# machine (PermissionError on /private on the Linux CI runner). It is now
# created lazily, immediately before the first checkpoint path is used.
# Nothing else about the script, its results or its tests changes.

# Frozen population and repetition counts -- not changed after seeing results.
POPULATION = {
    "021": ("PRAGNESH S. SHAH_Sr. Associate-IT Ops_Intileo", 10),
    "001": ("AMAN_CV_BCA", 3),
    "013": ("Kishan Yadav_Data Scientist_Intileo", 3),
}


def _run_baseline_with_retry(markdown_path: Path):
    """Unchanged from T13/T14: production's own real MAX_ATTEMPTS=3
    schema-retry policy, via the real, unmodified
    WorkExperienceExtractor. This is BASELINE's pre-existing production
    behavior, not a new retry policy introduced by T15."""
    extractor = WorkExperienceExtractor()
    last_exc = None
    for attempt in range(1, MAX_ATTEMPTS_BASELINE + 1):
        try:
            t0 = time.perf_counter()
            result = extractor.extract(markdown_path, model_name=MODEL)
            elapsed = time.perf_counter() - t0
            return {
                "status": "ok", "attempts": attempt, "elapsed_s": elapsed,
                "n_calls": attempt, "json_object": result.json_object, "error": None,
            }
        except (InvalidJSONError, SchemaValidationError) as e:
            last_exc = e
    return {"status": "failed", "attempts": MAX_ATTEMPTS_BASELINE, "elapsed_s": None,
            "n_calls": MAX_ATTEMPTS_BASELINE, "json_object": None, "error": str(last_exc)}


def _run_entity_first(markdown_path: Path, client: OllamaClient, validator: SchemaValidator):
    """ZERO retries on Stage A or Stage B -- deliberate, per T15's own
    primary rule. A parse/schema failure at either stage is recorded
    exactly once, never retried."""
    markdown_text = markdown_path.read_text(encoding="utf-8")
    t0 = time.perf_counter()

    stage_a_prompt = f"{_STAGE_A_PROMPT}\n\n{markdown_text}"
    stage_a_t0 = time.perf_counter()
    stage_a_response = client.generate_text(model=MODEL, prompt=stage_a_prompt)
    stage_a_elapsed = time.perf_counter() - stage_a_t0
    try:
        stage_a_json = _extract_json(stage_a_response.response)
        entities = stage_a_json.get("entities", [])
        if not isinstance(entities, list):
            entities = []
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {"status": "stage_a_parse_failed", "n_calls": 1, "elapsed_s": elapsed,
                "stage_a_elapsed_s": stage_a_elapsed, "stage_b_elapsed_s": None,
                "error": str(e), "raw_stage_a": stage_a_response.response,
                "stage_a_entities": None, "json_object": None,
                "stage_b_entity_fixation_violation": None, "schema_status": None}

    if not entities:
        elapsed = time.perf_counter() - t0
        return {"status": "ok", "n_calls": 1, "elapsed_s": elapsed,
                "stage_a_elapsed_s": stage_a_elapsed, "stage_b_elapsed_s": None,
                "json_object": {"work_experience": []}, "stage_a_entities": entities,
                "stage_b_entity_fixation_violation": False, "schema_status": None}

    stage_b_prompt = _STAGE_B_PROMPT_TEMPLATE.format(
        entity_list_json=json.dumps(entities), n_entities=len(entities),
    ) + f"\n\n{markdown_text}"
    stage_b_t0 = time.perf_counter()
    stage_b_response = client.generate_text(model=MODEL, prompt=stage_b_prompt)
    stage_b_elapsed = time.perf_counter() - stage_b_t0
    elapsed = time.perf_counter() - t0

    try:
        stage_b_json = _extract_json(stage_b_response.response)
    except Exception as e:
        return {"status": "stage_b_parse_failed", "n_calls": 2, "elapsed_s": elapsed,
                "stage_a_elapsed_s": stage_a_elapsed, "stage_b_elapsed_s": stage_b_elapsed,
                "error": str(e), "stage_a_entities": entities,
                "raw_stage_b": stage_b_response.response, "json_object": None,
                "stage_b_entity_fixation_violation": None, "schema_status": None}

    try:
        validator.validate(stage_b_json)
        schema_status = "valid"
    except Exception as e:
        schema_status = f"invalid: {e}"

    returned_jobs = stage_b_json.get("work_experience", []) if isinstance(stage_b_json, dict) else []
    violation = check_entity_fixation_violation(entities, returned_jobs)

    return {
        "status": ("ok" if schema_status == "valid" or schema_status is None else "schema_failed"),
        "n_calls": 2, "elapsed_s": elapsed,
        "stage_a_elapsed_s": stage_a_elapsed, "stage_b_elapsed_s": stage_b_elapsed,
        "json_object": stage_b_json, "stage_a_entities": entities,
        "schema_status": schema_status, "stage_b_entity_fixation_violation": violation,
    }


def main() -> None:
    client = OllamaClient()
    validator = SchemaValidator(SCHEMA_PATH)
    rows = []

    for rid, (stem, n_repeats) in POPULATION.items():
        md_path = MARKDOWN_DIR / f"{stem}.md"

        for run in range(1, n_repeats + 1):
            print(f"\n=== {rid} baseline run {run}/{n_repeats} ===", flush=True)
            b = _run_baseline_with_retry(md_path)
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            with (CKPT_DIR / f"{rid}__baseline_{run}.pkl").open("wb") as f:
                pickle.dump({"resume_id": rid, "stem": stem, "condition": "baseline", "run": run, **b}, f)
            n_jobs = len(b["json_object"].get("work_experience", [])) if b["json_object"] else None
            print(f"  status={b['status']} n_calls={b['n_calls']} elapsed={b['elapsed_s']} n_jobs={n_jobs}", flush=True)
            rows.append({"resume_id": rid, "condition": "baseline", "run": run, "status": b["status"],
                         "n_calls": b["n_calls"], "elapsed_s": b["elapsed_s"], "n_jobs": n_jobs})

        for run in range(1, n_repeats + 1):
            print(f"\n=== {rid} entity_first run {run}/{n_repeats} ===", flush=True)
            ef = _run_entity_first(md_path, client, validator)
            with (CKPT_DIR / f"{rid}__entity_first_{run}.pkl").open("wb") as f:
                pickle.dump({"resume_id": rid, "stem": stem, "condition": "entity_first", "run": run, **ef}, f)
            n_jobs = len(ef["json_object"].get("work_experience", [])) if ef.get("json_object") else None
            n_stage_a = len(ef["stage_a_entities"]) if ef.get("stage_a_entities") else None
            print(f"  status={ef['status']} n_calls={ef['n_calls']} elapsed={ef['elapsed_s']} "
                  f"n_stage_a_entities={n_stage_a} n_jobs={n_jobs} "
                  f"violation={ef.get('stage_b_entity_fixation_violation')}", flush=True)
            rows.append({"resume_id": rid, "condition": "entity_first", "run": run, "status": ef["status"],
                         "n_calls": ef["n_calls"], "elapsed_s": ef["elapsed_s"], "n_jobs": n_jobs})

    out = Path("benchmark/baselines/t15_pilot_results.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)", flush=True)
    print("\nT15 PILOT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
