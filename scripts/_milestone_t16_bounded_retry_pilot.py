"""
scripts/_milestone_t16_bounded_retry_pilot.py

Milestone T16/Phase 2C -- bounded Entity-First retry pilot, per the
frozen protocol in docs/T16_PHASE2C_BOUNDED_RETRY_PRE_REGISTRATION.md.

Research-only. Imports T13's own prompts/functions DIRECTLY (not
re-typed), the same objects T14/T15 already imported. Does not modify
WorkExperienceLocator, does not modify production code, does not add
retry logic to production.

AT MOST ONE RETRY PER STAGE, per run, per the frozen CASE 1-4 policy:
- Stage A structural failure -> retry Stage A once (same prompt).
- Stage B structural/schema/fixation failure -> retry Stage B once
  (same prompt, same fixed Stage-A entity list -- Stage A never rerun).
- Both stages structurally successful but recall incomplete -> NO
  retry, classified CONTENT_FAILURE_AFTER_STRUCTURAL_SUCCESS.

Runs 10 fresh Entity-First runs for resume 021, then 5 fresh runs for
resume 013, in that exact frozen order. No baseline condition is run.

Writes:
  benchmark/baselines/t16_pilot_results.csv
  <scratchpad>/t16_checkpoints/<run_id>.pkl
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
    _STAGE_A_PROMPT,
    _STAGE_B_PROMPT_TEMPLATE,
    _extract_json,
    check_entity_fixation_violation,
)
from services.ollama_client import OllamaClient  # noqa: E402
from validators.schema_validator import SchemaValidator  # noqa: E402
from config import SCHEMAS_DIR  # noqa: E402

SCHEMA_PATH = SCHEMAS_DIR / "modules" / "work_experience_schema.json"
MARKDOWN_DIR = Path("output/markdown")
CKPT_DIR = Path(
    "/private/tmp/claude-501/-Users-ruchigupta-Documents-ResumeAIBenchmark/"
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/t16_checkpoints"
)
# NOTE (remediation CI gate fix, 2026-10-17): CKPT_DIR is this script's original,
# machine-specific scratch location, kept verbatim so the recorded milestone
# stays reproducible on the machine it ran on. It used to be created HERE, at
# import time, which made merely importing this module fail on any other
# machine (PermissionError on /private on the Linux CI runner). It is now
# created lazily, immediately before the first checkpoint path is used.
# Nothing else about the script, its results or its tests changes.

# Frozen population and run order -- not changed after seeing results.
RUN_PLAN = (
    [("021", "PRAGNESH S. SHAH_Sr. Associate-IT Ops_Intileo", i) for i in range(1, 11)]
    + [("013", "Kishan Yadav_Data Scientist_Intileo", i) for i in range(1, 6)]
)


def _call_stage_a(client, markdown_text):
    prompt = f"{_STAGE_A_PROMPT}\n\n{markdown_text}"
    t0 = time.perf_counter()
    response = client.generate_text(model=MODEL, prompt=prompt)
    elapsed = time.perf_counter() - t0
    try:
        parsed = _extract_json(response.response)
        entities = parsed.get("entities", [])
        if not isinstance(entities, list):
            entities = []
        return {"ok": True, "entities": entities, "elapsed_s": elapsed,
                "raw": response.response, "error": None}
    except Exception as e:
        return {"ok": False, "entities": None, "elapsed_s": elapsed,
                "raw": response.response, "error": str(e)}


def _call_stage_b(client, validator, markdown_text, entities):
    prompt = _STAGE_B_PROMPT_TEMPLATE.format(
        entity_list_json=json.dumps(entities), n_entities=len(entities),
    ) + f"\n\n{markdown_text}"
    t0 = time.perf_counter()
    response = client.generate_text(model=MODEL, prompt=prompt)
    elapsed = time.perf_counter() - t0
    try:
        parsed = _extract_json(response.response)
    except Exception as e:
        return {"ok": False, "json_object": None, "elapsed_s": elapsed,
                "raw": response.response, "error": str(e), "schema_status": None,
                "fixation_violation": None}

    try:
        validator.validate(parsed)
        schema_status = "valid"
    except Exception as e:
        schema_status = f"invalid: {e}"

    returned_jobs = parsed.get("work_experience", []) if isinstance(parsed, dict) else []
    violation = check_entity_fixation_violation(entities, returned_jobs)

    ok = (schema_status == "valid") and (not violation)
    return {"ok": ok, "json_object": parsed, "elapsed_s": elapsed,
            "raw": response.response, "error": None,
            "schema_status": schema_status, "fixation_violation": violation}


def run_one(rid: str, stem: str, run_idx: int, client: OllamaClient, validator: SchemaValidator) -> dict:
    run_id = f"{rid}_{run_idx}"
    markdown_text = (MARKDOWN_DIR / f"{stem}.md").read_text(encoding="utf-8")

    record = {
        "run_id": run_id, "resume_id": rid, "stem": stem, "run": run_idx,
        "stage_a_initial": None, "stage_a_retry": None,
        "stage_b_initial": None, "stage_b_retry": None,
        "final_json_object": None, "final_stage_a_entities": None,
        "total_calls": 0, "total_elapsed_s": 0.0,
        "primary_failure_class": None,
    }
    t_run_start = time.perf_counter()

    # --- Stage A (initial) ---
    sa1 = _call_stage_a(client, markdown_text)
    record["stage_a_initial"] = sa1
    record["total_calls"] += 1
    record["total_elapsed_s"] += sa1["elapsed_s"]

    stage_a_ok = sa1["ok"]
    entities = sa1["entities"]

    if not stage_a_ok:
        # CASE 1: Stage A structural failure -> retry Stage A exactly once.
        sa2 = _call_stage_a(client, markdown_text)
        record["stage_a_retry"] = sa2
        record["total_calls"] += 1
        record["total_elapsed_s"] += sa2["elapsed_s"]
        if sa2["ok"]:
            stage_a_ok = True
            entities = sa2["entities"]
        else:
            record["primary_failure_class"] = "STAGE_A_RETRY_FAILURE"
            record["total_elapsed_s"] = time.perf_counter() - t_run_start
            return record

    if not entities:
        # No entities identified at all (structurally valid, empty list) --
        # not a parse failure; recall against a non-empty GT will be 0,
        # classified downstream as CONTENT_FAILURE_AFTER_STRUCTURAL_SUCCESS.
        record["final_json_object"] = {"work_experience": []}
        record["final_stage_a_entities"] = entities
        record["total_elapsed_s"] = time.perf_counter() - t_run_start
        return record

    record["final_stage_a_entities"] = entities

    # --- Stage B (initial) ---
    sb1 = _call_stage_b(client, validator, markdown_text, entities)
    record["stage_b_initial"] = sb1
    record["total_calls"] += 1
    record["total_elapsed_s"] += sb1["elapsed_s"]

    if sb1["ok"]:
        record["final_json_object"] = sb1["json_object"]
        record["total_elapsed_s"] = time.perf_counter() - t_run_start
        return record

    # CASE 2/4: Stage B structural/schema/fixation failure -> retry Stage B once.
    sb2 = _call_stage_b(client, validator, markdown_text, entities)
    record["stage_b_retry"] = sb2
    record["total_calls"] += 1
    record["total_elapsed_s"] += sb2["elapsed_s"]

    if sb2["ok"]:
        record["final_json_object"] = sb2["json_object"]
    else:
        if sb1["json_object"] is None and sb2["json_object"] is None:
            record["primary_failure_class"] = "STAGE_B_RETRY_FAILURE"
        elif sb2.get("fixation_violation") or sb1.get("fixation_violation"):
            record["primary_failure_class"] = "ENTITY_FIXATION_FAILURE"
        else:
            record["primary_failure_class"] = "STAGE_B_RETRY_FAILURE"
        # Preserve the retry's own (failed) json_object for evidence, if any.
        record["final_json_object"] = sb2["json_object"] or sb1["json_object"]

    record["total_elapsed_s"] = time.perf_counter() - t_run_start
    return record


def main() -> None:
    client = OllamaClient()
    validator = SchemaValidator(SCHEMA_PATH)
    rows = []

    for rid, stem, run_idx in RUN_PLAN:
        print(f"\n=== run {rid}_{run_idx} ===", flush=True)
        record = run_one(rid, stem, run_idx, client, validator)
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        with (CKPT_DIR / f"{record['run_id']}.pkl").open("wb") as f:
            pickle.dump(record, f)

        n_jobs = (
            len(record["final_json_object"].get("work_experience", []))
            if record.get("final_json_object") else 0
        )
        print(f"  calls={record['total_calls']} elapsed={record['total_elapsed_s']:.1f}s "
              f"stage_a_retry={'yes' if record['stage_a_retry'] else 'no'} "
              f"stage_b_retry={'yes' if record['stage_b_retry'] else 'no'} "
              f"failure_class={record['primary_failure_class']} n_jobs={n_jobs}", flush=True)

        rows.append({
            "run_id": record["run_id"], "resume_id": rid, "run": run_idx,
            "total_calls": record["total_calls"], "total_elapsed_s": record["total_elapsed_s"],
            "stage_a_retried": bool(record["stage_a_retry"]),
            "stage_b_retried": bool(record["stage_b_retry"]),
            "primary_failure_class": record["primary_failure_class"],
            "n_jobs": n_jobs,
        })

    out = Path("benchmark/baselines/t16_pilot_results.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)", flush=True)
    print("\nT16 PILOT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
