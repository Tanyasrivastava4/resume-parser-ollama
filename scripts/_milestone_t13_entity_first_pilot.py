"""
scripts/_milestone_t13_entity_first_pilot.py

Milestone T13/Phase 2C -- controlled repeated calls, per the frozen
protocol in docs/T13_PHASE2C_ENTITY_FIRST_PRE_REGISTRATION.md.

Research-only, isolated exactly like T2's own original prototype
(scripts/_milestone_t2_phase0_entity_first_prototype.py, re-read in
full before writing this file). NOT imported by services/ or
pipeline/. Does not modify WorkExperienceLocator, any production
prompt, or work_experience_schema.json. BASELINE reuses the real,
unmodified WorkExperienceExtractor directly.

Runs N=3 BASELINE + N=3 ENTITY-FIRST (2 calls each: Stage A + batched
Stage B) real qwen3:1.7b calls per resume, for 001/006/013/016/021.
Never one call per entity.

Writes:
  benchmark/baselines/t13_pilot_results.csv
  <scratchpad>/t13_checkpoints/<resume_id>__<condition>_<run>.pkl
"""
from __future__ import annotations

import csv
import json
import pickle
import re
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from services.ollama_client import OllamaClient  # noqa: E402
from services.work_experience_extractor import WorkExperienceExtractor  # noqa: E402
from services.base_module_extractor import InvalidJSONError  # noqa: E402
from validators.schema_validator import SchemaValidator, SchemaValidationError  # noqa: E402
from validators.misrouting_detector import _fuzzy_contains  # noqa: E402
from config import SCHEMAS_DIR  # noqa: E402

MODEL = "qwen3:1.7b"
SCHEMA_PATH = SCHEMAS_DIR / "modules" / "work_experience_schema.json"
MARKDOWN_DIR = Path("output/markdown")
CKPT_DIR = Path(
    "/private/tmp/claude-501/-Users-ruchigupta-Documents-ResumeAIBenchmark/"
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/t13_checkpoints"
)
# NOTE (remediation CI gate fix, 2026-10-17): CKPT_DIR is this script's
# original, machine-specific scratch location, kept verbatim so the recorded
# milestone remains reproducible on the machine it ran on. It used to be
# created HERE, at import time, which made merely importing this module
# (which the test suite does) fail on any other machine (PermissionError on
# /private on the Linux CI runner). It is now created lazily, immediately
# before the first checkpoint is written. Nothing else about the script, its
# results or its tests changes.

N_REPEATS = 3
MAX_ATTEMPTS_BASELINE = 3  # matches production's own retry policy, via WorkExperienceExtractor itself

POPULATION = {
    "001": "AMAN_CV_BCA",
    "006": "AshishKumarSingh_RBM_Pune_Stonelam",
    "013": "Kishan Yadav_Data Scientist_Intileo",
    "016": "Mukul Kanojia_Full Stack Developer_Intileo",
    "021": "PRAGNESH S. SHAH_Sr. Associate-IT Ops_Intileo",
}

# ---------------------------------------------------------------------------
# Stage A / Stage B prompts -- finalized here before the first call, per the
# frozen pre-registration. Not edited after any call.
# ---------------------------------------------------------------------------

_STAGE_A_PROMPT = """You are an enterprise Resume Parser performing ONLY Work Experience entity identification.

You are given Markdown extracted from a resume.

Your ONLY task: identify each distinct, genuine employment entity (full-time roles, part-time roles, internships, freelance engagements, consulting stints, self-employed periods) that appears in this resume's Work Experience section.

Do NOT extract responsibilities, technologies, or any other detail. Do NOT include Education, Skills, Projects, Certifications, Awards, Achievements, Publications, or Conferences content -- a new section heading such as "Projects", "Education", "Skills", or "Certifications" ends the work experience content above it; nothing under those headings is an employment entity.

For each genuine employment entity, in the EXACT order it appears in the resume, report:
- ordinal (1-based position in this list)
- company (exactly as written in the resume, or null if not explicitly identifiable)
- designation (exactly as written in the resume, or null if not explicitly identifiable)
- start_date (exactly as written, or null)
- end_date (exactly as written, or null if ongoing/not stated)

Output ONLY valid JSON in this exact shape, nothing else:
{"entities": [{"ordinal": 1, "company": "...", "designation": "...", "start_date": "...", "end_date": "..."}]}

If there is no genuine work experience at all, output {"entities": []}.
"""

_STAGE_B_PROMPT_TEMPLATE = """You are an enterprise Resume Parser.

You are given Markdown extracted from a resume, AND a pre-identified list of the EXACT employment entities (in order) that exist in this resume's Work Experience section:

{entity_list_json}

Your task: for EACH of these entities, in this exact order and count, extract its full details from the resume.

Do NOT add, remove, merge, or split any entity. You MUST produce EXACTLY {n_entities} entries in the output array, one per entity listed above, in the same order.

Do NOT extract any of the following, even partially: Personal information, contact details, skills, languages, professional summary, Education, Certifications, Projects, Awards, Achievements, Publications, Conferences.

STRICT RULES:
1. Output ONLY valid JSON. Never explain anything. Never add markdown formatting. Nothing before or after the JSON.
2. Never invent, infer, or guess any value. If a value is not explicitly present in the resume, return `null` for it.
3. Each entry must include all ten fields: company, designation, employment_type, location, start_date, end_date, currently_working, duration, responsibilities, technologies -- use null for any field not explicitly present. currently_working must be true or false (never null). responsibilities and technologies must be arrays (use [] if none).
4. Preserve chronology and exact wording from the resume for responsibilities and technologies.

Output shape: {{"work_experience": [ {{...}}, {{...}} ]}}
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def check_entity_fixation_violation(stage_a_entities: list, returned_jobs: list) -> bool:
    """True if Stage B's returned work_experience array does not have
    the same length as Stage A's fixed entity list, or if any
    positional company identity does not fuzzy-match Stage A's own
    value at that position -- i.e. Stage B added, removed, or
    reordered entities relative to the boundaries Stage A already
    fixed. A pure function, independently testable without any LLM
    call."""
    if len(returned_jobs) != len(stage_a_entities):
        return True
    for stage_a_entity, returned_job in zip(stage_a_entities, returned_jobs):
        if not isinstance(returned_job, dict):
            return True  # structurally malformed -- cannot be the intact, fixed entity
        sa_company = str(stage_a_entity.get("company") or "")
        rb_company = str(returned_job.get("company") or "")
        if sa_company and rb_company and not _fuzzy_contains(sa_company, rb_company):
            return True
    return False


def _run_baseline_with_retry(markdown_path: Path):
    """Reuses the real, unmodified WorkExperienceExtractor -- exactly
    production's own contract, including its own MAX_ATTEMPTS=3 retry
    policy on InvalidJSONError/SchemaValidationError."""
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
    """Exactly two real calls -- Stage A (segmentation), then one
    batched Stage B call for ALL identified entities together. Zero
    retries, per the frozen protocol (matching T2's own explicit
    design). Verifies, in code, that Stage B did not add/remove/
    reorder entities relative to Stage A's own fixed list."""
    markdown_text = markdown_path.read_text(encoding="utf-8")
    t0 = time.perf_counter()

    # Stage A
    stage_a_prompt = f"{_STAGE_A_PROMPT}\n\n{markdown_text}"
    stage_a_response = client.generate_text(model=MODEL, prompt=stage_a_prompt)
    try:
        stage_a_json = _extract_json(stage_a_response.response)
        entities = stage_a_json.get("entities", [])
        if not isinstance(entities, list):
            entities = []
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {"status": "stage_a_parse_failed", "n_calls": 1, "elapsed_s": elapsed,
                "error": str(e), "raw_stage_a": stage_a_response.response,
                "stage_a_entities": None, "json_object": None,
                "stage_b_entity_fixation_violation": None}

    if not entities:
        elapsed = time.perf_counter() - t0
        return {"status": "ok", "n_calls": 1, "elapsed_s": elapsed,
                "json_object": {"work_experience": []}, "stage_a_entities": entities,
                "stage_b_entity_fixation_violation": False}

    # Stage B -- ONE batched call for all entities together
    stage_b_prompt = _STAGE_B_PROMPT_TEMPLATE.format(
        entity_list_json=json.dumps(entities), n_entities=len(entities),
    ) + f"\n\n{markdown_text}"
    stage_b_response = client.generate_text(model=MODEL, prompt=stage_b_prompt)
    elapsed = time.perf_counter() - t0

    try:
        stage_b_json = _extract_json(stage_b_response.response)
    except Exception as e:
        return {"status": "stage_b_parse_failed", "n_calls": 2, "elapsed_s": elapsed,
                "error": str(e), "stage_a_entities": entities,
                "raw_stage_b": stage_b_response.response, "json_object": None,
                "stage_b_entity_fixation_violation": None}

    try:
        validator.validate(stage_b_json)
        schema_status = "valid"
    except Exception as e:
        schema_status = f"invalid: {e}"

    # Code-level entity-fixation check (not just prompt instruction).
    returned_jobs = stage_b_json.get("work_experience", []) if isinstance(stage_b_json, dict) else []
    violation = check_entity_fixation_violation(entities, returned_jobs)

    return {
        "status": "ok", "n_calls": 2, "elapsed_s": elapsed,
        "json_object": stage_b_json, "stage_a_entities": entities,
        "schema_status": schema_status, "stage_b_entity_fixation_violation": violation,
    }


def main() -> None:
    client = OllamaClient()
    validator = SchemaValidator(SCHEMA_PATH)
    rows = []

    for rid, stem in POPULATION.items():
        md_path = MARKDOWN_DIR / f"{stem}.md"

        for run in range(1, N_REPEATS + 1):
            print(f"\n=== {rid} baseline run {run}/{N_REPEATS} ===", flush=True)
            b = _run_baseline_with_retry(md_path)
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            with (CKPT_DIR / f"{rid}__baseline_{run}.pkl").open("wb") as f:
                pickle.dump({"resume_id": rid, "stem": stem, "condition": "baseline", "run": run, **b}, f)
            n_jobs = len(b["json_object"].get("work_experience", [])) if b["json_object"] else None
            print(f"  status={b['status']} n_calls={b['n_calls']} elapsed={b['elapsed_s']} n_jobs={n_jobs}", flush=True)
            rows.append({"resume_id": rid, "condition": "baseline", "run": run, "status": b["status"],
                         "n_calls": b["n_calls"], "elapsed_s": b["elapsed_s"], "n_jobs": n_jobs})

        for run in range(1, N_REPEATS + 1):
            print(f"\n=== {rid} entity_first run {run}/{N_REPEATS} ===", flush=True)
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

    out = Path("benchmark/baselines/t13_pilot_results.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)", flush=True)
    print("\nT13 PILOT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
