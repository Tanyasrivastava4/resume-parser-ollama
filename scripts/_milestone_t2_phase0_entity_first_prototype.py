"""
scripts/_milestone_t2_phase0_entity_first_prototype.py

Milestone T2 / Phase 0 -- Phase C: EXPERIMENTAL, RESEARCH-ONLY entity-
first Work Experience prototype. NOT production code, NOT imported by
services/ or pipeline/, does not modify WorkExperienceExtractor or any
production prompt/schema. Isolated exactly like every prior real-call
research script in this project (V5.6/V6.1's own pattern).

Compares two conditions on the SAME markdown/model/schema:

  BASELINE: exactly WorkExperienceExtractor's own production contract
  (one call: prompt + full markdown -> full work_experience array),
  reused directly, not reimplemented.

  ENTITY-FIRST: two calls, batched at section level (never one call per
  entity, per the governing architecture document's own explicit
  instruction):
    Stage 1 (segmentation): a NEW, narrower prompt asking the model
    ONLY to identify how many distinct employment entities exist and
    each one's company name, in resume order -- no other fields.
    Stage 2 (field extraction): a NEW prompt that supplies the Stage-1
    entity list back to the model and asks it to fill in the full
    production schema for EXACTLY those entities, in that exact order
    and count -- reusing the production schema unchanged.

Both conditions use the same model (qwen3:1.7b, the current baseline),
the same source markdown, and are validated against the SAME production
work_experience_schema.json. Real Ollama calls. Checkpointed per
(resume, condition, run) slot, resumable, matching V5.6/V6.1's own
resilience pattern.
"""
from __future__ import annotations

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
from validators.schema_validator import SchemaValidator  # noqa: E402
from config import SCHEMAS_DIR  # noqa: E402

MODEL = "qwen3:1.7b"  # current production baseline model -- unchanged
SCHEMA_PATH = SCHEMAS_DIR / "modules" / "work_experience_schema.json"
CKPT_DIR = Path(
    "/private/tmp/claude-501/-Users-ruchigupta-Documents-ResumeAIBenchmark/"
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/t2_phase0_checkpoints"
)
# NOTE (remediation CI gate fix, 2026-10-17): CKPT_DIR is this script's
# original, machine-specific scratch location, kept verbatim so the recorded
# milestone remains reproducible on the machine it ran on. It used to be
# created HERE, at import time, which made merely importing this module
# (which the test suite does) fail on any other machine (PermissionError on
# /private on the Linux CI runner). It is now created lazily, immediately
# before the first checkpoint is written. Nothing else about the script, its
# results or its tests changes.

_SEGMENTATION_PROMPT = """You are an enterprise Resume Parser performing ONLY entity segmentation.

You are given Markdown extracted from a resume.

Your ONLY task: identify each distinct, genuine employment entity (full-time roles, part-time roles, internships, freelance engagements, consulting stints, self-employed periods) in the Work Experience section of this resume.

Do NOT extract full details. Do NOT include Education, Skills, Projects, Certifications, Awards, Achievements, Publications, or Conferences content -- a new section heading such as "Projects", "Education", "Skills", or "Certifications" ends the work experience content above it; nothing under those headings is an employment entity.

For each genuine employment entity, in the EXACT order they appear in the resume, report only its company/organization name, exactly as written in the resume.

Output ONLY valid JSON in this exact shape, nothing else:
{"entities": ["Company Name 1", "Company Name 2", ...]}

If there is no genuine work experience at all, output {"entities": []}.
"""

_FIELD_EXTRACTION_PROMPT_TEMPLATE = """You are an enterprise Resume Parser.

You are given Markdown extracted from a resume, AND a pre-identified list of the EXACT employment entities (in order) that exist in this resume's Work Experience section:

{entity_list_json}

Your task: for EACH of these entities, in this exact order and count (do not add, remove, merge, or split any), extract its full details from the resume.

Do NOT extract any of the following, even partially: Personal information, contact details, skills, languages, professional summary, Education, Certifications, Projects, Awards, Achievements, Publications, Conferences.

STRICT RULES:
1. Output ONLY valid JSON. Never explain anything. Never add markdown formatting. Nothing before or after the JSON.
2. Never invent, infer, or guess any value. If a value is not explicitly present in the resume, return `null` for it.
3. You MUST produce EXACTLY {n_entities} entries in the output array, one per entity listed above, in the same order.
4. Each entry must include all ten fields: company, designation, employment_type, location, start_date, end_date, currently_working, duration, responsibilities, technologies -- use null for any field not explicitly present. currently_working must be true or false (never null). responsibilities and technologies must be arrays (use [] if none).
5. Preserve chronology and exact wording from the resume for responsibilities and technologies.

Output shape: {{"work_experience": [ {{...}}, {{...}} ]}}
"""


def _extract_json(text: str) -> dict:
    """Parse a model response as JSON, tolerating a stray code fence
    (unlike production's own _parse_json, which deliberately does not
    -- this experimental harness is more permissive on purpose, since
    the new segmentation/field-extraction prompts are themselves
    experimental and not yet hardened the way the production prompt is)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def run_baseline(markdown_path: Path, client: OllamaClient) -> dict:
    """Reuses the EXACT production WorkExperienceExtractor contract --
    not reimplemented. Real call. A genuine production-contract failure
    (InvalidJSONError/SchemaValidationError -- the same failures
    production's own retry-then-fail behavior would surface) is caught
    and RECORDED as a real, disclosed failed slot, not silently
    swallowed or retried beyond what production itself would do -- this
    experimental harness does not add retry behavior baseline doesn't
    already have in this direct-call form (matching V5.6/V6.1's own
    "one predeclared retry, then record as a genuine failure" pattern
    would require an explicit retry policy; this prototype has none by
    design, so ANY failure here is recorded as-is, once, no retry)."""
    extractor = WorkExperienceExtractor(ollama_client=client)
    t0 = time.perf_counter()
    try:
        result = extractor.extract(markdown_path, model_name=MODEL)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "condition": "baseline", "n_calls": 1, "elapsed_seconds": elapsed,
            "status": "failed", "error_type": type(e).__name__, "error": str(e),
        }
    elapsed = time.perf_counter() - t0
    return {
        "condition": "baseline", "n_calls": 1, "elapsed_seconds": elapsed,
        "json_object": result.json_object, "status": "ok",
    }


def run_entity_first(markdown_path: Path, client: OllamaClient, validator: SchemaValidator) -> dict:
    """Two real calls, batched at section level. Isolated -- does not
    call WorkExperienceExtractor or touch its schema/prompt files."""
    markdown_text = markdown_path.read_text(encoding="utf-8")
    t0 = time.perf_counter()
    n_calls = 0

    # Stage 1: segmentation
    seg_prompt = f"{_SEGMENTATION_PROMPT}\n\n{markdown_text}"
    seg_response = client.generate_text(model=MODEL, prompt=seg_prompt)
    n_calls += 1
    try:
        seg_json = _extract_json(seg_response.response)
        entities = seg_json.get("entities", [])
        if not isinstance(entities, list):
            entities = []
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {"condition": "entity_first", "n_calls": n_calls, "elapsed_seconds": elapsed,
                "status": "segmentation_parse_failed", "error": str(e), "raw_segmentation": seg_response.response}

    if not entities:
        elapsed = time.perf_counter() - t0
        return {"condition": "entity_first", "n_calls": n_calls, "elapsed_seconds": elapsed,
                "status": "ok", "json_object": {"work_experience": []}, "segmented_entities": entities}

    # Stage 2: field extraction, batched (ONE call for all entities)
    field_prompt = _FIELD_EXTRACTION_PROMPT_TEMPLATE.format(
        entity_list_json=json.dumps(entities), n_entities=len(entities),
    ) + f"\n\n{markdown_text}"
    field_response = client.generate_text(model=MODEL, prompt=field_prompt)
    n_calls += 1
    elapsed = time.perf_counter() - t0

    try:
        field_json = _extract_json(field_response.response)
    except Exception as e:
        return {"condition": "entity_first", "n_calls": n_calls, "elapsed_seconds": elapsed,
                "status": "field_extraction_parse_failed", "error": str(e),
                "segmented_entities": entities, "raw_field_extraction": field_response.response}

    try:
        validator.validate(field_json)
        schema_status = "valid"
    except Exception as e:
        schema_status = f"invalid: {e}"

    return {
        "condition": "entity_first", "n_calls": n_calls, "elapsed_seconds": elapsed,
        "status": "ok", "json_object": field_json, "segmented_entities": entities,
        "schema_status": schema_status,
    }


def run_slot(resume_id: str, stem: str, condition: str, run_idx: int, markdown_dir: Path) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{resume_id}__{condition}__run{run_idx}.pkl"
    if ckpt_path.exists():
        print(f"SKIP (already checkpointed): {ckpt_path.name}")
        return

    client = OllamaClient()
    validator = SchemaValidator(SCHEMA_PATH)
    markdown_path = markdown_dir / f"{stem}.md"

    print(f"RUNNING: {resume_id} {condition} run{run_idx} ...")
    if condition == "baseline":
        result = run_baseline(markdown_path, client)
    else:
        result = run_entity_first(markdown_path, client, validator)

    with ckpt_path.open("wb") as f:
        pickle.dump({"resume_id": resume_id, "condition": condition, "run_index": run_idx, "result": result}, f)
    print(f"DONE: {resume_id} {condition} run{run_idx} -> status={result.get('status')} "
          f"n_calls={result.get('n_calls')} elapsed={result.get('elapsed_seconds', 0):.1f}s")
