"""
scripts/_milestone_t23_llm_decision.py

Milestone T23/Phase 2K -- narrow LLM decision layer for exactly the
SEMANTICALLY_AMBIGUOUS junctions identified by
scripts/_milestone_t23_ambiguity_detection.py.

Contract (frozen in docs/T23_PHASE2K_SEMANTIC_ASSEMBLY_AMBIGUITY_PREREGISTRATION.md):
the LLM receives exactly one ambiguity object (two candidate IDs plus
local text context) and must return KEEP_SEPARATE or MERGE_SPECIFIED,
referencing only the two supplied candidate IDs. No entity
reconstruction, no confirmed_entities-style summary (the explicit T21
lesson).

Uses services.ollama_client.OllamaClient.generate_text() unmodified --
the same production client and qwen3:1.7b model used in T21/T21B, no
temperature override, num_ctx=8192 (all unmodified defaults).
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from services.ollama_client import OllamaClient  # noqa: E402
from scripts._milestone_t23_ambiguity_detection import AmbiguityRecord  # noqa: E402

MODEL = "qwen3:1.7b"

ALLOWED_DECISIONS = {"KEEP_SEPARATE", "MERGE_SPECIFIED"}

_PROMPT_TEMPLATE = """You are resolving ONE narrow, pre-identified ambiguity about two existing candidate text spans from a resume, produced by a deterministic (non-AI) process. You are NOT extracting a resume. You are NOT discovering entities. You are NOT allowed to invent, add, or omit any candidate.

Both candidate spans below have already been judged, by an earlier separate step, to represent real Work Experience content. Your ONLY question is whether these two specific spans describe the SAME real job (e.g. one is a bare role/company heading and the other supplies its missing details), or whether they are TWO SEPARATE, DIFFERENT real roles that happen to sit next to each other in the resume (e.g. two different promotions or two different job titles at the same employer, each a distinct role in the person's career).

Neither span below contains an explicit date range -- that is exactly why this is ambiguous and cannot be resolved by structure alone.

CANDIDATE A (id={candidate_a_id}):
{text_a}

CANDIDATE B (id={candidate_b_id}):
{text_b}

RULES:
1. You may ONLY return one of these two decisions: "KEEP_SEPARATE" or "MERGE_SPECIFIED".
2. You may NOT invent a candidate ID. You may NOT reference any candidate ID other than {candidate_a_id} and {candidate_b_id}.
3. You may NOT create a company name, role name, or any new entity object. You are only choosing whether these two EXISTING spans belong together or not.
4. If A and B describe genuinely different roles/titles (even at the same employer), choose KEEP_SEPARATE.
5. If A and B are fragments of the same single role description (e.g. a heading with no other detail, immediately completed by B's content), choose MERGE_SPECIFIED.
6. Return ONLY the JSON object below. No prose, no explanation, no markdown fences.

Return exactly this JSON structure:
{{
  "ambiguity_id": "{ambiguity_id}",
  "decision": "KEEP_SEPARATE|MERGE_SPECIFIED",
  "merge_groups": [] or [["{candidate_a_id}", "{candidate_b_id}"]]
}}
"""


@dataclass
class LLMDecisionResult:
    ambiguity_id: str
    candidate_ids: List[str]
    raw_response: str = ""
    retry_raw_response: Optional[str] = None
    parsed: Optional[Dict[str, Any]] = None
    structural_retry_used: bool = False
    latency_seconds: float = 0.0
    error: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)


def build_prompt(ambiguity_id: str, candidate_a_id: str, candidate_b_id: str, text_a: str, text_b: str) -> str:
    return _PROMPT_TEMPLATE.format(
        ambiguity_id=ambiguity_id, candidate_a_id=candidate_a_id, candidate_b_id=candidate_b_id,
        text_a=text_a, text_b=text_b,
    )


def _extract_json_object(text: str) -> Optional[str]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def validate_response(parsed: Any, ambiguity_id: str, valid_ids: set) -> List[str]:
    errors = []
    if not isinstance(parsed, dict):
        return ["response is not a JSON object"]

    if parsed.get("ambiguity_id") != ambiguity_id:
        errors.append(f"invalid/mismatched ambiguity_id: {parsed.get('ambiguity_id')!r} (expected {ambiguity_id!r})")

    decision = parsed.get("decision")
    if decision not in ALLOWED_DECISIONS:
        errors.append(f"unsupported decision {decision!r}")

    merge_groups = parsed.get("merge_groups")
    if not isinstance(merge_groups, list):
        errors.append("merge_groups missing or not a list")
        merge_groups = []

    if decision == "KEEP_SEPARATE" and merge_groups:
        errors.append("KEEP_SEPARATE must have empty merge_groups")
    if decision == "MERGE_SPECIFIED" and not merge_groups:
        errors.append("MERGE_SPECIFIED must include a non-empty merge_groups")

    for group in merge_groups:
        if not isinstance(group, list):
            errors.append(f"malformed merge group (not a list): {group!r}")
            continue
        for cid in group:
            if cid not in valid_ids:
                errors.append(f"INVENTED or out-of-group candidate_id in merge_groups: {cid!r}")
        if set(group) != valid_ids and group:
            errors.append(f"merge group does not exactly match the ambiguity's own candidate pair: {group!r}")

    return errors


def resolve_ambiguity(rec: AmbiguityRecord, ambiguity_id: str, text_a: str, text_b: str,
                       client: OllamaClient) -> LLMDecisionResult:
    valid_ids = {rec.candidate_a_id, rec.candidate_b_id}
    prompt = build_prompt(ambiguity_id, rec.candidate_a_id, rec.candidate_b_id, text_a, text_b)
    result = LLMDecisionResult(ambiguity_id=ambiguity_id, candidate_ids=sorted(valid_ids))

    for attempt in range(2):  # 1 initial call + at most 1 bounded structural retry
        start = time.perf_counter()
        try:
            response = client.generate_text(MODEL, prompt)
        except Exception as exc:  # noqa: BLE001
            result.latency_seconds += time.perf_counter() - start
            result.error = f"call failed: {exc}"
            return result
        result.latency_seconds += time.perf_counter() - start
        raw_text = response.response
        if attempt == 0:
            result.raw_response = raw_text
        else:
            result.retry_raw_response = raw_text

        json_text = _extract_json_object(raw_text)
        if json_text is None:
            if attempt == 0:
                result.structural_retry_used = True
                continue
            result.error = "no JSON object found in response (after 1 structural retry)"
            return result
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError as exc:
            if attempt == 0:
                result.structural_retry_used = True
                continue
            result.error = f"JSON parse failed after retry: {exc}"
            return result

        errors = validate_response(parsed, ambiguity_id, valid_ids)
        if errors and attempt == 0:
            result.structural_retry_used = True
            continue
        if errors:
            result.error = f"invalid response after 1 structural retry: {errors}"
            result.validation_errors = errors
            return result

        result.parsed = parsed
        result.validation_errors = errors
        return result

    return result
