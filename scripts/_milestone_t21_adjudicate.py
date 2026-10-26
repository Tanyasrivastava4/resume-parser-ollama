"""
scripts/_milestone_t21_adjudicate.py

Milestone T21/Phase 2H -- LLM candidate ambiguity resolution pilot.

Architecture under test:
  Deterministic candidates (T20's generate_candidates_t20, unmodified)
  -> frozen candidate inventory -> ONE batched LLM adjudication call
  per resume -> confirmed entities, expressed ONLY in terms of
  existing candidate IDs.

The LLM is NEVER the source of entity discovery. It receives a frozen
list of candidate IDs and may only KEEP, DISCARD, MERGE, or CLASSIFY
them. It may not invent a candidate ID, a new source span, a company,
a designation, or a date. If the correct answer cannot be expressed
using existing candidate IDs, the contract requires
`unrepresentable_by_candidates: true` instead of a free-form answer.

Uses services.ollama_client.OllamaClient.generate_text() unmodified --
the same production client T13-T16 used, with the same "qwen3:1.7b"
model, no temperature override (project/production default), no
context-length override (production default num_ctx=8192).

No production file modified. No service/pipeline/schema/prompt file
touched -- the adjudication prompt lives ONLY in this research script,
not in prompts/.
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
from scripts._milestone_t18_blocks import blockify  # noqa: E402
from scripts._milestone_t20_candidates import generate_candidates_t20  # noqa: E402

MODEL = "qwen3:1.7b"
MARKDOWN_DIR = Path("output/markdown")

VALID_DECISIONS = {"KEEP", "DISCARD", "MERGE", "CLASSIFY"}
VALID_CLASSIFICATIONS = {"WORK_EXPERIENCE", "PROJECT", "OTHER_NON_EXPERIENCE"}

_PROMPT_TEMPLATE = """You are adjudicating a FROZEN list of candidate text spans extracted from one resume by a deterministic (non-AI) process. You are NOT extracting a resume from scratch. You are NOT allowed to invent, discover, or add any candidate that is not already in the list below.

RULES (must follow exactly):
1. The candidate IDs below are fixed and cannot be changed or invented. You may only refer to IDs that appear in the list.
2. Candidate text is derived directly from the source document. You cannot create new source spans or new text.
3. You must produce a decision for EVERY candidate ID listed: KEEP, DISCARD, MERGE, or CLASSIFY.
4. Multiple candidate IDs MAY represent the same real Work Experience entity -- use MERGE and list all the candidate IDs that belong together in "merge_group".
5. A section heading by itself (e.g. "Work Experience", "Experience", "Skills", "Education") is NOT a Work Experience entity -- DISCARD it.
6. A Project or personal-project heading is NOT Work Experience -- CLASSIFY it as PROJECT, do not treat it as a confirmed Work Experience entity.
7. Prefer DISCARD when a candidate is clearly not a Work Experience entity (contact info, education, skills, certifications, summary, table rows, isolated section headings).
8. You MUST preserve (KEEP or MERGE, classified WORK_EXPERIENCE) any candidate that represents a real job, even if it is unusual: undated, an internship, a same-employer re-entry/promotion role, or an old/tail entry. Do not discard a real job merely because it looks unusual.
9. Do NOT use your own general knowledge of resumes to invent an entity that is not backed by one or more of the candidate IDs below.
10. Every "confirmed_entities" entry must list one or more EXISTING candidate_id values from the list below. You may never introduce an id that is not in the list.
11. If you believe the correct entity inventory cannot be fully expressed using only the candidate IDs given (e.g. a single candidate actually contains two distinct jobs and there is no way to split it using existing IDs), do not invent a split. Instead set "unrepresentable_by_candidates": true for that situation and explain nothing further.
12. Return ONLY the JSON object described below. No prose, no explanation, no markdown fences.

CANDIDATE LIST for this resume:
{candidates_json}

Return exactly this JSON structure and nothing else:
{{
  "candidate_decisions": [
    {{"candidate_id": "<one of the IDs above>", "decision": "KEEP|DISCARD|MERGE|CLASSIFY", "classification": "WORK_EXPERIENCE|PROJECT|OTHER_NON_EXPERIENCE|null", "merge_group": null or ["<candidate_id>", "<candidate_id>", ...]}}
  ],
  "confirmed_entities": [
    {{"entity_id": "E1", "candidate_ids": ["<one or more existing candidate_id values>"], "classification": "WORK_EXPERIENCE"}}
  ],
  "unrepresentable_by_candidates": false
}}
"""


@dataclass
class AdjudicationCallResult:
    resume_id: str
    stem: str
    n_candidates: int
    raw_response: str = ""
    parsed: Optional[Dict[str, Any]] = None
    structural_retry_used: bool = False
    latency_seconds: float = 0.0
    error: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)


def _preview(blocks, extent_start, extent_end, max_chars=220):
    text = " | ".join(blocks[i].stripped for i in range(extent_start, extent_end + 1))
    return text[:max_chars]


def build_candidate_inventory(blocks, candidates) -> List[Dict[str, Any]]:
    items = []
    for c in candidates:
        items.append({
            "candidate_id": f"C{c.candidate_id}",
            "anchor_text": c.anchor_text[:120],
            "section_label": c.containing_section_label,
            "n_blocks": c.n_blocks_in_extent,
            "flags": sorted(k for k, v in c.flags.items() if v),
            "text_preview": _preview(blocks, c.extent_start, c.extent_end),
        })
    return items


def build_prompt(candidate_inventory: List[Dict[str, Any]]) -> str:
    return _PROMPT_TEMPLATE.format(
        candidates_json=json.dumps(candidate_inventory, ensure_ascii=False, indent=2)
    )


def _extract_json_object(text: str) -> Optional[str]:
    """Find the first top-level {...} JSON object in the model's raw
    text output, tolerating a <think>...</think> preamble or markdown
    fences (structural leniency only -- never a semantic retry)."""
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


def validate_response(parsed: Dict[str, Any], valid_ids: set) -> List[str]:
    """Independently verify the candidate-only constraint. Returns a
    list of violation strings (empty list = fully valid)."""
    errors = []
    if not isinstance(parsed, dict):
        return ["response is not a JSON object"]

    if parsed.get("unrepresentable_by_candidates") is True:
        return errors  # a valid, honest "cannot represent" answer

    decisions = parsed.get("candidate_decisions")
    if not isinstance(decisions, list):
        errors.append("candidate_decisions missing or not a list")
        decisions = []

    seen_decision_ids = set()
    for d in decisions:
        if not isinstance(d, dict):
            errors.append(f"decision entry not an object: {d!r}")
            continue
        cid = d.get("candidate_id")
        if cid not in valid_ids:
            errors.append(f"INVENTED candidate_id in candidate_decisions: {cid!r}")
        if cid in seen_decision_ids:
            errors.append(f"duplicate candidate_decisions entry for {cid!r}")
        seen_decision_ids.add(cid)
        decision = d.get("decision")
        if decision not in VALID_DECISIONS:
            errors.append(f"unsupported decision {decision!r} for {cid!r}")
        cls = d.get("classification")
        if cls is not None and cls not in VALID_CLASSIFICATIONS:
            errors.append(f"unsupported classification {cls!r} for {cid!r}")
        mg = d.get("merge_group")
        if mg is not None:
            if not isinstance(mg, list):
                errors.append(f"merge_group for {cid!r} is not a list")
            else:
                for mid in mg:
                    if mid not in valid_ids:
                        errors.append(f"INVENTED candidate_id in merge_group: {mid!r}")

    missing = valid_ids - seen_decision_ids
    if missing:
        errors.append(f"missing decisions for candidate_ids: {sorted(missing)}")

    entities = parsed.get("confirmed_entities")
    if not isinstance(entities, list):
        errors.append("confirmed_entities missing or not a list")
        entities = []

    seen_entity_ids = set()
    for e in entities:
        if not isinstance(e, dict):
            errors.append(f"confirmed_entities entry not an object: {e!r}")
            continue
        eid = e.get("entity_id")
        if eid in seen_entity_ids:
            errors.append(f"duplicate entity_id: {eid!r}")
        seen_entity_ids.add(eid)
        cids = e.get("candidate_ids")
        if not isinstance(cids, list) or not cids:
            errors.append(f"entity {eid!r} has no candidate_ids")
            cids = []
        for cid in cids:
            if cid not in valid_ids:
                errors.append(f"INVENTED candidate_id in confirmed_entities: {cid!r} (entity {eid!r})")
        cls = e.get("classification")
        if cls not in VALID_CLASSIFICATIONS:
            errors.append(f"entity {eid!r} has unsupported classification {cls!r}")

    return errors


def adjudicate_resume(rid: str, stem: str, client: OllamaClient) -> AdjudicationCallResult:
    blocks = blockify((MARKDOWN_DIR / f"{stem}.md").read_text(encoding="utf-8"))
    candidates = generate_candidates_t20(blocks)
    inventory = build_candidate_inventory(blocks, candidates)
    valid_ids = {item["candidate_id"] for item in inventory}
    prompt = build_prompt(inventory)

    result = AdjudicationCallResult(resume_id=rid, stem=stem, n_candidates=len(candidates))

    for attempt in range(2):  # 1 initial call + 1 bounded STRUCTURAL retry only
        start = time.perf_counter()
        try:
            response = client.generate_text(MODEL, prompt)
        except Exception as exc:  # noqa: BLE001
            result.latency_seconds += time.perf_counter() - start
            result.error = f"call failed: {exc}"
            return result
        elapsed = time.perf_counter() - start
        result.latency_seconds += elapsed
        raw_text = response.response
        result.raw_response = raw_text

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

        errors = validate_response(parsed, valid_ids)
        if errors and attempt == 0:
            # A schema/structural violation (e.g. invented ID, missing
            # field) is grounds for exactly one bounded structural
            # retry, per protocol -- never a semantic re-roll.
            result.structural_retry_used = True
            continue

        result.parsed = parsed
        result.validation_errors = errors
        return result

    return result
