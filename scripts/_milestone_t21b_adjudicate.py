"""
scripts/_milestone_t21b_adjudicate.py

Milestone T21B/Phase 2I -- candidate-DECISION-ONLY diagnostic.

Isolates the candidate-level decision mechanism from T21's own
confirmed_entities assembly step, which T21 found collapsed to exactly
one entity in all 9 resumes regardless of true count. T21B tests
whether removing that assembly step reveals a materially stronger
candidate-level decision signal.

Reuses, UNMODIFIED:
- scripts._milestone_t20_candidates.generate_candidates_t20 (the exact
  T20/T21 deterministic candidate generator -- not re-implemented,
  not tuned).
- scripts._milestone_t21_adjudicate.build_candidate_inventory (the
  exact T21 candidate-summarization function used to build the
  per-candidate prompt payload) -- confirmed byte-for-byte reproduces
  T21's own frozen reports/t21_baseline_candidate_inventory.csv before
  this script was written.
- services.ollama_client.OllamaClient.generate_text() -- the same
  production client, same qwen3:1.7b model, no temperature override,
  same num_ctx=8192, same timeout.
- The same JSON-extraction tolerance (_extract_json_object) and the
  same bounded-structural-retry-only policy T21 used.

Does NOT reuse T21's validate_response() as-is, because that function
also validates a confirmed_entities section that no longer exists in
the T21B contract. A new, narrower validator is defined here that
checks ONLY the candidate_decisions contract: every supplied candidate
ID must receive exactly one decision, no invented IDs, no duplicates,
only the allowed decision/classification vocabulary.

NO confirmed_entities field exists anywhere in this contract. The LLM
performs NO entity assembly, NO merge, NO split.
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
from scripts._milestone_t21_adjudicate import build_candidate_inventory  # noqa: E402 (reused unmodified)

MODEL = "qwen3:1.7b"
MARKDOWN_DIR = Path("output/markdown")

# T21B's own explicit, frozen vocabulary (per this milestone's brief --
# narrower on decisions (no MERGE), broader on classification (adds
# EDUCATION) than T21's own vocabulary, which is NOT reused verbatim
# here because the brief explicitly mandates this exact set).
VALID_DECISIONS = {"KEEP", "DISCARD", "CLASSIFY"}
VALID_CLASSIFICATIONS = {"WORK_EXPERIENCE", "PROJECT", "EDUCATION", "OTHER_NON_EXPERIENCE"}

_PROMPT_TEMPLATE = """You are classifying a FROZEN list of candidate text spans extracted from one resume by a deterministic (non-AI) process. You are NOT extracting a resume from scratch. You are NOT allowed to invent, discover, or add any candidate that is not already in the list below.

Treat every candidate independently. There may be zero, one, or many Work Experience candidates in this list -- the number is NOT predetermined and is not related to how many candidates you see. Do not summarize the candidates. Do not create an entity list. Do not collapse multiple candidates into one entity. Your ONLY job is to give one independent decision for every candidate listed.

RULES (must follow exactly):
1. The candidate IDs below are fixed and cannot be changed or invented. You may only refer to IDs that appear in the list.
2. Candidate text is derived directly from the source document. You cannot create new source spans or new text.
3. You must produce EXACTLY ONE decision for EVERY candidate ID listed, and nothing else. Do not skip any candidate. Do not output an ID twice. Do not output an ID that is not in the list.
4. Allowed decision values: KEEP, DISCARD, CLASSIFY.
5. Allowed classification values (only relevant when decision is KEEP or CLASSIFY): WORK_EXPERIENCE, PROJECT, EDUCATION, OTHER_NON_EXPERIENCE.
6. A section heading by itself (e.g. "Work Experience", "Experience", "Skills", "Education") is NOT a Work Experience entity -- DISCARD it, or CLASSIFY it as OTHER_NON_EXPERIENCE.
7. A Project or personal-project heading is NOT Work Experience -- CLASSIFY it as PROJECT.
8. An education entry (degree, school, certification) is NOT Work Experience -- CLASSIFY it as EDUCATION.
9. Prefer DISCARD or CLASSIFY as OTHER_NON_EXPERIENCE when a candidate is clearly not a Work Experience entity (contact info, skills, summary, table rows, isolated section headings).
10. You MUST classify a candidate as WORK_EXPERIENCE (KEEP or CLASSIFY) if it represents a real job, even if it is unusual: undated, an internship, a same-employer re-entry/promotion role, or an old/tail entry. Do not discard a real job merely because it looks unusual.
11. Do NOT use your own general knowledge of resumes to invent a judgment not backed by the candidate's own text.
12. If you cannot determine the correct classification with confidence, you must still provide your best decision using only the allowed vocabulary -- never leave a candidate out.
13. Return ONLY the JSON object described below. No prose, no explanation, no markdown fences. Do not include any "confirmed_entities" field, entity IDs, merged groups, spans, or extracted fields of any kind.

Example (illustrates handling MULTIPLE candidates with DIFFERENT outcomes -- this is NOT a hint about how many Work Experience candidates exist in the real list below):
Given candidates C0 ("## Skills"), C1 ("### Acme Corp — Senior Engineer, Jan 2020 - Present"), C2 ("### Acme Corp — Junior Engineer, 2018 - 2020"):
{{
  "candidate_decisions": [
    {{"candidate_id": "C0", "decision": "DISCARD", "classification": "OTHER_NON_EXPERIENCE"}},
    {{"candidate_id": "C1", "decision": "KEEP", "classification": "WORK_EXPERIENCE"}},
    {{"candidate_id": "C2", "decision": "KEEP", "classification": "WORK_EXPERIENCE"}}
  ]
}}
(Note both C1 and C2 were independently kept as WORK_EXPERIENCE -- there is no limit of one.)

CANDIDATE LIST for this resume:
{candidates_json}

Return exactly this JSON structure and nothing else, with one entry per candidate ID above:
{{
  "candidate_decisions": [
    {{"candidate_id": "<one of the IDs above>", "decision": "KEEP|DISCARD|CLASSIFY", "classification": "WORK_EXPERIENCE|PROJECT|EDUCATION|OTHER_NON_EXPERIENCE|null"}}
  ]
}}
"""


@dataclass
class T21BCallResult:
    resume_id: str
    stem: str
    n_candidates: int
    raw_response: str = ""
    retry_raw_response: Optional[str] = None
    parsed: Optional[Dict[str, Any]] = None
    structural_retry_used: bool = False
    latency_seconds: float = 0.0
    error: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)


def build_prompt(candidate_inventory: List[Dict[str, Any]]) -> str:
    return _PROMPT_TEMPLATE.format(
        candidates_json=json.dumps(candidate_inventory, ensure_ascii=False, indent=2)
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


def validate_response(parsed: Any, valid_ids: set) -> List[str]:
    """Validate ONLY the candidate_decisions contract. No
    confirmed_entities field exists in T21B's contract at all -- if a
    response includes one, that is itself flagged as a rule violation
    (rule 13: 'no confirmed_entities field of any kind')."""
    errors = []
    if not isinstance(parsed, dict):
        return ["response is not a JSON object"]

    if "confirmed_entities" in parsed:
        errors.append("response illegally includes a confirmed_entities field (forbidden by the T21B contract)")

    decisions = parsed.get("candidate_decisions")
    if not isinstance(decisions, list):
        errors.append("candidate_decisions missing or not a list")
        decisions = []

    seen_ids = set()
    for d in decisions:
        if not isinstance(d, dict):
            errors.append(f"decision entry not an object: {d!r}")
            continue
        cid = d.get("candidate_id")
        if cid not in valid_ids:
            errors.append(f"INVENTED candidate_id: {cid!r}")
        if cid in seen_ids:
            errors.append(f"DUPLICATE candidate_decisions entry for {cid!r}")
        seen_ids.add(cid)
        decision = d.get("decision")
        if decision not in VALID_DECISIONS:
            errors.append(f"unsupported decision {decision!r} for {cid!r}")
        cls = d.get("classification")
        if cls is not None and cls not in VALID_CLASSIFICATIONS:
            errors.append(f"unsupported classification {cls!r} for {cid!r}")

    missing = valid_ids - seen_ids
    if missing:
        errors.append(f"MISSING decisions for candidate_ids: {sorted(missing, key=lambda s: int(s[1:]))}")

    return errors


def adjudicate_resume(rid: str, stem: str, client: OllamaClient) -> T21BCallResult:
    blocks = blockify((MARKDOWN_DIR / f"{stem}.md").read_text(encoding="utf-8"))
    candidates = generate_candidates_t20(blocks)
    inventory = build_candidate_inventory(blocks, candidates)
    valid_ids = {item["candidate_id"] for item in inventory}
    prompt = build_prompt(inventory)

    result = T21BCallResult(resume_id=rid, stem=stem, n_candidates=len(candidates))

    for attempt in range(2):  # 1 initial call + at most 1 bounded STRUCTURAL retry
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

        errors = validate_response(parsed, valid_ids)
        if errors and attempt == 0:
            result.structural_retry_used = True
            continue

        result.parsed = parsed
        result.validation_errors = errors
        return result

    return result
