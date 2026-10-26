"""
scripts/_milestone_t23_assemble.py

Milestone T23/Phase 2K -- hybrid deterministic + narrow-LLM entity
assembly.

Architecture:
  Frozen accepted candidates (T21B, unmodified)
    -> ambiguity detection (scripts/_milestone_t23_ambiguity_detection.py)
    -> CLEAR_MERGE / CLEAR_SEPARATE junctions: T22's own decision reused
       verbatim, no LLM involved.
    -> SEMANTICALLY_AMBIGUOUS junctions only: one narrow LLM call per
       ambiguity (scripts/_milestone_t23_llm_decision.py), producing
       KEEP_SEPARATE or MERGE_SPECIFIED for that exact pair.
    -> Deterministic final assembly: entities are built by literally
       re-running T22's own grouping logic, with the ONLY change being
       that an ambiguous junction's merge/no-merge outcome is taken
       from the LLM's decision instead of the date-based default,
       wherever the LLM successfully produced a valid decision.

T22's own assemble_entities() and its date-based rule are NOT modified
in place -- this module implements its own entity-building pass, reusing
T22's own building blocks (Entity dataclass, _candidate_has_own_date)
unchanged, so T22's own file remains re-runnable byte-for-byte as the
frozen baseline.

No LLM call is made for CLEAR_MERGE or CLEAR_SEPARATE junctions.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from services.ollama_client import OllamaClient  # noqa: E402
from scripts._milestone_t18_blocks import blockify  # noqa: E402
from scripts._milestone_t20_candidates import generate_candidates_t20, Candidate  # noqa: E402
from scripts._milestone_t22_assemble import (  # noqa: E402
    Entity,
    load_t21b_decisions,
    is_accepted,
    _candidate_has_own_date,
)
from scripts._milestone_t23_ambiguity_detection import classify_junction, _local_context  # noqa: E402
from scripts._milestone_t23_llm_decision import resolve_ambiguity, LLMDecisionResult  # noqa: E402

MARKDOWN_DIR = Path("output/markdown")


def _candidate_text(blocks, c: Candidate, max_chars: int = 500) -> str:
    return " | ".join(blocks[i].stripped for i in range(c.extent_start, c.extent_end + 1))[:max_chars]


def assemble_entities_hybrid(rid: str, blocks, accepted: List[Candidate], client: Optional[OllamaClient]):
    """Returns (entities, llm_call_log) where llm_call_log is a list of
    LLMDecisionResult, one per SEMANTICALLY_AMBIGUOUS junction actually
    escalated (empty if none for this resume -- e.g. every mandatory
    control other than U2)."""
    entities: List[Entity] = []
    llm_calls: List[LLMDecisionResult] = []
    if not accepted:
        return entities, llm_calls

    # Pre-classify every junction once, exactly as T23's own detector does.
    junctions = {}
    for i in range(len(accepted) - 1):
        a, b = accepted[i], accepted[i + 1]
        if b.extent_start == a.extent_end + 1:
            junctions[i] = classify_junction(blocks, a, b)

    ambiguity_counter = 0
    current = Entity(entity_id="E0", candidate_ids=[f"C{accepted[0].candidate_id}"],
                      extent_start=accepted[0].extent_start, extent_end=accepted[0].extent_end)
    entities.append(current)

    for i in range(1, len(accepted)):
        prev = accepted[i - 1]
        cur = accepted[i]
        j = junctions.get(i - 1)

        should_merge = False
        if j is not None:
            if j.classification == "CLEAR_MERGE":
                should_merge = True
            elif j.classification == "CLEAR_SEPARATE":
                should_merge = False
            elif j.classification == "SEMANTICALLY_AMBIGUOUS":
                ambiguity_counter += 1
                ambiguity_id = f"{rid}_AMB_{ambiguity_counter:03d}"
                text_a = _candidate_text(blocks, prev)
                text_b = _candidate_text(blocks, cur)
                if client is None:
                    raise RuntimeError("SEMANTICALLY_AMBIGUOUS junction found but no OllamaClient provided")
                result = resolve_ambiguity(j, ambiguity_id, text_a, text_b, client)
                llm_calls.append(result)
                if result.parsed is not None:
                    should_merge = result.parsed.get("decision") == "MERGE_SPECIFIED"
                else:
                    # LLM failed even after the one bounded structural
                    # retry -- fall back to T22's own frozen
                    # deterministic decision (never silently drop the
                    # candidate, never invent a resolution).
                    should_merge = not _candidate_has_own_date(blocks, prev)

        if should_merge:
            entities[-1].candidate_ids.append(f"C{cur.candidate_id}")
            entities[-1].extent_end = cur.extent_end
        else:
            entities.append(Entity(entity_id=f"E{len(entities)}", candidate_ids=[f"C{cur.candidate_id}"],
                                    extent_start=cur.extent_start, extent_end=cur.extent_end))

    for e in entities:
        for k in range(e.extent_start, e.extent_end + 1):
            b = blocks[k]
            if b.date_range_present:
                e.date_evidence.append(b.stripped)
        for cid in e.candidate_ids:
            cand = next(c for c in accepted if f"C{c.candidate_id}" == cid)
            e.company_evidence.append(cand.anchor_text)

    return entities, llm_calls


def run_resume_hybrid(rid: str, stem: str, decisions_by_resume: Dict[str, Dict[str, dict]],
                       client: Optional[OllamaClient]):
    blocks = blockify((MARKDOWN_DIR / f"{stem}.md").read_text(encoding="utf-8"))
    candidates = generate_candidates_t20(blocks)
    dmap = decisions_by_resume.get(rid, {})
    accepted = sorted(
        [c for c in candidates if is_accepted(dmap.get(f"C{c.candidate_id}"))],
        key=lambda c: c.extent_start,
    )
    entities, llm_calls = assemble_entities_hybrid(rid, blocks, accepted, client)
    return blocks, candidates, accepted, entities, llm_calls
