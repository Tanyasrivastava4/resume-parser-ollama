"""
scripts/_milestone_t22_assemble.py

Milestone T22/Phase 2J -- deterministic entity assembly from T21B's
frozen, accepted candidate decisions.

ZERO LLM CALLS. This module makes no network/model calls of any kind.

FROZEN INPUTS (read-only, never modified):
- T20/T21's own generate_candidates_t20() -- the deterministic
  candidate generator, unchanged.
- reports/t21b_candidate_decisions.csv -- T21B's own frozen candidate
  decisions, unchanged.

ACCEPTED-CANDIDATE RULE (reused verbatim from T21B's own
_milestone_t21b_evaluate.py, not redefined here):
    decision in {"KEEP", "CLASSIFY"} AND classification == "WORK_EXPERIENCE"

ASSEMBLY RULE (the one minimal rule frozen for T22, per the design note
in docs/T22_DETERMINISTIC_ENTITY_ASSEMBLY.md):

Sort accepted candidates by source order (extent_start). Scan left to
right. A candidate merges FORWARD into the entity currently being built
if and only if:
  (a) it is 0-gap adjacent to the immediately preceding accepted
      candidate (i.e. previous.extent_end + 1 == this.extent_start,
      with no other accepted candidate positioned between them), AND
  (b) the immediately preceding accepted candidate's own extent
      contains ZERO date-bearing blocks (via the pre-existing,
      unmodified Block.date_range_present feature).
Otherwise, the candidate starts a new entity.

This is a purely structural rule: it never reads company, title, or
any other field text, and never merges on company-string similarity
alone (i.e. "same employer does NOT automatically mean same entity").

Known, disclosed limitation (not patched): a sequence of consecutive,
genuinely separate, individually undated same-employer roles (no
candidate in the sequence has its own date) will over-merge under this
rule, because the only signal used (own-date presence) cannot
distinguish "incomplete fragment awaiting its date" from "a complete
but undated role." See the T22 report for the exact, disclosed case
(U2's Acting Manager / Analyst / Learning Specialist / Account
Protection Specialist chain).
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_t18_blocks import blockify  # noqa: E402
from scripts._milestone_t20_candidates import generate_candidates_t20, Candidate  # noqa: E402

ACCEPTED_DECISIONS = {"KEEP", "CLASSIFY"}
ACCEPTED_CLASSIFICATION = "WORK_EXPERIENCE"

DECISIONS_CSV = Path("reports/t21b_candidate_decisions.csv")
MARKDOWN_DIR = Path("output/markdown")


@dataclass
class Entity:
    entity_id: str
    candidate_ids: List[str] = field(default_factory=list)
    extent_start: int = 0
    extent_end: int = 0
    company_evidence: List[str] = field(default_factory=list)
    designation_evidence: List[str] = field(default_factory=list)
    date_evidence: List[str] = field(default_factory=list)


def load_t21b_decisions() -> Dict[str, Dict[str, dict]]:
    """Reads T21B's own frozen decisions CSV, unmodified. Returns
    {resume_id: {candidate_id: row}}."""
    by_resume: Dict[str, Dict[str, dict]] = {}
    with DECISIONS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_resume.setdefault(row["resume_id"], {})[row["candidate_id"]] = row
    return by_resume


def is_accepted(decision_row: Optional[dict]) -> bool:
    """The exact T21B accepted-candidate rule, reused verbatim."""
    return (
        decision_row is not None
        and decision_row["decision"] in ACCEPTED_DECISIONS
        and decision_row.get("classification") == ACCEPTED_CLASSIFICATION
    )


def _candidate_has_own_date(blocks, candidate: Candidate) -> bool:
    return any(blocks[i].date_range_present for i in range(candidate.extent_start, candidate.extent_end + 1))


def assemble_entities(blocks, accepted: List[Candidate]) -> List[Entity]:
    """The one frozen T22 assembly rule. `accepted` must already be
    sorted by extent_start. Returns entities in source order."""
    entities: List[Entity] = []
    if not accepted:
        return entities

    current = Entity(entity_id=f"E{len(entities)}", candidate_ids=[f"C{accepted[0].candidate_id}"],
                      extent_start=accepted[0].extent_start, extent_end=accepted[0].extent_end)
    entities.append(current)

    for i in range(1, len(accepted)):
        prev = accepted[i - 1]
        cur = accepted[i]
        zero_gap = cur.extent_start == prev.extent_end + 1
        prev_lacks_date = not _candidate_has_own_date(blocks, prev)
        if zero_gap and prev_lacks_date:
            # Merge cur forward into the currently open entity.
            entities[-1].candidate_ids.append(f"C{cur.candidate_id}")
            entities[-1].extent_end = cur.extent_end
        else:
            new_entity = Entity(entity_id=f"E{len(entities)}", candidate_ids=[f"C{cur.candidate_id}"],
                                 extent_start=cur.extent_start, extent_end=cur.extent_end)
            entities.append(new_entity)

    for e in entities:
        for i in range(e.extent_start, e.extent_end + 1):
            b = blocks[i]
            if b.date_range_present:
                e.date_evidence.append(b.stripped)
        # Company/designation evidence: the anchor text of each member
        # candidate, verbatim -- no semantic extraction, no
        # normalization, just the raw structural anchor already
        # computed by the frozen candidate generator.
        for cid in e.candidate_ids:
            cand = next(c for c in accepted if f"C{c.candidate_id}" == cid)
            e.company_evidence.append(cand.anchor_text)

    return entities


def run_resume(rid: str, stem: str, decisions_by_resume: Dict[str, Dict[str, dict]]):
    blocks = blockify((MARKDOWN_DIR / f"{stem}.md").read_text(encoding="utf-8"))
    candidates = generate_candidates_t20(blocks)
    decisions = decisions_by_resume.get(rid, {})
    accepted = sorted(
        [c for c in candidates if is_accepted(decisions.get(f"C{c.candidate_id}"))],
        key=lambda c: c.extent_start,
    )
    entities = assemble_entities(blocks, accepted)
    return blocks, candidates, accepted, entities
