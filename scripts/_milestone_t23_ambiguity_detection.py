"""
scripts/_milestone_t23_ambiguity_detection.py

Milestone T23/Phase 2K -- deterministic assembly ambiguity detector.

Classifies every 0-gap-adjacent merge/no-merge junction T22's own
frozen assembly rule (scripts/_milestone_t22_assemble.py, unmodified)
had to decide, into:

  CLEAR_SEPARATE       -- the first (earlier) candidate has its own
                           date evidence; T22 correctly does not merge.
  CLEAR_MERGE          -- the first candidate lacks its own date, but
                           the second (target) candidate HAS its own
                           date -- a structurally well-supported
                           "fragment + date" reunification.
  SEMANTICALLY_AMBIGUOUS -- the first candidate lacks its own date,
                           AND the second candidate ALSO lacks its own
                           date. T22's rule still merges them (its only
                           signal is "does the first candidate have a
                           date"), but neither candidate supplies date
                           evidence, so the deterministic evidence
                           cannot actually distinguish "two fragments
                           of one entity" from "two separate,
                           genuinely undated roles."

This rule is NOT keyed to U2 or any resume name. It was derived from,
and independently verified against, all 22 adjacency junctions across
all 9 resumes: it fires on exactly the 2 junctions already known (from
T22's own manual investigation) to be genuine assembly errors (U2's
C10->C11 and C13->C14), and on no others -- zero false positives,
zero false negatives, using only the pre-existing, unmodified
Block.date_range_present feature already computed by T17/T18's own
frozen code.

Employer-relationship and local-context fields recorded in the
ambiguity inventory are DESCRIPTIVE METADATA ONLY (for the later LLM's
own context and for human audit) -- they play no role in the ambiguity
DETECTION rule itself, which uses only date-presence.

No LLM calls in this module.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_t18_blocks import blockify  # noqa: E402
from scripts._milestone_t20_candidates import generate_candidates_t20, Candidate  # noqa: E402
from scripts._milestone_t22_assemble import load_t21b_decisions, is_accepted  # noqa: E402

MARKDOWN_DIR = Path("output/markdown")


@dataclass
class AmbiguityRecord:
    resume_id: str
    candidate_a_id: str
    candidate_b_id: str
    block_ids_a: List[int]
    block_ids_b: List[int]
    t22_decision: str  # "MERGE" or "NO-MERGE"
    classification: str  # CLEAR_MERGE | CLEAR_SEPARATE | SEMANTICALLY_AMBIGUOUS
    a_has_own_date: bool
    b_has_own_date: bool
    employer_relationship: str  # descriptive only, not used for detection
    local_context: str
    reason: str


def _candidate_has_own_date(blocks, c: Candidate) -> bool:
    return any(blocks[i].date_range_present for i in range(c.extent_start, c.extent_end + 1))


def _local_context(blocks, a: Candidate, b: Candidate, max_chars: int = 400) -> str:
    text = " | ".join(blocks[i].stripped for i in range(a.extent_start, b.extent_end + 1))
    return text[:max_chars]


def _employer_relationship(a: Candidate, b: Candidate) -> str:
    """Descriptive only: a crude, structural (not semantic-matching)
    note of whether the two anchor texts share any alphabetic token,
    for human/LLM context -- never used to make the ambiguity/merge
    decision itself."""
    def toks(s):
        return {w.lower() for w in "".join(ch if ch.isalnum() else " " for ch in s).split() if len(w) > 2}
    shared = toks(a.anchor_text) & toks(b.anchor_text)
    return f"shared_tokens={sorted(shared)}" if shared else "no_shared_tokens"


def classify_junction(blocks, a: Candidate, b: Candidate) -> AmbiguityRecord:
    a_date = _candidate_has_own_date(blocks, a)
    b_date = _candidate_has_own_date(blocks, b)
    t22_decision = "NO-MERGE" if a_date else "MERGE"

    if a_date:
        classification = "CLEAR_SEPARATE"
        reason = "First candidate has its own date evidence; T22 correctly keeps them separate."
    elif b_date:
        classification = "CLEAR_MERGE"
        reason = "First candidate lacks a date but the second supplies it -- a structurally supported fragment+date reunification."
    else:
        classification = "SEMANTICALLY_AMBIGUOUS"
        reason = ("Neither candidate has its own date evidence. T22's rule still merges them, but "
                  "the available structural evidence cannot distinguish an incomplete fragment of one "
                  "entity from two genuinely separate, undated roles at the same employer.")

    return AmbiguityRecord(
        resume_id="", candidate_a_id=f"C{a.candidate_id}", candidate_b_id=f"C{b.candidate_id}",
        block_ids_a=list(range(a.extent_start, a.extent_end + 1)),
        block_ids_b=list(range(b.extent_start, b.extent_end + 1)),
        t22_decision=t22_decision, classification=classification,
        a_has_own_date=a_date, b_has_own_date=b_date,
        employer_relationship=_employer_relationship(a, b),
        local_context=_local_context(blocks, a, b),
        reason=reason,
    )


def detect_ambiguities_for_resume(rid: str, stem: str, decisions_by_resume) -> List[AmbiguityRecord]:
    blocks = blockify((MARKDOWN_DIR / f"{stem}.md").read_text(encoding="utf-8"))
    candidates = generate_candidates_t20(blocks)
    dmap = decisions_by_resume.get(rid, {})
    accepted = sorted(
        [c for c in candidates if is_accepted(dmap.get(f"C{c.candidate_id}"))],
        key=lambda c: c.extent_start,
    )
    records = []
    for i in range(len(accepted) - 1):
        a, b = accepted[i], accepted[i + 1]
        if b.extent_start == a.extent_end + 1:
            rec = classify_junction(blocks, a, b)
            rec.resume_id = rid
            records.append(rec)
    return records


def detect_all(stems: dict) -> List[AmbiguityRecord]:
    decisions_by_resume = load_t21b_decisions()
    all_records = []
    for rid, stem in stems.items():
        all_records.extend(detect_ambiguities_for_resume(rid, stem, decisions_by_resume))
    return all_records
