"""
scripts/_milestone_t18_candidates.py

Milestone T18/Phase 2E -- Mechanism B fix (one-blank-line merge), per
docs/T18_PHASE2E_MECHANISM_TRACE_AND_PLAN.md.

Mechanism A (section-label cascade) was implemented and initially
combined with Mechanism B here, but was REVERTED after the mandatory
001 gate revealed a corpus-wide label-collapse defect -- see
scripts/_milestone_t18_blocks.py's module docstring for the full
account. Per explicit user direction, this module now uses T17's own,
unmodified containing_section_labels() for the section-label input.
T18's shipped result reflects Mechanism B only.

Reuses T17's own Candidate dataclass, _ANCHOR_TYPES, _iqr_bounds,
_apply_safety_flags, and containing_section_labels completely
unchanged (imported, not copied). Only the anchor-group-extension rule
inside generate_candidates() is replaced. T17's own
scripts/_milestone_t17_candidates.py is not modified, so the frozen
T17 baseline remains re-runnable byte-for-byte unchanged.

No LLM calls. No production file modified.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_t17_blocks import Block, containing_section_labels  # noqa: E402
from scripts._milestone_t17_candidates import (  # noqa: E402
    Candidate,
    _ANCHOR_TYPES,
    _CANDIDATE_CAP,
    _apply_safety_flags,
)


def generate_candidates_v2(blocks: List[Block]) -> List[Candidate]:
    """Candidate-generation pass with Mechanism B's fix: once an anchor
    group has accumulated a block whose `date_range_present` feature is
    True, a further anchor block is only added to the SAME group if
    blank_lines_before == 0 (no blank line at all); the original T17
    tolerance of blank_lines_before <= 1 continues to apply only to
    anchor blocks reached BEFORE the group's own first
    date_range_present block. See
    docs/T18_PHASE2E_MECHANISM_TRACE_AND_PLAN.md, Mechanism B, for the
    full evidence and tuning-subset verification behind this rule.

    NOTE: the "date signal" here is the block's own `date_range_present`
    feature (computed by regex on every block regardless of type), NOT
    `block_type == "date_bearing"`. T17's own block_type precedence
    (separator > table_row > heading_like > date_bearing > bullet >
    paragraph) means a standalone bold or heading-shaped line that also
    contains a date range is classified `heading_like`, never
    `date_bearing` -- which is the common, real-corpus case for a job's
    own date line (e.g. a bold `**Jan 2023 - Feb 2026**` line). An
    earlier draft of this fix checked `block_type == "date_bearing"`
    directly and, as a result, never fired on any of these
    heading-shaped date lines; this was caught during tuning-subset
    verification (re-run against the corrected condition below) before
    any held-out/gate resume was used to validate the fix, per the
    frozen tuning discipline.

    Everything else (extent computation, candidate cap, structural
    features, safety-flag application, and now also the section-label
    input itself, since Mechanism A was reverted) is unchanged from
    T17's own generate_candidates() -- only the anchor-group-extension
    condition differs.
    """
    if not blocks:
        return []

    section_labels = containing_section_labels(blocks)

    anchor_group_starts: List[int] = []
    i = 0
    n = len(blocks)
    while i < n:
        if blocks[i].block_type in _ANCHOR_TYPES:
            group_start = i
            j = i + 1
            seen_date = blocks[i].date_range_present
            while j < n and blocks[j].block_type in _ANCHOR_TYPES:
                tolerance = 0 if seen_date else 1
                if blocks[j].blank_lines_before > tolerance:
                    break
                if blocks[j].date_range_present:
                    seen_date = True
                j += 1
            anchor_group_starts.append(group_start)
            i = j
        else:
            i += 1

    if not anchor_group_starts:
        return []

    candidates: List[Candidate] = []
    for idx, start in enumerate(anchor_group_starts):
        if idx + 1 < len(anchor_group_starts):
            extent_end = anchor_group_starts[idx + 1] - 1
        else:
            extent_end = n - 1

        group_end = start
        seen_date = blocks[start].date_range_present
        while group_end + 1 <= extent_end and group_end + 1 < n:
            nxt = blocks[group_end + 1]
            if nxt.block_type not in _ANCHOR_TYPES:
                break
            tolerance = 0 if seen_date else 1
            if nxt.blank_lines_before > tolerance:
                break
            if nxt.date_range_present:
                seen_date = True
            group_end += 1

        extent_blocks = blocks[start : extent_end + 1]
        cand = Candidate(
            candidate_id=len(candidates),
            anchor_block_ids=list(range(start, group_end + 1)),
            extent_start=start,
            extent_end=extent_end,
            containing_section_label=section_labels[start],
            n_blocks_in_extent=len(extent_blocks),
            contains_bullet_run=any(b.block_type == "bullet" for b in extent_blocks),
            anchor_text=blocks[start].stripped,
        )
        candidates.append(cand)
        if len(candidates) >= _CANDIDATE_CAP:
            break

    _apply_safety_flags(candidates, blocks, section_labels)
    return candidates
