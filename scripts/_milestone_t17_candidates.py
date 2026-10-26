"""
scripts/_milestone_t17_candidates.py

Milestone T17/Phase 2D -- Candidate-generation pass + Safety-flagging
pass, per the frozen rules in
docs/T17_PHASE2D_CANDIDATE_GENERATION_PRE_REGISTRATION.md.

Research-only. Operates on the Block sequence from
scripts/_milestone_t17_blocks.py. A candidate is NOT a confirmed
entity -- no type classification (Job/Project/Education/Other), no
field extraction, no semantic merge/split. Does not modify any
production file.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_t17_blocks import Block, containing_section_labels  # noqa: E402
from validators.misrouting_detector import _fuzzy_contains  # noqa: E402

_ANCHOR_TYPES = ("heading_like", "table_row", "date_bearing")
_CANDIDATE_CAP = 200  # generic safety bound, not corpus-tuned -- see pre-registration


@dataclass
class Candidate:
    """A structural hypothesis, not a confirmed Work Experience entity.
    Anchor + probable extent, per the frozen pre-registration -- never
    a hard character crop."""

    candidate_id: int
    anchor_block_ids: List[int]
    extent_start: int  # block_id
    extent_end: int  # block_id, inclusive
    containing_section_label: Optional[str]
    n_blocks_in_extent: int
    contains_bullet_run: bool
    anchor_text: str  # first anchor block's stripped text
    flags: dict = field(default_factory=dict)


def generate_candidates(blocks: List[Block]) -> List[Candidate]:
    """Candidate-generation pass. Deliberately over-inclusive, per the
    frozen pre-registration -- does not classify, extract fields, or
    semantically merge/split. Whole-document scope (not restricted to
    any one section), per the mandatory 001 analysis requirement."""
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
            while (
                j < n
                and blocks[j].block_type in _ANCHOR_TYPES
                and blocks[j].blank_lines_before <= 1
            ):
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

        # Recompute this group's own block range (its own anchor run,
        # not the extent) to populate anchor_block_ids precisely.
        group_end = start
        while (
            group_end + 1 <= extent_end
            and group_end + 1 < n
            and blocks[group_end + 1].block_type in _ANCHOR_TYPES
            and blocks[group_end + 1].blank_lines_before <= 1
        ):
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


def _iqr_bounds(values: List[int]) -> tuple:
    if len(values) < 4:
        return (float("-inf"), float("inf"))  # too few points for a meaningful IQR
    s = sorted(values)
    n = len(s)
    q1 = s[n // 4]
    q3 = s[(3 * n) // 4]
    iqr = q3 - q1
    return (q1 - 1.5 * iqr, q3 + 1.5 * iqr)


def _apply_safety_flags(
    candidates: List[Candidate], blocks: List[Block], section_labels: List[Optional[str]]
) -> None:
    lengths = [c.n_blocks_in_extent for c in candidates]
    lo, hi = _iqr_bounds(lengths)
    last_block_id = len(blocks) - 1

    for cand in candidates:
        extent_labels = set(section_labels[cand.extent_start : cand.extent_end + 1])
        crosses = len(extent_labels - {cand.containing_section_label}) > 0

        length_outlier = not (lo <= cand.n_blocks_in_extent <= hi)

        page_break_adjacent = any(
            blocks[i].page_break_adjacent for i in range(cand.extent_start, cand.extent_end + 1)
        )

        low_confidence_terminus = cand.extent_end == last_block_id

        outside_vocab = cand.containing_section_label != "experience"

        misrouting_overlap = False
        for other in candidates:
            if other.candidate_id == cand.candidate_id:
                continue
            if other.containing_section_label == cand.containing_section_label:
                continue
            if _fuzzy_contains(cand.anchor_text, other.anchor_text):
                misrouting_overlap = True
                break

        cand.flags = {
            "crosses_section_boundary": crosses,
            "length_outlier": length_outlier,
            "page_break_adjacent": page_break_adjacent,
            "low_confidence_terminus": low_confidence_terminus,
            "outside_known_experience_vocabulary": outside_vocab,
            "misrouting_detector_overlap": misrouting_overlap,
        }
