"""
scripts/_milestone_t17_blocks.py

Milestone T17/Phase 2D -- Blocking pass + Feature-tagging pass, per the
frozen rules in docs/T17_PHASE2D_CANDIDATE_GENERATION_PRE_REGISTRATION.md.

Research-only. Operates on the existing, real, already-in-production
normalized content (output/markdown/<stem>.md) -- no parallel
representation is invented. Imports WorkExperienceLocator's own
regex constants and _looks_like_heading directly, unmodified, to
guarantee zero drift from the production utility. Does not modify
WorkExperienceLocator or any production file.

No LLM calls. No trees/graphs/coordinate IR/embeddings/learned
classifiers.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from services.work_experience_locator import (  # noqa: E402
    WorkExperienceLocator,
    _HEADING_PATTERN,
    _STANDALONE_BOLD_PATTERN,
    _LEADING_LIST_MARKER_PATTERN,
    _TABLE_ROW_PATTERN,
    _DATE_RANGE_PATTERN,
)

_SEPARATOR_PATTERN = re.compile(r"^```(?:markdown)?\s*$", re.IGNORECASE)
_PAGE_OPEN_PATTERN = re.compile(r"^```markdown\s*$", re.IGNORECASE)
_NUMBERED_LIST_PATTERN = re.compile(r"^\d+[.)]\s+")


@dataclass
class Block:
    """One atomic block -- one non-blank line of the existing
    normalized markdown content, plus cheap, deterministic, per-block
    metadata. See the frozen pre-registration for the exact rule
    behind every field."""

    block_id: int
    text: str
    stripped: str
    line_number: int
    page_number: int
    blank_lines_before: int
    indent_depth: int
    indent_delta: int
    block_type: str  # separator | table_row | heading_like | date_bearing | bullet | paragraph
    heading_likeness: bool
    date_range_present: bool
    bullet_run_after: bool = False  # filled in a second pass (needs lookahead)
    page_break_adjacent: bool = False  # filled in a second pass (needs lookaround)
    section_distance: Optional[int] = None
    repeated_pattern_id: int = -1  # filled in a third pass


def _is_heading_like(stripped: str) -> bool:
    if _HEADING_PATTERN.match(stripped):
        return True
    candidate = _LEADING_LIST_MARKER_PATTERN.sub("", stripped)
    if _STANDALONE_BOLD_PATTERN.match(candidate):
        return True
    return WorkExperienceLocator._looks_like_heading(stripped)  # noqa: SLF001


def _classify_block_type(stripped: str) -> str:
    if _SEPARATOR_PATTERN.match(stripped):
        return "separator"
    if _TABLE_ROW_PATTERN.match(stripped):
        return "table_row"
    if _is_heading_like(stripped):
        return "heading_like"
    if _DATE_RANGE_PATTERN.search(stripped):
        return "date_bearing"
    if stripped and stripped[0] in WorkExperienceLocator._BULLET_CHARS:  # noqa: SLF001
        return "bullet"
    if _NUMBERED_LIST_PATTERN.match(stripped):
        return "bullet"
    return "paragraph"


def blockify(markdown_text: str) -> List[Block]:
    """Derive the flat, ordered Atomic Block sequence from `markdown_text`.

    Pure function of the existing normalized content -- no I/O, no
    corpus-specific branching.
    """
    lines = markdown_text.split("\n")
    blocks: List[Block] = []
    page_number = 1
    blank_run = 0
    prev_indent = 0

    for line_number, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped:
            blank_run += 1
            continue

        if _PAGE_OPEN_PATTERN.match(stripped) and blocks:
            # A page-open fence increments the page counter for itself
            # and everything after it -- but not for content already
            # assigned to the previous page.
            page_number += 1

        indent_depth = (len(raw_line) - len(raw_line.lstrip())) // 2
        block_type = _classify_block_type(stripped)
        heading_likeness = _is_heading_like(stripped)
        date_range_present = bool(_DATE_RANGE_PATTERN.search(stripped))

        block = Block(
            block_id=len(blocks),
            text=raw_line,
            stripped=stripped,
            line_number=line_number,
            page_number=page_number,
            blank_lines_before=blank_run,
            indent_depth=indent_depth,
            indent_delta=indent_depth - prev_indent,
            block_type=block_type,
            heading_likeness=heading_likeness,
            date_range_present=date_range_present,
        )
        blocks.append(block)
        prev_indent = indent_depth
        blank_run = 0

    _fill_lookahead_lookaround_features(blocks)
    _fill_section_distance(blocks)
    _fill_repeated_pattern_id(blocks)
    return blocks


def _fill_lookahead_lookaround_features(blocks: List[Block]) -> None:
    for i, b in enumerate(blocks):
        if i + 1 < len(blocks) and blocks[i + 1].block_type == "bullet":
            b.bullet_run_after = True
        is_sep = b.block_type == "separator"
        prev_sep = i > 0 and blocks[i - 1].block_type == "separator"
        next_sep = i + 1 < len(blocks) and blocks[i + 1].block_type == "separator"
        b.page_break_adjacent = is_sep or prev_sep or next_sep


def _fill_section_distance(blocks: List[Block]) -> None:
    distance_since_heading: Optional[int] = None
    for b in blocks:
        if b.block_type in ("heading_like", "table_row"):
            distance_since_heading = 0
        elif distance_since_heading is not None:
            distance_since_heading += 1
        b.section_distance = distance_since_heading


def _fill_repeated_pattern_id(blocks: List[Block]) -> None:
    seen: dict = {}
    for b in blocks:
        key = (b.block_type, b.heading_likeness, b.date_range_present, b.indent_depth)
        if key not in seen:
            seen[key] = len(seen)
        b.repeated_pattern_id = seen[key]


def section_label_for_heading_text(stripped_heading_text: str) -> str:
    """Per §2 of the frozen pre-registration: label a heading's own
    text using the existing, already-frozen locator vocabulary,
    reused unmodified. Diagnostic/safety-flag use only -- never used
    to exclude a candidate."""
    from services.work_experience_locator import (  # noqa: E402
        _EXPERIENCE_SECTION_NAMES,
        _NON_EXPERIENCE_SECTION_KEYWORDS,
    )

    text = _HEADING_PATTERN.sub(lambda m: m.group(2), stripped_heading_text).strip()
    text = text.rstrip(":").strip().lower()
    if text in _EXPERIENCE_SECTION_NAMES:
        return "experience"
    for kw in _NON_EXPERIENCE_SECTION_KEYWORDS:
        if kw in text:
            return f"non_experience:{kw}"
    return f"unlabeled:{text[:40]}"


def containing_section_labels(blocks: List[Block]) -> List[Optional[str]]:
    """Per block, the label (per §2) of the most recent preceding
    heading_like block's own text, or None if none precedes it yet."""
    labels: List[Optional[str]] = []
    current: Optional[str] = None
    for b in blocks:
        if b.block_type == "heading_like":
            current = section_label_for_heading_text(b.stripped)
        labels.append(current)
    return labels
