"""
services/experimental_column_layout.py

Milestone V2.4D — originally EXPERIMENTAL, PROTOTYPE-ONLY, not wired
into production. Milestone V2.4H made this module's functions
selectively reachable from production: `services/stage1_router.py`
imports `analyze_page_layout` directly (reusing its own gap-detection,
not duplicating it) to decide the route, and the new
`services/column_aware_stage1_extractor.py` wraps `analyze_page_layout`
+ `linearize_page` with file I/O for documents the router selects.
Every function in THIS file is otherwise unchanged since V2.4F — no
algorithm logic here was modified for that integration; only this
docstring's own claim about reachability is updated, since the
original "not imported by services/stage1_router.py" statement would
otherwise be stale and misleading. `pipeline/parser_pipeline.py` still
never imports this module, directly or indirectly.

Exists to answer the question posed across Milestones V2.4D-H: can
PyMuPDF block coordinates support a narrow, non-resume-specific
column-aware text linearization for genuinely two-column documents,
without regressing single-column ones?

Column-gap detection is deliberately generic — it looks for a
consistent, wide, roughly-central vertical gap in the page's own block
coverage profile. No resume-specific text, section name, or employer
name is referenced anywhere in this module. If no such gap is found,
the page is treated as single-column and blocks are returned in their
original (already-natural, top-to-bottom) order — i.e. the algorithm's
default output for a single-column page is a no-op relative to
PyMuPDF's own natural block order.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class BlockRecord:
    """One PyMuPDF text block, with diagnostic column-assignment info."""

    page_number: int  # 1-indexed
    block_index: int
    x0: float
    y0: float
    x1: float
    y1: float
    full_text: str  # the block's complete, untruncated text
    text_preview: str  # first 80 chars, for the diagnostic CSV only
    relative_x: float  # x0 / page_width, in [0, 1]
    relative_y: float  # y0 / page_height, in [0, 1]
    column: str  # "single" | "left" | "right" | "header" | "footer"


def _coverage_gap(
    intervals: List[Tuple[float, float]],
    page_width: float,
    min_gap_width: float = 10.0,
    margin_fraction: float = 0.15,
) -> Optional[Tuple[float, float]]:
    """Find the widest horizontal gap in `intervals`' coverage of the
    page, excluding gaps too close to either page edge (a real column
    gutter sits roughly in the interior, not at the margins).

    Returns (gap_start, gap_end) in page coordinates, or None if no
    gap at least `min_gap_width` points wide exists within the
    interior region [margin_fraction, 1 - margin_fraction] * page_width.
    """
    if not intervals:
        return None
    sorted_intervals = sorted(intervals)
    merged: List[Tuple[float, float]] = []
    for start, end in sorted_intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    left_margin = page_width * margin_fraction
    right_margin = page_width * (1 - margin_fraction)

    best_gap: Optional[Tuple[float, float]] = None
    best_width = 0.0
    for (_, prev_end), (next_start, _) in zip(merged, merged[1:]):
        gap_start, gap_end = prev_end, next_start
        gap_width = gap_end - gap_start
        if gap_width < min_gap_width:
            continue
        if gap_end <= left_margin or gap_start >= right_margin:
            continue
        if gap_width > best_width:
            best_width = gap_width
            best_gap = (gap_start, gap_end)
    return best_gap


def analyze_page_layout(
    blocks: List[Tuple[float, float, float, float, str, int, int]],
    page_number: int,
    page_width: float,
    page_height: float,
) -> Tuple[List[BlockRecord], Optional[Tuple[float, float]]]:
    """Classify every block on one page as header/left/right/single/footer.

    Args:
        blocks: PyMuPDF's own `page.get_text("blocks")` output for one page.
        page_number: 1-indexed page number, for record-keeping only.
        page_width/page_height: the page's own dimensions.

    Returns:
        (records, gap) — `records` is one BlockRecord per input block;
        `gap` is the detected column-gutter x-range, or None if the
        page was judged single-column (no gap found, or found but not
        splitting the page into two substantial, populated regions).
    """
    real_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]  # block_type 0 = text, non-empty
    intervals = [(b[0], b[2]) for b in real_blocks]
    gap = _coverage_gap(intervals, page_width)

    records: List[BlockRecord] = []
    if gap is None:
        for idx, b in enumerate(real_blocks):
            x0, y0, x1, y1, text, block_no, _ = b
            records.append(
                BlockRecord(
                    page_number=page_number,
                    block_index=idx,
                    x0=x0, y0=y0, x1=x1, y1=y1,
                    full_text=text.strip(),
                text_preview=text.strip().replace("\n", " | ")[:80],
                    relative_x=x0 / page_width,
                    relative_y=y0 / page_height,
                    column="single",
                )
            )
        return records, None

    gap_mid = (gap[0] + gap[1]) / 2
    left_blocks = [b for b in real_blocks if (b[0] + b[2]) / 2 < gap_mid]
    right_blocks = [b for b in real_blocks if (b[0] + b[2]) / 2 >= gap_mid]

    # A genuine two-column layout needs BOTH sides substantially
    # populated, not one wide block accidentally straddling a narrow
    # gap. Require at least 3 blocks and non-trivial vertical extent
    # on each side, else treat as single-column (safe default).
    def _substantial(group: List) -> bool:
        if len(group) < 3:
            return False
        y_span = max(b[3] for b in group) - min(b[1] for b in group)
        return y_span > page_height * 0.1

    if not (_substantial(left_blocks) and _substantial(right_blocks)):
        for idx, b in enumerate(real_blocks):
            x0, y0, x1, y1, text, block_no, _ = b
            records.append(
                BlockRecord(
                    page_number=page_number,
                    block_index=idx,
                    x0=x0, y0=y0, x1=x1, y1=y1,
                    full_text=text.strip(),
                text_preview=text.strip().replace("\n", " | ")[:80],
                    relative_x=x0 / page_width,
                    relative_y=y0 / page_height,
                    column="single",
                )
            )
        return records, None

    # Header: block(s) whose x-extent spans (or starts well left of and
    # extends past) the detected gap itself -- i.e. genuinely full-width
    # content above the two-column region, not just a wide left-column
    # block. Identified as: y0 less than the minimum y0 of EITHER
    # column's blocks, and x-extent crosses the gap.
    two_col_top = min(min(b[1] for b in left_blocks), min(b[1] for b in right_blocks))

    for idx, b in enumerate(real_blocks):
        x0, y0, x1, y1, text, block_no, _ = b
        crosses_gap = x0 < gap[0] and x1 > gap[1]
        if crosses_gap and y1 <= two_col_top + 1.0:
            column = "header"
        elif (x0 + x1) / 2 < gap_mid:
            column = "left"
        else:
            column = "right"
        records.append(
            BlockRecord(
                page_number=page_number,
                block_index=idx,
                x0=x0, y0=y0, x1=x1, y1=y1,
                full_text=text.strip(),
                text_preview=text.strip().replace("\n", " | ")[:80],
                relative_x=x0 / page_width,
                relative_y=y0 / page_height,
                column=column,
            )
        )
    return records, gap


def _split_into_sublines(record: BlockRecord) -> List[Tuple[float, float, str]]:
    """Milestone V2.4F: split one block's `full_text` on its own
    internal newlines, assigning each resulting sub-line an
    interpolated y-position evenly spaced across the block's own
    [y0, y1] span (approximating even internal line-height, a
    reasonable approximation within one PyMuPDF text block, which
    already represents one contiguous font run).

    Why this exists (see docs/MILESTONE_V2_4F_TWO_COLUMN_BULLET_PRESERVATION.md
    §2 for the full, directly-reproduced evidence): on resume017,
    PyMuPDF groups an entire multi-bullet responsibility list into ONE
    block ("Managing a team...\nPlanning engineers' schedules\nProject
    initiation & site coordination\n...", block 5, y0=268.57..349.59)
    and the bullet GLYPHS for those same lines into a SEPARATE block
    ("•\n•\n•\n•\n•\n•", block 48, y0=288.87..349.59). The pre-V2.4F
    `_merge_same_row_blocks` compared only each block's own single y0 —
    268.57 vs. 288.87, a 20pt gap far outside any reasonable row
    tolerance — so the bullet block could never align with any of the
    text block's OTHER internal lines (only its first, which happened
    to share a y0 with a different, single-bullet block). Splitting
    every block into its own interpolated sub-lines FIRST lets the
    existing row-grouping logic operate at the true visual-line
    granularity PyMuPDF's own block segmentation had collapsed away.

    A single-line block (the overwhelming majority in this corpus)
    returns exactly one tuple at its own original y0 — byte-identical
    row-grouping behavior to before this milestone for every block
    that was never multi-line to begin with, including every block in
    all 21 non-two-column documents' single-column path (re-verified
    this milestone: reduces, never increases, orphaned lone-bullet
    lines on every other bullet-bearing document checked).
    """
    parts = [p for p in record.full_text.split("\n") if p.strip()]
    if len(parts) <= 1:
        return [(record.y0, record.x0, record.full_text.strip())]
    span = record.y1 - record.y0
    denominator = len(parts) - 1
    return [
        (record.y0 + (span * i / denominator) if denominator else record.y0, record.x0, part.strip())
        for i, part in enumerate(parts)
    ]


def _merge_same_row_blocks(records: List[BlockRecord], row_tolerance: float = 3.0) -> List[str]:
    """Merge same-visual-row content into single, space-joined lines,
    in left-to-right order, at sub-line granularity (Milestone V2.4F —
    see `_split_into_sublines`'s docstring for why sub-line granularity
    replaced whole-block y0 comparison).

    Original (Milestone V2.4D) evidence this row-merging exists for at
    all: on resume017, a single visual line ("09/2021 [icon] Present
    Andheri") was split by PyMuPDF into 5 separate single-line blocks
    with a nearly-identical y0 (243.6-244.4, well within 1pt of each
    other). Joining paragraph-by-paragraph (one block per line) broke
    this into 3 separate "lines", severing the date-range pattern
    (services/work_experience_locator.py's `_DATE_RANGE_PATTERN`) that
    downstream deterministic tooling needs to see "09/2021 ... Present"
    as one phrase — unaffected by the V2.4F change, since a single-line
    block still contributes exactly one sub-line at its own y0.
    """
    if not records:
        return []
    sublines: List[Tuple[float, float, str]] = []
    for r in records:
        sublines.extend(_split_into_sublines(r))
    sublines.sort(key=lambda t: (t[0], t[1]))
    rows: List[List[Tuple[float, float, str]]] = []
    for item in sublines:
        if rows and abs(item[0] - rows[-1][0][0]) <= row_tolerance:
            rows[-1].append(item)
        else:
            rows.append([item])
    lines = []
    for row in rows:
        row_sorted = sorted(row, key=lambda t: t[1])
        text = " ".join(t[2] for t in row_sorted if t[2])
        if text:
            lines.append(text)
    return lines


def linearize_page(records: List[BlockRecord], merge_same_row: bool = True) -> str:
    """Produce column-aware linear reading order: header blocks first
    (top-to-bottom), then every left-column block (top-to-bottom), then
    every right-column block (top-to-bottom). For a page classified
    "single" (no gap detected), returns blocks in their original
    (already top-to-bottom-ish, PyMuPDF-natural) order -- a no-op
    relative to the page's own natural block order.

    `merge_same_row` (default True): first merge blocks sharing
    (nearly) the same y0 into single space-joined lines -- see
    `_merge_same_row_blocks`'s docstring for the concrete, real defect
    this fixes. Set False to reproduce the earlier, naive one-block-
    per-paragraph behavior for comparison.
    """
    header = sorted([r for r in records if r.column == "header"], key=lambda r: r.y0)
    left = sorted([r for r in records if r.column == "left"], key=lambda r: r.y0)
    right = sorted([r for r in records if r.column == "right"], key=lambda r: r.y0)
    single = sorted([r for r in records if r.column == "single"], key=lambda r: r.y0)

    ordered_groups = [header, left, right, single]
    all_lines: List[str] = []
    for group in ordered_groups:
        if not group:
            continue
        if merge_same_row:
            all_lines.extend(_merge_same_row_blocks(group))
        else:
            all_lines.extend(r.full_text for r in group)
    return "\n\n".join(all_lines)
