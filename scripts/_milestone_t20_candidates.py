"""
scripts/_milestone_t20_candidates.py

Milestone T20/Phase 2G -- ONE narrow deterministic refinement targeting
the section-heading/first-job merge pattern T19 confirmed in 5/5
unseen resumes.

ROOT CAUSE (confirmed by direct anchor_block_ids inspection before any
code was written -- see docs/T20_SECTION_HEADING_FIRST_JOB_REFINEMENT.md,
"Exact mechanism/root cause"): in 4 of 5 T19 unseen resumes (U2, U3,
U4, U5), the block whose own text is a recognized Work-Experience
section title (e.g. "## Work Experience") becomes the START of its own
anchor group, and T18's existing anchor-group-extension loop (Mechanism
B, unchanged here) then extends that SAME group into the immediately
following first-job heading, because nothing in the extension
condition distinguishes a section-title heading from a job heading.

This is a genuinely different code path from a second, DISTINCT
manifestation found in the 5th resume (U1), where the section heading
is instead swallowed as a *non-first* element of an anchor group
STARTED by an earlier, unrelated anchor block (a table_row). Per the
frozen T20 hypothesis's own literal wording ("a section-label heading
should NOT be allowed to become part of the first Work Experience
candidate's anchor/extension chain when a valid first-job anchor
immediately follows it") and per T20's explicit instruction not to
tune more than one mechanism, ONLY the first manifestation (the
section heading's OWN group extending forward) is targeted here. The
second manifestation (U1) is deliberately NOT fixed -- see the T20
report's "New failure mechanism discovered" section; it is recorded as
a future hypothesis only.

THE ONE CHANGE: when a NEW anchor group is about to be opened at
`blocks[i]`, and `blocks[i]` is a `heading_like` block whose own text
(via T17's own unmodified `section_label_for_heading_text`) is
recognized as `"experience"`, the group is forced to be a SINGLETON --
the extension while-loop is skipped entirely for that group. No other
anchor-group-formation behavior changes: T18's own date_range_present-
based tolerance tightening (Mechanism B) is completely unmodified and
still governs every other anchor group, including the *next* group
(the real first job's own group, which starts immediately after the
now-singleton section-heading group and extends via the exact same,
unmodified Mechanism B rules as before).

Reuses T17's own Candidate dataclass, _ANCHOR_TYPES, _CANDIDATE_CAP,
_apply_safety_flags, containing_section_labels, and
section_label_for_heading_text completely unchanged (imported, not
copied). Reuses T18's own containing_section_labels usage pattern
(Mechanism A remains reverted, exactly as T18 shipped it). Only the
anchor-group-formation loop gains the one new singleton-guard
described above; the extent-recomputation loop's own group_end
recomputation is updated identically (same guard, same reasoning) so
the extent boundary stays consistent with the anchor-group boundary --
mirroring the precedent T18 itself set when Mechanism B's tolerance
change had to be applied consistently in both loops.

No LLM calls. No production file modified. T17's and T18's own files
are not modified -- this is a new, additive module.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_t17_blocks import (  # noqa: E402
    Block,
    containing_section_labels,
    section_label_for_heading_text,
)
from scripts._milestone_t17_candidates import (  # noqa: E402
    Candidate,
    _ANCHOR_TYPES,
    _CANDIDATE_CAP,
    _apply_safety_flags,
)


def _is_section_title_heading(block: Block) -> bool:
    """True iff `block` is itself (by its own text, not a cascaded
    running label) a recognized Work-Experience section title. Reuses
    T17's own section_label_for_heading_text unchanged -- no new
    classification is introduced."""
    if block.block_type != "heading_like":
        return False
    return section_label_for_heading_text(block.stripped) == "experience"


def generate_candidates_t20(blocks: List[Block]) -> List[Candidate]:
    """T18's generate_candidates_v2, with exactly one addition: a
    section-title heading (per _is_section_title_heading) never
    extends its own anchor group past itself. Everything else --
    Mechanism B's date_range_present-based tolerance tightening, the
    section-label input (T17's own containing_section_labels,
    Mechanism A still reverted), extent computation, candidate cap,
    structural features, safety-flag application -- is byte-identical
    to T18's own generate_candidates_v2.
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
            if _is_section_title_heading(blocks[i]):
                # THE ONE CHANGE (formation loop): a section-title
                # heading is always a singleton anchor group -- do not
                # enter the extension loop at all.
                anchor_group_starts.append(group_start)
                i = group_start + 1
                continue
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

        if _is_section_title_heading(blocks[start]):
            # THE ONE CHANGE (extent-recomputation loop): mirror the
            # same singleton guard so the extent boundary matches the
            # anchor-group boundary computed above.
            group_end = start
        else:
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
