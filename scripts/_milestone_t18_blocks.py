"""
scripts/_milestone_t18_blocks.py

Milestone T18/Phase 2E -- Mechanism A (section-label cascade),
per docs/T18_PHASE2E_MECHANISM_TRACE_AND_PLAN.md.

**MECHANISM A FIX REVERTED -- NOT USED IN THE LIVE T18 EVALUATION PATH.**
`containing_section_labels_v2()` below was implemented and passed its
own tuning-subset verification, but the MANDATORY 001 gate check (a
held-out resume) revealed a corpus-wide defect not caught during
tuning: any resume whose first heading is a document-title heading
shallower than its real section headings (e.g. `# <Name>` at level 1
followed by `## <Section>` headings at level 2) permanently locks
`established_level` at the title heading's own level, so no real
section heading ever again qualifies to update the running label --
collapsing the label for the entire rest of the document to the
title heading's own text. Confirmed corpus-wide (all 22 resumes showed
severe label-diversity collapse relative to T17, not just 001), and
confirmed to be the reason the previously-observed
`crosses_section_boundary`/bleed-dependent-coverage improvement was an
artifact of destroyed label diversity, not a genuine safety fix.

Per explicit user direction after this finding was reported, Mechanism
A's fix is REVERTED: the live T18 evaluation path
(`scripts/_milestone_t18_candidates.py`) uses T17's own, unmodified
`containing_section_labels()` directly. `containing_section_labels_v2`
and `heading_level` are retained below, unused, purely as disclosed
evidence of what was attempted and why it was abandoned -- see the
T18 final report for the full account. T18's shipped result reflects
Mechanism B only.

Reuses T17's own blockify() completely unchanged (imported, not
copied) -- the Blocking/Feature-tagging passes themselves are not
touched. T17's own scripts/_milestone_t17_blocks.py is not modified,
so the frozen T17 baseline remains re-runnable byte-for-byte unchanged.

No LLM calls. No production file modified.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_t17_blocks import (  # noqa: E402,F401
    Block,
    blockify,
    section_label_for_heading_text,
    _HEADING_PATTERN,
)


def heading_level(stripped_text: str) -> Optional[int]:
    """UNUSED in the live T18 path (Mechanism A reverted -- see module
    docstring). Retained for disclosure only.

    Markdown '#' heading level (1-6), from the same, already-
    existing _HEADING_PATTERN T17 already uses -- not a new signal.
    Returns None for a heading that matched only via
    _STANDALONE_BOLD_PATTERN or _looks_like_heading (no inherent
    '#'-level), matching Mechanism A's own documented design."""
    match = _HEADING_PATTERN.match(stripped_text)
    if not match:
        return None
    return len(match.group(1))


def containing_section_labels_v2(blocks: List[Block]) -> List[Optional[str]]:
    """UNUSED in the live T18 path (Mechanism A reverted -- see module
    docstring for the corpus-wide label-collapse defect this version
    was found to have at the mandatory 001 gate). Retained for
    disclosure only; the live path uses T17's own
    containing_section_labels() unchanged.

    Per block, the label of the most recent preceding heading_like
    block whose own '#' level is <= the currently-established level
    (or the first heading_like block seen at all, if none is
    established yet) -- fixing the T17 cascade where every heading_like
    block, including job/entity headings nested within an already-
    labeled section, overwrote the running label unconditionally.

    A heading_like block with no '#'-level (bold-only/plain-text) never
    overwrites an already-established '#'-level label -- see
    docs/T18_PHASE2E_MECHANISM_TRACE_AND_PLAN.md, Mechanism A, for the
    full evidence and tuning-subset verification behind this rule.
    """
    labels: List[Optional[str]] = []
    current: Optional[str] = None
    established_level: Optional[int] = None

    for b in blocks:
        if b.block_type == "heading_like":
            level = heading_level(b.stripped)
            if current is None:
                current = section_label_for_heading_text(b.stripped)
                established_level = level
            elif level is not None and (established_level is None or level <= established_level):
                current = section_label_for_heading_text(b.stripped)
                established_level = level
            # else: nested job/entity heading -- does not overwrite the
            # running section label (the Mechanism A fix).
        labels.append(current)
    return labels
