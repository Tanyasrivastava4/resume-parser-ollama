"""
scripts/_milestone_t9_phase2b_span_extraction.py

Milestone T9/Phase 2B -- research-only helper.

Derives the deterministic "Work Experience SECTION span" (the whole
Experience section's text -- section heading through section end, NOT
the individual per-job JobSpan objects `WorkExperienceLocator.locate()`
already returns) by calling ONLY the real, existing, UNMODIFIED
`WorkExperienceLocator` private methods it already uses internally
(`_find_section_start`, `_find_section_end`,
`_find_repeated_section_name_markers`). No new boundary-detection logic
is written here -- this module only calls into the real locator and
slices the result. `services/work_experience_locator.py` itself is
never modified.

This is NOT used by production code and is NOT imported by
`pipeline/parser_pipeline.py` or any production module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from services.work_experience_locator import WorkExperienceLocator


@dataclass
class SectionSpanResult:
    found: bool
    start_line_0indexed: Optional[int]
    end_line_0indexed: Optional[int]  # exclusive
    path: str  # "repeated_marker" | "normal" | "not_found"
    section_text: Optional[str]


def extract_section_span(markdown_text: str, locator: Optional[WorkExperienceLocator] = None) -> SectionSpanResult:
    """Return the whole Work Experience SECTION's text, using only the
    real WorkExperienceLocator's own existing private boundary logic
    (unmodified, called exactly as locate() itself calls it)."""
    loc = locator or WorkExperienceLocator()
    lines: List[str] = markdown_text.splitlines()

    repeated_markers = loc._find_repeated_section_name_markers(lines)  # noqa: SLF001
    if repeated_markers:
        body_start = repeated_markers[0]
        section_end = loc._find_section_end(lines, repeated_markers[-1], section_level=0)  # noqa: SLF001
        section_text = "\n".join(lines[body_start:section_end])
        return SectionSpanResult(
            found=True, start_line_0indexed=body_start, end_line_0indexed=section_end,
            path="repeated_marker", section_text=section_text,
        )

    section_start, section_level = loc._find_section_start(lines)  # noqa: SLF001
    if section_start is None:
        return SectionSpanResult(
            found=False, start_line_0indexed=None, end_line_0indexed=None,
            path="not_found", section_text=None,
        )

    section_end = loc._find_section_end(lines, section_start, section_level)  # noqa: SLF001
    section_text = "\n".join(lines[section_start:section_end])
    return SectionSpanResult(
        found=True, start_line_0indexed=section_start, end_line_0indexed=section_end,
        path="normal", section_text=section_text,
    )
