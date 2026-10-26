"""
scripts/_milestone_v6_6_boundary_correction.py

Milestone V6.6 -- narrowly-scoped correction to the Skills-section
END-BOUNDARY logic in scripts/_milestone_v5_9_signal_corrections.py's
`find_skills_section`. RESEARCH/EVALUATION LAYER ONLY -- this function
lives only in `scripts/`, imported nowhere by `services/` or
`pipeline/` (confirmed by direct grep before this milestone began);
`services/skills_extractor.py` is NOT touched, and was confirmed NOT to
contain any boundary-detection logic of its own.

ROOT CAUSE (reproduced and characterized in this milestone, Phase 1):
V5.9's (and, before it, V5.8's) end-boundary loop terminates the
Skills body ONLY at (a) a same-or-shallower '#'-prefixed markdown
heading, or (b) a closing code-fence line. It has NO check for a
genuine next-section title that is NOT '#'-prefixed. This is a
markdown-format-dependence defect, not a resume-005-specific one:
direct inspection in this milestone found it affects FOUR of the six
V5.6-lineage resumes when scored against
output/v2_4h_benchmark/stage1/markdown/ (which uses bare-text section
titles throughout, not '#' headings) -- 005 (confirmed in V6.5),
**013 and 016 (both newly discovered in this milestone -- their
Skills body overran all the way to embedded PDF hyperlink
annotations)**, and **021 (newly discovered -- overran deep into
Work Experience job listings)**. 023 was already correctly bounded
(a code fence happened to terminate it). 001 is unaffected by THIS
defect specifically -- it has an already-documented, different,
unrelated Stage-1 defect (column-bleed fusing "Summary" and "Technical
Skills," V2.6C/V5.7) that prevents its Skills heading from being found
at all under this lineage, which this milestone does not touch.

FIX: add a bare-text "next real resume section" check as an
ADDITIONAL end-boundary condition, alongside the existing '#'-heading
and code-fence checks (neither of which is modified or removed).
Vocabulary reused, not invented: the same COMMON_NEXT_SECTIONS list
V5.7's own (uncommitted, described-in-prose) design already
established, MINUS "Languages"/"Language" (deliberately excluded --
nested Languages content is already handled by V5.9's own
`_find_nonskills_language_spans`, which needs to see INSIDE a nested
Languages subheading to apply its denylist check; terminating the
section AT that heading instead would break that already-validated
mechanism for resumes 001, 008, 013, 020), PLUS "Reference" (singular)
-- added because resume 005's real source uses the singular form,
directly evidenced, not guessed.

MATCH RULE, evidence-calibrated: a non-'#' line qualifies as a section
terminator only if, after stripping leading/trailing whitespace and
trailing divider/punctuation characters, it (a) exactly equals a
vocabulary term (case-insensitive), OR (b) starts with a vocabulary
term as a whole word AND the full cleaned line is <=40 characters --
the second condition exists specifically to catch real compound
titles ("Awards and Achievements", "Awards & Recognition," both
directly observed in this corpus) while excluding ordinary prose or
bullet content that might merely CONTAIN one of these words mid-
sentence (verified: no legitimate Skills-body content anywhere in the
22-resume corpus, under either Stage-1 lineage, was found to
false-trigger this rule -- see the milestone report's corpus-wide
scan). Bullet lines are excluded from this check entirely (a section
title is never itself a bullet).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Reused, unmodified, from V5.9's own module -- not reimplemented.
from scripts._milestone_v5_9_signal_corrections import (  # noqa: E402
    HEADING_LINE_RE,
    _BULLET_LINE,
    _FENCE_LINE,
    _find_nonskills_language_spans,
    _heading_level,
    _is_table_like,
    _normalize_heading_text,
    apparent_comma_item_count,  # noqa: F401 -- re-exported for callers, unchanged
    in_section_frac,  # noqa: F401 -- re-exported for callers, unchanged
)

# Reused verbatim from V5.7's own established list (per V5.8's own
# docstring: "V5.7's own fill_ratio/in_section_frac computation code
# was not preserved in this repository... described in prose, not in
# committed code" -- this is that same described vocabulary, now
# finally committed here for the first time), MINUS "Languages"/
# "Language" (deliberately excluded, see module docstring), PLUS
# "Reference" (singular, added -- directly evidenced by resume 005's
# real source text, not guessed).
NEXT_SECTION_VOCAB = [
    "Experience", "Work Experience", "Professional Experience", "Education",
    "Projects", "Project", "Certifications", "Certification", "Achievements",
    "Personal Details", "Declaration", "References", "Reference",
    "Summary", "Professional Summary", "Objective", "Awards",
    "Publications", "Internship", "Internships", "Training", "Trainings",
    "Extra Curricular", "Hobbies", "Interests",
]
_NEXT_SECTION_VOCAB_LOWER = sorted((v.lower() for v in NEXT_SECTION_VOCAB), key=len, reverse=True)
_MAX_TITLE_LINE_LEN = 40


def _is_bare_next_section_title(line: str) -> bool:
    """The new check this milestone adds. Only ever called on lines
    that are NOT already a '#'-heading (the caller checks that first)
    and are not bullets."""
    if _BULLET_LINE.match(line):
        return False
    cleaned = re.sub(r"[\s_\-:=~]+$", "", line.strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    if not cleaned:
        return False
    for v in _NEXT_SECTION_VOCAB_LOWER:
        if cleaned == v:
            return True
        if len(cleaned) <= _MAX_TITLE_LINE_LEN and re.match(r"^" + re.escape(v) + r"\b", cleaned):
            return True
    return False


def find_skills_section(markdown_text: str) -> dict:
    """Corrected replacement for V5.9's function of the same name.
    IDENTICAL in every other respect (heading-vocabulary matching,
    table-content-implausibility check, nested-Languages-span
    exclusion, tokenization) -- only the end-boundary loop gains one
    new termination condition, `_is_bare_next_section_title`, alongside
    the two it already had ('#'-heading, code fence)."""
    lines = markdown_text.split("\n")

    from scripts._milestone_v5_8_full_corpus_validation import (
        _VOCAB_NORMALIZED,
        _FALLBACK_NORMALIZED,
    )

    genuine_hits: list[tuple[int, str]] = []
    fallback_hits: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        norm = _normalize_heading_text(line)
        if not norm:
            continue
        if norm in _VOCAB_NORMALIZED:
            genuine_hits.append((i, _VOCAB_NORMALIZED[norm]))
        elif norm in _FALLBACK_NORMALIZED:
            fallback_hits.append((i, norm))

    if not genuine_hits and not fallback_hits:
        return {
            "found": False, "heading_matched": None, "is_fallback": False,
            "start_line": None, "end_line": None, "body_text": "",
            "cleaned_body_text": "", "excluded_nonskills_spans": [],
            "boundary_content_implausible": False, "end_reason": None,
            "multiple_candidates": [], "note": "no vocabulary heading found",
        }

    if genuine_hits:
        start_line, heading_matched = genuine_hits[0]
        is_fallback = False
        other_candidates = [h for (_, h) in genuine_hits[1:]]
    else:
        start_line, heading_matched = fallback_hits[0]
        is_fallback = True
        other_candidates = []

    own_level = _heading_level(lines[start_line]) or 1
    end_line = len(lines)
    end_reason = "end_of_document"
    for j in range(start_line + 1, len(lines)):
        line = lines[j]
        if _FENCE_LINE.match(line):
            end_line = j
            end_reason = "code_fence"
            break
        level = _heading_level(line)
        if level is not None and level <= own_level:
            end_line = j
            end_reason = "markdown_heading"
            break
        # NEW in V6.6: a genuine, non-'#' next-section title.
        if _is_bare_next_section_title(line):
            end_line = j
            end_reason = "bare_text_next_section_title"
            break

    body_lines = lines[start_line + 1:end_line]
    body_text = "\n".join(body_lines).strip()

    implausible = _is_table_like(body_text)

    nonskills_spans = _find_nonskills_language_spans(body_lines)
    if nonskills_spans:
        excluded_idx = set()
        for s, e in nonskills_spans:
            excluded_idx.update(range(s, e))
        cleaned_lines = [l for idx, l in enumerate(body_lines) if idx not in excluded_idx]
        cleaned_body_text = "\n".join(cleaned_lines).strip()
    else:
        cleaned_body_text = body_text

    note = ""
    if implausible:
        note = "body is structurally table-like -- content implausible as a Skills list"
    elif is_fallback:
        note = "only the 'Tools' fallback heading matched"
    elif other_candidates:
        note = f"multiple genuine vocabulary headings found; using the first ({heading_matched!r}); others: {other_candidates!r}"
    elif not body_text:
        note = "heading matched but body is empty"
    if nonskills_spans:
        note = (note + "; " if note else "") + f"{len(nonskills_spans)} nested non-Skills 'Languages' span(s) excluded"
    if end_reason == "bare_text_next_section_title":
        note = (note + "; " if note else "") + "section terminated by a bare-text (non-'#') next-section title (V6.6 correction)"

    return {
        "found": True,
        "heading_matched": heading_matched,
        "is_fallback": is_fallback,
        "start_line": start_line,
        "end_line": end_line,
        "end_reason": end_reason,
        "body_text": body_text,
        "cleaned_body_text": cleaned_body_text,
        "excluded_nonskills_spans": nonskills_spans,
        "boundary_content_implausible": implausible,
        "multiple_candidates": other_candidates,
        "note": note,
    }
