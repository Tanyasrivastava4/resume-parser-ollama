"""
scripts/_milestone_v5_9_signal_corrections.py

Milestone V5.9 -- narrowly-scoped, deterministic corrections to the
three V5.8-traced Skills selective-escalation measurement-signal
problems:

  A. Parenthetical-comma tokenizer overcount
     (`apparent_comma_item_count` split on EVERY comma, including
     commas inside parentheses, e.g. "Microsoft Office (Excel, Word,
     PowerPoint)" was counted as 3 items instead of 1).
  B. Nested non-Skills ("Languages") subheading inflation
     (a nested `### Languages` / `## Languages` subsection listing
     human-language proficiency was swept into the Skills body and
     counted, inflating the denominator).
  C. Resume-006-class boundary-safety gap
     (the heading-vocabulary matcher accepted a bare "Skills" line
     immediately preceding an unrelated markdown TABLE -- an Education
     table -- with no content-plausibility check at all).

This is a RESEARCH/MEASUREMENT-LAYER correction only. `services/`,
`pipeline/`, `prompts/`, `schemas/`, `validators/` are untouched.
V5.8's own committed script (`_milestone_v5_8_full_corpus_validation.py`)
is intentionally NOT modified -- it remains the unaltered historical
record of what V5.8 actually measured. This module reuses V5.8's own
I/O/mapping utilities unchanged (imported directly) and replaces only
the three functions responsible for the traced problems.

Zero LLM calls. Zero GT access in any signal computation below (GT is
used only by the separate offline-evaluation script, exactly as in
V5.8, never here).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Reused, unmodified, from V5.8's own module -- not reimplemented.
from scripts._milestone_v5_8_full_corpus_validation import (  # noqa: E402
    _normalize_heading_text,
)

def _load_programming_language_denylist() -> set[str]:
    """Reads PROGRAMMING_LANGUAGE_DENYLIST's literal value directly out
    of services/languages_extractor.py's source text, WITHOUT importing
    that module. `services/languages_extractor.py` imports
    `services/base_module_extractor.py` -> `services/ollama_client.py`,
    which performs real Ollama HTTP calls and requires the `httpx`
    package -- importing the module (even without calling anything)
    would pull live model-calling code into this milestone's strictly
    zero-LLM-call research script, and would fail outright in any
    environment without `httpx` installed (confirmed: it failed here).
    Reading the already-evidenced, already-curated constant's source
    text is a deliberate, disclosed choice to reuse that existing
    abstraction's VALUE without its import-time side effects -- not a
    reimplementation or reinvention of the vocabulary itself."""
    import ast

    src = (_PROJECT_ROOT / "services" / "languages_extractor.py").read_text(encoding="utf-8")
    marker = "PROGRAMMING_LANGUAGE_DENYLIST = "
    start = src.index(marker) + len(marker)
    end = src.index("}", start) + 1
    return set(ast.literal_eval(src[start:end]))


# Reused, unmodified VALUE, from the existing production Languages
# module -- an already-evidenced, already-curated vocabulary
# distinguishing programming-language tokens from everything else (see
# services/languages_extractor.py's own commit history: re-verified
# there to have zero overlap with real GT "languages" values). This is
# the single decision this milestone's fix for problem B depends on,
# and it is NOT reinvented here.
PROGRAMMING_LANGUAGE_DENYLIST = _load_programming_language_denylist()

HEADING_LINE_RE = re.compile(r"^\s{0,3}(#{1,6})\s*(.+?)\s*$")
_FENCE_LINE = re.compile(r"^\s*```")
_DIVIDER_LINE_RE = re.compile(r"^[\s\-_=~]{3,}$")
_BULLET_LINE = re.compile(r"^\s*[-–—•*]\s*")

# Problem C: a body dominated by markdown-table syntax is structurally
# implausible as a Skills list (this corpus's real Skills sections are
# always comma-lists or bullet lists, never tables -- verified across
# all 22 resumes, see the milestone report's corpus scan). Requires
# BOTH at least one genuine header-separator row (e.g. "|---|---|")
# AND at least 2 pipe-delimited rows, to avoid flagging a single
# incidental "|" character.
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")


def _heading_level(line: str) -> int | None:
    m = HEADING_LINE_RE.match(line)
    return len(m.group(1)) if m else None


def _is_table_like(body_text: str) -> bool:
    lines = body_text.split("\n")
    n_rows = sum(1 for l in lines if _TABLE_ROW_RE.match(l))
    n_sep = sum(1 for l in lines if _TABLE_SEP_RE.match(l))
    return n_sep >= 1 and n_rows >= 2


def _protect_paren_commas(line: str) -> str:
    """Replace commas found INSIDE a single-level parenthetical group
    with a placeholder, so a later comma-split does not treat them as
    top-level item separators. Corpus-verified (all 22 resumes) to
    contain no nested parentheses, so single-level protection is
    sufficient -- not a general-purpose parser, a narrow fix matched
    to the evidence."""
    return re.sub(r"\([^()]*\)", lambda m: m.group(0).replace(",", "\x00"), line)


def _tokenize_line_items(line: str) -> list[str]:
    """Split one content line into apparent item tokens, protecting
    parenthetical commas (problem A) and skipping heading/divider
    lines entirely (a related overcount this milestone also found
    during reproduction: heading lines like '### Software Proficiency'
    and divider lines like '---' were previously counted as fake
    'items' of their own)."""
    stripped = line.strip()
    if not stripped:
        return []
    if HEADING_LINE_RE.match(line) or _FENCE_LINE.match(line):
        return []
    if _DIVIDER_LINE_RE.match(stripped):
        return []
    content = _BULLET_LINE.sub("", line).strip()
    if not content:
        return []
    protected = _protect_paren_commas(content)
    tokens = []
    for tok in protected.split(","):
        tok = tok.replace("\x00", ",").strip().strip(":").strip()
        if len(tok) >= 2:
            tokens.append(tok)
    return tokens


def apparent_comma_item_count(body_text: str) -> int:
    """Corrected replacement for V5.8's function of the same name.
    Fixes problem A (parenthetical commas) and the related
    heading/divider-line overcount found during this milestone's own
    reproduction of problem B."""
    if not body_text.strip():
        return 0
    count = 0
    for raw_line in body_text.split("\n"):
        count += len(_tokenize_line_items(raw_line))
    return count


def _find_nonskills_language_spans(lines: list[str]) -> list[tuple[int, int]]:
    """Problem B. Find nested (any-level) headings within the body
    literally named 'Languages'/'Language', and classify each span
    (heading to the next heading of any level, or end of body) as
    non-Skills ONLY if every item token within it normalizes to
    something NOT in PROGRAMMING_LANGUAGE_DENYLIST -- i.e. it looks
    like human-language proficiency content (e.g. 'English', 'Hindi'),
    not a legitimate programming-languages Skills subcategory (e.g.
    'JavaScript, TypeScript, Python', confirmed real content for
    resumes 001 and 013 in this exact corpus -- these must NOT be
    excluded, and are the reason this check requires actual content
    inspection rather than matching the heading text alone)."""
    spans = []
    for i, line in enumerate(lines):
        m = HEADING_LINE_RE.match(line)
        if not m:
            continue
        heading_text = m.group(2).strip().rstrip(":").strip().lower()
        if heading_text not in ("languages", "language"):
            continue
        end = len(lines)
        for j in range(i + 1, len(lines)):
            if HEADING_LINE_RE.match(lines[j]):
                end = j
                break
        items: list[str] = []
        for k in range(i + 1, end):
            items.extend(_tokenize_line_items(lines[k]))
        if not items:
            continue
        normalized = [tok.strip().lower() for tok in items]
        all_programming = all(tok in PROGRAMMING_LANGUAGE_DENYLIST for tok in normalized)
        if not all_programming:
            spans.append((i, end))
    return spans


def find_skills_section(markdown_text: str) -> dict:
    """Corrected replacement for V5.8's function of the same name.
    Boundary DETECTION logic (which heading matches, where the section
    ends by level) is UNCHANGED from V5.8 -- that logic was found
    CLEAR on 18/22 resumes and is not touched. This adds:

      - `boundary_content_implausible` (problem C): True if the
        located body is structurally table-like (see _is_table_like).
        When True, `found` remains True (a heading WAS matched -- this
        is honestly reported, not hidden) but a caller MUST treat the
        signal as invalid; this function does not silently drop it.
      - `cleaned_body_text` (problem B): body_text with any
        classified-non-Skills nested 'Languages' span's lines removed.
        `body_text` (the original, uncorrected span) is still returned
        unchanged for audit/comparison.
      - `excluded_nonskills_spans`: line ranges excluded, for
        transparency.
    """
    lines = markdown_text.split("\n")

    # Reuse V5.8's own vocabulary-matching tables exactly (imported,
    # not reimplemented) -- see _normalize_heading_text (reused above)
    # for the exact same matching rule V5.8 used.
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
            "boundary_content_implausible": False,
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
    for j in range(start_line + 1, len(lines)):
        line = lines[j]
        if _FENCE_LINE.match(line):
            end_line = j
            break
        level = _heading_level(line)
        if level is not None and level <= own_level:
            end_line = j
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

    return {
        "found": True,
        "heading_matched": heading_matched,
        "is_fallback": is_fallback,
        "start_line": start_line,
        "end_line": end_line,
        "body_text": body_text,
        "cleaned_body_text": cleaned_body_text,
        "excluded_nonskills_spans": nonskills_spans,
        "boundary_content_implausible": implausible,
        "multiple_candidates": other_candidates,
        "note": note,
    }


def _normalize_for_containment(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def in_section_frac(extracted_items: list[str], body_text: str) -> tuple[float | None, int, int]:
    """Unchanged in logic from V5.8 -- reproduced here (not imported)
    only so this module has zero runtime dependency on V5.8's own
    function objects, since callers must pass the CORRECTED
    (cleaned_body_text) body, not the original body_text."""
    if not extracted_items:
        return None, 0, 0
    norm_body = _normalize_for_containment(body_text)
    n_found = 0
    for item in extracted_items:
        norm_item = _normalize_for_containment(str(item))
        if norm_item and norm_item in norm_body:
            n_found += 1
    return n_found / len(extracted_items), n_found, len(extracted_items)
