"""
scripts/_milestone_v5_8_full_corpus_validation.py

Milestone V5.8 -- full-corpus (22-resume) validation of the V5.7 Skills
selective-escalation routing signals (fill_ratio, in_section_frac) and
the Skills-section boundary detector they depend on.

ZERO NEW LLM CALLS. This script only reads already-existing, already-
computed artifacts:

  - benchmark/baselines/milestone_23_67/checkpoints/<markdown_stem>.pkl
    -- the current, frozen, qwen3:1.7b production-baseline extraction
    (Milestone 23.67, run 22 Aug, confirmed via direct code inspection
    of config.MARKDOWN_DIR / evaluation/parser_evaluator.py that this
    run used output/markdown/ as its Stage-1 source -- the same
    directory this script reads from below, so the boundary/signal
    computation below is against the EXACT text the model actually saw,
    not a different Stage-1 snapshot).
  - output/markdown/<markdown_filename>.md -- the real Stage-1 source
    text for the same run (config.MARKDOWN_DIR).
  - benchmark/ground_truth/adjudicated_v1/mapping.csv -- the canonical
    22-resume corpus definition (resume012 permanently excluded, per
    Milestone 23.66/23.67's own established precedent).

GT (`expected` field on the skills FieldComparison) is read ONLY for
offline scoring columns in the output CSVs. It is NEVER used as an
input to fill_ratio or in_section_frac, and is never used to choose a
threshold beyond documented, disclosed exploration.

IMPORTANT DISCLOSED METHODOLOGY NOTE: V5.7's own fill_ratio /
in_section_frac computation code was not preserved in this repository
(no scripts/v5_7_*.py file exists) -- V5.7's CSV deliverables and
milestone report describe the method in prose, not in committed code.
This script is therefore a faithful RECONSTRUCTION of that described
method, not a byte-identical re-run of V5.7's own private script. Every
design decision that required a judgment call (exact heading-boundary
end condition, exact comma/bullet tokenization rule for
n_apparent_comma_items, exact fuzzy-match definition for
in_section_frac) is documented inline below, and is applied UNIFORMLY
and IDENTICALLY across all 22 resumes -- never tuned per-resume to
produce a desired result.

Nothing under services/, pipeline/, validators/, schemas/, prompts/, or
evaluation/ is imported for its LLM-calling behavior -- only
evaluation/parser_evaluator.py's FieldComparison/ComparisonResult
*data classes* are needed to unpickle the checkpoint files (pickle
requires the original class definitions to be importable).
"""
from __future__ import annotations

import csv
import pickle
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Only needed so pickle.load() can resolve the FieldComparison/
# ComparisonResult/DiagnosticResult classes already used by the frozen
# Milestone 23.67 checkpoints. No extraction/LLM code is invoked.
from validators.json_comparator import (  # noqa: F401,E402
    ComparisonResult,
    DiagnosticResult,
    FieldComparison,
)

GT_MAPPING = Path("benchmark/ground_truth/adjudicated_v1/mapping.csv")
CHECKPOINT_DIR = Path("benchmark/baselines/milestone_23_67/checkpoints")
MARKDOWN_DIR = Path("output/markdown")
PDF_DIR = Path("input/resumes")
OUT_DIR = Path("benchmark/baselines")

# --------------------------------------------------------------------
# Skills-section heading vocabulary -- reused VERBATIM from V3.0's own
# corpus-scanned 11-heading list (docs/MILESTONE_V3_0_SKILLS_STRUCTURE_
# FOUNDATION.md, Step 2), not reinvented for this milestone. "Tools" is
# a fallback-only match (per V3.0's own explicit rule: never preferred
# over a genuine Skills-named heading). "Strengths" is deliberately
# excluded (V3.0 confirmed it is a distinct concept, not Skills).
# --------------------------------------------------------------------
SKILLS_HEADINGS = [
    "skills",
    "technical skills",
    "core competencies",
    "competencies",
    "skills & competencies",
    "skill set summary",
    "technical & soft skills",
    "it proficiency",
    "technical qualifications",
    "technical qualification",
    "skills and technical qualifications",
]
FALLBACK_HEADINGS = ["tools"]

# A broad, generic markdown-heading line pattern, used only to find
# where a detected Skills section body ENDS. Captures the '#' run so
# its LEVEL can be compared against the matched Skills heading's own
# level -- a nested subheading (e.g. '## Languages:' under a level-1
# '# Technical Skills') is real Skills content (V3.0's own documented
# "nested subheadings organizing categories" structural form), not a
# section boundary, and must NOT end the section. Only a heading at the
# SAME level or shallower ends it.
_HEADING_LINE = re.compile(r"^\s{0,3}(#{1,6})\s*(.+?)\s*$")
_FENCE_LINE = re.compile(r"^\s*```")


def _heading_level(line: str) -> int | None:
    match = _HEADING_LINE.match(line)
    return len(match.group(1)) if match else None


def _normalize_heading_text(raw: str) -> str:
    """Normalize a candidate heading line for vocabulary comparison.

    Applies, in order: (1) strip a leading form-feed character (V5.7's
    own found-and-fixed native-PDF page-break artifact, e.g.
    '\\x0cTechnical Skills'); (2) strip leading markdown '#' marks;
    (3) strip trailing divider/punctuation characters (V5.7's own
    found-and-fixed 'SKILLS & TECHNICAL QUALIFICATIONS_____' case);
    (4) lowercase; (5) collapse internal whitespace; (6) normalize
    '&'/'and' to a single canonical token, mirroring V3.0's own existing
    interchangeability rule for other headings (e.g. 'Skills &
    Competencies').
    """
    text = raw.lstrip("\x0c")
    text = re.sub(r"^#{1,6}\s*", "", text)
    text = text.strip()
    text = text.rstrip("_-:.─—– \t")
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"\s*&\s*", " and ", text)
    return text


_VOCAB_NORMALIZED = {re.sub(r"\s*&\s*", " and ", h): h for h in SKILLS_HEADINGS}
_FALLBACK_NORMALIZED = {h: h for h in FALLBACK_HEADINGS}


def find_skills_section(markdown_text: str) -> dict:
    """Locate the Skills section in a Stage-1 markdown document.

    Returns a dict with:
      found: bool
      heading_matched: str | None -- the exact vocabulary string matched
      is_fallback: bool -- True if only the 'Tools' fallback matched
      start_line: int | None -- 0-indexed line of the heading itself
      end_line: int | None -- 0-indexed line where the body ends (excl.)
      body_text: str
      multiple_candidates: list[str] -- every OTHER genuine vocabulary
        heading also found in the document (for manual ambiguity review;
        empty in the normal case).
      note: str -- free-text note on anything the automated pass itself
        flagged as worth manual inspection.
    """
    lines = markdown_text.split("\n")

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
            "found": False,
            "heading_matched": None,
            "is_fallback": False,
            "start_line": None,
            "end_line": None,
            "body_text": "",
            "multiple_candidates": [],
            "note": "no vocabulary heading (genuine or fallback) found anywhere in the document",
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

    # Body ends at: the next markdown heading line at the SAME level or
    # SHALLOWER than the matched heading itself (a deeper subheading,
    # e.g. '## Languages:' nested under a level-1 '# Technical Skills',
    # is real Skills content -- V3.0's own documented "nested
    # subheadings organizing categories" structural form -- and must
    # NOT end the section; a real bug in this script's first draft,
    # found and fixed by direct testing against this exact case before
    # trusting the detector further, not glossed over), OR a closing
    # code-fence line (this Stage-1 source wraps logical blocks in
    # ```markdown ... ``` fences -- a Skills section is frequently, but
    # not always, its own complete fence block; the closing fence is
    # checked as an additional, independent, level-agnostic end
    # condition, since a fence boundary is a hard document-structure
    # boundary regardless of heading nesting), OR end of document --
    # whichever comes first.
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

    body_text = "\n".join(lines[start_line + 1 : end_line]).strip()

    note = ""
    if is_fallback:
        note = "only the 'Tools' fallback heading matched -- no genuine Skills-vocabulary heading found"
    elif other_candidates:
        note = f"multiple genuine vocabulary headings found; using the first ({heading_matched!r}); others: {other_candidates!r}"
    elif not body_text:
        note = "heading matched but body is empty"

    return {
        "found": True,
        "heading_matched": heading_matched,
        "is_fallback": is_fallback,
        "start_line": start_line,
        "end_line": end_line,
        "body_text": body_text,
        "multiple_candidates": other_candidates,
        "note": note,
    }


_BULLET_LINE = re.compile(r"^\s*[-–—•*]\s*")


def apparent_comma_item_count(body_text: str) -> int:
    """A simple, transparent proxy for 'how much content is apparently
    being asked for' in a Skills-section body.

    RECONSTRUCTION NOTE (see module docstring): V5.7's own tokenizer
    code was not preserved. This implementation: splits the body first
    on newlines, strips a leading bullet marker (-, --, *, bullet
    character) from each line if present, then further splits each
    resulting line on commas, and counts every non-empty, non-trivial
    (>=2 characters after stripping) resulting token as one apparent
    item. This handles both the labeled comma-list form ('Category: A,
    B, C') and the one-bullet-per-item form seen in this corpus (V3.0's
    own documented structural diversity), without assuming either form
    exclusively.
    """
    if not body_text.strip():
        return 0
    count = 0
    for raw_line in body_text.split("\n"):
        line = _BULLET_LINE.sub("", raw_line).strip()
        if not line:
            continue
        # A labeled category line by itself (e.g. "Category:") with no
        # content after the colon contributes 0 items, not 1.
        for token in line.split(","):
            token = token.strip().strip(":").strip()
            if len(token) >= 2:
                count += 1
    return count


def _normalize_for_containment(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def in_section_frac(extracted_items: list[str], body_text: str) -> tuple[float | None, int, int]:
    """Fraction of extracted_items whose normalized text is found as a
    substring of the normalized Skills-section body text.

    Returns (fraction_or_None, n_found, n_total). Returns (None, 0, 0)
    if extracted_items is empty (undefined, not zero -- an empty
    extraction is a different situation from "extracted things that
    aren't in the section").
    """
    if not extracted_items:
        return None, 0, 0
    norm_body = _normalize_for_containment(body_text)
    n_found = 0
    for item in extracted_items:
        norm_item = _normalize_for_containment(str(item))
        if norm_item and norm_item in norm_body:
            n_found += 1
    return n_found / len(extracted_items), n_found, len(extracted_items)


def load_mapping() -> list[dict]:
    rows = []
    with GT_MAPPING.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_baseline_skills(markdown_filename: str) -> dict:
    """Loads the Milestone 23.67 checkpoint for one resume and returns
    the skills FieldComparison's actual (baseline-extracted) and
    expected (GT, offline-only) lists, plus run status.
    """
    stem = Path(markdown_filename).stem
    ckpt_path = CHECKPOINT_DIR / f"{stem}.pkl"
    if not ckpt_path.exists():
        return {"status": "checkpoint_missing", "actual": None, "expected": None}
    with ckpt_path.open("rb") as f:
        data = pickle.load(f)
    if data.get("status") != "ok":
        return {"status": data.get("status", "unknown"), "actual": None, "expected": None}
    comparison = data["comparison"]
    skills_field = next((fr for fr in comparison.field_results if fr.field_path == "skills"), None)
    if skills_field is None:
        return {"status": "no_skills_field", "actual": None, "expected": None}
    return {
        "status": "ok",
        "actual": skills_field.actual or [],
        "expected": skills_field.expected or [],
    }


if __name__ == "__main__":
    print("This module is imported by the V5.8 analysis driver scripts; "
          "see scripts/_milestone_v5_8_run_analysis.py for the executable entry point.")
