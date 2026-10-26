"""
evaluation/contact_information_evidence_report.py

Contact Information Evidence Reporting (Milestone V3.9A).

Pure, read-only aggregation of ContactInformationEvidenceRecord objects
(services/evidence_record.py, Milestone V3.8; wired into
ParserPipeline.run() in Milestone V3.9) into corpus-level, field-level,
and resume-level CSV-ready rows for human inspection.

This module does NOT:
  - run the parser pipeline itself (callers pass already-produced
    ContactInformationEvidenceRecord objects in);
  - modify, delete, replace, reject, or autofill any extracted value --
    it never even sees an extracted contact_information dict, only the
    already-built evidence record's own fields;
  - invent a numeric confidence score, a new evidence label, or any
    new semantics -- every row emitted here uses the exact 6 closed
    `evidence_summary` labels ContactFieldEvidence already defines
    (NO_EXTRACTED_VALUE, NO_EVIDENCE_SOURCES_AVAILABLE,
    FULLY_CORROBORATED, PARTIALLY_CORROBORATED, NOT_CORROBORATED,
    EVIDENCE_DISAGREEMENT), copied verbatim, never renamed or reinterpreted;
  - treat NO_EXTRACTED_VALUE, NO_EVIDENCE_SOURCES_AVAILABLE, or
    NOT_CORROBORATED as a negative verdict -- these are reported as
    plain counts alongside every other label, with no severity, ranking,
    or "problem" framing applied by this module.

Every function here is pure: given the same list of
ContactInformationEvidenceRecord objects, it returns the same rows,
every time, with no side effects (no file writes, no pipeline calls).
Callers are responsible for writing the returned rows to CSV.
"""

from collections import Counter
from typing import Any, Dict, List, Tuple

from services.evidence_record import ContactInformationEvidenceRecord

# The exact 6 closed evidence_summary labels, copied verbatim from
# services/evidence_record.py's own module-level constants -- listed
# here only so this module's own row-ordering/zero-filling is stable
# and complete even for labels that happen not to occur in a given
# corpus (so a 0 count is reported explicitly, not silently omitted).
EVIDENCE_LABELS = (
    "NO_EXTRACTED_VALUE",
    "NO_EVIDENCE_SOURCES_AVAILABLE",
    "FULLY_CORROBORATED",
    "PARTIALLY_CORROBORATED",
    "NOT_CORROBORATED",
    "EVIDENCE_DISAGREEMENT",
)

# Labels that represent the extracted value having at least one
# corroborating source (visible text and/or annotation). Used only to
# compute a "corroborated vs. not" rollup count in the corpus summary --
# this is a plain re-grouping of the existing labels, not a new label
# and not a score.
_CORROBORATED_LABELS = frozenset({"FULLY_CORROBORATED", "PARTIALLY_CORROBORATED"})

# Labels that represent "no populated value to evaluate at all" or "no
# evidence source could be checked" -- kept separate from disagreement
# and from corroboration, per the explicit rule that absence of
# evidence must never be conflated with a wrong value.
_ABSENCE_LABELS = frozenset({"NO_EXTRACTED_VALUE", "NO_EVIDENCE_SOURCES_AVAILABLE", "NOT_CORROBORATED"})


def evidence_records_to_field_rows(
    records: List[Tuple[str, str, ContactInformationEvidenceRecord]]
) -> List[Dict[str, Any]]:
    """Flatten a list of (resume_id, stem, record) into one row per
    resume x Contact Information field -- the finest-grained, fully
    traceable representation every other aggregation in this module is
    built from. Never drops a field, never merges two fields' evidence
    together.

    Args:
        records: (resume_id, stem, ContactInformationEvidenceRecord)
            triples, one per resume.

    Returns:
        A list of plain dicts, one per (resume, field), each holding
        every ContactFieldEvidence attribute verbatim plus the owning
        resume_id/stem -- suitable for direct CSV writing.
    """
    rows: List[Dict[str, Any]] = []
    for resume_id, stem, record in records:
        for f in record.fields:
            rows.append({
                "resume_id": resume_id,
                "stem": stem,
                "field": f.field_name,
                "extracted_values": " | ".join(f.extracted_values),
                "extracted_value_count": len(f.extracted_values),
                "visible_text_checked": f.visible_text_checked,
                "visible_text_matches": " | ".join(f.visible_text_matches),
                "annotations_checked": f.annotations_checked,
                "annotation_matches": " | ".join(f.annotation_matches),
                "unmatched_same_type_annotations": " | ".join(f.unmatched_same_type_annotations),
                "evidence_summary": f.evidence_summary,
            })
    return rows


def build_corpus_summary(field_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One corpus-level summary dict from the field-level rows.

    Reports, using the exact existing terminology and nothing invented:
      - total resumes and total (resume, field) records evaluated,
      - total populated fields (extracted_value_count > 0) vs. empty,
      - a count for every one of the 6 closed evidence_summary labels
        (0 if a label never occurs in this corpus -- never omitted),
      - a plain re-grouping into corroborated / absence-or-unchecked /
        disagreement counts, computed only from the existing label
        counts above, not a new signal.
    """
    resume_ids = {r["resume_id"] for r in field_rows}
    total_records = len(field_rows)
    populated = [r for r in field_rows if r["extracted_value_count"] > 0]
    empty = [r for r in field_rows if r["extracted_value_count"] == 0]

    label_counts = Counter(r["evidence_summary"] for r in field_rows)

    summary: Dict[str, Any] = {
        "total_resumes": len(resume_ids),
        "total_resume_field_records": total_records,
        "total_populated_fields": len(populated),
        "total_empty_fields": len(empty),
    }
    for label in EVIDENCE_LABELS:
        summary[f"count_{label}"] = label_counts.get(label, 0)

    summary["count_corroborated_any_source"] = sum(
        label_counts.get(lbl, 0) for lbl in _CORROBORATED_LABELS
    )
    summary["count_absence_or_unchecked"] = sum(
        label_counts.get(lbl, 0) for lbl in _ABSENCE_LABELS
    )
    summary["count_disagreement"] = label_counts.get("EVIDENCE_DISAGREEMENT", 0)

    return summary


def build_field_level_distribution(field_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per (Contact Information field, evidence_summary label)
    combination, with the count of resumes falling into it -- the
    field-level distribution CSV. Every field x every label combination
    is emitted, including zero counts, so an absent row is never
    ambiguous with "not computed"."""
    field_names = sorted({r["field"] for r in field_rows}) or [
        "email", "phone", "linkedin", "github", "portfolio",
    ]
    counts = Counter((r["field"], r["evidence_summary"]) for r in field_rows)

    rows: List[Dict[str, Any]] = []
    for field_name in field_names:
        for label in EVIDENCE_LABELS:
            rows.append({
                "field": field_name,
                "evidence_summary": label,
                "count": counts.get((field_name, label), 0),
            })
    return rows


def build_resume_level_rollup(field_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per resume, with a count of that resume's 5 fields
    falling into each evidence_summary label -- lets a reviewer see, at
    a glance, whether any single resume has an unusual concentration of
    a particular label (e.g. several EVIDENCE_DISAGREEMENT fields),
    without collapsing anything into a single score."""
    by_resume: Dict[Tuple[str, str], Counter] = {}
    for r in field_rows:
        key = (r["resume_id"], r["stem"])
        by_resume.setdefault(key, Counter())[r["evidence_summary"]] += 1

    rows: List[Dict[str, Any]] = []
    for (resume_id, stem), label_counter in sorted(by_resume.items()):
        row: Dict[str, Any] = {"resume_id": resume_id, "stem": stem}
        for label in EVIDENCE_LABELS:
            row[label] = label_counter.get(label, 0)
        rows.append(row)
    return rows


def build_edge_case_rows(field_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rows requiring manual human attention: every EVIDENCE_DISAGREEMENT
    case, plus every case with at least one unmatched-same-type
    annotation even when the overall evidence_summary is
    FULLY_CORROBORATED (the V3.7 placeholder-mailto/stale-LinkedIn
    pattern -- worth surfacing to a reviewer even though it did not
    change the field's own status). Never implies these are wrong
    values -- see this module's own top docstring."""
    return [
        r for r in field_rows
        if r["evidence_summary"] == "EVIDENCE_DISAGREEMENT"
        or r["unmatched_same_type_annotations"]
    ]
