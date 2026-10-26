"""
evaluation/human_review_evidence_report.py

Human Review Evidence Report (Milestone V5.1).

Pure, read-only synthesis of the six per-module observational evidence
records already exposed on ParserPipelineResult (Contact Information
since Milestone V3.9; Work Experience/Education/Languages/Professional
Summary/Personal Information since Milestone V5.0) into a single,
per-resume, human-facing report structure -- for VISIBILITY and
EXPLANATION only, never for automated decision-making.

This is the "Option D" report Milestone V4.9 identified as the natural
next step after Milestone V5.0's own "Option B" wiring, and it follows
the exact architectural precedent evaluation/contact_information_
evidence_report.py already established (Milestone V3.9A): pure
functions, given the same already-built evidence, return the same rows
every time, with no side effects.

This module does NOT:
  - run the parser pipeline itself (callers pass an already-produced
    ParserPipelineResult in -- see build_resume_evidence_report());
  - call an LLM, or depend on one being available;
  - modify, delete, replace, reject, or repair any extracted value, or
    the ParserPipelineResult / evidence objects it reads from (every
    function here only reads attributes -- see the module's own tests
    for a direct mutation-safety proof);
  - invent a numeric confidence score, a severity score, a cross-module
    ranking, or a single PASS/FAIL/GOOD/BAD/HIGH-CONFIDENCE/LOW-
    CONFIDENCE status -- every explanation is generated deterministically
    from each module's own already-existing, already-documented evidence
    fields, using the presentation vocabulary Milestone V4.9's own Step 4
    already designed and traced to real states (see MILESTONE_V4_9_
    PARSER_EVIDENCE_VISIBILITY_ASSESSMENT.md);
  - flatten six modules' differently-shaped native evidence into one
    common shape -- Work Experience/Education keep their own raw
    per-entry fields (locator_matched, match_confidence,
    degree_evidence, institution_evidence, review_priority,
    validation_actions) rather than being coerced into a label the way
    Contact Information/Languages/Professional Summary/Personal
    Information already have one;
  - report on Skills -- Skills has no evidence function (Milestones
    V4.2/V4.3 closed that investigation, NO-GO), so Skills is
    deliberately absent from every report this module produces, not
    silently omitted as an oversight.

Module-to-evidence mapping this report design is built from (Step 1's
required mapping -- see docs/MILESTONE_V5_1_HUMAN_REVIEW_EVIDENCE_
REPORT.md for the full table with citations):

  Work Experience   -- ModuleEvidenceRecord/EntryEvidence (raw fields,
                        no single closed label). CAN support: "a
                        matching span was found in the source text for
                        this entry" / "no matching span was found, and
                        this entry's company/designation fields are
                        identical -- a pattern this corpus has
                        previously found concentrated in entries that
                        did not correspond to a real job." CANNOT
                        support: a claim that a no-match entry IS
                        incorrect (Milestone V2.9's own corpus found
                        66.7-87% of no-match entries still GT-correct).
  Education         -- ModuleEvidenceRecord/EntryEvidence (raw fields:
                        locator_matched, degree_evidence,
                        institution_evidence -- booleans, no numeric
                        signal). CAN support: "degree/institution text
                        was found in a matching source span." CANNOT
                        support: any numeric confidence -- Education
                        genuinely has none (Milestone V2.7's own
                        finding).
  Contact Information -- ContactFieldEvidence.evidence_summary (6
                        closed labels). CAN support: source-text and/or
                        PDF-annotation corroboration. CANNOT support:
                        that a non-corroborated value is wrong
                        (Milestone V3.7 found confirmed false-positive-
                        risk cases -- a placeholder mailto:, a stale
                        LinkedIn annotation -- where disagreement was
                        NOT a real defect).
  Languages         -- LanguageValueEvidence.evidence_summary (3 closed
                        labels). CAN support: whole-document text
                        presence. CANNOT support: detecting semantic
                        misattribution -- Milestone V4.0 directly found
                        this check catches pure hallucination but not a
                        word appearing in an unrelated (e.g. project-
                        description) context.
  Professional Summary -- ProfessionalSummaryEvidence.evidence_summary
                        (4 closed labels) + token_overlap_ratio (raw
                        measurement, not a confidence score). CAN
                        support: verbatim/near-verbatim source
                        presence. CANNOT support: a "ground truth
                        disagrees" claim -- GT is never a parameter to
                        this evidence at all (Milestone V4.4 found the
                        dominant "differ" mechanism is GT itself
                        diverging from source, not extraction error).
  Personal Information -- NameCompletenessEvidence.evidence_summary (3
                        closed labels) + LocationEvidence.evidence_summary
                        (4 closed labels). Name-completeness CAN
                        support: "not every full_name token appears in
                        the first/last name split" (0 FP/FN
                        corpus-wide, Milestone V4.7) -- CANNOT support:
                        which field should be corrected. Location CAN
                        support: verbatim source-text presence --
                        CANNOT support: distinguishing correct
                        attribution from a real, verbatim value copied
                        from the wrong section (Milestone V4.7 found 2
                        confirmed cases of exactly this).
"""

from typing import Any, Dict, List, Optional

# The fixed, ordered set of modules this report covers. Skills is
# deliberately absent -- see this module's own top docstring.
REPORT_MODULES = (
    "personal_information",
    "contact_information",
    "professional_summary",
    "languages",
    "work_experience",
    "education",
)

_EVIDENCE_NOT_AVAILABLE = "(module evidence unavailable)"


def _entry(module: str, item_id: str, native_state: Dict[str, Any], explanation: str,
           limitations: Optional[List[str]] = None) -> Dict[str, Any]:
    """One report row, always the same shape regardless of module --
    ONLY the shared envelope (module/item_id/explanation/limitations) is
    common; native_state itself stays whatever shape that module's own
    evidence already has (a dict of its own field names/values, never
    coerced into a shared label)."""
    return {
        "module": module,
        "item_id": item_id,
        "native_evidence_state": dict(native_state),
        "explanation": explanation,
        "limitations": list(limitations or []),
    }


def _module_unavailable_entry(module: str) -> Dict[str, Any]:
    """A single placeholder row for a module whose evidence field is
    None on the ParserPipelineResult (the module itself failed before
    evidence-building could run) -- emitted explicitly so this is never
    silently indistinguishable from "0 findings"."""
    return _entry(
        module, "(module)", {"evidence_available": False},
        "Evidence generation did not run for this module on this resume "
        "(the module extractor may have failed before evidence-building "
        "could run). This is not a statement about the value's "
        "correctness -- it means no evidence check occurred at all.",
    )


# ---------------------------------------------------------------------------
# Contact Information
# ---------------------------------------------------------------------------

_CI_EXPLANATIONS = {
    "NO_EXTRACTED_VALUE": "No value was extracted for this field; there is nothing to evaluate.",
    "NO_EVIDENCE_SOURCES_AVAILABLE": "No source text or PDF annotation data was available to check this "
        "value against; this does not indicate an error.",
    "FULLY_CORROBORATED": "Source evidence was found for this value.",
    "PARTIALLY_CORROBORATED": "Some, but not all, extracted values for this field were corroborated by "
        "available source evidence.",
    "NOT_CORROBORATED": "No matching source evidence was found for this value; this alone does not "
        "indicate the value is incorrect.",
    "EVIDENCE_DISAGREEMENT": "An informational discrepancy was found between this value and available "
        "PDF annotation evidence; this alone does not indicate the value is "
        "incorrect -- this project has previously confirmed cases (a placeholder "
        "mailto: address, a stale LinkedIn annotation) where the disagreement was "
        "not a real defect.",
}


def build_contact_information_report_entries(evidence) -> List[Dict[str, Any]]:
    """One entry per Contact Information field (always 5, in schema
    order, mirroring ContactInformationEvidenceRecord's own contract),
    or a single module-unavailable placeholder if `evidence` is None."""
    if evidence is None:
        return [_module_unavailable_entry("contact_information")]

    entries = []
    for f in evidence.fields:
        limitations = ["Corroboration is observational evidence, not proof of correctness."]
        if f.unmatched_same_type_annotations:
            limitations.append(
                "Other same-type PDF annotations exist that do not match this value; "
                "see this project's documented false-positive patterns before treating "
                "this as a defect."
            )
        explanation = _CI_EXPLANATIONS.get(f.evidence_summary, f.evidence_summary)
        entries.append(_entry(
            "contact_information", f.field_name,
            {
                "evidence_summary": f.evidence_summary,
                "extracted_values": list(f.extracted_values),
                "visible_text_checked": f.visible_text_checked,
                "visible_text_matches": list(f.visible_text_matches),
                "annotations_checked": f.annotations_checked,
                "annotation_matches": list(f.annotation_matches),
                "unmatched_same_type_annotations": list(f.unmatched_same_type_annotations),
            },
            explanation, limitations,
        ))
    return entries


# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------

_LANG_EXPLANATIONS = {
    "SOURCE_TEXT_PRESENT": "Source evidence was found for this value.",
    "NO_SOURCE_TEXT_EVIDENCE": "No matching source text was found for this value; this alone does not "
        "indicate the value is incorrect.",
    "NO_SOURCE_TEXT_AVAILABLE": "Source text was not available to check this value against.",
}

_LANG_MISATTRIBUTION_LIMITATION = (
    "This check cannot distinguish a value with no textual anchor from a value "
    "whose text appears elsewhere in the document in an unrelated context (semantic "
    "misattribution) -- it only detects whether the word is present or absent."
)


def build_languages_report_entries(evidence) -> List[Dict[str, Any]]:
    """One entry per extracted Languages value, or a single module-
    unavailable placeholder if `evidence` is None, or a single explicit
    "no values extracted" placeholder if the languages array is empty
    (an empty list is itself a complete, correct result, not missing
    data -- see LanguagesEvidenceRecord's own docstring)."""
    if evidence is None:
        return [_module_unavailable_entry("languages")]
    if not evidence.values:
        return [_entry(
            "languages", "(no values extracted)", {"evidence_available": True, "value_count": 0},
            "No languages were extracted for this resume; there is nothing to evaluate.",
        )]

    entries = []
    for i, v in enumerate(evidence.values):
        explanation = _LANG_EXPLANATIONS.get(v.evidence_summary, v.evidence_summary)
        limitations = [_LANG_MISATTRIBUTION_LIMITATION] if v.evidence_summary == "NO_SOURCE_TEXT_EVIDENCE" else []
        entries.append(_entry(
            "languages", f"value[{i}]: {v.value}",
            {
                "value": v.value,
                "evidence_summary": v.evidence_summary,
                "source_text_checked": v.source_text_checked,
                "source_text_present": v.source_text_present,
            },
            explanation, limitations,
        ))
    return entries


# ---------------------------------------------------------------------------
# Professional Summary
# ---------------------------------------------------------------------------

_SUMMARY_EXPLANATIONS = {
    "NO_SUMMARY_EXTRACTED": "No professional summary was extracted; there is nothing to evaluate.",
    "NO_SOURCE_TEXT_AVAILABLE": "Source text was not available to check this value against.",
    "SOURCE_TEXT_VERBATIM_MATCH": "Source evidence was found for this value (a verbatim match in the "
        "source text).",
    "NO_VERBATIM_MATCH_FOUND": "No exact verbatim match was found in the source text; this does not "
        "necessarily indicate unsupported content -- see the token overlap ratio "
        "alongside this state. Ground truth is never used to make this determination.",
}


def build_professional_summary_report_entries(evidence) -> List[Dict[str, Any]]:
    """A single entry (Professional Summary has exactly one value per
    resume, never a list -- see ProfessionalSummaryEvidence's own
    docstring), or a single module-unavailable placeholder if `evidence`
    is None."""
    if evidence is None:
        return [_module_unavailable_entry("professional_summary")]

    explanation = _SUMMARY_EXPLANATIONS.get(evidence.evidence_summary, evidence.evidence_summary)
    limitations = []
    if evidence.evidence_summary == "NO_VERBATIM_MATCH_FOUND":
        limitations.append(
            "This project has previously found this state can arise from a single "
            "formatting/OCR-character difference in otherwise-correct content, not only "
            "from genuinely unsupported content -- the token overlap ratio distinguishes "
            "these two cases better than this state alone."
        )
    limitations.append("Ground truth is never used to produce this evidence.")

    return [_entry(
        "professional_summary", "summary",
        {
            "evidence_summary": evidence.evidence_summary,
            "extracted_value": evidence.extracted_value,
            "source_text_checked": evidence.source_text_checked,
            "whole_document_verbatim_match": evidence.whole_document_verbatim_match,
            "token_overlap_ratio": evidence.token_overlap_ratio,
        },
        explanation, limitations,
    )]


# ---------------------------------------------------------------------------
# Personal Information
# ---------------------------------------------------------------------------

_PI_NAME_EXPLANATIONS = {
    "NO_FULL_NAME_TO_CHECK": "No full name was extracted; there is nothing to check for completeness.",
    "NAME_TOKENS_FULLY_REPRESENTED": "Every substantial word in the extracted full name is represented "
        "in the first/last name split.",
    "NAME_TOKENS_MISSING": "A name-completeness discrepancy was detected; the evidence does not "
        "identify how the name should be corrected.",
}

_PI_LOCATION_EXPLANATIONS = {
    "NO_LOCATION_EXTRACTED": "No location was extracted; there is nothing to evaluate.",
    "NO_SOURCE_TEXT_AVAILABLE": "Source text was not available to check this value against.",
    "SOURCE_TEXT_VERBATIM_MATCH": "Source evidence was found for this value.",
    "NO_VERBATIM_MATCH_FOUND": "No matching source evidence was found for this value; this alone does "
        "not indicate the value is incorrect.",
}

_PI_LOCATION_LIMITATION = (
    "The available evidence cannot distinguish wrong-section attribution -- a real, "
    "verbatim value copied from the wrong part of the resume (e.g. an employer's or "
    "institution's location, not the candidate's own) produces the same "
    "SOURCE_TEXT_VERBATIM_MATCH state as a genuinely correct value. This project has "
    "previously confirmed real cases of exactly this."
)


def build_personal_information_report_entries(evidence) -> List[Dict[str, Any]]:
    """Always exactly two entries when evidence is available
    (name_completeness, location) -- PersonalInformationEvidenceRecord's
    own fixed two-relationship shape (see that dataclass's own
    docstring for why it isn't a list) -- or a single module-unavailable
    placeholder if `evidence` is None."""
    if evidence is None:
        return [_module_unavailable_entry("personal_information")]

    name_ev = evidence.name_completeness
    name_explanation = _PI_NAME_EXPLANATIONS.get(name_ev.evidence_summary, name_ev.evidence_summary)
    name_limitations = [
        "This signal is diagnostic only -- it never implies an automatic correction, "
        "and does not identify which of first_name/last_name should hold a missing token."
    ] if name_ev.evidence_summary != "NO_FULL_NAME_TO_CHECK" else []

    loc_ev = evidence.location
    loc_explanation = _PI_LOCATION_EXPLANATIONS.get(loc_ev.evidence_summary, loc_ev.evidence_summary)
    loc_limitations = [_PI_LOCATION_LIMITATION] if loc_ev.evidence_summary == "SOURCE_TEXT_VERBATIM_MATCH" else []

    return [
        _entry(
            "personal_information", "name_completeness",
            {
                "evidence_summary": name_ev.evidence_summary,
                "full_name": name_ev.full_name,
                "first_name": name_ev.first_name,
                "last_name": name_ev.last_name,
                "checked": name_ev.checked,
                "missing_tokens": list(name_ev.missing_tokens),
            },
            name_explanation, name_limitations,
        ),
        _entry(
            "personal_information", "location",
            {
                "evidence_summary": loc_ev.evidence_summary,
                "extracted_value": loc_ev.extracted_value,
                "source_text_checked": loc_ev.source_text_checked,
                "verbatim_in_source": loc_ev.verbatim_in_source,
            },
            loc_explanation, loc_limitations,
        ),
    ]


# ---------------------------------------------------------------------------
# Work Experience / Education
# ---------------------------------------------------------------------------
#
# Neither module has a single closed evidence_summary label (see the
# module-level docstring's mapping table) -- native_evidence_state for
# these two modules is the RAW per-entry field values themselves
# (locator_matched, match_confidence, degree_evidence,
# institution_evidence, review_priority, validation_actions), never a
# synthesized label. The explanation sentence below is generated purely
# from combinations of these already-existing, already-documented
# fields -- no new evidence state is invented.

_WE_REVIEW_PRIORITY_EXPLANATIONS = {
    "NONE": "Matching source evidence was found for this entry.",
    "ELEVATED": "No matching locator evidence was found, and this entry's company and "
        "designation fields are identical -- a pattern this project has previously "
        "found concentrated in entries that did not correspond to a real job. This "
        "is diagnostic only and does not by itself mean this entry is incorrect.",
    "STANDARD": "No matching locator evidence was found for this entry; this alone does not "
        "indicate the entry is incorrect -- most entries without a locator match in "
        "this project's own corpus were still confirmed correct.",
}

_LOCATOR_UNAVAILABLE_EXPLANATION = (
    "Locator evidence was not available for this entry (for example, no source "
    "markdown was supplied, or the module's target section was not found in this "
    "resume). This is not a statement about correctness."
)


def _validation_actions_note(validation_actions) -> Optional[str]:
    if not validation_actions:
        return None
    rule_ids = ", ".join(a.rule_id for a in validation_actions)
    return (
        f"{len(validation_actions)} field(s) on this entry were automatically "
        f"adjusted by the deterministic validator before evidence generation "
        f"(rule(s): {rule_ids}). This is validation behavior, distinct from the "
        f"observational evidence reported here."
    )


def build_work_experience_report_entries(evidence) -> List[Dict[str, Any]]:
    """One entry per Work Experience array entry, or a single module-
    unavailable placeholder if `evidence` is None, or a single explicit
    "no entries extracted" placeholder if the array itself is empty."""
    if evidence is None:
        return [_module_unavailable_entry("work_experience")]
    if not evidence.entries:
        return [_entry(
            "work_experience", "(no entries extracted)", {"evidence_available": True, "entry_count": 0},
            "No work experience entries were extracted for this resume; there is nothing to evaluate.",
        )]

    entries = []
    for e in evidence.entries:
        if e.locator_matched is None:
            explanation = _LOCATOR_UNAVAILABLE_EXPLANATION
        elif e.review_priority is not None:
            explanation = _WE_REVIEW_PRIORITY_EXPLANATIONS[e.review_priority]
        else:
            explanation = (
                "Matching source evidence was found for this entry."
                if e.locator_matched else
                "No matching locator evidence was found for this entry; this alone does "
                "not indicate the entry is incorrect."
            )
        limitations = [
            "Absence of a locator match is not, by itself, evidence that this entry is incorrect."
        ]
        note = _validation_actions_note(e.validation_actions)
        if note:
            limitations.append(note)
        entries.append(_entry(
            "work_experience", f"entry[{e.entry_index}]",
            {
                "locator_matched": e.locator_matched,
                "match_confidence": e.match_confidence,
                "company_equals_designation": e.company_equals_designation,
                "review_priority": e.review_priority,
                "validation_action_count": len(e.validation_actions),
            },
            explanation, limitations,
        ))
    return entries


def build_education_report_entries(evidence) -> List[Dict[str, Any]]:
    """One entry per Education array entry, or a single module-
    unavailable placeholder if `evidence` is None, or a single explicit
    "no entries extracted" placeholder if the array itself is empty."""
    if evidence is None:
        return [_module_unavailable_entry("education")]
    if not evidence.entries:
        return [_entry(
            "education", "(no entries extracted)", {"evidence_available": True, "entry_count": 0},
            "No education entries were extracted for this resume; there is nothing to evaluate.",
        )]

    entries = []
    for e in evidence.entries:
        if e.locator_matched is None:
            explanation = _LOCATOR_UNAVAILABLE_EXPLANATION
        elif e.locator_matched:
            explanation = "Matching source evidence was found for this entry (native boolean signal, not a numeric score)."
        else:
            explanation = (
                "No matching locator evidence was found for this entry; this alone does "
                "not indicate the entry is incorrect."
            )
        limitations = [
            "Education has no numeric confidence signal -- degree_evidence/"
            "institution_evidence are native booleans, never coerced into a score.",
            "Absence of a locator match is not, by itself, evidence that this entry is incorrect.",
        ]
        note = _validation_actions_note(e.validation_actions)
        if note:
            limitations.append(note)
        entries.append(_entry(
            "education", f"entry[{e.entry_index}]",
            {
                "locator_matched": e.locator_matched,
                "degree_evidence": e.degree_evidence,
                "institution_evidence": e.institution_evidence,
                "validation_action_count": len(e.validation_actions),
            },
            explanation, limitations,
        ))
    return entries


# ---------------------------------------------------------------------------
# Top-level report assembly
# ---------------------------------------------------------------------------

_MODULE_BUILDERS = {
    "personal_information": lambda r: build_personal_information_report_entries(r.personal_information_evidence),
    "contact_information": lambda r: build_contact_information_report_entries(r.contact_information_evidence),
    "professional_summary": lambda r: build_professional_summary_report_entries(r.professional_summary_evidence),
    "languages": lambda r: build_languages_report_entries(r.languages_evidence),
    "work_experience": lambda r: build_work_experience_report_entries(r.work_experience_evidence),
    "education": lambda r: build_education_report_entries(r.education_evidence),
}


def build_resume_evidence_report(resume_id: str, stem: str, result) -> Dict[str, Any]:
    """Build the full, per-resume human-review evidence report from an
    ALREADY-BUILT ParserPipelineResult -- this function never runs the
    pipeline itself, never calls an LLM, and never reads or writes
    `result` or any of its evidence objects except through plain
    attribute access (no mutation anywhere in this module -- see
    tests/test_human_review_evidence_report.py's own mutation-safety
    proof).

    Args:
        resume_id: Caller-supplied identifier for this resume (e.g. the
            mapping.csv ResumeID, or any other stable string).
        stem: Caller-supplied filename stem for this resume.
        result: An already-produced ParserPipelineResult (from
            ParserPipeline.run(), already completed).

    Returns:
        A plain, JSON-serializable dict:
            {
                "resume_id": ..., "stem": ...,
                "modules": {
                    "personal_information": [...entries...],
                    "contact_information": [...],
                    "professional_summary": [...],
                    "languages": [...],
                    "work_experience": [...],
                    "education": [...],
                },
            }
        Skills is never a key here -- see this module's own top
        docstring for why.
    """
    modules: Dict[str, List[Dict[str, Any]]] = {}
    for module_key in REPORT_MODULES:
        modules[module_key] = _MODULE_BUILDERS[module_key](result)

    return {
        "resume_id": resume_id,
        "stem": stem,
        "modules": modules,
    }


def report_to_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten one resume's report into plain, CSV-ready rows -- one row
    per (module, item_id), mirroring evaluation/contact_information_
    evidence_report.py's own evidence_records_to_field_rows() shape.
    Pure -- never mutates `report`."""
    rows: List[Dict[str, Any]] = []
    for module_key, entries in report["modules"].items():
        for e in entries:
            rows.append({
                "resume_id": report["resume_id"],
                "stem": report["stem"],
                "module": e["module"],
                "item_id": e["item_id"],
                "native_evidence_state": repr(e["native_evidence_state"]),
                "explanation": e["explanation"],
                "limitation_count": len(e["limitations"]),
                "limitations": " | ".join(e["limitations"]),
            })
    return rows
