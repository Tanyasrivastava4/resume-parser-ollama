"""
services/evidence_record.py

Common Evidence Record (Milestone V2.7).

An additive, standalone, purely OBSERVATIONAL evidence-recording layer
for Work Experience, Education, and (Milestone V3.9) Contact Information.

build_education_evidence() and build_work_experience_evidence() are NOT
wired into pipeline/parser_pipeline.py and do not run during normal
pipeline execution — each is called, if at all, AFTER a resume has
already been parsed and validated, to build a read-only summary of the
evidence that already exists in that run's own outputs (schema
validation result, locator output, ValidationReport issues, Stage-1
route). Neither re-derives, re-scores, or re-interprets anything; every
field here is a direct pass-through of a fact some other, already-
existing, already-tested component already produced.

build_contact_information_evidence() (Milestone V3.8) IS wired into
pipeline/parser_pipeline.py as of Milestone V3.9 — see that module's own
docstring for the exact seam. This is a deliberate, narrow exception, not
a reversal of the "unwired" design above: Contact Information has no
existing validator or locator to call instead, and V3.9's own corpus
regression proof (docs/MILESTONE_V3_9_CONTACT_INFORMATION_EVIDENCE_
INTEGRATION.md) demonstrates the wiring is genuinely additive — the
pipeline's returned contact_information value is provably unchanged, and
the evidence record is exposed only as a new, separate, optional
ParserPipelineResult field, never consumed by ParserMerger or
SchemaValidator.

This module deliberately does NOT:
  - compute or store a single normalized confidence score (Milestone
    V2.7's own corpus investigation, docs/
    MILESTONE_V2_7_EVIDENCE_FRAMEWORK_CALIBRATION.md Step "Evaluation",
    found Work Experience's match confidence is a genuine continuous
    signal while Education has no numeric signal at all -- only
    categorical/boolean evidence. Forcing both into one shared 0-1
    scale would misrepresent Education's evidence as more precise than
    it is. See that milestone's own decision for the full reasoning.);
  - decide ACCEPT/REJECT/REPAIR/ESCALATE for any entry;
  - modify, remove, or repair any field of any entry;
  - re-run or duplicate any locator/validator logic -- every value here
    comes from calling the EXISTING WorkExperienceLocator/
    EducationLocator/*Validator classes exactly as they already are.

What IS common across both modules, and is what this module actually
standardizes, is the SHAPE of the record (which facts are captured,
under which names) -- not the values' own representation, which stays
module-native (a float for Work Experience's match confidence, booleans
for Education's presence flags) rather than being coerced into a shared
type.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from services.education_locator import EducationLocator
from services.education_validator import EducationValidator
from services.work_experience_locator import WorkExperienceLocator
from services.work_experience_validator import WorkExperienceValidator


@dataclass
class ValidationActionRecord:
    """One validation rule's recorded outcome for one entry -- a direct,
    unmodified pass-through of a ValidationIssue (from either
    EducationValidator or WorkExperienceValidator; both already share
    this exact shape by design, per each validator's own docstring).

    Attributes:
        rule_id: The WEV-xxx/EDV-xxx rule that fired.
        severity: As recorded by the originating validator.
        auto_fixed: Whether the rule corrected the entry (True) or was
            informational only (False, e.g. EDV-012).
        message: The originating validator's own message, unmodified.
    """

    rule_id: str
    severity: str
    auto_fixed: bool
    message: str


@dataclass
class EntryEvidence:
    """Per-entry evidence for one module's one array entry. Fields that
    don't apply to a given module (e.g. match_confidence for Education,
    which has no numeric signal) are left None -- never defaulted to a
    fabricated 0.0 or False, since that would misrepresent "this signal
    doesn't exist for this module" as "this signal was checked and came
    back empty."

    Attributes:
        entry_index: Index within the module's own array.
        locator_matched: Whether the module's locator found ANY
            corroborating span/candidate for this entry. None if the
            module has no locator concept at all, or if the locator
            never engaged with this resume (e.g. section_found=False)
            -- in which case this is explicitly None, not False, to
            preserve the "absence is not contradiction" distinction
            this milestone's own investigation confirmed is mandatory
            for both modules (locator absence correlates with 63-71%
            GT-correctness in the real corpus, not 0% -- see the
            milestone report's Evaluation section).
        match_confidence: Work Experience's own numeric [0, 1] score
            from WorkExperienceLocator.match_jobs_to_spans(), when a
            match exists. Always None for Education, which has no
            equivalent numeric signal -- NOT coerced to a float.
        degree_evidence / institution_evidence: Education's own boolean
            presence flags for the matched candidate (see
            EducationCandidate), when a match exists. Always None for
            Work Experience.
        validation_actions: Every ValidationActionRecord for this
            specific entry_index, in the order the validator applied
            them (empty list if none fired).
        company_equals_designation: Milestone V2.9, Work Experience
            only (always None for Education). Whether this entry's own
            `company` and `designation` fields are identical (case/
            whitespace-normalized) -- a deterministic, GT-independent
            structural check on already-extracted field values, not a
            new locator/validator signal. See review_priority's own
            docstring for why this is informative and its measured
            limitation.
        review_priority: Milestone V2.9, Work Experience only (always
            None for Education this milestone -- explicitly out of
            scope, see docs/MILESTONE_V2_9_WE_REVIEW_PRIORITIZATION.md).
            One of "NONE", "STANDARD", or "ELEVATED". This is NOT a
            correctness judgment, NOT a confidence score, and NEVER
            implies an entry should be rejected, removed, or repaired
            -- it exists only to help a human reviewer decide what to
            look at first. Computed purely from already-existing
            evidence (locator_matched, company_equals_designation), no
            new signal invented:
              - "NONE": locator_matched is True (a confident span match
                exists for this job).
              - "ELEVATED": locator_matched is not True AND
                company_equals_designation is True -- in this project's
                own 85-entry Work Experience corpus, this exact
                combination occurred 7 times, all 7 on entries that
                turned out NOT to correspond to any real GT job (100%
                precision) -- but all 7 came from a single resume
                (Krishankant Soni_GAD-UK_GT-Intileo), so this is a
                narrowly-validated, single-document pattern, not a
                broadly-confirmed one. Treat as "worth a closer look,"
                not as proven.
              - "STANDARD": locator_matched is not True and
                company_equals_designation is not True -- the majority
                (23/30) of this corpus's no-match population, measured
                at 87.0% GT-correct -- only modestly below this
                corpus's own overall baseline (93.3%). A real but weak
                signal; explicitly NOT a basis for rejection.
    """

    entry_index: int
    locator_matched: Optional[bool] = None
    match_confidence: Optional[float] = None
    degree_evidence: Optional[bool] = None
    institution_evidence: Optional[bool] = None
    company_equals_designation: Optional[bool] = None
    review_priority: Optional[str] = None
    validation_actions: List[ValidationActionRecord] = field(default_factory=list)


@dataclass
class ModuleEvidenceRecord:
    """The full observational evidence record for one module's output on
    one resume.

    Attributes:
        module: "work_experience" or "education".
        schema_valid: Whether this module's JSON passed
            SchemaValidator.validate() -- structural only, matching
            that validator's own documented scope.
        stage1_route: The document-level route
            (Stage1Router.ROUTE_NATIVE / ROUTE_OCR_FALLBACK /
            ROUTE_COLUMN_AWARE_NATIVE), when known. None if not
            supplied by the caller (this module never calls
            Stage1Router itself -- see the module docstring's "reuse,
            don't duplicate" principle).
        locator_section_found: Whether the module's locator found its
            target section at all. None for a module with no locator
            concept.
        entries: Per-entry evidence, in the same order as the module's
            own array.
    """

    module: str
    schema_valid: Optional[bool]
    stage1_route: Optional[str]
    locator_section_found: Optional[bool]
    entries: List[EntryEvidence] = field(default_factory=list)


def _issue_to_action(issue: Any) -> ValidationActionRecord:
    """Convert a WEV/EDV ValidationIssue (both share this exact shape)
    into a ValidationActionRecord -- a direct field copy, nothing
    computed."""
    return ValidationActionRecord(
        rule_id=issue.rule_id,
        severity=issue.severity,
        auto_fixed=issue.auto_fixed,
        message=issue.message,
    )


# Milestone V2.9: review-priority states. Not a numeric score -- three
# plain labels, each with a fully-stated, deterministic rationale (see
# EntryEvidence.review_priority's own docstring for the exact evidence
# and measured precision/recall behind each one,
# docs/MILESTONE_V2_9_WE_REVIEW_PRIORITIZATION.md for the full
# investigation, and benchmark/baselines/v2_9_review_priority_simulation.csv
# for the underlying numbers).
_WE_REVIEW_PRIORITY_NONE = "NONE"
_WE_REVIEW_PRIORITY_STANDARD = "STANDARD"
_WE_REVIEW_PRIORITY_ELEVATED = "ELEVATED"


def _company_equals_designation(entry: Dict[str, Any]) -> bool:
    """Milestone V2.9: whether this Work Experience entry's own
    `company` and `designation` fields are identical after normalizing
    case and surrounding whitespace. A real employer/role pair is very
    unlikely to have byte-identical company and designation text; in
    this project's own corpus, every real occurrence of this pattern
    (7 of 85 entries, all from Krishankant Soni_GAD-UK_GT-Intileo) was
    an entry that did not correspond to any real GT job -- the model
    appears to have extracted a single client/engagement-domain label
    into both fields when it had no genuine separate designation.
    A purely deterministic string comparison on already-extracted
    field values -- no locator call, no new signal invented."""
    company = entry.get("company")
    designation = entry.get("designation")
    if not isinstance(company, str) or not company.strip():
        return False
    if not isinstance(designation, str) or not designation.strip():
        return False
    return company.strip().lower() == designation.strip().lower()


def _we_review_priority(
    locator_matched: Optional[bool], company_equals_designation: bool
) -> Optional[str]:
    """Milestone V2.9: combine two already-computed, existing evidence
    signals into one interpretable label. Returns None only when
    locator_matched itself is None (the locator never engaged with
    this resume at all -- e.g. no markdown was supplied), since there
    is then no evidence basis to assign any priority at all, matching
    this module's own "never guess when a signal is unavailable"
    convention elsewhere.

    This function makes no correctness judgment and never implies an
    entry should be rejected, removed, or repaired -- see
    EntryEvidence.review_priority's own docstring for the full,
    corpus-measured rationale behind each label."""
    if locator_matched is None:
        return None
    if locator_matched:
        return _WE_REVIEW_PRIORITY_NONE
    if company_equals_designation:
        return _WE_REVIEW_PRIORITY_ELEVATED
    return _WE_REVIEW_PRIORITY_STANDARD


def build_education_evidence(
    education_entries: List[Dict[str, Any]],
    markdown_text: Optional[str],
    validation_report: Any,
    schema_valid: Optional[bool] = None,
    stage1_route: Optional[str] = None,
    locator: Optional[EducationLocator] = None,
) -> ModuleEvidenceRecord:
    """Build an observational evidence record for one resume's already-
    produced Education output.

    Args:
        education_entries: The (already validated) education array.
        markdown_text: The resume's Stage-1 markdown, or None if
            unavailable -- when None, locator_section_found and every
            entry's locator_matched are left None (never guessed).
        validation_report: The EducationValidator.ValidationReport
            already produced for this resume (from
            EducationValidator.validate() -- this function does not
            call validate() itself, it only reads its already-produced
            report, so it can never change what the validator did).
        schema_valid: Whether this module's JSON passed schema
            validation, if known.
        stage1_route: The document's Stage1Router route, if known.
        locator: Defaults to a new EducationLocator() -- only used to
            RE-READ (not re-validate) candidate spans for the
            per-entry locator_matched/degree_evidence/
            institution_evidence fields, exactly mirroring what EDV-012
            itself already checks (see education_validator.py) -- not
            a second, different algorithm.

    Returns:
        A ModuleEvidenceRecord. Never mutates education_entries,
        markdown_text, or validation_report.
    """
    locator = locator or EducationLocator()

    entries: List[EntryEvidence] = []
    section_found: Optional[bool] = None
    result = None
    if markdown_text:
        result = locator.locate(markdown_text)
        section_found = result.section_found

    actions_by_entry: Dict[int, List[ValidationActionRecord]] = {}
    if validation_report is not None:
        for issue in validation_report.issues:
            actions_by_entry.setdefault(issue.entry_index, []).append(_issue_to_action(issue))

    for index, entry in enumerate(education_entries):
        locator_matched: Optional[bool] = None
        degree_evidence: Optional[bool] = None
        institution_evidence: Optional[bool] = None

        if result is not None and result.section_found:
            locator_matched = False
            degree = entry.get("degree")
            institution = entry.get("institution")
            for candidate in result.candidates:
                degree_found = (
                    EducationValidator._edv_012_present_in(degree, candidate.raw_text)
                    if isinstance(degree, str) and degree.strip()
                    else False
                )
                institution_found = (
                    EducationValidator._edv_012_present_in(institution, candidate.raw_text)
                    if isinstance(institution, str) and institution.strip()
                    else True
                )
                if degree_found and institution_found:
                    locator_matched = True
                    degree_evidence = True
                    institution_evidence = (
                        True if isinstance(institution, str) and institution.strip() else None
                    )
                    break

        entries.append(
            EntryEvidence(
                entry_index=index,
                locator_matched=locator_matched,
                match_confidence=None,  # Education has no numeric signal -- see module docstring.
                degree_evidence=degree_evidence,
                institution_evidence=institution_evidence,
                validation_actions=actions_by_entry.get(index, []),
            )
        )

    return ModuleEvidenceRecord(
        module="education",
        schema_valid=schema_valid,
        stage1_route=stage1_route,
        locator_section_found=section_found,
        entries=entries,
    )


def build_work_experience_evidence(
    work_experience_entries: List[Dict[str, Any]],
    markdown_text: Optional[str],
    validation_report: Any,
    schema_valid: Optional[bool] = None,
    stage1_route: Optional[str] = None,
    locator: Optional[WorkExperienceLocator] = None,
) -> ModuleEvidenceRecord:
    """Build an observational evidence record for one resume's already-
    produced Work Experience output. Mirrors build_education_evidence's
    own shape and read-only contract exactly; see that function's
    docstring for the shared design principles.

    Args:
        work_experience_entries: The (already validated) work_experience
            array.
        markdown_text: The resume's Stage-1 markdown, or None.
        validation_report: The WorkExperienceValidator.ValidationReport
            already produced for this resume.
        schema_valid: Whether this module's JSON passed schema
            validation, if known.
        stage1_route: The document's Stage1Router route, if known.
        locator: Defaults to a new WorkExperienceLocator() -- only used
            to re-locate spans and re-run match_jobs_to_spans() (the
            existing, unmodified matching algorithm) for the per-entry
            match_confidence field.
    """
    locator = locator or WorkExperienceLocator()

    section_found: Optional[bool] = None
    matches: Dict[int, Any] = {}
    if markdown_text:
        spans = locator.locate(markdown_text)
        section_found = len(spans) > 0
        if spans:
            matches = locator.match_jobs_to_spans(work_experience_entries, spans)

    actions_by_entry: Dict[int, List[ValidationActionRecord]] = {}
    if validation_report is not None:
        for issue in validation_report.issues:
            actions_by_entry.setdefault(issue.job_index, []).append(_issue_to_action(issue))

    entries: List[EntryEvidence] = []
    for index, entry in enumerate(work_experience_entries):
        locator_matched: Optional[bool] = None
        match_confidence: Optional[float] = None
        if section_found is not None:
            locator_matched = index in matches
            if locator_matched:
                match_confidence = matches[index].confidence

        company_equals_designation = _company_equals_designation(entry)
        review_priority = _we_review_priority(locator_matched, company_equals_designation)

        entries.append(
            EntryEvidence(
                entry_index=index,
                locator_matched=locator_matched,
                match_confidence=match_confidence,
                degree_evidence=None,  # Work Experience has no equivalent -- see module docstring.
                institution_evidence=None,
                company_equals_designation=company_equals_designation,
                review_priority=review_priority,
                validation_actions=actions_by_entry.get(index, []),
            )
        )

    return ModuleEvidenceRecord(
        module="work_experience",
        schema_valid=schema_valid,
        stage1_route=stage1_route,
        locator_section_found=section_found,
        entries=entries,
    )


# ---------------------------------------------------------------------------
# Contact Information (Milestone V3.8)
# ---------------------------------------------------------------------------
#
# Contact Information's own schema (schemas/modules/contact_information_
# schema.json) is a flat object of 5 fields (email, phone as arrays;
# linkedin, github, portfolio as string-or-null) -- not an array of
# entries the way Education and Work Experience are. ModuleEvidenceRecord/
# EntryEvidence's entry_index-based shape does not fit that (there is no
# "entry" to index), so this section deliberately does NOT reuse those two
# dataclasses -- forcing a fake entry_index onto a fixed set of scalar/
# array fields would misrepresent the data's own shape, the same
# "reuse only where it fits" judgment this module already made once
# before (Work Experience's match_confidence vs Education's boolean
# evidence, see the module docstring above). What IS reused: the review_
# priority-style closed-label-set pattern (ContactFieldEvidence.
# evidence_summary below), and this project's now-established V3.6/V3.7
# corpus findings, cited inline below.
#
# Milestone V3.6 found no ContactInformationLocator/Validator exists.
# Milestone V3.7 investigated PDF hyperlink annotations (via PyMuPDF's
# page.get_links(), the same primitive services/native_stage1_extractor.
# py's HyperlinkAnnotation already uses) and reached CONDITIONAL GO for
# annotations as OBSERVATIONAL evidence only -- explicitly NOT for
# merging into Stage-1 input (Option 3, NO-GO), and explicitly not for
# any value modification, since it found concrete false-positive cases:
# a PDF whose mailto: destination was an uncustomized template placeholder
# ("x@gmail.com") coexisting with a correct visible-text email, and a PDF
# whose linkedin hyperlink destination was itself stale/wrong while the
# visible text, extraction, and GT all agreed on a different (correct)
# value. Both are proof that annotation content must never override an
# already-extracted value -- see build_contact_information_evidence's
# own docstring for how this milestone's design responds to that finding.

# Must stay in sync with the exact literal
# services/native_stage1_extractor.py's _write_output() writes when it
# appends undiscovered hyperlinks to a resume's Stage-1 markdown (see
# that method's docstring). This module does not import that literal as
# a shared constant to avoid coupling an additive, standalone evidence
# module to a change in NativeStage1Extractor's internals for a single
# string -- tests/test_evidence_record.py directly verifies this constant
# against a real, already-produced corpus markdown file so any drift is
# caught, not silently ignored.
_HYPERLINK_ANNOTATION_FOOTER_MARKER = "--- Embedded Hyperlink Annotations "

_CONTACT_INFO_FIELD_ORDER = ("email", "phone", "linkedin", "github", "portfolio")

# Closed set of evidence_summary labels -- deliberately not a numeric
# score (see this module's own top docstring and Milestone V2.7's
# decision, which this milestone follows exactly for Contact
# Information too).
_CI_NO_EXTRACTED_VALUE = "NO_EXTRACTED_VALUE"
_CI_NO_EVIDENCE_SOURCES_AVAILABLE = "NO_EVIDENCE_SOURCES_AVAILABLE"
_CI_FULLY_CORROBORATED = "FULLY_CORROBORATED"
_CI_PARTIALLY_CORROBORATED = "PARTIALLY_CORROBORATED"
_CI_NOT_CORROBORATED = "NOT_CORROBORATED"
_CI_EVIDENCE_DISAGREEMENT = "EVIDENCE_DISAGREEMENT"


@dataclass
class ContactFieldEvidence:
    """Observational evidence for one Contact Information field
    (email/phone/linkedin/github/portfolio) on one resume.

    Attributes:
        field_name: One of "email", "phone", "linkedin", "github",
            "portfolio" -- always one of these 5, always present in
            ContactInformationEvidenceRecord.fields even when this
            field has no extracted value.
        extracted_values: The field's own already-extracted value(s),
            copied verbatim -- never re-typed, re-ordered, or mutated.
            Always a list for uniform handling: a scalar field's single
            string (linkedin/github/portfolio) becomes a 1-item list,
            None becomes []; an array field's (email/phone) own list is
            copied as-is.
        visible_text_checked: True if Stage-1 markdown was available
            and searched (excluding the "Embedded Hyperlink
            Annotations" footer itself, which is annotation-derived,
            not visible printed text) -- None if no markdown was
            supplied. Never False: either the text was available to
            check, or it wasn't.
        visible_text_matches: Subset of extracted_values confirmed
            present in the visible body text (case/whitespace-
            insensitive for URLs' scheme/www/trailing-slash; digit-only
            for phone). Empty if visible_text_checked is not True, or
            if checked but nothing matched -- these two empty-list
            cases are NOT the same thing, which is exactly why
            visible_text_checked is tracked separately rather than
            being inferred from list emptiness.
        annotations_checked: True if a PDF was supplied and PyMuPDF
            successfully opened it -- None if no PDF was supplied or it
            could not be opened (never treated as "no annotations
            exist," per the explicit safety rule against treating
            evidence-access failure as evidence of absence).
        annotation_matches: Subset of extracted_values confirmed
            present among same-type hyperlink annotation destinations
            (mailto: for email, tel: for phone, linkedin.com/lnkd.in
            for linkedin, a bare github.com/<user> profile-root path
            for github -- deeper github.com/<user>/<repo> paths are
            excluded per V3.7's confirmed finding that those are
            Projects-section links, not profile evidence; portfolio has
            no independent domain signal per V3.6/V3.7, so it is only
            ever matched, never used for disagreement -- see
            evidence_summary).
        unmatched_same_type_annotations: Same-type annotation
            destinations that do NOT match any extracted_values,
            recorded exactly as found -- this list is always populated
            when such annotations exist, regardless of whether
            annotation_matches is also non-empty (evidence is never
            hidden), but it only escalates evidence_summary to
            EVIDENCE_DISAGREEMENT when annotation_matches AND
            visible_text_matches are BOTH empty for this field (see
            _ci_evidence_summary). When at least one same-type
            annotation already corroborates the field (e.g. Milestone
            V3.7's resume 005, which had 1 correct email annotation and
            2 unrelated References-section email annotations), the
            other same-type annotations are shown here but are not
            treated as a disagreement, since treating them as one would
            have been a demonstrated false alarm.
        evidence_summary: One of NO_EXTRACTED_VALUE,
            NO_EVIDENCE_SOURCES_AVAILABLE, FULLY_CORROBORATED,
            PARTIALLY_CORROBORATED, NOT_CORROBORATED, or
            EVIDENCE_DISAGREEMENT. NEVER a numeric score, and NEVER an
            ACCEPT/REJECT/REPAIR instruction to any caller -- this
            field exists to help a human reviewer or a future validator
            decide what to look at, exactly as WE's review_priority
            already does; it must not be used to drop, add, or alter
            any value in the parser's own output.
    """

    field_name: str
    extracted_values: List[str]
    visible_text_checked: Optional[bool]
    visible_text_matches: List[str]
    annotations_checked: Optional[bool]
    annotation_matches: List[str]
    unmatched_same_type_annotations: List[str]
    evidence_summary: str


@dataclass
class ContactInformationEvidenceRecord:
    """The full observational evidence record for one resume's already-
    produced Contact Information output. See the module-level comment
    above this class for why this does not reuse ModuleEvidenceRecord/
    EntryEvidence.

    Attributes:
        schema_valid: Whether this module's JSON passed
            SchemaValidator.validate(), if known.
        stage1_route: The document's Stage1Router route, if known. This
            function never calls Stage1Router itself.
        fields: Exactly 5 ContactFieldEvidence records, always in
            schema field order (email, phone, linkedin, github,
            portfolio), regardless of which fields have a value.
    """

    schema_valid: Optional[bool]
    stage1_route: Optional[str]
    fields: List[ContactFieldEvidence] = field(default_factory=list)


def _ci_normalize_url(value: str) -> str:
    """Strip scheme, leading 'www.', and a trailing slash for comparison
    purposes only -- never used to alter a stored value. Safe/lossless
    per V3.6's corpus-verified finding (0 false positives across the 4
    confirmed same-profile formatting-difference cases)."""
    v = value.strip().lower()
    v = re.sub(r"^https?://(www\.)?", "", v)
    v = re.sub(r"^www\.", "", v)
    return v.rstrip("/")


def _ci_normalize_digits(value: str) -> str:
    """Strip every non-digit character -- safe/lossless per V3.6's
    corpus-verified phone-comparison finding. Phone only -- see
    _ci_normalize_email, not this function, for email comparison."""
    return re.sub(r"\D", "", value or "")


def _ci_normalize_email(value: str) -> str:
    """Case/whitespace-insensitive comparison only -- an email address
    is compared as a string, never digit-stripped (digit-stripping an
    email would compare only its incidental digit characters, e.g.
    'ankushmehta243@gmail.com' -> '243', which is not a meaningful or
    safe comparison and both false-matches on unrelated values sharing
    digits and false-misses on any digit-free email)."""
    return (value or "").strip().lower()


def _ci_normalize_for_field(field_name: str, value: str) -> str:
    """Dispatch to the correct comparison-only normalization for a given
    Contact Information field. Never used to alter a stored value."""
    if field_name in ("linkedin", "github", "portfolio"):
        return _ci_normalize_url(value)
    if field_name == "phone":
        return _ci_normalize_digits(value)
    return _ci_normalize_email(value)  # field_name == "email"


def _ci_as_list(value: Any) -> List[str]:
    """Normalize a schema field's own value into a list of strings for
    uniform handling, without altering the underlying stored value this
    function is only ever given a read of. None/[] -> []; a scalar
    string -> [string]; a list -> copied as-is (only str items kept)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str) and v.strip()]
    return []


def _ci_visible_body_text(markdown_text: Optional[str]) -> Optional[str]:
    """The resume's own visible, printed text -- excludes the "Embedded
    Hyperlink Annotations" footer (that block is itself annotation-
    derived, not visible printed text; see V3.6/V3.7's own explicit
    distinction). Returns None if no markdown was supplied at all
    (never an empty string standing in for "unavailable")."""
    if markdown_text is None:
        return None
    body, _, _ = markdown_text.partition(_HYPERLINK_ANNOTATION_FOOTER_MARKER)
    return body


def _ci_extract_pdf_hyperlinks(pdf_path: Optional[Path]) -> Optional[List[Tuple[int, str]]]:
    """Best-effort, read-only PDF hyperlink-annotation extraction via
    PyMuPDF's page.get_links() -- the same primitive services/
    native_stage1_extractor.py's HyperlinkAnnotation already uses (see
    that module's own docstring for why fitz is imported deferred, not
    at module load time; this function follows the same convention).

    Returns None (never an empty list standing in for "checked, found
    nothing") if pdf_path is None, the file does not exist, or PyMuPDF
    cannot open it -- per the explicit safety rule against treating
    evidence-access failure as evidence of absence. Never writes
    anything, never re-runs Stage-1 extraction, never touches
    NativeStage1Extractor -- this is a narrow, standalone read of the
    PDF's own link annotations only.
    """
    if pdf_path is None:
        return None
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return None
    try:
        import fitz  # PyMuPDF -- deferred import, matches the codebase's
        # existing convention (see native_stage1_extractor.py,
        # stage1_router.py, column_aware_stage1_extractor.py).
    except ImportError:
        return None

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return None

    try:
        links: List[Tuple[int, str]] = []
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            for link in page.get_links():
                if link.get("kind") == fitz.LINK_URI:
                    uri = link.get("uri", "")
                    if uri:
                        links.append((page_index + 1, uri))
        return links
    except Exception:
        return None
    finally:
        doc.close()


_GITHUB_PROFILE_ROOT_RE = re.compile(r"github\.com/([^/?#]+)/?$", re.IGNORECASE)


def _ci_same_type_annotation_values(
    field_name: str, annotation_uris: List[str]
) -> List[str]:
    """Filter raw annotation destination URIs down to the ones plausibly
    ABOUT this specific field, using only hard structural signals
    (URI scheme/domain/path shape) -- never anchor text or field-name
    guessing, per V3.7's explicit "inspect the real evidence" finding.

    email: mailto: scheme -> the address itself.
    phone: tel: scheme -> the number itself.
    linkedin: linkedin.com or lnkd.in domain -> the URI as-is.
    github: a BARE github.com/<user> profile-root path only -- a
        deeper path (github.com/<user>/<repo>) is excluded entirely,
        per V3.7's confirmed finding that those are Projects-section
        repo links, not profile evidence.
    portfolio: no independent structural signal exists (V3.6/V3.7 both
        found no reliable domain pattern) -- returns [] always; a
        portfolio value can still be corroborated by an exact/
        normalized match against ANY annotation URI (see
        _ci_build_field_evidence), just never used to flag a
        disagreement, since there is no reliable way to tell whether a
        non-matching URL was ever claiming to be a portfolio link at
        all.
    """
    if field_name == "email":
        return [u[len("mailto:"):] for u in annotation_uris if u.lower().startswith("mailto:")]
    if field_name == "phone":
        return [u[len("tel:"):] for u in annotation_uris if u.lower().startswith("tel:")]
    if field_name == "linkedin":
        return [u for u in annotation_uris if "linkedin.com" in u.lower() or "lnkd.in" in u.lower()]
    if field_name == "github":
        return [u for u in annotation_uris if _GITHUB_PROFILE_ROOT_RE.search(u)]
    return []  # portfolio: see docstring.


def _ci_evidence_summary(
    extracted_values: List[str],
    visible_text_checked: Optional[bool],
    visible_text_matches: List[str],
    annotations_checked: Optional[bool],
    annotation_matches: List[str],
    unmatched_same_type_annotations: List[str],
) -> str:
    """Deterministic, closed-set label -- see ContactFieldEvidence.
    evidence_summary's own docstring for the full definition of each
    label and why EVIDENCE_DISAGREEMENT only fires when there is
    literally no corroboration from either source."""
    if not extracted_values:
        return _CI_NO_EXTRACTED_VALUE

    if not visible_text_checked and not annotations_checked:
        return _CI_NO_EVIDENCE_SOURCES_AVAILABLE

    corroborated = set(visible_text_matches) | set(annotation_matches)
    n_corroborated = sum(1 for v in extracted_values if v in corroborated)

    if n_corroborated == len(extracted_values):
        return _CI_FULLY_CORROBORATED
    if n_corroborated > 0:
        return _CI_PARTIALLY_CORROBORATED
    if unmatched_same_type_annotations:
        return _CI_EVIDENCE_DISAGREEMENT
    return _CI_NOT_CORROBORATED


def _ci_build_field_evidence(
    field_name: str,
    extracted_values: List[str],
    visible_body_text: Optional[str],
    annotation_uris: Optional[List[str]],
) -> ContactFieldEvidence:
    """Build one field's ContactFieldEvidence. Pure -- never mutates any
    argument. `annotation_uris` is the FULL raw list of every hyperlink
    URI found anywhere in the PDF (all fields), from which this function
    itself filters same-type candidates -- callers should not pre-filter."""
    visible_text_checked: Optional[bool] = None
    visible_text_matches: List[str] = []
    if visible_body_text is not None:
        visible_text_checked = True
        if field_name == "phone":
            body_haystack = _ci_normalize_digits(visible_body_text)
        else:
            body_haystack = re.sub(r"\s+", " ", visible_body_text).lower()
        for v in extracted_values:
            needle = _ci_normalize_for_field(field_name, v)
            if needle and needle in body_haystack:
                visible_text_matches.append(v)

    annotations_checked: Optional[bool] = None
    annotation_matches: List[str] = []
    unmatched_same_type_annotations: List[str] = []
    if annotation_uris is not None:
        annotations_checked = True
        same_type = _ci_same_type_annotation_values(field_name, annotation_uris)
        # Portfolio has no same-type filter (see docstring) but can still
        # be corroborated by an exact match against any raw annotation URI.
        candidates = same_type if field_name != "portfolio" else annotation_uris
        norm_extracted = {_ci_normalize_for_field(field_name, v): v for v in extracted_values}
        matched_original_values = set()
        for uri in candidates:
            key = _ci_normalize_for_field(field_name, uri)
            if key and key in norm_extracted:
                matched_original_values.add(norm_extracted[key])
            elif field_name != "portfolio":
                unmatched_same_type_annotations.append(uri)
        annotation_matches = [v for v in extracted_values if v in matched_original_values]

    evidence_summary = _ci_evidence_summary(
        extracted_values,
        visible_text_checked,
        visible_text_matches,
        annotations_checked,
        annotation_matches,
        unmatched_same_type_annotations,
    )

    return ContactFieldEvidence(
        field_name=field_name,
        extracted_values=list(extracted_values),
        visible_text_checked=visible_text_checked,
        visible_text_matches=visible_text_matches,
        annotations_checked=annotations_checked,
        annotation_matches=annotation_matches,
        unmatched_same_type_annotations=unmatched_same_type_annotations,
        evidence_summary=evidence_summary,
    )


def build_contact_information_evidence(
    contact_information: Dict[str, Any],
    markdown_text: Optional[str] = None,
    pdf_path: Optional[Path] = None,
    schema_valid: Optional[bool] = None,
    stage1_route: Optional[str] = None,
) -> ContactInformationEvidenceRecord:
    """Build an observational evidence record for one resume's already-
    produced Contact Information output (Milestone V3.8).

    This function NEVER modifies `contact_information` and never returns
    a value the caller should use to replace, add to, or remove from it
    -- it is a strictly read-only, additive annotation layer, exactly
    like build_education_evidence/build_work_experience_evidence above,
    extended to Contact Information's own (differently-shaped) schema.
    See the module-level comment above this section, and Milestone
    V3.7's own report, for the corpus evidence this design responds to.

    Args:
        contact_information: The already-extracted, already-validated
            contact_information object (the 5-field dict this module's
            schema defines) -- read only, never mutated.
        markdown_text: The resume's Stage-1 markdown, or None if
            unavailable. When None, every field's visible_text_checked
            is left None (never guessed).
        pdf_path: Path to the resume's original frozen PDF, or None.
            When None, or when PyMuPDF cannot open it, every field's
            annotations_checked is left None (never guessed, never
            treated as "no annotations exist").
        schema_valid: Whether this module's JSON passed schema
            validation, if known.
        stage1_route: The document's Stage1Router route, if known.

    Returns:
        A ContactInformationEvidenceRecord with exactly 5
        ContactFieldEvidence entries, in schema field order.
    """
    visible_body_text = _ci_visible_body_text(markdown_text)
    raw_annotation_uris: Optional[List[str]] = None
    raw_links = _ci_extract_pdf_hyperlinks(pdf_path)
    if raw_links is not None:
        raw_annotation_uris = [uri for _page, uri in raw_links]

    fields: List[ContactFieldEvidence] = []
    for field_name in _CONTACT_INFO_FIELD_ORDER:
        extracted_values = _ci_as_list(contact_information.get(field_name))
        fields.append(
            _ci_build_field_evidence(
                field_name, extracted_values, visible_body_text, raw_annotation_uris
            )
        )

    return ContactInformationEvidenceRecord(
        schema_valid=schema_valid,
        stage1_route=stage1_route,
        fields=fields,
    )


# ---------------------------------------------------------------------------
# Languages (Milestone V4.1)
# ---------------------------------------------------------------------------
#
# Languages' own schema (schemas/modules/languages_schema.json) is a flat
# array of plain strings -- not an array of dict entries (Education/Work
# Experience) and not an object of named fields (Contact Information).
# There is no sub-field to attach EntryEvidence's degree_evidence/
# institution_evidence/match_confidence/company_equals_designation/
# review_priority to (a language value has no sub-fields at all -- it IS
# the whole value), and there is no fixed field name to key
# ContactFieldEvidence-style records by. Reusing either existing shape
# would carry several always-None attributes that simply don't apply,
# the same "reuse only where it fits" judgment this module already made
# twice before (Contact Information vs. Education/Work Experience, see
# the comment above the Contact Information section). A new, minimal,
# per-VALUE record is used instead.
#
# Milestone V4.0 (docs/MILESTONE_V4_0_LANGUAGES_VALIDATION_EXTRACTION_
# ASSESSMENT.md) investigated Languages with the same evidence-first
# methodology as Contact Information and found:
#   - no LanguagesValidator or locator exists;
#   - the real corpus failure mode is 100% precision-type (false
#     positives), 0% recall-type (0 missing values in 22 resumes);
#   - a whole-document, item-level, case-insensitive, whole-word
#     text-presence check is SAFE on the real corpus (0 false positives
#     across the 23 genuinely-correct extracted values) but PARTIAL: it
#     catches the "pure hallucination" mechanism (both confirmed cases,
#     zero textual anchor anywhere) but cannot catch the "project-
#     description misattribution" mechanism (the word IS genuinely
#     present in the source text, just describing something other than
#     the candidate's own language ability -- a semantic distinction
#     this check cannot make);
#   - a section-heading-window locator was tested and rejected (9 false
#     positives on real correct values, since this corpus contains
#     resumes with multiple ambiguous "language"-matching headings --
#     e.g. a Skills "Languages: Python, ..." subheading vs. the real
#     "Language known - English, Hindi, ..." line).
# This section implements exactly the demonstrated, safe rule above --
# nothing broader. It does not attempt section-heading disambiguation,
# and it does not attempt to detect project-description misattribution
# (V4.0's own explicit non-goal for this milestone).

# Closed set of evidence_summary labels for Languages -- reused
# verbatim from V4.0's own experiment CSV terminology
# (benchmark/baselines/v4_0_deterministic_evidence_experiment.csv),
# not renamed or reinterpreted, plus one additional state
# (_LANG_NO_SOURCE_TEXT_AVAILABLE) for "never checked at all," mirroring
# Contact Information's NO_EVIDENCE_SOURCES_AVAILABLE distinction
# between "checked, absent" and "not checked" -- required so absence of
# evidence is never conflated with a negative finding. Only 3 states
# exist here (not Contact Information's 6) because Languages has only
# one evidence source (Stage-1 text), not two (text + annotations), so
# there is no second source to corroborate against or disagree with.
_LANG_SOURCE_TEXT_PRESENT = "SOURCE_TEXT_PRESENT"
_LANG_NO_SOURCE_TEXT_EVIDENCE = "NO_SOURCE_TEXT_EVIDENCE"
_LANG_NO_SOURCE_TEXT_AVAILABLE = "NO_SOURCE_TEXT_AVAILABLE"


@dataclass
class LanguageValueEvidence:
    """Observational evidence for one extracted Languages value.

    Attributes:
        value: The extracted language string, copied verbatim -- never
            re-typed, re-cased, or mutated.
        source_text_checked: True if Stage-1 markdown was available and
            searched. None if no markdown was supplied at all (never
            guessed, never coerced to False) -- the same "checked vs.
            not checked" distinction Contact Information's
            visible_text_checked already makes.
        source_text_present: True if `value` was found verbatim
            (case-insensitive, whole-word) anywhere in the Stage-1
            source text; False if checked and not found; None if not
            checked at all (source_text_checked is None).
        evidence_summary: One of SOURCE_TEXT_PRESENT,
            NO_SOURCE_TEXT_EVIDENCE, or NO_SOURCE_TEXT_AVAILABLE. NEVER
            a numeric score, and NEVER an ACCEPT/REJECT/REPAIR
            instruction. Per V4.0's own direct finding,
            NO_SOURCE_TEXT_EVIDENCE catches the "pure hallucination"
            failure mechanism but NOT the "project-description
            misattribution" mechanism (where the word IS present in the
            source, just in an unrelated semantic role) -- a value
            showing SOURCE_TEXT_PRESENT is therefore evidence the word
            appears in the document, not a correctness verdict, and a
            value showing NO_SOURCE_TEXT_EVIDENCE must never be
            silently treated as proof of an error and must never
            trigger removal, rejection, or repair by any caller.
    """

    value: str
    source_text_checked: Optional[bool]
    source_text_present: Optional[bool]
    evidence_summary: str


@dataclass
class LanguagesEvidenceRecord:
    """The full observational evidence record for one resume's already-
    produced Languages output. See the module-level comment above this
    section for why this does not reuse EntryEvidence/ModuleEvidenceRecord
    or ContactFieldEvidence/ContactInformationEvidenceRecord.

    Attributes:
        schema_valid: Whether this module's JSON passed
            SchemaValidator.validate(), if known.
        stage1_route: The document's Stage1Router route, if known. This
            function never calls Stage1Router itself.
        values: One LanguageValueEvidence per extracted language value,
            in the same order as the module's own array. Empty if the
            extracted languages array is empty -- there is no fixed
            slot to fill (unlike Contact Information's always-5-fields
            shape), so an empty list is itself the complete, correct
            representation of "nothing was extracted."
    """

    schema_valid: Optional[bool]
    stage1_route: Optional[str]
    values: List[LanguageValueEvidence] = field(default_factory=list)


def build_languages_evidence(
    languages: List[str],
    markdown_text: Optional[str] = None,
    schema_valid: Optional[bool] = None,
    stage1_route: Optional[str] = None,
) -> LanguagesEvidenceRecord:
    """Build an observational evidence record for one resume's already-
    produced Languages output (Milestone V4.1).

    This function NEVER modifies `languages` and never returns a value
    the caller should use to replace, add to, remove from, or reorder
    it -- it is a strictly read-only, additive annotation layer,
    exactly like build_education_evidence/build_work_experience_evidence/
    build_contact_information_evidence above. See the module-level
    comment above this section, and Milestone V4.0's own report, for the
    corpus evidence this design responds to.

    Args:
        languages: The already-extracted, already-validated languages
            array (the plain list of strings this module's schema
            defines) -- read only, never mutated.
        markdown_text: The resume's Stage-1 markdown, or None if
            unavailable. When None, every value's source_text_checked
            is left None (never guessed).
        schema_valid: Whether this module's JSON passed schema
            validation, if known.
        stage1_route: The document's Stage1Router route, if known.

    Returns:
        A LanguagesEvidenceRecord with one LanguageValueEvidence per
        entry in `languages`, in the same order.
    """
    values: List[LanguageValueEvidence] = []
    for value in languages:
        source_text_checked: Optional[bool] = None
        source_text_present: Optional[bool] = None
        if markdown_text is not None:
            source_text_checked = True
            if isinstance(value, str) and value.strip():
                pattern = r"\b" + re.escape(value) + r"\b"
                source_text_present = bool(re.search(pattern, markdown_text, re.IGNORECASE))
            else:
                source_text_present = False

        if source_text_checked is None:
            evidence_summary = _LANG_NO_SOURCE_TEXT_AVAILABLE
        elif source_text_present:
            evidence_summary = _LANG_SOURCE_TEXT_PRESENT
        else:
            evidence_summary = _LANG_NO_SOURCE_TEXT_EVIDENCE

        values.append(
            LanguageValueEvidence(
                value=value,
                source_text_checked=source_text_checked,
                source_text_present=source_text_present,
                evidence_summary=evidence_summary,
            )
        )

    return LanguagesEvidenceRecord(
        schema_valid=schema_valid,
        stage1_route=stage1_route,
        values=values,
    )


# ---------------------------------------------------------------------------
# Professional Summary (Milestone V4.5)
# ---------------------------------------------------------------------------
#
# Professional Summary's own schema (schemas/modules/professional_summary_
# schema.json) is a single free-form scalar -- {"professional_summary":
# "string" | null} -- not an array (Education/Work Experience/Languages)
# and not an object of named fields (Contact Information). There is
# exactly one value to evaluate per resume, never zero-to-N of them, so
# neither EntryEvidence's entry_index-based shape nor ContactFieldEvidence's
# field-name-keyed shape fits -- forcing either on would misrepresent a
# single scalar as a collection, the same "reuse only where it fits"
# judgment this module has now made three times (see the comments above
# the Contact Information and Languages sections). A new, minimal,
# single-record structure is used instead -- no list at all.
#
# Milestone V4.4 (docs/MILESTONE_V4_4_SUMMARY_EXTRACTION_EVIDENCE_
# ASSESSMENT.md) investigated Professional Summary with the same
# evidence-first methodology as Contact Information and Languages and
# found:
#   - no validator, locator, or preprocess_json() hook exists for this
#     module -- zero deterministic logic touches it today;
#   - the prompt's own rule 6 requires verbatim/near-verbatim extraction,
#     not paraphrase or synthesis -- unlike a true summarization task,
#     source-text presence is a directly-motivated signal here;
#   - presence/absence agreement between GT and extraction is perfect
#     across the 22-resume corpus (0 missing, 0 extra) -- all
#     disagreement is about content, when both are non-null;
#   - the DOMINANT disagreement mechanism (5 of 9 "differ" cases,
#     confirmed by directly reading real Stage-1 source text) is GT
#     itself diverging from source (editorial paraphrase/trimming/
#     normalization by the human adjudicator) -- extraction was the
#     MORE verbatim-faithful side in those cases, not the error. This is
#     why GT must never be used as a runtime evidence source for this
#     module: comparing extraction to GT would in that majority of cases
#     penalize the more correct output;
#   - a whole-document, whitespace-normalized verbatim-substring check
#     is safe (0 false positives against genuinely-verbatim content) but
#     has a real, corpus-demonstrated precision limit on its own: 2 of 3
#     non-matching cases in the corpus (013, 020) were pure formatting/
#     single-OCR-character artifacts, not real unsupported content --
#     resolved by a secondary token-overlap ratio (fraction of the
#     extraction's own word-tokens also found anywhere in the source),
#     which showed 1.000 for both false flags and a distinctly lower
#     0.926 for the one genuine case (010);
#   - no numeric threshold was calibrated or is proposed here -- V4.4
#     explicitly found only 2 data points for the ratio signal, marked
#     CALIBRATION REQUIRED, not asserted.
# This section implements exactly the demonstrated, safe signal above --
# nothing broader. It does not use GT at runtime (GT is never a
# parameter to build_professional_summary_evidence()), does not invent a
# confidence score (token_overlap_ratio is a raw, transparent
# measurement, not a calibrated probability -- the same distinction
# EntryEvidence.match_confidence's own docstring already draws for Work
# Experience's real, measured, non-"confidence-score" ratio), and does
# not classify a non-matching case as "incorrect" or "unsupported" --
# only as "no whole-document verbatim match found," a factual statement
# about what was and was not directly observed.

# Closed set of evidence_summary labels -- reused verbatim where V4.4
# already used this exact terminology
# (benchmark/baselines/v4_4_evidence_experiment_results.csv), plus two
# states for the cases that terminology didn't need to distinguish
# (no extracted value at all; source unavailable to check).
_SUMMARY_NO_SUMMARY_EXTRACTED = "NO_SUMMARY_EXTRACTED"
_SUMMARY_NO_SOURCE_TEXT_AVAILABLE = "NO_SOURCE_TEXT_AVAILABLE"
_SUMMARY_SOURCE_TEXT_VERBATIM_MATCH = "SOURCE_TEXT_VERBATIM_MATCH"
_SUMMARY_NO_VERBATIM_MATCH_FOUND = "NO_VERBATIM_MATCH_FOUND"


@dataclass
class ProfessionalSummaryEvidence:
    """Observational evidence for one resume's already-extracted
    Professional Summary value. Exactly one of these exists per resume
    (never a list) -- see the module-level comment above this section
    for why.

    Attributes:
        extracted_value: The extracted professional_summary value,
            copied verbatim -- never re-typed, re-cased, reformatted, or
            mutated. None if the module extracted no summary at all
            (a legitimate, common outcome -- 7 of 22 resumes in V4.4's
            own corpus, always agreeing correctly with GT).
        source_text_checked: True if Stage-1 markdown was available and
            searched. None if no markdown was supplied at all (never
            guessed, never coerced to False) -- the same "checked vs.
            not checked" distinction Contact Information's
            visible_text_checked and Languages' source_text_checked
            already make.
        whole_document_verbatim_match: True if extracted_value (after
            whitespace normalization: collapse all whitespace runs,
            casefold) is found as a single contiguous substring
            anywhere in the Stage-1 source text; False if checked and
            not found this way; None if not checked at all
            (source_text_checked is None) or extracted_value is None.
            V4.4 directly demonstrated this is NOT the same as "the
            content is unsupported" -- 2 of the 3 corpus cases where
            this came back False were pure formatting/OCR-artifact
            false flags, not real content problems. See
            token_overlap_ratio.
        token_overlap_ratio: Fraction (0.0-1.0) of extracted_value's own
            word-tokens that also appear anywhere in the Stage-1 source
            text, case-insensitive. A raw, directly-computed
            measurement, exactly like EntryEvidence.match_confidence
            for Work Experience -- NOT a calibrated confidence score,
            NOT a correctness probability, and NOT paired with any
            threshold (V4.4 found only 2 data points for this signal
            and explicitly declined to calibrate one -- CALIBRATION
            REQUIRED for any future threshold-based use). None if not
            checked (source_text_checked is None) or extracted_value is
            None/empty.
        evidence_summary: One of NO_SUMMARY_EXTRACTED,
            NO_SOURCE_TEXT_AVAILABLE, SOURCE_TEXT_VERBATIM_MATCH, or
            NO_VERBATIM_MATCH_FOUND. NEVER a numeric score, and NEVER an
            ACCEPT/REJECT/REPAIR instruction to any caller.
            NO_VERBATIM_MATCH_FOUND is a factual statement ("no
            whole-document contiguous match was found"), explicitly NOT
            a claim that the content is incorrect or unsupported --
            token_overlap_ratio must be read alongside it before any
            human or future caller draws a conclusion, per V4.4's own
            direct demonstration that this state is frequently a
            formatting artifact, not a content problem.
    """

    extracted_value: Optional[str]
    source_text_checked: Optional[bool]
    whole_document_verbatim_match: Optional[bool]
    token_overlap_ratio: Optional[float]
    evidence_summary: str


def _summary_normalize_ws(value: str) -> str:
    """Collapse all whitespace runs to a single space and casefold --
    comparison-only, never used to alter a stored value. Mirrors V4.4's
    own experiment methodology exactly (re.sub(r"\\s+", " ", ...))."""
    return re.sub(r"\s+", " ", value).strip().casefold()


_SUMMARY_TOKEN_RE = re.compile(r"[a-z0-9']+")


def build_professional_summary_evidence(
    professional_summary: Optional[str],
    markdown_text: Optional[str] = None,
    schema_valid: Optional[bool] = None,
    stage1_route: Optional[str] = None,
) -> ProfessionalSummaryEvidence:
    """Build an observational evidence record for one resume's already-
    produced Professional Summary output (Milestone V4.5).

    This function NEVER modifies `professional_summary` and never
    returns a value the caller should use to replace, rewrite, or
    remove it -- it is a strictly read-only, additive annotation layer,
    exactly like build_education_evidence/build_work_experience_evidence/
    build_contact_information_evidence/build_languages_evidence above.
    It NEVER accepts or reads a ground-truth value -- GT is not a
    parameter and is not used at runtime, per V4.4's own explicit
    finding that GT frequently diverges from source and must never be
    used as a Summary correction/contradiction signal. See the
    module-level comment above this section, and Milestone V4.4's own
    report, for the corpus evidence this design responds to.

    Args:
        professional_summary: The already-extracted, already-validated
            professional_summary value (a plain string, or None) --
            read only, never mutated.
        markdown_text: The resume's Stage-1 markdown, or None if
            unavailable. When None, source_text_checked,
            whole_document_verbatim_match, and token_overlap_ratio are
            all left None (never guessed).
        schema_valid: Whether this module's JSON passed schema
            validation, if known.
        stage1_route: The document's Stage1Router route, if known.

    Returns:
        A single ProfessionalSummaryEvidence record.
    """
    if not isinstance(professional_summary, str) or not professional_summary.strip():
        return ProfessionalSummaryEvidence(
            extracted_value=None,
            source_text_checked=None,
            whole_document_verbatim_match=None,
            token_overlap_ratio=None,
            evidence_summary=_SUMMARY_NO_SUMMARY_EXTRACTED,
        )

    source_text_checked: Optional[bool] = None
    verbatim_match: Optional[bool] = None
    overlap_ratio: Optional[float] = None

    if markdown_text is not None:
        source_text_checked = True
        md_norm = _summary_normalize_ws(markdown_text)
        value_norm = _summary_normalize_ws(professional_summary)
        verbatim_match = value_norm in md_norm

        value_tokens = set(_SUMMARY_TOKEN_RE.findall(value_norm))
        md_tokens = set(_SUMMARY_TOKEN_RE.findall(md_norm))
        overlap_ratio = (
            len(value_tokens & md_tokens) / len(value_tokens) if value_tokens else None
        )

    if source_text_checked is None:
        evidence_summary = _SUMMARY_NO_SOURCE_TEXT_AVAILABLE
    elif verbatim_match:
        evidence_summary = _SUMMARY_SOURCE_TEXT_VERBATIM_MATCH
    else:
        evidence_summary = _SUMMARY_NO_VERBATIM_MATCH_FOUND

    return ProfessionalSummaryEvidence(
        extracted_value=professional_summary,
        source_text_checked=source_text_checked,
        whole_document_verbatim_match=verbatim_match,
        token_overlap_ratio=overlap_ratio,
        evidence_summary=evidence_summary,
    )


# ---------------------------------------------------------------------------
# Personal Information (Milestone V4.8)
# ---------------------------------------------------------------------------
#
# Personal Information's own schema (schemas/modules/personal_information_
# schema.json) is 4 scalar fields -- full_name, first_name, last_name,
# current_location, each string-or-null -- but unlike Professional
# Summary (one lone scalar), this module has TWO genuinely distinct
# evidence relationships to represent: a CROSS-FIELD check (does
# first_name + last_name represent every substantial token in
# full_name?) and a single-field source-corroboration check
# (current_location vs. Stage-1 text). Neither ModuleEvidenceRecord
# (entry-indexed -- there are no entries here), ContactFieldEvidence
# (per-independent-named-field -- full_name/first_name/last_name are
# not independent, they are one relationship), nor
# ProfessionalSummaryEvidence (exactly one scalar) fits this shape --
# the same "reuse only where it fits" judgment this module has now made
# four times. A new top-level record with two small, independent nested
# evidence structures is used instead -- the smallest structure that
# still matches the real relationship between these fields.
#
# Milestone V4.7 (docs/MILESTONE_V4_7_PERSONAL_INFORMATION_ASSESSMENT.md)
# investigated Personal Information with the same evidence-first
# methodology as the other modules and found:
#   - a pre-existing, narrow preprocess_json() hook (Milestone 23.9.1)
#     already derives first_name/last_name from full_name in the one
#     case where the model leaves both null -- this evidence layer does
#     not duplicate or re-implement that logic, it only observes the
#     FINAL, already-processed output, exactly like every other
#     build_*_evidence() function in this module;
#   - a purely internal, GT-independent signal -- does every
#     substantial (2+ character) word-token in full_name also appear
#     somewhere in first_name + last_name? -- achieved 0 false
#     positives and 0 false negatives across the full 22-resume corpus
#     (single-character tokens, typically middle-initial abbreviations
#     like "S.", are excluded: including them produced exactly 1 false
#     positive, a case where both GT's own convention and the model's
#     split legitimately drop the initial);
#   - a whole-document verbatim-presence check for current_location
#     (mirroring Languages'/Professional Summary's own already-validated
#     methodology) correctly caught the one confirmed pure-fabrication
#     case in the corpus, but is explicitly UNABLE to detect the other
#     two real defect categories V4.7 found: wrong-section attribution
#     (a real, verbatim value copied from the wrong part of the resume,
#     e.g. a university's or employer's location) and boundary
#     truncation (a real, verbatim, but incomplete address) -- both
#     produce "verbatim in source" exactly like a fully correct value
#     does. This limitation is preserved explicitly in this
#     implementation, not glossed over.
# This section implements exactly these two demonstrated signals --
# nothing broader. It does not attempt wrong-section-attribution
# detection (V4.7 found no safe deterministic method for this), does
# not rewrite or move any field, and does not use GT at runtime.

_PI_NAME_NO_FULL_NAME_TO_CHECK = "NO_FULL_NAME_TO_CHECK"
_PI_NAME_TOKENS_FULLY_REPRESENTED = "NAME_TOKENS_FULLY_REPRESENTED"
_PI_NAME_TOKENS_MISSING = "NAME_TOKENS_MISSING"

_PI_LOCATION_NO_LOCATION_EXTRACTED = "NO_LOCATION_EXTRACTED"
_PI_LOCATION_NO_SOURCE_TEXT_AVAILABLE = "NO_SOURCE_TEXT_AVAILABLE"
_PI_LOCATION_SOURCE_TEXT_VERBATIM_MATCH = "SOURCE_TEXT_VERBATIM_MATCH"
_PI_LOCATION_NO_VERBATIM_MATCH_FOUND = "NO_VERBATIM_MATCH_FOUND"

# Tokens shorter than this are excluded from the name-completeness
# check -- V4.7's own direct corpus finding: including single-character
# tokens (almost always a middle-initial abbreviation like "S.") creates
# a false positive on a resume where BOTH ground truth's own convention
# and the model's own name split legitimately drop the initial from
# first_name/last_name (while correctly keeping it in full_name).
_PI_NAME_MIN_TOKEN_LENGTH = 2

_PI_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _pi_substantial_tokens(value: Optional[str]) -> set:
    """Case-folded word-tokens of `value`, excluding any shorter than
    _PI_NAME_MIN_TOKEN_LENGTH. Comparison-only -- never used to alter a
    stored value."""
    if not value:
        return set()
    return {t for t in _PI_TOKEN_RE.findall(value.casefold()) if len(t) >= _PI_NAME_MIN_TOKEN_LENGTH}


@dataclass
class NameCompletenessEvidence:
    """Observational evidence for whether first_name + last_name
    together represent every substantial token already present in
    full_name -- V4.7's own directly-demonstrated, 0-false-positive/
    0-false-negative corpus finding.

    Attributes:
        full_name: Copied verbatim from the extracted output. None if
            not extracted.
        first_name: Copied verbatim. None if not extracted.
        last_name: Copied verbatim. None if not extracted.
        checked: True if full_name was present and non-empty (so a
            comparison was actually performed). None if full_name
            itself is missing/empty -- there is nothing to check
            completeness against, and this is explicitly NOT the same
            as "checked and found incomplete." Preserves the same
            "checked vs. not checked" distinction every other evidence
            builder in this module already makes.
        missing_tokens: Substantial (2+ character) word-tokens found in
            full_name but not found anywhere in first_name or
            last_name, case-folded, in no particular order. Empty list
            if checked and nothing is missing, or if not checked at all
            -- callers must read `checked` to tell these two empty-list
            cases apart, not infer "checked" from list emptiness.
        evidence_summary: One of NO_FULL_NAME_TO_CHECK,
            NAME_TOKENS_FULLY_REPRESENTED, or NAME_TOKENS_MISSING.
            NEVER a numeric score. NAME_TOKENS_MISSING is a factual,
            diagnostic statement ("these tokens were not found in the
            split") -- it does not identify WHICH of first_name/
            last_name should have held the missing token, does not
            imply which field is "wrong," and must never be used to
            automatically move, insert, or repair any field.
    """

    full_name: Optional[str]
    first_name: Optional[str]
    last_name: Optional[str]
    checked: Optional[bool]
    missing_tokens: List[str]
    evidence_summary: str


@dataclass
class LocationEvidence:
    """Observational evidence for whether current_location is found
    verbatim (whitespace-normalized) anywhere in the Stage-1 source
    text -- V4.7's own directly-demonstrated signal and its explicit,
    corpus-confirmed limitations.

    Attributes:
        extracted_value: Copied verbatim. None if not extracted.
        source_text_checked: True if Stage-1 markdown was available and
            searched. None if no markdown was supplied (never guessed).
        verbatim_in_source: True if found as a contiguous,
            whitespace-normalized substring anywhere in the source
            text; False if checked and not found; None if not checked
            or extracted_value is None.
        evidence_summary: One of NO_LOCATION_EXTRACTED,
            NO_SOURCE_TEXT_AVAILABLE, SOURCE_TEXT_VERBATIM_MATCH, or
            NO_VERBATIM_MATCH_FOUND.

            CRITICAL, corpus-demonstrated limitations (V4.7) that
            callers must respect: (1) SOURCE_TEXT_VERBATIM_MATCH is NOT
            proof of correct semantic attribution -- V4.7 found 2
            confirmed cases where a verbatim, source-present location
            was copied from the WRONG part of the resume (a university's
            or an employer's location, not the candidate's own address)
            and this signal cannot and does not distinguish that case
            from a genuinely correct one. (2) NO_VERBATIM_MATCH_FOUND
            can indicate pure fabrication (V4.7 found exactly 1 such
            case) but must never, on its own, justify automatically
            deleting or overwriting a value -- absence of a verbatim
            match is observational evidence only. (3) This signal does
            NOT detect boundary truncation (a real, verbatim, but
            incomplete address) at all -- a truncated value matches
            this check just as cleanly as a complete one.
    """

    extracted_value: Optional[str]
    source_text_checked: Optional[bool]
    verbatim_in_source: Optional[bool]
    evidence_summary: str


@dataclass
class PersonalInformationEvidenceRecord:
    """The full observational evidence record for one resume's already-
    produced Personal Information output. See the module-level comment
    above this section for why this uses two small nested evidence
    structures rather than reusing any of ModuleEvidenceRecord/
    ContactInformationEvidenceRecord/ProfessionalSummaryEvidence.

    Attributes:
        schema_valid: Whether this module's JSON passed
            SchemaValidator.validate(), if known.
        stage1_route: The document's Stage1Router route, if known. This
            function never calls Stage1Router itself.
        name_completeness: See NameCompletenessEvidence.
        location: See LocationEvidence.
    """

    schema_valid: Optional[bool]
    stage1_route: Optional[str]
    name_completeness: NameCompletenessEvidence
    location: LocationEvidence


def _pi_normalize_ws(value: str) -> str:
    """Collapse whitespace runs to a single space and casefold --
    comparison-only, never used to alter a stored value. Mirrors every
    other whole-document verbatim check already in this module."""
    return re.sub(r"\s+", " ", value).strip().casefold()


def build_personal_information_evidence(
    personal_information: Dict[str, Any],
    markdown_text: Optional[str] = None,
    schema_valid: Optional[bool] = None,
    stage1_route: Optional[str] = None,
) -> PersonalInformationEvidenceRecord:
    """Build an observational evidence record for one resume's already-
    produced Personal Information output (Milestone V4.8).

    This function NEVER modifies `personal_information` and never
    returns a value the caller should use to replace, move, or rewrite
    any of its fields -- it is a strictly read-only, additive
    annotation layer, exactly like every other build_*_evidence()
    function in this module. It never accepts or reads a ground-truth
    value. It does not duplicate or re-run Milestone 23.9.1's
    preprocess_json() name-splitting logic -- it only observes
    whatever first_name/last_name the caller's already-final output
    contains, however they were produced.

    Args:
        personal_information: The already-extracted, already-validated
            personal_information object (the 4-field dict this
            module's schema defines) -- read only, never mutated.
        markdown_text: The resume's Stage-1 markdown, or None if
            unavailable. When None, the location evidence's
            source_text_checked/verbatim_in_source are left None.
        schema_valid: Whether this module's JSON passed schema
            validation, if known.
        stage1_route: The document's Stage1Router route, if known.

    Returns:
        A PersonalInformationEvidenceRecord.
    """
    full_name = personal_information.get("full_name")
    first_name = personal_information.get("first_name")
    last_name = personal_information.get("last_name")
    current_location = personal_information.get("current_location")

    # --- Name completeness ---
    if not isinstance(full_name, str) or not full_name.strip():
        name_checked: Optional[bool] = None
        missing_tokens: List[str] = []
        name_summary = _PI_NAME_NO_FULL_NAME_TO_CHECK
    else:
        name_checked = True
        full_tokens = _pi_substantial_tokens(full_name)
        split_tokens = _pi_substantial_tokens(first_name) | _pi_substantial_tokens(last_name)
        missing_tokens = sorted(full_tokens - split_tokens)
        name_summary = (
            _PI_NAME_TOKENS_MISSING if missing_tokens else _PI_NAME_TOKENS_FULLY_REPRESENTED
        )

    name_evidence = NameCompletenessEvidence(
        full_name=full_name,
        first_name=first_name,
        last_name=last_name,
        checked=name_checked,
        missing_tokens=missing_tokens,
        evidence_summary=name_summary,
    )

    # --- Location source evidence ---
    if not isinstance(current_location, str) or not current_location.strip():
        loc_checked: Optional[bool] = None
        verbatim: Optional[bool] = None
        loc_summary = _PI_LOCATION_NO_LOCATION_EXTRACTED
    elif markdown_text is None:
        loc_checked = None
        verbatim = None
        loc_summary = _PI_LOCATION_NO_SOURCE_TEXT_AVAILABLE
    else:
        loc_checked = True
        verbatim = _pi_normalize_ws(current_location) in _pi_normalize_ws(markdown_text)
        loc_summary = (
            _PI_LOCATION_SOURCE_TEXT_VERBATIM_MATCH if verbatim else _PI_LOCATION_NO_VERBATIM_MATCH_FOUND
        )

    location_evidence = LocationEvidence(
        extracted_value=current_location,
        source_text_checked=loc_checked,
        verbatim_in_source=verbatim,
        evidence_summary=loc_summary,
    )

    return PersonalInformationEvidenceRecord(
        schema_valid=schema_valid,
        stage1_route=stage1_route,
        name_completeness=name_evidence,
        location=location_evidence,
    )
