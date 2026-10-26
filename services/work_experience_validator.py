"""
services/work_experience_validator.py

Work Experience Validator — Phase 1 (Milestone 17.6).

Deterministic post-extraction validation for WorkExperienceExtractor's
output, implementing only the five "Quick Win" rules from
docs/WORK_EXPERIENCE_VALIDATION.md (Milestone 17.5) that need no source
markdown text, no job-to-text-span matching, and no retry logic:

    WEV-001  end_date sentinel strings ("Present", "Current", etc.)
             -> auto-fixed (end_date=null, currently_working=true)
    WEV-002  duration that exactly equals the computed date-difference
             -> auto-fixed (duration=null)
    WEV-003  employment_type outside a fixed whitelist
             -> auto-fixed (employment_type=null, never a guessed
                replacement)
    WEV-011  start_date after end_date
             -> flagged only, never corrected
    WEV-013  currently_working=true despite a stated end_date
             -> auto-fixed (currently_working=false)

Every other rule in docs/WORK_EXPERIENCE_VALIDATION.md (WEV-004 through
WEV-010, WEV-012, WEV-014, WEV-015) needs the source markdown, a
job-to-text-span locator, or retry logic — none of that exists here,
and none of it is implemented in this milestone.

This module knows nothing about ParserPipeline or Ollama. It takes a
work_experience array (already extracted) and returns a corrected array
plus a report of every issue found — a pure, in-memory, side-effect-free
transformation. The input list (and its job dicts) are never mutated;
validate() always returns new objects.

WEV-002's known, deliberate Phase 1 limitation: this validator cannot
confirm whether a duration that happens to equal the computed
date-difference was genuinely written in the resume (a coincidence) or
was computed by the model (a rule violation). Every exact match is
treated as computed and nulled out, per the milestone's own instruction.

Milestone 17.7: validate()'s signature gained a `markdown` parameter —
services/work_experience_locator.py (the job-to-text-span locator
docs/WORK_EXPERIENCE_VALIDATION.md called for) now exists as standalone
infrastructure, and ParserPipeline now reads the source markdown once
and passes it through here, establishing the API every future rule
needing markdown (WEV-004, WEV-007, WEV-008, WEV-009, WEV-014, WEV-015)
will use. Phase 1's five rules did not use it yet in that milestone.

Milestone 17.8 adds the first markdown-aware rule:

    WEV-015  a technology listed on a job that cannot be found (even
              as a loose variant) inside that job's own located
              markdown span -> auto-fixed (removed from the
              technologies list; never added, never guessed)

WEV-015 is docs/WORK_EXPERIENCE_VALIDATION.md's own rule ID for this
exact behavior ("Technology not mentioned in job" — the granular,
per-item companion to WEV-004's holistic whole-list check). The
Milestone 17.8 request text itself referred to this as "WEV-005", but
WEV-005 in the design doc is a different, unrelated rule (Duplicate
companies) — this is implemented and reported as WEV-015 throughout,
per the design doc, and the discrepancy is called out in
docs/TECHNOLOGY_VALIDATION_RESULTS.md.

WEV-015 uses services/work_experience_locator.py's WorkExperienceLocator
to find each job's own span of source markdown, and only removes items
absent from that job's own span text; it never searches outside that
span (in particular, never the Skills section, which is naturally
outside the located Experience-section spans), never adds a
technology, and calls no LLM. When a job cannot be confidently matched
to exactly one span, its technologies are left completely unchanged
and a Warning-severity issue is emitted instead of guessing.

WEV-001 through WEV-013 still ignore `markdown` entirely — only
WEV-015 reads it.

Milestone 23.15 adds three more markdown-aware rules, reusing WEV-015's
exact infrastructure (the same `self.locator.locate()` +
`match_jobs_to_spans()` call already made for WEV-015 in `validate()`,
not a second, separate lookup):

    WEV-016  employment_type present but unsupported by that job's own
              markdown span -> auto-fixed (employment_type=null)
    WEV-017  location present but unsupported by that job's own
              markdown span -> auto-fixed (location=null)
    WEV-018  duration present but unsupported by that job's own
              markdown span -> auto-fixed (duration=null)

These complete work `docs/WORK_EXPERIENCE_VALIDATION.md`'s own original
design already specified but Phase 1 explicitly deferred for lack of
markdown access: WEV-003's original spec (§2) already called for a
markdown-presence check ("does not appear ... within that job's own
text span"), but the *shipped* `_apply_wev_003` above only checks
whitelist membership — a cheaper, markdown-free approximation, not the
full rule. Likewise WEV-002's original spec's condition (b) ("the exact
duration string cannot be found verbatim anywhere in that job's own
text span") was never built; the shipped `_apply_wev_002` only
implements condition (a), computed-value detection. WEV-016/018 are
that deferred markdown-check half, now added as their own rules rather
than folded into WEV-002/003's existing methods, so neither rule's
already-tested behavior changes. `location` was never assigned a rule
ID in the original 15-rule design at all — WEV-017 is a genuinely new
rule, not a deferred one.

Root-cause evidence for why this is worth building now:
`docs/MILESTONE_23_14_WORK_EXPERIENCE_RESIDUAL_ERROR_ANALYSIS.md` found
111 of ~265 field-level `work_experience` mismatches (~42%, across 20
of 21 evaluated resumes) are exactly this pattern — `employment_type`/
`location`/`duration` populated when ground truth is `null` — the
single largest concrete, already-diagnosed opportunity remaining in
this module.

Matching strategy — deliberately NOT the same as WEV-015's: WEV-015
uses a whole-token (space-padded) substring check specifically to stop
a short technology name like "R" or "Go" from matching as a run of
letters inside an unrelated word (e.g. "r" inside "for"). WEV-016/017/
018 use a plain normalized substring check instead (still fully
deterministic — the same `_NORMALIZE_PATTERN` normalization WEV-015
uses, just without the token-boundary padding) — a real, verified,
GT-confirmed-correct case in this project's own dataset
(`Dattatray S. Phad_Desktop Support Engineer_Intileo`'s
"Nex-Z Solutions (Freelancer)" job, `employment_type: "Freelance"`)
would be *wrongly nulled* by a whole-token check, since "freelance" is
not a whole token inside "freelancer" — the source states the noun
"Freelancer" (as part of the company's own parenthetical), not the
adjective "Freelance" (the schema's employment_type value), and only a
plain substring check recognizes the second as legitimately derived
from the first. `employment_type`/`location`/`duration` values are not
short, generic, collision-prone tokens the way a bare technology
abbreviation can be, so the false-positive risk WEV-015's token-padding
guards against does not apply the same way here — this was verified
directly against this project's own real dataset (every known-legitimate
`employment_type`/`location`/`duration` value in the current 23-resume
ground truth was checked against its own matched job span; see
docs/MILESTONE_23_15_WORK_EXPERIENCE_PROVENANCE_VALIDATOR_RESULTS.md
§5-§6 for the full verification), not assumed or invented to make more
examples pass.

Like WEV-015: a job with no confident span match (`span is None`) has
its `employment_type`/`location`/`duration` left completely unchanged,
with a Warning-severity issue instead of a guess — this is the same
safety net that already protects `AMAN_CV_BCA`'s and
`AshishKumarSingh_RBM_Pune_Stonelam`'s real, GT-confirmed-legitimate
values, both of which this project's own `WorkExperienceLocator` cannot
currently find any confident span for at all (a pre-existing locator
limitation, not something these new rules introduce or worsen).

Milestone 17.9 built WorkExperienceLocator.match_jobs_to_spans() — a
confidence-based matcher combining company similarity, designation
similarity, date overlap, and occurrence order — as standalone
infrastructure, not yet used anywhere. Milestone 17.10 wires it in
here: WEV-015 now delegates its job-to-span matching entirely to
`self.locator.match_jobs_to_spans()` instead of the normalized-
company-name-only lookup Milestone 17.8 originally used (removed in
this milestone). This measurably improves how often WEV-015 finds a
confident match — see docs/TECHNOLOGY_VALIDATION_V2_RESULTS.md for the
real, measured before/after numbers — while the technology-provenance
check itself (§ _apply_wev_015: keep an item if it's found, even as a
loose variant, in the matched span's text; otherwise remove it; leave
everything unchanged and warn if there's no confident match) is
completely unchanged. Each matched job's `ValidationIssue` now also
carries the winning match's `confidence` score (see `ValidationIssue.
confidence`), propagated straight from the JobSpan
`match_jobs_to_spans()` returned.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dateutil import parser as date_parser

from services.work_experience_locator import JobSpan, WorkExperienceLocator

logger = logging.getLogger(__name__)

# WEV-001: ongoing-status words that must never appear literally in
# end_date — matching prompts/modules/work_experience.md's own rule 10.
_ONGOING_SENTINELS = frozenset({"present", "current", "till date", "ongoing", "now"})

# WEV-003: the only employment_type values Phase 1 accepts (matched
# case-insensitively). Anything else present is removed — never
# replaced with a guessed value.
_ALLOWED_EMPLOYMENT_TYPES = frozenset(
    {
        "internship",
        "freelance",
        "contract",
        "part-time",
        "full-time",
        "temporary",
        "apprenticeship",
        "consultant",
    }
)

# Fixed reference date used only to fill in a missing day-of-month (or
# missing year) when parsing a "Month Year" or bare "Year" date string.
# Never today's real date, so parsing the same input always produces
# the same result regardless of when this code runs.
_DATE_PARSE_DEFAULT = datetime(1900, 1, 1)

_YEARS_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b", re.IGNORECASE)
_MONTHS_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:months?|mos?)\b", re.IGNORECASE)

# WEV-015 technology-presence check: collapse everything but letters
# and digits to a single space, so e.g. "Node.js" and "Node JS"
# normalize identically. (Job-to-span *matching* itself no longer uses
# this — since Milestone 17.10 it's fully delegated to
# WorkExperienceLocator.match_jobs_to_spans(), which does its own,
# separate normalization.)
_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9]+")

# WEV-016/017/018 (Milestone 23.15): field name -> (schema key, rule id,
# human-readable label), driving one shared implementation instead of
# three near-identical methods. Order matches the schema/§ documentation
# order (employment_type, location, duration).
_PROVENANCE_FIELDS: Tuple[Tuple[str, str, str], ...] = (
    ("employment_type", "WEV-016", "employment_type"),
    ("location", "WEV-017", "location"),
    ("duration", "WEV-018", "duration"),
)


@dataclass
class ValidationIssue:
    """One rule's finding for one job entry.

    Attributes:
        job_index: Index of the affected job within the work_experience
            array (post-validation order — this validator never
            reorders or removes entries, so the index is stable).
        rule_id: The WEV-xxx rule that produced this issue.
        severity: Human-readable severity, matching the per-rule
            severity documented in docs/WORK_EXPERIENCE_VALIDATION.md.
        message: A human-readable description of what was found (and,
            for auto-fixed issues, what was changed).
        auto_fixed: Whether this validator corrected the job entry in
            place (True) or only flagged it, leaving the entry
            unchanged (False, e.g. WEV-011).
        confidence: The job-to-span match confidence (0-1) that
            produced this issue, for WEV-015 issues on a confidently-
            matched job (Milestone 17.10 — propagated straight from
            WorkExperienceLocator.match_jobs_to_spans()). None for
            every other rule, and for WEV-015's own unmatched-job
            warning (there's no winning match to report a confidence
            for).
    """

    job_index: int
    rule_id: str
    severity: str
    message: str
    auto_fixed: bool
    confidence: Optional[float] = None


@dataclass
class ValidationReport:
    """Aggregate result of validating one resume's work_experience array.

    Attributes:
        total_jobs: Number of job entries validated.
        issues: Every ValidationIssue found, in the order rules were
            applied (WEV-001, WEV-002, WEV-003, WEV-011, WEV-013 for
            job 0, then the same order for job 1, and so on).
    """

    total_jobs: int
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def auto_fixed_count(self) -> int:
        """Number of issues this validator corrected in place."""
        return sum(1 for issue in self.issues if issue.auto_fixed)

    @property
    def flagged_count(self) -> int:
        """Number of issues recorded but left uncorrected (e.g. WEV-011)."""
        return sum(1 for issue in self.issues if not issue.auto_fixed)

    def issues_by_rule(self) -> Dict[str, int]:
        """Count of issues per rule ID, across every job."""
        counts: Dict[str, int] = {}
        for issue in self.issues:
            counts[issue.rule_id] = counts.get(issue.rule_id, 0) + 1
        return counts


class WorkExperienceValidator:
    """Deterministic validator for WorkExperienceExtractor's output.

    Applies WEV-001, WEV-002, WEV-003, WEV-011, and WEV-013 (Phase 1,
    Milestone 17.6), then WEV-015 (Milestone 17.8), then WEV-016/017/018
    (Milestone 23.15), to every job in a work_experience array. WEV-001
    through WEV-013 are pure Python logic against the array's own
    fields and ignore `markdown` entirely; WEV-015 and WEV-016/017/018
    read it, via WorkExperienceLocator. There are no model calls or
    retries anywhere in this class. Stateless: a single instance can be
    reused freely across resumes.
    """

    def __init__(self, locator: Optional[WorkExperienceLocator] = None) -> None:
        """Args:
        locator: Locates each job's own markdown span, for WEV-015.
            Defaults to a new WorkExperienceLocator().
        """
        self.locator = locator or WorkExperienceLocator()

    def validate(
        self, markdown: str, work_experience_json: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], ValidationReport]:
        """Validate (and where safe, correct) a work_experience array.

        Args:
            markdown: The resume's full source markdown text. Used only
                by WEV-015 (Milestone 17.8), to locate each job's own
                span and check its technologies against that span. Every
                other rule ignores this parameter.
            work_experience_json: The array exactly as
                WorkExperienceExtractor returned it (a list of job
                dicts, each with the ten fields
                schemas/modules/work_experience_schema.json requires).

        Returns:
            A (validated_work_experience, report) tuple. The returned
            array is a new list of new dicts — `work_experience_json`
            itself and its job dicts are never mutated.
        """
        validated = [dict(job) for job in work_experience_json]
        report = ValidationReport(total_jobs=len(validated))

        for index, job in enumerate(validated):
            self._apply_wev_001(index, job, report)
            self._apply_wev_002(index, job, report)
            self._apply_wev_003(index, job, report)
            self._apply_wev_011(index, job, report)
            self._apply_wev_013(index, job, report)

        # WEV-015 and WEV-016/017/018 (Milestone 23.15) run in one
        # shared second pass, after every job has already been through
        # WEV-001..013: matching needs every job's (possibly-corrected)
        # fields up front, and — since Milestone 17.10 —
        # WorkExperienceLocator.match_jobs_to_spans() itself scores
        # every (job, span) pair against every other, so it needs the
        # whole `validated` list at once rather than one job at a time.
        # The new rules reuse this exact same locate()/match_jobs_to_
        # spans() call and its span_by_job_index result — no second,
        # separate lookup.
        spans = self.locator.locate(markdown) if markdown else []
        span_by_job_index = self.locator.match_jobs_to_spans(validated, spans)
        for index, job in enumerate(validated):
            self._apply_wev_015(index, job, report, span_by_job_index.get(index))
            self._apply_provenance_rules(index, job, report, span_by_job_index.get(index))

        return validated, report

    @staticmethod
    def _apply_wev_001(index: int, job: Dict[str, Any], report: ValidationReport) -> None:
        """WEV-001: end_date must never contain an ongoing-status word —
        that belongs exclusively in currently_working."""
        end_date = job.get("end_date")
        if not isinstance(end_date, str):
            return
        if end_date.strip().lower() in _ONGOING_SENTINELS:
            original = end_date
            job["end_date"] = None
            job["currently_working"] = True
            report.issues.append(
                ValidationIssue(
                    job_index=index,
                    rule_id="WEV-001",
                    severity="Medium",
                    message=(
                        f"end_date was the ongoing-status word {original!r}; "
                        "set end_date=null and currently_working=true."
                    ),
                    auto_fixed=True,
                )
            )

    @staticmethod
    def _apply_wev_002(index: int, job: Dict[str, Any], report: ValidationReport) -> None:
        """WEV-002: null out a duration that exactly equals the computed
        date-difference between start_date and end_date.

        Phase 1 has no source markdown to check whether this duration
        was genuinely stated, so every exact match is treated as
        computed (see the module docstring). Skipped entirely if
        start_date, end_date, or duration is missing or unparseable,
        since there is then nothing to compare against.
        """
        duration = job.get("duration")
        if not isinstance(duration, str) or not duration.strip():
            return

        start = _parse_year_month(job.get("start_date"))
        end = _parse_year_month(job.get("end_date"))
        if start is None or end is None:
            return

        stated_months = _parse_duration_to_months(duration)
        if stated_months is None:
            return

        computed_months = _months_between(start, end)
        if stated_months == computed_months:
            original = duration
            job["duration"] = None
            report.issues.append(
                ValidationIssue(
                    job_index=index,
                    rule_id="WEV-002",
                    severity="High",
                    message=(
                        f"duration {original!r} exactly matches the "
                        f"{computed_months}-month gap between start_date and "
                        "end_date; treated as computed and set to null "
                        "(Phase 1 cannot verify it was explicitly stated in "
                        "the resume)."
                    ),
                    auto_fixed=True,
                )
            )

    @staticmethod
    def _apply_wev_003(index: int, job: Dict[str, Any], report: ValidationReport) -> None:
        """WEV-003: remove an employment_type outside the fixed
        whitelist. Never replaces it with a guessed value — only null
        or the value already present."""
        employment_type = job.get("employment_type")
        if not isinstance(employment_type, str) or not employment_type.strip():
            return
        if employment_type.strip().lower() not in _ALLOWED_EMPLOYMENT_TYPES:
            original = employment_type
            job["employment_type"] = None
            report.issues.append(
                ValidationIssue(
                    job_index=index,
                    rule_id="WEV-003",
                    severity="Low",
                    message=(
                        f"employment_type {original!r} is not one of the "
                        "allowed values; set to null."
                    ),
                    auto_fixed=True,
                )
            )

    @staticmethod
    def _apply_wev_011(index: int, job: Dict[str, Any], report: ValidationReport) -> None:
        """WEV-011: flag (never correct) a start_date that is
        chronologically after end_date."""
        start = _parse_year_month(job.get("start_date"))
        end = _parse_year_month(job.get("end_date"))
        if start is None or end is None:
            return
        if start > end:
            report.issues.append(
                ValidationIssue(
                    job_index=index,
                    rule_id="WEV-011",
                    severity="Medium",
                    message=(
                        f"start_date {job.get('start_date')!r} is after "
                        f"end_date {job.get('end_date')!r}."
                    ),
                    auto_fixed=False,
                )
            )

    @staticmethod
    def _apply_wev_015(
        index: int, job: Dict[str, Any], report: ValidationReport, span: Optional[JobSpan]
    ) -> None:
        """WEV-015: remove any technology that cannot be found (even as
        a loose variant) inside this job's own located markdown span.

        Never adds a technology, never searches outside `span` (in
        particular, never the Skills section — it's structurally
        outside every job's located span, since
        WorkExperienceLocator's section boundaries stop before it).
        If `span` is None (no confident job-to-span match — see
        WorkExperienceLocator.match_jobs_to_spans(), which `validate()`
        calls to build the span-by-job-index mapping this method
        receives), technologies are left completely unchanged and a
        Warning is emitted instead of guessing. This check itself
        (present-in-span-text -> keep; absent -> remove) is unchanged
        since Milestone 17.8 — only how `span` gets chosen changed, in
        Milestone 17.10.
        """
        technologies = job.get("technologies")
        if not isinstance(technologies, list) or not technologies:
            return

        if span is None:
            report.issues.append(
                ValidationIssue(
                    job_index=index,
                    rule_id="WEV-015",
                    severity="Warning",
                    message=(
                        "Could not confidently match this job to a markdown "
                        "work-experience span; technologies left unchanged."
                    ),
                    auto_fixed=False,
                )
            )
            return

        kept: List[Any] = []
        removed: List[Any] = []
        for technology in technologies:
            if _technology_present_in_span(technology, span.markdown_text):
                kept.append(technology)
            else:
                removed.append(technology)

        if removed:
            job["technologies"] = kept
            report.issues.append(
                ValidationIssue(
                    job_index=index,
                    rule_id="WEV-015",
                    severity="Medium",
                    message=(
                        f"Removed {len(removed)} technology item(s) not found "
                        f"in this job's own markdown span (match confidence "
                        f"{span.confidence:.2f}): {removed!r}."
                    ),
                    auto_fixed=True,
                    confidence=span.confidence,
                )
            )

    @staticmethod
    def _apply_provenance_rules(
        index: int, job: Dict[str, Any], report: ValidationReport, span: Optional[JobSpan]
    ) -> None:
        """WEV-016/017/018 (Milestone 23.15): null `employment_type`,
        `location`, and `duration` respectively if the job has a
        confidently-matched span and the field's value cannot be found
        (as a plain normalized substring — see the module docstring for
        why this differs from WEV-015's whole-token check) anywhere
        inside that job's own span.

        Mirrors WEV-015's own safety rules exactly, for all three
        fields: never adds or guesses a replacement value (`null` is
        the only correction ever made); never searches outside the
        matched span; and if `span` is None (no confident job-to-span
        match), every one of these fields is left completely unchanged
        and a single Warning-severity issue is emitted instead of
        guessing — the same behavior WEV-015 already has for exactly
        this situation, applied here to a job that has no company/
        designation/date signal strong enough to locate at all.
        """
        if span is None:
            has_any_value = any(
                isinstance(job.get(field_name), str) and job[field_name].strip()
                for field_name, _rule_id, _label in _PROVENANCE_FIELDS
            )
            if has_any_value:
                report.issues.append(
                    ValidationIssue(
                        job_index=index,
                        rule_id="WEV-016/017/018",
                        severity="Warning",
                        message=(
                            "Could not confidently match this job to a markdown "
                            "work-experience span; employment_type/location/"
                            "duration left unchanged."
                        ),
                        auto_fixed=False,
                    )
                )
            return

        for field_name, rule_id, label in _PROVENANCE_FIELDS:
            value = job.get(field_name)
            if not isinstance(value, str) or not value.strip():
                continue
            if _value_present_in_span(value, span.markdown_text):
                continue
            original = value
            job[field_name] = None
            report.issues.append(
                ValidationIssue(
                    job_index=index,
                    rule_id=rule_id,
                    severity="Medium",
                    message=(
                        f"{label} {original!r} not found in this job's own "
                        f"markdown span (match confidence {span.confidence:.2f}); "
                        f"set {label}=null."
                    ),
                    auto_fixed=True,
                    confidence=span.confidence,
                )
            )

    @staticmethod
    def _apply_wev_013(index: int, job: Dict[str, Any], report: ValidationReport) -> None:
        """WEV-013: a stated end_date always wins over currently_working
        — never mark a role with a real end date as currently ongoing."""
        if job.get("currently_working") is True and job.get("end_date") is not None:
            job["currently_working"] = False
            report.issues.append(
                ValidationIssue(
                    job_index=index,
                    rule_id="WEV-013",
                    severity="Critical",
                    message=(
                        "currently_working was true despite end_date="
                        f"{job.get('end_date')!r}; set currently_working=false."
                    ),
                    auto_fixed=True,
                )
            )


def _parse_year_month(date_value: Any) -> Optional[Tuple[int, int]]:
    """Parse a date string into a (year, month) tuple, or None if it's
    missing, not a string, or unparseable. Uses a fixed default date
    (_DATE_PARSE_DEFAULT) so a missing day or year component never
    depends on today's real date, keeping every rule fully
    deterministic."""
    if not isinstance(date_value, str) or not date_value.strip():
        return None
    try:
        parsed = date_parser.parse(date_value, default=_DATE_PARSE_DEFAULT)
    except (ValueError, OverflowError, TypeError):
        return None
    return parsed.year, parsed.month


def _months_between(start: Tuple[int, int], end: Tuple[int, int]) -> int:
    """Whole calendar months between two (year, month) tuples. Day-of-
    month is deliberately not considered, since it's frequently unknown
    (defaulted) rather than genuinely stated."""
    start_year, start_month = start
    end_year, end_month = end
    return (end_year - start_year) * 12 + (end_month - start_month)


def _technology_present_in_span(technology: Any, span_text: str) -> bool:
    """WEV-015: whether `technology` can be found, even as a loose
    variant, inside `span_text`.

    Both strings are normalized (lowercased, punctuation/whitespace
    collapsed to single spaces) so e.g. "Node.js" and "Node JS"
    normalize identically to "node js".
    The match is then a whole-token substring check (padding both sides
    with a space) rather than a raw substring check on the
    fully-concatenated text, specifically so a short technology name
    (e.g. "R", "Go") only matches when it appears as its own token in
    the span — not merely as a run of letters inside an unrelated
    word (e.g. "r" inside "for").

    A non-string or blank technology is treated as present (returns
    True — nothing meaningful to check, so it's left alone rather than
    removed).
    """
    if not isinstance(technology, str) or not technology.strip():
        return True
    technology_normalized = _NORMALIZE_PATTERN.sub(" ", technology.lower()).strip()
    if not technology_normalized:
        return True
    span_normalized = _NORMALIZE_PATTERN.sub(" ", span_text.lower()).strip()
    return f" {technology_normalized} " in f" {span_normalized} "


def _value_present_in_span(value: str, span_text: str) -> bool:
    """WEV-016/017/018: whether `value` can be found as a plain
    normalized substring inside `span_text`.

    Both strings are normalized identically to `_technology_present_in_span`
    above (lowercased, punctuation/whitespace collapsed to single
    spaces) — but, deliberately, WITHOUT that function's whole-token
    space-padding. See the module docstring for the concrete, real-
    dataset case (`Dattatray S. Phad`'s "Freelance" derived from
    "Nex-Z Solutions (Freelancer)") that a whole-token check would
    wrongly fail. `employment_type`/`location`/`duration` values are
    typically multi-character phrases (e.g. "Full-time", "Mumbai,
    India", "2.8 Years"), not the short, generic, collision-prone
    single tokens (e.g. "R", "Go") WEV-015's token-padding exists to
    protect against — so a plain substring check is the right,
    evidence-based choice for these three fields specifically, not a
    new strategy adopted merely to pass more examples.
    """
    value_normalized = _NORMALIZE_PATTERN.sub(" ", value.lower()).strip()
    if not value_normalized:
        return True
    span_normalized = _NORMALIZE_PATTERN.sub(" ", span_text.lower()).strip()
    return value_normalized in span_normalized


def _parse_duration_to_months(duration: str) -> Optional[int]:
    """Parse a human-written duration string (e.g. "2 yrs 3 mos", "8
    years and 1 month", "7 Months", "2.8 Years") into a total number of
    months. Returns None if neither a years nor a months component is
    found, since there's then nothing numeric to compare."""
    years_match = _YEARS_PATTERN.search(duration)
    months_match = _MONTHS_PATTERN.search(duration)
    if years_match is None and months_match is None:
        return None
    years = float(years_match.group(1)) if years_match else 0.0
    months = float(months_match.group(1)) if months_match else 0.0
    return round(years * 12 + months)
