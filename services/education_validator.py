"""
services/education_validator.py

Education Validator — Phase 1 (Milestone 18.10).

Deterministic post-extraction validation for EducationExtractor's
output, implementing only the "Quick Win" rules
docs/EDUCATION_VALIDATION.md (Milestone 18.9) scoped for a first build —
the same "Quick Win" tier docs/WORK_EXPERIENCE_VALIDATION.md (Milestone
17.5) established as the precedent this document explicitly follows:

    EDV-001  an ongoing-status word ("Pursuing", "Ongoing", "In
              Progress") or a bare completion word ("Completed")
              literally inside end_date
              -> auto-fixed (ongoing word: end_date=null,
                 currently_studying=true; completion word: end_date=
                 null only, currently_studying left as-is)
    EDV-008  grade that exactly matches, or is a verbatim substring
              of, degree or specialization
              -> auto-fixed (grade=null)
    EDV-009  institution that exactly matches, or contains all of,
              degree's own text
              -> auto-fixed (institution=null)
    EDV-010  a stated date range dumped whole into start_date (sub-
              pattern 1), a single date duplicated into both
              start_date and end_date (sub-pattern 2), or a single
              lone date left only in start_date with end_date null
              (sub-pattern 3, Milestone 23.24)
              -> auto-fixed (split the range; move the duplicated/
                 lone value to end_date and null start_date)
    EDV-011  currently_studying=true despite a non-null, non-blank
              grade (boolean-only fix — see below)
              -> auto-fixed (currently_studying=false; end_date is
                 never reconstructed)

Every other rule docs/EDUCATION_VALIDATION.md §2/§3 documents
(EDV-002 through EDV-007, EDV-012, and the markdown-aware/cross-module/
human-review-oriented variants of any rule above) needs source
markdown, a different module's already-extracted output, or a human
review surface that doesn't exist yet — none of that is implemented in
this milestone, per docs/EDUCATION_VALIDATION.md §3's own "Quick Wins"
tier and §7's Phase 1 implementation plan.

This module knows nothing about ParserPipeline or Ollama. It takes an
education array (already extracted) and returns a corrected array plus
a report of every issue found — a pure, in-memory, side-effect-free
transformation. The input list (and its entry dicts) are never mutated;
validate() always returns new objects.

EDV-010's scope: sub-pattern 1 (a stated range like "2008-2010"
landing whole in start_date), sub-pattern 2 (a single date duplicated
into both start_date and end_date), and, as of Milestone 23.24,
sub-pattern 3 (a single lone date left only in start_date, end_date
null) are all implemented, all unambiguous, pure string/date-shape
operations.

Sub-pattern 3's history is worth recording plainly: `docs/
EDUCATION_VALIDATION.md` (Milestone 18.9) originally deferred it —
"moving a single date from start_date to end_date without confirming
via markdown risks being wrong exactly as often as right; flag rather
than guess." That reasoning was sound *given the evidence available at
the time* (no dataset-wide scan had been done). Milestone 23.24
performed exactly that scan — a full, direct comparison of every
education entry's start_date/end_date across this project's own
23-resume ground truth against current parser output — and found this
pattern is **not** a coin flip in this dataset: all 27 observed
instances, across 9 resumes, are one-directional (ground truth always
has the lone date in end_date; the model always puts it in start_date,
verbatim, unchanged), and **zero** ground-truth education entries
anywhere in the dataset have the opposite shape (a real start_date
with a genuinely null end_date) — including the dataset's one true
currently_studying=true entry, which has *both* dates null, not a lone
start_date. See docs/MILESTONE_23_24_EDUCATION_DATE_SWAP_RESULTS.md
for the full investigation. This rule remains narrow: it only ever
moves a value already present, never invents one, and only fires when
end_date is genuinely empty (a populated end_date of any kind, even a
value sub-pattern 1/2 would also match, is handled by those rules
first, in the order below, and this rule never overwrites an
already-populated end_date).

EDV-011's known, deliberate Phase 1 scope: only the boolean
(currently_studying) is corrected. The real end_date this entry should
have had is never reconstructed here — docs/EDUCATION_VALIDATION.md §2
is explicit that recovering it would need markdown-aware re-extraction,
out of this rule's scope.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from services.education_locator import EducationLocator

logger = logging.getLogger(__name__)

# EDV-001: ongoing-status words that must never appear literally in
# end_date -- an entry still in progress belongs exclusively in
# currently_studying. Mirrors work_experience_validator's own
# _ONGOING_SENTINELS list (present/current/till date/now), plus the
# two education-specific ongoing words docs/EDUCATION_VALIDATION.md's
# EDV-001 spec names explicitly (Pursuing, In Progress).
_ONGOING_SENTINELS = frozenset(
    {"pursuing", "ongoing", "in progress", "present", "current", "till date", "now"}
)

# EDV-001: a bare completion word doesn't itself confirm the entry is
# *not* ongoing (that's currently_studying's own job) -- it only means
# end_date, as a literal word rather than a real date, must be nulled.
_COMPLETION_SENTINELS = frozenset({"completed"})

# EDV-010 sub-pattern 1: a stated year range ("2008-2010", "2008–2010")
# dumped whole into start_date instead of being split into start_date/
# end_date.
_DATE_RANGE_PATTERN = re.compile(r"^\s*(\d{4})\s*[-–—]\s*(\d{4})\s*$")

# EDV-008: the minimum normalized length a *substring* (not exact)
# match must have before it's treated as a real echo, so a trivial
# short string (e.g. a single letter) can't falsely trigger against
# unrelated field text.
_MIN_SUBSTRING_ECHO_LENGTH = 3

# EDV-012 (Milestone V2.6D): whole-word normalization, mirroring
# services/work_experience_locator.py's own _normalize_text/
# _text_present_in_block helpers exactly (not reinvented) -- lowercase,
# collapse non-alphanumeric runs to a single space, then check
# substring presence padded with spaces so a short value only matches
# its own whole token, never a run of letters inside an unrelated word.
_EDV_012_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass
class ValidationIssue:
    """One rule's finding for one education entry.

    Mirrors services/work_experience_validator.py's ValidationIssue
    shape exactly, per docs/EDUCATION_VALIDATION.md §4's instruction to
    reuse that reporting structure rather than invent a new one --
    only the job-specific field name (job_index) is renamed to this
    module's own entry_index, since an education array holds entries,
    not jobs.

    Attributes:
        entry_index: Index of the affected entry within the education
            array (post-validation order -- this validator never
            reorders or removes entries, so the index is stable).
        rule_id: The EDV-xxx rule that produced this issue.
        severity: Human-readable severity, matching the per-rule
            severity documented in docs/EDUCATION_VALIDATION.md.
        message: A human-readable description of what was found (and,
            for auto-fixed issues, what was changed).
        auto_fixed: Whether this validator corrected the entry in
            place (True) or only flagged it, leaving the entry
            unchanged (False -- no Phase 1 rule currently does this,
            since every Quick Win rule is auto-fixable, but the field
            is kept for shape-parity with WorkExperienceValidator and
            for any future rule added to this same class).
    """

    entry_index: int
    rule_id: str
    severity: str
    message: str
    auto_fixed: bool


@dataclass
class ValidationReport:
    """Aggregate result of validating one resume's education array.

    Mirrors services/work_experience_validator.py's ValidationReport
    shape exactly (total_jobs -> total_entries).

    Attributes:
        total_entries: Number of education entries validated.
        issues: Every ValidationIssue found, in the order rules were
            applied (EDV-001, EDV-008, EDV-009, EDV-010, EDV-011 for
            entry 0, then the same order for entry 1, and so on).
    """

    total_entries: int
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def auto_fixed_count(self) -> int:
        """Number of issues this validator corrected in place."""
        return sum(1 for issue in self.issues if issue.auto_fixed)

    @property
    def flagged_count(self) -> int:
        """Number of issues recorded but left uncorrected."""
        return sum(1 for issue in self.issues if not issue.auto_fixed)

    def issues_by_rule(self) -> Dict[str, int]:
        """Count of issues per rule ID, across every entry."""
        counts: Dict[str, int] = {}
        for issue in self.issues:
            counts[issue.rule_id] = counts.get(issue.rule_id, 0) + 1
        return counts


class EducationValidator:
    """Deterministic validator for EducationExtractor's output.

    Applies EDV-001, EDV-008, EDV-009, EDV-010 (sub-patterns 1/2 only),
    and EDV-011 (boolean-only) -- Phase 1's Quick Win rules, per
    docs/EDUCATION_VALIDATION.md §3/§7 -- to every entry in an
    education array. Every one of those rules is pure Python logic
    against the array's own fields; none reads source markdown, a
    different module's output, or calls a model. Stateless: a single
    instance can be reused freely across resumes.

    Milestone V2.6D adds EDV-012 (markdown-aware, additive only -- see
    that method's own docstring): when the resume's source markdown is
    supplied, each entry is checked against EducationLocator's
    candidate spans for corroborating evidence. This is the first rule
    in this class that reads source markdown, filling exactly the slot
    the module docstring above already reserved ("EDV-002 through
    EDV-007, EDV-012... need source markdown"). It is purely additive:
    it never modifies, flags for removal, or otherwise constrains any
    entry -- see EDV-012's own docstring for why "locator found no
    evidence" is never treated as evidence the LLM is wrong (Milestone
    V2.6D's central, corpus-verified finding).
    """

    def __init__(self, education_locator: Optional[EducationLocator] = None):
        """
        Args:
            education_locator: Defaults to a new EducationLocator().
                Only used by EDV-012, and only when validate() is
                called with markdown_text.
        """
        self.education_locator = education_locator or EducationLocator()

    def validate(
        self,
        education_json: List[Dict[str, Any]],
        markdown_text: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], ValidationReport]:
        """Validate (and where safe, correct) an education array.

        Args:
            education_json: The array exactly as EducationExtractor
                returned it (a list of entry dicts, each with the nine
                fields schemas/modules/education_schema.json
                requires).
            markdown_text: The resume's full source markdown (Milestone
                V2.6D). Optional and defaults to None for backward
                compatibility with every existing caller -- when
                omitted, behavior is byte-for-byte identical to before
                this milestone (EDV-012 simply does not run). Mirrors
                WorkExperienceValidator.validate()'s own
                (markdown_text, jobs) precedent.

        Returns:
            A (validated_education, report) tuple. The returned array
            is a new list of new dicts -- `education_json` itself and
            its entry dicts are never mutated.
        """
        validated = [dict(entry) for entry in education_json]
        report = ValidationReport(total_entries=len(validated))

        for index, entry in enumerate(validated):
            entry_is_confirmed_ongoing = self._apply_edv_001(index, entry, report)
            self._apply_edv_008(index, entry, report)
            self._apply_edv_009(index, entry, report)
            self._apply_edv_010(index, entry, report, skip_sub_pattern_3=entry_is_confirmed_ongoing)
            self._apply_edv_011(index, entry, report)

        if markdown_text:
            self._apply_edv_012(validated, markdown_text, report)

        return validated, report

    @staticmethod
    def _apply_edv_001(index: int, entry: Dict[str, Any], report: ValidationReport) -> bool:
        """EDV-001: end_date must never hold a literal ongoing-status
        word (that belongs exclusively in currently_studying) or a
        bare completion word (which isn't a real date either).

        Returns:
            True only when the ongoing-status-word branch fired (i.e.
            entry['currently_studying'] was just set True *because of
            an explicit ongoing-status word in end_date*, a strong,
            source-confirmed signal). Milestone 23.24's EDV-010
            sub-pattern 3 uses this to skip an entry whose start_date-
            only, end_date-null shape was just deliberately created
            here to represent a genuinely ongoing entry -- not the
            model-omission shape sub-pattern 3 targets. The bare-
            completion-word branch returns False: "Completed" with no
            real date is a different situation (see that branch's own
            docstring text) and does not by itself confirm the entry
            is ongoing, so it must not suppress sub-pattern 3.
        """
        end_date = entry.get("end_date")
        if not isinstance(end_date, str):
            return False
        normalized = end_date.strip().lower()

        if normalized in _ONGOING_SENTINELS:
            original = end_date
            entry["end_date"] = None
            entry["currently_studying"] = True
            report.issues.append(
                ValidationIssue(
                    entry_index=index,
                    rule_id="EDV-001",
                    severity="Medium",
                    message=(
                        f"end_date was the ongoing-status word {original!r}; "
                        "set end_date=null and currently_studying=true."
                    ),
                    auto_fixed=True,
                )
            )
            return True
        elif normalized in _COMPLETION_SENTINELS:
            original = end_date
            entry["end_date"] = None
            report.issues.append(
                ValidationIssue(
                    entry_index=index,
                    rule_id="EDV-001",
                    severity="Medium",
                    message=(
                        f"end_date was the bare completion word {original!r}, "
                        "not a real date; set end_date=null "
                        "(currently_studying left unchanged -- a completion "
                        "word doesn't itself confirm the entry isn't ongoing)."
                    ),
                    auto_fixed=True,
                )
            )
        return False

    @staticmethod
    def _apply_edv_008(index: int, entry: Dict[str, Any], report: ValidationReport) -> None:
        """EDV-008: null out a grade that exactly matches, or is a
        verbatim substring of, degree or specialization -- a
        duplicated value, not a real grade."""
        grade = entry.get("grade")
        if not isinstance(grade, str) or not grade.strip():
            return
        grade_normalized = grade.strip().lower()

        for sibling_field in ("degree", "specialization"):
            sibling_value = entry.get(sibling_field)
            if not isinstance(sibling_value, str) or not sibling_value.strip():
                continue
            sibling_normalized = sibling_value.strip().lower()

            is_exact_match = grade_normalized == sibling_normalized
            is_substring_match = (
                len(grade_normalized) >= _MIN_SUBSTRING_ECHO_LENGTH
                and grade_normalized in sibling_normalized
            )
            if is_exact_match or is_substring_match:
                original = grade
                entry["grade"] = None
                report.issues.append(
                    ValidationIssue(
                        entry_index=index,
                        rule_id="EDV-008",
                        severity="Medium",
                        message=(
                            f"grade {original!r} duplicates {sibling_field} "
                            f"{sibling_value!r}, not a real grade; set "
                            "grade=null."
                        ),
                        auto_fixed=True,
                    )
                )
                return

    @staticmethod
    def _apply_edv_009(index: int, entry: Dict[str, Any], report: ValidationReport) -> None:
        """EDV-009: null out an institution that exactly matches, or
        contains all of, degree's own text -- a duplicated value, not
        a real institution."""
        institution = entry.get("institution")
        degree = entry.get("degree")
        if not isinstance(institution, str) or not institution.strip():
            return
        if not isinstance(degree, str) or not degree.strip():
            return

        institution_normalized = institution.strip().lower()
        degree_normalized = degree.strip().lower()

        if institution_normalized == degree_normalized or (
            degree_normalized in institution_normalized
        ):
            original = institution
            entry["institution"] = None
            report.issues.append(
                ValidationIssue(
                    entry_index=index,
                    rule_id="EDV-009",
                    severity="High",
                    message=(
                        f"institution {original!r} duplicates degree "
                        f"{degree!r}, not a real institution; set "
                        "institution=null."
                    ),
                    auto_fixed=True,
                )
            )

    @staticmethod
    def _apply_edv_010(
        index: int,
        entry: Dict[str, Any],
        report: ValidationReport,
        skip_sub_pattern_3: bool = False,
    ) -> None:
        """EDV-010: split a stated date range dumped whole into
        start_date (sub-pattern 1), move a single date duplicated into
        both start_date and end_date over to end_date alone (sub-
        pattern 2), or move a single lone date left only in start_date
        over to end_date (sub-pattern 3, Milestone 23.24) -- see the
        module docstring for the dataset-wide evidence behind
        sub-pattern 3 specifically.

        Sub-patterns are checked in this order, and each returns
        immediately on a match, so a single entry is never touched by
        more than one sub-pattern in the same call.

        Args:
            skip_sub_pattern_3: True when this same validate() call's
                EDV-001 step just set currently_studying=True on this
                entry because of an explicit ongoing-status word in
                end_date -- a genuinely ongoing entry legitimately has
                only a start_date and a null end_date, which is
                exactly sub-pattern 3's trigger shape. Sub-patterns 1
                and 2 are unaffected by this flag; only sub-pattern 3
                has this ambiguity.
        """
        start_date = entry.get("start_date")
        end_date = entry.get("end_date")

        if isinstance(start_date, str):
            range_match = _DATE_RANGE_PATTERN.match(start_date)
            if range_match is not None:
                end_is_confirming = end_date is None or (
                    isinstance(end_date, str) and end_date.strip() and end_date.strip() in start_date
                )
                if end_is_confirming:
                    first_year, second_year = range_match.group(1), range_match.group(2)
                    original_start = start_date
                    original_end = end_date
                    entry["start_date"] = first_year
                    entry["end_date"] = second_year
                    report.issues.append(
                        ValidationIssue(
                            entry_index=index,
                            rule_id="EDV-010",
                            severity="Medium",
                            message=(
                                f"start_date {original_start!r} was a date "
                                f"range (end_date was {original_end!r}); "
                                f"split into start_date={first_year!r}, "
                                f"end_date={second_year!r}."
                            ),
                            auto_fixed=True,
                        )
                    )
                    return

        if (
            isinstance(start_date, str)
            and isinstance(end_date, str)
            and start_date.strip()
            and start_date.strip() == end_date.strip()
        ):
            original_start = start_date
            entry["start_date"] = None
            report.issues.append(
                ValidationIssue(
                    entry_index=index,
                    rule_id="EDV-010",
                    severity="Medium",
                    message=(
                        f"start_date and end_date were both {original_start!r} "
                        "(a single date duplicated into both fields); set "
                        "start_date=null, leaving the value in end_date."
                    ),
                    auto_fixed=True,
                )
            )
            return

        # Sub-pattern 3 (Milestone 23.24): a single lone date left only
        # in start_date, with end_date genuinely null -- the dataset-
        # wide pattern this rule now corrects. Only fires when
        # start_date is a real, non-blank string and end_date is
        # exactly None (not an empty string, not any other falsy
        # value that isn't the schema's actual null) -- an end_date
        # that already holds any string, however unrelated, is never
        # overwritten by this rule; sub-patterns 1/2 above have
        # already had first claim on any string-shaped start_date, so
        # reaching this point means neither matched. Never fires when
        # skip_sub_pattern_3 is True (this same entry's end_date was
        # just deliberately nulled by EDV-001 because of an explicit
        # ongoing-status word -- a genuinely ongoing entry, not the
        # model-omission shape this rule targets).
        if (
            not skip_sub_pattern_3
            and isinstance(start_date, str)
            and start_date.strip()
            and end_date is None
        ):
            original_start = start_date
            entry["start_date"] = None
            entry["end_date"] = original_start
            report.issues.append(
                ValidationIssue(
                    entry_index=index,
                    rule_id="EDV-010",
                    severity="Medium",
                    message=(
                        f"start_date was the lone date {original_start!r} with "
                        "end_date null; moved the value to end_date and set "
                        "start_date=null (Milestone 23.24: this dataset's own "
                        "ground truth never places a real start_date-only, "
                        "end_date-null entry -- see "
                        "docs/MILESTONE_23_24_EDUCATION_DATE_SWAP_RESULTS.md)."
                    ),
                    auto_fixed=True,
                )
            )

    @staticmethod
    def _apply_edv_011(index: int, entry: Dict[str, Any], report: ValidationReport) -> None:
        """EDV-011 (boolean-only): currently_studying=true is strong
        evidence of a rule violation when the entry also has a
        non-null, non-blank grade -- a completed, graded result and
        "still studying" are close to mutually exclusive for the same
        entry. Only the boolean is corrected; end_date is never
        reconstructed here (see the module docstring)."""
        grade = entry.get("grade")
        if entry.get("currently_studying") is True and isinstance(grade, str) and grade.strip():
            entry["currently_studying"] = False
            report.issues.append(
                ValidationIssue(
                    entry_index=index,
                    rule_id="EDV-011",
                    severity="Critical",
                    message=(
                        f"currently_studying was true despite grade="
                        f"{grade!r}; set currently_studying=false "
                        "(end_date is not reconstructed by this rule)."
                    ),
                    auto_fixed=True,
                )
            )

    @staticmethod
    def _edv_012_normalize(text: str) -> str:
        """Lowercase and collapse any run of non-alphanumeric characters
        to a single space -- identical normalization to
        services/work_experience_locator.py's own _normalize_text, not
        reinvented."""
        return _EDV_012_NORMALIZE_PATTERN.sub(" ", text.lower()).strip()

    @classmethod
    def _edv_012_present_in(cls, value: Optional[str], blob: str) -> bool:
        """True if `value`, normalized, appears as a whole-token
        substring inside `blob` (also normalized) -- identical
        mechanism to work_experience_locator.py's own
        _text_present_in_block: space-padded so a short value only
        matches its own token, never a run of letters inside an
        unrelated word. False (not None) when `value` is missing/blank
        -- EDV-012 treats "this entry doesn't state an institution" as
        "nothing to corroborate for that field," never as a strike
        against the entry (see _apply_edv_012's own docstring)."""
        if not isinstance(value, str) or not value.strip():
            return False
        norm_value = cls._edv_012_normalize(value)
        if not norm_value:
            return False
        norm_blob = cls._edv_012_normalize(blob)
        return f" {norm_value} " in f" {norm_blob} "

    def _apply_edv_012(
        self,
        entries: List[Dict[str, Any]],
        markdown_text: str,
        report: ValidationReport,
    ) -> None:
        """EDV-012 (Milestone V2.6D): markdown-aware, PURELY ADDITIVE
        corroboration evidence -- never modifies, flags for removal, or
        otherwise constrains any entry. Records an Info-severity,
        auto_fixed=False issue for an entry when EducationLocator finds
        a single candidate span in the source markdown whose own raw
        text contains BOTH this entry's degree AND (when the entry
        states one) institution, verbatim as whole tokens.

        This rule intentionally does NOT have a corresponding
        "contradiction" or "unsupported" counterpart. Milestone V2.6D's
        own corpus-wide simulation (docs/
        MILESTONE_V2_6D_EDUCATION_VALIDATOR_READINESS.md, Step 4) found
        that "EducationLocator did not find supporting evidence for
        this entry" occurs for reasons that have NOTHING to do with
        whether the entry is correct: the section was never located at
        all (a Stage-1 extraction problem), the located boundary is
        wrong (a Stage-1 transcription problem, confirmed on
        AshishKumarSingh_RBM_Pune_Stonelam), the entry's own text is
        damaged by a font/ligature artifact in this representation
        (confirmed on Arjun Sharma_Data Architect_GT), or the entry's
        degree name simply falls outside the locator's recognized
        keyword vocabulary (confirmed on
        Nilesh-Shivram-Magare-CV-Updated (1)). Every one of these is a
        locator/Stage-1 limitation, not evidence the LLM's entry is
        wrong -- treating "not found" as "unsupported" would risk
        incorrectly removing a real, correct entry, exactly the
        high-severity regression this milestone's own investigation was
        built to guard against. Only the positive, corroborating
        direction is implemented.

        Args:
            entries: The (already Phase-1-validated) education array
                for this resume.
            markdown_text: The resume's full source markdown.
            report: Mutated in place -- one Info issue appended per
                entry with corroborating evidence found; entries
                without it receive no issue at all (silence, not a
                flag).
        """
        result = self.education_locator.locate(markdown_text)
        if not result.section_found or not result.candidates:
            return

        for index, entry in enumerate(entries):
            degree = entry.get("degree")
            institution = entry.get("institution")
            if not isinstance(degree, str) or not degree.strip():
                continue

            for candidate in result.candidates:
                degree_found = self._edv_012_present_in(degree, candidate.raw_text)
                institution_found = (
                    self._edv_012_present_in(institution, candidate.raw_text)
                    if isinstance(institution, str) and institution.strip()
                    else True  # entry states no institution -- nothing to corroborate for that field
                )
                if degree_found and institution_found:
                    report.issues.append(
                        ValidationIssue(
                            entry_index=index,
                            rule_id="EDV-012",
                            severity="Info",
                            message=(
                                f"Corroborated by source markdown: degree {degree!r}"
                                + (f" and institution {institution!r}" if isinstance(institution, str) and institution.strip() else "")
                                + " both found verbatim in a located Education candidate span. "
                                "Informational only -- no field was changed."
                            ),
                            auto_fixed=False,
                        )
                    )
                    break
