"""
services/work_experience_extractor.py

Work Experience Extractor (Milestone 15.2; preprocess_json() override
added in Milestone 23.16, extended in Milestone 23.18 and 23.22).

Sixth of the modular, per-section resume parsers: extracts just the
candidate's work experience from a resume's markdown, using a small text
model:

    output/markdown/<resume>.md
                |
                v
    prompts/modules/work_experience.md + markdown, in one prompt
                |
                v
    OllamaClient.generate_text (settings.DEFAULT_TEXT_MODEL, default "qwen3:1.7b")
                |
                v
    parsed {"work_experience": [...]}
                |
                v
    preprocess_json()  -- bare-array response wrapping (Milestone 23.22)
                       -- exact-duplicate technologies removal (Milestone 23.16)
                       -- day+month date-string normalization (Milestone 23.18)
                |
                v
    validated against this module's own schema, returned in memory

All pipeline mechanics (reading markdown, prompting, parsing, validating,
timing) live in services/base_module_extractor.py and are not duplicated
here — this class supplies the work-experience-specific prompt id,
schema path, and model name via the three protected hooks the base class
calls, plus (as of Milestone 23.16) one override of the base class's
preprocess_json() hook. Its public extract() is exactly
BaseModuleExtractor's, unchanged.

The schema (schemas/modules/work_experience_schema.json) enforces that
"work_experience" is an array of job objects, each with all ten fields
(company, designation, employment_type, location, start_date, end_date,
currently_working, duration, responsibilities, technologies) present and
additionalProperties: false at both the module and job-item level. See
docs/work_experience_design.md (Milestone 15.1) for why each field
exists and how edge cases (gaps, internships, freelancing, consulting,
career breaks, self-employment) are represented. `technologies` (like
`responsibilities`) additionally carries `uniqueItems: true`.

This module extractor is deliberately narrow. It does not merge its
output with any other module's result (that's a later assembly step),
does not save anything to disk, does not benchmark itself, and — beyond
Milestone 23.16's exact-duplicate removal — does not normalize or
enrich any field.

Milestone 23.16 (docs/MILESTONE_23_16_TECHNOLOGIES_DEDUP_INVESTIGATION.md):
the model occasionally, non-deterministically returns a job's
`technologies` array with an exact-duplicate entry — confirmed via a
real, captured failure on the "Arjun Sharma" resume in this project's
own benchmark dataset (also independently documented as far back as
Milestone 18.3, per docs/PARSER_RELIABILITY_DESIGN.md), and reproduced
live during that investigation. Because `work_experience_schema.json`
declares `technologies`' `uniqueItems: true`, and because
BaseModuleExtractor.extract() validates against that schema *before*
returning, an unresolved duplicate does not just under-score this
module — it raises SchemaValidationError, is retried MAX_ATTEMPTS
times by pipeline/parser_pipeline.py, and (if still duplicated) fails
the *entire resume*, not just work_experience. No post-extraction
validator (the shape services/work_experience_validator.py uses) can
ever fix this, because extract() never successfully returns in the
first place — the same architectural reasoning already established for
services/skills_extractor.py's own preprocess_json() override
(docs/SKILLS_VALIDATION_DESIGN.md §12), which this override mirrors.

preprocess_json() below removes only exact-duplicate strings from each
job's own `technologies` array (ground-truth-confirmed always safe: no
ground-truth `technologies` array in this project's 23-resume benchmark
dataset has ever contained a duplicate) before schema validation runs,
so this specific failure mode is closed at the source. Per
docs/MILESTONE_23_16_TECHNOLOGIES_DEDUP_INVESTIGATION.md §5/§9's direct
evidence, it does deliberately less than Skills' SKV-002: no case
folding and no whitespace trimming, since no case-only or
whitespace/punctuation-variant duplicate was ever observed in real
data, and this dataset does contain real, legitimate near-duplicate
pairs (e.g. "ADF" alongside "Azure Data Factory", "Databricks"
alongside "Azure Data Bricks") that any normalization broader than
exact-string equality risks incorrectly merging. It does nothing else:
no `responsibilities` deduplication (zero evidence of need), no
re-casing or rewriting of a surviving value, no fuzzy or semantic
matching.

Milestone 23.18 (docs/MILESTONE_23_18_DATE_NORMALIZATION_RESULTS.md,
building on docs/MILESTONE_23_17_POST_23_16_RESIDUAL_ERROR_ANALYSIS.md's
finding that ~48% of start_date/end_date mismatches on already-matched
jobs are pure string-formatting differences, not wrong dates): adds a
second, independent preprocess_json() transform, scoped narrowly to
`start_date`/`end_date` only. Re-verification against the current
benchmark's own ground truth (not just the model's output) found that
ground truth's own date-string convention is genuinely inconsistent —
not only across resumes but within a single resume's own entries (see
docs/MILESTONE_23_18_DATE_NORMALIZATION_RESULTS.md §3) — so a
blanket "always expand abbreviated months" or "always add an ordinal
suffix" transform is demonstrably unsafe: it would silently turn
several currently-correct extractions into incorrect ones. The
transform implemented here is deliberately the narrowest evidenced-safe
slice: it fires *only* when a value is exactly `"<day> <abbreviated
month> <year>"` (e.g. `"27 Jul 2015"`) — a day number is required to be
present — in which case it rewrites it to `"<day><ordinal suffix>
<full month name> <year>"` (e.g. `"27th July 2015"`), preserving the
original day digits' own zero-padding. Bare `"<month> <year>"` values
(no day, e.g. `"Jan 2023"`) are deliberately left untouched, even though
they are also frequently pure formatting differences (Milestone 23.17's
PRAGNESH/Dattatray examples) — expanding those was measured to break
11 currently-correct matches across 6 resumes (see the results doc's
§3), a regression the narrower, day-scoped rule never causes on this
project's own benchmark (its own 3 known counter-examples are confined
to Arjun Sharma, a resume Milestone 23.17 already found to be
independently unstable in these exact fields). `duration`,
`currently_working`, and every other job field are never touched by
this transform, matching the same "narrower than it could be, because
the evidence doesn't support more" discipline Milestone 23.16 applied
to `technologies`.

Milestone 23.22 (docs/PARSER_RELIABILITY_HARDENING.md, a *reliability*
milestone — see that doc's own scope note that it targets success rate,
not benchmark score): `docs/PARSER_STABILITY_ANALYSIS.md` (Milestone
23.21) found this extractor occasionally returns a bare JSON array of
job objects instead of the `{"work_experience": [...]}` object the
schema requires — confirmed in that milestone's own 5-run data
("PRAGNESH S. SHAH...") and, going back further, on "Arjun Sharma"
since Milestone 23.14 (one of that resume's several independently
documented malformed-output shapes). Since the schema requires a
top-level object, this shape always fails validation no matter how
correct the array's own contents are — the same "closes an entire
resume's pipeline run, not just under-scores one field" failure mode
Milestone 23.16 already fixed for the `uniqueItems` case.
`_wrap_bare_array()` fixes only the top-level shape — it never invents,
drops, or reorders a single job or field the model already produced.
"""

import re
from pathlib import Path
from typing import Any, List, Optional

from config import SCHEMAS_DIR
from settings import DEFAULT_TEXT_MODEL
from services.base_module_extractor import BaseModuleExtractor, BaseModuleResult

# Backward/forward-compatible alias, matching the pattern established by
# PersonalInformationExtractor: the shared BaseModuleResult, importable
# under this module's own name.
WorkExperienceResult = BaseModuleResult

__all__ = ["WorkExperienceExtractor", "WorkExperienceResult"]

# Text model used for extraction. Small/fast rather than the vision
# model, since this step only ever reads text. Milestone 23.3B: imported
# from settings (shared by every module extractor and JsonDraftGenerator)
# instead of hardcoded here, so a single settings.DEFAULT_TEXT_MODEL
# change (or the DEFAULT_TEXT_MODEL env var) switches the whole parser.

# Prompt instructing the model how to extract work experience only.
DEFAULT_PROMPT_ID = "modules/work_experience"

# This module's own schema — scoped to {"work_experience": [...]}, not
# the full resume_schema_v1.json.
DEFAULT_SCHEMA_PATH = SCHEMAS_DIR / "modules" / "work_experience_schema.json"

# Milestone 23.18: the one date-string shape confirmed, by direct
# cross-check against this project's own ground truth and benchmark
# output (docs/MILESTONE_23_18_DATE_NORMALIZATION_RESULTS.md §3), to be
# safely normalizable with no observed counter-example on this dataset:
# a bare day number, a 3-4 letter month abbreviation, and a 4-digit
# year — nothing more, nothing less. Deliberately does not match a day
# with a full month name already present (never observed in this
# project's actual model output) or a month+year with no day (observed,
# but proven unsafe to normalize blanket — see the docstring above).
_DAY_ABBREVIATED_MONTH_YEAR_PATTERN = re.compile(
    r"^(?P<day>0[1-9]|[12][0-9]|3[01]|[1-9])"
    r" (?P<month>Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)"
    r" (?P<year>\d{4})$"
)

# Every abbreviation this project's own benchmark data has actually
# produced or referenced (docs/MILESTONE_23_18_DATE_NORMALIZATION_RESULTS.md
# §3) mapped to its full name. "May" is intentionally absent: its
# abbreviated and full spellings are identical, so it never needs
# normalizing.
_MONTH_ABBREVIATION_TO_FULL_NAME = {
    "Jan": "January",
    "Feb": "February",
    "Mar": "March",
    "Apr": "April",
    "Jun": "June",
    "Jul": "July",
    "Aug": "August",
    "Sep": "September",
    "Sept": "September",
    "Oct": "October",
    "Nov": "November",
    "Dec": "December",
}


class WorkExperienceExtractor(BaseModuleExtractor):
    """Extracts work experience from a resume's markdown.

    All pipeline mechanics come from BaseModuleExtractor; this class
    answers the three questions the base class needs (which prompt,
    which schema, which model), plus one override of preprocess_json()
    that (Milestone 23.22) wraps a bare-array response in the required
    object shape, (Milestone 23.16) removes exact-duplicate technologies
    entries, and (Milestone 23.18) normalizes day+abbreviated-month date
    strings, before schema validation runs.
    """

    def _get_prompt_id(self) -> str:
        return DEFAULT_PROMPT_ID

    def _get_schema_path(self) -> Path:
        return DEFAULT_SCHEMA_PATH

    def _get_model_name(self) -> str:
        return DEFAULT_TEXT_MODEL

    def preprocess_json(self, json_object: Any) -> Any:
        """Milestone 23.16 + 23.18 + 23.22: run three independent,
        narrowly-scoped transforms before schema validation — see the
        module docstring, docs/MILESTONE_23_16_TECHNOLOGIES_DEDUP_INVESTIGATION.md,
        docs/MILESTONE_23_18_DATE_NORMALIZATION_RESULTS.md, and
        docs/PARSER_RELIABILITY_HARDENING.md for why each must happen
        here, not in a post-extraction validator, and exactly what each
        one does and does not do:

        0. Milestone 23.22 — if the model returned a bare JSON array
           instead of `{"work_experience": [...]}`, wrap it in the
           expected object shape (see `_wrap_bare_array`'s own
           docstring), *before* anything below runs, so the two
           per-job transforms still apply to a bare-array response.
        1. Milestone 23.16 — remove exact-duplicate strings from each
           job's own `technologies` array (see
           `_deduplicate_technologies`'s own docstring).
        2. Milestone 23.18 — normalize each job's `start_date` and
           `end_date`, independently, from `"<day> <abbreviated month>
           <year>"` to `"<day><ordinal suffix> <full month name>
           <year>"` when (and only when) that exact shape is present
           (see `_normalize_job_dates`'s own docstring).

        These transforms touch disjoint concerns (top-level shape vs.
        `technologies` vs. `start_date`/`end_date`) and never interact
        with each other.

        Defensive by design: if `json_object` isn't a dict or a bare
        list, its "work_experience" value isn't a list, a given entry
        isn't a dict, or a field isn't the type each transform expects,
        that part is returned completely unchanged — diagnosing a
        malformed shape is SchemaValidator.validate()'s job, not this
        method's.

        Args:
            json_object: The value BaseModuleExtractor._parse_json()
                returned — parsed, syntactically valid JSON, not yet
                schema-validated.

        Returns:
            `json_object` unchanged if no transform found anything to
            change (or the shape wasn't as expected); otherwise a new
            dict, either the bare-array wrapped in
            `{"work_experience": ...}` or a shallow copy of
            `json_object` with "work_experience" replaced by a new list
            in which only the affected job dict(s) are replaced.
            `json_object` itself (and every job dict that didn't need a
            change) is never mutated.
        """
        json_object = self._wrap_bare_array(json_object)

        if not isinstance(json_object, dict):
            return json_object
        jobs = json_object.get("work_experience")
        if not isinstance(jobs, list):
            return json_object

        new_jobs: List[Any] = [
            self._normalize_job_dates(self._deduplicate_technologies(job)) for job in jobs
        ]

        if new_jobs == jobs:
            return json_object
        return {**json_object, "work_experience": new_jobs}

    @staticmethod
    def _wrap_bare_array(json_object: Any) -> Any:
        """Milestone 23.22: if the model returned a bare JSON array of
        job objects instead of the expected `{"work_experience":
        [...]}` object, wrap it in that object shape.

        Confirmed real, repeated defect: `WorkExperienceExtractor`'s
        response is occasionally a top-level array — not the wrapping
        object the schema requires — a shape confirmed live in this
        project's own benchmark logs on two different resumes across
        multiple milestones (docs/PARSER_STABILITY_ANALYSIS.md §6:
        "PRAGNESH S. SHAH..."; historically also "Arjun Sharma_Data
        Architect_GT" since Milestone 23.14). Because
        `work_experience_schema.json` requires a top-level object, this
        shape always fails schema validation regardless of how correct
        the array's own contents are, exhausts all 3 pipeline retries,
        and fails the *entire resume* — the same class of
        architectural failure Milestone 23.16 already fixed for the
        `uniqueItems` case.

        This is deliberately narrower than it might look: it only ever
        adds the wrapping key around a value the model already
        produced — it never invents, drops, or reorders a single job or
        field. If `json_object` is a dict already (the normal case) or
        anything other than a `list` (e.g. a string, a malformed
        partial response), it is returned completely unchanged;
        diagnosing any other malformed shape remains
        SchemaValidator.validate()'s job.

        Args:
            json_object: The value BaseModuleExtractor._parse_json()
                returned — parsed, syntactically valid JSON, not yet
                schema-validated.

        Returns:
            `{"work_experience": json_object}` if `json_object` is a
            bare `list`; `json_object` itself, completely unchanged,
            otherwise.
        """
        if isinstance(json_object, list):
            return {"work_experience": json_object}
        return json_object

    @staticmethod
    def _deduplicate_technologies(job: Any) -> Any:
        """Return `job` with its own `technologies` array's exact
        duplicates removed (first occurrence kept), or `job` itself,
        unchanged, if there was nothing to remove or the shape wasn't
        as expected. Never mutates `job`."""
        if not isinstance(job, dict):
            return job
        technologies = job.get("technologies")
        if not isinstance(technologies, list):
            return job

        deduplicated: List[Any] = []
        seen = set()
        for technology in technologies:
            if isinstance(technology, str):
                if technology in seen:
                    continue
                seen.add(technology)
            deduplicated.append(technology)

        if deduplicated == technologies:
            return job
        return {**job, "technologies": deduplicated}

    @staticmethod
    def _normalize_job_dates(job: Any) -> Any:
        """Milestone 23.18: return `job` with `start_date` and
        `end_date` each independently normalized by
        `_normalize_date_string`, or `job` itself, unchanged, if neither
        field needed a change or the shape wasn't as expected. Never
        mutates `job`."""
        if not isinstance(job, dict):
            return job
        original_start = job.get("start_date")
        original_end = job.get("end_date")
        new_start = WorkExperienceExtractor._normalize_date_string(original_start)
        new_end = WorkExperienceExtractor._normalize_date_string(original_end)

        if new_start == original_start and new_end == original_end:
            return job
        return {**job, "start_date": new_start, "end_date": new_end}

    @staticmethod
    def _normalize_date_string(value: Optional[str]) -> Optional[str]:
        """Milestone 23.18: rewrite `value` from `"<day> <abbreviated
        month> <year>"` (e.g. `"27 Jul 2015"`) to `"<day><ordinal
        suffix> <full month name> <year>"` (e.g. `"27th July 2015"`) if
        it matches that exact shape; otherwise return `value` completely
        unchanged.

        This is the one shape
        docs/MILESTONE_23_18_DATE_NORMALIZATION_RESULTS.md §3 confirmed
        is safe to normalize on this project's own ground truth and
        benchmark output, with no observed counter-example. In
        particular, this method deliberately does **not** touch:

        - `None` (missing dates are never invented — returned as-is).
        - A non-string value (defensive; schema validation's job to
          flag, not this method's).
        - A bare `"<month> <year>"` value with no day number (e.g.
          `"Jan 2023"`) — also frequently pure formatting in this
          project's data, but proven (§3) to break 11 currently-correct
          matches across 6 resumes if normalized blanket, so left
          alone.
        - A value that already has a full month name, an ordinal
          suffix, or both (e.g. `"27th July 2015"`, `"27 July 2015"`)
          — never observed as a real shape in this project's model
          output, so out of this evidenced scope; returned unchanged
          rather than guessed at.
        - Anything else that doesn't match the exact `"<day>
          <abbreviated month> <year>"` shape (extra words, a date
          range, a quarter, a bare year, numeric `MM/YYYY`, ISO
          `YYYY-MM`, etc.) — every one of these is either already
          unambiguous as-is or too ambiguous to safely rewrite; none is
          altered.

        The original day digits' own zero-padding is preserved exactly
        (`"07"` becomes `"07th"`, never `"7th"`) — this rewrites
        representation only, never the underlying calendar date.

        Args:
            value: A job's `start_date` or `end_date` value, or `None`.

        Returns:
            The normalized string, or `value` itself, completely
            unchanged, if no normalization applies.
        """
        if not isinstance(value, str):
            return value

        match = _DAY_ABBREVIATED_MONTH_YEAR_PATTERN.match(value)
        if match is None:
            return value

        day = match.group("day")
        month_abbreviation = match.group("month")
        year = match.group("year")

        full_month_name = _MONTH_ABBREVIATION_TO_FULL_NAME[month_abbreviation]
        day_with_ordinal_suffix = day + WorkExperienceExtractor._ordinal_suffix(day)

        return f"{day_with_ordinal_suffix} {full_month_name} {year}"

    @staticmethod
    def _ordinal_suffix(day_digits: str) -> str:
        """Return the standard English ordinal suffix ("st"/"nd"/"rd"/"th")
        for a 1-2 digit day-of-month string, per the usual 11th/12th/13th
        exception (e.g. "11" and "21" both end in "1", but "11" takes
        "th" while "21" takes "st")."""
        day_number = int(day_digits)
        if 10 <= day_number % 100 <= 20:
            return "th"
        return {1: "st", 2: "nd", 3: "rd"}.get(day_number % 10, "th")
