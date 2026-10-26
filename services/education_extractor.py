"""
services/education_extractor.py

Education Extractor (Milestone 18.2; preprocess_json() override added
in Milestone 23.22).

Seventh of the modular, per-section resume parsers: extracts just the
candidate's education history from a resume's markdown, using a small
text model:

    output/markdown/<resume>.md
                |
                v
    prompts/modules/education.md + markdown, in one prompt
                |
                v
    OllamaClient.generate_text (settings.DEFAULT_TEXT_MODEL, default "qwen3:1.7b")
                |
                v
    parsed {"education": [...]}, validated against this module's own
    schema, returned in memory

All pipeline mechanics (reading markdown, prompting, parsing, validating,
timing) live in services/base_module_extractor.py and are not duplicated
here — this class only supplies the education-specific prompt id, schema
path, and model name via the three protected hooks the base class calls.
Its public extract() is exactly BaseModuleExtractor's, unchanged.

The schema (schemas/modules/education_schema.json) enforces that
"education" is an array of entry objects, each with all nine fields
(institution, degree, specialization, location, start_date, end_date,
currently_studying, grade, description) present and
additionalProperties: false at both the module and entry-item level.
See docs/EDUCATION_DESIGN.md (Milestone 18.1) for why each field exists
and how the real extraction challenges found in this project's own
dataset (multiple degrees, certifications mistaken for education,
missing dates, CGPA/percentage grades, tables, multi-column layouts)
are meant to be handled.

This module extractor is deliberately narrow, matching every other
module extractor in this project: it does not merge its output with any
other module's result, does not save anything to disk, does not
benchmark itself, and does not normalize, enrich, or validate against
the source markdown beyond its own schema. It is not wired into
ParserPipeline in this milestone (see docs/EDUCATION_EXTRACTOR_RESULTS.md).

Milestone 23.22 (docs/PARSER_RELIABILITY_HARDENING.md, following
docs/PARSER_STABILITY_ANALYSIS.md's finding that this extractor's
`currently_studying` field is this parser's single dominant reliability
weak point): `education_schema.json` declares `currently_studying` a
required, non-nullable `boolean`, but the model occasionally returns
JSON `null` for it instead — confirmed as a real, repeatable, 100%-
reproducible defect on "Rima_Karmakar_CV" (reproduced live: all 3
retry attempts returned the identical `null`, every time), which failed
that resume's entire pipeline run in 3 of 5 identical benchmark runs
(Milestone 23.21). Because this field is required, `null` is invalid
regardless of what the real answer is, and `BaseModuleExtractor.extract()`
validates *before* returning, so an unresolved `null` here doesn't
just under-score this module — it exhausts all 3 pipeline retries and
fails the *entire resume*, the same architectural failure mode
Milestone 23.16 already fixed for `WorkExperienceExtractor`'s
`technologies` field.

preprocess_json() below coerces a `null` `currently_studying` to
`False` before schema validation. This is a narrower kind of fix than
this project's other `preprocess_json()` overrides (which only ever
remove or reformat a value that's already present, never invent one) —
justified specifically because `currently_studying` is a *required*
field with only two valid values, so `null` is never a legitimate
answer to preserve, and because the evidence for which of the two
valid values to prefer is unusually strong: 67 of 68 `education`
entries in this project's own ground truth (98.5%) have
`currently_studying: False`, and the one real, live-reproduced case of
this exact defect (Rima_Karmakar_CV) has `False` as its own confirmed
ground truth. It does nothing else: no other field is touched, and a
`currently_studying` that is already a real boolean (`True` or `False`)
is never altered.
"""

from pathlib import Path
from typing import Any, List

from config import SCHEMAS_DIR
from settings import DEFAULT_TEXT_MODEL
from services.base_module_extractor import BaseModuleExtractor, BaseModuleResult

# Backward/forward-compatible alias, matching the pattern established by
# WorkExperienceExtractor: the shared BaseModuleResult, importable under
# this module's own name.
EducationResult = BaseModuleResult

__all__ = ["EducationExtractor", "EducationResult"]

# Text model used for extraction. Small/fast rather than the vision
# model, since this step only ever reads text. Milestone 23.3B: imported
# from settings (shared by every module extractor and JsonDraftGenerator)
# instead of hardcoded here, so a single settings.DEFAULT_TEXT_MODEL
# change (or the DEFAULT_TEXT_MODEL env var) switches the whole parser.

# Prompt instructing the model how to extract education only.
DEFAULT_PROMPT_ID = "modules/education"

# This module's own schema — scoped to {"education": [...]}, not the
# full resume_schema_v1.json.
DEFAULT_SCHEMA_PATH = SCHEMAS_DIR / "modules" / "education_schema.json"


class EducationExtractor(BaseModuleExtractor):
    """Extracts education history from a resume's markdown.

    All pipeline mechanics come from BaseModuleExtractor; this class
    answers the three questions the base class needs (which prompt,
    which schema, which model), plus (Milestone 23.22) one override of
    preprocess_json() that coerces a null `currently_studying` to
    `False` before schema validation runs.
    """

    def _get_prompt_id(self) -> str:
        return DEFAULT_PROMPT_ID

    def _get_schema_path(self) -> Path:
        return DEFAULT_SCHEMA_PATH

    def _get_model_name(self) -> str:
        return DEFAULT_TEXT_MODEL

    def preprocess_json(self, json_object: Any) -> Any:
        """Milestone 23.22: coerce a null `currently_studying` to
        `False` in each education entry before schema validation — see
        the module docstring and
        docs/PARSER_RELIABILITY_HARDENING.md for why this must happen
        here (this field is required and non-nullable; `null` always
        fails schema validation and, after 3 failed retries, the whole
        resume) and why `False` specifically is evidence-justified, not
        guessed.

        Defensive by design: if `json_object` isn't a dict, its
        "education" value isn't a list, or a given entry isn't a dict,
        that part is returned completely unchanged — diagnosing a
        malformed shape is SchemaValidator.validate()'s job, not this
        method's. An entry whose `currently_studying` is already a real
        boolean, or is simply absent from the entry, is never touched.

        Args:
            json_object: The value BaseModuleExtractor._parse_json()
                returned — parsed, syntactically valid JSON, not yet
                schema-validated.

        Returns:
            `json_object` unchanged if no entry's `currently_studying`
            was `null`; otherwise a new dict, a shallow copy of
            `json_object` with "education" replaced by a new list in
            which only the affected entry dict(s) are replaced.
            `json_object` itself (and every entry dict that didn't need
            a change) is never mutated.
        """
        if not isinstance(json_object, dict):
            return json_object
        entries = json_object.get("education")
        if not isinstance(entries, list):
            return json_object

        new_entries: List[Any] = [self._default_currently_studying(entry) for entry in entries]

        if new_entries == entries:
            return json_object
        return {**json_object, "education": new_entries}

    @staticmethod
    def _default_currently_studying(entry: Any) -> Any:
        """Return `entry` with a null `currently_studying` coerced to
        `False`, or `entry` itself, unchanged, if it wasn't null or the
        shape wasn't as expected. Never mutates `entry`."""
        if not isinstance(entry, dict):
            return entry
        if "currently_studying" not in entry or entry["currently_studying"] is not None:
            return entry
        return {**entry, "currently_studying": False}
