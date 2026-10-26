"""
services/personal_information_extractor.py

Personal Information Extractor (Milestone 14.1; refactored onto the
shared BaseModuleExtractor pipeline in Milestone 14.2; preprocess_json()
override added in Milestone 23.9.1).

First of the modular, per-section resume parsers: extracts just the
candidate's personal information from a resume's markdown, using a small
text model:

    output/markdown/<resume>.md
                |
                v
    prompts/modules/personal_information.md + markdown, in one prompt
                |
                v
    OllamaClient.generate_text (settings.DEFAULT_TEXT_MODEL, default "qwen3:1.7b")
                |
                v
    parsed {"personal_information": {...}}
                |
                v
    preprocess_json()  -- derive first_name/last_name from full_name
                |
                v
    validated against this module's own schema, returned in memory

All of that pipeline logic (reading markdown, prompting, parsing,
validating, timing) now lives in services/base_module_extractor.py; this
class only supplies the personal-information-specific prompt id, schema
path, and model name via the three protected hooks the base class calls,
plus (as of Milestone 23.9.1) one override of the base class's
preprocess_json() hook. Its public API is otherwise unchanged from
before the refactor: same constructor, same extract() signature and
behavior, same exceptions and result shape (the names below are kept as
aliases for the shared base classes, so existing imports keep working).

This module extractor is deliberately narrow. It does not merge its
output with any other module's result (that's a later assembly step),
does not benchmark itself, and — beyond Milestone 23.9.1's name-splitting
— does not normalize or enrich any field.

Milestone 23.9.1 (docs/MILESTONE_23_9_PLAN.md, Fix 1;
docs/MILESTONE_23_9_1_RESULTS.md): the model occasionally populates
full_name but leaves first_name/last_name both null, confirmed real on
2 of 23 resumes in this project's own benchmark dataset ("CA Ishika
Sharma Resume", "Chirag Sharma_AVP_Intileo"). preprocess_json() below
derives first_name/last_name from full_name in exactly that one case,
using the same first-token/remainder-joined convention already present
in this project's own ground truth (e.g. "Nilesh Shivram Magare" ->
first_name="Nilesh", last_name="Shivram Magare"). It never overwrites an
already-populated first_name or last_name, never touches full_name
itself, and does nothing when full_name is null or not a string — a
deliberately narrow, deterministic split with no generalized name-
parsing heuristics (no title/prefix stripping, no inference from any
other field). See docs/MILESTONE_23_9_1_RESULTS.md for a documented
known limitation this narrow scope implies for names that include a
professional prefix/title (e.g. "CA Ishika Sharma").
"""

from pathlib import Path
from typing import Any

from config import SCHEMAS_DIR
from settings import DEFAULT_TEXT_MODEL
from services.base_module_extractor import (
    BaseModuleExtractor,
    BaseModuleExtractorError,
    BaseModuleResult,
    InvalidJSONError,
    MarkdownNotFoundError,
)

# Backward-compatible aliases. Before Milestone 14.2's refactor, this
# module defined its own exception and result classes; they're kept
# importable under their original names (now just the shared base
# classes from BaseModuleExtractor) so existing code doesn't have to
# change.
PersonalInformationExtractorError = BaseModuleExtractorError
PersonalInformationResult = BaseModuleResult

__all__ = [
    "PersonalInformationExtractor",
    "PersonalInformationResult",
    "PersonalInformationExtractorError",
    "MarkdownNotFoundError",
    "InvalidJSONError",
]

# Text model used for extraction. Small/fast rather than the vision
# model, since this step only ever reads text. Milestone 23.3B: imported
# from settings (shared by every module extractor and JsonDraftGenerator)
# instead of hardcoded here, so a single settings.DEFAULT_TEXT_MODEL
# change (or the DEFAULT_TEXT_MODEL env var) switches the whole parser.

# Prompt instructing the model how to extract personal information only.
DEFAULT_PROMPT_ID = "modules/personal_information"

# This module's own schema — scoped to {"personal_information": {...}},
# not the full resume_schema_v1.json.
DEFAULT_SCHEMA_PATH = SCHEMAS_DIR / "modules" / "personal_information_schema.json"


class PersonalInformationExtractor(BaseModuleExtractor):
    """Extracts personal information from a resume's markdown.

    All pipeline mechanics come from BaseModuleExtractor; this class only
    answers the three questions the base class needs: which prompt,
    which schema, and which model.
    """

    def _get_prompt_id(self) -> str:
        return DEFAULT_PROMPT_ID

    def _get_schema_path(self) -> Path:
        return DEFAULT_SCHEMA_PATH

    def _get_model_name(self) -> str:
        return DEFAULT_TEXT_MODEL

    def preprocess_json(self, json_object: Any) -> Any:
        """Milestone 23.9.1 (Fix 1, docs/MILESTONE_23_9_PLAN.md): derive
        first_name/last_name from full_name when the model left both
        null.

        Only fires when full_name is a non-empty string AND both
        first_name and last_name are already null — an already-
        populated first_name/last_name (from the model itself) is never
        touched, matched, or overwritten. full_name itself is never
        modified.

        Split rule: the first whitespace-delimited token becomes
        first_name; every remaining token, rejoined with a single
        space, becomes last_name. This is the exact convention already
        present in this project's own ground truth for multi-token
        names (e.g. "Nilesh Shivram Magare" -> first_name="Nilesh",
        last_name="Shivram Magare") — not an invented rule. For a
        single-token full_name (no whitespace at all — never observed
        in this project's ground truth, handled defensively, not
        verified against real data), first_name becomes that one token
        and last_name stays null, rather than guessing a last name that
        was never written.

        Deliberately does nothing else: no stripping of professional
        titles/prefixes (e.g. "CA", "Dr.", "Mr."), no casing changes, no
        inference from current_location or any other field. This means
        a full_name that includes a leading title (e.g. "CA Ishika
        Sharma") will split into first_name="CA", last_name="Ishika
        Sharma" — not matching a ground-truth convention that treats
        the title as separate from the name — a known, documented
        limitation of this narrow scope, not a bug; see
        docs/MILESTONE_23_9_1_RESULTS.md.

        Args:
            json_object: The value BaseModuleExtractor._parse_json()
                returned — parsed, syntactically valid JSON, not yet
                schema-validated.

        Returns:
            `json_object` unchanged (same object, not a copy) if the
            derivation condition isn't met; otherwise a new dict, a
            shallow copy of `json_object` with "personal_information"
            replaced by a shallow copy that has first_name/last_name
            filled in. `json_object` itself is never mutated.
        """
        if not isinstance(json_object, dict):
            return json_object
        personal_information = json_object.get("personal_information")
        if not isinstance(personal_information, dict):
            return json_object

        full_name = personal_information.get("full_name")
        first_name = personal_information.get("first_name")
        last_name = personal_information.get("last_name")

        if not isinstance(full_name, str) or not full_name.strip():
            return json_object
        if first_name is not None or last_name is not None:
            return json_object

        tokens = full_name.split()
        if not tokens:
            return json_object

        derived_first_name = tokens[0]
        derived_last_name = " ".join(tokens[1:]) if len(tokens) > 1 else None

        updated_personal_information = dict(personal_information)
        updated_personal_information["first_name"] = derived_first_name
        updated_personal_information["last_name"] = derived_last_name

        updated_json_object = dict(json_object)
        updated_json_object["personal_information"] = updated_personal_information
        return updated_json_object
