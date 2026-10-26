"""
services/languages_extractor.py

Languages Extractor (Milestone 14.6; preprocess_json() override added in
Milestone 23.9.2).

Fifth of the modular, per-section resume parsers: extracts just the
candidate's spoken/written languages from a resume's markdown, using a
small text model:

    output/markdown/<resume>.md
                |
                v
    prompts/modules/languages.md + markdown, in one prompt
                |
                v
    OllamaClient.generate_text (settings.DEFAULT_TEXT_MODEL, default "qwen3:1.7b")
                |
                v
    parsed {"languages": [...]}
                |
                v
    preprocess_json()  -- drop known programming/technical terms
                |
                v
    validated against this module's own schema, returned in memory

All of that pipeline logic (reading markdown, prompting, parsing,
validating, timing) lives in services/base_module_extractor.py; this
class only supplies the languages-specific prompt id, schema path, and
model name via the three protected hooks the base class calls, plus (as
of Milestone 23.9.2) one override of the base class's preprocess_json()
hook. Its public extract() is exactly BaseModuleExtractor's, unchanged.

The schema (schemas/modules/languages_schema.json) enforces that
"languages" is an array of plain strings only: no null items, and no
duplicate values (uniqueItems).

This module extractor is deliberately narrow. It does not merge its
output with any other module's result (that's a later assembly step),
does not save anything to disk, does not benchmark itself, and — beyond
Milestone 23.9.2's denylist filter — does not normalize or enrich any
field.

Milestone 23.9.2 (docs/MILESTONE_23_9_PLAN.md, Fix 5;
docs/MILESTONE_23_9_2_RESULTS.md): the prompt (rule 3) already
explicitly forbids programming/scripting/markup language names from
appearing in "languages" — the model still occasionally violates this
already-explicit rule when a resume's Skills/Technical-Skills section
contains its own subsection literally headed "Languages" (e.g. a
"- Languages: Java, Python, JavaScript, HTML/CSS" line under
"## SKILLS"), confirmed live on 1 of 23 resumes in this project's own
benchmark dataset ("Mukul Kanojia_Full Stack Developer_Intileo",
`"JavaScript"` leaked; ground truth `[]`) and structurally present
(same trigger shape, no live leak observed this run) on 7 further
resumes. preprocess_json() below removes any already-extracted entry
matching a fixed, closed set of known programming/technical term names
— see PROGRAMMING_LANGUAGE_DENYLIST below and
docs/MILESTONE_23_9_2_RESULTS.md for the full evidence this exact list
was built from. It never touches any entry not on this list, so a
genuine spoken/written language is never at risk: every value in this
project's own 23-resume ground truth was checked directly against this
list and found to have zero overlap. This hook has no access to the
source markdown, so it cannot detect *which heading* a leaked value
came from — it filters by value identity alone, which is sufficient
here only because no term on this closed list can ever also be a real
spoken/written language name.
"""

from pathlib import Path
from typing import Any, List

from config import SCHEMAS_DIR
from settings import DEFAULT_TEXT_MODEL
from services.base_module_extractor import BaseModuleExtractor

# Text model used for extraction. Small/fast rather than the vision
# model, since this step only ever reads text. Milestone 23.3B: imported
# from settings (shared by every module extractor and JsonDraftGenerator)
# instead of hardcoded here, so a single settings.DEFAULT_TEXT_MODEL
# change (or the DEFAULT_TEXT_MODEL env var) switches the whole parser.

# Prompt instructing the model how to extract languages only.
DEFAULT_PROMPT_ID = "modules/languages"

# This module's own schema — scoped to {"languages": [...]}, not the
# full resume_schema_v1.json.
DEFAULT_SCHEMA_PATH = SCHEMAS_DIR / "modules" / "languages_schema.json"

# Milestone 23.9.2 (docs/MILESTONE_23_9_PLAN.md, Fix 5; see this module's
# docstring above and docs/MILESTONE_23_9_2_RESULTS.md for the full
# evidence trail). A closed, fixed set of programming/scripting/markup/
# technology term names that must never appear in "languages", matched
# by exact value after `.strip().casefold()` — never a substring match,
# so this can only ever remove an entry that IS one of these terms,
# never one that merely contains one.
#
# The first 15 terms are exactly prompts/modules/languages.md rule 3's
# own enumeration (already the parser's own stated definition of "not a
# human language") — reused, not re-invented, and re-verified this
# milestone to have zero overlap with any of this project's own 23
# ground-truth "languages" values. The remaining 3 (`c`, `pyspark`,
# `vb.net`) were added from this milestone's own fresh markdown scan of
# all 23 resumes: each appears verbatim inside a resume's own
# "Languages"-labeled technical-skills line/heading (Arjun Sharma's and
# PRAGNESH S. SHAH's, specifically) and was independently re-verified to
# have zero overlap with any ground-truth language value before being
# added. No term was added on suspicion alone — every entry here traces
# to either the prompt's own existing rule or a directly-observed
# trigger-context resume in this dataset.
PROGRAMMING_LANGUAGE_DENYLIST = {
    "python",
    "java",
    "sql",
    "c++",
    "c#",
    "javascript",
    "html",
    "css",
    "r",
    "go",
    "kotlin",
    "swift",
    "typescript",
    "scala",
    "matlab",
    "c",
    "pyspark",
    "vb.net",
}


class LanguagesExtractor(BaseModuleExtractor):
    """Extracts spoken/written languages from a resume's markdown.

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
        """Milestone 23.9.2 (Fix 5, docs/MILESTONE_23_9_PLAN.md): drop
        any entry in "languages" that exactly matches a known
        programming/technical term, per PROGRAMMING_LANGUAGE_DENYLIST
        above.

        A value is removed only if `value.strip().casefold()` exactly
        equals one of the denylist's normalized entries — not if it
        merely contains one as a substring (e.g. a hypothetical
        language-proficiency phrase that happened to include the
        letter "R" would not match, since the comparison is against
        the whole, trimmed string, not a substring search). Every other
        entry — including every entry when none of them are on the
        denylist — passes through completely unchanged, in its original
        order and original casing.

        Deliberately does nothing else: no section-heading awareness
        (this hook is never given the source markdown, only the
        already-parsed JSON, so it cannot know which markdown heading a
        value came from — see the module docstring), no fuzzy or
        substring matching, no normalization of a surviving value's
        casing or spacing, no qualifier-phrase handling (a separate,
        not-yet-implemented concern — see docs/MILESTONE_23_9_2_RESULTS.md
        §14).

        Defensive by design, mirroring SkillsExtractor.preprocess_json()
        and PersonalInformationExtractor.preprocess_json(): if
        `json_object` isn't a dict, or its "languages" value isn't a
        list, this returns the input completely unchanged. A non-string
        entry is never matched against the denylist (it can't be
        `.strip()`'d or `.casefold()`'d) and is always kept — schema
        validation, not this hook, is responsible for rejecting a
        non-string item.

        Args:
            json_object: The value BaseModuleExtractor._parse_json()
                returned — parsed, syntactically valid JSON, not yet
                schema-validated.

        Returns:
            `json_object` unchanged (same object, not a copy) if no
            entry matched the denylist (or the shape wasn't as
            expected); otherwise a new dict, a shallow copy of
            `json_object` with "languages" replaced by a filtered list.
            `json_object` itself is never mutated.
        """
        if not isinstance(json_object, dict):
            return json_object
        languages = json_object.get("languages")
        if not isinstance(languages, list):
            return json_object

        filtered: List[Any] = [
            value
            for value in languages
            if not (
                isinstance(value, str)
                and value.strip().casefold() in PROGRAMMING_LANGUAGE_DENYLIST
            )
        ]

        if filtered == languages:
            return json_object

        updated_json_object = dict(json_object)
        updated_json_object["languages"] = filtered
        return updated_json_object
