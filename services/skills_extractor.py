"""
services/skills_extractor.py

Skills Extractor (Milestone 14.4; preprocess_json() override added in
Milestone 22.6).

Third of the modular, per-section resume parsers: extracts just the
candidate's skills from a resume's markdown, using a small text model:

    output/markdown/<resume>.md
                |
                v
    prompts/modules/skills.md + markdown, in one prompt
                |
                v
    OllamaClient.generate_text (settings.DEFAULT_TEXT_MODEL, default "qwen3:1.7b")
                |
                v
    parsed {"skills": [...]}
                |
                v
    preprocess_json()  -- SKV-001 + SKV-002 duplicate removal
                |
                v
    validated against this module's own schema, returned in memory

All pipeline mechanics (reading markdown, prompting, parsing, validating,
timing) live in services/base_module_extractor.py and are not duplicated
here — this class supplies the skills-specific prompt id, schema path,
and model name via the three protected hooks the base class calls, plus
(as of Milestone 22.6) one override of the base class's preprocess_json()
hook. Its public extract() is exactly BaseModuleExtractor's, unchanged.

The schema (schemas/modules/skills_schema.json) enforces that "skills" is
an array of plain strings only: no nested objects, no null items, and no
duplicate values (uniqueItems).

This module extractor is deliberately narrow. It does not merge its
output with any other module's result (that's a later assembly step),
does not save anything to disk, does not benchmark itself, and — beyond
Milestone 22.6's duplicate removal — does not normalize or enrich any
field.

Milestone 22.6 (docs/SKILLS_VALIDATION_DESIGN.md §1, §12, §13, rules
SKV-001/SKV-002): the model occasionally, genuinely-non-deterministically
returns a `skills` array with a repeated entry — confirmed real, not
hypothetical, on the "Kishan Yadav" resume in this project's own
benchmark dataset, where the source markdown itself lists the same tool
(e.g. "Azure ML Pipelines") under two different category headings and
the model sometimes carries both mentions through unmerged. Because
`skills_schema.json` declares `uniqueItems: true`, and because
BaseModuleExtractor.extract() validates against that schema *before*
returning, an unresolved duplicate does not just under-score this
module — it raises SchemaValidationError, is retried
MAX_ATTEMPTS times by pipeline/parser_pipeline.py, and (if still
duplicated) fails the *entire resume*, not just skills. No
post-extraction validator (the shape services/work_experience_validator.py
and services/education_validator.py use) can ever fix this, because
extract() never successfully returns in the first place — see
docs/SKILLS_VALIDATION_DESIGN.md §12 for the full reasoning.
preprocess_json() below removes exact and case-insensitive duplicates
(ground-truth-confirmed always safe: no ground-truth skills array in
this project's 23-resume benchmark dataset has ever contained a
duplicate, exact or case-insensitive) before schema validation runs, so
this specific failure mode is closed at the source. It does nothing
else beyond what Milestone 23.32 added (below): no splitting of
comma/parenthetical lists, no casing/spelling normalization of a
*surviving* value, no cross-section provenance filtering — every one of
those was investigated and explicitly deferred or rejected in
docs/SKILLS_VALIDATION_DESIGN.md, and none of them belongs in this
narrowly-scoped hook.

Milestone 23.32 (docs/MILESTONE_23_31_SKILLS_RESIDUAL_ERROR_ANALYSIS.md
§14/§18, "P0-A"/"P0-B"; docs/MILESTONE_23_32_SKILLS_P0_FIXES_RESULTS.md):
two additional, narrow, deterministic corrections added to
preprocess_json(), run before the SKV-001/002 dedup pass above so that
any duplicate either one happens to create is still caught by it:

  1. P0-A (`_strip_leading_conjunction_artifact`): removes a literal
     leading "and " or "& " from an individual skill string — a
     confirmed, reproducible artifact of the model splitting an
     Oxford-comma list (e.g. "Program strategy, enterprise roadmaps,
     and success metrics") but leaving the trailing member's own
     conjunction word attached (observed live: Chirag Sharma's
     "and success metrics" / "and dependency management" / "and
     solution design"). Exhaustively confirmed against all 23
     ground-truth files (0 legitimate skills begin with "and "/"& ")
     and all 23 current extractor outputs (0 ambiguous or legitimate
     collisions; every other "and"/"&" occurrence in the dataset is
     mid-string, never matched by this anchored rule) before being
     implemented — see the Milestone 23.32 results doc §3/§13.
  2. P0-B (`_remove_spoken_language_entries`, SKV-003 as designed in
     docs/SKILLS_VALIDATION_DESIGN.md §5/§13 but not implemented until
     now): removes an entry that is *exactly* a spoken/human language
     name, optionally followed by a parenthetical (e.g. "English
     (Professional Working Proficiency)") — content that belongs to
     the `languages` module, not `skills`. A bounded, whole-string
     match against a small, curated, disjoint-from-technology list;
     never a substring search. Confirmed live: CA Ishika Sharma's
     `skills` currently includes "English"/"Hindi", sourced from a
     `### Languages` subsection nested under `## Technical & Soft
     Skills` — content the resume's own `languages` module already
     captures correctly and independently, so removing it here is pure
     duplicate-content cleanup, not a data loss. Exhaustively scanned
     against all 417 current skill values and all 23 ground-truth
     files before being implemented — 0 collisions found with any
     programming language, framework, or technology name (see the
     Milestone 23.32 results doc §4/§13).

Both operate on the `skills` array's own string contents only —
`preprocess_json()` never receives the source markdown, section
metadata, or locator spans (confirmed by direct read of
BaseModuleExtractor.extract()), so neither rule can be, or attempts to
be, provenance-aware; each is a narrow, content-only, whole-string-or-
anchored-prefix match, evidenced safe against the complete current
dataset rather than a broader heuristic.
"""

import re
from pathlib import Path
from typing import Any, List

from config import SCHEMAS_DIR
from services.base_module_extractor import BaseModuleExtractor
from settings import DEFAULT_TEXT_MODEL

# Text model used for extraction. Small/fast rather than the vision
# model, since this step only ever reads text. Milestone 23.3B: imported
# from settings (shared by every module extractor and JsonDraftGenerator)
# instead of hardcoded here, so a single settings.DEFAULT_TEXT_MODEL
# change (or the DEFAULT_TEXT_MODEL env var) switches the whole parser.

# Prompt instructing the model how to extract skills only.
DEFAULT_PROMPT_ID = "modules/skills"

# This module's own schema — scoped to {"skills": [...]}, not the full
# resume_schema_v1.json.
DEFAULT_SCHEMA_PATH = SCHEMAS_DIR / "modules" / "skills_schema.json"

# Milestone 23.32, P0-A: a leading "and "/"& " conjunction left over from
# the model splitting an Oxford-comma list but not detaching the final
# member's own conjunction word. Anchored to the very start of the
# string (`^`) and requires trailing whitespace, so it can never match
# "and"/"&" appearing mid-string (e.g. "Program and project management",
# "Wireless device configuration & troubleshooting") — both confirmed,
# real, currently-correct dataset values that must never be touched. See
# the module docstring and docs/MILESTONE_23_32_SKILLS_P0_FIXES_RESULTS.md
# §3 for the exhaustive dataset scan this pattern was checked against.
_LEADING_CONJUNCTION_PATTERN = re.compile(r"^(?:and|&)\s+", re.IGNORECASE)

# Milestone 23.32, P0-B (SKV-003, designed in
# docs/SKILLS_VALIDATION_DESIGN.md §5/§13): a small, curated, finite set
# of spoken/human language names — deliberately disjoint from any
# programming language, framework, or technology name in this project's
# dataset (Python, Java, C, C++, R, Go, Rust, ... are never on it by
# construction). Matched only as a bounded whole string (optionally
# followed by a parenthetical proficiency descriptor, e.g. "English
# (Professional Working Proficiency)") — never as a substring search
# inside a longer, legitimately-technical phrase. See the module
# docstring and docs/MILESTONE_23_32_SKILLS_P0_FIXES_RESULTS.md §4 for
# the exhaustive scan (all 417 current skill values, all 23
# ground-truth files) this list and pattern were checked against.
_SPOKEN_LANGUAGES = (
    "English", "Hindi", "Marathi", "French", "German", "Spanish", "Tamil",
    "Telugu", "Kannada", "Bengali", "Punjabi", "Gujarati", "Urdu", "Arabic",
    "Mandarin", "Chinese", "Japanese", "Korean", "Portuguese", "Italian",
    "Russian", "Malayalam",
)
_SPOKEN_LANGUAGE_PATTERN = re.compile(
    r"^(?:" + "|".join(re.escape(language) for language in _SPOKEN_LANGUAGES) + r")"
    r"(?:\s*\([^)]*\))?$",
    re.IGNORECASE,
)


class SkillsExtractor(BaseModuleExtractor):
    """Extracts skills from a resume's markdown.

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

    @staticmethod
    def _strip_leading_conjunction_artifacts(skills: List[Any]) -> List[Any]:
        """P0-A (Milestone 23.32, docs/MILESTONE_23_31_SKILLS_RESIDUAL_ERROR_ANALYSIS.md
        §14/§18, docs/MILESTONE_23_32_SKILLS_P0_FIXES_RESULTS.md §3):
        remove a literal leading "and " or "& " from an individual
        skill string, preserving the remainder exactly.

        Real, confirmed cause: the model splits an Oxford-comma list
        ("Program strategy, enterprise roadmaps, and success metrics")
        into separate array entries but leaves the final member's own
        conjunction word attached instead of detaching it
        ("and success metrics" instead of "success metrics") — observed
        live on Chirag Sharma's current output, 3 instances, all 3
        confirmed against ground truth to be exact matches once
        stripped.

        Anchored to the very start of the string only (`^`), so
        "and"/"&" appearing anywhere else in a skill — mid-string
        ("Program and project management"), as part of a real compound
        skill ("Wireless device configuration & troubleshooting") — is
        never touched; only a genuine leading conjunction, followed by
        whitespace, is stripped. Never strips if doing so would leave
        an empty string (defensive; not observed in this dataset).
        Non-string entries are returned unchanged, exactly as the
        dedup pass below already does.

        Args:
            skills: The raw "skills" list, not yet deduplicated.

        Returns:
            A new list, same length and order, with each matching
            entry's leading conjunction removed. Never mutates `skills`.
        """
        result: List[Any] = []
        for skill in skills:
            if isinstance(skill, str):
                stripped = _LEADING_CONJUNCTION_PATTERN.sub("", skill, count=1)
                if stripped and stripped != skill:
                    result.append(stripped)
                    continue
            result.append(skill)
        return result

    @staticmethod
    def _remove_spoken_language_entries(skills: List[Any]) -> List[Any]:
        """P0-B / SKV-003 (Milestone 22.5 design,
        docs/SKILLS_VALIDATION_DESIGN.md §5/§13; implemented Milestone
        23.32, docs/MILESTONE_23_32_SKILLS_P0_FIXES_RESULTS.md §4):
        remove any entry that is exactly a spoken/human language name
        (optionally followed by a parenthetical proficiency
        descriptor), per `_SPOKEN_LANGUAGE_PATTERN`.

        Real, confirmed cause: a nested `### Languages` subsection
        under a `## ... Skills` heading sometimes gets pulled into the
        skills array despite the prompt's own rule 6 already
        forbidding it — observed live on CA Ishika Sharma's current
        output ("English", "Hindi"), content her own `languages` module
        output already captures correctly and independently, so this
        removal is duplicate-content cleanup, not data loss.

        Bounded whole-string match only (never a substring search
        inside a longer phrase) against a small, curated, finite list
        deliberately disjoint from any programming language, framework,
        or technology name — "Java", "C", "R", "English NLP", "Hindi
        OCR" and similar are structurally unreachable by this pattern.
        Exhaustively checked against all 417 current skill values and
        all 23 ground-truth files before being written (0 collisions
        found either way) — see the Milestone 23.32 results doc §4.

        Args:
            skills: The "skills" list, after `_strip_leading_conjunction_artifacts`.

        Returns:
            A new list with every matching entry removed, order
            otherwise preserved. Never mutates `skills`.
        """
        return [
            skill
            for skill in skills
            if not (isinstance(skill, str) and _SPOKEN_LANGUAGE_PATTERN.match(skill.strip()))
        ]

    def preprocess_json(self, json_object: Any) -> Any:
        """P0-A + P0-B (Milestone 23.32) + SKV-001 + SKV-002 (Milestone
        22.6): three narrow, independent, deterministic corrections to
        `skills` before schema validation — see the module docstring
        for why this must happen here, not in a post-extraction
        validator.

        Applied in this order: (1) `_strip_leading_conjunction_artifacts`
        (P0-A), (2) `_remove_spoken_language_entries` (P0-B), (3) the
        SKV-001/002 duplicate-removal pass below. P0-A/P0-B run first
        so that any duplicate either one happens to create (e.g. an
        "and Python" artifact stripped down to a "Python" that was
        already present elsewhere in the same array) is still caught
        by the existing dedup pass — not observed in the current
        dataset, but a deliberate ordering choice, not an accident.

        SKV-001 + SKV-002: a single normalized-key pass implements both
        rules at once: two entries with the exact same characters are
        trivially also case-insensitive duplicates of each other, so
        there is no separate "exact-only" pass to write — SKV-002's
        check is a strict superset of SKV-001's. The key used to detect
        a duplicate is `skill.strip().casefold()`; the *surviving* copy
        (always the first occurrence encountered) is returned exactly
        as the model wrote it, never re-cased or otherwise rewritten —
        only which duplicate to drop is decided by the normalized key,
        per docs/SKILLS_VALIDATION_DESIGN.md §8's safe-vs-semantic
        normalization line. Original order is preserved.

        Deliberately does nothing else beyond these three rules — no
        general splitting/joining, no casing/spelling normalization of
        a surviving value, no cross-section provenance filtering (see
        the module docstring).

        Defensive by design: if `json_object` isn't a dict, or its
        "skills" value isn't a list, this returns the input completely
        unchanged. Diagnosing a malformed shape is
        SchemaValidator.validate()'s job, not this method's — it only
        ever touches a recognizably list-shaped skills array. A
        non-string entry is never touched by any of the three rules
        (schema validation, not this hook, is responsible for
        rejecting a non-string item) — it is always kept, so this
        method can never silently drop a value schema validation would
        otherwise have reported as an error.

        Args:
            json_object: The value BaseModuleExtractor._parse_json()
                returned — parsed, syntactically valid JSON, not yet
                schema-validated.

        Returns:
            `json_object` unchanged if none of the three rules changed
            anything (or the shape wasn't as expected); otherwise a new
            dict, a shallow copy of `json_object` with "skills"
            replaced by the corrected list. `json_object` itself is
            never mutated.
        """
        if not isinstance(json_object, dict):
            return json_object
        skills = json_object.get("skills")
        if not isinstance(skills, list):
            return json_object

        conjunction_stripped = self._strip_leading_conjunction_artifacts(skills)
        language_removed = self._remove_spoken_language_entries(conjunction_stripped)

        deduplicated: List[Any] = []
        seen_keys = set()
        for skill in language_removed:
            if isinstance(skill, str):
                key = skill.strip().casefold()
                if key in seen_keys:
                    continue
                seen_keys.add(key)
            deduplicated.append(skill)

        if deduplicated == skills:
            return json_object
        return {**json_object, "skills": deduplicated}
