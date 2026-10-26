"""
services/parser_merger.py

Parser Merger (Milestone 14.7 prerequisite).

Each module extractor (PersonalInformationExtractor,
ContactInformationExtractor, SkillsExtractor, ProfessionalSummaryExtractor,
LanguagesExtractor) returns a JSON object with exactly one top-level key,
e.g. {"personal_information": {...}}. This module combines several such
objects into a single resume dict, e.g. {"personal_information": {...},
"contact_information": {...}, "skills": [...], ...} — nothing more.

This does no normalization, no enrichment, and no schema validation of
its own (that's validators/schema_validator.py, run separately by the
pipeline after merging). It is a pure, side-effect-free combination step.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ParserMergerError(Exception):
    """Base exception for all Parser Merger failures."""


class MergeConflictError(ParserMergerError):
    """Raised when two module results define the same top-level key."""


class ParserMerger:
    """Merges module extractor results into one combined resume dict."""

    def merge(self, module_json_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge several module JSON objects into one combined dict.

        Args:
            module_json_objects: One JSON object per module extractor
                result (e.g. PersonalInformationExtractor's
                `json_object`), each expected to have exactly one
                top-level key.

        Returns:
            A single dict containing every module's top-level key,
            unioned together with no other transformation.

        Raises:
            MergeConflictError: If two module objects define the same
                top-level key — a real bug worth surfacing immediately
                rather than silently letting one overwrite the other.
        """
        merged: Dict[str, Any] = {}
        for module_json_object in module_json_objects:
            for key, value in module_json_object.items():
                if key in merged:
                    logger.error("Merge conflict: key '%s' defined by more than one module", key)
                    raise MergeConflictError(
                        f"Key '{key}' is defined by more than one module result."
                    )
                merged[key] = value

        logger.debug("Merged %d module result(s) into keys: %s", len(module_json_objects), list(merged))
        return merged
