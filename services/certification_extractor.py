"""
services/certification_extractor.py

Certification Extractor (Milestone T3/Phase 1 -- schema foundation).

SCAFFOLDING ONLY: this class exists as a complete, ready module
extractor mirroring the exact same architecture as every other module
extractor (WorkExperienceExtractor, EducationExtractor,
ProfessionalSummaryExtractor), but has NOT been added to
pipeline/parser_pipeline.py's `_module_extractors` list and has NOT
been validated against any real Ollama call in this milestone
(Milestone T3/Phase 1's own target LLM call count was zero). Wiring
this into the live pipeline is a deliberate, disclosed, DEFERRED next
step -- see docs/MILESTONE_T3_PHASE1_SCHEMA_FOUNDATION.md Part E for
the explicit reasoning (activating an unvalidated new extractor in
production without any real-call evidence would itself be a broader,
unauthorized change, not a schema-foundation change).

    output/markdown/<resume>.md
                |
                v
    prompts/modules/certifications.md + markdown, in one prompt
                |
                v
    OllamaClient.generate_text (settings.DEFAULT_TEXT_MODEL, default "qwen3:1.7b")
                |
                v
    parsed {"certifications": [...]}, validated against
    schemas/modules/certifications_schema.json, returned in memory

All pipeline mechanics (reading markdown, prompting, parsing, validating,
timing) live in services/base_module_extractor.py and are not duplicated
here. This module extractor is deliberately narrow -- like
ProfessionalSummaryExtractor, it performs no preprocess_json()
normalization of its own (no evidenced need was found in this
milestone's Part B field-design work; unlike WorkExperienceExtractor,
no real checkpoint data exists yet to motivate one).
"""

from pathlib import Path

from config import SCHEMAS_DIR
from settings import DEFAULT_TEXT_MODEL
from services.base_module_extractor import BaseModuleExtractor, BaseModuleResult

CertificationResult = BaseModuleResult

__all__ = ["CertificationExtractor", "CertificationResult"]

DEFAULT_PROMPT_ID = "modules/certifications"
DEFAULT_SCHEMA_PATH = SCHEMAS_DIR / "modules" / "certifications_schema.json"


class CertificationExtractor(BaseModuleExtractor):
    """Extracts certifications from a resume's markdown.

    All pipeline mechanics come from BaseModuleExtractor; this class
    answers the three questions the base class needs (which prompt,
    which schema, which model). No preprocess_json() override.
    """

    def _get_prompt_id(self) -> str:
        return DEFAULT_PROMPT_ID

    def _get_schema_path(self) -> Path:
        return DEFAULT_SCHEMA_PATH

    def _get_model_name(self) -> str:
        return DEFAULT_TEXT_MODEL
