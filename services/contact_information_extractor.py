"""
services/contact_information_extractor.py

Contact Information Extractor (Milestone 14.3).

Second of the modular, per-section resume parsers: extracts just the
candidate's contact information from a resume's markdown, using a small
text model:

    output/markdown/<resume>.md
                |
                v
    prompts/modules/contact_information.md + markdown, in one prompt
                |
                v
    OllamaClient.generate_text (settings.DEFAULT_TEXT_MODEL, default "qwen3:1.7b")
                |
                v
    parsed {"contact_information": {...}}, validated against this
    module's own schema, returned in memory

All pipeline mechanics (reading markdown, prompting, parsing, validating,
timing) live in services/base_module_extractor.py and are not duplicated
here — this class only supplies the contact-information-specific prompt
id, schema path, and model name via the three protected hooks the base
class calls. Its public extract() is exactly BaseModuleExtractor's,
unchanged.

This module extractor is deliberately narrow. It does not merge its
output with any other module's result (that's a later assembly step),
does not save anything to disk, does not benchmark itself, and does not
normalize or enrich any field.
"""

from pathlib import Path

from config import SCHEMAS_DIR
from settings import DEFAULT_TEXT_MODEL
from services.base_module_extractor import BaseModuleExtractor

# Text model used for extraction. Small/fast rather than the vision
# model, since this step only ever reads text. Milestone 23.3B: imported
# from settings (shared by every module extractor and JsonDraftGenerator)
# instead of hardcoded here, so a single settings.DEFAULT_TEXT_MODEL
# change (or the DEFAULT_TEXT_MODEL env var) switches the whole parser.

# Prompt instructing the model how to extract contact information only.
DEFAULT_PROMPT_ID = "modules/contact_information"

# This module's own schema — scoped to {"contact_information": {...}},
# not the full resume_schema_v1.json.
DEFAULT_SCHEMA_PATH = SCHEMAS_DIR / "modules" / "contact_information_schema.json"


class ContactInformationExtractor(BaseModuleExtractor):
    """Extracts contact information from a resume's markdown.

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
