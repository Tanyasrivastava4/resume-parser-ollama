"""
services/base_module_extractor.py

Base Module Extractor (Milestone 14.2).

Generic pipeline shared by every per-section resume module extractor
(Personal Information today; Contact Information, Professional Summary,
Skills, Languages, etc. in later milestones):

    read markdown -> load prompt -> OllamaClient.generate_text() ->
    parse JSON -> preprocess_json() -> validate against a schema ->
    measure timing -> return

This base class knows nothing about any specific module. It has no idea
what "personal information" or "skills" even mean — every module-specific
detail (which prompt to load, which schema to validate against, which
model to use) comes from three protected hook methods that each concrete
subclass implements: _get_prompt_id(), _get_schema_path(), and
_get_model_name(). Subclasses only need to answer those three questions;
everything else (reading the file, calling Ollama, parsing, validating,
timing) is implemented exactly once, here.

This is a pure refactor: it does not change what any existing module
extractor does, only how much of its code is shared.

Milestone 22.6: a fourth, non-abstract hook, preprocess_json(), was
added between parsing and schema validation — see that method's own
docstring, and docs/SKILLS_VALIDATION_DESIGN.md §12, for why deterministic
cleanup that needs to happen before a module's own schema check (not
after, the way a post-extraction validator like
services/work_experience_validator.py runs) belongs here. Its default
implementation is the identity function, so every extractor that
doesn't override it — every one of them except
services/skills_extractor.py as of this milestone — is provably
unaffected; see tests/test_base_module_extractor.py's
TestPreprocessJsonDefault for the exact guarantee this makes.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from services.ollama_client import OllamaClient
from services.prompt_manager import PromptManager
from validators.schema_validator import SchemaValidator

logger = logging.getLogger(__name__)


class BaseModuleExtractorError(Exception):
    """Base exception for all module extractor failures."""


class MarkdownNotFoundError(BaseModuleExtractorError):
    """Raised when the given markdown path does not exist or is not a file."""


class InvalidJSONError(BaseModuleExtractorError):
    """Raised when the model's response is not syntactically valid JSON."""


@dataclass
class BaseModuleResult:
    """Result of running one module extractor over one resume.

    Attributes:
        json_object: The parsed JSON value returned by the model, exactly
            as returned apart from having passed schema validation — not
            enriched, not normalized, not merged with any other module's
            output.
        processing_time: Wall-clock time, in seconds, for the whole
            extraction (reading markdown, prompting, parsing, validating).
    """

    json_object: Any
    processing_time: float


class BaseModuleExtractor(ABC):
    """Generic read -> prompt -> model -> parse -> validate pipeline for
    a single resume-section module extractor.

    Composes PromptManager (prompt loading), OllamaClient (model
    inference), and SchemaValidator (structural validation) — all used
    exactly as they already exist. Concrete subclasses supply
    module-specific detail purely by implementing _get_prompt_id(),
    _get_schema_path(), and _get_model_name(); this class has no
    knowledge of what any particular module extracts, and cannot be
    instantiated directly.
    """

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        prompt_manager: Optional[PromptManager] = None,
        schema_validator: Optional[SchemaValidator] = None,
    ) -> None:
        """Initialize the extractor.

        Args:
            ollama_client: OllamaClient to use for inference. Defaults to
                a new OllamaClient() built from settings.py.
            prompt_manager: PromptManager to use for loading the module
                prompt. Defaults to a new PromptManager() rooted at
                config.PROMPTS_DIR.
            schema_validator: SchemaValidator used to check the parsed
                result before returning it. Defaults to a new
                SchemaValidator() validating against this subclass's own
                schema, from its _get_schema_path().
        """
        self.ollama_client = ollama_client or OllamaClient()
        self.prompt_manager = prompt_manager or PromptManager()
        self.schema_validator = schema_validator or SchemaValidator(
            schema_path=self._get_schema_path()
        )

    @abstractmethod
    def _get_prompt_id(self) -> str:
        """Return the PromptManager prompt id for this module, e.g.
        "modules/personal_information"."""
        raise NotImplementedError

    @abstractmethod
    def _get_schema_path(self) -> Path:
        """Return the path to this module's own JSON Schema file."""
        raise NotImplementedError

    @abstractmethod
    def _get_model_name(self) -> str:
        """Return the default Ollama text model name for this module."""
        raise NotImplementedError

    def preprocess_json(self, json_object: Any) -> Any:
        """Milestone 22.6: generic pre-schema-validation hook.

        Called on the freshly-parsed JSON, after `_parse_json()` and
        before `self.schema_validator.validate()` — i.e. before the
        one point in `extract()` that can raise `SchemaValidationError`
        and fail the whole step (after `MAX_ATTEMPTS` retries, the
        whole resume — see pipeline/parser_pipeline.py). A subclass
        needing to deterministically repair something the model
        produced that would otherwise trip its own schema (e.g. a
        `uniqueItems` violation) belongs here, not in a post-extraction
        validator: any validator positioned after `extract()` returns
        (the way services/work_experience_validator.py and
        services/education_validator.py are positioned in
        pipeline/parser_pipeline.py) never runs at all if
        `extract()` itself fails — see docs/SKILLS_VALIDATION_DESIGN.md
        §12 for the full architectural reasoning this hook implements.

        This base implementation is a no-op — it exists so every
        subclass gets a well-defined extension point without needing
        to override `extract()`'s whole body, but changes nothing by
        default. Only a subclass that explicitly overrides this method
        (Milestone 22.6: services/skills_extractor.py) behaves any
        differently from before this method existed.

        Args:
            json_object: The value `_parse_json()` returned — parsed,
                syntactically valid JSON, not yet schema-validated.

        Returns:
            The value to schema-validate and return. Default: the
            input, unchanged.
        """
        return json_object

    def extract(
        self,
        markdown_path: Union[str, Path],
        model_name: Optional[str] = None,
        prompt_id: Optional[str] = None,
    ) -> BaseModuleResult:
        """Extract this module's section from a resume's markdown.

        Args:
            markdown_path: Path to the source markdown file, e.g.
                output/markdown/resume.md.
            model_name: Name of an installed Ollama text model. Defaults
                to this subclass's _get_model_name().
            prompt_id: Prompt identifier to load via PromptManager.
                Defaults to this subclass's _get_prompt_id().

        Returns:
            A BaseModuleResult describing what was produced.

        Raises:
            MarkdownNotFoundError: If `markdown_path` does not exist or
                is not a file.
            PromptNotFoundError: If `prompt_id` has no matching prompt
                file (from services.prompt_manager).
            InvalidJSONError: If the model's response is not
                syntactically valid JSON.
            SchemaValidationError: If the parsed result does not conform
                to this module's schema (from
                validators.schema_validator).
            ModelNotFoundError: If `model_name` is not installed in
                Ollama (from services.ollama_client).
            OllamaNotRunningError: If the Ollama server cannot be reached
                (from services.ollama_client).
            OllamaConnectionError: If a different connection-level error
                occurs while calling Ollama (from services.ollama_client).
        """
        model_name = model_name or self._get_model_name()
        prompt_id = prompt_id or self._get_prompt_id()

        start_time = time.perf_counter()
        markdown_path = Path(markdown_path)

        if not markdown_path.exists() or not markdown_path.is_file():
            logger.error("Markdown file not found: %s", markdown_path)
            raise MarkdownNotFoundError(f"Markdown file not found: {markdown_path}")

        document_name = markdown_path.stem
        logger.info(
            "Extracting %s from %s with model=%s prompt=%s",
            type(self).__name__,
            markdown_path,
            model_name,
            prompt_id,
        )

        markdown_text = markdown_path.read_text(encoding="utf-8")
        prompt_text = self.prompt_manager.load(prompt_id)
        combined_prompt = f"{prompt_text}\n\n{markdown_text}"

        response = self.ollama_client.generate_text(model=model_name, prompt=combined_prompt)

        json_object = self._parse_json(response.response, document_name)
        json_object = self.preprocess_json(json_object)
        self.schema_validator.validate(json_object)
        processing_time = time.perf_counter() - start_time

        logger.info(
            "Extracted %s from %s in %.2fs",
            type(self).__name__,
            markdown_path,
            processing_time,
        )

        return BaseModuleResult(json_object=json_object, processing_time=processing_time)

    @staticmethod
    def _parse_json(response_text: str, document_name: str) -> Any:
        """Parse `response_text` as JSON.

        Raises:
            InvalidJSONError: If `response_text` isn't syntactically
                valid JSON. The response is not cleaned up or
                reinterpreted first (e.g. stripping stray markdown code
                fences) — module prompts already instruct the model to
                return raw JSON only, so a response that doesn't parse is
                treated as a real failure to surface, not something to
                silently patch around.

                The one deliberate exception (Milestone 23.4, per
                docs/KRISHANKANT_JSON_FAILURE_ANALYSIS.md) is
                `strict=False`: the model occasionally emits an
                unescaped literal control character (typically a raw
                newline or tab) inside a JSON string value instead of
                the escaped `\\n`/`\\t` sequence — a narrow, well-
                understood JSON-generation quirk, not the kind of
                structural error (missing commas, unterminated strings,
                stray prose) this method still surfaces loudly.
                `strict=False` permits only that one class of otherwise-
                valid-looking control character inside a string; it does
                not relax any other part of JSON's grammar.
        """
        try:
            return json.loads(response_text, strict=False)
        except json.JSONDecodeError as exc:
            logger.error("Model returned invalid JSON for %s: %s", document_name, exc)
            raise InvalidJSONError(
                f"Model response for '{document_name}' is not valid JSON: {exc}"
            ) from exc
