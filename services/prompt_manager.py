"""
services/prompt_manager.py

Prompt Manager (Milestone 9).

Loads prompt templates from the prompts/ directory and renders them by
substituting `{{variable}}` placeholders. Every pipeline in
PROJECT_PLAN.md will eventually need prompts (e.g. a vision-model
extraction prompt, a markdown-to-JSON prompt); this module is the one
shared way to load and fill them in, so that logic isn't duplicated per
pipeline.

This module knows nothing about resumes, pipelines, or parsing — it only
reads text files and does string substitution. What a prompt says and how
its output gets used is entirely up to the caller.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Optional

from config import PROMPTS_DIR

logger = logging.getLogger(__name__)

# Prompt templates are markdown files, one per prompt, addressed by an id
# like "resume/extract_markdown" -> prompts/resume/extract_markdown.md.
# The id's "/" segments map directly onto nested folders, so arbitrarily
# deep folder structures are supported for free.
PROMPT_FILE_EXTENSION = ".md"

# Matches {{variable_name}} placeholders, tolerating extra whitespace
# inside the braces (e.g. "{{ candidate_name }}").
_PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*(\w+)\s*\}\}")


class PromptManagerError(Exception):
    """Base exception for all Prompt Manager failures."""


class PromptNotFoundError(PromptManagerError):
    """Raised when a requested prompt id has no matching file on disk."""


class PromptRenderError(PromptManagerError):
    """Raised when a prompt template cannot be rendered, e.g. because a
    placeholder it references was not supplied a value."""


class PromptManager:
    """Loads and renders prompt templates from a prompts directory.

    Templates are plain markdown files containing `{{variable}}`
    placeholders. Raw template text is cached in memory the first time
    each prompt id is loaded, so repeated calls don't re-read the file
    from disk.
    """

    def __init__(self, prompts_dir: Path = PROMPTS_DIR) -> None:
        """Initialize the manager.

        Args:
            prompts_dir: Root directory that prompt ids are resolved
                against. Defaults to config.PROMPTS_DIR.
        """
        self.prompts_dir = Path(prompts_dir).resolve()
        self._cache: Dict[str, str] = {}

    def load(self, prompt_id: str) -> str:
        """Load a prompt's raw template text, using the in-memory cache
        after the first read.

        Args:
            prompt_id: Slash-separated prompt identifier, e.g.
                "resume/extract_markdown" for
                prompts/resume/extract_markdown.md.

        Returns:
            The template's raw text, unrendered (placeholders intact).

        Raises:
            PromptNotFoundError: If no file exists at the resolved path.
        """
        if prompt_id in self._cache:
            logger.debug("Prompt cache hit: %s", prompt_id)
            return self._cache[prompt_id]

        path = self._resolve_path(prompt_id)

        if not path.is_file():
            logger.error("Prompt not found: %s (resolved to %s)", prompt_id, path)
            raise PromptNotFoundError(f"No prompt found for id '{prompt_id}'")

        logger.info("Loading prompt %s from %s", prompt_id, path)
        text = path.read_text(encoding="utf-8")
        self._cache[prompt_id] = text
        return text

    def render(self, prompt_id: str, variables: Optional[Dict[str, str]] = None) -> str:
        """Load a prompt and substitute its `{{variable}}` placeholders.

        Args:
            prompt_id: Slash-separated prompt identifier, as in `load`.
            variables: Mapping of placeholder name to replacement value.
                Values are converted to `str` before substitution.

        Returns:
            The rendered prompt text, with every placeholder replaced.

        Raises:
            PromptNotFoundError: If no file exists for `prompt_id`.
            PromptRenderError: If the template references a placeholder
                that is missing from `variables`.
        """
        template = self.load(prompt_id)
        variables = variables or {}

        placeholders = set(_PLACEHOLDER_PATTERN.findall(template))
        missing = placeholders - variables.keys()
        if missing:
            logger.error(
                "Prompt '%s' is missing variable(s): %s", prompt_id, sorted(missing)
            )
            raise PromptRenderError(
                f"Prompt '{prompt_id}' is missing value(s) for: {sorted(missing)}"
            )

        rendered = _PLACEHOLDER_PATTERN.sub(
            lambda match: str(variables[match.group(1)]), template
        )
        logger.debug("Rendered prompt %s with variables %s", prompt_id, sorted(variables))
        return rendered

    def clear_cache(self) -> None:
        """Drop all cached prompt text, forcing the next `load`/`render`
        call for each prompt id to re-read it from disk."""
        logger.debug("Clearing prompt cache (%d entr(y/ies))", len(self._cache))
        self._cache.clear()

    def _resolve_path(self, prompt_id: str) -> Path:
        """Resolve a prompt id to an absolute file path under prompts_dir.

        Raises:
            PromptNotFoundError: If `prompt_id` would resolve outside of
                prompts_dir (e.g. via "..").
        """
        relative = Path(*prompt_id.split("/")).with_suffix(PROMPT_FILE_EXTENSION)
        path = (self.prompts_dir / relative).resolve()

        if self.prompts_dir not in path.parents and path != self.prompts_dir:
            logger.error("Prompt id resolves outside prompts_dir: %s", prompt_id)
            raise PromptNotFoundError(f"No prompt found for id '{prompt_id}'")

        return path
