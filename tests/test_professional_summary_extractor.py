"""
tests/test_professional_summary_extractor.py

Tests for services/professional_summary_extractor.py (Milestone 14.5).

PromptManager and SchemaValidator run for real (cheap, deterministic,
local operations — the former against a temporary prompt file, the
latter against the project's real
schemas/modules/professional_summary_schema.json), so these tests
exercise the real prompt-loading and validation paths. Only OllamaClient
is mocked in the unit tests, since it depends on a running local model
server.

One integration test at the bottom talks to a real local Ollama server,
the real qwen3:1.7b text model, and the project's real
prompts/modules/professional_summary.md prompt. It skips itself (rather
than failing) if Ollama or the model isn't available.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services.base_module_extractor import (
    BaseModuleResult,
    InvalidJSONError,
    MarkdownNotFoundError,
)
from services.ollama_client import OllamaClient
from services.professional_summary_extractor import ProfessionalSummaryExtractor
from services.prompt_manager import PromptManager
from validators.schema_validator import SchemaValidationError

INTEGRATION_TEXT_MODEL = "qwen3:1.7b"

VALID_JSON_OBJECT = {
    "professional_summary": "Mathematician and writer with a focus on computing."
}


@pytest.fixture
def prompts_dir(tmp_path):
    """A temporary prompts/ tree with one module prompt, so tests never
    touch the project's real prompts/ directory."""
    modules_dir = tmp_path / "prompts" / "modules"
    modules_dir.mkdir(parents=True)
    (modules_dir / "professional_summary.md").write_text(
        "Extract only the candidate's professional summary as JSON.",
        encoding="utf-8",
    )
    return modules_dir.parent


@pytest.fixture
def mock_ollama_client():
    client = MagicMock(spec=OllamaClient)
    client.generate_text.return_value = SimpleNamespace(
        response=json.dumps(VALID_JSON_OBJECT)
    )
    return client


@pytest.fixture
def extractor(mock_ollama_client, prompts_dir):
    return ProfessionalSummaryExtractor(
        ollama_client=mock_ollama_client,
        prompt_manager=PromptManager(prompts_dir=prompts_dir),
    )


class TestExtract:
    def test_valid_markdown_returns_expected_result(self, tmp_path, extractor):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text(
            "# Ada Lovelace\n\n## Summary\n\n"
            "Mathematician and writer with a focus on computing.",
            encoding="utf-8",
        )

        result = extractor.extract(markdown_path)

        assert isinstance(result, BaseModuleResult)
        assert result.json_object == VALID_JSON_OBJECT
        assert result.processing_time >= 0

    def test_prompt_is_sent_with_markdown_appended(
        self, tmp_path, extractor, mock_ollama_client, prompts_dir
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Ada Lovelace\n\n## Summary", encoding="utf-8")

        extractor.extract(markdown_path)

        prompt_text = (prompts_dir / "modules" / "professional_summary.md").read_text()
        call_kwargs = mock_ollama_client.generate_text.call_args.kwargs
        assert call_kwargs["model"] == "qwen3:1.7b"
        assert call_kwargs["prompt"] == f"{prompt_text}\n\n# Ada Lovelace\n\n## Summary"

    def test_empty_summary_returns_null(self, tmp_path, extractor, mock_ollama_client):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text(
            "# Ada Lovelace\n\nNo summary section present.", encoding="utf-8"
        )
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({"professional_summary": None})
        )

        result = extractor.extract(markdown_path)

        assert result.json_object == {"professional_summary": None}

    def test_invalid_json_raises_invalid_json_error(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response="Sure, here is the JSON: {not valid json}"
        )

        with pytest.raises(InvalidJSONError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_missing_key(self, tmp_path, extractor, mock_ollama_client):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({})
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_array_value(self, tmp_path, extractor, mock_ollama_client):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({"professional_summary": ["Mathematician", "Writer"]})
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_object_value(self, tmp_path, extractor, mock_ollama_client):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({"professional_summary": {"text": "Mathematician"}})
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_extra_field(self, tmp_path, extractor, mock_ollama_client):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        object_with_extra_field = {
            **VALID_JSON_OBJECT,
            "skills": ["Mathematics"],
        }
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps(object_with_extra_field)
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_missing_markdown_raises_markdown_not_found(self, tmp_path, extractor):
        missing_path = tmp_path / "does_not_exist.md"

        with pytest.raises(MarkdownNotFoundError):
            extractor.extract(missing_path)

    def test_result_is_not_saved_to_disk(self, tmp_path, extractor):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text(
            "# Ada Lovelace\n\n## Summary\n\nMathematician.", encoding="utf-8"
        )

        result = extractor.extract(markdown_path)

        assert not hasattr(result, "json_path")


@pytest.mark.integration
def test_real_ollama_extract_integration(tmp_path):
    """Optional integration test against a real local Ollama server, the
    real qwen3:1.7b model, and the project's real
    prompts/modules/professional_summary.md prompt. Skips itself if
    either the server or the model isn't available."""
    ollama_client = OllamaClient()

    if not ollama_client.is_running():
        pytest.skip("Ollama is not running locally; skipping integration test.")
    if not ollama_client.model_exists(INTEGRATION_TEXT_MODEL):
        pytest.skip(f"Model '{INTEGRATION_TEXT_MODEL}' is not installed; skipping.")

    markdown_path = tmp_path / "resume.md"
    markdown_path.write_text(
        "# Ada Lovelace\n\n## Summary\n\nMathematician and writer with a focus "
        "on computing, known for work on the Analytical Engine.\n\n"
        "## Skills\n\n- Mathematics\n- Analytical writing\n",
        encoding="utf-8",
    )

    extractor = ProfessionalSummaryExtractor(ollama_client=ollama_client)

    result = extractor.extract(markdown_path, prompt_id="modules/professional_summary")

    assert isinstance(result, BaseModuleResult)
    assert isinstance(result.json_object, dict)
    assert set(result.json_object.keys()) == {"professional_summary"}
    assert isinstance(result.json_object["professional_summary"], (str, type(None)))
