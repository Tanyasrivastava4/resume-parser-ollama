"""
tests/test_contact_information_extractor.py

Tests for services/contact_information_extractor.py (Milestone 14.3).

PromptManager and SchemaValidator run for real (cheap, deterministic,
local operations — the former against a temporary prompt file, the
latter against the project's real
schemas/modules/contact_information_schema.json), so these tests
exercise the real prompt-loading and validation paths. Only OllamaClient
is mocked in the unit tests, since it depends on a running local model
server.

One integration test at the bottom talks to a real local Ollama server,
the real qwen3:1.7b text model, and the project's real
prompts/modules/contact_information.md prompt. It skips itself (rather
than failing) if Ollama or the model isn't available.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services.base_module_extractor import BaseModuleResult, InvalidJSONError, MarkdownNotFoundError
from services.contact_information_extractor import ContactInformationExtractor
from services.ollama_client import OllamaClient
from services.prompt_manager import PromptManager
from validators.schema_validator import SchemaValidationError

INTEGRATION_TEXT_MODEL = "qwen3:1.7b"

VALID_JSON_OBJECT = {
    "contact_information": {
        "email": ["ada@example.com"],
        "phone": [],
        "linkedin": "https://linkedin.com/in/ada",
        "github": None,
        "portfolio": None,
    }
}


@pytest.fixture
def prompts_dir(tmp_path):
    """A temporary prompts/ tree with one module prompt, so tests never
    touch the project's real prompts/ directory."""
    modules_dir = tmp_path / "prompts" / "modules"
    modules_dir.mkdir(parents=True)
    (modules_dir / "contact_information.md").write_text(
        "Extract only the candidate's contact information as JSON.",
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
    return ContactInformationExtractor(
        ollama_client=mock_ollama_client,
        prompt_manager=PromptManager(prompts_dir=prompts_dir),
    )


class TestExtract:
    def test_valid_markdown_returns_expected_result(self, tmp_path, extractor):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text(
            "# Ada Lovelace\n\nEmail: ada@example.com", encoding="utf-8"
        )

        result = extractor.extract(markdown_path)

        assert isinstance(result, BaseModuleResult)
        assert result.json_object == VALID_JSON_OBJECT
        assert result.processing_time >= 0

    def test_prompt_is_sent_with_markdown_appended(
        self, tmp_path, extractor, mock_ollama_client, prompts_dir
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Ada Lovelace\n\nContact info.", encoding="utf-8")

        extractor.extract(markdown_path)

        prompt_text = (prompts_dir / "modules" / "contact_information.md").read_text()
        call_kwargs = mock_ollama_client.generate_text.call_args.kwargs
        assert call_kwargs["model"] == "qwen3:1.7b"
        assert call_kwargs["prompt"] == f"{prompt_text}\n\n# Ada Lovelace\n\nContact info."

    def test_missing_markdown_raises_markdown_not_found(self, tmp_path, extractor):
        missing_path = tmp_path / "does_not_exist.md"

        with pytest.raises(MarkdownNotFoundError):
            extractor.extract(missing_path)

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

    def test_schema_failure_on_missing_required_field_raises(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        incomplete_object = {
            "contact_information": {
                "email": [],
                "phone": [],
                "linkedin": None,
                "github": None,
                # portfolio is missing
            }
        }
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps(incomplete_object)
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_missing_contact_information_object(
        self, tmp_path, extractor, mock_ollama_client
    ):
        """The root object itself must contain "contact_information" —
        a response missing that key entirely must fail validation."""
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({"email": ["ada@example.com"]})
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_wrong_data_type(
        self, tmp_path, extractor, mock_ollama_client
    ):
        """email must be an array; a string here must fail validation
        rather than being silently accepted."""
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        wrong_type_object = {
            "contact_information": {
                "email": "ada@example.com",  # should be an array
                "phone": [],
                "linkedin": None,
                "github": None,
                "portfolio": None,
            }
        }
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps(wrong_type_object)
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_extra_field_at_root(
        self, tmp_path, extractor, mock_ollama_client
    ):
        """The module schema only allows the "contact_information" key
        at the root — any other top-level module's data mixed in must
        fail validation, not be silently accepted."""
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        object_with_extra_section = {
            **VALID_JSON_OBJECT,
            "skills": ["Mathematics"],
        }
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps(object_with_extra_section)
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_extra_field_inside_contact_information(
        self, tmp_path, extractor, mock_ollama_client
    ):
        """Extra fields nested inside contact_information itself must
        also be rejected, not just extras at the root."""
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        object_with_extra_nested_field = {
            "contact_information": {
                **VALID_JSON_OBJECT["contact_information"],
                "fax": "555-1234",
            }
        }
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps(object_with_extra_nested_field)
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_result_is_not_saved_to_disk(self, tmp_path, extractor):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Ada Lovelace", encoding="utf-8")

        result = extractor.extract(markdown_path)

        assert not hasattr(result, "json_path")


@pytest.mark.integration
def test_real_ollama_extract_integration(tmp_path):
    """Optional integration test against a real local Ollama server, the
    real qwen3:1.7b model, and the project's real
    prompts/modules/contact_information.md prompt. Skips itself if
    either the server or the model isn't available."""
    ollama_client = OllamaClient()

    if not ollama_client.is_running():
        pytest.skip("Ollama is not running locally; skipping integration test.")
    if not ollama_client.model_exists(INTEGRATION_TEXT_MODEL):
        pytest.skip(f"Model '{INTEGRATION_TEXT_MODEL}' is not installed; skipping.")

    markdown_path = tmp_path / "resume.md"
    markdown_path.write_text(
        "# Ada Lovelace\n\nEmail: ada@example.com | Phone: +1-555-0100\n"
        "LinkedIn: https://linkedin.com/in/ada\n\n"
        "## Skills\n\n- Mathematics\n- Analytical writing\n",
        encoding="utf-8",
    )

    extractor = ContactInformationExtractor(ollama_client=ollama_client)

    result = extractor.extract(markdown_path, prompt_id="modules/contact_information")

    assert isinstance(result, BaseModuleResult)
    assert isinstance(result.json_object, dict)
    assert set(result.json_object.keys()) == {"contact_information"}
