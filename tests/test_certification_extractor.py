"""
tests/test_certification_extractor.py

Tests for services/certification_extractor.py (Milestone T3/Phase 1).

ZERO real Ollama calls -- OllamaClient is mocked throughout; the real,
committed prompts/modules/certifications.md and
schemas/modules/certifications_schema.json are used (via the default
PromptManager/SchemaValidator), so these tests exercise the real
prompt-loading and schema-validation paths without any network/model
call. Mirrors tests/test_education_extractor.py's own established
convention.
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services.base_module_extractor import BaseModuleResult, InvalidJSONError
from services.certification_extractor import CertificationExtractor, CertificationResult
from services.ollama_client import OllamaClient
from validators.schema_validator import SchemaValidationError


def _entry(**overrides):
    base = {
        "name": "Microsoft Certified: Azure Solution Architect Expert (AZ-305)",
        "issuing_organization": "Microsoft",
        "issue_date": "March 2025",
        "expiry_date": "March 2026",
    }
    base.update(overrides)
    return base


VALID_JSON_OBJECT = {"certifications": [_entry()]}
EMPTY_JSON_OBJECT = {"certifications": []}


@pytest.fixture
def mock_ollama_client():
    client = MagicMock(spec=OllamaClient)
    client.generate_text.return_value = SimpleNamespace(response=json.dumps(VALID_JSON_OBJECT))
    return client


@pytest.fixture
def extractor(mock_ollama_client):
    return CertificationExtractor(ollama_client=mock_ollama_client)


def _respond(mock_ollama_client, json_object):
    mock_ollama_client.generate_text.return_value = SimpleNamespace(response=json.dumps(json_object))


class TestExtract:
    def test_valid_markdown_returns_expected_result(self, tmp_path, extractor):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume\n\n## Certifications\n\nAzure Solution Architect Expert", encoding="utf-8")

        result = extractor.extract(markdown_path)

        assert isinstance(result, BaseModuleResult)
        assert isinstance(result, CertificationResult)
        assert result.json_object == VALID_JSON_OBJECT
        assert result.processing_time >= 0

    def test_real_prompt_is_loaded_and_sent(self, tmp_path, extractor, mock_ollama_client):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        extractor.extract(markdown_path)

        call_kwargs = mock_ollama_client.generate_text.call_args.kwargs
        assert call_kwargs["model"] == "qwen3:1.7b"
        # Confirms the REAL committed prompt file loaded without error
        # and its own anti-misrouting instructions are present.
        assert "certifications" in call_kwargs["prompt"].lower()
        assert "Skills" in call_kwargs["prompt"]
        assert "Projects" in call_kwargs["prompt"]
        assert call_kwargs["prompt"].endswith("# Resume")

    def test_empty_certifications_is_valid(self, tmp_path, extractor, mock_ollama_client):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume\n\nNo certifications here.", encoding="utf-8")
        _respond(mock_ollama_client, EMPTY_JSON_OBJECT)

        result = extractor.extract(markdown_path)

        assert result.json_object == {"certifications": []}

    def test_multiple_certifications(self, tmp_path, extractor, mock_ollama_client):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        multi = {"certifications": [
            _entry(name="AWS Certified Solutions Architect"),
            _entry(name="PMP", issuing_organization="PMI", issue_date=None, expiry_date=None),
        ]}
        _respond(mock_ollama_client, multi)

        result = extractor.extract(markdown_path)

        assert len(result.json_object["certifications"]) == 2

    def test_missing_required_field_raises_schema_validation_error(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        invalid = {"certifications": [{"name": "PMP"}]}  # missing 3 required fields
        _respond(mock_ollama_client, invalid)

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_invalid_json_raises_invalid_json_error(self, tmp_path, extractor, mock_ollama_client):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(response="not json")

        with pytest.raises(InvalidJSONError):
            extractor.extract(markdown_path)
