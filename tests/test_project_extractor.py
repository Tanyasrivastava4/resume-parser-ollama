"""
tests/test_project_extractor.py

Tests for services/project_extractor.py (Milestone T3/Phase 1).

ZERO real Ollama calls -- OllamaClient is mocked throughout; the real,
committed prompts/modules/projects.md and
schemas/modules/projects_schema.json are used, mirroring
tests/test_certification_extractor.py's own convention.
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services.base_module_extractor import BaseModuleResult, InvalidJSONError
from services.ollama_client import OllamaClient
from services.project_extractor import ProjectExtractor, ProjectResult
from validators.schema_validator import SchemaValidationError


def _entry(**overrides):
    base = {
        "name": "Senraa",
        "description": "A fashion-based web platform with AR try-on",
        "role": None,
        "start_date": None,
        "end_date": None,
        "highlights": ["Built AR try-on feature", "Implemented responsive UI"],
        "technologies": ["React.js", "Node.js"],
        "url": "https://senraa.com/",
    }
    base.update(overrides)
    return base


VALID_JSON_OBJECT = {"projects": [_entry()]}
EMPTY_JSON_OBJECT = {"projects": []}


@pytest.fixture
def mock_ollama_client():
    client = MagicMock(spec=OllamaClient)
    client.generate_text.return_value = SimpleNamespace(response=json.dumps(VALID_JSON_OBJECT))
    return client


@pytest.fixture
def extractor(mock_ollama_client):
    return ProjectExtractor(ollama_client=mock_ollama_client)


def _respond(mock_ollama_client, json_object):
    mock_ollama_client.generate_text.return_value = SimpleNamespace(response=json.dumps(json_object))


class TestExtract:
    def test_valid_markdown_returns_expected_result(self, tmp_path, extractor):
        markdown_path = tmp_path / "resume.md"
        # Milestone T6/Phase 1C: the markdown fixture must actually
        # contain the URL text the mocked response returns, matching
        # the real corpus's own convention (e.g. resume 016's
        # "### Senraa (https://senraa.com/)") -- otherwise
        # ProjectExtractor's own URL field-fidelity correction
        # (services/project_extractor.py's _ground_project_urls,
        # grounded in the real resume-016 defect) correctly nulls out
        # a URL this fixture never actually made source-present.
        markdown_path.write_text(
            "# Resume\n\n## Projects\n\n### Senraa (https://senraa.com/)", encoding="utf-8"
        )

        result = extractor.extract(markdown_path)

        assert isinstance(result, BaseModuleResult)
        assert isinstance(result, ProjectResult)
        assert result.json_object == VALID_JSON_OBJECT

    def test_real_prompt_is_loaded_and_sent(self, tmp_path, extractor, mock_ollama_client):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        extractor.extract(markdown_path)

        call_kwargs = mock_ollama_client.generate_text.call_args.kwargs
        assert call_kwargs["model"] == "qwen3:1.7b"
        prompt = call_kwargs["prompt"]
        assert "projects" in prompt.lower()
        # Confirms the real prompt's anti-misrouting instruction (the
        # exact T2-found resume-001 failure) is present.
        assert "Work Experience" in prompt
        assert "employer" in prompt.lower() or "employment" in prompt.lower()

    def test_empty_projects_is_valid(self, tmp_path, extractor, mock_ollama_client):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume\n\nNo projects listed.", encoding="utf-8")
        _respond(mock_ollama_client, EMPTY_JSON_OBJECT)

        result = extractor.extract(markdown_path)

        assert result.json_object == {"projects": []}

    def test_multiple_projects(self, tmp_path, extractor, mock_ollama_client):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        multi = {"projects": [_entry(name="Senraa"), _entry(name="Bewakoof.com Clone", url=None)]}
        _respond(mock_ollama_client, multi)

        result = extractor.extract(markdown_path)

        assert len(result.json_object["projects"]) == 2

    def test_missing_required_field_raises_schema_validation_error(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        invalid = {"projects": [{"name": "Senraa"}]}  # missing 7 required fields
        _respond(mock_ollama_client, invalid)

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_invalid_json_raises_invalid_json_error(self, tmp_path, extractor, mock_ollama_client):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(response="not json")

        with pytest.raises(InvalidJSONError):
            extractor.extract(markdown_path)
