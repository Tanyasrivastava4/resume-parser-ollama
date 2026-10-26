"""
tests/test_personal_information_extractor.py

Tests for services/personal_information_extractor.py (Milestone 14.1).

PromptManager and SchemaValidator run for real (cheap, deterministic,
local operations — the former against a temporary prompt file, the
latter against the project's real
schemas/modules/personal_information_schema.json), so these tests
exercise the real prompt-loading and validation paths. Only OllamaClient
is mocked in the unit tests, since it depends on a running local model
server.

One integration test at the bottom talks to a real local Ollama server,
the real qwen3:1.7b text model, and the project's real
prompts/modules/personal_information.md prompt. It skips itself (rather
than failing) if Ollama or the model isn't available.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services.ollama_client import OllamaClient
from services.personal_information_extractor import (
    InvalidJSONError,
    MarkdownNotFoundError,
    PersonalInformationExtractor,
    PersonalInformationResult,
)
from services.prompt_manager import PromptManager
from validators.schema_validator import SchemaValidationError

INTEGRATION_TEXT_MODEL = "qwen3:1.7b"

VALID_JSON_OBJECT = {
    "personal_information": {
        "full_name": "Ada Lovelace",
        "first_name": "Ada",
        "last_name": "Lovelace",
        "current_location": None,
    }
}


@pytest.fixture
def prompts_dir(tmp_path):
    """A temporary prompts/ tree with one module prompt, so tests never
    touch the project's real prompts/ directory."""
    modules_dir = tmp_path / "prompts" / "modules"
    modules_dir.mkdir(parents=True)
    (modules_dir / "personal_information.md").write_text(
        "Extract only the candidate's personal information as JSON.",
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
    return PersonalInformationExtractor(
        ollama_client=mock_ollama_client,
        prompt_manager=PromptManager(prompts_dir=prompts_dir),
    )


class TestExtract:
    def test_valid_markdown_returns_expected_result(self, tmp_path, extractor):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Ada Lovelace\n\nMathematician.", encoding="utf-8")

        result = extractor.extract(markdown_path)

        assert isinstance(result, PersonalInformationResult)
        assert result.json_object == VALID_JSON_OBJECT
        assert result.processing_time >= 0

    def test_prompt_is_sent_with_markdown_appended(
        self, tmp_path, extractor, mock_ollama_client, prompts_dir
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Ada Lovelace\n\nMathematician.", encoding="utf-8")

        extractor.extract(markdown_path)

        prompt_text = (prompts_dir / "modules" / "personal_information.md").read_text()
        call_kwargs = mock_ollama_client.generate_text.call_args.kwargs
        assert call_kwargs["model"] == "qwen3:1.7b"
        assert call_kwargs["prompt"] == (
            f"{prompt_text}\n\n# Ada Lovelace\n\nMathematician."
        )

    def test_empty_markdown_still_produces_a_result(self, tmp_path, extractor):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("", encoding="utf-8")

        result = extractor.extract(markdown_path)

        assert result.json_object == VALID_JSON_OBJECT

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

    def test_schema_failure_raises_schema_validation_error(
        self, tmp_path, extractor, mock_ollama_client
    ):
        """Syntactically valid JSON that doesn't conform to this
        module's schema (missing a required field here) must raise
        SchemaValidationError."""
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        incomplete_object = {
            "personal_information": {
                "full_name": "Ada Lovelace",
                "first_name": "Ada",
                # last_name and current_location are missing
            }
        }
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps(incomplete_object)
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_extra_field_at_root(
        self, tmp_path, extractor, mock_ollama_client
    ):
        """The module schema only allows the "personal_information" key
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

    def test_missing_markdown_raises_markdown_not_found(self, tmp_path, extractor):
        missing_path = tmp_path / "does_not_exist.md"

        with pytest.raises(MarkdownNotFoundError):
            extractor.extract(missing_path)

    def test_result_is_not_saved_to_disk(self, tmp_path, extractor):
        """This module extractor only returns an in-memory result — it
        has no output directory of its own (merging/saving is a later
        milestone's job)."""
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Ada Lovelace", encoding="utf-8")

        result = extractor.extract(markdown_path)

        assert not hasattr(result, "json_path")


class TestPreprocessJson:
    """Milestone 23.9.1 (Fix 1, docs/MILESTONE_23_9_PLAN.md).
    preprocess_json() exercised directly, in isolation from
    extract()/Ollama/schema validation — see TestExtract for the same
    behavior exercised through the full extract() pipeline."""

    def test_full_name_populated_first_and_last_null_are_derived(self, extractor):
        """Reproduces the real dataset case (Chirag Sharma_AVP_Intileo):
        full_name populated, first_name/last_name both null."""
        result = extractor.preprocess_json(
            {
                "personal_information": {
                    "full_name": "Chirag Sharma",
                    "first_name": None,
                    "last_name": None,
                    "current_location": None,
                }
            }
        )

        assert result == {
            "personal_information": {
                "full_name": "Chirag Sharma",
                "first_name": "Chirag",
                "last_name": "Sharma",
                "current_location": None,
            }
        }

    def test_multi_token_full_name_joins_remainder_into_last_name(self, extractor):
        """3+ tokens: first token alone becomes first_name, every
        remaining token rejoined becomes last_name — the exact
        convention already present in this project's own ground truth
        (e.g. "Nilesh Shivram Magare" -> first_name="Nilesh",
        last_name="Shivram Magare")."""
        result = extractor.preprocess_json(
            {
                "personal_information": {
                    "full_name": "Nilesh Shivram Magare",
                    "first_name": None,
                    "last_name": None,
                    "current_location": None,
                }
            }
        )

        assert result["personal_information"]["first_name"] == "Nilesh"
        assert result["personal_information"]["last_name"] == "Shivram Magare"

    def test_already_populated_first_and_last_name_are_never_touched(self, extractor):
        """A model-provided first_name/last_name is never overwritten,
        even if it looks inconsistent with full_name — this hook only
        fills in a genuine null, never corrects an existing value."""
        original = {
            "personal_information": {
                "full_name": "Ada Lovelace",
                "first_name": "Ada",
                "last_name": "Lovelace",
                "current_location": None,
            }
        }

        result = extractor.preprocess_json(original)

        assert result is original

    def test_only_first_name_populated_is_left_untouched(self, extractor):
        """Only fires when BOTH first_name and last_name are null — a
        partially-populated pair (e.g. the model supplied first_name
        but not last_name) is left exactly as the model returned it,
        not silently "completed"."""
        original = {
            "personal_information": {
                "full_name": "Ada Lovelace",
                "first_name": "Ada",
                "last_name": None,
                "current_location": None,
            }
        }

        result = extractor.preprocess_json(original)

        assert result is original

    def test_single_token_full_name_sets_first_name_only(self, extractor):
        """Defensive, documented behavior for a case not present in this
        project's ground truth (checked directly — zero single-token
        full_name entries exist): first_name becomes the one token;
        last_name stays null rather than guessing one that was never
        written."""
        result = extractor.preprocess_json(
            {
                "personal_information": {
                    "full_name": "Madonna",
                    "first_name": None,
                    "last_name": None,
                    "current_location": None,
                }
            }
        )

        assert result["personal_information"]["first_name"] == "Madonna"
        assert result["personal_information"]["last_name"] is None

    def test_null_full_name_creates_no_fabricated_names(self, extractor):
        """No full_name at all -> nothing to split from; first_name and
        last_name must stay null, never fabricated."""
        original = {
            "personal_information": {
                "full_name": None,
                "first_name": None,
                "last_name": None,
                "current_location": None,
            }
        }

        result = extractor.preprocess_json(original)

        assert result is original

    def test_empty_string_full_name_creates_no_fabricated_names(self, extractor):
        original = {
            "personal_information": {
                "full_name": "",
                "first_name": None,
                "last_name": None,
                "current_location": None,
            }
        }

        result = extractor.preprocess_json(original)

        assert result is original

    def test_full_name_is_never_modified(self, extractor):
        result = extractor.preprocess_json(
            {
                "personal_information": {
                    "full_name": "Chirag Sharma",
                    "first_name": None,
                    "last_name": None,
                    "current_location": None,
                }
            }
        )

        assert result["personal_information"]["full_name"] == "Chirag Sharma"

    def test_unrelated_field_current_location_is_preserved_unchanged(self, extractor):
        result = extractor.preprocess_json(
            {
                "personal_information": {
                    "full_name": "Jane Doe",
                    "first_name": None,
                    "last_name": None,
                    "current_location": "New York, USA",
                }
            }
        )

        assert result["personal_information"]["current_location"] == "New York, USA"

    def test_malformed_shape_returns_input_unchanged(self, extractor):
        """Diagnosing a malformed shape is SchemaValidator.validate()'s
        job, not this method's — it only ever touches a recognizably
        dict-shaped personal_information object."""
        original = {"not_personal_information": 1}

        result = extractor.preprocess_json(original)

        assert result is original

    def test_non_dict_input_returns_input_unchanged(self, extractor):
        original = ["unexpected", "shape"]

        result = extractor.preprocess_json(original)

        assert result is original

    def test_known_limitation_professional_title_prefix_is_not_stripped(self, extractor):
        """Documented, deliberate scope limitation (docs/MILESTONE_23_9_1
        _RESULTS.md): a full_name that includes a leading professional
        title/prefix (e.g. "CA" for Chartered Accountant, matching the
        real "CA Ishika Sharma Resume" dataset entry) is split by
        whitespace token exactly like any other name — this hook does
        NOT recognize or strip titles, since doing so would be exactly
        the kind of generalized name-parsing heuristic Milestone 23.9.1
        was explicitly scoped to avoid. This test documents the known,
        accepted limitation, not a bug."""
        result = extractor.preprocess_json(
            {
                "personal_information": {
                    "full_name": "CA Ishika Sharma",
                    "first_name": None,
                    "last_name": None,
                    "current_location": None,
                }
            }
        )

        assert result["personal_information"]["first_name"] == "CA"
        assert result["personal_information"]["last_name"] == "Ishika Sharma"


@pytest.mark.integration
def test_real_ollama_extract_integration(tmp_path):
    """Optional integration test against a real local Ollama server, the
    real qwen3:1.7b model, and the project's real
    prompts/modules/personal_information.md prompt. Skips itself if
    either the server or the model isn't available."""
    ollama_client = OllamaClient()

    if not ollama_client.is_running():
        pytest.skip("Ollama is not running locally; skipping integration test.")
    if not ollama_client.model_exists(INTEGRATION_TEXT_MODEL):
        pytest.skip(f"Model '{INTEGRATION_TEXT_MODEL}' is not installed; skipping.")

    markdown_path = tmp_path / "resume.md"
    markdown_path.write_text(
        "# Ada Lovelace\n\nMathematician and writer, based in London.\n\n"
        "## Skills\n\n- Mathematics\n- Analytical writing\n",
        encoding="utf-8",
    )

    extractor = PersonalInformationExtractor(ollama_client=ollama_client)

    result = extractor.extract(markdown_path, prompt_id="modules/personal_information")

    assert isinstance(result, PersonalInformationResult)
    assert isinstance(result.json_object, dict)
    assert set(result.json_object.keys()) == {"personal_information"}
