"""
tests/test_base_module_extractor.py

Tests for services/base_module_extractor.py (Milestone 14.2).

BaseModuleExtractor is abstract (its three _get_* hooks must be
implemented by a subclass), so these tests define a small
DummyModuleExtractor purely to exercise the shared generic pipeline — it
has no relationship to any real resume module (personal information,
skills, etc.), which is exactly the point: the base class must not know
or care what's being extracted.

PromptManager and SchemaValidator run for real against temporary files;
only OllamaClient is mocked in the unit tests. One integration test at
the bottom talks to a real local Ollama server using the dummy
extractor's own trivial prompt/schema, confirming the generic pipeline
works end to end independent of any specific module extractor.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services.base_module_extractor import (
    BaseModuleExtractor,
    BaseModuleResult,
    InvalidJSONError,
    MarkdownNotFoundError,
)
from services.ollama_client import OllamaClient
from services.prompt_manager import PromptManager
from validators.schema_validator import SchemaValidationError

DUMMY_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {"greeting": {"type": ["string", "null"]}},
    "required": ["greeting"],
    "additionalProperties": False,
}

VALID_JSON_OBJECT = {"greeting": "hello"}


class DummyModuleExtractor(BaseModuleExtractor):
    """A minimal, made-up module extractor used only to exercise
    BaseModuleExtractor's generic pipeline in isolation from any real
    resume module."""

    def __init__(self, schema_path, **kwargs):
        self._schema_path = schema_path
        super().__init__(**kwargs)

    def _get_prompt_id(self) -> str:
        return "dummy/greeting"

    def _get_schema_path(self):
        return self._schema_path

    def _get_model_name(self) -> str:
        return "qwen3:1.7b"


@pytest.fixture
def schema_path(tmp_path):
    path = tmp_path / "dummy_schema.json"
    path.write_text(json.dumps(DUMMY_SCHEMA), encoding="utf-8")
    return path


@pytest.fixture
def prompts_dir(tmp_path):
    """A temporary prompts/ tree with one made-up prompt, so tests never
    touch the project's real prompts/ directory."""
    dummy_dir = tmp_path / "prompts" / "dummy"
    dummy_dir.mkdir(parents=True)
    (dummy_dir / "greeting.md").write_text(
        "Return a JSON greeting.", encoding="utf-8"
    )
    return dummy_dir.parent


@pytest.fixture
def mock_ollama_client():
    client = MagicMock(spec=OllamaClient)
    client.generate_text.return_value = SimpleNamespace(
        response=json.dumps(VALID_JSON_OBJECT)
    )
    return client


@pytest.fixture
def extractor(mock_ollama_client, prompts_dir, schema_path):
    return DummyModuleExtractor(
        schema_path=schema_path,
        ollama_client=mock_ollama_client,
        prompt_manager=PromptManager(prompts_dir=prompts_dir),
    )


class TestSubclassOverride:
    def test_cannot_instantiate_base_class_directly(self):
        with pytest.raises(TypeError):
            BaseModuleExtractor()

    def test_subclass_hooks_return_expected_values(self, extractor, schema_path):
        assert extractor._get_prompt_id() == "dummy/greeting"
        assert extractor._get_schema_path() == schema_path
        assert extractor._get_model_name() == "qwen3:1.7b"

    def test_incomplete_subclass_cannot_be_instantiated(self):
        """A subclass that forgets to implement one of the three
        required hooks must fail at instantiation, not silently work
        with missing behavior."""

        class IncompleteExtractor(BaseModuleExtractor):
            def _get_prompt_id(self) -> str:
                return "x"

            # _get_schema_path and _get_model_name intentionally omitted.

        with pytest.raises(TypeError):
            IncompleteExtractor()

    def test_extract_uses_hooks_for_model_and_prompt_by_default(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Hello", encoding="utf-8")

        extractor.extract(markdown_path)

        call_kwargs = mock_ollama_client.generate_text.call_args.kwargs
        assert call_kwargs["model"] == extractor._get_model_name()


class TestPromptLoading:
    def test_loads_prompt_via_prompt_manager_and_appends_markdown(
        self, tmp_path, extractor, mock_ollama_client, prompts_dir
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Hello", encoding="utf-8")

        extractor.extract(markdown_path)

        prompt_text = (prompts_dir / "dummy" / "greeting.md").read_text()
        call_kwargs = mock_ollama_client.generate_text.call_args.kwargs
        assert call_kwargs["prompt"] == f"{prompt_text}\n\n# Hello"

    def test_explicit_prompt_id_overrides_the_hook(
        self, tmp_path, extractor, mock_ollama_client, prompts_dir
    ):
        (prompts_dir / "dummy" / "other.md").write_text(
            "A different prompt.", encoding="utf-8"
        )
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Hello", encoding="utf-8")

        extractor.extract(markdown_path, prompt_id="dummy/other")

        call_kwargs = mock_ollama_client.generate_text.call_args.kwargs
        assert call_kwargs["prompt"] == "A different prompt.\n\n# Hello"

    def test_explicit_model_name_overrides_the_hook(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Hello", encoding="utf-8")

        extractor.extract(markdown_path, model_name="qwen2.5vl:3b")

        call_kwargs = mock_ollama_client.generate_text.call_args.kwargs
        assert call_kwargs["model"] == "qwen2.5vl:3b"


class TestSchemaLoading:
    def test_default_schema_validator_is_built_from_get_schema_path(
        self, extractor, schema_path
    ):
        assert extractor.schema_validator.schema_path == schema_path

    def test_schema_validation_failure_raises(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Hello", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({"unexpected_field": "surprise"})
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)


class TestJSONParsing:
    def test_valid_json_is_parsed(self, tmp_path, extractor):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Hello", encoding="utf-8")

        result = extractor.extract(markdown_path)

        assert result.json_object == VALID_JSON_OBJECT

    def test_invalid_json_raises(self, tmp_path, extractor, mock_ollama_client):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Hello", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response="Sure, here you go: not valid json"
        )

        with pytest.raises(InvalidJSONError):
            extractor.extract(markdown_path)

    def test_control_character_in_json_string_is_tolerated(
        self, tmp_path, extractor, mock_ollama_client
    ):
        """Milestone 23.4: a response that is syntactically valid JSON
        except for an unescaped control character (e.g. a literal
        newline) inside a string value must still parse successfully.
        _parse_json() calls json.loads() with strict=False specifically
        to permit this common LLM-JSON-generation quirk — see
        docs/KRISHANKANT_JSON_FAILURE_ANALYSIS.md and
        docs/JSON_CONTROL_CHARACTER_ROBUSTNESS_RESULTS.md."""
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Hello", encoding="utf-8")
        # json.dumps() would escape a real newline as "\\n" — build the
        # response text by hand instead, so the response the mock
        # returns contains an actual raw newline byte inside the quoted
        # string, exactly like the defect this test guards against.
        response_with_raw_control_char = json.dumps(VALID_JSON_OBJECT).replace(
            '"hello"', '"hel\nlo"'
        )
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=response_with_raw_control_char
        )

        result = extractor.extract(markdown_path)

        assert result.json_object == {"greeting": "hel\nlo"}

    def test_unterminated_string_still_raises(
        self, tmp_path, extractor, mock_ollama_client
    ):
        """Milestone 23.4 regression guard: strict=False only relaxes
        control-character handling inside strings — a structurally
        malformed response (an unterminated string) must still be
        rejected exactly as before."""
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Hello", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response='{"greeting": "hello}'
        )

        with pytest.raises(InvalidJSONError):
            extractor.extract(markdown_path)


class TestValidation:
    def test_valid_result_passes_through_unchanged(self, tmp_path, extractor):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Hello", encoding="utf-8")

        result = extractor.extract(markdown_path)

        assert result.json_object == VALID_JSON_OBJECT


class TestTiming:
    def test_processing_time_is_measured_and_non_negative(self, tmp_path, extractor):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Hello", encoding="utf-8")

        result = extractor.extract(markdown_path)

        assert isinstance(result, BaseModuleResult)
        assert isinstance(result.processing_time, float)
        assert result.processing_time >= 0


class TestMissingMarkdown:
    def test_missing_markdown_raises(self, tmp_path, extractor):
        with pytest.raises(MarkdownNotFoundError):
            extractor.extract(tmp_path / "does_not_exist.md")


class TestPreprocessJsonHook:
    """Milestone 22.6: the generic preprocess_json() hook, and its
    wiring into extract() between JSON parsing and schema validation.
    Uses DummyModuleExtractor (no override -- confirms the default,
    no-op behavior every existing module extractor still gets) and a
    second, purpose-built dummy subclass that does override the hook
    (confirms the extension point itself works generically, before
    ever touching services/skills_extractor.py's real use of it -- see
    tests/test_skills_extractor.py for that)."""

    def test_default_preprocess_json_is_the_identity_function(self, extractor):
        """The base implementation must return its input completely
        unchanged -- not an equal copy, the exact same object -- for
        any input, including shapes no real module extractor would
        ever actually produce, since a no-op has no business
        inspecting its argument at all."""
        for value in (
            VALID_JSON_OBJECT,
            {"anything": ["at", "all"]},
            ["not", "even", "a", "dict"],
            None,
            "not a dict either",
        ):
            assert extractor.preprocess_json(value) is value

    def test_extract_calls_preprocess_json_with_the_parsed_object(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Hello", encoding="utf-8")
        extractor.preprocess_json = MagicMock(side_effect=lambda obj: obj)

        extractor.extract(markdown_path)

        extractor.preprocess_json.assert_called_once_with(VALID_JSON_OBJECT)

    def test_default_hook_leaves_existing_module_behavior_byte_identical(
        self, tmp_path, extractor
    ):
        """Requirement 3: every existing module extractor -- exercised
        here via DummyModuleExtractor, which does not override
        preprocess_json, exactly like every real module extractor
        except SkillsExtractor as of this milestone -- must behave
        exactly as it did before this hook was added."""
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Hello", encoding="utf-8")

        result = extractor.extract(markdown_path)

        assert result.json_object == VALID_JSON_OBJECT

    def test_overriding_subclass_can_repair_output_before_schema_validation(
        self, tmp_path, mock_ollama_client, prompts_dir, schema_path
    ):
        """Requirement 2: preprocess_json() runs strictly between JSON
        parsing and schema validation -- proven generically here (no
        Skills-specific code at all) by having a dummy subclass's
        override repair a value that would otherwise fail the dummy
        schema's own validation, and confirming extract() succeeds."""

        class RepairingExtractor(DummyModuleExtractor):
            def preprocess_json(self, json_object):
                if isinstance(json_object, dict) and "greeting" not in json_object:
                    return {**json_object, "greeting": None}
                return json_object

        mock_ollama_client.generate_text.return_value = SimpleNamespace(response=json.dumps({}))
        # Without the override, `{}` fails DUMMY_SCHEMA's required
        # "greeting" field -- confirmed by the base DummyModuleExtractor's
        # own existing test_schema_validation_failure_raises test using
        # the same kind of malformed (here, incomplete) response.
        repairing_extractor = RepairingExtractor(
            schema_path=schema_path,
            ollama_client=mock_ollama_client,
            prompt_manager=PromptManager(prompts_dir=prompts_dir),
        )

        result = repairing_extractor.extract(markdown_path=_write_markdown(tmp_path))

        assert result.json_object == {"greeting": None}

    def test_preprocess_json_return_value_is_what_gets_schema_validated(
        self, tmp_path, mock_ollama_client, prompts_dir, schema_path
    ):
        """The exact value preprocess_json() returns -- not the
        original parsed object -- must be what SchemaValidator sees
        and what extract() ultimately returns."""

        class BreakingExtractor(DummyModuleExtractor):
            def preprocess_json(self, json_object):
                return {"totally": "different", "shape": True}

        breaking_extractor = BreakingExtractor(
            schema_path=schema_path,
            ollama_client=mock_ollama_client,
            prompt_manager=PromptManager(prompts_dir=prompts_dir),
        )

        with pytest.raises(SchemaValidationError):
            breaking_extractor.extract(markdown_path=_write_markdown(tmp_path))


def _write_markdown(tmp_path):
    markdown_path = tmp_path / "resume.md"
    markdown_path.write_text("# Hello", encoding="utf-8")
    return markdown_path


@pytest.mark.integration
def test_real_ollama_generic_pipeline_integration(tmp_path, schema_path, prompts_dir):
    """Confirms the generic base pipeline works end to end against a
    real local Ollama server, using nothing module-specific — just the
    DummyModuleExtractor's own trivial prompt/schema. Skips itself if
    Ollama or the model isn't available."""
    ollama_client = OllamaClient()

    if not ollama_client.is_running():
        pytest.skip("Ollama is not running locally; skipping integration test.")
    if not ollama_client.model_exists("qwen3:1.7b"):
        pytest.skip("Model 'qwen3:1.7b' is not installed; skipping.")

    (prompts_dir / "dummy" / "greeting.md").write_text(
        "You must return ONLY the following exact JSON structure, with no "
        "other text, no markdown, and no explanation:\n\n"
        '{"greeting": "hello"}\n\n'
        'Replace "hello" with a short greeting of your choice, but keep the '
        'exact key name "greeting".',
        encoding="utf-8",
    )
    markdown_path = tmp_path / "resume.md"
    markdown_path.write_text("# Anything\n", encoding="utf-8")

    extractor = DummyModuleExtractor(
        schema_path=schema_path,
        ollama_client=ollama_client,
        prompt_manager=PromptManager(prompts_dir=prompts_dir),
    )

    result = extractor.extract(markdown_path)

    assert isinstance(result, BaseModuleResult)
    assert isinstance(result.json_object, dict)
    assert set(result.json_object.keys()) == {"greeting"}
