"""
tests/test_languages_extractor.py

Tests for services/languages_extractor.py (Milestone 14.6).

PromptManager and SchemaValidator run for real (cheap, deterministic,
local operations — the former against a temporary prompt file, the
latter against the project's real schemas/modules/languages_schema.json),
so these tests exercise the real prompt-loading and validation paths.
Only OllamaClient is mocked in the unit tests, since it depends on a
running local model server.

One integration test at the bottom talks to a real local Ollama server,
the real qwen3:1.7b text model, and the project's real
prompts/modules/languages.md prompt. It skips itself (rather than
failing) if Ollama or the model isn't available.
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
from services.languages_extractor import LanguagesExtractor
from services.ollama_client import OllamaClient
from services.prompt_manager import PromptManager
from validators.schema_validator import SchemaValidationError

INTEGRATION_TEXT_MODEL = "qwen3:1.7b"

VALID_JSON_OBJECT = {"languages": ["English", "French"]}


@pytest.fixture
def prompts_dir(tmp_path):
    """A temporary prompts/ tree with one module prompt, so tests never
    touch the project's real prompts/ directory."""
    modules_dir = tmp_path / "prompts" / "modules"
    modules_dir.mkdir(parents=True)
    (modules_dir / "languages.md").write_text(
        "Extract only the candidate's languages as JSON.", encoding="utf-8"
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
    return LanguagesExtractor(
        ollama_client=mock_ollama_client,
        prompt_manager=PromptManager(prompts_dir=prompts_dir),
    )


class TestExtract:
    def test_valid_markdown_returns_expected_result(self, tmp_path, extractor):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text(
            "# Ada Lovelace\n\n## Languages\n\n- English\n- French",
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
        markdown_path.write_text("# Ada Lovelace\n\n## Languages", encoding="utf-8")

        extractor.extract(markdown_path)

        prompt_text = (prompts_dir / "modules" / "languages.md").read_text()
        call_kwargs = mock_ollama_client.generate_text.call_args.kwargs
        assert call_kwargs["model"] == "qwen3:1.7b"
        assert call_kwargs["prompt"] == f"{prompt_text}\n\n# Ada Lovelace\n\n## Languages"

    def test_empty_languages_section_returns_empty_array(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text(
            "# Ada Lovelace\n\nNo languages listed.", encoding="utf-8"
        )
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({"languages": []})
        )

        result = extractor.extract(markdown_path)

        assert result.json_object == {"languages": []}

    def test_duplicate_languages_rejected(self, tmp_path, extractor, mock_ollama_client):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({"languages": ["English", "English"]})
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

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

    def test_schema_failure_on_missing_languages_array(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({})
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_languages_not_an_array(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({"languages": "English"})
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_null_item(self, tmp_path, extractor, mock_ollama_client):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({"languages": ["English", None]})
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_nested_object_item(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps(
                {"languages": [{"name": "English", "level": "Native"}]}
            )
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_extra_field_at_root(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({"languages": ["English"], "skills": ["Python"]})
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
            "# Ada Lovelace\n\n## Languages\n\n- English", encoding="utf-8"
        )

        result = extractor.extract(markdown_path)

        assert not hasattr(result, "json_path")


class TestPreprocessJson:
    """Milestone 23.9.2 (Fix 5): preprocess_json() drops known
    programming/technical terms from "languages". These tests call
    preprocess_json() directly, mirroring the style of
    tests/test_skills_extractor.py::TestPreprocessJson and
    tests/test_personal_information_extractor.py::TestPreprocessJson —
    no markdown, no model, no schema validation involved."""

    def test_confirmed_javascript_leak_is_removed(self, extractor):
        # The exact confirmed live case: Mukul Kanojia_Full Stack
        # Developer_Intileo's current output, before this fix.
        result = extractor.preprocess_json({"languages": ["JavaScript"]})

        assert result["languages"] == []

    def test_legitimate_spoken_language_remains_extractable(self, extractor):
        json_object = {"languages": ["Tamil", "Hindi", "Marathi", "English"]}

        result = extractor.preprocess_json(json_object)

        assert result is json_object
        assert result["languages"] == ["Tamil", "Hindi", "Marathi", "English"]

    def test_programming_language_mixed_with_real_languages_is_removed_only(
        self, extractor
    ):
        # Mirrors the actual source shape ("Languages: Java, Python,
        # JavaScript, HTML/CSS") landing next to genuinely spoken ones —
        # only the denylisted entries are dropped; the real language
        # survives, in its original position.
        result = extractor.preprocess_json(
            {"languages": ["Java", "Python", "JavaScript", "Tamil"]}
        )

        assert result["languages"] == ["Tamil"]

    def test_programming_language_present_only_in_skills_never_leaks_in(
        self, extractor
    ):
        # A programming language mentioned elsewhere in the resume (i.e.
        # never placed into the "languages" array by the model in the
        # first place) must not appear merely because it exists in the
        # document — preprocess_json only ever filters, it never adds.
        json_object = {"languages": ["English", "Hindi"]}

        result = extractor.preprocess_json(json_object)

        assert result is json_object
        assert "Python" not in result["languages"]

    def test_case_and_whitespace_variants_are_still_matched(self, extractor):
        result = extractor.preprocess_json(
            {"languages": ["  javascript ", "JAVASCRIPT", "Hindi"]}
        )

        assert result["languages"] == ["Hindi"]

    def test_exact_match_only_not_substring_match(self, extractor):
        # A value that merely contains a denylisted term as a substring
        # (not an exact match after strip/casefold) must survive —
        # demonstrates this is not a broad/fuzzy filter.
        result = extractor.preprocess_json(
            {"languages": ["Javascript Enthusiast", "R Programming"]}
        )

        assert result["languages"] == ["Javascript Enthusiast", "R Programming"]

    def test_additional_evidence_based_terms_are_removed(self, extractor):
        # c, pyspark, vb.net — added from this milestone's own fresh
        # scan (Arjun Sharma, PRAGNESH S. SHAH), not from the original
        # prompt rule 3 enumeration.
        result = extractor.preprocess_json(
            {"languages": ["C", "Pyspark", "VB.NET", "Marathi"]}
        )

        assert result["languages"] == ["Marathi"]

    def test_empty_languages_list_stays_empty(self, extractor):
        json_object = {"languages": []}

        result = extractor.preprocess_json(json_object)

        assert result is json_object
        assert result["languages"] == []

    def test_missing_languages_key_returns_input_unchanged(self, extractor):
        json_object = {"not_languages": ["JavaScript"]}

        result = extractor.preprocess_json(json_object)

        assert result is json_object

    def test_non_list_languages_value_returns_input_unchanged(self, extractor):
        json_object = {"languages": "English"}

        result = extractor.preprocess_json(json_object)

        assert result is json_object

    def test_non_dict_input_returns_input_unchanged(self, extractor):
        json_object = ["JavaScript", "English"]

        result = extractor.preprocess_json(json_object)

        assert result is json_object

    def test_non_string_entries_are_preserved_not_crashed_or_dropped(
        self, extractor
    ):
        # Schema validation (not this hook) is responsible for rejecting
        # a non-string item — preprocess_json must not crash on one or
        # silently discard it.
        json_object = {"languages": [None, 42, "JavaScript", "Hindi"]}

        result = extractor.preprocess_json(json_object)

        assert result["languages"] == [None, 42, "Hindi"]

    def test_survivor_order_and_casing_are_preserved_exactly(self, extractor):
        result = extractor.preprocess_json(
            {"languages": ["English", "JavaScript", "Marathi", "Python", "Hindi"]}
        )

        assert result["languages"] == ["English", "Marathi", "Hindi"]


class TestQualifierPhraseRegressionGuards:
    """Milestone 23.9.3: deterministic regression guards proving the
    Milestone 23.9.2 denylist behavior and general preprocess_json
    pass-through are unaffected by this milestone's prompt-only change
    (prompts/modules/languages.md rule 4). No new preprocess_json logic
    was added this milestone — these tests exist to make that
    non-regression explicit and named, per this milestone's own test
    requirements, rather than relying solely on the pre-existing
    Milestone 23.9.2 tests above to imply it."""

    def test_legitimate_bare_language_unchanged(self, extractor):
        # Requirement 4: a legitimate bare language remains unchanged.
        json_object = {"languages": ["Tamil", "Marathi", "English", "Hindi"]}

        result = extractor.preprocess_json(json_object)

        assert result is json_object
        assert result["languages"] == ["Tamil", "Marathi", "English", "Hindi"]

    def test_programming_language_still_excluded_by_denylist(self, extractor):
        # Requirement 5: the Milestone 23.9.2 denylist still applies —
        # this milestone must not weaken or bypass it.
        result = extractor.preprocess_json(
            {"languages": ["Tamil", "JavaScript", "Marathi", "Python"]}
        )

        assert result["languages"] == ["Tamil", "Marathi"]

    def test_empty_languages_list_still_stays_empty(self, extractor):
        # Requirement 7: empty/no-language input remains empty.
        json_object = {"languages": []}

        result = extractor.preprocess_json(json_object)

        assert result is json_object
        assert result["languages"] == []


@pytest.mark.integration
class TestQualifierPhraseIntegration:
    """Milestone 23.9.3: integration tests against a real local Ollama
    server, the real qwen3:1.7b model, and the project's real,
    just-updated prompts/modules/languages.md — the only way to
    actually verify a prompt-wording change, since this is a model-
    behavior change, not a deterministic code path (see
    docs/MILESTONE_23_9_3_RESULTS.md §4). Each test skips itself if
    Ollama or the model isn't available, matching the file's existing
    integration test's convention.

    Because Ollama is non-deterministic (extensively documented
    throughout this project), a single run of these tests passing or
    failing is one data point, not proof — the targeted regression-suite
    benchmark (docs/MILESTONE_23_9_3_RESULTS.md §7-§8) is the primary
    evidence for whether this prompt change actually helps; these tests
    exist to catch a clear, reproducible failure (e.g. the qualifier
    phrase itself leaking into the output) rather than to guarantee
    the language is extracted on every single run.
    """

    @staticmethod
    def _skip_if_ollama_unavailable(ollama_client):
        if not ollama_client.is_running():
            pytest.skip("Ollama is not running locally; skipping integration test.")
        if not ollama_client.model_exists(INTEGRATION_TEXT_MODEL):
            pytest.skip(f"Model '{INTEGRATION_TEXT_MODEL}' is not installed; skipping.")

    def test_slight_understanding_of_tamil_extracts_tamil(self, tmp_path):
        # Requirement 1: "Slight understanding of Tamil" -> Tamil.
        ollama_client = OllamaClient()
        self._skip_if_ollama_unavailable(ollama_client)

        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text(
            "# Test Candidate\n\n## Personal Details\n\n"
            "- Language known - English, Hindi, Slight understanding of Tamil\n",
            encoding="utf-8",
        )

        extractor = LanguagesExtractor(ollama_client=ollama_client)
        result = extractor.extract(markdown_path, prompt_id="modules/languages")

        languages = result.json_object["languages"]
        assert "Tamil" in languages
        # The qualifier phrase itself must never appear verbatim.
        assert not any("understanding" in lang.lower() for lang in languages)

    def test_can_understand_marathi_extracts_marathi(self, tmp_path):
        # Requirement 2: "Can understand Marathi" -> Marathi.
        ollama_client = OllamaClient()
        self._skip_if_ollama_unavailable(ollama_client)

        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text(
            "# Test Candidate\n\n## Personal Details\n\n"
            "- Languages Known: Hindi, English & Can understand Marathi\n",
            encoding="utf-8",
        )

        extractor = LanguagesExtractor(ollama_client=ollama_client)
        result = extractor.extract(markdown_path, prompt_id="modules/languages")

        languages = result.json_object["languages"]
        assert "Marathi" in languages
        assert not any("understand" in lang.lower() for lang in languages)

    def test_exact_dataset_phrasing_ashishkumarsingh(self, tmp_path):
        # Requirement 3: equivalent qualifier phrasing found in the
        # actual dataset — the literal source line from
        # AshishKumarSingh_RBM_Pune_Stonelam.md, not a paraphrase, with
        # its real surrounding context.
        ollama_client = OllamaClient()
        self._skip_if_ollama_unavailable(ollama_client)

        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text(
            "# Test Candidate\n\n"
            "- **Gender:** Male\n"
            "- **Father's Name:** Mr. Test Father\n"
            "- **Languages Known:** Hindi, English & Can understand Marathi\n\n"
            "## Address\nSome address line.\n",
            encoding="utf-8",
        )

        extractor = LanguagesExtractor(ollama_client=ollama_client)
        result = extractor.extract(markdown_path, prompt_id="modules/languages")

        languages = result.json_object["languages"]
        assert "Hindi" in languages
        assert "English" in languages
        assert "Marathi" in languages

    def test_project_description_language_mention_not_extracted(self, tmp_path):
        # Requirement 6: a sentence containing a language name outside
        # the intended Languages context must not be incorrectly
        # extracted — the literal pattern from Anushka_ (1).md's own
        # markdown, a project description, not a proficiency claim.
        #
        # Deliberately uses a fuller, realistic snippet (not a single
        # bare bullet in isolation): a minimal one-line markdown
        # containing only this sentence was tried first and found, via
        # a dedicated A/B check against the pre-Milestone-23.9.3 prompt
        # (docs/MILESTONE_23_9_3_RESULTS.md §11), to trigger an
        # unrelated, already-documented, pre-existing hallucination
        # pattern (ISSUE-06 in docs/PARSER_ROADMAP.md: the model
        # inventing "the resume is in English" when no language section
        # exists at all) on BOTH the old and new prompt equally — not
        # something this milestone's change causes or is responsible
        # for fixing. This richer snippet mirrors
        # output/markdown/Anushka_ (1).md's own real structure (a named
        # candidate, a proper section heading, realistic surrounding
        # content), which the real dataset file itself extracts
        # correctly ([] — see docs/MILESTONE_23_9_3_RESULTS.md §8).
        ollama_client = OllamaClient()
        self._skip_if_ollama_unavailable(ollama_client)

        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text(
            "# Test Candidate\n\n"
            "## Summary\n\n"
            "AI/ML engineer with experience building NLP systems.\n\n"
            "## Projects\n\n"
            "- Developed a machine translation model for English to "
            "Hindi Translation using PyTorch and Hugging Face's "
            "Transformer Library.\n"
            "- Built a REST API for a recommendation engine using "
            "FastAPI and PostgreSQL.\n\n"
            "## Skills\n\n"
            "- Python, SQL, Django REST Framework\n",
            encoding="utf-8",
        )

        extractor = LanguagesExtractor(ollama_client=ollama_client)
        result = extractor.extract(markdown_path, prompt_id="modules/languages")

        assert result.json_object["languages"] == []


@pytest.mark.integration
def test_real_ollama_extract_integration(tmp_path):
    """Optional integration test against a real local Ollama server, the
    real qwen3:1.7b model, and the project's real
    prompts/modules/languages.md prompt. Skips itself if either the
    server or the model isn't available."""
    ollama_client = OllamaClient()

    if not ollama_client.is_running():
        pytest.skip("Ollama is not running locally; skipping integration test.")
    if not ollama_client.model_exists(INTEGRATION_TEXT_MODEL):
        pytest.skip(f"Model '{INTEGRATION_TEXT_MODEL}' is not installed; skipping.")

    markdown_path = tmp_path / "resume.md"
    markdown_path.write_text(
        "# Ada Lovelace\n\n## Languages\n\n- English\n- French\n\n"
        "## Skills\n\n- Mathematics\n- Python\n",
        encoding="utf-8",
    )

    extractor = LanguagesExtractor(ollama_client=ollama_client)

    result = extractor.extract(markdown_path, prompt_id="modules/languages")

    assert isinstance(result, BaseModuleResult)
    assert isinstance(result.json_object, dict)
    assert set(result.json_object.keys()) == {"languages"}
    assert isinstance(result.json_object["languages"], list)
