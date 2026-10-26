"""
tests/test_skills_extractor.py

Tests for services/skills_extractor.py (Milestone 14.4).

PromptManager and SchemaValidator run for real (cheap, deterministic,
local operations — the former against a temporary prompt file, the
latter against the project's real schemas/modules/skills_schema.json),
so these tests exercise the real prompt-loading and validation paths.
Only OllamaClient is mocked in the unit tests, since it depends on a
running local model server.

One integration test at the bottom talks to a real local Ollama server,
the real qwen3:1.7b text model, and the project's real
prompts/modules/skills.md prompt. It skips itself (rather than failing)
if Ollama or the model isn't available.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services.base_module_extractor import BaseModuleResult, InvalidJSONError, MarkdownNotFoundError
from services.ollama_client import OllamaClient
from services.prompt_manager import PromptManager
from services.skills_extractor import SkillsExtractor
from validators.schema_validator import SchemaValidationError

INTEGRATION_TEXT_MODEL = "qwen3:1.7b"

VALID_JSON_OBJECT = {"skills": ["Python", "SQL", "Data Analysis"]}


@pytest.fixture
def prompts_dir(tmp_path):
    """A temporary prompts/ tree with one module prompt, so tests never
    touch the project's real prompts/ directory."""
    modules_dir = tmp_path / "prompts" / "modules"
    modules_dir.mkdir(parents=True)
    (modules_dir / "skills.md").write_text(
        "Extract only the candidate's skills as JSON.", encoding="utf-8"
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
    return SkillsExtractor(
        ollama_client=mock_ollama_client,
        prompt_manager=PromptManager(prompts_dir=prompts_dir),
    )


class TestExtract:
    def test_valid_markdown_returns_expected_result(self, tmp_path, extractor):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text(
            "# Ada Lovelace\n\n## Skills\n\n- Python\n- SQL\n- Data Analysis",
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
        markdown_path.write_text("# Ada Lovelace\n\n## Skills", encoding="utf-8")

        extractor.extract(markdown_path)

        prompt_text = (prompts_dir / "modules" / "skills.md").read_text()
        call_kwargs = mock_ollama_client.generate_text.call_args.kwargs
        assert call_kwargs["model"] == "qwen3:1.7b"
        assert call_kwargs["prompt"] == f"{prompt_text}\n\n# Ada Lovelace\n\n## Skills"

    def test_empty_skills_section_returns_empty_array(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Ada Lovelace\n\nNo skills listed.", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({"skills": []})
        )

        result = extractor.extract(markdown_path)

        assert result.json_object == {"skills": []}

    def test_duplicate_skills_are_deduplicated_not_rejected(
        self, tmp_path, extractor, mock_ollama_client
    ):
        """Milestone 22.6 (SKV-001): a duplicate skill used to reach
        schema validation unchanged and raise SchemaValidationError
        (see docs/SKILLS_VALIDATION_DESIGN.md §1/§12 for why that used
        to fail the *entire resume*, not just this module) -- as of
        this milestone, SkillsExtractor.preprocess_json() removes the
        duplicate before schema validation ever sees it, so extract()
        now succeeds. See TestPreprocessJson below for preprocess_json()
        exercised directly, and
        tests/test_base_module_extractor.py::TestPreprocessJsonHook for
        the generic hook this relies on."""
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({"skills": ["Python", "Python"]})
        )

        result = extractor.extract(markdown_path)

        assert result.json_object == {"skills": ["Python"]}

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

    def test_schema_failure_on_missing_skills_array(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({})
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_skills_not_an_array(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({"skills": "Python"})
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_nested_object_item(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({"skills": [{"name": "Python"}]})
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_null_item(self, tmp_path, extractor, mock_ollama_client):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({"skills": ["Python", None]})
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_extra_field_at_root(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({"skills": ["Python"], "languages": ["English"]})
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_missing_markdown_raises_markdown_not_found(self, tmp_path, extractor):
        missing_path = tmp_path / "does_not_exist.md"

        with pytest.raises(MarkdownNotFoundError):
            extractor.extract(missing_path)

    def test_result_is_not_saved_to_disk(self, tmp_path, extractor):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Ada Lovelace\n\n## Skills\n\n- Python", encoding="utf-8")

        result = extractor.extract(markdown_path)

        assert not hasattr(result, "json_path")


class TestPreprocessJson:
    """Milestone 22.6 (SKV-001 + SKV-002,
    docs/SKILLS_VALIDATION_DESIGN.md §1/§12/§13). preprocess_json()
    exercised directly, in isolation from extract()/Ollama/schema
    validation -- see TestExtract.
    test_duplicate_skills_are_deduplicated_not_rejected above for the
    same behavior exercised through the full extract() pipeline."""

    def test_exact_duplicates_removed_keeping_first_occurrence(self, extractor):
        result = extractor.preprocess_json({"skills": ["Python", "SQL", "Python"]})

        assert result == {"skills": ["Python", "SQL"]}

    def test_case_insensitive_duplicates_removed_surviving_copy_not_recased(
        self, extractor
    ):
        """SKV-002: the *first* occurrence's exact casing survives --
        this is a dedup decision, not a normalization one (see
        docs/SKILLS_VALIDATION_DESIGN.md §8's safe-vs-semantic line)."""
        result = extractor.preprocess_json({"skills": ["Power BI", "Power Bi", "Excel"]})

        assert result == {"skills": ["Power BI", "Excel"]}

    def test_whitespace_variant_counts_as_a_duplicate(self, extractor):
        result = extractor.preprocess_json({"skills": ["Python", " python ", "SQL"]})

        assert result == {"skills": ["Python", "SQL"]}

    def test_source_level_duplicate_across_two_categories_kishan_yadav_shape(
        self, extractor
    ):
        """Reproduces the real dataset case docs/SKILLS_VALIDATION_DESIGN.md
        §1 confirmed caused Kishan Yadav's resume to fail schema
        validation entirely: the same tool genuinely listed under two
        different category headings, carried through unmerged into one
        flat array."""
        result = extractor.preprocess_json(
            {
                "skills": [
                    "Microsoft Azure ML",
                    "Azure ML Pipelines",
                    "Databricks",
                    "MLflow",
                    "GitHub Actions",
                    "Azure ML Pipelines",
                    "CI/CD Pipelines",
                ]
            }
        )

        assert result["skills"].count("Azure ML Pipelines") == 1
        assert result == {
            "skills": [
                "Microsoft Azure ML",
                "Azure ML Pipelines",
                "Databricks",
                "MLflow",
                "GitHub Actions",
                "CI/CD Pipelines",
            ]
        }

    def test_no_duplicates_returns_the_same_object_unchanged(self, extractor):
        original = {"skills": ["Python", "SQL"]}

        result = extractor.preprocess_json(original)

        assert result is original

    def test_empty_skills_array_unaffected(self, extractor):
        original = {"skills": []}

        result = extractor.preprocess_json(original)

        assert result is original

    def test_non_string_entries_are_never_deduplicated_against_each_other(
        self, extractor
    ):
        """A non-string entry is a schema violation SchemaValidator
        should report -- preprocess_json() must not mask it by
        treating repeated non-strings as duplicates to silently drop."""
        result = extractor.preprocess_json({"skills": ["Python", None, None, 123, 123]})

        assert result == {"skills": ["Python", None, None, 123, 123]}

    def test_malformed_skills_value_passed_through_unchanged(self, extractor):
        original = {"skills": "not a list"}

        result = extractor.preprocess_json(original)

        assert result is original

    def test_non_dict_input_passed_through_unchanged(self, extractor):
        for value in (["not", "a", "dict"], None, "a string", 42):
            assert extractor.preprocess_json(value) is value

    def test_missing_skills_key_passed_through_unchanged(self, extractor):
        original = {"something_else": ["a", "a"]}

        result = extractor.preprocess_json(original)

        assert result is original

    def test_does_not_mutate_the_input_dict_or_list(self, extractor):
        original_skills = ["Python", "Python", "SQL"]
        original = {"skills": original_skills}

        result = extractor.preprocess_json(original)

        assert original == {"skills": ["Python", "Python", "SQL"]}
        assert original_skills == ["Python", "Python", "SQL"]
        assert result is not original


class TestPreprocessJsonLeadingConjunctionArtifact:
    """Milestone 23.32, P0-A
    (docs/MILESTONE_23_31_SKILLS_RESIDUAL_ERROR_ANALYSIS.md §14/§18,
    docs/MILESTONE_23_32_SKILLS_P0_FIXES_RESULTS.md §3). Exercised
    directly against preprocess_json(), in isolation from
    extract()/Ollama/schema validation, mirroring TestPreprocessJson
    above."""

    def test_leading_and_is_stripped(self, extractor):
        result = extractor.preprocess_json({"skills": ["and Python"]})

        assert result == {"skills": ["Python"]}

    def test_leading_ampersand_is_stripped(self, extractor):
        result = extractor.preprocess_json({"skills": ["& React"]})

        assert result == {"skills": ["React"]}

    def test_and_appearing_mid_string_is_never_touched(self, extractor):
        result = extractor.preprocess_json({"skills": ["Python and Django"]})

        assert result == {"skills": ["Python and Django"]}

    def test_ampersand_inside_a_legitimate_skill_is_never_touched(self, extractor):
        result = extractor.preprocess_json({"skills": ["C & C++"]})

        assert result == {"skills": ["C & C++"]}

    def test_legitimate_leading_punctuation_is_never_touched(self, extractor):
        """A leading "&" or "and" is only ever stripped when it is the
        literal artifact this rule targets -- unrelated leading
        punctuation/casing must survive completely untouched."""
        result = extractor.preprocess_json({"skills": ["C#", "#DevOps", "(ISC)2"]})

        assert result == {"skills": ["C#", "#DevOps", "(ISC)2"]}

    def test_none_and_empty_string_entries_are_never_touched(self, extractor):
        result = extractor.preprocess_json({"skills": ["and Python", None, ""]})

        assert result == {"skills": ["Python", None, ""]}

    def test_non_string_entries_are_never_touched(self, extractor):
        result = extractor.preprocess_json({"skills": [123, ["and", "nested"]]})

        assert result == {"skills": [123, ["and", "nested"]]}

    def test_multiple_skills_are_each_processed_independently(self, extractor):
        result = extractor.preprocess_json(
            {
                "skills": [
                    "Program strategy",
                    "enterprise roadmaps",
                    "and success metrics",
                    "Risk",
                    "issue",
                    "and dependency management",
                    "Requirements gathering",
                    "BRDs",
                    "FRDs",
                    "and solution design",
                ]
            }
        )

        assert result == {
            "skills": [
                "Program strategy",
                "enterprise roadmaps",
                "success metrics",
                "Risk",
                "issue",
                "dependency management",
                "Requirements gathering",
                "BRDs",
                "FRDs",
                "solution design",
            ]
        }

    def test_no_leading_conjunction_returns_the_same_object_unchanged(self, extractor):
        original = {"skills": ["Python", "SQL"]}

        result = extractor.preprocess_json(original)

        assert result is original

    def test_case_insensitive_and_is_stripped(self, extractor):
        result = extractor.preprocess_json({"skills": ["AND Python", "And SQL"]})

        assert result == {"skills": ["Python", "SQL"]}

    def test_chirag_sharma_real_dataset_regression(self, extractor):
        """Milestone 23.32 §2/§11: reproduces the exact live artifact
        confirmed on Chirag Sharma_AVP_Intileo's current
        output/json_final skills array -- all 3 items should recover
        their exact ground-truth form once the leading conjunction is
        stripped."""
        result = extractor.preprocess_json(
            {
                "skills": [
                    "Program strategy",
                    "enterprise roadmaps",
                    "and success metrics",
                    "Risk",
                    "issue",
                    "and dependency management",
                    "Requirements gathering",
                    "BRDs",
                    "FRDs",
                    "and solution design",
                    "Budgeting",
                ]
            }
        )

        assert "and success metrics" not in result["skills"]
        assert "and dependency management" not in result["skills"]
        assert "and solution design" not in result["skills"]
        assert "success metrics" in result["skills"]
        assert "dependency management" in result["skills"]
        assert "solution design" in result["skills"]


class TestPreprocessJsonSpokenLanguageRemoval:
    """Milestone 23.32, P0-B / SKV-003
    (docs/SKILLS_VALIDATION_DESIGN.md §5/§13,
    docs/MILESTONE_23_32_SKILLS_P0_FIXES_RESULTS.md §4)."""

    def test_confirmed_spoken_language_is_removed(self, extractor):
        result = extractor.preprocess_json({"skills": ["Python", "English"]})

        assert result == {"skills": ["Python"]}

    def test_spoken_language_with_proficiency_parenthetical_is_removed(self, extractor):
        result = extractor.preprocess_json(
            {"skills": ["Python", "English (Professional Working Proficiency)"]}
        )

        assert result == {"skills": ["Python"]}

    def test_multiple_spoken_languages_are_all_removed(self, extractor):
        result = extractor.preprocess_json(
            {"skills": ["Tally ERP", "English", "Hindi", "MS Excel"]}
        )

        assert result == {"skills": ["Tally ERP", "MS Excel"]}

    def test_legitimate_programming_language_names_are_preserved(self, extractor):
        """Java/C/R are real programming languages this rule must never
        touch, despite sharing a domain concept ("language") with the
        spoken-language blocklist."""
        result = extractor.preprocess_json({"skills": ["Java", "C", "R", "Python"]})

        assert result == {"skills": ["Java", "C", "R", "Python"]}

    def test_technology_containing_a_language_term_is_preserved(self, extractor):
        """A longer, legitimately-technical phrase that happens to
        contain a spoken-language word must never match -- this rule is
        a bounded whole-string match, never a substring search."""
        result = extractor.preprocess_json({"skills": ["English NLP", "Hindi OCR"]})

        assert result == {"skills": ["English NLP", "Hindi OCR"]}

    def test_ambiguous_value_is_preserved(self, extractor):
        """A value that is not an exact (optionally parenthetical-
        decorated) match to the curated spoken-language list is always
        kept -- this rule never guesses."""
        result = extractor.preprocess_json({"skills": ["English Language Teaching"]})

        assert result == {"skills": ["English Language Teaching"]}

    def test_unrelated_skills_are_unaffected(self, extractor):
        original = {"skills": ["Python", "SQL", "Power BI"]}

        result = extractor.preprocess_json(original)

        assert result is original

    def test_order_is_preserved_around_a_removed_entry(self, extractor):
        result = extractor.preprocess_json(
            {"skills": ["Tally ERP", "CompuOffice", "English", "MS Word", "Hindi", "MS Excel"]}
        )

        assert result == {"skills": ["Tally ERP", "CompuOffice", "MS Word", "MS Excel"]}

    def test_case_insensitive_match_is_removed(self, extractor):
        result = extractor.preprocess_json({"skills": ["english", "HINDI", "Python"]})

        assert result == {"skills": ["Python"]}

    def test_ca_ishika_sharma_real_dataset_regression(self, extractor):
        """Milestone 23.32 §2/§11: reproduces the exact live
        contamination confirmed on CA Ishika Sharma Resume's current
        output/json_final skills array -- English/Hindi should be
        removed, the 5 real technical skills preserved exactly, in
        order."""
        result = extractor.preprocess_json(
            {
                "skills": [
                    "Tally ERP",
                    "CompuOffice",
                    "MS Word",
                    "MS Excel",
                    "PowerPoint",
                    "English",
                    "Hindi",
                ]
            }
        )

        assert result == {
            "skills": ["Tally ERP", "CompuOffice", "MS Word", "MS Excel", "PowerPoint"]
        }

    def test_duplicate_removal_still_applies_after_language_removal(self, extractor):
        """SKV-001/002 still runs after P0-A/P0-B -- unaffected by
        their addition."""
        result = extractor.preprocess_json(
            {"skills": ["Python", "Python", "English", "Hindi"]}
        )

        assert result == {"skills": ["Python"]}


@pytest.mark.integration
def test_real_ollama_extract_integration(tmp_path):
    """Optional integration test against a real local Ollama server, the
    real qwen3:1.7b model, and the project's real
    prompts/modules/skills.md prompt. Skips itself if either the server
    or the model isn't available."""
    ollama_client = OllamaClient()

    if not ollama_client.is_running():
        pytest.skip("Ollama is not running locally; skipping integration test.")
    if not ollama_client.model_exists(INTEGRATION_TEXT_MODEL):
        pytest.skip(f"Model '{INTEGRATION_TEXT_MODEL}' is not installed; skipping.")

    markdown_path = tmp_path / "resume.md"
    markdown_path.write_text(
        "# Ada Lovelace\n\n## Skills\n\n- Mathematics\n- Analytical writing\n- Python\n",
        encoding="utf-8",
    )

    extractor = SkillsExtractor(ollama_client=ollama_client)

    result = extractor.extract(markdown_path, prompt_id="modules/skills")

    assert isinstance(result, BaseModuleResult)
    assert isinstance(result.json_object, dict)
    assert set(result.json_object.keys()) == {"skills"}
    assert isinstance(result.json_object["skills"], list)
