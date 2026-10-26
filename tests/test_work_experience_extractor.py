"""
tests/test_work_experience_extractor.py

Tests for services/work_experience_extractor.py (Milestone 15.2).

PromptManager and SchemaValidator run for real (cheap, deterministic,
local operations — the former against a temporary prompt file, the
latter against the project's real
schemas/modules/work_experience_schema.json), so these tests exercise
the real prompt-loading and validation paths. Only OllamaClient is
mocked in the unit tests, since it depends on a running local model
server.

One integration test at the bottom talks to a real local Ollama server,
the real qwen3:1.7b text model, and the project's real
prompts/modules/work_experience.md prompt. It skips itself (rather than
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
from services.ollama_client import OllamaClient
from services.prompt_manager import PromptManager
from services.work_experience_extractor import WorkExperienceExtractor, WorkExperienceResult
from validators.schema_validator import SchemaValidationError

INTEGRATION_TEXT_MODEL = "qwen3:1.7b"

VALID_JOB = {
    "company": "Acme Corp",
    "designation": "Senior Data Analyst",
    "employment_type": "Full-time",
    "location": "Bengaluru, India",
    "start_date": "Jun 2022",
    "end_date": None,
    "currently_working": True,
    "duration": None,
    "responsibilities": ["Built dashboards used by the finance team"],
    "technologies": ["Python", "SQL"],
}

VALID_JSON_OBJECT = {"work_experience": [VALID_JOB]}


@pytest.fixture
def prompts_dir(tmp_path):
    """A temporary prompts/ tree with one module prompt, so tests never
    touch the project's real prompts/ directory."""
    modules_dir = tmp_path / "prompts" / "modules"
    modules_dir.mkdir(parents=True)
    (modules_dir / "work_experience.md").write_text(
        "Extract only the candidate's work experience as JSON.", encoding="utf-8"
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
    return WorkExperienceExtractor(
        ollama_client=mock_ollama_client,
        prompt_manager=PromptManager(prompts_dir=prompts_dir),
    )


class TestExtract:
    def test_valid_markdown_returns_expected_result(self, tmp_path, extractor):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text(
            "# Ada Lovelace\n\n## Experience\n\n"
            "Senior Data Analyst, Acme Corp, Jun 2022 - Present",
            encoding="utf-8",
        )

        result = extractor.extract(markdown_path)

        assert isinstance(result, BaseModuleResult)
        assert isinstance(result, WorkExperienceResult)
        assert result.json_object == VALID_JSON_OBJECT
        assert result.processing_time >= 0

    def test_prompt_is_sent_with_markdown_appended(
        self, tmp_path, extractor, mock_ollama_client, prompts_dir
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Ada Lovelace\n\n## Experience", encoding="utf-8")

        extractor.extract(markdown_path)

        prompt_text = (prompts_dir / "modules" / "work_experience.md").read_text()
        call_kwargs = mock_ollama_client.generate_text.call_args.kwargs
        assert call_kwargs["model"] == "qwen3:1.7b"
        assert call_kwargs["prompt"] == f"{prompt_text}\n\n# Ada Lovelace\n\n## Experience"

    def test_empty_work_experience_returns_empty_array(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Ada Lovelace\n\nNo experience listed.", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({"work_experience": []})
        )

        result = extractor.extract(markdown_path)

        assert result.json_object == {"work_experience": []}

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

    def test_schema_failure_on_missing_work_experience_key(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({})
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_work_experience_not_an_array(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({"work_experience": "Acme Corp"})
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_job_missing_required_field(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        incomplete_job = dict(VALID_JOB)
        del incomplete_job["technologies"]
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({"work_experience": [incomplete_job]})
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_job_extra_field(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        job_with_extra = dict(VALID_JOB)
        job_with_extra["salary"] = "100000"
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({"work_experience": [job_with_extra]})
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_currently_working_not_boolean(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        bad_job = dict(VALID_JOB)
        bad_job["currently_working"] = "yes"
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({"work_experience": [bad_job]})
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_responsibilities_not_an_array(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        bad_job = dict(VALID_JOB)
        bad_job["responsibilities"] = "Did stuff"
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({"work_experience": [bad_job]})
        )

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_extra_field_at_root(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({"work_experience": [], "skills": ["Python"]})
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
            "# Ada Lovelace\n\n## Experience\n\nSenior Data Analyst, Acme Corp",
            encoding="utf-8",
        )

        result = extractor.extract(markdown_path)

        assert not hasattr(result, "json_path")


class TestPreprocessJson:
    """Milestone 23.16: preprocess_json() removes exact-duplicate
    `technologies` entries before schema validation — see
    docs/MILESTONE_23_16_TECHNOLOGIES_DEDUP_INVESTIGATION.md §5/§9/§10
    for the evidence and design behind exactly what this does and does
    not do. preprocess_json() exercised directly, in isolation from
    extract()/Ollama/schema validation, the same way
    tests/test_skills_extractor.py's equivalent class does; the last
    test in this class exercises the full extract() pipeline instead,
    to prove the real Arjun Sharma failure is actually prevented end to
    end, not just at the unit level.
    """

    def test_exact_duplicate_technology_removed_keeping_first_occurrence(self, extractor):
        job = dict(VALID_JOB, technologies=["Python", "Pandas", "Python", "SQL", "Pandas"])

        result = extractor.preprocess_json({"work_experience": [job]})

        assert result["work_experience"][0]["technologies"] == ["Python", "Pandas", "SQL"]

    def test_multiple_distinct_duplicates_all_removed(self, extractor):
        job = dict(
            VALID_JOB,
            technologies=["A", "B", "A", "C", "B", "D", "C"],
        )

        result = extractor.preprocess_json({"work_experience": [job]})

        assert result["work_experience"][0]["technologies"] == ["A", "B", "C", "D"]

    def test_duplicate_appearing_non_consecutively_is_removed(self, extractor):
        """The real Arjun Sharma pattern shape: a duplicated cluster
        separated by unrelated items in between, not adjacent repeats."""
        job = dict(
            VALID_JOB,
            technologies=["Django", "Flask", "SQL", "Github", "Django", "Flask"],
        )

        result = extractor.preprocess_json({"work_experience": [job]})

        assert result["work_experience"][0]["technologies"] == [
            "Django",
            "Flask",
            "SQL",
            "Github",
        ]

    def test_first_occurrence_order_is_preserved(self, extractor):
        job = dict(VALID_JOB, technologies=["SQL", "Python", "SQL", "Java", "Python"])

        result = extractor.preprocess_json({"work_experience": [job]})

        assert result["work_experience"][0]["technologies"] == ["SQL", "Python", "Java"]

    def test_already_unique_array_is_returned_unchanged(self, extractor):
        job = dict(VALID_JOB, technologies=["Python", "SQL", "Java"])
        original = {"work_experience": [job]}

        result = extractor.preprocess_json(original)

        assert result is original

    def test_empty_technologies_array_is_unaffected(self, extractor):
        job = dict(VALID_JOB, technologies=[])
        original = {"work_experience": [job]}

        result = extractor.preprocess_json(original)

        assert result is original

    def test_null_technologies_is_unaffected(self, extractor):
        """technologies isn't schema-valid as null (must be an array),
        but preprocess_json() must not itself crash on it or treat it
        as a shape to fix — that's SchemaValidator's job."""
        job = dict(VALID_JOB, technologies=None)
        original = {"work_experience": [job]}

        result = extractor.preprocess_json(original)

        assert result is original

    def test_missing_technologies_key_is_unaffected(self, extractor):
        job = dict(VALID_JOB)
        del job["technologies"]
        original = {"work_experience": [job]}

        result = extractor.preprocess_json(original)

        assert result is original

    def test_legitimate_near_duplicates_adf_and_azure_data_factory_both_kept(self, extractor):
        """Real dataset example
        (docs/MILESTONE_23_16_TECHNOLOGIES_DEDUP_INVESTIGATION.md §7):
        these are two distinct strings the model wrote for two
        different-looking mentions in the source text, not spelling
        variants of the same duplicate — must never be merged."""
        job = dict(VALID_JOB, technologies=["ADF", "Azure Data Factory", "Databricks"])
        original = {"work_experience": [job]}

        result = extractor.preprocess_json(original)

        assert result is original
        assert result["work_experience"][0]["technologies"] == [
            "ADF",
            "Azure Data Factory",
            "Databricks",
        ]

    def test_technologies_deduplicated_independently_per_job(self, extractor):
        job_a = dict(VALID_JOB, company="Acme Corp", technologies=["Python", "Python", "SQL"])
        job_b = dict(VALID_JOB, company="Beta LLC", technologies=["SQL", "SQL", "Java"])

        result = extractor.preprocess_json({"work_experience": [job_a, job_b]})

        assert result["work_experience"][0]["technologies"] == ["Python", "SQL"]
        assert result["work_experience"][1]["technologies"] == ["SQL", "Java"]

    def test_responsibilities_are_never_touched(self, extractor):
        job = dict(
            VALID_JOB,
            technologies=["Python", "Python"],
            responsibilities=["Did a thing", "Did a thing"],
        )

        result = extractor.preprocess_json({"work_experience": [job]})

        # technologies deduplicated...
        assert result["work_experience"][0]["technologies"] == ["Python"]
        # ...but responsibilities' own duplicate is left completely alone.
        assert result["work_experience"][0]["responsibilities"] == [
            "Did a thing",
            "Did a thing",
        ]

    def test_all_other_job_fields_are_byte_for_byte_unchanged(self, extractor):
        job = dict(VALID_JOB, technologies=["Python", "Python"])

        result = extractor.preprocess_json({"work_experience": [job]})

        changed_job = result["work_experience"][0]
        for key in (
            "company",
            "designation",
            "employment_type",
            "location",
            "start_date",
            "end_date",
            "currently_working",
            "duration",
            "responsibilities",
        ):
            assert changed_job[key] == job[key]

    def test_arjun_sharma_real_duplicate_pattern_is_fully_deduplicated(self, extractor):
        """The exact array captured from this project's own real
        benchmark failure
        (docs/MILESTONE_23_16_TECHNOLOGIES_DEDUP_INVESTIGATION.md §7):
        'Django', 'Flask', 'Tensorflow', 'Pytorch' each appeared twice,
        tripping technologies' uniqueItems constraint and failing the
        whole resume's pipeline run."""
        job = dict(
            VALID_JOB,
            technologies=[
                "Databricks", "Azure", "Kafka", "ADF", "Data bases", "Warehouses",
                "Analytical systems", "Python", "Django", "Flask", "Tensorflow",
                "Pytorch", "SQL", "Visual studio", "Github", "Django", "Flask",
                "Tensorflow", "Pytorch",
            ],
        )

        result = extractor.preprocess_json({"work_experience": [job]})

        deduplicated = result["work_experience"][0]["technologies"]
        assert deduplicated == [
            "Databricks", "Azure", "Kafka", "ADF", "Data bases", "Warehouses",
            "Analytical systems", "Python", "Django", "Flask", "Tensorflow",
            "Pytorch", "SQL", "Visual studio", "Github",
        ]
        assert len(deduplicated) == len(set(deduplicated))

    def test_full_extract_pipeline_recovers_from_a_would_be_unique_items_failure(
        self, tmp_path, extractor, mock_ollama_client
    ):
        """Milestone 23.16's actual goal, proven end to end: the exact
        Arjun Sharma duplicate-technologies response, run through the
        real extract() pipeline (parse -> preprocess_json() ->
        SchemaValidator.validate()), must now succeed instead of
        raising SchemaValidationError."""
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        duplicated_job = dict(
            VALID_JOB,
            technologies=[
                "Databricks", "Azure", "Kafka", "ADF", "Data bases", "Warehouses",
                "Analytical systems", "Python", "Django", "Flask", "Tensorflow",
                "Pytorch", "SQL", "Visual studio", "Github", "Django", "Flask",
                "Tensorflow", "Pytorch",
            ],
        )
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps({"work_experience": [duplicated_job]})
        )

        result = extractor.extract(markdown_path)

        technologies = result.json_object["work_experience"][0]["technologies"]
        assert len(technologies) == len(set(technologies))
        assert technologies.count("Django") == 1
        assert technologies.count("Pytorch") == 1


class TestBareArrayWrapping:
    """Milestone 23.22 (docs/PARSER_RELIABILITY_HARDENING.md): preprocess_json()
    wraps a bare JSON array response in `{"work_experience": [...]}`
    before schema validation — see services/work_experience_extractor.py's
    own module/class docstrings for the evidence (PRAGNESH S. SHAH...,
    Milestone 23.21's stability analysis; Arjun Sharma, historically
    since Milestone 23.14). Exercised directly, in isolation from
    extract()/Ollama/schema validation, the same way this file's other
    TestPreprocessJson-style classes do; the last test in this class
    exercises the full extract() pipeline instead, to prove a real
    bare-array response is actually recovered end to end, not just at
    the unit level.
    """

    def test_bare_array_is_wrapped_in_work_experience_key(self, extractor):
        bare_array = [dict(VALID_JOB)]

        result = extractor.preprocess_json(bare_array)

        assert result == {"work_experience": [VALID_JOB]}

    def test_real_pragnesh_style_multi_job_bare_array_is_wrapped(self, extractor):
        """Shape reproduced from this project's own benchmark logs
        (docs/PARSER_STABILITY_ANALYSIS.md §6): several complete job
        objects returned as a top-level array, no wrapping object."""
        jobs = [
            dict(VALID_JOB, company="Comfort Securities Ltd."),
            dict(VALID_JOB, company="Eagle Security and Personnel Service"),
            dict(VALID_JOB, company="KMI Business Technologies Pvt. Ltd."),
        ]

        result = extractor.preprocess_json(list(jobs))

        assert isinstance(result, dict)
        assert list(result.keys()) == ["work_experience"]
        assert len(result["work_experience"]) == 3
        assert result["work_experience"][0]["company"] == "Comfort Securities Ltd."

    def test_wrapped_array_still_gets_technologies_and_date_transforms(self, extractor):
        """The bare-array wrap happens first, so the two other
        preprocess_json() transforms still apply to its contents —
        composition proof, mirroring
        TestDateNormalization::test_composes_correctly_with_technologies_deduplication."""
        bare_array = [
            dict(
                VALID_JOB,
                start_date="27 Jul 2015",
                end_date="01 Jun 2015",
                technologies=["Python", "Python", "SQL"],
            )
        ]

        result = extractor.preprocess_json(bare_array)

        job = result["work_experience"][0]
        assert job["start_date"] == "27th July 2015"
        assert job["end_date"] == "01st June 2015"
        assert job["technologies"] == ["Python", "SQL"]

    def test_already_wrapped_object_is_unaffected(self, extractor):
        original = {"work_experience": [dict(VALID_JOB)]}

        result = extractor.preprocess_json(original)

        assert result["work_experience"] == original["work_experience"]

    def test_empty_bare_array_is_wrapped_into_empty_work_experience(self, extractor):
        result = extractor.preprocess_json([])

        assert result == {"work_experience": []}

    def test_non_list_non_dict_json_object_is_unaffected(self, extractor):
        assert extractor.preprocess_json("not valid at all") == "not valid at all"
        assert extractor.preprocess_json(None) is None
        assert extractor.preprocess_json(42) == 42

    def test_full_extract_pipeline_recovers_from_a_bare_array_response(
        self, tmp_path, extractor, mock_ollama_client
    ):
        """Milestone 23.22's actual goal, proven end to end: a bare-array
        response, run through the real extract() pipeline (parse ->
        preprocess_json() -> SchemaValidator.validate()), must now
        succeed instead of raising SchemaValidationError."""
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response=json.dumps([dict(VALID_JOB)])
        )

        result = extractor.extract(markdown_path)

        assert result.json_object == {"work_experience": [VALID_JOB]}


class TestDateNormalization:
    """Milestone 23.18: preprocess_json() normalizes `start_date`/
    `end_date` values that are exactly `"<day> <abbreviated month>
    <year>"` (e.g. `"27 Jul 2015"`) to `"<day><ordinal suffix> <full
    month name> <year>"` (e.g. `"27th July 2015"`) before schema
    validation — see docs/MILESTONE_23_18_DATE_NORMALIZATION_RESULTS.md
    §3 for why this exact, narrow shape (day required present) was
    chosen over the broader "always expand abbreviated months"
    transform originally considered: the broader version was measured,
    against this project's own ground truth, to silently break 11
    currently-correct matches across 6 resumes. `technologies`
    deduplication (Milestone 23.16) is covered by TestPreprocessJson
    above; this class covers only the date-normalization half of
    preprocess_json(), exercised the same way, in isolation from
    extract()/Ollama/schema validation.
    """

    # ---- real dataset examples (AshishKumarSingh_RBM_Pune_Stonelam,
    # docs/MILESTONE_23_18_DATE_NORMALIZATION_RESULTS.md §3) ----

    def test_real_example_start_date_day_with_st_suffix(self, extractor):
        job = dict(VALID_JOB, start_date="01 Jun 2015", end_date=None)

        result = extractor.preprocess_json({"work_experience": [job]})

        assert result["work_experience"][0]["start_date"] == "01st June 2015"

    def test_real_example_end_date_day_with_st_suffix(self, extractor):
        job = dict(VALID_JOB, start_date=None, end_date="01 Jun 2015")

        result = extractor.preprocess_json({"work_experience": [job]})

        assert result["work_experience"][0]["end_date"] == "01st June 2015"

    def test_real_example_th_suffix(self, extractor):
        job = dict(VALID_JOB, start_date="07 Nov 2014", end_date="27 Oct 2014")

        result = extractor.preprocess_json({"work_experience": [job]})

        updated = result["work_experience"][0]
        assert updated["start_date"] == "07th November 2014"
        assert updated["end_date"] == "27th October 2014"

    def test_real_example_nd_suffix(self, extractor):
        job = dict(VALID_JOB, start_date="12 Dec 2009", end_date="08 Apr 2013")

        result = extractor.preprocess_json({"work_experience": [job]})

        updated = result["work_experience"][0]
        assert updated["start_date"] == "12th December 2009"
        assert updated["end_date"] == "08th April 2013"

    def test_real_example_both_start_and_end_normalized_in_same_entry(self, extractor):
        """AshishKumarSingh's GLAZE ROCKS entry: both fields need
        normalizing, independently, in the same job."""
        job = dict(VALID_JOB, start_date="07 Nov 2014", end_date="01 Jun 2015")

        result = extractor.preprocess_json({"work_experience": [job]})

        updated = result["work_experience"][0]
        assert updated["start_date"] == "07th November 2014"
        assert updated["end_date"] == "01st June 2015"

    def test_real_example_normalized_across_multiple_job_entries(self, extractor):
        """AshishKumarSingh's own 4 affected entries, normalized
        independently of one another within one resume's array."""
        jobs = [
            dict(VALID_JOB, company="CLASSIC FLOORINGS", start_date="27 Jul 2015", end_date=None),
            dict(VALID_JOB, company="GLAZE ROCKS", start_date="07 Nov 2014", end_date="01 Jun 2015"),
            dict(VALID_JOB, company="ESCON ELEVATORS", start_date="09 Apr 2013", end_date="27 Oct 2014"),
            dict(VALID_JOB, company="RAJ STEELS", start_date="12 Dec 2009", end_date="08 Apr 2013"),
        ]

        result = extractor.preprocess_json({"work_experience": jobs})

        updated = result["work_experience"]
        assert updated[0]["start_date"] == "27th July 2015"
        assert updated[1]["start_date"] == "07th November 2014"
        assert updated[1]["end_date"] == "01st June 2015"
        assert updated[2]["start_date"] == "09th April 2013"
        assert updated[2]["end_date"] == "27th October 2014"
        assert updated[3]["start_date"] == "12th December 2009"
        assert updated[3]["end_date"] == "08th April 2013"

    # ---- ordinal-suffix rule coverage ----

    @pytest.mark.parametrize(
        "day, expected_suffix",
        [
            ("1", "1st"), ("2", "2nd"), ("3", "3rd"), ("4", "4th"),
            ("11", "11th"), ("12", "12th"), ("13", "13th"),
            ("21", "21st"), ("22", "22nd"), ("23", "23rd"),
            ("31", "31st"),
        ],
    )
    def test_ordinal_suffix_follows_standard_english_rule(self, extractor, day, expected_suffix):
        job = dict(VALID_JOB, start_date=f"{day} Jan 2020", end_date=None)

        result = extractor.preprocess_json({"work_experience": [job]})

        assert result["work_experience"][0]["start_date"] == f"{expected_suffix} January 2020"

    def test_leading_zero_in_day_is_preserved_not_reformatted(self, extractor):
        """'07' must become '07th', never '7th' — the digits themselves
        are never rewritten, only the suffix is appended."""
        job = dict(VALID_JOB, start_date="07 Jan 2020", end_date=None)

        result = extractor.preprocess_json({"work_experience": [job]})

        assert result["work_experience"][0]["start_date"] == "07th January 2020"

    # ---- month-abbreviation coverage ----

    @pytest.mark.parametrize(
        "abbreviation, full_name",
        [
            ("Jan", "January"), ("Feb", "February"), ("Mar", "March"),
            ("Apr", "April"), ("Jun", "June"), ("Jul", "July"),
            ("Aug", "August"), ("Sep", "September"), ("Sept", "September"),
            ("Oct", "October"), ("Nov", "November"), ("Dec", "December"),
        ],
    )
    def test_every_abbreviated_month_is_expanded(self, extractor, abbreviation, full_name):
        job = dict(VALID_JOB, start_date=f"15 {abbreviation} 2020", end_date=None)

        result = extractor.preprocess_json({"work_experience": [job]})

        assert result["work_experience"][0]["start_date"] == f"15th {full_name} 2020"

    # ---- already-canonical values are left unchanged ----

    def test_already_full_month_with_ordinal_is_unchanged(self, extractor):
        job = dict(VALID_JOB, start_date="27th July 2015", end_date=None)
        original = {"work_experience": [job]}

        result = extractor.preprocess_json(original)

        assert result is original

    def test_full_month_without_ordinal_is_left_out_of_scope(self, extractor):
        """Never observed as a real model-output shape in this
        project's data (§3) — deliberately not guessed at."""
        job = dict(VALID_JOB, start_date="27 July 2015", end_date=None)
        original = {"work_experience": [job]}

        result = extractor.preprocess_json(original)

        assert result is original

    # ---- negative controls: must NOT be normalized ----

    def test_bare_month_and_year_is_not_normalized(self, extractor):
        """Explicitly out of scope per §3: expanding month-only values
        (no day) was measured to break 11 currently-correct matches
        elsewhere in this project's own benchmark."""
        job = dict(VALID_JOB, start_date="Jan 2023", end_date=None)
        original = {"work_experience": [job]}

        result = extractor.preprocess_json(original)

        assert result is original

    def test_bare_full_month_and_year_is_not_normalized(self, extractor):
        job = dict(VALID_JOB, start_date="January 2023", end_date=None)
        original = {"work_experience": [job]}

        result = extractor.preprocess_json(original)

        assert result is original

    def test_numeric_mm_yyyy_is_not_normalized(self, extractor):
        job = dict(VALID_JOB, start_date="09/2021", end_date=None)
        original = {"work_experience": [job]}

        result = extractor.preprocess_json(original)

        assert result is original

    def test_iso_yyyy_mm_is_not_normalized(self, extractor):
        job = dict(VALID_JOB, start_date="2024-01", end_date=None)
        original = {"work_experience": [job]}

        result = extractor.preprocess_json(original)

        assert result is original

    def test_bare_year_is_not_normalized(self, extractor):
        job = dict(VALID_JOB, start_date="2022", end_date=None)
        original = {"work_experience": [job]}

        result = extractor.preprocess_json(original)

        assert result is original

    def test_ambiguous_free_text_date_is_not_normalized(self, extractor):
        """'Present' and similarly ambiguous free-text values must
        never be altered or interpreted."""
        job = dict(VALID_JOB, start_date="Jan 2022", end_date="Present")
        original = {"work_experience": [job]}

        result = extractor.preprocess_json(original)

        assert result is original

    def test_null_start_and_end_date_are_unaffected(self, extractor):
        job = dict(VALID_JOB, start_date=None, end_date=None)
        original = {"work_experience": [job]}

        result = extractor.preprocess_json(original)

        assert result is original

    def test_missing_date_is_never_inferred(self, extractor):
        """A currently-working job with no end_date must stay None —
        this transform never invents a value for a missing date."""
        job = dict(VALID_JOB, start_date="15 Jun 2020", end_date=None, currently_working=True)

        result = extractor.preprocess_json({"work_experience": [job]})

        assert result["work_experience"][0]["end_date"] is None

    # ---- scope: only start_date/end_date are ever touched ----

    def test_duration_is_never_touched_even_when_date_like(self, extractor):
        job = dict(
            VALID_JOB,
            start_date="27 Jul 2015",
            end_date=None,
            duration="8 Years & 1 month",
        )

        result = extractor.preprocess_json({"work_experience": [job]})

        assert result["work_experience"][0]["duration"] == "8 Years & 1 month"

    def test_currently_working_is_never_touched(self, extractor):
        job = dict(VALID_JOB, start_date="27 Jul 2015", end_date=None, currently_working=True)

        result = extractor.preprocess_json({"work_experience": [job]})

        assert result["work_experience"][0]["currently_working"] is True

    def test_all_other_job_fields_are_byte_for_byte_unchanged(self, extractor):
        job = dict(VALID_JOB, start_date="27 Jul 2015")

        result = extractor.preprocess_json({"work_experience": [job]})

        updated = result["work_experience"][0]
        for key in (
            "company",
            "designation",
            "employment_type",
            "location",
            "currently_working",
            "duration",
            "responsibilities",
            "technologies",
        ):
            assert updated[key] == job[key]

    def test_composes_correctly_with_technologies_deduplication(self, extractor):
        """Both Milestone 23.16 and 23.18 transforms fire together on
        the same job without interfering with each other."""
        job = dict(
            VALID_JOB,
            start_date="27 Jul 2015",
            end_date="01 Jun 2015",
            technologies=["Python", "Python", "SQL"],
        )

        result = extractor.preprocess_json({"work_experience": [job]})

        updated = result["work_experience"][0]
        assert updated["start_date"] == "27th July 2015"
        assert updated["end_date"] == "01st June 2015"
        assert updated["technologies"] == ["Python", "SQL"]

    def test_regression_underlying_calendar_date_is_never_changed(self, extractor):
        """Proves the transform rewrites representation only: the day,
        month, and year encoded in the output always match the input,
        for a sweep of real and synthetic day/month/year combinations."""
        month_names = {
            "Jan": ("January", 1), "Mar": ("March", 3), "Sept": ("September", 9),
            "Dec": ("December", 12),
        }
        for abbr, (full, month_num) in month_names.items():
            for day in ("01", "11", "22", "31" if month_num != 2 else "28"):
                job = dict(VALID_JOB, start_date=f"{day} {abbr} 2019", end_date=None)
                result = extractor.preprocess_json({"work_experience": [job]})
                normalized = result["work_experience"][0]["start_date"]
                # Same day digits, same full month name, same year — only
                # an ordinal suffix and month-name expansion were added.
                assert normalized.startswith(day)
                assert full in normalized
                assert normalized.endswith("2019")

    def test_already_unique_technologies_and_already_canonical_dates_return_original(self, extractor):
        """When neither the 23.16 nor the 23.18 transform has anything
        to do, preprocess_json() must return the exact original object
        (identity), not just an equal copy."""
        job = dict(
            VALID_JOB,
            start_date="27th July 2015",
            end_date=None,
            technologies=["Python", "SQL"],
        )
        original = {"work_experience": [job]}

        result = extractor.preprocess_json(original)

        assert result is original


@pytest.mark.integration
def test_real_ollama_extract_integration(tmp_path):
    """Optional integration test against a real local Ollama server, the
    real qwen3:1.7b model, and the project's real
    prompts/modules/work_experience.md prompt. Skips itself if either
    the server or the model isn't available."""
    ollama_client = OllamaClient()

    if not ollama_client.is_running():
        pytest.skip("Ollama is not running locally; skipping integration test.")
    if not ollama_client.model_exists(INTEGRATION_TEXT_MODEL):
        pytest.skip(f"Model '{INTEGRATION_TEXT_MODEL}' is not installed; skipping.")

    markdown_path = tmp_path / "resume.md"
    markdown_path.write_text(
        "# Ada Lovelace\n\n## Work Experience\n\n"
        "**Senior Data Analyst**, Acme Corp — Bengaluru, India\n"
        "Jun 2022 - Present\n"
        "- Built dashboards used by the finance team for monthly close\n"
        "- Led migration of reporting pipeline to a new warehouse\n"
        "Technologies: Python, SQL, Power BI\n",
        encoding="utf-8",
    )

    extractor = WorkExperienceExtractor(ollama_client=ollama_client)

    result = extractor.extract(markdown_path, prompt_id="modules/work_experience")

    assert isinstance(result, BaseModuleResult)
    assert isinstance(result.json_object, dict)
    assert set(result.json_object.keys()) == {"work_experience"}
    assert isinstance(result.json_object["work_experience"], list)
