"""
tests/test_education_extractor.py

Tests for services/education_extractor.py (Milestone 18.2).

PromptManager and SchemaValidator run for real (cheap, deterministic,
local operations — the former against a temporary prompt file, the
latter against the project's real
schemas/modules/education_schema.json), so these tests exercise the
real prompt-loading and validation paths. Only OllamaClient is mocked
in the unit tests, since it depends on a running local model server.

The scenario-shaped tests (multiple degrees, CGPA, percentages,
tables, certifications, pursuing, missing dates, empty education) each
mock a realistic model response representing that scenario already
handled correctly, and confirm it parses and validates against the
module schema — mirroring tests/test_work_experience_extractor.py's
own convention, since prompt-following behavior itself (whether the
real model actually handles each scenario correctly) can only be
checked against a real Ollama server, covered by the integration test
at the bottom and by the real-resume manual testing in
docs/EDUCATION_EXTRACTOR_RESULTS.md.

One integration test at the bottom talks to a real local Ollama server,
the real qwen3:1.7b text model, and the project's real
prompts/modules/education.md prompt. It skips itself (rather than
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
from services.education_extractor import EducationExtractor, EducationResult
from services.ollama_client import OllamaClient
from services.prompt_manager import PromptManager
from validators.schema_validator import SchemaValidationError

INTEGRATION_TEXT_MODEL = "qwen3:1.7b"


def _entry(**overrides):
    """A minimal, otherwise-valid education entry with all nine schema
    fields, overridden as needed per test."""
    base = {
        "institution": "Acme University",
        "degree": "B.Tech",
        "specialization": "Computer Science",
        "location": "Bengaluru, India",
        "start_date": "2018",
        "end_date": "2022",
        "currently_studying": False,
        "grade": None,
        "description": None,
    }
    base.update(overrides)
    return base


VALID_ENTRY = _entry()
VALID_JSON_OBJECT = {"education": [VALID_ENTRY]}


@pytest.fixture
def prompts_dir(tmp_path):
    """A temporary prompts/ tree with one module prompt, so tests never
    touch the project's real prompts/ directory."""
    modules_dir = tmp_path / "prompts" / "modules"
    modules_dir.mkdir(parents=True)
    (modules_dir / "education.md").write_text(
        "Extract only the candidate's education as JSON.", encoding="utf-8"
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
    return EducationExtractor(
        ollama_client=mock_ollama_client,
        prompt_manager=PromptManager(prompts_dir=prompts_dir),
    )


def _respond(mock_ollama_client, json_object):
    mock_ollama_client.generate_text.return_value = SimpleNamespace(
        response=json.dumps(json_object)
    )


class TestExtract:
    def test_valid_markdown_returns_expected_result(self, tmp_path, extractor):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text(
            "# Ada Lovelace\n\n## Education\n\nB.Tech, Acme University, 2018 - 2022",
            encoding="utf-8",
        )

        result = extractor.extract(markdown_path)

        assert isinstance(result, BaseModuleResult)
        assert isinstance(result, EducationResult)
        assert result.json_object == VALID_JSON_OBJECT
        assert result.processing_time >= 0

    def test_prompt_is_sent_with_markdown_appended(
        self, tmp_path, extractor, mock_ollama_client, prompts_dir
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Ada Lovelace\n\n## Education", encoding="utf-8")

        extractor.extract(markdown_path)

        prompt_text = (prompts_dir / "modules" / "education.md").read_text()
        call_kwargs = mock_ollama_client.generate_text.call_args.kwargs
        assert call_kwargs["model"] == "qwen3:1.7b"
        assert call_kwargs["prompt"] == f"{prompt_text}\n\n# Ada Lovelace\n\n## Education"

    def test_invalid_json_raises_invalid_json_error(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        _respond(mock_ollama_client, None)
        mock_ollama_client.generate_text.return_value = SimpleNamespace(
            response="Sure, here is the JSON: {not valid json}"
        )

        with pytest.raises(InvalidJSONError):
            extractor.extract(markdown_path)

    def test_missing_markdown_raises_markdown_not_found(self, tmp_path, extractor):
        missing_path = tmp_path / "does_not_exist.md"

        with pytest.raises(MarkdownNotFoundError):
            extractor.extract(missing_path)

    def test_result_is_not_saved_to_disk(self, tmp_path, extractor):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text(
            "# Ada Lovelace\n\n## Education\n\nB.Tech, Acme University",
            encoding="utf-8",
        )

        result = extractor.extract(markdown_path)

        assert not hasattr(result, "json_path")


class TestSchemaValidation:
    """Structural rejection cases, mirroring
    test_work_experience_extractor.py's own coverage."""

    def test_schema_failure_on_missing_education_key(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        _respond(mock_ollama_client, {})

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_education_not_an_array(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        _respond(mock_ollama_client, {"education": "Acme University"})

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_entry_missing_required_field(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        incomplete_entry = dict(VALID_ENTRY)
        del incomplete_entry["grade"]
        _respond(mock_ollama_client, {"education": [incomplete_entry]})

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_entry_extra_field(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        entry_with_extra = dict(VALID_ENTRY)
        entry_with_extra["gpa_scale"] = "4.0"
        _respond(mock_ollama_client, {"education": [entry_with_extra]})

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_currently_studying_not_boolean(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        bad_entry = dict(VALID_ENTRY)
        bad_entry["currently_studying"] = "yes"
        _respond(mock_ollama_client, {"education": [bad_entry]})

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_institution_as_array(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        bad_entry = dict(VALID_ENTRY)
        bad_entry["institution"] = ["Acme University"]
        _respond(mock_ollama_client, {"education": [bad_entry]})

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)

    def test_schema_failure_on_extra_field_at_root(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        _respond(mock_ollama_client, {"education": [], "skills": ["Python"]})

        with pytest.raises(SchemaValidationError):
            extractor.extract(markdown_path)


class TestRealDatasetScenarios:
    """Each test mocks a realistic model response for one real
    extraction scenario docs/EDUCATION_DESIGN.md identified from this
    project's own dataset, and confirms the result parses and
    validates correctly. These do not test whether the real model
    actually produces this output (that's the integration test and
    docs/EDUCATION_EXTRACTOR_RESULTS.md's manual real-resume testing)
    — only that this shape is accepted end to end."""

    def test_multiple_degrees(self, tmp_path, extractor, mock_ollama_client):
        """Mirrors Dattatray S. Phad: three entries, newest-first,
        oldest schooling still included."""
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume\n\n## Education", encoding="utf-8")
        entries = [
            _entry(
                institution="Ismail Yusuf College, Mumbai",
                degree="Bachelor of Commerce (B.Com)",
                specialization=None,
                location=None,
                start_date="2013",
                end_date="2017",
                grade=None,
            ),
            _entry(
                institution="Murarrao Rane Jr. College, Mumbai",
                degree="Higher Secondary Certificate (HSC)",
                specialization=None,
                location=None,
                start_date="2011",
                end_date="2013",
                grade=None,
            ),
            _entry(
                institution="Nandadeep Vidyalaya, Mumbai",
                degree="Secondary School Certificate (SSC)",
                specialization=None,
                location=None,
                start_date="2007",
                end_date="2010",
                grade=None,
            ),
        ]
        _respond(mock_ollama_client, {"education": entries})

        result = extractor.extract(markdown_path)

        assert len(result.json_object["education"]) == 3
        assert [e["degree"] for e in result.json_object["education"]] == [
            "Bachelor of Commerce (B.Com)",
            "Higher Secondary Certificate (HSC)",
            "Secondary School Certificate (SSC)",
        ]

    def test_cgpa_grade_is_preserved_verbatim(self, tmp_path, extractor, mock_ollama_client):
        """Mirrors Anushka_ (1): CGPA on a 0-10 scale, kept as-is, never
        converted to a percentage."""
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume\n\n## Education", encoding="utf-8")
        entry = _entry(
            institution="Allenhouse Institute of Technology",
            degree="BTech(CSE)",
            specialization=None,
            location="Kanpur, UP",
            start_date=None,
            end_date=None,
            grade="9.0 / 10.00",
        )
        _respond(mock_ollama_client, {"education": [entry]})

        result = extractor.extract(markdown_path)

        assert result.json_object["education"][0]["grade"] == "9.0 / 10.00"

    def test_percentage_grade_is_preserved_verbatim(
        self, tmp_path, extractor, mock_ollama_client
    ):
        """Mirrors Krishankant Soni's table: a percentage grade with the
        '%' sign kept exactly as written."""
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume\n\n## Education", encoding="utf-8")
        entry = _entry(
            institution="Central Board of Secondary Education",
            degree="Class 12",
            specialization=None,
            location=None,
            start_date=None,
            end_date="Mar 2021",
            grade="96.8%",
            description="1st in School",
        )
        _respond(mock_ollama_client, {"education": [entry]})

        result = extractor.extract(markdown_path)

        assert result.json_object["education"][0]["grade"] == "96.8%"
        assert result.json_object["education"][0]["description"] == "1st in School"

    def test_mixed_grade_scales_in_the_same_resume(
        self, tmp_path, extractor, mock_ollama_client
    ):
        """Mirrors Anushka_ (1): CGPA for one entry, percentage for
        another entry in the SAME resume — the extractor must not
        assume one grade format applies resume-wide."""
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume\n\n## Education", encoding="utf-8")
        entries = [
            _entry(institution="Allenhouse Institute of Technology", grade="9.0 / 10.00"),
            _entry(
                institution="Dr. Virendra Swarup Memorial Public School",
                degree="12th ISC",
                grade="83.00 / 100.00",
            ),
        ]
        _respond(mock_ollama_client, {"education": entries})

        result = extractor.extract(markdown_path)

        grades = [e["grade"] for e in result.json_object["education"]]
        assert grades == ["9.0 / 10.00", "83.00 / 100.00"]

    def test_table_sourced_entries_with_a_remarks_column(
        self, tmp_path, extractor, mock_ollama_client
    ):
        """Mirrors Krishankant Soni's 5-column table: Examination | Year
        | Institution | Performance (%) | Remarks — each row becomes
        one entry, with the Remarks column mapped to description."""
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume\n\n## Professional & Educational Qualifications", encoding="utf-8")
        entries = [
            _entry(
                institution="Institute of Chartered Accountants of India",
                degree="CA Final",
                specialization=None,
                location=None,
                start_date=None,
                end_date="Jan 2026",
                grade="51.67%",
                description="First Attempt",
            ),
            _entry(
                institution="University of Rajasthan",
                degree="B.Com",
                specialization=None,
                location=None,
                start_date=None,
                end_date="Mar 2024",
                grade=None,
                description=None,
            ),
        ]
        _respond(mock_ollama_client, {"education": entries})

        result = extractor.extract(markdown_path)

        assert len(result.json_object["education"]) == 2
        assert result.json_object["education"][0]["description"] == "First Attempt"
        assert result.json_object["education"][1]["grade"] is None

    def test_certification_is_not_present_alongside_a_degree(
        self, tmp_path, extractor, mock_ollama_client
    ):
        """Mirrors Rima_Karmakar_CV's "Education & Certifications"
        heading: only the real degree should appear; the ISTQB
        certification lines must not become an education entry."""
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text(
            "# Resume\n\n# Education & Certifications\n\n"
            "- M.Sc. in Information Technology\n"
            "- ISTQB Foundation Level & ISTQB Agile Extension\n",
            encoding="utf-8",
        )
        entry = _entry(
            institution=None,
            degree="M.Sc. in Information Technology",
            specialization="Information Technology",
            location=None,
            start_date=None,
            end_date=None,
            grade=None,
        )
        _respond(mock_ollama_client, {"education": [entry]})

        result = extractor.extract(markdown_path)

        assert len(result.json_object["education"]) == 1
        degrees = [e["degree"] for e in result.json_object["education"]]
        assert not any("istqb" in (d or "").lower() for d in degrees)

    def test_pursuing_sentinel_maps_to_currently_studying(
        self, tmp_path, extractor, mock_ollama_client
    ):
        """Mirrors AMAN_CV_BCA: the literal word "Pursuing" in place of
        a date must become end_date=null, currently_studying=true —
        never a literal "Pursuing" string in end_date."""
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume\n\n## Education", encoding="utf-8")
        entry = _entry(
            institution="IGNOU, Delhi",
            degree="Bachelor of Computer Applications (BCA)",
            specialization=None,
            location=None,
            start_date=None,
            end_date=None,
            currently_studying=True,
            grade=None,
        )
        _respond(mock_ollama_client, {"education": [entry]})

        result = extractor.extract(markdown_path)

        assert result.json_object["education"][0]["end_date"] is None
        assert result.json_object["education"][0]["currently_studying"] is True

    def test_missing_start_date_with_only_a_passing_year(
        self, tmp_path, extractor, mock_ollama_client
    ):
        """Mirrors Arjun Sharma's table ("YR. OF PASSING" only, no start
        year column at all) and PRAGNESH S. SHAH (no dates at all) —
        start_date legitimately null is the common case, not an edge
        case, and must not raise or be guessed at."""
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume\n\n## Educational Qualifications", encoding="utf-8")
        entry = _entry(
            institution="Anna University Chennai",
            degree="B.E (E.C.E)",
            specialization=None,
            location=None,
            start_date=None,
            end_date="2018",
            grade="70.54%",
        )
        _respond(mock_ollama_client, {"education": [entry]})

        result = extractor.extract(markdown_path)

        assert result.json_object["education"][0]["start_date"] is None
        assert result.json_object["education"][0]["end_date"] == "2018"

    def test_no_dates_at_all_for_any_entry(self, tmp_path, extractor, mock_ollama_client):
        """Mirrors PRAGNESH S. SHAH: a real, entirely date-free
        education section (BCA/Diploma/HSC/SSC bullets, no year
        anywhere) — both dates null must be a valid, unexceptional
        entry."""
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume\n\n## Educational Qualification", encoding="utf-8")
        entry = _entry(
            institution="Yashwantrao Chavan Maharashtra Open University",
            degree="BCA",
            specialization=None,
            location=None,
            start_date=None,
            end_date=None,
            currently_studying=False,
            grade=None,
        )
        _respond(mock_ollama_client, {"education": [entry]})

        result = extractor.extract(markdown_path)

        assert result.json_object["education"][0]["start_date"] is None
        assert result.json_object["education"][0]["end_date"] is None
        assert result.json_object["education"][0]["currently_studying"] is False

    def test_empty_education_returns_empty_array(
        self, tmp_path, extractor, mock_ollama_client
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Ada Lovelace\n\nNo education listed.", encoding="utf-8")
        _respond(mock_ollama_client, {"education": []})

        result = extractor.extract(markdown_path)

        assert result.json_object == {"education": []}


class TestPreprocessJson:
    """Milestone 23.22 (docs/PARSER_RELIABILITY_HARDENING.md): preprocess_json()
    coerces a null `currently_studying` to `False` before schema
    validation — see services/education_extractor.py's own module/class
    docstrings for the evidence (Rima_Karmakar_CV's real, live-reproduced,
    100%-repeatable failure). Exercised directly, in isolation from
    extract()/Ollama/schema validation, the same way
    tests/test_work_experience_extractor.py's equivalent class does; the
    last test in this class exercises the full extract() pipeline
    instead, to prove the real Rima_Karmakar_CV failure is actually
    prevented end to end, not just at the unit level.
    """

    def test_null_currently_studying_is_coerced_to_false(self, extractor):
        entry = _entry(currently_studying=None)

        result = extractor.preprocess_json({"education": [entry]})

        assert result["education"][0]["currently_studying"] is False

    def test_real_rima_karmakar_case_all_fields_null_except_degree(self, extractor):
        """The exact shape reproduced live against the real model and
        the real Rima_Karmakar_CV.md — every field null except degree."""
        entry = {
            "institution": None,
            "degree": "M.Sc. in Information Technology",
            "specialization": None,
            "location": None,
            "start_date": None,
            "end_date": None,
            "currently_studying": None,
            "grade": None,
            "description": None,
        }

        result = extractor.preprocess_json({"education": [entry]})

        assert result["education"][0]["currently_studying"] is False
        # Every other field is left exactly as the model produced it —
        # this fix touches currently_studying only.
        assert result["education"][0]["degree"] == "M.Sc. in Information Technology"
        assert result["education"][0]["institution"] is None

    def test_already_false_is_unaffected(self, extractor):
        entry = _entry(currently_studying=False)
        original = {"education": [entry]}

        result = extractor.preprocess_json(original)

        assert result is original

    def test_already_true_is_unaffected(self, extractor):
        """Real, confirmed case (AMAN_CV_BCA's BCA degree, this
        project's own ground truth): currently_studying=True must never
        be flipped by this fix — only null is ever touched."""
        entry = _entry(currently_studying=True)
        original = {"education": [entry]}

        result = extractor.preprocess_json(original)

        assert result is original
        assert result["education"][0]["currently_studying"] is True

    def test_multiple_entries_each_coerced_independently(self, extractor):
        entries = [
            _entry(institution="A University", currently_studying=None),
            _entry(institution="B University", currently_studying=True),
            _entry(institution="C University", currently_studying=None),
        ]

        result = extractor.preprocess_json({"education": entries})

        assert result["education"][0]["currently_studying"] is False
        assert result["education"][1]["currently_studying"] is True
        assert result["education"][2]["currently_studying"] is False

    def test_missing_currently_studying_key_is_unaffected(self, extractor):
        entry = _entry()
        del entry["currently_studying"]
        original = {"education": [entry]}

        result = extractor.preprocess_json(original)

        assert result is original

    def test_empty_education_array_is_unaffected(self, extractor):
        original = {"education": []}

        result = extractor.preprocess_json(original)

        assert result is original

    def test_non_dict_json_object_is_unaffected(self, extractor):
        assert extractor.preprocess_json(["not", "a", "dict"]) == ["not", "a", "dict"]
        assert extractor.preprocess_json(None) is None

    def test_education_not_a_list_is_unaffected(self, extractor):
        original = {"education": "not a list"}

        result = extractor.preprocess_json(original)

        assert result is original

    def test_entry_not_a_dict_is_unaffected(self, extractor):
        original = {"education": ["not a dict"]}

        result = extractor.preprocess_json(original)

        assert result is original

    def test_all_other_fields_are_byte_for_byte_unchanged(self, extractor):
        entry = _entry(currently_studying=None, grade="8.5 CGPA", description="Dean's list")

        result = extractor.preprocess_json({"education": [entry]})

        changed_entry = result["education"][0]
        for key in (
            "institution",
            "degree",
            "specialization",
            "location",
            "start_date",
            "end_date",
            "grade",
            "description",
        ):
            assert changed_entry[key] == entry[key]

    def test_full_extract_pipeline_recovers_from_a_would_be_boolean_failure(
        self, tmp_path, extractor, mock_ollama_client
    ):
        """Milestone 23.22's actual goal, proven end to end: the exact
        Rima_Karmakar_CV response shape, run through the real extract()
        pipeline (parse -> preprocess_json() -> SchemaValidator.validate()),
        must now succeed instead of raising SchemaValidationError."""
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")
        null_studying_entry = {
            "institution": None,
            "degree": "M.Sc. in Information Technology",
            "specialization": None,
            "location": None,
            "start_date": None,
            "end_date": None,
            "currently_studying": None,
            "grade": None,
            "description": None,
        }
        _respond(mock_ollama_client, {"education": [null_studying_entry]})

        result = extractor.extract(markdown_path)

        assert result.json_object["education"][0]["currently_studying"] is False


@pytest.mark.integration
def test_real_ollama_extract_integration(tmp_path):
    """Optional integration test against a real local Ollama server, the
    real qwen3:1.7b model, and the project's real
    prompts/modules/education.md prompt. Skips itself if either the
    server or the model isn't available."""
    ollama_client = OllamaClient()

    if not ollama_client.is_running():
        pytest.skip("Ollama is not running locally; skipping integration test.")
    if not ollama_client.model_exists(INTEGRATION_TEXT_MODEL):
        pytest.skip(f"Model '{INTEGRATION_TEXT_MODEL}' is not installed; skipping.")

    markdown_path = tmp_path / "resume.md"
    markdown_path.write_text(
        "# Ada Lovelace\n\n## Education\n\n"
        "### Bachelor of Technology\n"
        "**Acme Institute of Technology** — Bengaluru, India\n"
        "2018 - 2022\n"
        "CGPA: 8.5 / 10.00\n",
        encoding="utf-8",
    )

    extractor = EducationExtractor(ollama_client=ollama_client)

    result = extractor.extract(markdown_path, prompt_id="modules/education")

    assert isinstance(result, BaseModuleResult)
    assert isinstance(result.json_object, dict)
    assert set(result.json_object.keys()) == {"education"}
    assert isinstance(result.json_object["education"], list)
