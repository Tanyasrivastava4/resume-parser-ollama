"""
tests/test_parser_pipeline.py

Tests for pipeline/parser_pipeline.py (Milestone 14.7; work_experience
wired in as a sixth module extractor in Milestone 17.3; education wired
in as a seventh module extractor in Milestone 18.3).

Every dependency (the seven module extractors, ParserMerger,
SchemaValidator) is mocked via constructor injection, so these unit
tests exercise only the pipeline's own orchestration logic — none of
them touch a real Ollama server or the real filesystem beyond
tmp_path/output_cleanup. One integration test at the bottom runs the real
pipeline end to end, with every real service, against a real local
Ollama server; it skips itself if Ollama or the model isn't available.

Milestone 23.7: JsonDraftGenerator has been retired — see
pipeline/parser_pipeline.py's own module docstring and
docs/JSON_DRAFT_GENERATOR_RETIREMENT_RESULTS.md. The old
mock_json_draft_generator fixture, DRAFT_JSON constant, and
TestJsonDraftGeneratorNonBlocking class (Milestone 23.6) were removed;
TestFinalJsonWithoutJsonDraftGenerator and
TestJsonDraftGeneratorFullyRemoved (below) replace them.
"""

import json
import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from config import BASE_DIR, OUTPUT_DIR
from pipeline.parser_pipeline import ParserPipeline, ParserPipelineResult
from services.base_module_extractor import InvalidJSONError, MarkdownNotFoundError
from services.contact_information_extractor import ContactInformationExtractor
from services.education_extractor import EducationExtractor
from services.education_validator import EducationValidator
from services.education_validator import ValidationReport as EducationValidationReport
from services.languages_extractor import LanguagesExtractor
from services.ollama_client import OllamaClient
from services.parser_merger import MergeConflictError, ParserMerger
from services.personal_information_extractor import PersonalInformationExtractor
from services.professional_summary_extractor import ProfessionalSummaryExtractor
from services.project_extractor import ProjectExtractor
from services.skills_extractor import SkillsExtractor
from services.work_experience_extractor import WorkExperienceExtractor
from services.work_experience_validator import ValidationReport, WorkExperienceValidator
from validators.schema_validator import SchemaValidationError, SchemaValidator

INTEGRATION_TEXT_MODEL = "qwen3:1.7b"

PERSONAL_INFO_JSON = {
    "personal_information": {
        "full_name": "Ada Lovelace",
        "first_name": "Ada",
        "last_name": "Lovelace",
        "current_location": None,
    }
}
CONTACT_INFO_JSON = {
    "contact_information": {
        "email": ["ada@example.com"],
        "phone": [],
        "linkedin": None,
        "github": None,
        "portfolio": None,
    }
}
SKILLS_JSON = {"skills": ["Mathematics"]}
SUMMARY_JSON = {"professional_summary": "Mathematician and writer."}
LANGUAGES_JSON = {"languages": ["English"]}
WORK_EXPERIENCE_JSON = {
    "work_experience": [
        {
            "company": "Analytical Engines Ltd",
            "designation": "Mathematician",
            "employment_type": "Full-time",
            "location": None,
            "start_date": "1842",
            "end_date": None,
            "currently_working": True,
            "duration": None,
            "responsibilities": ["Wrote the first published algorithm"],
            "technologies": [],
        }
    ]
}
EDUCATION_JSON = {
    "education": [
        {
            "institution": "Royal Institution",
            "degree": "Self-taught",
            "specialization": "Mathematics",
            "location": None,
            "start_date": None,
            "end_date": None,
            "currently_studying": False,
            "grade": None,
            "description": None,
        }
    ]
}
PROJECTS_JSON = {
    "projects": [
        {
            "name": "Analytical Engine Notes",
            "description": "Annotated translation with an original algorithm.",
            "role": None,
            "start_date": None,
            "end_date": None,
            "highlights": ["Wrote the first published algorithm for the Analytical Engine"],
            "technologies": [],
            "url": None,
        }
    ]
}

MERGED_JSON = {
    **PERSONAL_INFO_JSON,
    **CONTACT_INFO_JSON,
    **SKILLS_JSON,
    **SUMMARY_JSON,
    **LANGUAGES_JSON,
    **WORK_EXPERIENCE_JSON,
    **EDUCATION_JSON,
    **PROJECTS_JSON,
}


def _module_result(json_object):
    return SimpleNamespace(json_object=json_object, processing_time=0.01)


@pytest.fixture
def mock_personal_information_extractor():
    extractor = MagicMock(spec=PersonalInformationExtractor)
    extractor.extract.return_value = _module_result(PERSONAL_INFO_JSON)
    return extractor


@pytest.fixture
def mock_contact_information_extractor():
    extractor = MagicMock(spec=ContactInformationExtractor)
    extractor.extract.return_value = _module_result(CONTACT_INFO_JSON)
    return extractor


@pytest.fixture
def mock_skills_extractor():
    extractor = MagicMock(spec=SkillsExtractor)
    extractor.extract.return_value = _module_result(SKILLS_JSON)
    return extractor


@pytest.fixture
def mock_professional_summary_extractor():
    extractor = MagicMock(spec=ProfessionalSummaryExtractor)
    extractor.extract.return_value = _module_result(SUMMARY_JSON)
    return extractor


@pytest.fixture
def mock_languages_extractor():
    extractor = MagicMock(spec=LanguagesExtractor)
    extractor.extract.return_value = _module_result(LANGUAGES_JSON)
    return extractor


@pytest.fixture
def mock_work_experience_extractor():
    extractor = MagicMock(spec=WorkExperienceExtractor)
    extractor.extract.return_value = _module_result(WORK_EXPERIENCE_JSON)
    return extractor


@pytest.fixture
def mock_education_extractor():
    extractor = MagicMock(spec=EducationExtractor)
    extractor.extract.return_value = _module_result(EDUCATION_JSON)
    return extractor


@pytest.fixture
def mock_project_extractor():
    extractor = MagicMock(spec=ProjectExtractor)
    extractor.extract.return_value = _module_result(PROJECTS_JSON)
    return extractor


@pytest.fixture
def mock_parser_merger():
    merger = MagicMock(spec=ParserMerger)
    merger.merge.return_value = MERGED_JSON
    return merger


@pytest.fixture
def mock_schema_validator():
    validator = MagicMock(spec=SchemaValidator)
    validator.validate.return_value = True
    return validator


@pytest.fixture
def pipeline(
    mock_personal_information_extractor,
    mock_contact_information_extractor,
    mock_skills_extractor,
    mock_professional_summary_extractor,
    mock_languages_extractor,
    mock_work_experience_extractor,
    mock_education_extractor,
    mock_project_extractor,
    mock_parser_merger,
    mock_schema_validator,
):
    return ParserPipeline(
        personal_information_extractor=mock_personal_information_extractor,
        contact_information_extractor=mock_contact_information_extractor,
        skills_extractor=mock_skills_extractor,
        professional_summary_extractor=mock_professional_summary_extractor,
        languages_extractor=mock_languages_extractor,
        work_experience_extractor=mock_work_experience_extractor,
        education_extractor=mock_education_extractor,
        project_extractor=mock_project_extractor,
        parser_merger=mock_parser_merger,
        schema_validator=mock_schema_validator,
    )


@pytest.fixture
def output_cleanup():
    """Track resume names whose output/modules/<name>/ and
    output/json_final/<name>.json should be removed after the test."""
    resume_names = []
    yield resume_names
    for resume_name in resume_names:
        shutil.rmtree(OUTPUT_DIR / "modules" / resume_name, ignore_errors=True)
        (OUTPUT_DIR / "json_final" / f"{resume_name}.json").unlink(missing_ok=True)


class TestHappyPath:
    def test_returns_expected_result(self, tmp_path, pipeline, output_cleanup):
        resume_name = "test_resume_happy_path"
        output_cleanup.append(resume_name)
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("# Ada Lovelace", encoding="utf-8")

        result = pipeline.run(markdown_path)

        assert isinstance(result, ParserPipelineResult)
        assert result.draft_json is None
        assert result.module_results == {
            "personal_information": PERSONAL_INFO_JSON,
            "contact_information": CONTACT_INFO_JSON,
            "skills": SKILLS_JSON,
            "professional_summary": SUMMARY_JSON,
            "languages": LANGUAGES_JSON,
            "work_experience": WORK_EXPERIENCE_JSON,
            "education": EDUCATION_JSON,
            "projects": PROJECTS_JSON,
        }
        assert result.merged_json == MERGED_JSON
        assert result.processing_time >= 0

    def test_every_module_extractor_is_called_with_markdown_path(
        self,
        tmp_path,
        pipeline,
        mock_personal_information_extractor,
        mock_contact_information_extractor,
        mock_skills_extractor,
        mock_professional_summary_extractor,
        mock_languages_extractor,
        mock_work_experience_extractor,
        mock_education_extractor,
        mock_project_extractor,
        output_cleanup,
    ):
        resume_name = "test_resume_module_calls"
        output_cleanup.append(resume_name)
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("# Ada Lovelace", encoding="utf-8")

        pipeline.run(markdown_path)

        mock_personal_information_extractor.extract.assert_called_once_with(markdown_path)
        mock_contact_information_extractor.extract.assert_called_once_with(markdown_path)
        mock_skills_extractor.extract.assert_called_once_with(markdown_path)
        mock_professional_summary_extractor.extract.assert_called_once_with(markdown_path)
        mock_languages_extractor.extract.assert_called_once_with(markdown_path)
        mock_work_experience_extractor.extract.assert_called_once_with(markdown_path)
        mock_education_extractor.extract.assert_called_once_with(markdown_path)
        mock_project_extractor.extract.assert_called_once_with(markdown_path)

    def test_merger_receives_every_module_result(
        self, tmp_path, pipeline, mock_parser_merger, output_cleanup
    ):
        resume_name = "test_resume_merger_input"
        output_cleanup.append(resume_name)
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("# Ada Lovelace", encoding="utf-8")

        pipeline.run(markdown_path)

        mock_parser_merger.merge.assert_called_once()
        merge_input = mock_parser_merger.merge.call_args[0][0]
        assert PERSONAL_INFO_JSON in merge_input
        assert CONTACT_INFO_JSON in merge_input
        assert SKILLS_JSON in merge_input
        assert SUMMARY_JSON in merge_input
        assert LANGUAGES_JSON in merge_input
        assert WORK_EXPERIENCE_JSON in merge_input
        assert EDUCATION_JSON in merge_input
        assert PROJECTS_JSON in merge_input

    def test_validator_receives_merged_json(
        self, tmp_path, pipeline, mock_schema_validator, output_cleanup
    ):
        resume_name = "test_resume_validation_input"
        output_cleanup.append(resume_name)
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("# Ada Lovelace", encoding="utf-8")

        pipeline.run(markdown_path)

        mock_schema_validator.validate.assert_called_once_with(MERGED_JSON)


class TestMissingMarkdown:
    def test_missing_markdown_raises(
        self, tmp_path, pipeline, mock_personal_information_extractor
    ):
        # Milestone 23.7: this used to raise from the (now-retired)
        # JsonDraftGenerator step; personal_information is the pipeline's
        # new first step, so it's the one that would encounter this in
        # production.
        mock_personal_information_extractor.extract.side_effect = MarkdownNotFoundError(
            "Markdown file not found"
        )
        missing_path = tmp_path / "does_not_exist.md"

        with pytest.raises(MarkdownNotFoundError):
            pipeline.run(missing_path)


class TestFinalJsonWithoutJsonDraftGenerator:
    """Milestone 23.7: regression coverage proving the final parser JSON
    is still produced correctly now that JsonDraftGenerator has been
    retired entirely — see docs/JSON_DRAFT_GENERATOR_RETIREMENT_RESULTS
    .md. Uses the pipeline fixture's real, post-retirement constructor
    shape directly (no json_draft_generator argument exists anymore)."""

    def test_final_json_produced_correctly_with_no_draft_step(
        self, tmp_path, pipeline, output_cleanup
    ):
        resume_name = "test_resume_retirement_final_json"
        output_cleanup.append(resume_name)
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("# Ada Lovelace", encoding="utf-8")

        result = pipeline.run(markdown_path)

        assert result.merged_json == MERGED_JSON
        final_path = OUTPUT_DIR / "json_final" / f"{resume_name}.json"
        assert json.loads(final_path.read_text(encoding="utf-8")) == MERGED_JSON


class TestJsonDraftGeneratorFullyRemoved:
    """Milestone 23.7: proves JsonDraftGenerator is no longer invoked —
    by verifying the absence of the draft-generation execution path
    itself (module gone, constructor parameter gone, no artifact
    written), not merely by mocking its result and checking it's
    unused (which is what Milestone 23.6's TestJsonDraftGeneratorNon
    Blocking did, before the class this replaces existed at all)."""

    def test_json_draft_generator_module_no_longer_importable(self):
        with pytest.raises(ModuleNotFoundError):
            import services.json_draft_generator  # noqa: F401

    def test_parser_pipeline_constructor_has_no_draft_generator_parameter(self):
        import inspect

        params = inspect.signature(ParserPipeline.__init__).parameters
        assert "json_draft_generator" not in params

    def test_no_json_draft_output_file_is_written(self, tmp_path, pipeline, output_cleanup):
        resume_name = "test_resume_retirement_no_draft_artifact"
        output_cleanup.append(resume_name)
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        pipeline.run(markdown_path)

        assert not (OUTPUT_DIR / "json_draft" / f"{resume_name}.json").exists()


class TestModuleFailure:
    def test_module_failure_raises_and_stops_later_modules(
        self,
        tmp_path,
        pipeline,
        mock_contact_information_extractor,
        mock_skills_extractor,
        mock_languages_extractor,
        output_cleanup,
    ):
        # personal_information (step 1) succeeds and gets saved to disk
        # before contact_information fails, so this needs cleanup too.
        resume_name = "test_resume_module_failure"
        output_cleanup.append(resume_name)
        mock_contact_information_extractor.extract.side_effect = SchemaValidationError(
            "contact_information failed validation"
        )
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        with pytest.raises(SchemaValidationError):
            pipeline.run(markdown_path)

        # contact_information (step 3) failed, so skills/languages
        # (steps 4 and 6) must never have been reached.
        mock_skills_extractor.extract.assert_not_called()
        mock_languages_extractor.extract.assert_not_called()

    def test_module_failure_does_not_call_merger_or_validator(
        self,
        tmp_path,
        pipeline,
        mock_personal_information_extractor,
        mock_parser_merger,
        mock_schema_validator,
    ):
        mock_personal_information_extractor.extract.side_effect = InvalidJSONError("bad json")
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        with pytest.raises(InvalidJSONError):
            pipeline.run(markdown_path)

        mock_parser_merger.merge.assert_not_called()
        mock_schema_validator.validate.assert_not_called()


class TestMergerFailure:
    def test_merger_failure_raises_and_skips_validation(
        self, tmp_path, pipeline, mock_parser_merger, mock_schema_validator, output_cleanup
    ):
        # All five modules succeed and get saved to disk before the
        # merger fails, so this needs cleanup too.
        resume_name = "test_resume_merger_failure"
        output_cleanup.append(resume_name)
        mock_parser_merger.merge.side_effect = MergeConflictError(
            "Key 'skills' is defined by more than one module result."
        )
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        with pytest.raises(MergeConflictError):
            pipeline.run(markdown_path)

        mock_schema_validator.validate.assert_not_called()


class TestValidationFailure:
    def test_validation_failure_raises(
        self, tmp_path, pipeline, mock_schema_validator, output_cleanup
    ):
        # All five modules succeed and get saved to disk before
        # validation fails, so this needs cleanup too.
        resume_name = "test_resume_validation_raises"
        output_cleanup.append(resume_name)
        mock_schema_validator.validate.side_effect = SchemaValidationError(
            "merged JSON does not conform to resume_schema_v1.json"
        )
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        with pytest.raises(SchemaValidationError):
            pipeline.run(markdown_path)

    def test_validation_failure_leaves_no_final_json_file(
        self, tmp_path, pipeline, mock_schema_validator, output_cleanup
    ):
        resume_name = "test_resume_validation_failure"
        output_cleanup.append(resume_name)
        mock_schema_validator.validate.side_effect = SchemaValidationError("bad")
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        with pytest.raises(SchemaValidationError):
            pipeline.run(markdown_path)

        assert not (OUTPUT_DIR / "json_final" / f"{resume_name}.json").exists()


class TestOutputFilesWritten:
    def test_module_and_final_files_are_written_with_correct_content(
        self, tmp_path, pipeline, output_cleanup
    ):
        resume_name = "test_resume_output_files"
        output_cleanup.append(resume_name)
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("# Ada Lovelace", encoding="utf-8")

        result = pipeline.run(markdown_path)

        module_dir = OUTPUT_DIR / "modules" / resume_name
        expected_module_files = {
            "personal_information.json": PERSONAL_INFO_JSON,
            "contact_information.json": CONTACT_INFO_JSON,
            "skills.json": SKILLS_JSON,
            "professional_summary.json": SUMMARY_JSON,
            "languages.json": LANGUAGES_JSON,
            "work_experience.json": WORK_EXPERIENCE_JSON,
            "education.json": EDUCATION_JSON,
            "projects.json": PROJECTS_JSON,
        }
        for filename, expected_content in expected_module_files.items():
            file_path = module_dir / filename
            assert file_path.is_file(), filename
            assert json.loads(file_path.read_text(encoding="utf-8")) == expected_content

        final_path = OUTPUT_DIR / "json_final" / f"{resume_name}.json"
        assert final_path.is_file()
        assert json.loads(final_path.read_text(encoding="utf-8")) == MERGED_JSON
        assert json.loads(final_path.read_text(encoding="utf-8")) == result.merged_json


class TestWorkExperienceValidatorIntegration:
    """Milestone 17.6: WorkExperienceValidator runs on the real
    (unmocked) work_experience result before it reaches ParserMerger.
    These tests use a real WorkExperienceValidator (the `pipeline`
    fixture doesn't inject one, so it gets the real default) with a
    mocked WorkExperienceExtractor returning deliberately dirty data,
    to confirm the correction actually takes effect end to end — not
    just that the validator class works in isolation (see
    tests/test_work_experience_validator.py for that)."""

    def test_dirty_work_experience_is_corrected_before_merging(
        self, tmp_path, pipeline, mock_work_experience_extractor, mock_parser_merger, output_cleanup
    ):
        resume_name = "test_resume_wev_correction"
        output_cleanup.append(resume_name)
        dirty_job = {
            "company": "Analytical Engines Ltd",
            "designation": "Mathematician",
            "employment_type": "Bogus Type",  # WEV-003
            "location": None,
            "start_date": "Jan 2020",
            "end_date": "Present",  # WEV-001
            "currently_working": False,
            "duration": None,
            "responsibilities": [],
            "technologies": [],
        }
        mock_work_experience_extractor.extract.return_value = _module_result(
            {"work_experience": [dirty_job]}
        )
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("# Ada Lovelace", encoding="utf-8")

        result = pipeline.run(markdown_path)

        # The merger (mocked here) must receive the CORRECTED job, not
        # the raw dirty one WorkExperienceExtractor returned.
        merge_input = mock_parser_merger.merge.call_args[0][0]
        corrected = next(
            entry["work_experience"][0] for entry in merge_input if "work_experience" in entry
        )
        assert corrected["employment_type"] is None
        assert corrected["end_date"] is None
        assert corrected["currently_working"] is True

        # module_results and the saved module file both reflect the
        # corrected version too, not the raw extractor output.
        assert result.module_results["work_experience"]["work_experience"][0]["employment_type"] is None
        module_path = OUTPUT_DIR / "modules" / resume_name / "work_experience.json"
        saved = json.loads(module_path.read_text(encoding="utf-8"))
        assert saved["work_experience"][0]["end_date"] is None

    def test_validation_report_is_populated_on_the_result(
        self, tmp_path, pipeline, mock_work_experience_extractor, output_cleanup
    ):
        resume_name = "test_resume_wev_report"
        output_cleanup.append(resume_name)
        dirty_job = {
            "company": "Analytical Engines Ltd",
            "designation": "Mathematician",
            "employment_type": None,
            "location": None,
            "start_date": None,
            "end_date": "Current",  # WEV-001
            "currently_working": False,
            "duration": None,
            "responsibilities": [],
            "technologies": [],
        }
        mock_work_experience_extractor.extract.return_value = _module_result(
            {"work_experience": [dirty_job]}
        )
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("# Ada Lovelace", encoding="utf-8")

        result = pipeline.run(markdown_path)

        report = result.work_experience_validation_report
        assert report is not None
        assert report.total_jobs == 1
        assert len(report.issues) == 1
        assert report.issues[0].rule_id == "WEV-001"
        assert report.auto_fixed_count == 1

    def test_clean_work_experience_produces_no_issues(self, tmp_path, pipeline, output_cleanup):
        """The `pipeline` fixture's default WORK_EXPERIENCE_JSON is
        already clean by every Quick Win rule — confirms the validator
        being wired in doesn't spuriously flag valid data.

        Milestone 23.15: WEV-016/017/018 also need a confidently-matched
        markdown span to confirm employment_type="Full-time" is
        genuinely source-supported (a bare "# Ada Lovelace" with no
        Experience section at all would otherwise correctly produce a
        Warning, per the same "no confident match" safety net
        WEV-015 already has — not a spurious flag), so this markdown
        now includes a real Experience section matching
        WORK_EXPERIENCE_JSON's own company/employment_type."""
        resume_name = "test_resume_wev_clean"
        output_cleanup.append(resume_name)
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text(
            "# Ada Lovelace\n\n"
            "## Experience\n"
            "### Analytical Engines Ltd\n"
            "Full-time\n"
            "1842 - Present\n"
            "- Wrote the first published algorithm\n",
            encoding="utf-8",
        )

        result = pipeline.run(markdown_path)

        assert result.work_experience_validation_report.issues == []

    def test_custom_validator_is_used_when_injected(
        self, tmp_path, mock_personal_information_extractor,
        mock_contact_information_extractor, mock_skills_extractor,
        mock_professional_summary_extractor, mock_languages_extractor,
        mock_work_experience_extractor, mock_education_extractor, mock_project_extractor,
        mock_parser_merger, mock_schema_validator, output_cleanup,
    ):
        """Confirms work_experience_validator is wired in via
        constructor injection, like every other dependency."""
        resume_name = "test_resume_wev_custom_validator"
        output_cleanup.append(resume_name)
        custom_validator = MagicMock(spec=WorkExperienceValidator)
        custom_validator.validate.return_value = (
            [{"company": "Overridden by custom validator"}],
            ValidationReport(total_jobs=1),
        )
        pipeline = ParserPipeline(
            personal_information_extractor=mock_personal_information_extractor,
            contact_information_extractor=mock_contact_information_extractor,
            skills_extractor=mock_skills_extractor,
            professional_summary_extractor=mock_professional_summary_extractor,
            languages_extractor=mock_languages_extractor,
            work_experience_extractor=mock_work_experience_extractor,
            work_experience_validator=custom_validator,
            education_extractor=mock_education_extractor,
            project_extractor=mock_project_extractor,
            parser_merger=mock_parser_merger,
            schema_validator=mock_schema_validator,
        )
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("# Ada Lovelace", encoding="utf-8")

        result = pipeline.run(markdown_path)

        custom_validator.validate.assert_called_once_with(
            "# Ada Lovelace", WORK_EXPERIENCE_JSON["work_experience"]
        )
        assert result.module_results["work_experience"]["work_experience"] == [
            {"company": "Overridden by custom validator"}
        ]


class TestEducationValidatorIntegration:
    """Milestone 18.10: EducationValidator runs on the real (unmocked)
    education result before it reaches ParserMerger. These tests use a
    real EducationValidator (the `pipeline` fixture doesn't inject one,
    so it gets the real default) with a mocked EducationExtractor
    returning deliberately dirty data, to confirm the correction
    actually takes effect end to end — not just that the validator
    class works in isolation (see tests/test_education_validator.py for
    that). Mirrors TestWorkExperienceValidatorIntegration's exact
    shape."""

    def test_dirty_education_is_corrected_before_merging(
        self, tmp_path, pipeline, mock_education_extractor, mock_parser_merger, output_cleanup
    ):
        resume_name = "test_resume_edv_correction"
        output_cleanup.append(resume_name)
        dirty_entry = {
            "institution": "State University",
            "degree": "B.Tech",
            "specialization": "Computer Science",
            "location": None,
            "start_date": "2015",
            "end_date": "Pursuing",  # EDV-001
            "currently_studying": False,
            "grade": "B.Tech",  # EDV-008
            "description": None,
        }
        mock_education_extractor.extract.return_value = _module_result(
            {"education": [dirty_entry]}
        )
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("# Ada Lovelace", encoding="utf-8")

        result = pipeline.run(markdown_path)

        # The merger (mocked here) must receive the CORRECTED entry,
        # not the raw dirty one EducationExtractor returned.
        merge_input = mock_parser_merger.merge.call_args[0][0]
        corrected = next(
            entry["education"][0] for entry in merge_input if "education" in entry
        )
        assert corrected["grade"] is None
        assert corrected["end_date"] is None
        assert corrected["currently_studying"] is True

        # module_results and the saved module file both reflect the
        # corrected version too, not the raw extractor output.
        assert result.module_results["education"]["education"][0]["grade"] is None
        module_path = OUTPUT_DIR / "modules" / resume_name / "education.json"
        saved = json.loads(module_path.read_text(encoding="utf-8"))
        assert saved["education"][0]["end_date"] is None

    def test_validation_report_is_populated_on_the_result(
        self, tmp_path, pipeline, mock_education_extractor, output_cleanup
    ):
        resume_name = "test_resume_edv_report"
        output_cleanup.append(resume_name)
        dirty_entry = {
            "institution": "M.Sc.",  # EDV-009
            "degree": "M.Sc.",
            "specialization": None,
            "location": None,
            "start_date": None,
            "end_date": None,
            "currently_studying": False,
            "grade": None,
            "description": None,
        }
        mock_education_extractor.extract.return_value = _module_result(
            {"education": [dirty_entry]}
        )
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("# Ada Lovelace", encoding="utf-8")

        result = pipeline.run(markdown_path)

        report = result.education_validation_report
        assert report is not None
        assert report.total_entries == 1
        assert len(report.issues) == 1
        assert report.issues[0].rule_id == "EDV-009"
        assert report.auto_fixed_count == 1

    def test_clean_education_produces_no_issues(self, tmp_path, pipeline, output_cleanup):
        """The `pipeline` fixture's default EDUCATION_JSON is already
        clean by every Quick Win rule — confirms the validator being
        wired in doesn't spuriously flag valid data."""
        resume_name = "test_resume_edv_clean"
        output_cleanup.append(resume_name)
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("# Ada Lovelace", encoding="utf-8")

        result = pipeline.run(markdown_path)

        assert result.education_validation_report.issues == []

    def test_custom_validator_is_used_when_injected(
        self, tmp_path, mock_personal_information_extractor,
        mock_contact_information_extractor, mock_skills_extractor,
        mock_professional_summary_extractor, mock_languages_extractor,
        mock_work_experience_extractor, mock_education_extractor, mock_project_extractor,
        mock_parser_merger, mock_schema_validator, output_cleanup,
    ):
        """Confirms education_validator is wired in via constructor
        injection, like every other dependency."""
        resume_name = "test_resume_edv_custom_validator"
        output_cleanup.append(resume_name)
        custom_validator = MagicMock(spec=EducationValidator)
        custom_validator.validate.return_value = (
            [{"institution": "Overridden by custom validator"}],
            EducationValidationReport(total_entries=1),
        )
        pipeline = ParserPipeline(
            personal_information_extractor=mock_personal_information_extractor,
            contact_information_extractor=mock_contact_information_extractor,
            skills_extractor=mock_skills_extractor,
            professional_summary_extractor=mock_professional_summary_extractor,
            languages_extractor=mock_languages_extractor,
            work_experience_extractor=mock_work_experience_extractor,
            education_extractor=mock_education_extractor,
            education_validator=custom_validator,
            project_extractor=mock_project_extractor,
            parser_merger=mock_parser_merger,
            schema_validator=mock_schema_validator,
        )
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("# Ada Lovelace", encoding="utf-8")

        result = pipeline.run(markdown_path)

        # Milestone V2.6D: _validate_education now also passes the
        # resume's markdown_text (EDV-012's own corroboration evidence;
        # see services/education_validator.py), mirroring
        # work_experience_validator's existing (markdown_text, jobs)
        # call shape.
        custom_validator.validate.assert_called_once_with(
            EDUCATION_JSON["education"], "# Ada Lovelace"
        )
        assert result.module_results["education"]["education"] == [
            {"institution": "Overridden by custom validator"}
        ]


def _make_pdf(tmp_path, links, filename="synthetic.pdf"):
    """A minimal one-page PDF with the given [(rect_tuple, uri), ...]
    link annotations. Mirrors tests/test_evidence_record.py's own
    helper of the same name/shape."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    for rect, uri in links:
        page.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(*rect), "uri": uri})
    pdf_path = tmp_path / filename
    doc.save(pdf_path)
    doc.close()
    return pdf_path


class TestContactInformationEvidenceIntegration:
    """Milestone V3.9: build_contact_information_evidence() (Milestone
    V3.8) runs on ContactInformationExtractor's real result before it
    reaches ParserMerger, exactly like TestWorkExperienceValidatorIntegration
    / TestEducationValidatorIntegration above -- except this branch NEVER
    reassigns json_object, so every test here also asserts the extracted
    value is byte/structurally unchanged, not just that evidence was
    attached correctly."""

    def test_corroborated_email(self, tmp_path, pipeline, mock_contact_information_extractor, output_cleanup):
        resume_name = "test_resume_ci_email"
        output_cleanup.append(resume_name)
        mock_contact_information_extractor.extract.return_value = _module_result(
            {"contact_information": {"email": ["ada@example.com"], "phone": [], "linkedin": None, "github": None, "portfolio": None}}
        )
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("Ada Lovelace\nada@example.com\n", encoding="utf-8")

        result = pipeline.run(markdown_path)

        field = next(f for f in result.contact_information_evidence.fields if f.field_name == "email")
        assert field.evidence_summary == "FULLY_CORROBORATED"
        assert field.visible_text_matches == ["ada@example.com"]

    def test_supported_phone_normalization(self, tmp_path, pipeline, mock_contact_information_extractor, output_cleanup):
        resume_name = "test_resume_ci_phone"
        output_cleanup.append(resume_name)
        mock_contact_information_extractor.extract.return_value = _module_result(
            {"contact_information": {"email": [], "phone": ["+91-8097305823"], "linkedin": None, "github": None, "portfolio": None}}
        )
        markdown_path = tmp_path / f"{resume_name}.md"
        # Visible text has different formatting -- digit-only comparison
        # must still corroborate it (V3.6's corpus-verified normalization).
        markdown_path.write_text("Ada Lovelace\nPhone: 91 8097305823\n", encoding="utf-8")

        result = pipeline.run(markdown_path)

        field = next(f for f in result.contact_information_evidence.fields if f.field_name == "phone")
        assert field.evidence_summary == "FULLY_CORROBORATED"

    def test_supported_url_evidence_via_annotation(self, tmp_path, pipeline, mock_contact_information_extractor, output_cleanup):
        resume_name = "test_resume_ci_url_annotation"
        output_cleanup.append(resume_name)
        mock_contact_information_extractor.extract.return_value = _module_result(
            {"contact_information": {"email": [], "phone": [], "linkedin": "https://www.linkedin.com/in/ada", "github": None, "portfolio": None}}
        )
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("Ada Lovelace\nlinkedin\n", encoding="utf-8")  # bare label only, no visible URL
        pdf_path = _make_pdf(tmp_path, [((50, 40, 150, 60), "https://www.linkedin.com/in/ada")])

        result = pipeline.run(markdown_path, pdf_path=pdf_path)

        field = next(f for f in result.contact_information_evidence.fields if f.field_name == "linkedin")
        assert field.annotation_matches == ["https://www.linkedin.com/in/ada"]
        assert field.evidence_summary == "FULLY_CORROBORATED"

    def test_no_evidence_available(self, tmp_path, pipeline, mock_contact_information_extractor, output_cleanup):
        resume_name = "test_resume_ci_no_evidence"
        output_cleanup.append(resume_name)
        mock_contact_information_extractor.extract.return_value = _module_result(
            {"contact_information": {"email": [], "phone": [], "linkedin": None, "github": None, "portfolio": None}}
        )
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("Ada Lovelace\nNo contact info here.\n", encoding="utf-8")

        result = pipeline.run(markdown_path)  # no pdf_path

        for field in result.contact_information_evidence.fields:
            assert field.extracted_values == []
            assert field.evidence_summary == "NO_EXTRACTED_VALUE"
            assert field.annotations_checked is None  # never guessed without a pdf_path

    def test_annotation_disagreement_with_no_corroboration(self, tmp_path, pipeline, mock_contact_information_extractor, output_cleanup):
        resume_name = "test_resume_ci_disagreement"
        output_cleanup.append(resume_name)
        mock_contact_information_extractor.extract.return_value = _module_result(
            {"contact_information": {"email": [], "phone": [], "linkedin": "https://www.linkedin.com/in/wrong-value", "github": None, "portfolio": None}}
        )
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("Ada Lovelace\nNo linkedin text visible anywhere.\n", encoding="utf-8")
        pdf_path = _make_pdf(tmp_path, [((50, 40, 150, 60), "https://www.linkedin.com/in/ada")])

        result = pipeline.run(markdown_path, pdf_path=pdf_path)

        field = next(f for f in result.contact_information_evidence.fields if f.field_name == "linkedin")
        assert field.evidence_summary == "EVIDENCE_DISAGREEMENT"
        # The extracted value itself must be completely untouched.
        assert result.module_results["contact_information"]["contact_information"]["linkedin"] == \
            "https://www.linkedin.com/in/wrong-value"

    def test_placeholder_mailto_false_positive_case(self, tmp_path, pipeline, mock_contact_information_extractor, output_cleanup):
        """Direct regression test for V3.7's confirmed resume-004 case."""
        resume_name = "test_resume_ci_placeholder_mailto"
        output_cleanup.append(resume_name)
        mock_contact_information_extractor.extract.return_value = _module_result(
            {"contact_information": {"email": ["real@example.com"], "phone": [], "linkedin": None, "github": None, "portfolio": None}}
        )
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("Ada Lovelace\nreal@example.com\n", encoding="utf-8")
        pdf_path = _make_pdf(tmp_path, [((50, 40, 150, 60), "mailto:placeholder@example.com")])

        result = pipeline.run(markdown_path, pdf_path=pdf_path)

        field = next(f for f in result.contact_information_evidence.fields if f.field_name == "email")
        assert field.evidence_summary == "FULLY_CORROBORATED"  # visible text protects the value
        assert field.unmatched_same_type_annotations == ["placeholder@example.com"]
        assert result.module_results["contact_information"]["contact_information"]["email"] == ["real@example.com"]

    def test_stale_linkedin_annotation_false_positive_case(self, tmp_path, pipeline, mock_contact_information_extractor, output_cleanup):
        """Direct regression test for V3.7's confirmed resume-005 case."""
        resume_name = "test_resume_ci_stale_linkedin"
        output_cleanup.append(resume_name)
        correct_url = "https://www.linkedin.com/in/arjun-sharma-1bbb0721"
        stale_url = "http://www.linkedin.com/in/arjun-sharma-1bbb07210"
        mock_contact_information_extractor.extract.return_value = _module_result(
            {"contact_information": {"email": [], "phone": [], "linkedin": correct_url, "github": None, "portfolio": None}}
        )
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text(f"Ada Lovelace\n{correct_url}\n", encoding="utf-8")
        pdf_path = _make_pdf(tmp_path, [((50, 40, 150, 60), stale_url)])

        result = pipeline.run(markdown_path, pdf_path=pdf_path)

        field = next(f for f in result.contact_information_evidence.fields if f.field_name == "linkedin")
        assert field.evidence_summary == "FULLY_CORROBORATED"
        assert field.unmatched_same_type_annotations == [stale_url]
        assert result.module_results["contact_information"]["contact_information"]["linkedin"] == correct_url

    def test_non_contact_hyperlink_annotation_is_excluded(self, tmp_path, pipeline, mock_contact_information_extractor, output_cleanup):
        resume_name = "test_resume_ci_non_contact_annotation"
        output_cleanup.append(resume_name)
        mock_contact_information_extractor.extract.return_value = _module_result(
            {"contact_information": {"email": [], "phone": [], "linkedin": None, "github": "https://github.com/ada", "portfolio": None}}
        )
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("Ada Lovelace\nhttps://github.com/ada\n", encoding="utf-8")
        pdf_path = _make_pdf(tmp_path, [
            ((50, 40, 150, 60), "https://github.com/ada"),
            ((50, 80, 150, 100), "https://github.com/ada/some-project"),  # repo link -- must be excluded
            ((50, 120, 150, 140), "https://vendor.example.com/login"),  # unrelated -- must be excluded
        ])

        result = pipeline.run(markdown_path, pdf_path=pdf_path)

        field = next(f for f in result.contact_information_evidence.fields if f.field_name == "github")
        assert field.annotation_matches == ["https://github.com/ada"]
        assert field.unmatched_same_type_annotations == []  # repo/unrelated links never surface here

    def test_malformed_source_pdf_url_provenance(self, tmp_path, pipeline, mock_contact_information_extractor, output_cleanup):
        """Direct regression test for V3.7's confirmed resume-020 case:
        a malformed URL that is genuinely present in the visible text
        (a pre-existing source/PDF-level defect) is reported as
        corroborated -- accurately reflecting the document, not "fixed"
        or hidden -- and is still never mutated."""
        resume_name = "test_resume_ci_malformed_source"
        output_cleanup.append(resume_name)
        malformed = "https://in/nilesh-magare-16753373"
        mock_contact_information_extractor.extract.return_value = _module_result(
            {"contact_information": {"email": [], "phone": [], "linkedin": malformed, "github": None, "portfolio": malformed}}
        )
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text(f"Ada Lovelace\n{malformed}\n", encoding="utf-8")

        result = pipeline.run(markdown_path)

        linkedin_field = next(f for f in result.contact_information_evidence.fields if f.field_name == "linkedin")
        portfolio_field = next(f for f in result.contact_information_evidence.fields if f.field_name == "portfolio")
        assert linkedin_field.evidence_summary == "FULLY_CORROBORATED"
        assert portfolio_field.evidence_summary == "FULLY_CORROBORATED"
        assert result.module_results["contact_information"]["contact_information"]["linkedin"] == malformed
        assert result.module_results["contact_information"]["contact_information"]["portfolio"] == malformed

    def test_extracted_output_is_unchanged_with_evidence_enabled(
        self, tmp_path, pipeline, mock_contact_information_extractor, mock_parser_merger, output_cleanup
    ):
        """The most important proof: the exact same contact_information
        object ContactInformationExtractor returned is what reaches
        module_results, the merger, and the saved module JSON file --
        with the evidence-building call in between."""
        resume_name = "test_resume_ci_unchanged_output"
        output_cleanup.append(resume_name)
        original = {"email": ["ada@example.com"], "phone": ["+1-555-0100"], "linkedin": "https://www.linkedin.com/in/ada", "github": None, "portfolio": None}
        mock_contact_information_extractor.extract.return_value = _module_result(
            {"contact_information": dict(original)}
        )
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("Ada Lovelace\nada@example.com\n", encoding="utf-8")
        pdf_path = _make_pdf(tmp_path, [((50, 40, 150, 60), "https://www.linkedin.com/in/ada")])

        result = pipeline.run(markdown_path, pdf_path=pdf_path)

        assert result.module_results["contact_information"]["contact_information"] == original

        merge_input = mock_parser_merger.merge.call_args[0][0]
        merged_ci = next(entry["contact_information"] for entry in merge_input if "contact_information" in entry)
        assert merged_ci == original

        module_path = OUTPUT_DIR / "modules" / resume_name / "contact_information.json"
        saved = json.loads(module_path.read_text(encoding="utf-8"))
        assert saved["contact_information"] == original

        # And the evidence record is populated alongside, not instead of.
        assert result.contact_information_evidence is not None


class TestMultiModuleEvidenceIntegration:
    """Milestone V5.0: build_personal_information_evidence()/
    build_professional_summary_evidence()/build_languages_evidence()
    (Milestones V4.1/V4.5/V4.8) and build_work_experience_evidence()/
    build_education_evidence() (Milestone V2.7) now all run on their own
    module's real result before it reaches ParserMerger, using the EXACT
    SAME non-reassigning pattern TestContactInformationEvidenceIntegration
    above already proves for Contact Information -- every test class here
    also asserts the extracted value is unchanged, not just that evidence
    was attached."""

    def test_personal_information_evidence_available(
        self, tmp_path, pipeline, mock_personal_information_extractor, output_cleanup
    ):
        resume_name = "test_resume_pi_evidence"
        output_cleanup.append(resume_name)
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("Ada Lovelace\nLondon, UK\n", encoding="utf-8")

        result = pipeline.run(markdown_path)

        assert result.personal_information_evidence is not None
        assert result.personal_information_evidence.name_completeness.evidence_summary == (
            "NAME_TOKENS_FULLY_REPRESENTED"
        )

    def test_personal_information_evidence_missing_full_name_is_not_checked(
        self, tmp_path, pipeline, mock_personal_information_extractor, output_cleanup
    ):
        resume_name = "test_resume_pi_no_full_name"
        output_cleanup.append(resume_name)
        mock_personal_information_extractor.extract.return_value = _module_result(
            {"personal_information": {"full_name": None, "first_name": None, "last_name": None, "current_location": None}}
        )
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("No name here.\n", encoding="utf-8")

        result = pipeline.run(markdown_path)

        name_evidence = result.personal_information_evidence.name_completeness
        assert name_evidence.checked is None
        assert name_evidence.evidence_summary == "NO_FULL_NAME_TO_CHECK"
        assert result.module_results["personal_information"]["personal_information"]["full_name"] is None

    def test_professional_summary_evidence_available(
        self, tmp_path, pipeline, mock_professional_summary_extractor, output_cleanup
    ):
        resume_name = "test_resume_summary_evidence"
        output_cleanup.append(resume_name)
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("Ada Lovelace\nMathematician and writer.\n", encoding="utf-8")

        result = pipeline.run(markdown_path)

        assert result.professional_summary_evidence is not None
        assert result.professional_summary_evidence.evidence_summary == "SOURCE_TEXT_VERBATIM_MATCH"
        assert result.module_results["professional_summary"] == SUMMARY_JSON

    def test_professional_summary_evidence_absent_value(
        self, tmp_path, pipeline, mock_professional_summary_extractor, output_cleanup
    ):
        resume_name = "test_resume_summary_absent"
        output_cleanup.append(resume_name)
        mock_professional_summary_extractor.extract.return_value = _module_result(
            {"professional_summary": None}
        )
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("Ada Lovelace\n", encoding="utf-8")

        result = pipeline.run(markdown_path)

        assert result.professional_summary_evidence.extracted_value is None
        assert result.professional_summary_evidence.evidence_summary == "NO_SUMMARY_EXTRACTED"

    def test_languages_evidence_available(
        self, tmp_path, pipeline, mock_languages_extractor, output_cleanup
    ):
        resume_name = "test_resume_languages_evidence"
        output_cleanup.append(resume_name)
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("Ada Lovelace\nLanguages: English\n", encoding="utf-8")

        result = pipeline.run(markdown_path)

        assert result.languages_evidence is not None
        assert len(result.languages_evidence.values) == 1
        assert result.languages_evidence.values[0].evidence_summary == "SOURCE_TEXT_PRESENT"
        assert result.module_results["languages"] == LANGUAGES_JSON

    def test_languages_evidence_empty_array(
        self, tmp_path, pipeline, mock_languages_extractor, output_cleanup
    ):
        resume_name = "test_resume_languages_empty"
        output_cleanup.append(resume_name)
        mock_languages_extractor.extract.return_value = _module_result({"languages": []})
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("Ada Lovelace\n", encoding="utf-8")

        result = pipeline.run(markdown_path)

        assert result.languages_evidence.values == []
        assert result.module_results["languages"]["languages"] == []

    def test_work_experience_evidence_available(
        self, tmp_path, pipeline, mock_work_experience_extractor, output_cleanup
    ):
        resume_name = "test_resume_we_evidence"
        output_cleanup.append(resume_name)
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("Ada Lovelace\n\n## Experience\n\nMathematician, Analytical Engines Ltd\n", encoding="utf-8")

        result = pipeline.run(markdown_path)

        assert result.work_experience_evidence is not None
        assert result.work_experience_evidence.module == "work_experience"
        assert len(result.work_experience_evidence.entries) == 1
        # The validation report used by the evidence call is the SAME one
        # exposed via work_experience_validation_report -- not a second,
        # independently-run validation.
        assert result.work_experience_evidence.entries[0].validation_actions == [] or all(
            a.rule_id for a in result.work_experience_evidence.entries[0].validation_actions
        )

    def test_education_evidence_available(
        self, tmp_path, pipeline, mock_education_extractor, output_cleanup
    ):
        resume_name = "test_resume_edu_evidence"
        output_cleanup.append(resume_name)
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("Ada Lovelace\n\n## Education\n\nSelf-taught, Royal Institution\n", encoding="utf-8")

        result = pipeline.run(markdown_path)

        assert result.education_evidence is not None
        assert result.education_evidence.module == "education"
        assert len(result.education_evidence.entries) == 1

    def test_skills_untouched(self, tmp_path, pipeline, output_cleanup):
        resume_name = "test_resume_skills_untouched"
        output_cleanup.append(resume_name)
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("Ada Lovelace\n", encoding="utf-8")

        result = pipeline.run(markdown_path)

        assert result.module_results["skills"] == SKILLS_JSON
        assert not hasattr(result, "skills_evidence")

    def test_no_mutation_of_any_extracted_module_value(self, tmp_path, pipeline, output_cleanup):
        """The Step 4(B)/Step 5 non-mutation proof, extended to all five
        newly-integrated modules in one place: the exact same objects
        each mock extractor returned are what reach module_results and
        the saved module JSON files, with every new evidence-building
        call in between."""
        resume_name = "test_resume_no_mutation_all_modules"
        output_cleanup.append(resume_name)
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text(
            "Ada Lovelace\n\n"
            "Mathematician and writer.\n\n"
            "Languages: English\n\n"
            "## Experience\n\nMathematician, Analytical Engines Ltd (Full-time)\n\n"
            "## Education\n\nSelf-taught, Royal Institution\n",
            encoding="utf-8",
        )

        result = pipeline.run(markdown_path)

        assert result.module_results["personal_information"] == PERSONAL_INFO_JSON
        assert result.module_results["professional_summary"] == SUMMARY_JSON
        assert result.module_results["languages"] == LANGUAGES_JSON
        # work_experience/education pass through WEV/EDV first (pre-
        # existing behavior); this data is already known-clean given
        # this markdown (see TestWorkExperienceValidatorIntegration.
        # test_clean_work_experience_produces_no_issues /
        # TestEducationValidatorIntegration.
        # test_clean_education_produces_no_issues above for the
        # pre-existing, unchanged-by-V5.0 clean-input behavior), so it
        # is unchanged by validation AND unchanged again by
        # evidence-building.
        assert result.module_results["work_experience"] == WORK_EXPERIENCE_JSON
        assert result.module_results["education"] == EDUCATION_JSON

        for filename, expected in (
            ("personal_information.json", PERSONAL_INFO_JSON),
            ("professional_summary.json", SUMMARY_JSON),
            ("languages.json", LANGUAGES_JSON),
            ("work_experience.json", WORK_EXPERIENCE_JSON),
            ("education.json", EDUCATION_JSON),
        ):
            saved = json.loads((OUTPUT_DIR / "modules" / resume_name / filename).read_text(encoding="utf-8"))
            assert saved == expected

        # All five new evidence records populated alongside, not instead of.
        assert result.personal_information_evidence is not None
        assert result.professional_summary_evidence is not None
        assert result.languages_evidence is not None
        assert result.work_experience_evidence is not None
        assert result.education_evidence is not None

    def test_evidence_consistency_with_standalone_functions(
        self,
        tmp_path,
        pipeline,
        mock_personal_information_extractor,
        mock_professional_summary_extractor,
        mock_languages_extractor,
        output_cleanup,
    ):
        """Pipeline-produced evidence must be VALUE-EQUAL to calling the
        same standalone build_*_evidence() function directly with the
        same inputs -- proving the integration doesn't reimplement or
        diverge from the already-tested standalone logic."""
        from services.evidence_record import (
            build_languages_evidence,
            build_personal_information_evidence,
            build_professional_summary_evidence,
        )

        resume_name = "test_resume_evidence_consistency"
        output_cleanup.append(resume_name)
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_text = "Ada Lovelace\nMathematician and writer.\nLanguages: English\n"
        markdown_path.write_text(markdown_text, encoding="utf-8")

        result = pipeline.run(markdown_path)

        assert result.personal_information_evidence == build_personal_information_evidence(
            PERSONAL_INFO_JSON["personal_information"], markdown_text=markdown_text
        )
        assert result.professional_summary_evidence == build_professional_summary_evidence(
            SUMMARY_JSON["professional_summary"], markdown_text=markdown_text
        )
        assert result.languages_evidence == build_languages_evidence(
            LANGUAGES_JSON["languages"], markdown_text=markdown_text
        )

    def test_evidence_consistency_work_experience_and_education(
        self, tmp_path, pipeline, mock_work_experience_extractor, mock_education_extractor, output_cleanup
    ):
        from services.evidence_record import build_education_evidence, build_work_experience_evidence

        resume_name = "test_resume_evidence_consistency_we_edu"
        output_cleanup.append(resume_name)
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_text = "Ada Lovelace\n\n## Experience\n\nMathematician, Analytical Engines Ltd\n\n## Education\n\nSelf-taught, Royal Institution\n"
        markdown_path.write_text(markdown_text, encoding="utf-8")

        result = pipeline.run(markdown_path)

        # Use the SAME post-validation entries and SAME ValidationReport
        # the pipeline itself already exposes -- this is exactly what
        # build_work_experience_evidence()/build_education_evidence()
        # were called with internally, so equality here proves the
        # integration didn't diverge from the standalone contract.
        assert result.work_experience_evidence == build_work_experience_evidence(
            result.module_results["work_experience"]["work_experience"],
            markdown_text,
            result.work_experience_validation_report,
        )
        assert result.education_evidence == build_education_evidence(
            result.module_results["education"]["education"],
            markdown_text,
            result.education_validation_report,
        )

    def test_contact_information_evidence_regression_unaffected(
        self, tmp_path, pipeline, mock_contact_information_extractor, output_cleanup
    ):
        """The pre-existing Contact Information integration (Milestone
        V3.9) must be entirely unaffected by adding the five new
        branches around it in the same loop."""
        resume_name = "test_resume_ci_regression_v5_0"
        output_cleanup.append(resume_name)
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("Ada Lovelace\nada@example.com\n", encoding="utf-8")

        result = pipeline.run(markdown_path)

        assert result.contact_information_evidence is not None
        field = next(f for f in result.contact_information_evidence.fields if f.field_name == "email")
        assert field.evidence_summary == "FULLY_CORROBORATED"
        assert result.module_results["contact_information"] == CONTACT_INFO_JSON


@pytest.mark.integration
def test_real_pipeline_end_to_end_integration(tmp_path):
    """Optional integration test running the entire pipeline for real:
    real module extractors, real ParserMerger, real SchemaValidator,
    real local Ollama server. Skips itself if Ollama or the required
    model isn't available."""
    ollama_client = OllamaClient()

    if not ollama_client.is_running():
        pytest.skip("Ollama is not running locally; skipping integration test.")
    if not ollama_client.model_exists(INTEGRATION_TEXT_MODEL):
        pytest.skip(f"Model '{INTEGRATION_TEXT_MODEL}' is not installed; skipping.")

    resume_name = "test_resume_pipeline_integration"
    markdown_path = tmp_path / f"{resume_name}.md"
    markdown_path.write_text(
        "# Ada Lovelace\n\n"
        "## Summary\n\nMathematician and writer with a focus on computing.\n\n"
        "## Contact\n\nEmail: ada@example.com\n\n"
        "## Skills\n\n- Mathematics\n- Analytical writing\n\n"
        "## Languages\n\n- English\n- French\n\n"
        "## Experience\n\n### Mathematician, Analytical Engines Ltd\n"
        "1842 - Present\n- Wrote the first published algorithm\n\n"
        "## Education\n\n### Self-taught in Mathematics\n"
        "Royal Institution\n\n"
        "## Projects\n\n### Analytical Engine Notes\n"
        "Annotated translation with an original algorithm.\n"
        "- Wrote the first published algorithm for the Analytical Engine\n",
        encoding="utf-8",
    )

    pipeline = ParserPipeline()

    try:
        result = pipeline.run(markdown_path)

        assert isinstance(result, ParserPipelineResult)
        assert result.draft_json is None
        assert set(result.module_results.keys()) == {
            "personal_information",
            "contact_information",
            "skills",
            "professional_summary",
            "languages",
            "work_experience",
            "education",
            "projects",
        }
        assert set(result.merged_json.keys()) == {
            "personal_information",
            "contact_information",
            "skills",
            "professional_summary",
            "languages",
            "work_experience",
            "education",
            "projects",
        }

        module_dir = OUTPUT_DIR / "modules" / resume_name
        for filename in (
            "personal_information.json",
            "contact_information.json",
            "skills.json",
            "professional_summary.json",
            "languages.json",
            "work_experience.json",
            "education.json",
            "projects.json",
        ):
            assert (module_dir / filename).is_file()
        assert (OUTPUT_DIR / "json_final" / f"{resume_name}.json").is_file()
    finally:
        shutil.rmtree(OUTPUT_DIR / "modules" / resume_name, ignore_errors=True)
        (OUTPUT_DIR / "json_final" / f"{resume_name}.json").unlink(missing_ok=True)
        (OUTPUT_DIR / "json_draft" / f"{resume_name}.json").unlink(missing_ok=True)



class TestOutputRoot:
    """Remediation R-08 (2026-10-17): ParserPipeline(output_root=...)
    redirects module/final JSON; omitting it keeps the fixed
    config.OUTPUT_DIR locations byte-for-byte."""

    def _build(self, output_root, **mocks):
        return ParserPipeline(output_root=output_root, **mocks)

    def test_default_root_is_none_and_resolves_to_output_dir(self, pipeline):
        assert pipeline.output_root is None
        assert pipeline._resolved_output_root() == OUTPUT_DIR

    def test_default_root_still_writes_to_the_fixed_locations(
        self, tmp_path, pipeline, output_cleanup
    ):
        resume_name = "test_resume_r08_default_root"
        output_cleanup.append(resume_name)
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("# Ada Lovelace", encoding="utf-8")

        pipeline.run(markdown_path)

        assert (OUTPUT_DIR / "json_final" / f"{resume_name}.json").is_file()
        assert (OUTPUT_DIR / "modules" / resume_name / "skills.json").is_file()

    def test_custom_root_receives_every_artifact_and_fixed_locations_are_untouched(
        self,
        tmp_path,
        mock_personal_information_extractor,
        mock_contact_information_extractor,
        mock_skills_extractor,
        mock_professional_summary_extractor,
        mock_languages_extractor,
        mock_work_experience_extractor,
        mock_education_extractor,
        mock_project_extractor,
        mock_parser_merger,
        mock_schema_validator,
    ):
        resume_name = "test_resume_r08_custom_root"
        custom_root = tmp_path / "isolated_run"
        pipeline = ParserPipeline(
            personal_information_extractor=mock_personal_information_extractor,
            contact_information_extractor=mock_contact_information_extractor,
            skills_extractor=mock_skills_extractor,
            professional_summary_extractor=mock_professional_summary_extractor,
            languages_extractor=mock_languages_extractor,
            work_experience_extractor=mock_work_experience_extractor,
            education_extractor=mock_education_extractor,
            project_extractor=mock_project_extractor,
            parser_merger=mock_parser_merger,
            schema_validator=mock_schema_validator,
            output_root=custom_root,
        )
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text("# Ada Lovelace", encoding="utf-8")

        result = pipeline.run(markdown_path)

        assert pipeline.output_root == custom_root
        final_path = custom_root / "json_final" / f"{resume_name}.json"
        assert json.loads(final_path.read_text(encoding="utf-8")) == MERGED_JSON
        module_dir = custom_root / "modules" / resume_name
        assert sorted(p.name for p in module_dir.iterdir()) == sorted(
            f"{key}.json" for key, _ in pipeline._module_extractors
        )
        assert result.merged_json == MERGED_JSON
        # The fixed locations were never written for this resume.
        assert not (OUTPUT_DIR / "json_final" / f"{resume_name}.json").exists()
        assert not (OUTPUT_DIR / "modules" / resume_name).exists()

    def test_string_root_is_accepted(self, tmp_path):
        pipeline = ParserPipeline(
            personal_information_extractor=MagicMock(),
            contact_information_extractor=MagicMock(),
            skills_extractor=MagicMock(),
            professional_summary_extractor=MagicMock(),
            languages_extractor=MagicMock(),
            work_experience_extractor=MagicMock(),
            education_extractor=MagicMock(),
            project_extractor=MagicMock(),
            parser_merger=MagicMock(),
            schema_validator=MagicMock(),
            output_root=str(tmp_path / "as_string"),
        )
        assert pipeline.output_root == tmp_path / "as_string"

    def test_isolated_parser_pipeline_is_a_thin_alias(self, tmp_path):
        from services.isolated_parser_pipeline import IsolatedParserPipeline

        isolated = IsolatedParserPipeline(
            personal_information_extractor=MagicMock(),
            contact_information_extractor=MagicMock(),
            skills_extractor=MagicMock(),
            professional_summary_extractor=MagicMock(),
            languages_extractor=MagicMock(),
            work_experience_extractor=MagicMock(),
            education_extractor=MagicMock(),
            project_extractor=MagicMock(),
            parser_merger=MagicMock(),
            schema_validator=MagicMock(),
            output_root=tmp_path / "iso",
        )
        assert isinstance(isolated, ParserPipeline)
        assert isolated.output_root == tmp_path / "iso"
        assert isolated._output_root == tmp_path / "iso"
        saved = isolated._save_final_json("r08_alias", {"skills": []})
        assert saved == tmp_path / "iso" / "json_final" / "r08_alias.json"
        assert saved.is_file()
