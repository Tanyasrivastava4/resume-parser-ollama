"""
tests/test_parser_pipeline_retry.py

Tests for the per-module retry behavior added to pipeline/parser_pipeline
.py in Milestone 14.9.2.

Every dependency is mocked via constructor injection, exactly like
tests/test_parser_pipeline.py (which this file leaves untouched). These
tests focus specifically on retry mechanics: which exceptions get
retried, how many attempts are made, that only the failing step is
retried (not the whole resume), and that retry_count is reported
correctly on ParserPipelineResult.

Milestone 23.7: JsonDraftGenerator has been retired (see
docs/JSON_DRAFT_GENERATOR_RETIREMENT_RESULTS.md) — its own retry test
was removed, and the one test that also incidentally asserted on its
call count now asserts only on the module extractors it names
directly (personal_information/contact_information/skills), not on an
overall module count — unaffected by Milestone T5/Phase 1B's later
addition of an eighth module extractor (projects).
"""

import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from config import OUTPUT_DIR
from pipeline.parser_pipeline import MAX_ATTEMPTS, ParserPipeline, ParserPipelineResult
from services.base_module_extractor import InvalidJSONError, MarkdownNotFoundError
from services.contact_information_extractor import ContactInformationExtractor
from services.education_extractor import EducationExtractor
from services.languages_extractor import LanguagesExtractor
from services.ollama_client import ModelNotFoundError
from services.parser_merger import ParserMerger
from services.personal_information_extractor import PersonalInformationExtractor
from services.professional_summary_extractor import ProfessionalSummaryExtractor
from services.project_extractor import ProjectExtractor
from services.prompt_manager import PromptNotFoundError
from services.skills_extractor import SkillsExtractor
from services.work_experience_extractor import WorkExperienceExtractor
from validators.schema_validator import SchemaValidationError, SchemaValidator

PERSONAL_INFO_JSON = {"personal_information": {"full_name": "Ada Lovelace"}}
CONTACT_INFO_JSON = {"contact_information": {"email": []}}
SKILLS_JSON = {"skills": ["Python"]}
SUMMARY_JSON = {"professional_summary": None}
LANGUAGES_JSON = {"languages": ["English"]}
WORK_EXPERIENCE_JSON = {"work_experience": []}
EDUCATION_JSON = {"education": []}
PROJECTS_JSON = {"projects": []}

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


@pytest.fixture(autouse=True)
def output_cleanup():
    """Every test in this file uses the markdown filename "resume.md",
    and ParserPipeline always saves module/final output to the real
    config.OUTPUT_DIR regardless of where the source markdown lives —
    clean that up after every test so it doesn't leak into the real
    project tree."""
    yield
    shutil.rmtree(OUTPUT_DIR / "modules" / "resume", ignore_errors=True)
    (OUTPUT_DIR / "json_final" / "resume.json").unlink(missing_ok=True)


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


class TestMaxAttempts:
    def test_max_attempts_is_three(self):
        assert MAX_ATTEMPTS == 3


class TestRetryOnInvalidJSONError:
    def test_succeeds_on_second_attempt(
        self, tmp_path, pipeline, mock_skills_extractor
    ):
        mock_skills_extractor.extract.side_effect = [
            InvalidJSONError("bad json"),
            _module_result(SKILLS_JSON),
        ]
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        result = pipeline.run(markdown_path)

        assert result.merged_json == MERGED_JSON
        assert mock_skills_extractor.extract.call_count == 2
        assert result.retry_count == 1

    def test_succeeds_on_third_attempt(self, tmp_path, pipeline, mock_skills_extractor):
        mock_skills_extractor.extract.side_effect = [
            InvalidJSONError("bad json"),
            InvalidJSONError("bad json again"),
            _module_result(SKILLS_JSON),
        ]
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        result = pipeline.run(markdown_path)

        assert mock_skills_extractor.extract.call_count == 3
        assert result.retry_count == 2

    def test_raises_original_exception_after_exhausting_retries(
        self, tmp_path, pipeline, mock_skills_extractor
    ):
        mock_skills_extractor.extract.side_effect = InvalidJSONError("always bad")
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        with pytest.raises(InvalidJSONError, match="always bad"):
            pipeline.run(markdown_path)

        # 1 initial attempt + 2 retries = 3 total calls.
        assert mock_skills_extractor.extract.call_count == 3


class TestRetryOnSchemaValidationError:
    def test_succeeds_on_second_attempt(
        self, tmp_path, pipeline, mock_contact_information_extractor
    ):
        mock_contact_information_extractor.extract.side_effect = [
            SchemaValidationError("email must be an array"),
            _module_result(CONTACT_INFO_JSON),
        ]
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        result = pipeline.run(markdown_path)

        assert result.merged_json == MERGED_JSON
        assert mock_contact_information_extractor.extract.call_count == 2
        assert result.retry_count == 1

    def test_raises_original_exception_after_exhausting_retries(
        self, tmp_path, pipeline, mock_contact_information_extractor
    ):
        mock_contact_information_extractor.extract.side_effect = SchemaValidationError(
            "always invalid"
        )
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        with pytest.raises(SchemaValidationError, match="always invalid"):
            pipeline.run(markdown_path)

        assert mock_contact_information_extractor.extract.call_count == 3


class TestRetryScopedToFailingModuleOnly:
    def test_only_the_failing_module_is_retried_not_the_whole_resume(
        self,
        tmp_path,
        pipeline,
        mock_personal_information_extractor,
        mock_contact_information_extractor,
        mock_skills_extractor,
    ):
        """A failure in contact_information must not re-run
        personal_information — only contact_information itself is
        retried."""
        mock_contact_information_extractor.extract.side_effect = [
            SchemaValidationError("bad"),
            _module_result(CONTACT_INFO_JSON),
        ]
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        pipeline.run(markdown_path)

        assert mock_personal_information_extractor.extract.call_count == 1
        assert mock_contact_information_extractor.extract.call_count == 2
        assert mock_skills_extractor.extract.call_count == 1


class TestNonRetryableExceptions:
    @pytest.mark.parametrize(
        "exception",
        [
            MarkdownNotFoundError("markdown missing"),
            PromptNotFoundError("prompt missing"),
            ModelNotFoundError("model missing"),
        ],
    )
    def test_non_retryable_exception_is_not_retried(
        self, tmp_path, pipeline, mock_skills_extractor, exception
    ):
        mock_skills_extractor.extract.side_effect = exception
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        with pytest.raises(type(exception)):
            pipeline.run(markdown_path)

        mock_skills_extractor.extract.assert_called_once()


class TestRetryCountReporting:
    def test_retry_count_is_zero_when_everything_succeeds_first_try(
        self, tmp_path, pipeline
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        result = pipeline.run(markdown_path)

        assert isinstance(result, ParserPipelineResult)
        assert result.retry_count == 0

    def test_retry_count_sums_across_multiple_failing_steps(
        self,
        tmp_path,
        pipeline,
        mock_personal_information_extractor,
        mock_languages_extractor,
    ):
        mock_personal_information_extractor.extract.side_effect = [
            InvalidJSONError("bad"),
            _module_result(PERSONAL_INFO_JSON),
        ]
        mock_languages_extractor.extract.side_effect = [
            SchemaValidationError("bad"),
            SchemaValidationError("bad again"),
            _module_result(LANGUAGES_JSON),
        ]
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        result = pipeline.run(markdown_path)

        # personal_information: 1 retry, languages: 2 retries -> 3 total.
        assert result.retry_count == 3


class TestRetryLogging:
    def test_logs_retry_attempt_number_and_reason(
        self, tmp_path, pipeline, mock_skills_extractor, caplog
    ):
        mock_skills_extractor.extract.side_effect = [
            SchemaValidationError("email must be an array"),
            _module_result(SKILLS_JSON),
        ]
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        with caplog.at_level("WARNING", logger="pipeline.parser_pipeline"):
            pipeline.run(markdown_path)

        retry_messages = [r.message for r in caplog.records if "Retrying" in r.message]
        assert len(retry_messages) == 1
        assert "SkillsExtractor" in retry_messages[0]
        assert "2/3" in retry_messages[0]
        assert "SchemaValidationError" in retry_messages[0]
