"""
tests/test_resume_retry_queue.py

Tests for pipeline/resume_retry_queue.py (Milestone 14.9.3).

ParserPipeline is mocked via constructor injection for every unit test,
so these exercise only ResumeRetryQueue's own orchestration: does it
retry the whole resume exactly once on a retryable failure, does it
leave non-retryable failures alone, and does it report the right
attempt/reason in its logs. One integration test at the bottom runs a
real ResumeRetryQueue wrapping a real ParserPipeline against a real
local Ollama server, confirming the normal (non-retry) path works
end to end; it skips itself if Ollama or the model isn't available.
"""

import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from config import OUTPUT_DIR
from pipeline.parser_pipeline import ParserPipeline, ParserPipelineResult
from pipeline.resume_retry_queue import MAX_RESUME_ATTEMPTS, ResumeRetryQueue
from services.base_module_extractor import InvalidJSONError, MarkdownNotFoundError
from services.ollama_client import ModelNotFoundError, OllamaClient
from validators.schema_validator import SchemaValidationError

INTEGRATION_TEXT_MODEL = "qwen3:1.7b"

FAKE_RESULT = ParserPipelineResult(
    draft_json={"note": "draft"},
    module_results={"skills": {"skills": ["Python"]}},
    merged_json={"skills": ["Python"]},
    processing_time=0.05,
    retry_count=0,
)


@pytest.fixture
def mock_parser_pipeline():
    pipeline = MagicMock(spec=ParserPipeline)
    pipeline.run.return_value = FAKE_RESULT
    return pipeline


@pytest.fixture
def queue(mock_parser_pipeline):
    return ResumeRetryQueue(parser_pipeline=mock_parser_pipeline)


class TestMaxResumeAttempts:
    def test_max_resume_attempts_is_two(self):
        assert MAX_RESUME_ATTEMPTS == 2


class TestHappyPath:
    def test_succeeds_on_first_attempt_without_retry(
        self, tmp_path, queue, mock_parser_pipeline
    ):
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        result = queue.run(markdown_path)

        assert result is FAKE_RESULT
        mock_parser_pipeline.run.assert_called_once_with(markdown_path)


class TestResumeRetryOnInvalidJSONError:
    def test_succeeds_on_retry(self, tmp_path, queue, mock_parser_pipeline):
        mock_parser_pipeline.run.side_effect = [
            InvalidJSONError("SkillsExtractor failed: bad json"),
            FAKE_RESULT,
        ]
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        result = queue.run(markdown_path)

        assert result is FAKE_RESULT
        assert mock_parser_pipeline.run.call_count == 2

    def test_raises_after_exhausting_the_one_retry(
        self, tmp_path, queue, mock_parser_pipeline
    ):
        mock_parser_pipeline.run.side_effect = InvalidJSONError("still bad")
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        with pytest.raises(InvalidJSONError, match="still bad"):
            queue.run(markdown_path)

        # 1 initial attempt + 1 retry = 2 total calls.
        assert mock_parser_pipeline.run.call_count == 2


class TestResumeRetryOnSchemaValidationError:
    def test_succeeds_on_retry(self, tmp_path, queue, mock_parser_pipeline):
        mock_parser_pipeline.run.side_effect = [
            SchemaValidationError("SkillsExtractor failed: not unique"),
            FAKE_RESULT,
        ]
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        result = queue.run(markdown_path)

        assert result is FAKE_RESULT
        assert mock_parser_pipeline.run.call_count == 2

    def test_raises_after_exhausting_the_one_retry(
        self, tmp_path, queue, mock_parser_pipeline
    ):
        mock_parser_pipeline.run.side_effect = SchemaValidationError("still invalid")
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        with pytest.raises(SchemaValidationError, match="still invalid"):
            queue.run(markdown_path)

        assert mock_parser_pipeline.run.call_count == 2


class TestNonRetryableExceptions:
    @pytest.mark.parametrize(
        "exception",
        [
            MarkdownNotFoundError("markdown missing"),
            ModelNotFoundError("model missing"),
        ],
    )
    def test_non_retryable_exception_is_not_retried(
        self, tmp_path, queue, mock_parser_pipeline, exception
    ):
        mock_parser_pipeline.run.side_effect = exception
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        with pytest.raises(type(exception)):
            queue.run(markdown_path)

        mock_parser_pipeline.run.assert_called_once()


class TestRetryLogging:
    def test_logs_resume_retry_attempt_and_reason(
        self, tmp_path, queue, mock_parser_pipeline, caplog
    ):
        mock_parser_pipeline.run.side_effect = [
            SchemaValidationError("SkillsExtractor failed: not unique"),
            FAKE_RESULT,
        ]
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        with caplog.at_level("WARNING", logger="pipeline.resume_retry_queue"):
            queue.run(markdown_path)

        retry_messages = [r.message for r in caplog.records if "Resume retry" in r.message]
        assert len(retry_messages) == 1
        assert "attempt 2/2" in retry_messages[0].lower() or "2/2" in retry_messages[0]
        assert "SkillsExtractor failed" in retry_messages[0]

    def test_logs_final_failure_reason(self, tmp_path, queue, mock_parser_pipeline, caplog):
        mock_parser_pipeline.run.side_effect = InvalidJSONError(
            "SkillsExtractor failed: still bad"
        )
        markdown_path = tmp_path / "resume.md"
        markdown_path.write_text("# Resume", encoding="utf-8")

        with caplog.at_level("ERROR", logger="pipeline.resume_retry_queue"):
            with pytest.raises(InvalidJSONError):
                queue.run(markdown_path)

        failure_messages = [
            r.message for r in caplog.records if "marking failed" in r.message
        ]
        assert len(failure_messages) == 1
        assert "SkillsExtractor failed" in failure_messages[0]


@pytest.mark.integration
def test_real_resume_retry_queue_happy_path_integration(tmp_path):
    """Optional integration test: a real ResumeRetryQueue wrapping a
    real ParserPipeline, against a real local Ollama server. Skips
    itself if Ollama or the model isn't available. Only exercises the
    normal (non-retry) path — genuinely forcing a retryable failure
    against a real model isn't reliably reproducible."""
    ollama_client = OllamaClient()

    if not ollama_client.is_running():
        pytest.skip("Ollama is not running locally; skipping integration test.")
    if not ollama_client.model_exists(INTEGRATION_TEXT_MODEL):
        pytest.skip(f"Model '{INTEGRATION_TEXT_MODEL}' is not installed; skipping.")

    resume_name = "test_resume_retry_queue_integration"
    markdown_path = tmp_path / f"{resume_name}.md"
    markdown_path.write_text(
        "# Ada Lovelace\n\n## Skills\n\n- Mathematics\n- Analytical writing\n",
        encoding="utf-8",
    )

    queue = ResumeRetryQueue()

    try:
        result = queue.run(markdown_path)

        assert isinstance(result, ParserPipelineResult)
        assert isinstance(result.merged_json, dict)
    finally:
        shutil.rmtree(OUTPUT_DIR / "modules" / resume_name, ignore_errors=True)
        (OUTPUT_DIR / "json_final" / f"{resume_name}.json").unlink(missing_ok=True)
        (OUTPUT_DIR / "json_draft" / f"{resume_name}.json").unlink(missing_ok=True)
