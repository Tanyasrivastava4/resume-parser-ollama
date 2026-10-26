"""
pipeline/resume_retry_queue.py

Resume Retry Queue (Milestone 14.9.3).

Wraps ParserPipeline with one more layer of resilience, on top of the
per-module retry already built into ParserPipeline itself (Milestone
14.9.2). ParserPipeline already retries a single failing step (the JSON
draft, or any one module extractor) up to 3 attempts before giving up.
If a resume still fails after that — every module retry exhausted — this
queue retries the WHOLE resume once, from the very start of the
pipeline: a fresh draft, fresh module extractions. The model may simply
produce a better result on a clean attempt. If that second, whole-resume
attempt also fails, the resume is marked failed: the original exception
propagates to the caller, which is expected to catch it and move on to
the next resume (the same pattern BatchProcessor, ParserEvaluator, and
scripts/run_parser_pipeline.py already use).

This is pure orchestration. It does not change ParserPipeline's own
per-module retry behavior, does not touch any prompt, and does not
change any extractor/merger/validator logic.
"""

import logging
from pathlib import Path
from typing import Optional, Union

from pipeline.parser_pipeline import ParserPipeline, ParserPipelineResult
from services.base_module_extractor import InvalidJSONError
from validators.schema_validator import SchemaValidationError

logger = logging.getLogger(__name__)

# Total attempts for the WHOLE resume: 1 initial + 1 retry = 2 attempts.
# Each of those attempts already includes ParserPipeline's own
# per-module retries — this is a coarser, last-resort layer on top.
MAX_RESUME_ATTEMPTS = 2

# The same failure modes ParserPipeline's own per-module retry targets —
# a fresh attempt of the whole resume might simply produce a better
# result. Non-retryable errors (missing markdown/prompt/model, Ollama
# unreachable) are deterministic and would fail identically again, so
# they are never retried here either.
RETRYABLE_EXCEPTIONS = (InvalidJSONError, SchemaValidationError)


class ResumeRetryQueue:
    """Retries a whole resume once if it still fails after
    ParserPipeline's own per-module retries are exhausted.
    """

    def __init__(self, parser_pipeline: Optional[ParserPipeline] = None) -> None:
        """Initialize the queue.

        Args:
            parser_pipeline: ParserPipeline to run each resume through.
                Defaults to a new ParserPipeline().
        """
        self.parser_pipeline = parser_pipeline or ParserPipeline()

    def run(self, markdown_path: Union[str, Path]) -> ParserPipelineResult:
        """Run one resume through ParserPipeline, retrying the whole
        resume once if it fails with a retryable error.

        Args:
            markdown_path: Path to the source markdown file, e.g.
                output/markdown/resume.md.

        Returns:
            The ParserPipelineResult from whichever attempt succeeded.

        Raises:
            Whatever ParserPipeline.run() raised on the final attempt,
            unmodified, if every attempt failed with a retryable
            exception (InvalidJSONError or SchemaValidationError). Any
            non-retryable exception (e.g. MarkdownNotFoundError,
            PromptNotFoundError, ModelNotFoundError,
            OllamaNotRunningError) propagates immediately, with no
            resume-level retry at all.
        """
        attempt = 1
        while True:
            try:
                return self.parser_pipeline.run(markdown_path)
            except RETRYABLE_EXCEPTIONS as exc:
                if attempt >= MAX_RESUME_ATTEMPTS:
                    logger.error(
                        "Resume failed after %d attempt(s); marking failed. "
                        "Reason: %s",
                        attempt,
                        exc,
                    )
                    raise
                attempt += 1
                logger.warning(
                    "Resume retry — attempt %d/%d — reason: %s",
                    attempt,
                    MAX_RESUME_ATTEMPTS,
                    exc,
                )
