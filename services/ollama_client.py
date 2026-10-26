"""
services/ollama_client.py

Thin client around a local Ollama server (Milestone 8).

Every pipeline in PROJECT_PLAN.md runs its models (Qwen2.5-VL, Qwen 1.7B)
through Ollama, so this module gives them one shared way to talk to it:
check the server is up, list/verify installed models, and send a text or
vision prompt. It is a transport layer only.

Transient connection failures (OllamaConnectionError) are retried with
exponential backoff, configured via settings.OLLAMA_MAX_RETRIES and
settings.OLLAMA_RETRY_BACKOFF_SECONDS. A fully unreachable server, a
missing model, and Ollama's own response errors are never retried.

This module does not parse resumes, does not know about prompts, and does
not extract or interpret JSON from model output — callers get back the raw
`ollama.GenerateResponse` and decide what to do with it.
"""

import logging
import time
from pathlib import Path
from typing import List, Sequence, Union

import httpx
import ollama

from settings import (
    OLLAMA_HOST,
    OLLAMA_MAX_RETRIES,
    OLLAMA_RETRY_BACKOFF_SECONDS,
    OLLAMA_TIMEOUT,
)

logger = logging.getLogger(__name__)

# Ollama defaults to a 4096-token context window regardless of what a
# model actually supports (e.g. qwen2.5vl:3b supports 128K, per `ollama
# show qwen2.5vl:3b`), unless a request explicitly asks for more via this
# option.
DEFAULT_NUM_CTX = 8192


class OllamaClientError(Exception):
    """Base exception for all Ollama client failures."""


class OllamaNotRunningError(OllamaClientError):
    """Raised when the local Ollama server cannot be reached at all."""


class OllamaConnectionError(OllamaClientError):
    """Raised for connection-level failures other than the server being
    down, e.g. a timeout or an unexpected transport error."""


class ModelNotFoundError(OllamaClientError):
    """Raised when a requested model is not installed in Ollama."""


class OllamaClient:
    """A minimal wrapper around the local Ollama server.

    Responsibilities are limited to connectivity and raw prompt/response
    plumbing: checking the server is running, listing and verifying
    models, and sending text or vision prompts. It does not implement any
    prompt engineering, resume parsing, or response parsing — every method
    returns Ollama's raw response object untouched.
    """

    def __init__(
        self,
        host: str = OLLAMA_HOST,
        timeout: float = OLLAMA_TIMEOUT,
        max_retries: int = OLLAMA_MAX_RETRIES,
        retry_backoff_seconds: float = OLLAMA_RETRY_BACKOFF_SECONDS,
    ) -> None:
        """Initialize the client.

        Args:
            host: Base URL of the local Ollama server, e.g.
                "http://localhost:11434".
            timeout: Request timeout in seconds for calls to the server.
            max_retries: Maximum number of retries for a transient
                OllamaConnectionError, not counting the initial attempt.
                Defaults to settings.OLLAMA_MAX_RETRIES.
            retry_backoff_seconds: Base delay, in seconds, before the
                first retry; each subsequent retry doubles it. Defaults
                to settings.OLLAMA_RETRY_BACKOFF_SECONDS.
        """
        self.host = host
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._client = ollama.Client(host=host, timeout=timeout)

        # Number of retries the most recent call needed. Purely
        # informational (e.g. for DocumentExtractionResult.retry_count) —
        # nothing here depends on it being read.
        self.last_retry_count = 0

        logger.debug(
            "OllamaClient configured for host=%s timeout=%s max_retries=%s "
            "retry_backoff_seconds=%s",
            host,
            timeout,
            max_retries,
            retry_backoff_seconds,
        )

    def is_running(self) -> bool:
        """Check whether the local Ollama server is reachable.

        This is a plain health check: it never raises, it only reports
        True or False so callers can decide what to do next.

        Returns:
            True if the server responded, False if it could not be
            reached for any reason.
        """
        try:
            self._client.list()
            return True
        except Exception as exc:  # noqa: BLE001 - health check must not raise
            logger.warning("Ollama is not reachable at %s: %s", self.host, exc)
            return False

    def list_models(self) -> List[str]:
        """List the names of models currently installed in Ollama.

        Returns:
            A list of installed model names, e.g. ["qwen2.5vl:3b", "qwen3:1.7b"].

        Raises:
            OllamaNotRunningError: If the Ollama server cannot be reached.
            OllamaConnectionError: If a different connection-level error
                occurs (e.g. a timeout).
        """
        response = self._call(self._client.list)
        models = [model.model for model in response.models if model.model]
        logger.debug("Installed models: %s", models)
        return models

    def model_exists(self, model: str) -> bool:
        """Check whether a specific model is installed in Ollama.

        Args:
            model: Model name to look for, e.g. "qwen2.5vl:3b".

        Returns:
            True if the model is installed, False otherwise.

        Raises:
            OllamaNotRunningError: If the Ollama server cannot be reached.
            OllamaConnectionError: If a different connection-level error
                occurs (e.g. a timeout).
        """
        return model in self.list_models()

    def generate_text(self, model: str, prompt: str) -> ollama.GenerateResponse:
        """Send a text prompt to a model and return the raw response.

        Args:
            model: Name of an installed text model, e.g. "qwen3:1.7b".
            prompt: The prompt text to send.

        Returns:
            The raw `ollama.GenerateResponse` object, unparsed.

        Raises:
            ModelNotFoundError: If `model` is not installed in Ollama.
            OllamaNotRunningError: If the Ollama server cannot be reached.
            OllamaConnectionError: If a different connection-level error
                occurs (e.g. a timeout).
        """
        self._ensure_model_exists(model)
        logger.info("Sending text prompt to model %s", model)
        return self._call(
            self._client.generate,
            model=model,
            prompt=prompt,
            options={"num_ctx": DEFAULT_NUM_CTX},
        )

    def generate_vision(
        self,
        model: str,
        prompt: str,
        image_paths: Sequence[Union[str, Path]],
    ) -> ollama.GenerateResponse:
        """Send a prompt with one or more images to a vision model.

        Args:
            model: Name of an installed vision-capable model, e.g.
                "qwen2.5vl:3b".
            prompt: The prompt text to send alongside the images.
            image_paths: One or more paths to image files (e.g. the PNGs
                produced by services/pdf_loader.py) to attach to the
                prompt.

        Returns:
            The raw `ollama.GenerateResponse` object, unparsed.

        Raises:
            ModelNotFoundError: If `model` is not installed in Ollama.
            OllamaNotRunningError: If the Ollama server cannot be reached.
            OllamaConnectionError: If a different connection-level error
                occurs (e.g. a timeout).
        """
        self._ensure_model_exists(model)
        image_paths = [str(path) for path in image_paths]
        logger.info(
            "Sending vision prompt to model %s with %d image(s)",
            model,
            len(image_paths),
        )
        return self._call(
            self._client.generate,
            model=model,
            prompt=prompt,
            images=image_paths,
            options={"num_ctx": DEFAULT_NUM_CTX},
        )

    def _ensure_model_exists(self, model: str) -> None:
        """Raise ModelNotFoundError if `model` is not installed."""
        if not self.model_exists(model):
            logger.error("Requested model is not installed: %s", model)
            raise ModelNotFoundError(f"Model '{model}' is not installed in Ollama.")

    def _call(self, func, *args, **kwargs):
        """Call an underlying ollama.Client method, retrying on transient
        connection failures with exponential backoff.

        Only OllamaConnectionError (e.g. a timeout) is retried. A fully
        unreachable server (OllamaNotRunningError) and Ollama's own
        ResponseError (e.g. a bad request) are not retried, since trying
        again wouldn't help either of those. After `self.max_retries`
        retries are exhausted, the original OllamaConnectionError is
        raised.
        """
        self.last_retry_count = 0

        while True:
            try:
                result = self._invoke(func, *args, **kwargs)
                if self.last_retry_count:
                    logger.info(
                        "Ollama call succeeded after %d retr(y/ies)",
                        self.last_retry_count,
                    )
                return result
            except OllamaConnectionError as exc:
                if self.last_retry_count >= self.max_retries:
                    logger.error(
                        "Ollama call failed after %d retr(y/ies); giving up: %s",
                        self.last_retry_count,
                        exc,
                    )
                    raise
                wait_seconds = self.retry_backoff_seconds * (2**self.last_retry_count)
                self.last_retry_count += 1
                logger.warning(
                    "Retry %d/%d after %d seconds because of %s",
                    self.last_retry_count,
                    self.max_retries,
                    wait_seconds,
                    exc,
                )
                time.sleep(wait_seconds)

    def _invoke(self, func, *args, **kwargs):
        """Call `func` once, translating transport failures into this
        module's custom exceptions.

        `ConnectionError` (raised by the ollama library when it cannot
        reach the server at all) becomes OllamaNotRunningError; any other
        httpx-level transport error (e.g. a timeout) becomes
        OllamaConnectionError. Ollama's own `ResponseError` (e.g. a bad
        request) is left to propagate as-is.
        """
        try:
            return func(*args, **kwargs)
        except ConnectionError as exc:
            logger.error("Ollama server is not running at %s", self.host)
            raise OllamaNotRunningError(
                f"Could not reach Ollama at {self.host}. Is it running?"
            ) from exc
        except httpx.HTTPError as exc:
            logger.error("Connection error talking to Ollama at %s: %s", self.host, exc)
            raise OllamaConnectionError(
                f"Connection error talking to Ollama at {self.host}: {exc}"
            ) from exc
