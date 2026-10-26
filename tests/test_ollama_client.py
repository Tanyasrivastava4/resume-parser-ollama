"""
tests/test_ollama_client.py

Tests for services/ollama_client.py (Milestone 8).

Most tests mock the underlying ollama.Client so they exercise our
connectivity/model-verification/error-translation logic without needing a
real Ollama server or downloaded models. The client's constructor does not
itself make a network call, so a real OllamaClient can be built and then
have its internal `_client` replaced with a mock.

One integration test at the bottom talks to a real local Ollama server. It
skips itself (rather than failing) if Ollama isn't running or the expected
model isn't installed, so the mocked tests remain the source of truth in
environments without Ollama.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import ollama
import pytest

from services.ollama_client import (
    DEFAULT_NUM_CTX,
    ModelNotFoundError,
    OllamaClient,
    OllamaConnectionError,
    OllamaNotRunningError,
)
from settings import OLLAMA_MAX_RETRIES, OLLAMA_RETRY_BACKOFF_SECONDS, OLLAMA_TIMEOUT

# Model used by the optional integration test, matching Pipeline C in
# PROJECT_PLAN.md (the small text model).
INTEGRATION_TEXT_MODEL = "qwen3:1.7b"


def _fake_list_response(*model_names: str) -> SimpleNamespace:
    """Build a stand-in for ollama.Client.list()'s return value."""
    return SimpleNamespace(models=[SimpleNamespace(model=name) for name in model_names])


@pytest.fixture
def client_with_mock():
    """An OllamaClient whose underlying ollama.Client is a MagicMock."""
    client = OllamaClient(host="http://localhost:11434")
    client._client = MagicMock()
    return client


class TestIsRunning:
    def test_true_when_server_responds(self, client_with_mock):
        client_with_mock._client.list.return_value = _fake_list_response()

        assert client_with_mock.is_running() is True

    def test_false_when_server_unreachable(self, client_with_mock):
        client_with_mock._client.list.side_effect = ConnectionError("refused")

        assert client_with_mock.is_running() is False


class TestListModels:
    def test_returns_installed_model_names(self, client_with_mock):
        client_with_mock._client.list.return_value = _fake_list_response(
            "qwen2.5vl:3b", "qwen3:1.7b"
        )

        assert client_with_mock.list_models() == ["qwen2.5vl:3b", "qwen3:1.7b"]

    def test_raises_not_running_when_server_unreachable(self, client_with_mock):
        client_with_mock._client.list.side_effect = ConnectionError("refused")

        with pytest.raises(OllamaNotRunningError):
            client_with_mock.list_models()

    def test_raises_connection_error_on_timeout(self, client_with_mock, monkeypatch):
        monkeypatch.setattr("services.ollama_client.time.sleep", lambda _seconds: None)
        client_with_mock._client.list.side_effect = httpx.TimeoutException("timed out")

        with pytest.raises(OllamaConnectionError):
            client_with_mock.list_models()


class TestModelExists:
    def test_true_when_model_installed(self, client_with_mock):
        client_with_mock._client.list.return_value = _fake_list_response("qwen3:1.7b")

        assert client_with_mock.model_exists("qwen3:1.7b") is True

    def test_false_when_model_missing(self, client_with_mock):
        client_with_mock._client.list.return_value = _fake_list_response("qwen3:1.7b")

        assert client_with_mock.model_exists("does-not-exist:1b") is False


class TestGenerateText:
    def test_raises_model_not_found_without_calling_generate(self, client_with_mock):
        client_with_mock._client.list.return_value = _fake_list_response("qwen3:1.7b")

        with pytest.raises(ModelNotFoundError):
            client_with_mock.generate_text("does-not-exist:1b", "hello")

        client_with_mock._client.generate.assert_not_called()

    def test_returns_raw_response_unmodified(self, client_with_mock):
        client_with_mock._client.list.return_value = _fake_list_response("qwen3:1.7b")
        sentinel_response = SimpleNamespace(response="raw text output")
        client_with_mock._client.generate.return_value = sentinel_response

        result = client_with_mock.generate_text("qwen3:1.7b", "hello")

        assert result is sentinel_response
        client_with_mock._client.generate.assert_called_once_with(
            model="qwen3:1.7b", prompt="hello", options={"num_ctx": DEFAULT_NUM_CTX}
        )

    def test_raises_not_running_when_server_unreachable(self, client_with_mock):
        client_with_mock._client.list.side_effect = ConnectionError("refused")

        with pytest.raises(OllamaNotRunningError):
            client_with_mock.generate_text("qwen3:1.7b", "hello")


class TestGenerateVision:
    def test_raises_model_not_found_without_calling_generate(self, client_with_mock):
        client_with_mock._client.list.return_value = _fake_list_response("qwen2.5vl:3b")

        with pytest.raises(ModelNotFoundError):
            client_with_mock.generate_vision(
                "does-not-exist:3b", "describe this", ["page_001.png"]
            )

        client_with_mock._client.generate.assert_not_called()

    def test_returns_raw_response_with_images_forwarded(self, client_with_mock):
        client_with_mock._client.list.return_value = _fake_list_response("qwen2.5vl:3b")
        sentinel_response = SimpleNamespace(response="a description")
        client_with_mock._client.generate.return_value = sentinel_response

        result = client_with_mock.generate_vision(
            "qwen2.5vl:3b", "describe this", ["page_001.png", "page_002.png"]
        )

        assert result is sentinel_response
        client_with_mock._client.generate.assert_called_once_with(
            model="qwen2.5vl:3b",
            prompt="describe this",
            images=["page_001.png", "page_002.png"],
            options={"num_ctx": DEFAULT_NUM_CTX},
        )


class TestConfiguration:
    """Milestone 12.5: DPI/timeout/retry settings must be configurable
    via settings.py rather than hardcoded, without changing the public
    constructor's defaults-based behavior."""

    def test_default_timeout_comes_from_settings(self):
        client = OllamaClient()

        assert client.timeout == OLLAMA_TIMEOUT
        assert client._client._client.timeout.connect == OLLAMA_TIMEOUT

    def test_custom_timeout_overrides_the_default(self):
        client = OllamaClient(timeout=42)

        assert client.timeout == 42
        assert client._client._client.timeout.connect == 42

    def test_default_retry_settings_come_from_settings(self):
        client = OllamaClient()

        assert client.max_retries == OLLAMA_MAX_RETRIES
        assert client.retry_backoff_seconds == OLLAMA_RETRY_BACKOFF_SECONDS

    def test_custom_retry_settings_override_the_defaults(self):
        client = OllamaClient(max_retries=1, retry_backoff_seconds=0.5)

        assert client.max_retries == 1
        assert client.retry_backoff_seconds == 0.5


class TestRetryLogic:
    """Retry logic must engage only for OllamaConnectionError (e.g. a
    timeout), never for ModelNotFoundError or a fully-down server."""

    def test_retry_succeeds_on_second_attempt(self, client_with_mock, monkeypatch):
        sleep_calls = []
        monkeypatch.setattr("services.ollama_client.time.sleep", sleep_calls.append)
        client_with_mock._client.list.return_value = _fake_list_response("qwen2.5vl:3b")
        sentinel_response = SimpleNamespace(response="ok")
        client_with_mock._client.generate.side_effect = [
            httpx.TimeoutException("timed out"),
            sentinel_response,
        ]

        result = client_with_mock.generate_text("qwen2.5vl:3b", "hello")

        assert result is sentinel_response
        assert client_with_mock._client.generate.call_count == 2
        assert sleep_calls == [5.0]
        assert client_with_mock.last_retry_count == 1

    def test_retry_succeeds_on_third_attempt(self, client_with_mock, monkeypatch):
        sleep_calls = []
        monkeypatch.setattr("services.ollama_client.time.sleep", sleep_calls.append)
        client_with_mock._client.list.return_value = _fake_list_response("qwen2.5vl:3b")
        sentinel_response = SimpleNamespace(response="ok")
        client_with_mock._client.generate.side_effect = [
            httpx.TimeoutException("timed out"),
            httpx.TimeoutException("timed out"),
            sentinel_response,
        ]

        result = client_with_mock.generate_text("qwen2.5vl:3b", "hello")

        assert result is sentinel_response
        assert client_with_mock._client.generate.call_count == 3
        assert sleep_calls == [5.0, 10.0]
        assert client_with_mock.last_retry_count == 2

    def test_retry_ultimately_fails_after_max_retries(self, client_with_mock, monkeypatch):
        sleep_calls = []
        monkeypatch.setattr("services.ollama_client.time.sleep", sleep_calls.append)
        client_with_mock._client.list.return_value = _fake_list_response("qwen2.5vl:3b")
        client_with_mock._client.generate.side_effect = httpx.TimeoutException("timed out")

        with pytest.raises(OllamaConnectionError):
            client_with_mock.generate_text("qwen2.5vl:3b", "hello")

        # Initial attempt + 3 retries (the default OLLAMA_MAX_RETRIES) = 4 calls.
        assert client_with_mock._client.generate.call_count == 4
        assert sleep_calls == [5.0, 10.0, 20.0]

    def test_no_retry_for_model_not_found(self, client_with_mock, monkeypatch):
        sleep_mock = MagicMock()
        monkeypatch.setattr("services.ollama_client.time.sleep", sleep_mock)
        client_with_mock._client.list.return_value = _fake_list_response("qwen3:1.7b")

        with pytest.raises(ModelNotFoundError):
            client_with_mock.generate_text("does-not-exist:1b", "hello")

        client_with_mock._client.generate.assert_not_called()
        sleep_mock.assert_not_called()

    def test_no_retry_when_server_is_not_running(self, client_with_mock, monkeypatch):
        sleep_mock = MagicMock()
        monkeypatch.setattr("services.ollama_client.time.sleep", sleep_mock)
        client_with_mock._client.list.return_value = _fake_list_response("qwen2.5vl:3b")
        client_with_mock._client.generate.side_effect = ConnectionError("refused")

        with pytest.raises(OllamaNotRunningError):
            client_with_mock.generate_text("qwen2.5vl:3b", "hello")

        client_with_mock._client.generate.assert_called_once()
        sleep_mock.assert_not_called()

    def test_retry_log_messages_include_attempt_wait_and_reason(
        self, client_with_mock, monkeypatch, caplog
    ):
        monkeypatch.setattr("services.ollama_client.time.sleep", lambda _seconds: None)
        client_with_mock._client.list.return_value = _fake_list_response("qwen2.5vl:3b")
        sentinel_response = SimpleNamespace(response="ok")
        client_with_mock._client.generate.side_effect = [
            httpx.TimeoutException("timed out"),
            sentinel_response,
        ]

        with caplog.at_level("WARNING", logger="services.ollama_client"):
            client_with_mock.generate_text("qwen2.5vl:3b", "hello")

        retry_messages = [record.message for record in caplog.records if "Retry" in record.message]
        assert len(retry_messages) == 1
        assert "Retry 1/3 after 5" in retry_messages[0]
        assert "timed out" in retry_messages[0]


@pytest.mark.integration
def test_real_ollama_generate_text_integration():
    """Optional integration test against a real local Ollama server.

    Skips itself if Ollama isn't running or the expected model isn't
    installed, rather than failing, so this suite still passes in
    environments without a local Ollama setup.

    The underlying ollama.Client.generate is wrapped with a spy (not
    replaced — `wraps` still forwards through to the real call) so this
    test can confirm the num_ctx option actually reaches the real client
    call, not just that our own code intends to send it.
    """
    client = OllamaClient()

    if not client.is_running():
        pytest.skip("Ollama is not running locally; skipping integration test.")
    if not client.model_exists(INTEGRATION_TEXT_MODEL):
        pytest.skip(f"Model '{INTEGRATION_TEXT_MODEL}' is not installed; skipping.")

    with patch.object(
        client._client, "generate", wraps=client._client.generate
    ) as generate_spy:
        response = client.generate_text(
            INTEGRATION_TEXT_MODEL, "Reply with exactly one word: OK"
        )

    assert isinstance(response, ollama.GenerateResponse)
    assert response.response
    assert generate_spy.call_args.kwargs["options"] == {"num_ctx": DEFAULT_NUM_CTX}
