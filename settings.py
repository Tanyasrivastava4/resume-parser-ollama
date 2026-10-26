"""
settings.py

Environment-driven settings for the Resume AI Benchmark project.

Unlike config.py (static, shared, safe to commit), this module holds values
that vary per machine or per developer and may be sensitive: model
endpoints, API keys, and toggles. Values are read from environment
variables (optionally loaded from a local .env file) with sane defaults for
local development, and nothing here should be committed with real secrets.

Nothing in this file executes parsing logic. It only exposes settings that
other modules will read later, once implemented.
"""

import os

# Load a local .env file if python-dotenv is installed and a .env exists.
# This is optional tooling convenience, not a hard dependency of the app.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _get_bool(name: str, default: bool) -> bool:
    """Parse an environment variable as a boolean, falling back to default."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# Host for the local Ollama server, used by services/ollama_client.py to
# run every self-hosted model (Qwen2.5-VL, Qwen 1.7B, etc.) via one
# consistent API instead of separate per-model endpoints.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Default request timeout (in seconds) for calls to the Ollama server.
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "900"))

# Maximum number of retries for a transient Ollama connection failure
# (OllamaConnectionError), not counting the initial attempt. Failures that
# aren't transient connection issues (e.g. a missing model or prompt) are
# never retried.
OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))

# Base delay, in seconds, before the first retry. Each subsequent retry
# doubles the previous wait (exponential backoff), e.g. 5, 10, 20 for the
# default OLLAMA_MAX_RETRIES of 3.
OLLAMA_RETRY_BACKOFF_SECONDS = float(os.getenv("OLLAMA_RETRY_BACKOFF_SECONDS", "5"))

# Single, shared text model used by JsonDraftGenerator and every module
# extractor (PersonalInformationExtractor, ContactInformationExtractor,
# SkillsExtractor, ProfessionalSummaryExtractor, LanguagesExtractor,
# WorkExperienceExtractor, EducationExtractor) for the markdown-to-JSON
# stage of the pipeline (Milestone 23.3B). Each of those modules imports
# this one value (`from settings import DEFAULT_TEXT_MODEL`) instead of
# hardcoding its own local copy of the model name, so changing it here —
# or overriding it via the DEFAULT_TEXT_MODEL environment variable —
# switches the entire parser's text model in one place. This does not
# affect the vision model used for OCR/markdown generation (see
# QWEN_VL_ENDPOINT below and services/document_extractor.py), which is a
# separate, earlier pipeline stage with its own model selection, nor does
# it affect a model_name a caller explicitly passes to an individual
# extract()/generate() call, which still takes precedence over this
# default exactly as it already did before this setting existed.
DEFAULT_TEXT_MODEL = os.getenv("DEFAULT_TEXT_MODEL", "qwen3:1.7b")

# DPI used by services/pdf_loader.py to rasterize PDF pages to images.
# Higher values produce larger, sharper images at the cost of disk space
# and processing time.
PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "200"))

# OCR strategy used by services/document_extractor.py to process each
# page. "single_page" (the default) sends the whole page image to the
# vision model in one request, unchanged since Milestone 10. "split_columns"
# (Milestone 15C.3, experimental) additionally splits each non-blank page
# into left/right full-height halves and sends each half as its own
# request, for multi-column layouts where the vision model drops an
# entire column when given the full page in one shot (see
# docs/SIDEBAR_RECOGNITION_INVESTIGATION.md). The default must remain
# "single_page" so existing behavior is unaffected unless explicitly
# opted into.
OCR_STRATEGY = os.getenv("OCR_STRATEGY", "single_page")

# Endpoint for the self-hosted Qwen2.5-VL model used by Pipelines A and B.
QWEN_VL_ENDPOINT = os.getenv("QWEN_VL_ENDPOINT", "http://localhost:8000")

# Endpoint for the smaller Qwen 1.7B text model used by Pipeline C.
QWEN_SMALL_ENDPOINT = os.getenv("QWEN_SMALL_ENDPOINT", "http://localhost:8001")

# Endpoint / config for Docling, used by Pipeline C to convert PDF to
# markdown before the small model turns it into JSON.
DOCLING_ENDPOINT = os.getenv("DOCLING_ENDPOINT", "http://localhost:8002")

# Optional API key for any hosted model fallback. Left blank for a fully
# self-hosted setup.
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")

# Logging verbosity for the whole application.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Whether to run in debug mode (more verbose logging, no cleanup of
# intermediate files, etc.).
DEBUG = _get_bool("DEBUG", False)
