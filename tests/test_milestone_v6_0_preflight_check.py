"""
tests/test_milestone_v6_0_preflight_check.py

Tests for scripts/_milestone_v6_0_preflight_check.py (Milestone V6.0).

This script's entire purpose is to inspect real repository state (git-
tracked source files, the real GT/PDF manifests, `ollama list`, and a
fresh re-derivation of the V5.9 signal population) -- it is not a pure
function of synthetic inputs, so the most meaningful test of it is a
real, read-only run against the actual repository, exactly as a future
V6.1 execution session would perform it. That integration-style check
is exercised here (test_main_passes_in_current_environment); a small
number of pure-function unit tests cover the parts that ARE pure
functions (sha256_of).

No LLM calls. No production file is modified by any test here.
"""
import hashlib
import os

import pytest

from scripts._milestone_v6_0_preflight_check import (
    main,
    sha256_of,
)


class TestSha256Of:
    def test_matches_hashlib_directly(self, tmp_path):
        p = tmp_path / "sample.txt"
        p.write_bytes(b"hello world")
        assert sha256_of(p) == hashlib.sha256(b"hello world").hexdigest()

    def test_differs_for_different_content(self, tmp_path):
        p1 = tmp_path / "a.txt"
        p2 = tmp_path / "b.txt"
        p1.write_bytes(b"content A")
        p2.write_bytes(b"content B")
        assert sha256_of(p1) != sha256_of(p2)


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_V6_PREFLIGHT") != "1",
    reason=(
        "Opt-in V6.x preflight (set RUN_V6_PREFLIGHT=1). This test asserts "
        "the frozen Milestone V6.0 repository state (production-source "
        "manifest, qwen3:1.7b AND qwen3:8b installed). Milestones T3/T5 "
        "deliberately changed pipeline/parser_pipeline.py and "
        "schemas/resume_schema_v1.json after that freeze, and the Skills "
        "V7.1 execution that would re-derive the protocol is deferred "
        "(decision D4, 2026-10-17), so on every post-T3 checkout this "
        "preflight correctly reports drift. It talks to `ollama list` and "
        "is therefore also an integration test, not a deterministic one."
    ),
)
class TestMainIntegration:
    def test_main_passes_in_current_environment(self):
        """Real, read-only run against the actual repository -- exactly
        what a future V6.1 execution session would run first. Zero LLM
        calls (only `ollama list`, a metadata query, and read-only file
        hashing / GT-manifest / signal re-derivation)."""
        assert main() == 0
