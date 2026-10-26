"""
tests/test_milestone_t2_phase0_entity_first_prototype.py

Milestone T2 / Phase 0 -- focused tests for the deterministic
(non-LLM-call) parsing helper in the entity-first prototype harness.
Does NOT call Ollama -- the harness's own call-making functions
(run_baseline, run_entity_first, run_slot) are exercised only via the
16 real checkpoints already produced and evaluated in this milestone's
own report, not re-invoked here.
"""
from __future__ import annotations

import pytest

from scripts._milestone_t2_phase0_entity_first_prototype import _extract_json


class TestExtractJson:
    def test_plain_json_parses(self):
        assert _extract_json('{"entities": ["A", "B"]}') == {"entities": ["A", "B"]}

    def test_strips_json_code_fence(self):
        text = '```json\n{"entities": ["A"]}\n```'
        assert _extract_json(text) == {"entities": ["A"]}

    def test_strips_bare_code_fence(self):
        text = '```\n{"entities": []}\n```'
        assert _extract_json(text) == {"entities": []}

    def test_invalid_json_raises(self):
        with pytest.raises(Exception):
            _extract_json("not json at all")
