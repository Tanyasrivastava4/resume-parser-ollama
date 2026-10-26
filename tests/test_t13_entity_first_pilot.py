"""
tests/test_t13_entity_first_pilot.py

Milestone T13/Phase 2C -- focused, narrowly-scoped tests for the
research-only entity-first pilot code
(scripts/_milestone_t13_entity_first_pilot.py). These are NEW
diagnostic tests, distinct from the project's production regression
suite -- they test only the T13 research harness's own deterministic
helper functions, never invoking Ollama, never touching
WorkExperienceExtractor's production behavior, WorkExperienceLocator,
or any production prompt/schema file.
"""
from __future__ import annotations

import pytest

from scripts._milestone_t13_entity_first_pilot import (
    _extract_json,
    check_entity_fixation_violation,
)


class TestExtractJson:
    """Stage A/B raw-response parsing."""

    def test_plain_json(self):
        assert _extract_json('{"entities": []}') == {"entities": []}

    def test_json_wrapped_in_code_fence(self):
        text = '```json\n{"entities": [{"ordinal": 1, "company": "Acme"}]}\n```'
        assert _extract_json(text) == {"entities": [{"ordinal": 1, "company": "Acme"}]}

    def test_json_wrapped_in_bare_code_fence(self):
        text = '```\n{"work_experience": []}\n```'
        assert _extract_json(text) == {"work_experience": []}

    def test_malformed_json_raises(self):
        with pytest.raises(Exception):
            _extract_json("not json at all")

    def test_missing_fields_in_entity_are_preserved_as_given(self):
        # The parser itself does not fill in defaults -- that is the
        # caller's responsibility. Confirms no silent field injection.
        text = '{"entities": [{"ordinal": 1, "company": "Acme"}]}'
        result = _extract_json(text)
        assert "designation" not in result["entities"][0]


class TestEntityFixationViolation:
    """Stage B must never create, delete, merge, or split entities
    relative to Stage A's own fixed list -- this is verified in code,
    not merely assumed from the prompt's own instruction."""

    def test_identical_count_and_matching_companies_no_violation(self):
        stage_a = [{"ordinal": 1, "company": "Acme Corp"}, {"ordinal": 2, "company": "Widget Inc"}]
        returned = [{"company": "Acme Corp"}, {"company": "Widget Inc"}]
        assert check_entity_fixation_violation(stage_a, returned) is False

    def test_fuzzy_matching_company_variant_no_violation(self):
        stage_a = [{"ordinal": 1, "company": "Acme Corp"}]
        returned = [{"company": "Acme Corp (Subsidiary)"}]
        assert check_entity_fixation_violation(stage_a, returned) is False

    def test_extra_entity_is_a_violation(self):
        stage_a = [{"ordinal": 1, "company": "Acme Corp"}]
        returned = [{"company": "Acme Corp"}, {"company": "Extra Fabricated Co"}]
        assert check_entity_fixation_violation(stage_a, returned) is True

    def test_missing_entity_is_a_violation(self):
        stage_a = [{"ordinal": 1, "company": "Acme Corp"}, {"ordinal": 2, "company": "Widget Inc"}]
        returned = [{"company": "Acme Corp"}]
        assert check_entity_fixation_violation(stage_a, returned) is True

    def test_reordered_or_swapped_company_identity_is_a_violation(self):
        stage_a = [{"ordinal": 1, "company": "Acme Corp"}, {"ordinal": 2, "company": "Widget Inc"}]
        returned = [{"company": "Widget Inc"}, {"company": "Acme Corp"}]
        assert check_entity_fixation_violation(stage_a, returned) is True

    def test_empty_stage_a_and_empty_returned_no_violation(self):
        assert check_entity_fixation_violation([], []) is False

    def test_null_company_on_both_sides_treated_as_no_signal_not_a_violation(self):
        # Neither side has anything to compare -- absence of a signal
        # is not itself a violation (matches production's own
        # "never guess" convention elsewhere in this codebase).
        stage_a = [{"ordinal": 1, "company": None}]
        returned = [{"company": None}]
        assert check_entity_fixation_violation(stage_a, returned) is False

    def test_non_dict_returned_job_does_not_crash_and_is_treated_as_unmatched(self):
        stage_a = [{"ordinal": 1, "company": "Acme Corp"}]
        returned = ["not a dict"]
        assert check_entity_fixation_violation(stage_a, returned) is True
