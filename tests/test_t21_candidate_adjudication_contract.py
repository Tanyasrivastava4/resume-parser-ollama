"""
tests/test_t21_candidate_adjudication_contract.py

Milestone T21/Phase 2H -- verifies the candidate-only adjudication
contract's INDEPENDENT validation logic (validate_response), which is
what T21 relies on to prove candidate IDs are never invented, every
confirmed entity references existing candidates, merge groups reference
existing candidates, unrepresentable state is handled, and malformed/
duplicate/unsupported output is rejected.

Zero LLM calls -- these are pure unit tests of the validation function
against hand-constructed responses, not live model calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_t21_adjudicate import (  # noqa: E402
    validate_response,
    _extract_json_object,
)

VALID_IDS = {"C0", "C1", "C2", "C3"}


def _base_decisions():
    return [
        {"candidate_id": "C0", "decision": "KEEP", "classification": "WORK_EXPERIENCE", "merge_group": None},
        {"candidate_id": "C1", "decision": "DISCARD", "classification": None, "merge_group": None},
        {"candidate_id": "C2", "decision": "DISCARD", "classification": None, "merge_group": None},
        {"candidate_id": "C3", "decision": "DISCARD", "classification": None, "merge_group": None},
    ]


class TestValidResponseAccepted:
    def test_fully_valid_response_has_no_errors(self):
        parsed = {
            "candidate_decisions": _base_decisions(),
            "confirmed_entities": [
                {"entity_id": "E1", "candidate_ids": ["C0"], "classification": "WORK_EXPERIENCE"}
            ],
            "unrepresentable_by_candidates": False,
        }
        assert validate_response(parsed, VALID_IDS) == []

    def test_unrepresentable_state_is_accepted_without_further_checks(self):
        parsed = {"unrepresentable_by_candidates": True}
        assert validate_response(parsed, VALID_IDS) == []


class TestInventionRejected:
    def test_invented_candidate_id_in_decisions_is_rejected(self):
        decisions = _base_decisions()
        decisions.append({"candidate_id": "C99", "decision": "DISCARD", "classification": None, "merge_group": None})
        parsed = {"candidate_decisions": decisions, "confirmed_entities": []}
        errors = validate_response(parsed, VALID_IDS)
        assert any("INVENTED" in e and "C99" in e for e in errors)

    def test_invented_candidate_id_in_confirmed_entities_is_rejected(self):
        parsed = {
            "candidate_decisions": _base_decisions(),
            "confirmed_entities": [
                {"entity_id": "E1", "candidate_ids": ["C0", "C_MADE_UP"], "classification": "WORK_EXPERIENCE"}
            ],
        }
        errors = validate_response(parsed, VALID_IDS)
        assert any("INVENTED" in e and "C_MADE_UP" in e for e in errors)

    def test_invented_candidate_id_in_merge_group_is_rejected(self):
        decisions = _base_decisions()
        decisions[0]["merge_group"] = ["C0", "C_NOT_REAL"]
        parsed = {"candidate_decisions": decisions, "confirmed_entities": []}
        errors = validate_response(parsed, VALID_IDS)
        assert any("INVENTED" in e and "C_NOT_REAL" in e for e in errors)


class TestSchemaViolationsRejected:
    def test_missing_decision_for_a_candidate_is_flagged(self):
        decisions = _base_decisions()[:-1]  # drop C3
        parsed = {"candidate_decisions": decisions, "confirmed_entities": []}
        errors = validate_response(parsed, VALID_IDS)
        assert any("missing decisions" in e and "C3" in e for e in errors)

    def test_duplicate_decision_for_same_candidate_is_rejected(self):
        decisions = _base_decisions()
        decisions.append(dict(decisions[0]))
        parsed = {"candidate_decisions": decisions, "confirmed_entities": []}
        errors = validate_response(parsed, VALID_IDS)
        assert any("duplicate" in e.lower() and "C0" in e for e in errors)

    def test_unsupported_decision_value_is_rejected(self):
        decisions = _base_decisions()
        decisions[0]["decision"] = "MAYBE"
        parsed = {"candidate_decisions": decisions, "confirmed_entities": []}
        errors = validate_response(parsed, VALID_IDS)
        assert any("unsupported decision" in e for e in errors)

    def test_unsupported_classification_value_is_rejected(self):
        decisions = _base_decisions()
        decisions[0]["classification"] = "HOBBY"
        parsed = {"candidate_decisions": decisions, "confirmed_entities": []}
        errors = validate_response(parsed, VALID_IDS)
        assert any("unsupported classification" in e for e in errors)

    def test_confirmed_entity_with_no_candidate_ids_is_rejected(self):
        parsed = {
            "candidate_decisions": _base_decisions(),
            "confirmed_entities": [{"entity_id": "E1", "candidate_ids": [], "classification": "WORK_EXPERIENCE"}],
        }
        errors = validate_response(parsed, VALID_IDS)
        assert any("no candidate_ids" in e for e in errors)

    def test_confirmed_entity_missing_classification_is_rejected(self):
        parsed = {
            "candidate_decisions": _base_decisions(),
            "confirmed_entities": [{"entity_id": "E1", "candidate_ids": ["C0"], "classification": None}],
        }
        errors = validate_response(parsed, VALID_IDS)
        assert any("unsupported classification" in e for e in errors)

    def test_non_dict_response_is_rejected(self):
        errors = validate_response("not a dict", VALID_IDS)
        assert errors == ["response is not a JSON object"]

    def test_missing_candidate_decisions_key_is_rejected(self):
        parsed = {"confirmed_entities": []}
        errors = validate_response(parsed, VALID_IDS)
        assert any("candidate_decisions" in e for e in errors)


class TestJsonExtraction:
    def test_extracts_json_from_plain_response(self):
        text = '{"a": 1}'
        assert _extract_json_object(text) == '{"a": 1}'

    def test_extracts_json_after_think_block(self):
        text = "<think>reasoning here</think>\n{\"a\": 1}"
        assert _extract_json_object(text) == '{"a": 1}'

    def test_extracts_json_from_markdown_fence(self):
        text = "```json\n{\"a\": 1}\n```"
        assert _extract_json_object(text) == '{"a": 1}'

    def test_returns_none_when_no_json_present(self):
        assert _extract_json_object("no json here at all") is None
