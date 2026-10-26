"""
tests/test_t21b_candidate_decision_contract.py

Milestone T21B/Phase 2I -- verifies the candidate-DECISION-ONLY
contract's independent validation logic. No confirmed_entities concept
exists in this contract at all; a response that includes one is itself
flagged as a violation.

Zero LLM calls -- pure unit tests of validate_response() against
hand-constructed responses.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_t21b_adjudicate import (  # noqa: E402
    validate_response,
    _extract_json_object,
    VALID_DECISIONS,
    VALID_CLASSIFICATIONS,
)

VALID_IDS = {"C0", "C1", "C2"}


def _base_decisions():
    return [
        {"candidate_id": "C0", "decision": "DISCARD", "classification": "OTHER_NON_EXPERIENCE"},
        {"candidate_id": "C1", "decision": "KEEP", "classification": "WORK_EXPERIENCE"},
        {"candidate_id": "C2", "decision": "CLASSIFY", "classification": "PROJECT"},
    ]


class TestVocabularyIsExactlyAsSpecified:
    def test_decision_vocabulary_has_no_merge(self):
        assert VALID_DECISIONS == {"KEEP", "DISCARD", "CLASSIFY"}

    def test_classification_vocabulary_includes_education(self):
        assert VALID_CLASSIFICATIONS == {"WORK_EXPERIENCE", "PROJECT", "EDUCATION", "OTHER_NON_EXPERIENCE"}


class TestValidResponseAccepted:
    def test_fully_valid_response_has_no_errors(self):
        parsed = {"candidate_decisions": _base_decisions()}
        assert validate_response(parsed, VALID_IDS) == []


class TestNoEntityAssemblyAllowed:
    def test_confirmed_entities_field_is_itself_a_violation(self):
        parsed = {
            "candidate_decisions": _base_decisions(),
            "confirmed_entities": [{"entity_id": "E1", "candidate_ids": ["C1"]}],
        }
        errors = validate_response(parsed, VALID_IDS)
        assert any("confirmed_entities" in e for e in errors)


class TestInventionRejected:
    def test_invented_candidate_id_is_rejected(self):
        decisions = _base_decisions()
        decisions.append({"candidate_id": "C99", "decision": "DISCARD", "classification": None})
        parsed = {"candidate_decisions": decisions}
        errors = validate_response(parsed, VALID_IDS)
        assert any("INVENTED" in e and "C99" in e for e in errors)


class TestCompletenessRejected:
    def test_missing_decision_is_flagged(self):
        decisions = _base_decisions()[:-1]
        parsed = {"candidate_decisions": decisions}
        errors = validate_response(parsed, VALID_IDS)
        assert any("MISSING" in e and "C2" in e for e in errors)

    def test_duplicate_decision_is_rejected(self):
        decisions = _base_decisions()
        decisions.append(dict(decisions[0]))
        parsed = {"candidate_decisions": decisions}
        errors = validate_response(parsed, VALID_IDS)
        assert any("DUPLICATE" in e and "C0" in e for e in errors)


class TestUnsupportedValuesRejected:
    def test_unsupported_decision_rejected(self):
        decisions = _base_decisions()
        decisions[0]["decision"] = "MERGE"  # explicitly forbidden in T21B
        parsed = {"candidate_decisions": decisions}
        errors = validate_response(parsed, VALID_IDS)
        assert any("unsupported decision" in e for e in errors)

    def test_unsupported_classification_rejected(self):
        decisions = _base_decisions()
        decisions[0]["classification"] = "HOBBY"
        parsed = {"candidate_decisions": decisions}
        errors = validate_response(parsed, VALID_IDS)
        assert any("unsupported classification" in e for e in errors)

    def test_non_dict_response_rejected(self):
        assert validate_response("not a dict", VALID_IDS) == ["response is not a JSON object"]


class TestJsonExtraction:
    def test_extracts_plain_json(self):
        assert _extract_json_object('{"a": 1}') == '{"a": 1}'

    def test_extracts_json_after_think_block(self):
        text = "<think>reasoning</think>\n{\"a\": 1}"
        assert _extract_json_object(text) == '{"a": 1}'

    def test_extracts_json_from_fence(self):
        assert _extract_json_object("```json\n{\"a\": 1}\n```") == '{"a": 1}'

    def test_returns_none_when_absent(self):
        assert _extract_json_object("no json") is None
