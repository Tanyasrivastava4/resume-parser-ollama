"""
tests/test_review_result_contract.py

Tests for evaluation/review_result_contract.py (Milestone V5.2; V5.3
adds review_report_output_path()/write_review_result_contract() at the
bottom of this file). Pure in-memory logic against a minimal fake
ParserPipelineResult stand-in -- no Ollama, no real pipeline run.
Mirrors tests/test_human_review_evidence_report.py's own structure.
TestPersistence (V5.3) is the only class here that touches the real
filesystem, always via tmp_path -- never a real project directory.
"""

import copy
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytest

from evaluation.review_result_contract import (
    CONTRACT_VERSION,
    MODULES_WITHOUT_REVIEW_EVIDENCE,
    REVIEW_REPORTS_SUBDIR,
    build_review_result_contract,
    deserialize_review_result_contract,
    review_report_output_path,
    serialize_review_result_contract,
    write_review_result_contract,
)
from services.evidence_record import (
    LanguagesEvidenceRecord,
    LanguageValueEvidence,
    LocationEvidence,
    NameCompletenessEvidence,
    PersonalInformationEvidenceRecord,
)


@dataclass
class _FakeResult:
    """A minimal stand-in for ParserPipelineResult -- only the
    attributes build_review_result_contract() actually reads
    (merged_json, plus the six evidence attributes
    build_resume_evidence_report() reads)."""
    merged_json: Dict[str, Any]
    personal_information_evidence: Optional[Any] = None
    contact_information_evidence: Optional[Any] = None
    professional_summary_evidence: Optional[Any] = None
    languages_evidence: Optional[Any] = None
    work_experience_evidence: Optional[Any] = None
    education_evidence: Optional[Any] = None


def _sample_merged_json():
    return {
        "personal_information": {"full_name": "Ada Lovelace", "first_name": "Ada",
                                  "last_name": "Lovelace", "current_location": "London"},
        "contact_information": {"email": ["ada@example.com"], "phone": [], "linkedin": None,
                                 "github": None, "portfolio": None},
        "skills": ["Mathematics"],
        "professional_summary": "Mathematician and writer.",
        "languages": ["English"],
        "work_experience": [],
        "education": [],
    }


def _pi_evidence():
    return PersonalInformationEvidenceRecord(
        schema_valid=True, stage1_route=None,
        name_completeness=NameCompletenessEvidence(
            full_name="Ada Lovelace", first_name="Ada", last_name="Lovelace",
            checked=True, missing_tokens=[], evidence_summary="NAME_TOKENS_FULLY_REPRESENTED",
        ),
        location=LocationEvidence(
            extracted_value="London", source_text_checked=True,
            verbatim_in_source=True, evidence_summary="SOURCE_TEXT_VERBATIM_MATCH",
        ),
    )


def _lang_evidence():
    return LanguagesEvidenceRecord(
        schema_valid=True, stage1_route=None,
        values=[LanguageValueEvidence(value="English", source_text_checked=True,
                                       source_text_present=True, evidence_summary="SOURCE_TEXT_PRESENT")],
    )


class TestOldConsumerPathUnaffected:
    """The old consumer path -- reading result.merged_json directly,
    never calling this module at all -- must be completely unaffected
    by this module's existence."""

    def test_merged_json_unchanged_after_building_contract(self):
        merged = _sample_merged_json()
        result = _FakeResult(merged_json=merged, personal_information_evidence=_pi_evidence())
        snapshot = copy.deepcopy(merged)

        build_review_result_contract("001", "stem", result)

        assert result.merged_json == snapshot

    def test_building_contract_is_opt_in(self):
        """A consumer that never imports or calls anything from this
        module sees no behavior change at all -- trivially true since
        nothing in this module is invoked unless explicitly called, but
        verified here for documentation/regression purposes."""
        merged = _sample_merged_json()
        result = _FakeResult(merged_json=merged)
        # No call to build_review_result_contract() at all.
        assert result.merged_json == merged


class TestNewConsumerPath:
    def test_parsed_result_equals_merged_json_exactly(self):
        merged = _sample_merged_json()
        result = _FakeResult(merged_json=merged)
        contract = build_review_result_contract("001", "stem", result)
        assert contract["parsed_result"] == merged
        assert contract["parsed_result"] is result.merged_json  # same reference, never copied/altered

    def test_review_report_is_v51s_own_output_unmodified(self):
        result = _FakeResult(merged_json=_sample_merged_json(), personal_information_evidence=_pi_evidence())
        contract = build_review_result_contract("001", "stem", result)

        from evaluation.human_review_evidence_report import build_resume_evidence_report
        expected = build_resume_evidence_report("001", "stem", result)
        assert contract["review_report"] == expected

    def test_contract_version_present(self):
        result = _FakeResult(merged_json=_sample_merged_json())
        contract = build_review_result_contract("001", "stem", result)
        assert contract["contract_version"] == CONTRACT_VERSION == "1.0"

    def test_parsed_result_and_review_report_are_separate_top_level_keys(self):
        """Evidence must never be embedded inside parsed_result."""
        result = _FakeResult(merged_json=_sample_merged_json(), personal_information_evidence=_pi_evidence())
        contract = build_review_result_contract("001", "stem", result)
        assert "review_report" not in contract["parsed_result"]
        assert set(contract.keys()) == {
            "contract_version", "resume_id", "stem", "parsed_result",
            "review_report", "modules_without_review_evidence",
        }


class TestSkillsHonestRepresentation:
    def test_skills_present_in_parsed_result(self):
        result = _FakeResult(merged_json=_sample_merged_json())
        contract = build_review_result_contract("001", "stem", result)
        assert "skills" in contract["parsed_result"]
        assert contract["parsed_result"]["skills"] == ["Mathematics"]

    def test_skills_absent_from_review_report(self):
        result = _FakeResult(merged_json=_sample_merged_json())
        contract = build_review_result_contract("001", "stem", result)
        assert "skills" not in contract["review_report"]["modules"]

    def test_skills_listed_in_modules_without_review_evidence(self):
        result = _FakeResult(merged_json=_sample_merged_json())
        contract = build_review_result_contract("001", "stem", result)
        assert "skills" in contract["modules_without_review_evidence"]
        assert MODULES_WITHOUT_REVIEW_EVIDENCE == ("skills",)


class TestAllSixEvidenceModulesRepresented:
    def test_all_six_modules_present_when_evidence_available(self):
        result = _FakeResult(
            merged_json=_sample_merged_json(),
            personal_information_evidence=_pi_evidence(),
            languages_evidence=_lang_evidence(),
        )
        contract = build_review_result_contract("001", "stem", result)
        assert set(contract["review_report"]["modules"].keys()) == {
            "personal_information", "contact_information", "professional_summary",
            "languages", "work_experience", "education",
        }

    def test_all_none_evidence_still_produces_placeholders_not_omission(self):
        result = _FakeResult(merged_json=_sample_merged_json())
        contract = build_review_result_contract("001", "stem", result)
        for module_key, entries in contract["review_report"]["modules"].items():
            assert len(entries) >= 1  # never silently empty/omitted


class TestSerializationRoundTrip:
    def test_json_round_trip_is_exact(self):
        result = _FakeResult(merged_json=_sample_merged_json(), personal_information_evidence=_pi_evidence())
        contract = build_review_result_contract("001", "stem", result)

        payload = serialize_review_result_contract(contract)
        round_tripped = deserialize_review_result_contract(payload)

        assert round_tripped == contract

    def test_serialize_produces_valid_json(self):
        result = _FakeResult(merged_json=_sample_merged_json())
        contract = build_review_result_contract("001", "stem", result)
        payload = serialize_review_result_contract(contract)
        assert json.loads(payload) == contract  # standard json module agrees too

    def test_serialize_does_not_mutate_contract(self):
        result = _FakeResult(merged_json=_sample_merged_json(), personal_information_evidence=_pi_evidence())
        contract = build_review_result_contract("001", "stem", result)
        snapshot = copy.deepcopy(contract)
        serialize_review_result_contract(contract)
        assert contract == snapshot

    def test_no_evidence_state_strengthened_by_round_trip(self):
        """A NAME_TOKENS_MISSING / NO_VERBATIM_MATCH_FOUND state must
        remain byte-for-byte identical text after a JSON round trip --
        never silently upgraded to a stronger-sounding label."""
        evidence = PersonalInformationEvidenceRecord(
            schema_valid=True, stage1_route=None,
            name_completeness=NameCompletenessEvidence(
                full_name="Ada Lovelace", first_name="Ada", last_name=None,
                checked=True, missing_tokens=["lovelace"], evidence_summary="NAME_TOKENS_MISSING",
            ),
            location=LocationEvidence(
                extracted_value="Nowhere", source_text_checked=True,
                verbatim_in_source=False, evidence_summary="NO_VERBATIM_MATCH_FOUND",
            ),
        )
        result = _FakeResult(merged_json=_sample_merged_json(), personal_information_evidence=evidence)
        contract = build_review_result_contract("001", "stem", result)
        payload = serialize_review_result_contract(contract)
        round_tripped = deserialize_review_result_contract(payload)

        pi_entries = round_tripped["review_report"]["modules"]["personal_information"]
        name_entry = next(e for e in pi_entries if e["item_id"] == "name_completeness")
        loc_entry = next(e for e in pi_entries if e["item_id"] == "location")
        assert name_entry["native_evidence_state"]["evidence_summary"] == "NAME_TOKENS_MISSING"
        assert loc_entry["native_evidence_state"]["evidence_summary"] == "NO_VERBATIM_MATCH_FOUND"


class TestDeterminism:
    def test_building_contract_twice_is_identical(self):
        result = _FakeResult(merged_json=_sample_merged_json(), personal_information_evidence=_pi_evidence())
        contract1 = build_review_result_contract("001", "stem", result)
        contract2 = build_review_result_contract("001", "stem", result)
        assert contract1 == contract2

    def test_serialized_payload_is_identical_across_calls(self):
        result = _FakeResult(merged_json=_sample_merged_json(), personal_information_evidence=_pi_evidence())
        payload1 = serialize_review_result_contract(build_review_result_contract("001", "stem", result))
        payload2 = serialize_review_result_contract(build_review_result_contract("001", "stem", result))
        assert payload1 == payload2


class TestNoMutationOfInputResult:
    def test_does_not_mutate_result_object(self):
        result = _FakeResult(merged_json=_sample_merged_json(), personal_information_evidence=_pi_evidence())
        snapshot = copy.deepcopy(result)
        build_review_result_contract("001", "stem", result)
        assert result == snapshot


class TestReviewReportOutputPath:
    """Milestone V5.3."""

    def test_mirrors_json_final_naming_convention(self, tmp_path):
        path = review_report_output_path(tmp_path, "my_resume")
        assert path == tmp_path / REVIEW_REPORTS_SUBDIR / "my_resume.json"
        assert REVIEW_REPORTS_SUBDIR == "review_reports"

    def test_does_not_create_anything_on_disk(self, tmp_path):
        path = review_report_output_path(tmp_path, "my_resume")
        assert not path.exists()
        assert not path.parent.exists()

    def test_accepts_string_output_root(self, tmp_path):
        path = review_report_output_path(str(tmp_path), "my_resume")
        assert path == tmp_path / REVIEW_REPORTS_SUBDIR / "my_resume.json"


class TestWriteReviewResultContract:
    """Milestone V5.3. Real tmp_path file I/O -- never a real project
    directory."""

    def test_writes_a_readable_json_file_at_the_expected_path(self, tmp_path):
        result = _FakeResult(merged_json=_sample_merged_json(), personal_information_evidence=_pi_evidence())

        written_path = write_review_result_contract("001", "resume_stem", result, tmp_path)

        assert written_path == tmp_path / "review_reports" / "resume_stem.json"
        assert written_path.is_file()
        on_disk = json.loads(written_path.read_text(encoding="utf-8"))
        assert on_disk == build_review_result_contract("001", "resume_stem", result)

    def test_creates_the_review_reports_directory_if_missing(self, tmp_path):
        assert not (tmp_path / "review_reports").exists()
        result = _FakeResult(merged_json=_sample_merged_json())

        write_review_result_contract("001", "stem", result, tmp_path)

        assert (tmp_path / "review_reports").is_dir()

    def test_never_touches_json_final_or_modules_directories(self, tmp_path):
        """The persisted artifact must land only under review_reports/
        -- json_final/ and modules/ (ParserPipeline's own output) are
        never created or touched by this function."""
        result = _FakeResult(merged_json=_sample_merged_json())

        write_review_result_contract("001", "stem", result, tmp_path)

        assert not (tmp_path / "json_final").exists()
        assert not (tmp_path / "modules").exists()

    def test_does_not_mutate_result_object(self, tmp_path):
        result = _FakeResult(merged_json=_sample_merged_json(), personal_information_evidence=_pi_evidence())
        snapshot = copy.deepcopy(result)

        write_review_result_contract("001", "stem", result, tmp_path)

        assert result == snapshot

    def test_repeated_writes_are_deterministic(self, tmp_path):
        result = _FakeResult(merged_json=_sample_merged_json(), personal_information_evidence=_pi_evidence())

        path1 = write_review_result_contract("001", "stem", result, tmp_path)
        payload1 = path1.read_text(encoding="utf-8")
        path2 = write_review_result_contract("001", "stem", result, tmp_path)
        payload2 = path2.read_text(encoding="utf-8")

        assert path1 == path2
        assert payload1 == payload2

    def test_accepts_string_output_root(self, tmp_path):
        result = _FakeResult(merged_json=_sample_merged_json())
        written_path = write_review_result_contract("001", "stem", result, str(tmp_path))
        assert written_path.is_file()

    def test_raises_oserror_for_an_unwritable_target(self, tmp_path):
        """A file (not a directory) in the way of review_reports/ makes
        the target unwritable -- write_review_result_contract() must
        propagate this, not silently swallow it (that is the caller's
        responsibility, per this function's own docstring -- see
        scripts/run_parser_pipeline.py's own try/except around this
        exact call)."""
        (tmp_path / "review_reports").write_text("not a directory", encoding="utf-8")
        result = _FakeResult(merged_json=_sample_merged_json())

        with pytest.raises(OSError):
            write_review_result_contract("001", "stem", result, tmp_path)
