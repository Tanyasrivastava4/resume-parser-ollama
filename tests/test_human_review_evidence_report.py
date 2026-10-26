"""
tests/test_human_review_evidence_report.py

Tests for evaluation/human_review_evidence_report.py (Milestone V5.1).
Pure in-memory logic against directly-constructed evidence dataclass
objects -- no Ollama, no real pipeline run, no file I/O. Mirrors
tests/test_contact_information_evidence_report.py's own structure
(Milestone V3.9A), extended to all six report-covered modules.
"""

import copy
from dataclasses import dataclass
from typing import Any, Optional

import pytest

from evaluation.human_review_evidence_report import (
    REPORT_MODULES,
    build_contact_information_report_entries,
    build_education_report_entries,
    build_languages_report_entries,
    build_personal_information_report_entries,
    build_professional_summary_report_entries,
    build_resume_evidence_report,
    build_work_experience_report_entries,
    report_to_rows,
)
from services.evidence_record import (
    ContactFieldEvidence,
    ContactInformationEvidenceRecord,
    EntryEvidence,
    LanguagesEvidenceRecord,
    LanguageValueEvidence,
    LocationEvidence,
    ModuleEvidenceRecord,
    NameCompletenessEvidence,
    PersonalInformationEvidenceRecord,
    ProfessionalSummaryEvidence,
    ValidationActionRecord,
)


@dataclass
class _FakeResult:
    """A minimal stand-in for ParserPipelineResult -- only the six
    evidence attributes build_resume_evidence_report() actually reads."""
    personal_information_evidence: Optional[Any] = None
    contact_information_evidence: Optional[Any] = None
    professional_summary_evidence: Optional[Any] = None
    languages_evidence: Optional[Any] = None
    work_experience_evidence: Optional[Any] = None
    education_evidence: Optional[Any] = None


def _ci_field(field_name, **overrides):
    base = dict(
        field_name=field_name, extracted_values=[], visible_text_checked=None,
        visible_text_matches=[], annotations_checked=None, annotation_matches=[],
        unmatched_same_type_annotations=[], evidence_summary="NO_EXTRACTED_VALUE",
    )
    base.update(overrides)
    return ContactFieldEvidence(**base)


def _ci_record(*fields):
    return ContactInformationEvidenceRecord(schema_valid=True, stage1_route=None, fields=list(fields))


def _lang_value(value, **overrides):
    base = dict(value=value, source_text_checked=True, source_text_present=True,
                evidence_summary="SOURCE_TEXT_PRESENT")
    base.update(overrides)
    return LanguageValueEvidence(**base)


def _pi_record(**overrides):
    name = overrides.pop("name_completeness", NameCompletenessEvidence(
        full_name="Ada Lovelace", first_name="Ada", last_name="Lovelace",
        checked=True, missing_tokens=[], evidence_summary="NAME_TOKENS_FULLY_REPRESENTED",
    ))
    location = overrides.pop("location", LocationEvidence(
        extracted_value="London", source_text_checked=True,
        verbatim_in_source=True, evidence_summary="SOURCE_TEXT_VERBATIM_MATCH",
    ))
    return PersonalInformationEvidenceRecord(
        schema_valid=True, stage1_route=None, name_completeness=name, location=location,
    )


def _we_entry(entry_index, **overrides):
    base = dict(
        entry_index=entry_index, locator_matched=True, match_confidence=0.9,
        degree_evidence=None, institution_evidence=None,
        company_equals_designation=False, review_priority="NONE", validation_actions=[],
    )
    base.update(overrides)
    return EntryEvidence(**base)


def _edu_entry(entry_index, **overrides):
    base = dict(
        entry_index=entry_index, locator_matched=True, match_confidence=None,
        degree_evidence=True, institution_evidence=True,
        company_equals_designation=None, review_priority=None, validation_actions=[],
    )
    base.update(overrides)
    return EntryEvidence(**base)


def _module_record(module, entries, section_found=True):
    return ModuleEvidenceRecord(
        module=module, schema_valid=True, stage1_route=None,
        locator_section_found=section_found, entries=entries,
    )


class TestSkillsAbsence:
    def test_report_modules_excludes_skills(self):
        assert "skills" not in REPORT_MODULES

    def test_full_report_has_no_skills_key(self):
        result = _FakeResult()
        report = build_resume_evidence_report("001", "stem", result)
        assert "skills" not in report["modules"]


class TestContactInformationReportEntries:
    def test_normal_corroborated_case(self):
        record = _ci_record(
            _ci_field("email", extracted_values=["a@x.com"], visible_text_matches=["a@x.com"],
                      evidence_summary="FULLY_CORROBORATED"),
        )
        entries = build_contact_information_report_entries(record)
        assert len(entries) == 1
        assert entries[0]["item_id"] == "email"
        assert "Source evidence was found" in entries[0]["explanation"]
        assert "wrong" not in entries[0]["explanation"].lower()

    def test_evidence_disagreement_does_not_overstate(self):
        record = _ci_record(
            _ci_field("linkedin", extracted_values=["x"], evidence_summary="EVIDENCE_DISAGREEMENT"),
        )
        entries = build_contact_information_report_entries(record)
        assert "does not indicate the value is incorrect" in entries[0]["explanation"]
        assert "is wrong" not in entries[0]["explanation"]

    def test_unmatched_annotations_produce_limitation_note(self):
        record = _ci_record(
            _ci_field("email", extracted_values=["a@x.com"], evidence_summary="FULLY_CORROBORATED",
                      unmatched_same_type_annotations=["other@x.com"]),
        )
        entries = build_contact_information_report_entries(record)
        assert any("false-positive" in lim for lim in entries[0]["limitations"])

    def test_none_evidence_produces_single_unavailable_placeholder(self):
        entries = build_contact_information_report_entries(None)
        assert len(entries) == 1
        assert entries[0]["item_id"] == "(module)"
        assert entries[0]["native_evidence_state"]["evidence_available"] is False

    def test_does_not_mutate_input_record(self):
        record = _ci_record(_ci_field("email", extracted_values=["a@x.com"]))
        snapshot = copy.deepcopy(record)
        build_contact_information_report_entries(record)
        assert record == snapshot


class TestLanguagesReportEntries:
    def test_source_text_present_case(self):
        record = LanguagesEvidenceRecord(schema_valid=True, stage1_route=None, values=[_lang_value("English")])
        entries = build_languages_report_entries(record)
        assert len(entries) == 1
        assert "Source evidence was found" in entries[0]["explanation"]

    def test_no_source_text_evidence_includes_misattribution_limitation(self):
        record = LanguagesEvidenceRecord(
            schema_valid=True, stage1_route=None,
            values=[_lang_value("Hindi", source_text_present=False, evidence_summary="NO_SOURCE_TEXT_EVIDENCE")],
        )
        entries = build_languages_report_entries(record)
        assert "does not" in entries[0]["explanation"]
        assert any("misattribution" in lim for lim in entries[0]["limitations"])

    def test_empty_values_produces_explicit_no_values_placeholder(self):
        record = LanguagesEvidenceRecord(schema_valid=True, stage1_route=None, values=[])
        entries = build_languages_report_entries(record)
        assert len(entries) == 1
        assert entries[0]["item_id"] == "(no values extracted)"

    def test_none_evidence_produces_single_unavailable_placeholder(self):
        entries = build_languages_report_entries(None)
        assert len(entries) == 1
        assert entries[0]["native_evidence_state"]["evidence_available"] is False

    def test_does_not_mutate_input_record(self):
        record = LanguagesEvidenceRecord(schema_valid=True, stage1_route=None, values=[_lang_value("English")])
        snapshot = copy.deepcopy(record)
        build_languages_report_entries(record)
        assert record == snapshot


class TestProfessionalSummaryReportEntries:
    def test_verbatim_match_case(self):
        evidence = ProfessionalSummaryEvidence(
            extracted_value="A summary.", source_text_checked=True,
            whole_document_verbatim_match=True, token_overlap_ratio=1.0,
            evidence_summary="SOURCE_TEXT_VERBATIM_MATCH",
        )
        entries = build_professional_summary_report_entries(evidence)
        assert len(entries) == 1
        assert entries[0]["item_id"] == "summary"
        assert "Ground truth is never used" in " ".join(entries[0]["limitations"])

    def test_no_verbatim_match_does_not_claim_unsupported(self):
        evidence = ProfessionalSummaryEvidence(
            extracted_value="A summary.", source_text_checked=True,
            whole_document_verbatim_match=False, token_overlap_ratio=1.0,
            evidence_summary="NO_VERBATIM_MATCH_FOUND",
        )
        entries = build_professional_summary_report_entries(evidence)
        assert "does not necessarily indicate" in entries[0]["explanation"]
        assert "unsupported" not in entries[0]["explanation"].split("does not necessarily indicate")[0]

    def test_no_summary_extracted(self):
        evidence = ProfessionalSummaryEvidence(
            extracted_value=None, source_text_checked=None,
            whole_document_verbatim_match=None, token_overlap_ratio=None,
            evidence_summary="NO_SUMMARY_EXTRACTED",
        )
        entries = build_professional_summary_report_entries(evidence)
        assert "nothing to evaluate" in entries[0]["explanation"]

    def test_none_evidence_produces_single_unavailable_placeholder(self):
        entries = build_professional_summary_report_entries(None)
        assert len(entries) == 1
        assert entries[0]["native_evidence_state"]["evidence_available"] is False

    def test_does_not_mutate_input_record(self):
        evidence = ProfessionalSummaryEvidence(
            extracted_value="A summary.", source_text_checked=True,
            whole_document_verbatim_match=True, token_overlap_ratio=1.0,
            evidence_summary="SOURCE_TEXT_VERBATIM_MATCH",
        )
        snapshot = copy.deepcopy(evidence)
        build_professional_summary_report_entries(evidence)
        assert evidence == snapshot


class TestPersonalInformationReportEntries:
    def test_always_exactly_two_entries_when_available(self):
        record = _pi_record()
        entries = build_personal_information_report_entries(record)
        assert len(entries) == 2
        assert {e["item_id"] for e in entries} == {"name_completeness", "location"}

    def test_name_tokens_missing_is_diagnostic_only(self):
        record = _pi_record(name_completeness=NameCompletenessEvidence(
            full_name="Ada Lovelace", first_name="Ada", last_name=None,
            checked=True, missing_tokens=["lovelace"], evidence_summary="NAME_TOKENS_MISSING",
        ))
        entries = build_personal_information_report_entries(record)
        name_entry = next(e for e in entries if e["item_id"] == "name_completeness")
        assert "does not identify how the name should be corrected" in name_entry["explanation"]
        assert any("diagnostic only" in lim for lim in name_entry["limitations"])

    def test_location_verbatim_match_discloses_wrong_section_limitation(self):
        record = _pi_record(location=LocationEvidence(
            extracted_value="Kanpur, UP", source_text_checked=True,
            verbatim_in_source=True, evidence_summary="SOURCE_TEXT_VERBATIM_MATCH",
        ))
        entries = build_personal_information_report_entries(record)
        loc_entry = next(e for e in entries if e["item_id"] == "location")
        assert any("wrong-section attribution" in lim for lim in loc_entry["limitations"])

    def test_none_evidence_produces_single_unavailable_placeholder(self):
        entries = build_personal_information_report_entries(None)
        assert len(entries) == 1
        assert entries[0]["native_evidence_state"]["evidence_available"] is False

    def test_does_not_mutate_input_record(self):
        record = _pi_record()
        snapshot = copy.deepcopy(record)
        build_personal_information_report_entries(record)
        assert record == snapshot


class TestWorkExperienceReportEntries:
    def test_locator_matched_case(self):
        record = _module_record("work_experience", [_we_entry(0, locator_matched=True, review_priority="NONE")])
        entries = build_work_experience_report_entries(record)
        assert "Matching source evidence was found" in entries[0]["explanation"]

    def test_elevated_priority_does_not_assert_incorrectness(self):
        record = _module_record("work_experience", [
            _we_entry(0, locator_matched=False, company_equals_designation=True, review_priority="ELEVATED"),
        ])
        entries = build_work_experience_report_entries(record)
        assert "does not by itself mean" in entries[0]["explanation"]
        assert entries[0]["explanation"].count("incorrect") == 0 or "does not" in entries[0]["explanation"]

    def test_locator_unavailable_case(self):
        record = _module_record("work_experience", [_we_entry(0, locator_matched=None, review_priority=None)],
                                 section_found=None)
        entries = build_work_experience_report_entries(record)
        assert "not available" in entries[0]["explanation"]

    def test_validation_actions_produce_limitation_note(self):
        action = ValidationActionRecord(rule_id="WEV-016", severity="Medium", auto_fixed=True, message="nulled")
        record = _module_record("work_experience", [_we_entry(0, validation_actions=[action])])
        entries = build_work_experience_report_entries(record)
        assert any("WEV-016" in lim for lim in entries[0]["limitations"])

    def test_empty_entries_produces_explicit_placeholder(self):
        record = _module_record("work_experience", [])
        entries = build_work_experience_report_entries(record)
        assert len(entries) == 1
        assert entries[0]["item_id"] == "(no entries extracted)"

    def test_none_evidence_produces_single_unavailable_placeholder(self):
        entries = build_work_experience_report_entries(None)
        assert len(entries) == 1
        assert entries[0]["native_evidence_state"]["evidence_available"] is False

    def test_does_not_mutate_input_record(self):
        record = _module_record("work_experience", [_we_entry(0)])
        snapshot = copy.deepcopy(record)
        build_work_experience_report_entries(record)
        assert record == snapshot


class TestEducationReportEntries:
    def test_locator_matched_preserves_native_booleans(self):
        record = _module_record("education", [_edu_entry(0, degree_evidence=True, institution_evidence=False)])
        entries = build_education_report_entries(record)
        assert entries[0]["native_evidence_state"]["degree_evidence"] is True
        assert entries[0]["native_evidence_state"]["institution_evidence"] is False
        assert any("no numeric confidence signal" in lim for lim in entries[0]["limitations"])

    def test_no_match_does_not_assert_incorrectness(self):
        record = _module_record("education", [_edu_entry(0, locator_matched=False, degree_evidence=None, institution_evidence=None)])
        entries = build_education_report_entries(record)
        assert "does not" in entries[0]["explanation"]

    def test_empty_entries_produces_explicit_placeholder(self):
        record = _module_record("education", [])
        entries = build_education_report_entries(record)
        assert len(entries) == 1
        assert entries[0]["item_id"] == "(no entries extracted)"

    def test_none_evidence_produces_single_unavailable_placeholder(self):
        entries = build_education_report_entries(None)
        assert len(entries) == 1
        assert entries[0]["native_evidence_state"]["evidence_available"] is False

    def test_does_not_mutate_input_record(self):
        record = _module_record("education", [_edu_entry(0)])
        snapshot = copy.deepcopy(record)
        build_education_report_entries(record)
        assert record == snapshot


class TestFullReportAssembly:
    def test_all_six_modules_present(self):
        result = _FakeResult(
            personal_information_evidence=_pi_record(),
            contact_information_evidence=_ci_record(_ci_field("email", extracted_values=["a@x.com"])),
            professional_summary_evidence=ProfessionalSummaryEvidence(
                extracted_value="s", source_text_checked=True, whole_document_verbatim_match=True,
                token_overlap_ratio=1.0, evidence_summary="SOURCE_TEXT_VERBATIM_MATCH",
            ),
            languages_evidence=LanguagesEvidenceRecord(schema_valid=True, stage1_route=None, values=[_lang_value("English")]),
            work_experience_evidence=_module_record("work_experience", [_we_entry(0)]),
            education_evidence=_module_record("education", [_edu_entry(0)]),
        )
        report = build_resume_evidence_report("001", "stem", result)
        assert set(report["modules"].keys()) == set(REPORT_MODULES)
        assert all(len(entries) > 0 for entries in report["modules"].values())

    def test_all_none_evidence_still_produces_full_module_set_with_placeholders(self):
        result = _FakeResult()
        report = build_resume_evidence_report("001", "stem", result)
        assert set(report["modules"].keys()) == set(REPORT_MODULES)
        for entries in report["modules"].values():
            assert len(entries) == 1
            assert entries[0]["native_evidence_state"]["evidence_available"] is False

    def test_determinism_same_result_object(self):
        result = _FakeResult(personal_information_evidence=_pi_record())
        report1 = build_resume_evidence_report("001", "stem", result)
        report2 = build_resume_evidence_report("001", "stem", result)
        assert report1 == report2

    def test_does_not_mutate_the_result_object(self):
        result = _FakeResult(personal_information_evidence=_pi_record())
        snapshot = copy.deepcopy(result)
        build_resume_evidence_report("001", "stem", result)
        assert result == snapshot


class TestReportToRows:
    def test_flattens_every_module_entry_to_one_row(self):
        result = _FakeResult(personal_information_evidence=_pi_record())
        report = build_resume_evidence_report("001", "stem", result)
        rows = report_to_rows(report)

        pi_rows = [r for r in rows if r["module"] == "personal_information"]
        assert len(pi_rows) == 2  # name_completeness + location
        assert all(r["resume_id"] == "001" and r["stem"] == "stem" for r in rows)

    def test_does_not_mutate_the_report(self):
        result = _FakeResult(personal_information_evidence=_pi_record())
        report = build_resume_evidence_report("001", "stem", result)
        snapshot = copy.deepcopy(report)
        report_to_rows(report)
        assert report == snapshot


class TestNoOverstatingLanguage:
    """Direct scan of every explanation this module can produce for
    forbidden absolute/overstating words, per Step 4's explicit 'bad
    examples' list."""

    FORBIDDEN_SUBSTRINGS = (
        "is wrong", "failed validation", "high confidence", "low confidence",
        "needs immediate review", "automatically reject",
    )

    def _all_explanations(self):
        yield from (e["explanation"] for e in build_contact_information_report_entries(
            _ci_record(*[_ci_field(f, evidence_summary=s) for f, s in [
                ("email", "NO_EXTRACTED_VALUE"), ("phone", "NO_EVIDENCE_SOURCES_AVAILABLE"),
                ("linkedin", "FULLY_CORROBORATED"), ("github", "PARTIALLY_CORROBORATED"),
                ("portfolio", "EVIDENCE_DISAGREEMENT"),
            ]])
        ))
        for state in ("SOURCE_TEXT_PRESENT", "NO_SOURCE_TEXT_EVIDENCE", "NO_SOURCE_TEXT_AVAILABLE"):
            yield from (e["explanation"] for e in build_languages_report_entries(
                LanguagesEvidenceRecord(schema_valid=True, stage1_route=None,
                                         values=[_lang_value("X", evidence_summary=state)])
            ))
        for state in ("NO_SUMMARY_EXTRACTED", "NO_SOURCE_TEXT_AVAILABLE", "SOURCE_TEXT_VERBATIM_MATCH", "NO_VERBATIM_MATCH_FOUND"):
            yield from (e["explanation"] for e in build_professional_summary_report_entries(
                ProfessionalSummaryEvidence(extracted_value="s" if state != "NO_SUMMARY_EXTRACTED" else None,
                                             source_text_checked=True, whole_document_verbatim_match=None,
                                             token_overlap_ratio=None, evidence_summary=state)
            ))
        for we_kwargs in (
            dict(locator_matched=True, review_priority="NONE"),
            dict(locator_matched=False, review_priority="ELEVATED", company_equals_designation=True),
            dict(locator_matched=False, review_priority="STANDARD"),
            dict(locator_matched=None, review_priority=None),
        ):
            yield from (e["explanation"] for e in build_work_experience_report_entries(
                _module_record("work_experience", [_we_entry(0, **we_kwargs)])
            ))

    def test_no_forbidden_overstating_language(self):
        for explanation in self._all_explanations():
            lowered = explanation.lower()
            for forbidden in self.FORBIDDEN_SUBSTRINGS:
                assert forbidden not in lowered, f"Forbidden phrase {forbidden!r} found in: {explanation!r}"
