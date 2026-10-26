"""
tests/test_contact_information_evidence_report.py

Tests for evaluation/contact_information_evidence_report.py (Milestone
V3.9A). Pure in-memory logic against directly-constructed
ContactInformationEvidenceRecord/ContactFieldEvidence objects -- no
Ollama, no real pipeline run, no file I/O. Mirrors
tests/test_evidence_record.py's own structure.
"""

import copy

import pytest

from evaluation.contact_information_evidence_report import (
    build_corpus_summary,
    build_edge_case_rows,
    build_field_level_distribution,
    build_resume_level_rollup,
    evidence_records_to_field_rows,
)
from services.evidence_record import ContactFieldEvidence, ContactInformationEvidenceRecord


def _field(field_name, **overrides):
    base = dict(
        field_name=field_name,
        extracted_values=[],
        visible_text_checked=None,
        visible_text_matches=[],
        annotations_checked=None,
        annotation_matches=[],
        unmatched_same_type_annotations=[],
        evidence_summary="NO_EXTRACTED_VALUE",
    )
    base.update(overrides)
    return ContactFieldEvidence(**base)


def _record(*fields):
    return ContactInformationEvidenceRecord(schema_valid=True, stage1_route=None, fields=list(fields))


class TestFieldRowFlattening:
    def test_every_field_becomes_one_row(self):
        record = _record(
            _field("email", extracted_values=["a@example.com"], evidence_summary="FULLY_CORROBORATED"),
            _field("phone"),
            _field("linkedin"),
            _field("github"),
            _field("portfolio"),
        )
        rows = evidence_records_to_field_rows([("001", "Resume One", record)])

        assert len(rows) == 5
        assert rows[0]["resume_id"] == "001"
        assert rows[0]["stem"] == "Resume One"
        assert rows[0]["field"] == "email"
        assert rows[0]["extracted_value_count"] == 1
        assert rows[0]["evidence_summary"] == "FULLY_CORROBORATED"

    def test_does_not_mutate_the_input_record(self):
        record = _record(_field("email", extracted_values=["a@example.com"]))
        snapshot = copy.deepcopy(record)

        evidence_records_to_field_rows([("001", "stem", record)])

        assert record == snapshot

    def test_multiple_resumes_all_represented(self):
        r1 = _record(_field("email", extracted_values=["a@example.com"]))
        r2 = _record(_field("email", extracted_values=["b@example.com"]))

        rows = evidence_records_to_field_rows([("001", "one", r1), ("002", "two", r2)])

        assert {r["resume_id"] for r in rows} == {"001", "002"}


class TestCorpusSummary:
    def test_normal_corroborated_evidence_aggregation(self):
        rows = evidence_records_to_field_rows([
            ("001", "one", _record(_field("email", extracted_values=["a@x.com"], evidence_summary="FULLY_CORROBORATED"))),
        ])

        summary = build_corpus_summary(rows)

        assert summary["total_resumes"] == 1
        assert summary["total_resume_field_records"] == 1
        assert summary["total_populated_fields"] == 1
        assert summary["count_FULLY_CORROBORATED"] == 1
        assert summary["count_corroborated_any_source"] == 1
        assert summary["count_disagreement"] == 0

    def test_no_evidence_aggregation(self):
        rows = evidence_records_to_field_rows([
            ("001", "one", _record(_field("email"))),  # NO_EXTRACTED_VALUE (default)
        ])

        summary = build_corpus_summary(rows)

        assert summary["total_populated_fields"] == 0
        assert summary["total_empty_fields"] == 1
        assert summary["count_NO_EXTRACTED_VALUE"] == 1
        assert summary["count_absence_or_unchecked"] == 1
        # Absence must never be counted as disagreement or as corroborated.
        assert summary["count_disagreement"] == 0
        assert summary["count_corroborated_any_source"] == 0

    def test_disagreement_aggregation(self):
        rows = evidence_records_to_field_rows([
            ("001", "one", _record(_field(
                "linkedin", extracted_values=["https://linkedin.com/in/wrong"],
                unmatched_same_type_annotations=["https://linkedin.com/in/real"],
                evidence_summary="EVIDENCE_DISAGREEMENT",
            ))),
        ])

        summary = build_corpus_summary(rows)

        assert summary["count_EVIDENCE_DISAGREEMENT"] == 1
        assert summary["count_disagreement"] == 1
        assert summary["count_corroborated_any_source"] == 0

    def test_every_label_present_even_when_zero(self):
        rows = evidence_records_to_field_rows([("001", "one", _record(_field("email")))])

        summary = build_corpus_summary(rows)

        for label in ("NO_EXTRACTED_VALUE", "NO_EVIDENCE_SOURCES_AVAILABLE", "FULLY_CORROBORATED",
                      "PARTIALLY_CORROBORATED", "NOT_CORROBORATED", "EVIDENCE_DISAGREEMENT"):
            assert f"count_{label}" in summary


class TestFieldLevelGrouping:
    def test_counts_grouped_by_field_and_label(self):
        rows = evidence_records_to_field_rows([
            ("001", "one", _record(
                _field("email", extracted_values=["a@x.com"], evidence_summary="FULLY_CORROBORATED"),
                _field("phone", extracted_values=["555"], evidence_summary="FULLY_CORROBORATED"),
            )),
            ("002", "two", _record(
                _field("email", extracted_values=["b@x.com"], evidence_summary="NOT_CORROBORATED"),
                _field("phone"),
            )),
        ])

        dist = build_field_level_distribution(rows)

        email_fully = next(r for r in dist if r["field"] == "email" and r["evidence_summary"] == "FULLY_CORROBORATED")
        email_not = next(r for r in dist if r["field"] == "email" and r["evidence_summary"] == "NOT_CORROBORATED")
        phone_no_value = next(r for r in dist if r["field"] == "phone" and r["evidence_summary"] == "NO_EXTRACTED_VALUE")
        assert email_fully["count"] == 1
        assert email_not["count"] == 1
        assert phone_no_value["count"] == 1

    def test_zero_counts_are_still_emitted(self):
        rows = evidence_records_to_field_rows([("001", "one", _record(_field("email", extracted_values=["a@x.com"], evidence_summary="FULLY_CORROBORATED")))])

        dist = build_field_level_distribution(rows)

        disagreement_row = next(r for r in dist if r["field"] == "email" and r["evidence_summary"] == "EVIDENCE_DISAGREEMENT")
        assert disagreement_row["count"] == 0


class TestResumeLevelGrouping:
    def test_one_row_per_resume_with_label_counts(self):
        rows = evidence_records_to_field_rows([
            ("001", "one", _record(
                _field("email", extracted_values=["a@x.com"], evidence_summary="FULLY_CORROBORATED"),
                _field("phone"),
            )),
            ("002", "two", _record(
                _field("email", extracted_values=["b@x.com"], evidence_summary="EVIDENCE_DISAGREEMENT",
                       unmatched_same_type_annotations=["mailto:other@x.com"]),
            )),
        ])

        rollup = build_resume_level_rollup(rows)

        assert len(rollup) == 2
        r1 = next(r for r in rollup if r["resume_id"] == "001")
        r2 = next(r for r in rollup if r["resume_id"] == "002")
        assert r1["FULLY_CORROBORATED"] == 1
        assert r1["NO_EXTRACTED_VALUE"] == 1
        assert r2["EVIDENCE_DISAGREEMENT"] == 1


class TestKnownFalsePositiveAnnotationCases:
    def test_placeholder_mailto_case_is_edge_case_but_not_disagreement(self):
        """V3.7 resume 004 pattern: visible text corroborates, but a
        same-type annotation disagrees -- must surface as an edge case
        for review, while evidence_summary itself stays FULLY_CORROBORATED
        (never downgraded by the bad annotation)."""
        rows = evidence_records_to_field_rows([
            ("004", "placeholder_case", _record(_field(
                "email", extracted_values=["real@example.com"],
                visible_text_checked=True, visible_text_matches=["real@example.com"],
                annotations_checked=True, unmatched_same_type_annotations=["x@gmail.com"],
                evidence_summary="FULLY_CORROBORATED",
            ))),
        ])

        edge_cases = build_edge_case_rows(rows)
        summary = build_corpus_summary(rows)

        assert len(edge_cases) == 1
        assert edge_cases[0]["evidence_summary"] == "FULLY_CORROBORATED"
        assert edge_cases[0]["unmatched_same_type_annotations"] == "x@gmail.com"
        # Must count as corroborated, never as a disagreement.
        assert summary["count_corroborated_any_source"] == 1
        assert summary["count_disagreement"] == 0

    def test_stale_linkedin_annotation_case_is_edge_case_but_not_disagreement(self):
        """V3.7 resume 005 pattern -- same shape as the placeholder case,
        different field."""
        rows = evidence_records_to_field_rows([
            ("005", "stale_linkedin_case", _record(_field(
                "linkedin", extracted_values=["https://www.linkedin.com/in/arjun-sharma-1bbb0721"],
                visible_text_checked=True,
                visible_text_matches=["https://www.linkedin.com/in/arjun-sharma-1bbb0721"],
                annotations_checked=True,
                unmatched_same_type_annotations=["http://www.linkedin.com/in/arjun-sharma-1bbb07210"],
                evidence_summary="FULLY_CORROBORATED",
            ))),
        ])

        edge_cases = build_edge_case_rows(rows)

        assert len(edge_cases) == 1
        assert edge_cases[0]["field"] == "linkedin"
        assert edge_cases[0]["evidence_summary"] == "FULLY_CORROBORATED"

    def test_non_contact_annotation_never_appears_since_it_was_already_excluded_upstream(self):
        """Non-contact annotations are excluded by build_contact_
        information_evidence() itself (V3.8/V3.9) -- this reporting
        layer never sees them at all, so a resume whose only annotation
        was a non-contact link (e.g. a GitHub repo link) has an empty
        unmatched_same_type_annotations list, not a phantom edge case."""
        rows = evidence_records_to_field_rows([
            ("003", "repo_link_case", _record(_field(
                "github", extracted_values=["https://github.com/ada"],
                annotations_checked=True, annotation_matches=["https://github.com/ada"],
                unmatched_same_type_annotations=[],  # the repo-path link was already filtered out upstream
                evidence_summary="FULLY_CORROBORATED",
            ))),
        ])

        edge_cases = build_edge_case_rows(rows)

        assert edge_cases == []

    def test_missing_pdf_evidence_is_not_a_negative_verdict(self):
        rows = evidence_records_to_field_rows([
            ("010", "no_pdf_case", _record(_field(
                "email", extracted_values=["real@example.com"],
                visible_text_checked=True, visible_text_matches=["real@example.com"],
                annotations_checked=None,  # no PDF supplied -- never guessed as False
                evidence_summary="FULLY_CORROBORATED",
            ))),
        ])

        summary = build_corpus_summary(rows)
        edge_cases = build_edge_case_rows(rows)

        assert rows[0]["annotations_checked"] is None
        assert summary["count_disagreement"] == 0
        assert edge_cases == []  # no unmatched annotation, no disagreement -- nothing to flag


class TestZeroMutation:
    def test_reporting_never_mutates_the_source_records(self):
        record = _record(
            _field("email", extracted_values=["a@example.com"], evidence_summary="FULLY_CORROBORATED"),
            _field("linkedin", extracted_values=["https://linkedin.com/in/x"],
                   unmatched_same_type_annotations=["https://linkedin.com/in/y"],
                   evidence_summary="FULLY_CORROBORATED"),
        )
        snapshot = copy.deepcopy(record)
        records = [("001", "one", record)]

        rows = evidence_records_to_field_rows(records)
        build_corpus_summary(rows)
        build_field_level_distribution(rows)
        build_resume_level_rollup(rows)
        build_edge_case_rows(rows)

        assert record == snapshot

    def test_calling_twice_is_idempotent(self):
        record = _record(_field("email", extracted_values=["a@example.com"], evidence_summary="FULLY_CORROBORATED"))
        records = [("001", "one", record)]

        rows1 = evidence_records_to_field_rows(records)
        rows2 = evidence_records_to_field_rows(records)
        summary1 = build_corpus_summary(rows1)
        summary2 = build_corpus_summary(rows2)

        assert rows1 == rows2
        assert summary1 == summary2
