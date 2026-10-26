"""
tests/test_json_comparator.py

Tests for validators/json_comparator.py (Milestone 13.3).

No external services are involved (this module only compares Python
values and writes CSVs), so every test runs directly against
JsonComparator with no mocking. The "integration" test at the bottom
exercises the full comparator + CSV-writing workflow against real files
on disk (via tmp_path), as opposed to the unit tests above it which only
check compare()'s in-memory result.
"""

import csv

import pytest

from validators.json_comparator import (
    ComparisonResult,
    DiagnosticResult,
    ENTRY_STATUS_EXTRA,
    ENTRY_STATUS_FULL_MATCH,
    ENTRY_STATUS_MISSING,
    ENTRY_STATUS_PARTIAL_MATCH,
    FieldComparison,
    JsonComparator,
    STATUS_EXTRA,
    STATUS_FAIL,
    STATUS_MISSING,
    STATUS_PASS,
)


@pytest.fixture
def comparator():
    return JsonComparator()


class TestNestedObjects:
    def test_matching_nested_object_passes(self, comparator):
        expected = {"personal_information": {"full_name": "Ada Lovelace"}}
        actual = {"personal_information": {"full_name": "Ada Lovelace"}}

        result = comparator.compare(expected, actual)

        assert result.accuracy == 1.0
        [field_result] = result.field_results
        assert field_result.field_path == "personal_information.full_name"
        assert field_result.status == STATUS_PASS

    def test_differing_nested_value_fails(self, comparator):
        expected = {"personal_information": {"full_name": "Ada Lovelace"}}
        actual = {"personal_information": {"full_name": "Ada Byron"}}

        result = comparator.compare(expected, actual)

        [field_result] = result.field_results
        assert field_result.status == STATUS_FAIL
        assert result.matched_fields == 0

    def test_deeply_nested_objects_use_dotted_paths(self, comparator):
        expected = {"a": {"b": {"c": "value"}}}
        actual = {"a": {"b": {"c": "value"}}}

        result = comparator.compare(expected, actual)

        [field_result] = result.field_results
        assert field_result.field_path == "a.b.c"
        assert field_result.status == STATUS_PASS


class TestArrays:
    def test_arrays_with_same_elements_different_order_pass(self, comparator):
        expected = {"skills": ["Python", "SQL"]}
        actual = {"skills": ["SQL", "Python"]}

        result = comparator.compare(expected, actual)

        [field_result] = result.field_results
        assert field_result.field_path == "skills"
        assert field_result.status == STATUS_PASS

    def test_arrays_with_different_elements_fail(self, comparator):
        expected = {"skills": ["Python", "SQL"]}
        actual = {"skills": ["Python", "Java"]}

        result = comparator.compare(expected, actual)

        [field_result] = result.field_results
        assert field_result.status == STATUS_FAIL

    def test_empty_arrays_pass(self, comparator):
        result = comparator.compare({"skills": []}, {"skills": []})

        assert result.field_results[0].status == STATUS_PASS

    def test_arrays_are_compared_as_sets_so_duplicates_do_not_count(self, comparator):
        """Documents the literal 'compare as unordered sets' rule: a
        duplicated element collapses like a Python set, so this counts
        as a PASS even though the counts differ."""
        expected = {"skills": ["Python", "Python"]}
        actual = {"skills": ["Python"]}

        result = comparator.compare(expected, actual)

        assert result.field_results[0].status == STATUS_PASS


class TestNulls:
    def test_null_equals_null_passes(self, comparator):
        result = comparator.compare({"linkedin": None}, {"linkedin": None})

        assert result.field_results[0].status == STATUS_PASS
        assert result.accuracy == 1.0

    def test_null_vs_value_fails(self, comparator):
        result = comparator.compare({"linkedin": None}, {"linkedin": "https://x.com/ada"})

        assert result.field_results[0].status == STATUS_FAIL

    def test_value_vs_null_fails(self, comparator):
        result = comparator.compare({"linkedin": "https://x.com/ada"}, {"linkedin": None})

        assert result.field_results[0].status == STATUS_FAIL


class TestMissingFields:
    def test_field_absent_from_actual_is_missing(self, comparator):
        expected = {"full_name": "Ada Lovelace", "github": None}
        actual = {"full_name": "Ada Lovelace"}

        result = comparator.compare(expected, actual)

        missing = [r for r in result.field_results if r.status == STATUS_MISSING]
        assert len(missing) == 1
        assert missing[0].field_path == "github"
        assert missing[0].expected is None  # github's own expected value was null
        assert result.missing_fields == 1

    def test_missing_field_reduces_accuracy(self, comparator):
        expected = {"a": "1", "b": "2"}
        actual = {"a": "1"}

        result = comparator.compare(expected, actual)

        assert result.total_fields == 2
        assert result.matched_fields == 1
        assert result.missing_fields == 1
        assert result.accuracy == 0.5


class TestExtraFields:
    def test_field_absent_from_expected_is_extra(self, comparator):
        expected = {"full_name": "Ada Lovelace"}
        actual = {"full_name": "Ada Lovelace", "nickname": "Ada"}

        result = comparator.compare(expected, actual)

        extra = [r for r in result.field_results if r.status == STATUS_EXTRA]
        assert len(extra) == 1
        assert extra[0].field_path == "nickname"
        assert extra[0].actual == "Ada"
        assert result.extra_fields == 1

    def test_extra_field_reduces_accuracy(self, comparator):
        expected = {"a": "1"}
        actual = {"a": "1", "b": "2"}

        result = comparator.compare(expected, actual)

        assert result.total_fields == 2
        assert result.matched_fields == 1
        assert result.extra_fields == 1
        assert result.accuracy == 0.5


class TestFullMatch:
    def test_100_percent_match(self, comparator):
        expected = {
            "personal_information": {"full_name": "Ada Lovelace", "current_location": None},
            "skills": ["Mathematics", "Writing"],
            "languages": [],
        }
        actual = {
            "personal_information": {"full_name": "Ada Lovelace", "current_location": None},
            "skills": ["Writing", "Mathematics"],
            "languages": [],
        }

        result = comparator.compare(expected, actual)

        assert result.accuracy == 1.0
        assert result.missing_fields == 0
        assert result.extra_fields == 0
        assert result.matched_fields == result.total_fields


class TestZeroMatch:
    def test_0_percent_match_with_completely_disjoint_keys(self, comparator):
        expected = {"full_name": "Ada Lovelace", "email": ["ada@example.com"]}
        actual = {"nickname": "Ada", "handle": "@ada"}

        result = comparator.compare(expected, actual)

        assert result.accuracy == 0.0
        assert result.matched_fields == 0
        assert result.missing_fields == 2
        assert result.extra_fields == 2
        assert result.total_fields == 4

    def test_0_percent_match_with_same_keys_all_different_values(self, comparator):
        expected = {"full_name": "Ada Lovelace", "current_location": "London"}
        actual = {"full_name": "Grace Hopper", "current_location": "New York"}

        result = comparator.compare(expected, actual)

        assert result.accuracy == 0.0
        assert result.missing_fields == 0
        assert result.extra_fields == 0
        assert all(r.status == STATUS_FAIL for r in result.field_results)


class TestComparisonResultDataclasses:
    def test_field_comparison_is_a_plain_dataclass(self):
        record = FieldComparison(field_path="x", expected=1, actual=1, status=STATUS_PASS)
        assert record.field_path == "x"

    def test_comparison_result_defaults_field_results_to_empty_list(self):
        result = ComparisonResult(
            total_fields=0, matched_fields=0, missing_fields=0, extra_fields=0, accuracy=1.0
        )
        assert result.field_results == []


@pytest.mark.integration
def test_generate_reports_writes_three_csv_files_with_correct_content(tmp_path):
    """Exercises the full comparator + CSV-reporting workflow against
    real files on disk, for a small batch of resumes."""
    comparator = JsonComparator()

    comparisons = {
        "resume_one": comparator.compare(
            {"full_name": "Ada Lovelace", "skills": ["Math"]},
            {"full_name": "Ada Lovelace", "skills": ["Math"]},
        ),
        "resume_two": comparator.compare(
            {"full_name": "Grace Hopper", "skills": ["COBOL"]},
            {"full_name": "Grace H.", "skills": ["COBOL"], "nickname": "Amazing Grace"},
        ),
    }

    comparator.generate_reports(comparisons, output_dir=tmp_path)

    for filename in ("resume_accuracy.csv", "field_accuracy.csv", "summary.csv"):
        assert (tmp_path / filename).is_file()

    with (tmp_path / "resume_accuracy.csv").open(newline="", encoding="utf-8") as f:
        rows = {row["resume_name"]: row for row in csv.DictReader(f)}
    assert rows["resume_one"]["accuracy"] == "1.0000"
    assert rows["resume_two"]["extra_fields"] == "1"

    with (tmp_path / "field_accuracy.csv").open(newline="", encoding="utf-8") as f:
        field_rows = {row["field_path"]: row for row in csv.DictReader(f)}
    assert field_rows["full_name"]["total"] == "2"
    assert field_rows["full_name"]["matched"] == "1"
    assert "nickname" in field_rows
    assert field_rows["nickname"]["extra"] == "1"

    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as f:
        summary_row = next(csv.DictReader(f))
    assert summary_row["total_resumes"] == "2"
    # resume_one: 2/2 matched: resume_two: 1/3 matched -> 3 matched / 5 total fields overall.
    assert float(summary_row["overall_accuracy"]) == pytest.approx(3 / 5, abs=0.001)
    # Average of each resume's own accuracy: (1.0 + 1/3) / 2.
    assert float(summary_row["average_resume_accuracy"]) == pytest.approx(2 / 3, abs=0.001)


# ---------------------------------------------------------------------------
# Diagnostic mode (Milestone 20.3)
# ---------------------------------------------------------------------------
#
# compare_diagnostic() is an entirely separate analysis layered on top of
# compare() (see validators/json_comparator.py's "Diagnostic mode" section
# and docs/JSON_COMPARATOR_DIAGNOSTIC.md). These tests check three things
# the strict-mode tests above can't: that the strict score is untouched by
# (and unable to affect, or be affected by) diagnostic mode, that entry/
# field-level matching behaves as designed, and that the new reports are
# written correctly.


class TestDiagnosticNeverAffectsStrictScore:
    """Requirement 1/3: diagnostic mode must never change, and must never
    be changed by, the strict result."""

    def test_strict_result_embedded_in_diagnostic_matches_plain_compare(self, comparator):
        expected = {
            "work_experience": [
                {"company": "Acme", "duration": None, "technologies": ["Python"]}
            ]
        }
        actual = {
            "work_experience": [
                {"company": "Acme", "duration": "3 years", "technologies": ["Python"]}
            ]
        }

        plain = comparator.compare(expected, actual)
        diagnostic = comparator.compare_diagnostic(expected, actual)

        assert diagnostic.strict_result == plain
        assert diagnostic.strict_accuracy == plain.accuracy

    def test_calling_compare_diagnostic_does_not_change_a_later_plain_compare(self, comparator):
        expected = {"skills": ["Python", "SQL"]}
        actual = {"skills": ["Python", "SQL"]}

        comparator.compare_diagnostic(expected, actual)
        result = comparator.compare(expected, actual)

        assert result.accuracy == 1.0

    def test_diagnostic_strict_accuracy_is_harsh_even_when_entry_field_accuracy_is_high(
        self, comparator
    ):
        """The canonical case from docs/JSON_COMPARATOR_AUDIT.md: one
        entry, almost every sub-field right, still 0% strict."""
        expected = {
            "work_experience": [
                {
                    "company": "PSD and Associates",
                    "location": None,
                    "duration": None,
                    "responsibilities": ["Conducted statutory audit."],
                }
            ]
        }
        actual = {
            "work_experience": [
                {
                    "company": "PSD and Associates",
                    "location": "PSD and Associates",
                    "duration": "3 years",
                    "responsibilities": ["Conducted statutory audit."],
                }
            ]
        }

        result = comparator.compare_diagnostic(expected, actual)

        assert result.strict_accuracy == 0.0
        assert result.entry_accuracy == 1.0  # the job was found
        assert result.field_accuracy == pytest.approx(0.5)  # 2 of 4 sub-fields right


class TestEntryLevelMatching:
    def test_identical_single_entry_is_full_match(self, comparator):
        expected = {"work_experience": [{"company": "Acme", "location": "NY"}]}
        actual = {"work_experience": [{"company": "Acme", "location": "NY"}]}

        result = comparator.compare_diagnostic(expected, actual)

        we = result.array_diagnostics["work_experience"]
        assert we.expected_count == 1
        assert we.actual_count == 1
        [entry] = we.entries
        assert entry.status == ENTRY_STATUS_FULL_MATCH
        assert entry.expected_index == 0
        assert entry.actual_index == 0
        assert entry.match_score == 1.0
        assert we.entry_accuracy == 1.0
        assert we.field_accuracy == 1.0

    def test_entry_with_one_wrong_field_is_partial_match(self, comparator):
        expected = {"education": [{"institution": "MIT", "degree": "BSc"}]}
        actual = {"education": [{"institution": "MIT", "degree": "BA"}]}

        result = comparator.compare_diagnostic(expected, actual)

        [entry] = result.array_diagnostics["education"].entries
        assert entry.status == ENTRY_STATUS_PARTIAL_MATCH
        assert entry.match_score == pytest.approx(0.5)
        fields_by_name = {fc.field_name: fc for fc in entry.field_comparisons}
        assert fields_by_name["institution"].matched is True
        assert fields_by_name["degree"].matched is False
        assert fields_by_name["degree"].expected == "BSc"
        assert fields_by_name["degree"].actual == "BA"

    def test_dropped_entry_is_missing_not_extra(self, comparator):
        expected = {"work_experience": [{"company": "A"}, {"company": "B"}]}
        actual = {"work_experience": [{"company": "A"}]}

        result = comparator.compare_diagnostic(expected, actual)

        we = result.array_diagnostics["work_experience"]
        statuses = sorted(e.status for e in we.entries)
        assert statuses == sorted([ENTRY_STATUS_FULL_MATCH, ENTRY_STATUS_MISSING])
        missing = next(e for e in we.entries if e.status == ENTRY_STATUS_MISSING)
        assert missing.expected_index == 1
        assert missing.actual_index is None
        assert missing.field_comparisons == []
        assert we.entry_accuracy == 0.5  # 1 of 2 GT entries found

    def test_fabricated_entry_is_extra_not_missing(self, comparator):
        expected = {"work_experience": [{"company": "A"}]}
        actual = {"work_experience": [{"company": "A"}, {"company": "Fabricated Co"}]}

        result = comparator.compare_diagnostic(expected, actual)

        we = result.array_diagnostics["work_experience"]
        extra = next(e for e in we.entries if e.status == ENTRY_STATUS_EXTRA)
        assert extra.actual_index == 1
        assert extra.expected_index is None
        assert we.actual_count == 2
        assert we.expected_count == 1
        assert we.entry_accuracy == 1.0  # the one real GT entry was found

    def test_entries_with_nothing_in_common_are_not_arbitrarily_paired(self, comparator):
        """An expected entry and an actual entry that share zero
        sub-fields must be reported as MISSING + EXTRA, never matched to
        each other just because counts line up."""
        expected = {"work_experience": [{"company": "Acme", "location": "NY"}]}
        actual = {"work_experience": [{"designation": "Engineer", "duration": "2 years"}]}

        result = comparator.compare_diagnostic(expected, actual)

        we = result.array_diagnostics["work_experience"]
        statuses = sorted(e.status for e in we.entries)
        assert statuses == sorted([ENTRY_STATUS_MISSING, ENTRY_STATUS_EXTRA])

    def test_nested_list_order_does_not_affect_diagnostic_matching(self, comparator):
        """Closes the ordering gap noted in docs/JSON_COMPARATOR_AUDIT.md
        section 1 -- diagnostic mode treats technologies/responsibilities
        as unordered, unlike the strict comparator."""
        expected = {"work_experience": [{"company": "X", "technologies": ["A", "B", "C"]}]}
        actual = {"work_experience": [{"company": "X", "technologies": ["C", "B", "A"]}]}

        result = comparator.compare_diagnostic(expected, actual)

        [entry] = result.array_diagnostics["work_experience"].entries
        assert entry.status == ENTRY_STATUS_FULL_MATCH
        # ... while the strict result, unaffected, still fails on this exact case.
        assert result.strict_accuracy == 0.0

    def test_best_match_picks_highest_similarity_not_first_available(self, comparator):
        """Two GT entries, two actual entries: each GT entry must align
        to its true counterpart even when a worse partial overlap with
        the other actual entry exists."""
        expected = {
            "education": [
                {"institution": "MIT", "degree": "BSc", "grade": "A"},
                {"institution": "Harvard", "degree": "MSc", "grade": "A"},
            ]
        }
        actual = {
            "education": [
                {"institution": "Harvard", "degree": "MSc", "grade": "A"},  # exact match for GT[1]
                {"institution": "MIT", "degree": "BSc", "grade": "B"},  # partial match for GT[0]
            ]
        }

        result = comparator.compare_diagnostic(expected, actual)

        entries_by_expected = {
            e.expected_index: e for e in result.array_diagnostics["education"].entries
        }
        assert entries_by_expected[0].actual_index == 1
        assert entries_by_expected[0].status == ENTRY_STATUS_PARTIAL_MATCH
        assert entries_by_expected[1].actual_index == 0
        assert entries_by_expected[1].status == ENTRY_STATUS_FULL_MATCH


class TestScalarArraysAreLeftAlone:
    """Diagnostic mode only adds value for arrays of objects
    (work_experience, education) -- arrays of scalars (skills, languages)
    get nothing from entry/field alignment they don't already have from
    compare()'s own whole-array result."""

    def test_scalar_array_field_produces_no_array_diagnostic(self, comparator):
        expected = {"skills": ["Python", "SQL"]}
        actual = {"skills": ["SQL", "Java"]}

        result = comparator.compare_diagnostic(expected, actual)

        assert result.array_diagnostics == {}
        assert result.entry_accuracy == 1.0  # nothing to check -> vacuously 1.0
        assert result.field_accuracy == 1.0
        # ... but the strict result is still exactly what compare() gives.
        assert result.strict_accuracy == 0.0

    def test_empty_object_array_produces_no_mismatches(self, comparator):
        expected = {"work_experience": []}
        actual = {"work_experience": []}

        result = comparator.compare_diagnostic(expected, actual)

        assert result.array_diagnostics == {}
        assert result.strict_accuracy == 1.0


class TestMismatchByField:
    def test_mismatch_by_field_counts_only_matched_entries(self, comparator):
        expected = {
            "work_experience": [
                {"company": "A", "duration": None},  # will be dropped (MISSING)
                {"company": "B", "duration": None},  # will partially match
            ]
        }
        actual = {
            "work_experience": [
                {"company": "B", "duration": "1 year"},
            ]
        }

        result = comparator.compare_diagnostic(expected, actual)

        # Only the matched B/B pair contributes -- the missing "A" entry
        # has no sub-fields to attribute a mismatch to.
        assert result.mismatch_by_field == {
            "work_experience.company": {"total": 1, "matched": 1},
            "work_experience.duration": {"total": 1, "matched": 0},
        }


class TestDiagnosticResultDataclass:
    def test_diagnostic_result_defaults(self):
        result = DiagnosticResult(strict_accuracy=1.0, entry_accuracy=1.0, field_accuracy=1.0)
        assert result.array_diagnostics == {}
        assert result.mismatch_by_field == {}
        assert result.strict_result is None


@pytest.mark.integration
def test_generate_diagnostic_reports_writes_three_files_with_correct_content(tmp_path):
    """Exercises the full diagnostic-comparison + report-writing workflow
    against real files on disk, mirroring
    test_generate_reports_writes_three_csv_files_with_correct_content
    above but for generate_diagnostic_reports()."""
    comparator = JsonComparator()

    diagnostics = {
        "resume_one": comparator.compare_diagnostic(
            {"work_experience": [{"company": "Acme", "duration": None}]},
            {"work_experience": [{"company": "Acme", "duration": "2 years"}]},
        ),
        "resume_two": comparator.compare_diagnostic(
            {"work_experience": [{"company": "Beta", "duration": None}]},
            {"work_experience": [{"company": "Beta", "duration": None}]},
        ),
    }

    comparator.generate_diagnostic_reports(diagnostics, output_dir=tmp_path)

    for filename in (
        "diagnostic_summary.csv",
        "diagnostic_mismatch_by_field.csv",
        "diagnostic_report.md",
    ):
        assert (tmp_path / filename).is_file()

    with (tmp_path / "diagnostic_summary.csv").open(newline="", encoding="utf-8") as f:
        rows = {row["resume_name"]: row for row in csv.DictReader(f)}
    assert rows["resume_one"]["strict_accuracy"] == "0.0000"
    assert rows["resume_one"]["entry_accuracy"] == "1.0000"
    assert rows["resume_one"]["field_accuracy"] == "0.5000"
    assert rows["resume_two"]["strict_accuracy"] == "1.0000"
    assert rows["resume_two"]["field_accuracy"] == "1.0000"

    with (tmp_path / "diagnostic_mismatch_by_field.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        field_rows = {row["field_path"]: row for row in csv.DictReader(f)}
    # company: 2/2 matched across both resumes; duration: 1/2.
    assert field_rows["work_experience.company"]["total"] == "2"
    assert field_rows["work_experience.company"]["matched"] == "2"
    assert field_rows["work_experience.duration"]["total"] == "2"
    assert field_rows["work_experience.duration"]["matched"] == "1"

    report_text = (tmp_path / "diagnostic_report.md").read_text(encoding="utf-8")
    assert "# Diagnostic Comparison Report" in report_text
    assert "Strict accuracy" in report_text
    assert "Entry accuracy" in report_text
    assert "Field accuracy" in report_text
    assert "Mismatch Breakdown by Field" in report_text
    assert "work_experience.duration" in report_text


@pytest.mark.integration
def test_generate_reports_output_unchanged_when_diagnostic_reports_also_written(tmp_path):
    """Requirement 1: writing diagnostic reports into the same directory
    must not alter a single byte of the existing strict reports."""
    comparator = JsonComparator()
    expected = {"work_experience": [{"company": "Acme", "duration": None}]}
    actual = {"work_experience": [{"company": "Acme", "duration": "2 years"}]}

    strict_only_dir = tmp_path / "strict_only"
    comparisons = {"resume_one": comparator.compare(expected, actual)}
    comparator.generate_reports(comparisons, output_dir=strict_only_dir)

    both_dir = tmp_path / "both"
    diagnostics = {"resume_one": comparator.compare_diagnostic(expected, actual)}
    comparator.generate_reports(
        {name: diag.strict_result for name, diag in diagnostics.items()}, output_dir=both_dir
    )
    comparator.generate_diagnostic_reports(diagnostics, output_dir=both_dir)

    for filename in ("resume_accuracy.csv", "field_accuracy.csv", "summary.csv"):
        assert (strict_only_dir / filename).read_text(encoding="utf-8") == (
            both_dir / filename
        ).read_text(encoding="utf-8")
