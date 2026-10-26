"""
tests/test_education_validator.py

Tests for services/education_validator.py (Milestone 18.10, Phase 1 —
the "Quick Win" rules from docs/EDUCATION_VALIDATION.md).

Pure in-memory logic, no external dependencies (no Ollama, no markdown,
no other module's output) — every test runs directly against
EducationValidator with plain dicts, mirroring
tests/test_work_experience_validator.py's own structure.
"""

import copy

import pytest

from services.education_validator import (
    EducationValidator,
    ValidationIssue,
    ValidationReport,
)


def _entry(**overrides):
    """A minimal, otherwise-valid education entry with all nine schema
    fields, overridden as needed per test."""
    base = {
        "institution": "State University",
        "degree": "B.Tech",
        "specialization": "Computer Science",
        "location": None,
        "start_date": None,
        "end_date": None,
        "currently_studying": False,
        "grade": None,
        "description": None,
    }
    base.update(overrides)
    return base


@pytest.fixture
def validator():
    return EducationValidator()


class TestHappyPath:
    def test_empty_array_returns_empty_array_and_empty_report(self, validator):
        validated, report = validator.validate([])

        assert validated == []
        assert report.total_entries == 0
        assert report.issues == []
        assert report.auto_fixed_count == 0
        assert report.flagged_count == 0

    def test_fully_valid_entry_produces_no_issues(self, validator):
        entry = _entry(
            institution="State University",
            degree="B.Tech",
            specialization="Computer Science",
            grade="78%",
            start_date="2018",
            end_date="2022",
            currently_studying=False,
        )

        validated, report = validator.validate([entry])

        assert validated == [entry]
        assert report.total_entries == 1
        assert report.issues == []

    def test_input_list_and_entry_dicts_are_not_mutated(self, validator):
        entry = _entry(end_date="Pursuing", currently_studying=False)
        original = copy.deepcopy([entry])

        validator.validate([entry])

        assert [entry] == original

    def test_returned_entries_are_new_dicts(self, validator):
        entry = _entry()

        validated, _ = validator.validate([entry])

        assert validated[0] is not entry
        assert validated[0] == entry


class TestEDV001OngoingAndCompletionSentinels:
    @pytest.mark.parametrize(
        "sentinel", ["Pursuing", "pursuing", "Ongoing", "In Progress", "Present", "  Ongoing  "]
    )
    def test_ongoing_sentinel_end_date_is_converted(self, validator, sentinel):
        entry = _entry(end_date=sentinel, currently_studying=False)

        validated, report = validator.validate([entry])

        assert validated[0]["end_date"] is None
        assert validated[0]["currently_studying"] is True
        assert len(report.issues) == 1
        issue = report.issues[0]
        assert issue.rule_id == "EDV-001"
        assert issue.auto_fixed is True
        assert issue.entry_index == 0

    def test_bare_completion_word_nulls_end_date_only(self, validator):
        entry = _entry(end_date="Completed", currently_studying=False)

        validated, report = validator.validate([entry])

        assert validated[0]["end_date"] is None
        # currently_studying is left exactly as it was -- a completion
        # word alone doesn't confirm the entry isn't ongoing.
        assert validated[0]["currently_studying"] is False
        assert len(report.issues) == 1
        assert report.issues[0].rule_id == "EDV-001"
        assert report.issues[0].auto_fixed is True

    def test_bare_completion_word_leaves_true_currently_studying_untouched(self, validator):
        entry = _entry(end_date="Completed", currently_studying=True)

        validated, report = validator.validate([entry])

        assert validated[0]["end_date"] is None
        assert validated[0]["currently_studying"] is True

    def test_real_calendar_end_date_is_untouched(self, validator):
        entry = _entry(end_date="2022", currently_studying=False)

        validated, report = validator.validate([entry])

        assert validated[0]["end_date"] == "2022"
        assert report.issues == []

    def test_null_end_date_is_untouched(self, validator):
        entry = _entry(end_date=None, currently_studying=True)

        validated, report = validator.validate([entry])

        assert validated[0]["end_date"] is None
        assert report.issues == []


class TestEDV008GradeEchoesDegreeOrSpecialization:
    def test_grade_exactly_matching_degree_is_nulled(self, validator):
        entry = _entry(degree="B.Tech", grade="B.Tech")

        validated, report = validator.validate([entry])

        assert validated[0]["grade"] is None
        assert len(report.issues) == 1
        assert report.issues[0].rule_id == "EDV-008"
        assert report.issues[0].auto_fixed is True

    def test_grade_exactly_matching_specialization_is_nulled(self, validator):
        entry = _entry(specialization="General", grade="General")

        validated, report = validator.validate([entry])

        assert validated[0]["grade"] is None
        assert report.issues[0].rule_id == "EDV-008"

    def test_grade_substring_of_degree_is_nulled(self, validator):
        entry = _entry(degree="Senior Secondary (12th)", grade="Senior Secondary")

        validated, report = validator.validate([entry])

        assert validated[0]["grade"] is None
        assert report.issues[0].rule_id == "EDV-008"

    def test_case_insensitive_match(self, validator):
        entry = _entry(degree="B.Tech", grade="b.tech")

        validated, report = validator.validate([entry])

        assert validated[0]["grade"] is None

    def test_real_grade_is_untouched(self, validator):
        entry = _entry(degree="B.Tech", specialization="Computer Science", grade="78%")

        validated, report = validator.validate([entry])

        assert validated[0]["grade"] == "78%"
        assert report.issues == []

    def test_short_grade_does_not_falsely_substring_match(self, validator):
        """A short grade value shouldn't trigger a substring false
        positive against an unrelated degree that happens to contain
        the same short run of characters."""
        entry = _entry(degree="BA", grade="A")

        validated, report = validator.validate([entry])

        assert validated[0]["grade"] == "A"
        assert report.issues == []

    def test_null_grade_is_not_flagged(self, validator):
        entry = _entry(grade=None)

        validated, report = validator.validate([entry])

        assert validated[0]["grade"] is None
        assert report.issues == []


class TestEDV009InstitutionEchoesDegree:
    def test_institution_exactly_matching_degree_is_nulled(self, validator):
        entry = _entry(institution="M.Sc. in Information Technology", degree="M.Sc. in Information Technology")

        validated, report = validator.validate([entry])

        assert validated[0]["institution"] is None
        assert len(report.issues) == 1
        assert report.issues[0].rule_id == "EDV-009"
        assert report.issues[0].auto_fixed is True

    def test_institution_containing_degree_text_is_nulled(self, validator):
        entry = _entry(institution="M.Sc. in Information Technology", degree="M.Sc.")

        validated, report = validator.validate([entry])

        assert validated[0]["institution"] is None
        assert report.issues[0].rule_id == "EDV-009"

    def test_real_institution_is_untouched(self, validator):
        entry = _entry(institution="State University", degree="M.Sc.")

        validated, report = validator.validate([entry])

        assert validated[0]["institution"] == "State University"
        assert report.issues == []

    def test_null_institution_is_not_flagged(self, validator):
        entry = _entry(institution=None, degree="M.Sc.")

        validated, report = validator.validate([entry])

        assert validated[0]["institution"] is None
        assert report.issues == []

    def test_null_degree_means_no_comparison_possible(self, validator):
        entry = _entry(institution="State University", degree=None)

        validated, report = validator.validate([entry])

        assert validated[0]["institution"] == "State University"
        assert report.issues == []


class TestEDV010DateFieldMisassignment:
    def test_year_range_with_hyphen_is_split(self, validator):
        entry = _entry(start_date="2008-2010", end_date=None)

        validated, report = validator.validate([entry])

        assert validated[0]["start_date"] == "2008"
        assert validated[0]["end_date"] == "2010"
        assert len(report.issues) == 1
        assert report.issues[0].rule_id == "EDV-010"
        assert report.issues[0].auto_fixed is True

    def test_year_range_with_en_dash_is_split(self, validator):
        entry = _entry(start_date="2008–2010", end_date=None)

        validated, report = validator.validate([entry])

        assert validated[0]["start_date"] == "2008"
        assert validated[0]["end_date"] == "2010"

    def test_year_range_with_spaces_is_split(self, validator):
        entry = _entry(start_date="2008 - 2010", end_date=None)

        validated, report = validator.validate([entry])

        assert validated[0]["start_date"] == "2008"
        assert validated[0]["end_date"] == "2010"

    def test_year_range_with_confirming_end_date_substring_is_split(self, validator):
        entry = _entry(start_date="2008-2010", end_date="2010")

        validated, report = validator.validate([entry])

        assert validated[0]["start_date"] == "2008"
        assert validated[0]["end_date"] == "2010"

    def test_year_range_with_contradicting_end_date_is_left_untouched(self, validator):
        """end_date holds a real, unrelated value -- not null and not a
        substring of start_date -- so there's no confirming signal;
        EDV-010 leaves this alone rather than guessing."""
        entry = _entry(start_date="2008-2010", end_date="2015")

        validated, report = validator.validate([entry])

        assert validated[0]["start_date"] == "2008-2010"
        assert validated[0]["end_date"] == "2015"
        assert report.issues == []

    def test_duplicated_single_date_moves_to_end_date(self, validator):
        entry = _entry(start_date="2019", end_date="2019")

        validated, report = validator.validate([entry])

        assert validated[0]["start_date"] is None
        assert validated[0]["end_date"] == "2019"
        assert len(report.issues) == 1
        assert report.issues[0].rule_id == "EDV-010"
        assert report.issues[0].auto_fixed is True

    def test_distinct_real_dates_are_untouched(self, validator):
        entry = _entry(start_date="2018", end_date="2022")

        validated, report = validator.validate([entry])

        assert validated[0]["start_date"] == "2018"
        assert validated[0]["end_date"] == "2022"
        assert report.issues == []

    def test_both_dates_null_is_untouched(self, validator):
        entry = _entry(start_date=None, end_date=None)

        validated, report = validator.validate([entry])

        assert report.issues == []


class TestEDV010SubPattern3LoneDateMovesToEndDate:
    """Milestone 23.24: a single lone date left only in start_date,
    end_date null, moves to end_date. See
    docs/MILESTONE_23_24_EDUCATION_DATE_SWAP_RESULTS.md for the
    full-dataset evidence this rule is grounded in — every case here
    is a real shape confirmed present in this project's own ground
    truth / current parser output, not a hypothetical."""

    def test_confirmed_swap_case_bhavita_jain_style(self, validator):
        """Real case: Bhavita Jain_GAD_GT entry 0 (CA Final, table's
        own 'Year' column = 'May 2025', ground truth end_date='May
        2025', start_date=None; model output was start_date='May
        2025', end_date=None before this fix)."""
        entry = _entry(degree="CA Final", start_date="May 2025", end_date=None)

        validated, report = validator.validate([entry])

        assert validated[0]["start_date"] is None
        assert validated[0]["end_date"] == "May 2025"
        assert len(report.issues) == 1
        assert report.issues[0].rule_id == "EDV-010"
        assert report.issues[0].auto_fixed is True

    def test_second_independent_swap_case_mukul_kanojia_style(self, validator):
        """Real case: Mukul Kanojia_Full Stack Developer_Intileo entry
        1 (12th Grade, bulleted free-text list, single bare date
        'June 2016', no table, no column header — a structurally
        different source shape than the Bhavita Jain case above, same
        underlying defect)."""
        entry = _entry(degree="12th Grade (PCM)", start_date="June 2016", end_date=None)

        validated, report = validator.validate([entry])

        assert validated[0]["start_date"] is None
        assert validated[0]["end_date"] == "June 2016"

    def test_already_correct_start_and_end_is_unaffected(self, validator):
        """Negative control: a real two-date range (start != end, both
        populated) must never be touched by this sub-pattern."""
        entry = _entry(start_date="2018", end_date="2022")

        validated, report = validator.validate([entry])

        assert validated[0]["start_date"] == "2018"
        assert validated[0]["end_date"] == "2022"
        assert report.issues == []

    def test_incomplete_date_both_null_is_unaffected(self, validator):
        """Negative control: no date at all -- nothing to move."""
        entry = _entry(start_date=None, end_date=None)

        validated, report = validator.validate([entry])

        assert validated[0]["start_date"] is None
        assert validated[0]["end_date"] is None
        assert report.issues == []

    def test_ambiguous_empty_string_end_date_is_unaffected(self, validator):
        """Negative control: end_date is an empty string, not the
        schema's actual null -- this rule only fires on end_date is
        None exactly, never on any other falsy value, so a malformed
        empty-string end_date is left for schema validation to catch
        rather than silently overwritten."""
        entry = _entry(start_date="2019", end_date="")

        validated, report = validator.validate([entry])

        assert validated[0]["start_date"] == "2019"
        assert validated[0]["end_date"] == ""
        assert report.issues == []

    def test_unusual_iso_style_date_formatting_still_moves(self, validator):
        """Real case: Monika Yathamshetti_AM-GAD_GT (CA Final,
        '2022-05' ISO-style year-month, not a Western 'Month YYYY' or
        bare year) -- this rule moves whatever string is present,
        unchanged, regardless of its own internal format."""
        entry = _entry(start_date="2022-05", end_date=None)

        validated, report = validator.validate([entry])

        assert validated[0]["start_date"] is None
        assert validated[0]["end_date"] == "2022-05"

    def test_currently_studying_true_does_not_block_the_move(self, validator):
        """Real case: two dataset entries (Chirag Sharma_AVP_Intileo
        entry 0, Krishankant Soni_GAD-UK_GT-Intileo entry 3) had model
        output with currently_studying=True *and* the lone-date shape
        -- but ground truth confirms currently_studying=False and the
        swap direction is identical to every other case. This rule
        deliberately does not gate on currently_studying (see the
        module docstring and docs/MILESTONE_23_24_EDUCATION_DATE_SWAP_
        RESULTS.md §6 for why: zero ground-truth entries in this
        dataset — including its one true ongoing entry — have the
        start-only/end-null shape, so gating on currently_studying
        would only lose real fixes without preventing any observed
        risk). The currently_studying field itself is untouched by
        this rule either way -- fixing it is a different rule's job,
        out of this milestone's scope."""
        entry = _entry(start_date="Mar 2024", end_date=None, currently_studying=True)

        validated, report = validator.validate([entry])

        assert validated[0]["start_date"] is None
        assert validated[0]["end_date"] == "Mar 2024"
        assert validated[0]["currently_studying"] is True

    def test_range_string_in_start_date_is_handled_by_sub_pattern_1_not_3(self, validator):
        """A range string ('2008-2010') in start_date with end_date
        null matches sub-pattern 1 first (it returns immediately on
        match), so sub-pattern 3 never gets a chance to instead move
        the whole unsplit range string into end_date."""
        entry = _entry(start_date="2008-2010", end_date=None)

        validated, report = validator.validate([entry])

        assert validated[0]["start_date"] == "2008"
        assert validated[0]["end_date"] == "2010"
        assert len(report.issues) == 1

    def test_legitimate_ongoing_entry_via_edv_001_is_not_swapped(self, validator):
        """Critical negative control, found via the full test suite
        (tests/test_parser_pipeline.py's own EducationValidator
        integration test) while implementing this rule: an entry with
        a real start_date and end_date="Pursuing" is a genuinely
        ongoing entry with a known start date and no end date yet.
        EDV-001 (which runs first) correctly converts end_date to null
        and currently_studying to true -- but that null end_date must
        NOT then look, to sub-pattern 3, like the model-omission shape
        it targets. This is exactly the "legitimate case where start/
        end ordering could appear unusual" this rule must never touch
        (Milestone 23.24 Phase 6/8)."""
        entry = _entry(start_date="2015", end_date="Pursuing", currently_studying=False)

        validated, report = validator.validate([entry])

        assert validated[0]["start_date"] == "2015"
        assert validated[0]["end_date"] is None
        assert validated[0]["currently_studying"] is True
        rule_ids = [issue.rule_id for issue in report.issues]
        assert rule_ids == ["EDV-001"]

    def test_bare_completion_word_with_lone_start_date_still_moves(self, validator):
        """Contrast case: a bare completion word ("Completed", no real
        date) does NOT by itself confirm the entry is ongoing (see
        EDV-001's own docstring) -- unlike the ongoing-word branch
        above, this must not suppress sub-pattern 3."""
        entry = _entry(start_date="2019", end_date="Completed", currently_studying=False)

        validated, report = validator.validate([entry])

        assert validated[0]["start_date"] is None
        assert validated[0]["end_date"] == "2019"
        rule_ids = [issue.rule_id for issue in report.issues]
        assert rule_ids == ["EDV-001", "EDV-010"]


class TestEDV011CurrentlyStudyingContradictsGrade:
    def test_true_with_grade_is_corrected(self, validator):
        entry = _entry(currently_studying=True, grade="51.67%")

        validated, report = validator.validate([entry])

        assert validated[0]["currently_studying"] is False
        # end_date is never reconstructed by this rule.
        assert validated[0]["end_date"] is None
        assert len(report.issues) == 1
        issue = report.issues[0]
        assert issue.rule_id == "EDV-011"
        assert issue.auto_fixed is True

    def test_true_with_null_grade_is_untouched(self, validator):
        entry = _entry(currently_studying=True, grade=None)

        validated, report = validator.validate([entry])

        assert validated[0]["currently_studying"] is True
        assert report.issues == []

    def test_false_with_grade_is_untouched(self, validator):
        entry = _entry(currently_studying=False, grade="78%")

        validated, report = validator.validate([entry])

        assert validated[0]["currently_studying"] is False
        assert report.issues == []

    def test_true_with_blank_grade_is_untouched(self, validator):
        entry = _entry(currently_studying=True, grade="   ")

        validated, report = validator.validate([entry])

        assert validated[0]["currently_studying"] is True
        assert report.issues == []


class TestRuleInteraction:
    def test_edv_001_ongoing_fix_prevents_edv_011_from_also_firing(self, validator):
        """An entry with end_date="Pursuing" and a real grade (an
        inconsistent combination the model produced) is corrected by
        EDV-001 to currently_studying=true; EDV-011 then sees that same
        true+grade combination and flips it straight back to false --
        both rules fire, in order, and the entry ends up fully
        consistent (not ongoing, no ongoing-word end_date)."""
        entry = _entry(end_date="Pursuing", currently_studying=False, grade="78%")

        validated, report = validator.validate([entry])

        assert validated[0]["end_date"] is None
        assert validated[0]["currently_studying"] is False
        rule_ids = [issue.rule_id for issue in report.issues]
        assert rule_ids == ["EDV-001", "EDV-011"]

    def test_multiple_entries_each_validated_independently(self, validator):
        entries = [
            _entry(end_date="Pursuing", currently_studying=False),
            _entry(degree="B.Tech", grade="B.Tech"),
            _entry(institution="M.Sc.", degree="M.Sc."),
            _entry(start_date="2008-2010", end_date=None),
            _entry(currently_studying=True, grade="60%"),
        ]

        validated, report = validator.validate(entries)

        assert validated[0]["currently_studying"] is True
        assert validated[1]["grade"] is None
        assert validated[2]["institution"] is None
        assert validated[3]["start_date"] == "2008"
        assert validated[3]["end_date"] == "2010"
        assert validated[4]["currently_studying"] is False
        assert [issue.entry_index for issue in report.issues] == [0, 1, 2, 3, 4]
        assert [issue.rule_id for issue in report.issues] == [
            "EDV-001",
            "EDV-008",
            "EDV-009",
            "EDV-010",
            "EDV-011",
        ]

    def test_all_five_quick_win_rules_can_fire_on_one_entry(self, validator):
        entry = _entry(
            institution="B.Tech in Computer Science",
            degree="B.Tech in Computer Science",
            specialization="Computer Science",
            grade="B.Tech in Computer Science",
            start_date="2008-2010",
            end_date="Pursuing",
            currently_studying=False,
        )

        validated, report = validator.validate([entry])

        rule_ids = {issue.rule_id for issue in report.issues}
        assert rule_ids == {"EDV-001", "EDV-008", "EDV-009", "EDV-010"}
        # EDV-001 sets currently_studying=true from the "Pursuing"
        # end_date; grade is nulled by EDV-008 before EDV-011 ever runs,
        # so EDV-011 has nothing left to contradict and never fires.
        assert validated[0]["grade"] is None
        assert validated[0]["institution"] is None
        assert validated[0]["currently_studying"] is True


class TestValidationReport:
    def test_auto_fixed_and_flagged_counts(self, validator):
        entries = [
            _entry(end_date="Ongoing", currently_studying=False),
            _entry(degree="B.Tech", grade="B.Tech"),
        ]

        _, report = validator.validate(entries)

        assert report.total_entries == 2
        assert report.auto_fixed_count == 2
        assert report.flagged_count == 0

    def test_issues_by_rule(self, validator):
        entries = [
            _entry(end_date="Ongoing", currently_studying=False),
            _entry(end_date="In Progress", currently_studying=False),
            _entry(degree="B.Tech", grade="B.Tech"),
        ]

        _, report = validator.validate(entries)

        assert report.issues_by_rule() == {"EDV-001": 2, "EDV-008": 1}

    def test_no_flag_only_issues_exist_in_phase_1(self, validator):
        """Every Phase 1 Quick Win rule is auto-fixable -- there is no
        flag-only rule implemented in this milestone (EDV-011 corrects
        the boolean; it doesn't merely flag)."""
        entries = [
            _entry(end_date="Ongoing", currently_studying=False),
            _entry(degree="B.Tech", grade="B.Tech"),
            _entry(institution="M.Sc.", degree="M.Sc."),
            _entry(start_date="2008-2010", end_date=None),
            _entry(currently_studying=True, grade="60%"),
        ]

        _, report = validator.validate(entries)

        assert report.flagged_count == 0
        assert all(issue.auto_fixed for issue in report.issues)


class TestValidationIssueAndReportShapeMirrorsWorkExperienceValidator:
    """docs/EDUCATION_VALIDATION.md §4 calls for reusing
    WorkExperienceValidator's ValidationIssue/ValidationReport shape
    (job_index/entry_index, rule_id, severity, message, auto_fixed).
    These tests pin that shape directly."""

    def test_validation_issue_fields(self, validator):
        entry = _entry(end_date="Ongoing", currently_studying=False)

        _, report = validator.validate([entry])

        issue = report.issues[0]
        assert isinstance(issue, ValidationIssue)
        assert isinstance(issue.entry_index, int)
        assert isinstance(issue.rule_id, str)
        assert isinstance(issue.severity, str)
        assert isinstance(issue.message, str)
        assert isinstance(issue.auto_fixed, bool)

    def test_validation_report_type(self, validator):
        _, report = validator.validate([])

        assert isinstance(report, ValidationReport)
