"""
tests/test_work_experience_validator.py

Tests for services/work_experience_validator.py (Milestone 17.6, Phase 1
rules; Milestone 17.7 added the `markdown` parameter to validate()'s
signature, currently unused by every Phase 1 rule).

Pure in-memory logic, no external dependencies (no Ollama, no real
markdown parsing) — every test runs directly against
WorkExperienceValidator with plain dicts. Most tests pass an empty
string for `markdown` since Phase 1 ignores it entirely; a dedicated
TestMarkdownParameterIsAcceptedButIgnored class below confirms that
explicitly, including with real, non-empty markdown text.
"""

import copy

import pytest

from services.work_experience_validator import (
    ValidationIssue,
    ValidationReport,
    WorkExperienceValidator,
)


def _job(**overrides):
    """A minimal, otherwise-valid job entry with all ten schema fields,
    overridden as needed per test."""
    base = {
        "company": "Acme Corp",
        "designation": "Engineer",
        "employment_type": None,
        "location": None,
        "start_date": None,
        "end_date": None,
        "currently_working": False,
        "duration": None,
        "responsibilities": [],
        "technologies": [],
    }
    base.update(overrides)
    return base


@pytest.fixture
def validator():
    return WorkExperienceValidator()


class TestHappyPath:
    def test_empty_array_returns_empty_array_and_empty_report(self, validator):
        validated, report = validator.validate("", [])

        assert validated == []
        assert report.total_jobs == 0
        assert report.issues == []
        assert report.auto_fixed_count == 0
        assert report.flagged_count == 0

    def test_fully_valid_job_produces_no_issues(self, validator):
        """Milestone 23.15: since WEV-016/017/018 need a confident
        job-to-span match to confirm employment_type is genuinely
        supported (an empty-markdown call would otherwise now produce
        a Warning — see TestProvenanceRulesWEV016to018 below for that
        behavior specifically), this "fully valid" happy path is given
        real, matching markdown so the whole pipeline — Phase 1 rules
        plus WEV-015/016/017/018 — genuinely produces zero issues."""
        job = _job(
            company="Analytical Engines Ltd",
            employment_type="Full-time",
            start_date="Jun 2022",
            end_date=None,
            currently_working=True,
            duration=None,
        )
        markdown = (
            "## Experience\n### Analytical Engines Ltd\nFull-time\nJun 2022 - Present\n"
        )

        validated, report = validator.validate(markdown, [job])

        assert validated == [job]
        assert report.total_jobs == 1
        assert report.issues == []

    def test_input_list_and_job_dicts_are_not_mutated(self, validator):
        job = _job(end_date="Present", currently_working=True)
        original = copy.deepcopy([job])

        validator.validate("", [job])

        assert [job] == original


class TestWEV001PresentInEndDate:
    @pytest.mark.parametrize(
        "sentinel", ["Present", "present", "Current", "Till Date", "Ongoing", "Now", "  Present  "]
    )
    def test_sentinel_end_date_is_converted(self, validator, sentinel):
        job = _job(end_date=sentinel, currently_working=False)

        validated, report = validator.validate("", [job])

        assert validated[0]["end_date"] is None
        assert validated[0]["currently_working"] is True
        assert len(report.issues) == 1
        issue = report.issues[0]
        assert issue.rule_id == "WEV-001"
        assert issue.auto_fixed is True
        assert issue.job_index == 0

    def test_real_calendar_end_date_is_untouched(self, validator):
        job = _job(end_date="Jan 2022", currently_working=False)

        validated, report = validator.validate("", [job])

        assert validated[0]["end_date"] == "Jan 2022"
        assert validated[0]["currently_working"] is False
        assert report.issues == []

    def test_null_end_date_is_untouched(self, validator):
        job = _job(end_date=None, currently_working=True)

        validated, report = validator.validate("", [job])

        assert validated[0]["end_date"] is None
        assert report.issues == []


class TestWEV002ComputedDuration:
    def test_duration_exactly_matching_computed_gap_is_nulled(self, validator):
        job = _job(start_date="Jan 2023", end_date="Jan 2024", duration="1 year")

        validated, report = validator.validate("", [job])

        assert validated[0]["duration"] is None
        assert len(report.issues) == 1
        assert report.issues[0].rule_id == "WEV-002"
        assert report.issues[0].auto_fixed is True

    def test_duration_matching_months_and_years_combo(self, validator):
        # Jan 2020 -> Aug 2021 = 19 months = "1 year 7 months".
        job = _job(start_date="Jan 2020", end_date="Aug 2021", duration="1 year 7 months")

        validated, report = validator.validate("", [job])

        assert validated[0]["duration"] is None
        assert report.issues[0].rule_id == "WEV-002"

    def test_duration_not_matching_computed_gap_is_kept(self, validator):
        # Jan 2023 -> Jan 2024 = 12 months, stated duration says otherwise.
        job = _job(start_date="Jan 2023", end_date="Jan 2024", duration="3 years")

        validated, report = validator.validate("", [job])

        assert validated[0]["duration"] == "3 years"
        # Milestone 23.15: no markdown was given, so WEV-016/017/018
        # separately warn that duration's provenance couldn't be
        # confirmed either — that is a different, new rule, not a
        # regression of WEV-002's own behavior under test here.
        assert not any(issue.rule_id == "WEV-002" for issue in report.issues)

    def test_duration_kept_when_end_date_is_null(self, validator):
        """An ongoing role has no computable end date, so there's
        nothing to compare against — duration must be left alone even
        if it happens to equal some plausible-looking value."""
        job = _job(start_date="Jan 2023", end_date=None, currently_working=True, duration="1 year")

        validated, report = validator.validate("", [job])

        assert validated[0]["duration"] == "1 year"
        assert not any(issue.rule_id == "WEV-002" for issue in report.issues)

    def test_duration_kept_when_dates_are_unparseable(self, validator):
        job = _job(start_date="a while back", end_date="recently", duration="1 year")

        validated, report = validator.validate("", [job])

        assert validated[0]["duration"] == "1 year"
        assert not any(issue.rule_id == "WEV-002" for issue in report.issues)

    def test_duration_kept_when_duration_text_has_no_numeric_component(self, validator):
        job = _job(start_date="Jan 2023", end_date="Jan 2024", duration="a few months")

        validated, report = validator.validate("", [job])

        assert validated[0]["duration"] == "a few months"
        assert not any(issue.rule_id == "WEV-002" for issue in report.issues)

    def test_null_duration_is_not_flagged(self, validator):
        job = _job(start_date="Jan 2023", end_date="Jan 2024", duration=None)

        validated, report = validator.validate("", [job])

        assert validated[0]["duration"] is None
        assert report.issues == []

    def test_decimal_years_duration_is_parsed(self, validator):
        # 27 Jul 2015 -> 11 Aug 2023 truncated to year-month = Jul 2015 -> Aug 2023 = 97 months.
        job = _job(start_date="Jul 2015", end_date="Aug 2023", duration="8.08 years")

        validated, report = validator.validate("", [job])

        # 8.08 * 12 = 96.96 -> rounds to 97, matching the computed gap exactly.
        assert validated[0]["duration"] is None
        assert report.issues[0].rule_id == "WEV-002"


class TestWEV003EmploymentTypeWhitelist:
    @pytest.mark.parametrize(
        "allowed",
        [
            "Internship",
            "Freelance",
            "Contract",
            "Part-time",
            "Full-time",
            "Temporary",
            "Apprenticeship",
            "Consultant",
            "full-time",  # case-insensitive match
        ],
    )
    def test_allowed_values_are_kept(self, validator, allowed):
        job = _job(employment_type=allowed)

        validated, report = validator.validate("", [job])

        assert validated[0]["employment_type"] == allowed
        # Milestone 23.15: empty markdown means WEV-016 separately warns
        # that employment_type's provenance couldn't be confirmed — a
        # different, new rule, not a regression of WEV-003 under test here.
        assert not any(issue.rule_id == "WEV-003" for issue in report.issues)

    @pytest.mark.parametrize(
        "disallowed", ["Articleship Firm", "Full Time", "Self-employed", "Trainee", "Member (Virtual)"]
    )
    def test_disallowed_values_are_nulled_not_replaced(self, validator, disallowed):
        job = _job(employment_type=disallowed)

        validated, report = validator.validate("", [job])

        assert validated[0]["employment_type"] is None
        assert len(report.issues) == 1
        assert report.issues[0].rule_id == "WEV-003"
        assert report.issues[0].auto_fixed is True

    def test_null_employment_type_is_not_flagged(self, validator):
        job = _job(employment_type=None)

        validated, report = validator.validate("", [job])

        assert validated[0]["employment_type"] is None
        assert report.issues == []


class TestWEV011InvalidDateOrdering:
    def test_start_after_end_is_flagged_not_corrected(self, validator):
        job = _job(start_date="Jan 2024", end_date="Jan 2023")

        validated, report = validator.validate("", [job])

        # Unchanged — WEV-011 never corrects.
        assert validated[0]["start_date"] == "Jan 2024"
        assert validated[0]["end_date"] == "Jan 2023"
        assert len(report.issues) == 1
        issue = report.issues[0]
        assert issue.rule_id == "WEV-011"
        assert issue.auto_fixed is False

    def test_start_before_end_is_not_flagged(self, validator):
        job = _job(start_date="Jan 2023", end_date="Jan 2024")

        validated, report = validator.validate("", [job])

        assert report.issues == []

    def test_equal_start_and_end_is_not_flagged(self, validator):
        job = _job(start_date="Jan 2023", end_date="Jan 2023")

        validated, report = validator.validate("", [job])

        assert report.issues == []

    def test_missing_dates_are_not_flagged(self, validator):
        job = _job(start_date=None, end_date=None)

        validated, report = validator.validate("", [job])

        assert report.issues == []


class TestWEV013CurrentlyWorkingMismatch:
    def test_true_with_real_end_date_is_corrected(self, validator):
        job = _job(currently_working=True, end_date="Jan 2022")

        validated, report = validator.validate("", [job])

        assert validated[0]["currently_working"] is False
        assert validated[0]["end_date"] == "Jan 2022"
        assert len(report.issues) == 1
        issue = report.issues[0]
        assert issue.rule_id == "WEV-013"
        assert issue.auto_fixed is True

    def test_true_with_null_end_date_is_untouched(self, validator):
        job = _job(currently_working=True, end_date=None)

        validated, report = validator.validate("", [job])

        assert validated[0]["currently_working"] is True
        assert report.issues == []

    def test_false_with_real_end_date_is_untouched(self, validator):
        job = _job(currently_working=False, end_date="Jan 2022")

        validated, report = validator.validate("", [job])

        assert validated[0]["currently_working"] is False
        assert report.issues == []


class TestRuleInteraction:
    def test_wev_001_runs_before_wev_013_so_a_fixed_present_never_double_flags(self, validator):
        """A job with end_date="Present" and currently_working=False (the
        model got both fields wrong) should be corrected by WEV-001 to
        end_date=null/currently_working=true, and WEV-013 must then see
        a null end_date and do nothing further — not flip
        currently_working back to false."""
        job = _job(end_date="Present", currently_working=False)

        validated, report = validator.validate("", [job])

        assert validated[0]["end_date"] is None
        assert validated[0]["currently_working"] is True
        rule_ids = [issue.rule_id for issue in report.issues]
        assert rule_ids == ["WEV-001"]

    def test_multiple_jobs_each_validated_independently(self, validator):
        jobs = [
            _job(end_date="Present", currently_working=False),
            _job(employment_type="Bogus Type"),
            _job(start_date="Jan 2024", end_date="Jan 2023"),
        ]

        validated, report = validator.validate("", jobs)

        assert validated[0]["currently_working"] is True
        assert validated[1]["employment_type"] is None
        assert validated[2]["start_date"] == "Jan 2024"  # unchanged, flagged only
        assert [issue.job_index for issue in report.issues] == [0, 1, 2]
        assert [issue.rule_id for issue in report.issues] == ["WEV-001", "WEV-003", "WEV-011"]


class TestValidationReport:
    def test_auto_fixed_and_flagged_counts(self, validator):
        jobs = [
            _job(end_date="Present", currently_working=False),  # WEV-001, auto-fixed
            _job(start_date="Jan 2024", end_date="Jan 2023"),  # WEV-011, flagged only
        ]

        _, report = validator.validate("", jobs)

        assert report.auto_fixed_count == 1
        assert report.flagged_count == 1

    def test_issues_by_rule_counts_across_jobs(self, validator):
        jobs = [
            _job(end_date="Present", currently_working=False),
            _job(end_date="Current", currently_working=False),
            _job(employment_type="Bogus"),
        ]

        _, report = validator.validate("", jobs)

        counts = report.issues_by_rule()
        assert counts == {"WEV-001": 2, "WEV-003": 1}

    def test_validation_issue_is_a_plain_dataclass_with_expected_fields(self):
        issue = ValidationIssue(
            job_index=0, rule_id="WEV-001", severity="Medium", message="test", auto_fixed=True
        )

        assert issue.job_index == 0
        assert issue.rule_id == "WEV-001"
        assert issue.severity == "Medium"
        assert issue.auto_fixed is True

    def test_validation_report_defaults_to_empty_issues_list(self):
        report = ValidationReport(total_jobs=3)

        assert report.issues == []
        assert report.auto_fixed_count == 0
        assert report.flagged_count == 0
        assert report.issues_by_rule() == {}


class TestMarkdownParameterIsAcceptedButIgnored:
    """Milestone 17.7: validate() gained a `markdown` parameter to
    establish the API future markdown-aware rules will use. Phase 1's
    five original rules (WEV-001, 002, 003, 011, 013) must behave
    identically no matter what markdown is passed — including real,
    substantial, or even actively misleading markdown text.

    Milestone 17.8 added WEV-015, the first rule that actually reads
    `markdown` — but only when a job has a non-empty `technologies`
    list (see TestWEV015TechnologyProvenance below). Every job fixture
    in this class uses `_job()`'s default `technologies=[]`, so WEV-015
    is always a no-op here too — these tests still isolate and confirm
    only the original five rules' indifference to markdown content.
    """

    def test_empty_markdown_string(self, validator):
        job = _job(end_date="Present", currently_working=False)

        validated, report = validator.validate("", [job])

        assert validated[0]["end_date"] is None
        assert len(report.issues) == 1

    def test_real_markdown_text_does_not_change_phase_1_behavior(self, validator):
        job = _job(end_date="Present", currently_working=False)
        real_markdown = (
            "# Ada Lovelace\n\n## Experience\n### Mathematician\n"
            "**Analytical Engines Ltd**\nJan 2020 - Present\n"
        )

        validated, report = validator.validate(real_markdown, [job])

        assert validated[0]["end_date"] is None
        assert validated[0]["currently_working"] is True
        assert len(report.issues) == 1
        assert report.issues[0].rule_id == "WEV-001"

    def test_markdown_that_contradicts_the_json_does_not_change_phase_1_behavior(self, validator):
        """Even markdown that (if it were used) would suggest a
        different conclusion must have zero effect in Phase 1 — proof
        that `markdown` is genuinely inert here, not silently
        influencing the five existing rules."""
        job = _job(
            employment_type="Full-time", start_date="Jan 2020", end_date="Jan 2021", duration="1 year"
        )
        misleading_markdown = "## Experience\n### Something Else Entirely\nUnrelated text.\n"

        validated_a, report_a = validator.validate("", [job])
        validated_b, report_b = validator.validate(misleading_markdown, [job])

        assert validated_a == validated_b
        assert report_a.issues_by_rule() == report_b.issues_by_rule()

    def test_none_markdown_does_not_raise(self, validator):
        """Phase 1 never touches `markdown`, so even a caller passing
        None for it must not raise — reinforcing it's truly unused."""
        job = _job(end_date="Present", currently_working=False)

        validated, report = validator.validate(None, [job])

        assert validated[0]["end_date"] is None


class TestWEV015TechnologyProvenance:
    """Milestone 17.8: WEV-015 — remove any technology not found inside
    a job's own located markdown span. docs/WORK_EXPERIENCE_VALIDATION.md
    is the source of this rule's ID and behavior; the milestone request
    itself referred to it as "WEV-005" (that ID actually names a
    different, unrelated rule — Duplicate companies — in the design
    doc), so it is implemented and tested here as WEV-015 throughout.
    """

    def test_valid_technologies_already_present_in_the_job_span_are_kept(self, validator):
        job = _job(
            company="Analytical Engines Ltd",
            technologies=["Python", "SQL"],
        )
        markdown = (
            "## Experience\n"
            "### Analytical Engines Ltd\n"
            "Jan 2020 - Present\n"
            "- Built systems using Python and SQL.\n"
        )

        validated, report = validator.validate(markdown, [job])

        assert validated[0]["technologies"] == ["Python", "SQL"]
        assert report.issues == []

    def test_close_variant_spelling_is_still_recognized_as_present(self, validator):
        """The design doc's own worked example: "Node JS" vs "Node.js"
        must normalize to the same token sequence and not be removed."""
        job = _job(company="Acme Corp", technologies=["Node.js"])
        markdown = "## Experience\n### Acme Corp\n2020 - 2021\n- Built APIs with Node JS.\n"

        validated, report = validator.validate(markdown, [job])

        assert validated[0]["technologies"] == ["Node.js"]
        assert report.issues == []

    def test_technology_copied_from_skills_section_is_removed(self, validator):
        """A technology that only appears in the resume-wide Skills
        section — never inside this job's own span — must be removed,
        even though it's real text elsewhere in the same document. The
        rule must never search outside the matched JobSpan."""
        job = _job(
            company="Analytical Engines Ltd",
            technologies=["Python", "Kubernetes"],
        )
        markdown = (
            "## Experience\n"
            "### Analytical Engines Ltd\n"
            "Jan 2020 - Present\n"
            "- Built systems using Python.\n"
            "\n"
            "## Skills\n"
            "Python, Kubernetes, Docker\n"
        )

        validated, report = validator.validate(markdown, [job])

        assert validated[0]["technologies"] == ["Python"]
        assert len(report.issues) == 1
        issue = report.issues[0]
        assert issue.rule_id == "WEV-015"
        assert issue.auto_fixed is True
        assert "Kubernetes" in issue.message

    def test_multiple_jobs_are_each_checked_against_their_own_span_only(self, validator):
        job_a = _job(company="Analytical Engines Ltd", technologies=["Python"])
        job_b = _job(company="Babbage & Co", technologies=["COBOL"])
        markdown = (
            "## Experience\n"
            "### Analytical Engines Ltd\n"
            "2020 - 2021\n"
            "- Wrote programs in Python.\n"
            "### Babbage & Co\n"
            "2018 - 2019\n"
            "- Performed calculations using mechanical devices.\n"
        )

        validated, report = validator.validate(markdown, [job_a, job_b])

        assert validated[0]["technologies"] == ["Python"]
        assert validated[1]["technologies"] == []
        assert len(report.issues) == 1
        issue = report.issues[0]
        assert issue.job_index == 1
        assert issue.rule_id == "WEV-015"
        assert "COBOL" in issue.message

    def test_repeated_technology_that_is_present_is_kept_both_times(self, validator):
        """WEV-015 only removes items absent from the span — it doesn't
        de-duplicate (that's a different rule's job)."""
        job = _job(company="Acme Corp", technologies=["Python", "Python"])
        markdown = "## Experience\n### Acme Corp\n2020 - 2021\n- Used Python extensively.\n"

        validated, report = validator.validate(markdown, [job])

        assert validated[0]["technologies"] == ["Python", "Python"]
        assert report.issues == []

    def test_repeated_technology_that_is_absent_is_removed_both_times(self, validator):
        job = _job(company="Acme Corp", technologies=["Rust", "Rust"])
        markdown = "## Experience\n### Acme Corp\n2020 - 2021\n- Used Python extensively.\n"

        validated, report = validator.validate(markdown, [job])

        assert validated[0]["technologies"] == []
        assert len(report.issues) == 1
        assert report.issues[0].auto_fixed is True

    def test_unmatched_job_span_leaves_technologies_unchanged_and_warns(self, validator):
        """A job whose company appears nowhere in the located spans
        (e.g. a hallucinated company, or the section wasn't found at
        all) must never have its technologies guessed at."""
        job = _job(company="Nonexistent Corp", technologies=["Python", "SQL"])
        markdown = "## Experience\n### Analytical Engines Ltd\n2020 - 2021\n- Used Java.\n"

        validated, report = validator.validate(markdown, [job])

        assert validated[0]["technologies"] == ["Python", "SQL"]
        assert len(report.issues) == 1
        issue = report.issues[0]
        assert issue.rule_id == "WEV-015"
        assert issue.severity == "Warning"
        assert issue.auto_fixed is False

    def test_no_experience_section_at_all_leaves_technologies_unchanged_and_warns(self, validator):
        job = _job(company="Analytical Engines Ltd", technologies=["Python"])
        markdown = "# Just a name, no sections at all.\n"

        validated, report = validator.validate(markdown, [job])

        assert validated[0]["technologies"] == ["Python"]
        assert report.issues[0].severity == "Warning"

    def test_overlapping_but_distinct_company_names_are_not_cross_matched(self, validator):
        """"Tech Solutions" and "Tech Solutions Group" share a prefix
        but are different employers — the job named "Tech Solutions"
        must never be checked against "Tech Solutions Group"'s span
        (or vice versa)."""
        job_a = _job(company="Tech Solutions", technologies=["Java"])
        job_b = _job(company="Tech Solutions Group", technologies=["Python"])
        markdown = (
            "## Experience\n"
            "### Tech Solutions Group\n"
            "2022 - 2023\n"
            "- Worked with Python.\n"
            "### Tech Solutions\n"
            "2018 - 2019\n"
            "- Worked with Java.\n"
        )

        validated, report = validator.validate(markdown, [job_a, job_b])

        # Both correctly matched to their OWN, differently-named span —
        # neither job's real technology was wrongly removed.
        assert validated[0]["technologies"] == ["Java"]
        assert validated[1]["technologies"] == ["Python"]
        assert report.issues == []

    def test_same_company_reemployment_is_matched_positionally_in_document_order(self, validator):
        """Two jobs with the identical company name (legitimate
        re-employment, per docs/work_experience_design.md) must each be
        matched to the correct one of the two same-named spans, when
        both the JSON's job order and the document's span order agree
        (both list the earlier stint first) — the common case."""
        job_first_stint = _job(company="Acme Corp", technologies=["Java"])
        job_second_stint = _job(company="Acme Corp", technologies=["Python"])
        markdown = (
            "## Experience\n"
            "### Acme Corp\n"
            "2018 - 2019\n"
            "- First stint: worked with Java.\n"
            "### Acme Corp\n"
            "2022 - 2023\n"
            "- Second stint: worked with Python.\n"
        )

        validated, report = validator.validate(
            markdown, [job_first_stint, job_second_stint]
        )

        assert validated[0]["technologies"] == ["Java"]
        assert validated[1]["technologies"] == ["Python"]
        assert report.issues == []

    def test_same_company_jobs_out_of_chronological_order_can_mismatch_by_position(
        self, validator
    ):
        """Documented limitation: matching is purely positional (N-th
        same-company job <-> N-th same-company span), not
        date-aware. If a resume's JSON job order and its document's
        span order disagree for jobs sharing one company (e.g. JSON
        lists the newer role first, but the document itself lists
        chronologically), positional pairing can attach the wrong
        span. This test documents that tradeoff rather than hiding it —
        WEV-015 does not attempt date-based disambiguation."""
        job_listed_first_in_json = _job(company="Acme Corp", technologies=["Java"])
        job_listed_second_in_json = _job(company="Acme Corp", technologies=["Python"])
        markdown = (
            "## Experience\n"
            "### Acme Corp\n"
            "2022 - 2023\n"
            "- Second stint: worked with Python.\n"
            "### Acme Corp\n"
            "2018 - 2019\n"
            "- First stint: worked with Java.\n"
        )

        validated, report = validator.validate(
            markdown, [job_listed_first_in_json, job_listed_second_in_json]
        )

        # The first JSON job (real technology: Java) is matched to the
        # document's first span (which actually discusses Python), so
        # its real "Java" is incorrectly removed — the known, accepted
        # cost of positional-only matching without date awareness.
        assert validated[0]["technologies"] == []
        assert validated[1]["technologies"] == []

    def test_more_same_company_jobs_than_spans_leaves_the_overflow_job_unmatched(self, validator):
        job_a = _job(company="Acme Corp", technologies=["Java"])
        job_b = _job(company="Acme Corp", technologies=["Python"])
        markdown = "## Experience\n### Acme Corp\n2018 - 2019\n- Worked with Java.\n"

        validated, report = validator.validate(markdown, [job_a, job_b])

        assert validated[0]["technologies"] == ["Java"]
        # No second span exists for the second "Acme Corp" job — left
        # unchanged and flagged, not guessed at using the first span.
        assert validated[1]["technologies"] == ["Python"]
        assert len(report.issues) == 1
        assert report.issues[0].job_index == 1
        assert report.issues[0].severity == "Warning"

    def test_empty_technologies_produces_no_issue_even_when_unmatched(self, validator):
        job = _job(company="Nonexistent Corp", technologies=[])
        markdown = "## Experience\n### Analytical Engines Ltd\n2020 - 2021\n- Used Java.\n"

        validated, report = validator.validate(markdown, [job])

        assert validated[0]["technologies"] == []
        assert report.issues == []

    def test_missing_company_leaves_technologies_unchanged_and_warns(self, validator):
        job = _job(company=None, technologies=["Python"])
        markdown = "## Experience\n### Analytical Engines Ltd\n2020 - 2021\n- Used Python.\n"

        validated, report = validator.validate(markdown, [job])

        assert validated[0]["technologies"] == ["Python"]
        assert report.issues[0].severity == "Warning"

    def test_input_job_dicts_are_not_mutated_by_wev_015(self, validator):
        job = _job(company="Acme Corp", technologies=["Python", "Rust"])
        original = copy.deepcopy(job)
        markdown = "## Experience\n### Acme Corp\n2020 - 2021\n- Used Python.\n"

        validator.validate(markdown, [job])

        assert job == original

    def test_custom_locator_is_used_when_injected(self):
        """Milestone 17.10: the validator now delegates matching itself
        to the injected locator's own match_jobs_to_spans() (not just
        locate()) — this test configures both."""
        from unittest.mock import MagicMock

        from services.work_experience_locator import JobSpan, WorkExperienceLocator

        located_spans = [
            JobSpan(
                company="Analytical Engines Ltd",
                designation="Mathematician",
                date_range="Jan 2020 - Present",
                markdown_text="Only Python is mentioned here.",
                start_line=1,
                end_line=1,
            )
        ]
        matched_span = JobSpan(
            company="Analytical Engines Ltd",
            designation="Mathematician",
            date_range="Jan 2020 - Present",
            markdown_text="Only Python is mentioned here.",
            start_line=1,
            end_line=1,
            confidence=0.97,
        )
        custom_locator = MagicMock(spec=WorkExperienceLocator)
        custom_locator.locate.return_value = located_spans
        custom_locator.match_jobs_to_spans.return_value = {0: matched_span}
        validator = WorkExperienceValidator(locator=custom_locator)
        job = _job(company="Analytical Engines Ltd", technologies=["Python", "SQL"])

        validated, report = validator.validate("irrelevant markdown", [job])

        custom_locator.locate.assert_called_once_with("irrelevant markdown")
        # Not asserting the exact job-list argument here: _apply_wev_015
        # mutates that same (copied) job dict's `technologies` in place
        # right after this call, so by the time the assertion runs, the
        # mock's recorded reference reflects the *post*-mutation state —
        # a plain equality check against the pre-mutation `job` would
        # spuriously fail. The spans argument isn't mutated, so it's
        # still safe to assert exactly.
        custom_locator.match_jobs_to_spans.assert_called_once()
        call_args = custom_locator.match_jobs_to_spans.call_args
        assert len(call_args.args[0]) == 1
        assert call_args.args[0][0]["company"] == "Analytical Engines Ltd"
        assert call_args.args[1] == located_spans
        assert validated[0]["technologies"] == ["Python"]
        assert report.issues[0].rule_id == "WEV-015"
        assert report.issues[0].confidence == 0.97


class TestProvenanceRulesWEV016to018:
    """Milestone 23.15: WEV-016 (employment_type), WEV-017 (location),
    WEV-018 (duration) — null each field if the job has a confidently
    matched markdown span and the value cannot be found (as a plain
    normalized substring, not WEV-015's whole-token match — see the
    module docstring's "Matching strategy" section for why) anywhere in
    that job's own span.

    Several fixtures below reuse real company names, real span text
    excerpts, and real hallucination/legitimate-value pairs verified
    directly against this project's own 23-resume benchmark dataset
    during the Milestone 23.15 investigation, rather than only
    synthetic examples:

    - Dattatray S. Phad_Desktop Support Engineer_Intileo.md: the
      "Nex-Z Solutions (Freelancer)" job, ground-truth
      employment_type="Freelance" — a real, punctuation-containing
      legitimate value that a whole-token match would incorrectly
      null (see the module docstring), used here as the positive
      control that justifies this rule's plain-substring strategy.
    - Anandu Chandran_Area Credit Manager_SMFG-Intileo.md: the
      candidate's own address, "Punalur-691333, Kollam District,
      Kerala, India", is duplicated by the model as a hallucinated
      job location and "Full-time" employment_type on jobs whose own
      markdown span never mentions either — the real negative
      control this rule targets (Milestone 23.14's 111/265 finding).
    - Nilesh-Shivram-Magare-CV-Updated (1).md: job0 ("Quatre Agro
      Enterprises Pvt. Ltd. (Crystal Logistics Cool Chain Ltd.)") has
      a hallucinated duration "2 yrs 3 mos" nowhere in its span; job1
      ("Ambetronics Engineers Pvt. Ltd (4 yrs)") has a real, present
      "4 yrs" duration stated directly in the company heading.
    """

    # ---- 1-3: unsupported value -> null, one per field -------------

    def test_unsupported_employment_type_is_nulled(self, validator):
        """Real negative control: Anandu Chandran's repeated
        "Full-time" employment_type never appears anywhere in the
        job's own markdown span."""
        job = _job(
            company="Larsen & Turbo (L&T) Micro Finance Ltd",
            designation="FLO",
            employment_type="Full-time",
        )
        markdown = (
            "## Experience:\n\n"
            "- **Larsen & Turbo (L&T) Micro Finance Ltd**\n"
            "  - Worked as FLO from December 04 2017 to September 21 2020.\n"
            "    - Responsibilities: Sales, Operations, and Credit support\n"
        )

        validated, report = validator.validate(markdown, [job])

        assert validated[0]["employment_type"] is None
        matching = [issue for issue in report.issues if issue.rule_id == "WEV-016"]
        assert len(matching) == 1
        assert matching[0].auto_fixed is True
        assert matching[0].job_index == 0

    def test_unsupported_location_is_nulled(self, validator):
        """Real negative control: Anandu Chandran's own home address
        ("Punalur...Kerala, India") is hallucinated onto this job's
        location, but the job's own span never states a location."""
        job = _job(
            company="Larsen & Turbo (L&T) Micro Finance Ltd",
            designation="FLO",
            location="Punalur, Kerala, India",
        )
        markdown = (
            "## Experience:\n\n"
            "- **Larsen & Turbo (L&T) Micro Finance Ltd**\n"
            "  - Worked as FLO from December 04 2017 to September 21 2020.\n"
            "    - Responsibilities: Sales, Operations, and Credit support\n"
        )

        validated, report = validator.validate(markdown, [job])

        assert validated[0]["location"] is None
        matching = [issue for issue in report.issues if issue.rule_id == "WEV-017"]
        assert len(matching) == 1
        assert matching[0].auto_fixed is True

    def test_unsupported_duration_is_nulled(self, validator):
        """Real negative control: Nilesh Shivram Magare's job0 has a
        hallucinated "2 yrs 3 mos" duration; the real span (unlike
        job1's "(4 yrs)") never states a duration at all."""
        job = _job(
            company="Quatre Agro Enterprises Pvt. Ltd. (Crystal Logistics Cool Chain Ltd.)",
            designation="Technical Support Coordinator",
            duration="2 yrs 3 mos",
        )
        markdown = (
            "## Experience\n\n"
            "### Technical Support Coordinator\n"
            "**Quatre Agro Enterprises Pvt. Ltd. (Crystal Logistics Cool Chain Ltd.)**\n"
            "- Pre-sales technical validation and vendor sourcing.\n"
            "- Providing in-house technical support to other team members.\n\n"
            "**Goregaon, Mumbai**\n"
            "01/2026 – Present\n"
        )

        validated, report = validator.validate(markdown, [job])

        assert validated[0]["duration"] is None
        matching = [issue for issue in report.issues if issue.rule_id == "WEV-018"]
        assert len(matching) == 1
        assert matching[0].auto_fixed is True

    # ---- 4-6: source-supported value -> unchanged, one per field ---

    def test_source_supported_employment_type_is_kept(self, validator):
        """Real positive control: Dattatray Phad's "Freelance"
        employment_type is genuinely derived from the company
        heading's "(Freelancer)" parenthetical — this must survive
        as a plain normalized substring even though "freelance" is
        not a whole token inside "freelancer" (WEV-015's own
        whole-token convention would wrongly null this)."""
        job = _job(
            company="Nex-Z Solutions (Freelancer)",
            designation="IT Support Specialist",
            employment_type="Freelance",
        )
        markdown = (
            "## Nex-Z Solutions (Freelancer)\n"
            "**IT Support Specialist**\n"
            "Mumbai, India | January 2021 – May 2023\n\n"
            "- Provided hardware repair, upgrades, and virus protection services "
            "for clients\n"
        )

        validated, report = validator.validate(markdown, [job])

        assert validated[0]["employment_type"] == "Freelance"
        assert not any(issue.rule_id == "WEV-016" for issue in report.issues)

    def test_source_supported_location_is_kept(self, validator):
        """Real positive control: Dattatray Phad's "Mumbai, India"
        location is stated verbatim in the job's own span."""
        job = _job(
            company="Nex-Z Solutions (Freelancer)",
            designation="IT Support Specialist",
            location="Mumbai, India",
        )
        markdown = (
            "## Nex-Z Solutions (Freelancer)\n"
            "**IT Support Specialist**\n"
            "Mumbai, India | January 2021 – May 2023\n\n"
            "- Provided hardware repair, upgrades, and virus protection services "
            "for clients\n"
        )

        validated, report = validator.validate(markdown, [job])

        assert validated[0]["location"] == "Mumbai, India"
        assert not any(issue.rule_id == "WEV-017" for issue in report.issues)

    def test_source_supported_duration_is_kept(self, validator):
        """Real positive control: Nilesh Shivram Magare's job1 states
        "(4 yrs)" directly in its own company heading."""
        job = _job(
            company="Ambetronics Engineers Pvt. Ltd",
            designation="Senior Engineer in Customer Support and Service",
            duration="4 yrs",
        )
        markdown = (
            "### Senior Engineer in Customer Support and Service\n"
            "**Ambetronics Engineers Pvt. Ltd (4 yrs)**\n"
            "- Lead a support team of 10 to 12 service engineers.\n\n"
            "**Andheri, Mumbai**\n"
            "09/2021 – 12/2025\n"
        )

        validated, report = validator.validate(markdown, [job])

        assert validated[0]["duration"] == "4 yrs"
        assert not any(issue.rule_id == "WEV-018" for issue in report.issues)

    # ---- 7-8: null/empty input handling -----------------------------

    def test_null_input_remains_null(self, validator):
        """A job with all three fields already null and a confident
        span match must stay null and produce no provenance issues —
        there's nothing to check provenance for."""
        job = _job(
            company="Acme Corp",
            designation="Engineer",
            employment_type=None,
            location=None,
            duration=None,
        )
        markdown = "## Experience\n### Acme Corp\n2020 - 2021\n- Did engineering work.\n"

        validated, report = validator.validate(markdown, [job])

        assert validated[0]["employment_type"] is None
        assert validated[0]["location"] is None
        assert validated[0]["duration"] is None
        assert not any(
            issue.rule_id in ("WEV-016", "WEV-017", "WEV-018") for issue in report.issues
        )

    def test_empty_string_input_is_left_unchanged(self, validator):
        """Per the same convention already used by WEV-002/WEV-003 for
        an empty/whitespace-only string (`if not value.strip(): return`
        / `continue`): the value is left exactly as-is, not converted
        to null and not treated as "unsupported" — there is nothing to
        check provenance for."""
        job = _job(
            company="Acme Corp",
            designation="Engineer",
            employment_type="",
            location="   ",
            duration="",
        )
        markdown = "## Experience\n### Acme Corp\n2020 - 2021\n- Did engineering work.\n"

        validated, report = validator.validate(markdown, [job])

        assert validated[0]["employment_type"] == ""
        assert validated[0]["location"] == "   "
        assert validated[0]["duration"] == ""
        assert not any(
            issue.rule_id in ("WEV-016", "WEV-017", "WEV-018") for issue in report.issues
        )

    # ---- 9: unrelated fields byte-for-byte unchanged -----------------

    def test_unrelated_fields_are_byte_for_byte_unchanged(self, validator):
        job = _job(
            company="Larsen & Turbo (L&T) Micro Finance Ltd",
            designation="FLO",
            employment_type="Full-time",  # hallucinated, will be nulled
            location="Punalur, Kerala, India",  # hallucinated, will be nulled
            start_date="Dec 2017",
            end_date="Sep 2020",
            currently_working=False,
            duration=None,
            responsibilities=["Sales", "Operations", "Credit support"],
            technologies=[],
        )
        markdown = (
            "## Experience:\n\n"
            "- **Larsen & Turbo (L&T) Micro Finance Ltd**\n"
            "  - Worked as FLO from December 04 2017 to September 21 2020.\n"
            "    - Responsibilities: Sales, Operations, and Credit support\n"
        )

        validated, report = validator.validate(markdown, [job])

        # The two hallucinated fields under test WERE changed...
        assert validated[0]["employment_type"] is None
        assert validated[0]["location"] is None
        # ...but every other field is untouched, including its own
        # identity/date/array fields.
        assert validated[0]["company"] == "Larsen & Turbo (L&T) Micro Finance Ltd"
        assert validated[0]["designation"] == "FLO"
        assert validated[0]["start_date"] == "Dec 2017"
        assert validated[0]["end_date"] == "Sep 2020"
        assert validated[0]["currently_working"] is False
        assert validated[0]["duration"] is None
        assert validated[0]["responsibilities"] == ["Sales", "Operations", "Credit support"]
        assert validated[0]["technologies"] == []

    # ---- 10: multiple jobs checked independently ---------------------

    def test_multiple_jobs_are_each_checked_against_their_own_span_only(self, validator):
        job_hallucinated = _job(
            company="Larsen & Turbo (L&T) Micro Finance Ltd",
            designation="FLO",
            employment_type="Full-time",
            location="Punalur, Kerala, India",
        )
        job_legitimate = _job(
            company="Nex-Z Solutions (Freelancer)",
            designation="IT Support Specialist",
            employment_type="Freelance",
            location="Mumbai, India",
        )
        markdown = (
            "## Experience:\n\n"
            "- **Larsen & Turbo (L&T) Micro Finance Ltd**\n"
            "  - Worked as FLO from December 04 2017 to September 21 2020.\n\n"
            "## Nex-Z Solutions (Freelancer)\n"
            "**IT Support Specialist**\n"
            "Mumbai, India | January 2021 – May 2023\n"
        )

        validated, report = validator.validate(markdown, [job_hallucinated, job_legitimate])

        assert validated[0]["employment_type"] is None
        assert validated[0]["location"] is None
        assert validated[1]["employment_type"] == "Freelance"
        assert validated[1]["location"] == "Mumbai, India"
        job0_rules = {issue.rule_id for issue in report.issues if issue.job_index == 0}
        job1_rules = {issue.rule_id for issue in report.issues if issue.job_index == 1}
        assert {"WEV-016", "WEV-017"} <= job0_rules
        assert not ({"WEV-016", "WEV-017"} & job1_rules)

    # ---- 11: punctuation-containing legitimate values not nulled -----

    def test_legitimate_value_containing_punctuation_is_not_incorrectly_nulled(self, validator):
        """Same real Dattatray Phad case as the positive controls
        above, isolated here specifically for the punctuation
        question the milestone's validation question raises: the
        value "Freelance" must match inside the punctuation-bearing
        span text "(Freelancer)" via plain normalized substring."""
        job = _job(
            company="Nex-Z Solutions (Freelancer)",
            designation="IT Support Specialist",
            employment_type="Freelance",
        )
        markdown = "## Nex-Z Solutions (Freelancer)\n**IT Support Specialist**\nMumbai, India | January 2021 – May 2023\n"

        validated, report = validator.validate(markdown, [job])

        assert validated[0]["employment_type"] == "Freelance"
        assert not any(issue.rule_id == "WEV-016" for issue in report.issues)

    # ---- 12: existing WEV-015 technology provenance unchanged --------

    def test_wev_015_technology_provenance_behavior_is_unchanged_alongside_new_rules(
        self, validator
    ):
        """WEV-016/017/018 must not interfere with WEV-015's own,
        already-shipped technology-provenance behavior when both run
        against the same job in the same validate() call."""
        job = _job(
            company="Acme Corp",
            designation="Engineer",
            employment_type="Full-time",  # unsupported -> nulled by WEV-016
            technologies=["Python", "Kubernetes"],  # Kubernetes unsupported -> WEV-015
        )
        markdown = (
            "## Experience\n### Acme Corp\n2020 - 2021\n"
            "- Built systems using Python.\n"
            "\n## Skills\nPython, Kubernetes, Docker\n"
        )

        validated, report = validator.validate(markdown, [job])

        assert validated[0]["employment_type"] is None
        assert validated[0]["technologies"] == ["Python"]
        rule_ids = {issue.rule_id for issue in report.issues}
        assert "WEV-016" in rule_ids
        assert "WEV-015" in rule_ids

    # ---- unmatched-span safety net (mirrors WEV-015's own test) ------

    def test_unmatched_job_span_leaves_all_three_fields_unchanged_and_warns(self, validator):
        job = _job(
            company="Nonexistent Corp",
            employment_type="Full-time",
            location="Somewhere",
            duration="2 years",
        )
        markdown = "## Experience\n### Analytical Engines Ltd\n2020 - 2021\n- Used Java.\n"

        validated, report = validator.validate(markdown, [job])

        assert validated[0]["employment_type"] == "Full-time"
        assert validated[0]["location"] == "Somewhere"
        assert validated[0]["duration"] == "2 years"
        matching = [issue for issue in report.issues if issue.rule_id == "WEV-016/017/018"]
        assert len(matching) == 1
        assert matching[0].severity == "Warning"
        assert matching[0].auto_fixed is False

    def test_custom_locator_is_used_when_injected(self):
        """Milestone 23.15: confirms WEV-016/017/018 use the same
        injected locator's match_jobs_to_spans() result already
        computed once for WEV-015 (no second, separate lookup)."""
        from unittest.mock import MagicMock

        from services.work_experience_locator import JobSpan, WorkExperienceLocator

        located_spans = [
            JobSpan(
                company="Analytical Engines Ltd",
                designation="Mathematician",
                date_range="Jan 2020 - Present",
                markdown_text="Full-time role in London.",
                start_line=1,
                end_line=1,
            )
        ]
        matched_span = JobSpan(
            company="Analytical Engines Ltd",
            designation="Mathematician",
            date_range="Jan 2020 - Present",
            markdown_text="Full-time role in London.",
            start_line=1,
            end_line=1,
            confidence=0.9,
        )
        custom_locator = MagicMock(spec=WorkExperienceLocator)
        custom_locator.locate.return_value = located_spans
        custom_locator.match_jobs_to_spans.return_value = {0: matched_span}
        validator = WorkExperienceValidator(locator=custom_locator)
        job = _job(company="Analytical Engines Ltd", employment_type="Freelance")

        validated, report = validator.validate("irrelevant markdown", [job])

        custom_locator.locate.assert_called_once_with("irrelevant markdown")
        custom_locator.match_jobs_to_spans.assert_called_once()
        # "Freelance" is not present in "Full-time role in London." ->
        # nulled, proving the injected locator's span was genuinely used.
        assert validated[0]["employment_type"] is None
        assert any(issue.rule_id == "WEV-016" for issue in report.issues)
