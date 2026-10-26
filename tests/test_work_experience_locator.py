"""
tests/test_work_experience_locator.py

Tests for services/work_experience_locator.py (Milestone 17.7).

Pure in-memory logic, no external dependencies — every test runs
directly against WorkExperienceLocator with plain markdown strings.
Several tests use real markdown structure patterns confirmed to exist
in this project's own dataset (see the module docstring and
docs/WORK_EXPERIENCE_LOCATOR_RESULTS.md), not just idealized examples.
"""

import pytest

from services.work_experience_locator import JobSpan, WorkExperienceLocator


@pytest.fixture
def locator():
    return WorkExperienceLocator()


def _job(**overrides):
    """A minimal work_experience job dict, overridden as needed."""
    base = {
        "company": "Acme Corp",
        "designation": "Engineer",
        "start_date": None,
        "end_date": None,
    }
    base.update(overrides)
    return base


def _span(**overrides):
    """A minimal JobSpan, overridden as needed."""
    base = {
        "company": "Acme Corp",
        "designation": "Engineer",
        "date_range": None,
        "markdown_text": "Acme Corp\nEngineer",
        "start_line": 1,
        "end_line": 2,
    }
    base.update(overrides)
    return JobSpan(**base)


class TestNoExperienceSection:
    def test_no_heading_at_all_returns_empty_list(self, locator):
        text = "# Ada Lovelace\n\n## Skills\n\n- Mathematics\n"

        assert locator.locate(text) == []

    def test_empty_string_returns_empty_list(self, locator):
        assert locator.locate("") == []

    def test_experience_like_word_inside_a_sentence_is_not_a_section(self, locator):
        """A heading merely mentioning "experience" in passing (not
        being ITSELF one of the recognized section names) must not be
        treated as the work experience section."""
        text = "# Resume\n\n## My Experience With Cloud Migration Projects\n\nSome text.\n"

        assert locator.locate(text) == []


class TestSingleJob:
    def test_single_job_with_heading_marker(self, locator):
        text = (
            "# Ada Lovelace\n"
            "\n"
            "## Experience\n"
            "### Mathematician\n"
            "**Analytical Engines Ltd**\n"
            "Jan 2020 - Present\n"
            "- Wrote the first published algorithm\n"
            "\n"
            "## Education\n"
            "### Mathematics\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 1
        span = spans[0]
        assert isinstance(span, JobSpan)
        assert span.designation == "Mathematician"
        assert span.company == "Analytical Engines Ltd"
        assert span.date_range == "Jan 2020 - Present"
        assert "Wrote the first published algorithm" in span.markdown_text

    def test_single_job_is_the_only_content_and_section_runs_to_end_of_document(self, locator):
        text = "## Experience\n### Engineer\n**Acme Corp**\nJan 2020 - Dec 2021\n- Did things\n"

        spans = locator.locate(text)

        assert len(spans) == 1
        assert spans[0].company == "Acme Corp"
        assert spans[0].end_line == 5

    def test_start_and_end_line_are_1_indexed_and_bound_the_block(self, locator):
        text = (
            "## Experience\n"  # line 1
            "### Engineer\n"  # line 2
            "**Acme Corp**\n"  # line 3
            "Jan 2020 - Dec 2021\n"  # line 4
            "\n"
            "## Education\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 1
        assert spans[0].start_line == 2
        assert spans[0].end_line == 4


class TestMultipleJobs:
    def test_multiple_jobs_consistent_heading_level(self, locator):
        text = (
            "## Experience\n"
            "### Senior Engineer\n"
            "**Ambetronics Engineers Pvt. Ltd**\n"
            "09/2021 - Present\n"
            "- Managed a team\n"
            "\n"
            "### Maintenance Engineer\n"
            "**Amit Engineering & Construction Works**\n"
            "10/2019 - 08/2021\n"
            "- Kept things running\n"
            "\n"
            "## Education\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 2
        assert spans[0].designation == "Senior Engineer"
        assert spans[0].company == "Ambetronics Engineers Pvt. Ltd"
        assert spans[1].designation == "Maintenance Engineer"
        assert spans[1].company == "Amit Engineering & Construction Works"
        # Preserves document order.
        assert spans[0].start_line < spans[1].start_line

    def test_deeper_sub_heading_within_a_job_is_not_a_new_block(self, locator):
        """A '#### Responsibilities:' sub-label inside one job (a real
        pattern confirmed in this project's own generated markdown)
        must stay part of that job's block, not start a new one."""
        text = (
            "## Experience\n"
            "### Senior Engineer\n"
            "**Ambetronics Engineers Pvt. Ltd**\n"
            "09/2021 - Present\n"
            "\n"
            "#### Responsibilities:\n"
            "- Managed a team of 10-12 service engineers\n"
            "- Planning engineers' schedules\n"
            "\n"
            "### Maintenance Engineer\n"
            "**Amit Engineering & Construction Works**\n"
            "10/2019 - 08/2021\n"
            "\n"
            "## Education\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 2
        assert "Responsibilities" in spans[0].markdown_text
        assert "Managed a team" in spans[0].markdown_text
        assert spans[1].company == "Amit Engineering & Construction Works"

    def test_inconsistent_heading_levels_across_jobs_still_recovers_most_jobs(self, locator):
        """Confirmed real pattern (Dattatray S. Phad): the section
        heading is one level, the first job has no heading of its own
        (just bold text), and subsequent jobs use a heading level
        shallower than the section heading itself."""
        text = (
            "### Professional Experience\n"
            "**Lauren Information Technologies Pvt. Ltd**\n"
            "*IT Desktop Support Engineer*\n"
            "Jun 2025 – Present\n"
            "- Provide end-to-end technical support\n"
            "\n"
            "## IT HUB Computers\n"
            "**Desktop Support Engineer**\n"
            "June 2023 – March 2025\n"
            "- Delivered desktop support\n"
            "\n"
            "## Crystal Solutions Ltd\n"
            "**Hardware Assistant Trainee**\n"
            "July 2019 – September 2019\n"
            "- Assisted in installing desktops\n"
            "\n"
            "# EDUCATION\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 3
        assert spans[0].company == "Lauren Information Technologies Pvt. Ltd"
        assert "Provide end-to-end technical support" in spans[0].markdown_text
        assert spans[2].company == "Crystal Solutions Ltd"

    def test_section_heading_word_that_overlaps_a_non_experience_keyword_is_not_a_false_boundary(
        self, locator
    ):
        """Confirmed real bug found and fixed during development: a job
        titled '...& Projects' must not be mistaken for the start of a
        Projects section just because it contains that word — the
        heading is long/sentence-like, unlike a real section heading."""
        text = (
            "## Experience\n"
            "### Process Lead – Program Management\n"
            "* 04/2023 - Current\n"
            "IDFC First Bank\n"
            "- Defined enterprise-wide transformation roadmaps\n"
            "\n"
            "### Manager – Digital Transformation & Projects\n"
            "* 10/2020 - 03/2023\n"
            "IDFC First Bank\n"
            "- Led multiple digital transformation initiatives\n"
            "\n"
            "## Education\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 2
        assert spans[1].designation == "Manager – Digital Transformation & Projects"
        assert "Led multiple digital transformation initiatives" in spans[1].markdown_text


class TestOverlappingDateRanges:
    def test_two_concurrent_jobs_are_still_returned_as_two_separate_spans(self, locator):
        """Simultaneous roles are legitimate (see
        docs/work_experience_design.md) — the locator must not merge
        them just because their date ranges overlap; it only ever
        splits on structural markers, never on date logic."""
        text = (
            "## Experience\n"
            "### Full-time Engineer\n"
            "**Day Job Inc**\n"
            "Jan 2020 - Dec 2022\n"
            "- Did the main job\n"
            "\n"
            "### Evening Consultant\n"
            "**Side Gig LLC**\n"
            "Jun 2021 - Dec 2022\n"
            "- Did consulting work concurrently\n"
            "\n"
            "## Education\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 2
        assert spans[0].date_range == "Jan 2020 - Dec 2022"
        assert spans[1].date_range == "Jun 2021 - Dec 2022"
        assert spans[0].company == "Day Job Inc"
        assert spans[1].company == "Side Gig LLC"


class TestMissingCompany:
    def test_designation_only_heading_with_no_company_line_at_all(self, locator):
        text = (
            "## Experience\n"
            "### Freelance Writer\n"
            "Jan 2020 - Present\n"
            "- Wrote articles for various clients\n"
            "\n"
            "## Education\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 1
        assert spans[0].designation == "Freelance Writer"
        assert spans[0].company is None

    def test_company_line_present_but_not_recognizable_as_a_company(self, locator):
        """The line following the heading doesn't contain any of the
        recognized company-suffix keywords, so company stays None
        rather than guessing — matching the "if detectable" contract."""
        text = (
            "## Experience\n"
            "### Backend Engineer\n"
            "Riverside\n"
            "Jan 2020 - Present\n"
            "- Built APIs\n"
            "\n"
            "## Education\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 1
        assert spans[0].company is None
        assert spans[0].designation == "Backend Engineer"


class TestUnusualHeadings:
    @pytest.mark.parametrize(
        "heading",
        [
            "## Experience",
            "## Work Experience",
            "## Professional Experience",
            "## Employment History",
            "## Career History",
            "## Work History",
            "## EXPERIENCE",
            "## experience",
            "## Experience:",
            "# Experience",
            "### Experience",
        ],
    )
    def test_recognized_section_heading_variants(self, locator, heading):
        text = f"{heading}\n### Engineer\n**Acme Corp**\nJan 2020 - Present\n\n## Education\n"

        spans = locator.locate(text)

        assert len(spans) == 1
        assert spans[0].company == "Acme Corp"

    def test_unrecognized_heading_is_not_treated_as_experience_section(self, locator):
        text = "## Random Section\n### Engineer\n**Acme Corp**\nJan 2020 - Present\n"

        assert locator.locate(text) == []

    def test_section_ends_at_first_of_several_known_non_experience_headings(self, locator):
        text = (
            "## Experience\n"
            "### Engineer\n"
            "**Acme Corp**\n"
            "Jan 2020 - Present\n"
            "\n"
            "## Certifications\n"
            "- AWS Certified\n"
            "\n"
            "## Education\n"
            "### BSc\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 1
        assert "AWS Certified" not in spans[0].markdown_text


class TestMarkdownFormattingVariations:
    def test_bold_only_markers_no_headings(self, locator):
        text = (
            "## Experience\n"
            "**Acme Corp**\n"
            "*Engineer*\n"
            "Jan 2020 - Dec 2021\n"
            "- Did things\n"
            "\n"
            "**Other Corp**\n"
            "*Senior Engineer*\n"
            "Jan 2022 - Present\n"
            "- Did more things\n"
            "\n"
            "## Education\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 2
        assert spans[0].company == "Acme Corp"
        assert spans[1].company == "Other Corp"

    def test_plain_paragraph_jobs_with_no_heading_or_bold_split_on_date_lines(self, locator):
        """Confirmed real pattern (Anandu Chandran, later jobs): plain
        sentences with no markup at all, each containing its own date
        range, are the locator's last-resort split signal."""
        text = (
            "## Experience:\n"
            "# Working as Manager at Acme Corp from January 2020 to March 2021.\n"
            "\n"
            "# Working as Director at Acme Corp from March 2021 till now.\n"
            "\n"
            "## Personal Profile\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 2
        assert "January 2020 to March 2021" in spans[0].date_range
        assert spans[1].date_range is not None

    def test_bullet_wrapped_bold_company_line_is_cleaned_of_the_bullet_marker(self, locator):
        text = (
            "## Experience\n"
            "### Manager\n"
            "- **Larsen & Turbo Micro Finance Ltd**\n"
            "  - Worked from January 2020 to March 2021\n"
            "\n"
            "## Education\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 1
        assert spans[0].company == "Larsen & Turbo Micro Finance Ltd"
        assert not spans[0].company.startswith("-")
        assert "**" not in spans[0].company

    def test_no_structural_markers_at_all_returns_one_whole_section_span(self, locator):
        """Graceful degradation: if neither headings, bold lines, nor
        date-range patterns are found, the whole section becomes one
        span with company/designation/date_range all None, rather than
        crashing or returning nothing."""
        text = "## Experience\nI have worked in various roles over the years.\n\n## Education\n"

        spans = locator.locate(text)

        assert len(spans) == 1
        assert spans[0].company is None
        assert spans[0].designation is None
        assert spans[0].date_range is None
        assert "various roles" in spans[0].markdown_text

    def test_month_day_year_date_order_is_parsed_correctly(self, locator):
        """Confirmed real bug found and fixed during development: 'from
        December 04 2017 to September 21 2020' must not be misread as
        ending in a 2-digit year ('September 21')."""
        text = (
            "## Experience\n"
            "### FLO\n"
            "**Example Corp**\n"
            "Worked from December 04 2017 to September 21 2020.\n"
            "\n"
            "## Education\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 1
        assert spans[0].date_range == "December 04 2017 to September 21 2020"

    def test_day_before_month_date_format(self, locator):
        text = (
            "## Experience\n"
            "### Manager\n"
            "**Example Corp**\n"
            "27th July 2015 to 11th August 2023\n"
            "\n"
            "## Education\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 1
        assert "27th July 2015" in spans[0].date_range
        assert "11th August 2023" in spans[0].date_range

    def test_slash_date_format(self, locator):
        text = "## Experience\n### Engineer\n**Acme Corp**\n09/2021 - 08/2023\n\n## Education\n"

        spans = locator.locate(text)

        assert spans[0].date_range == "09/2021 - 08/2023"

    def test_ongoing_word_variants_in_date_range(self, locator):
        for word in ["Present", "Current", "Till Date", "Ongoing", "Now"]:
            text = f"## Experience\n### Engineer\n**Acme Corp**\nJan 2020 - {word}\n\n## Education\n"
            spans = WorkExperienceLocator().locate(text)
            assert spans[0].date_range == f"Jan 2020 - {word}"


class TestRealDatasetSamples:
    """Regression tests against real, previously-confirmed-tricky
    markdown structure from this project's own dataset (paraphrased
    here as fixtures rather than reading the real files, so this test
    doesn't depend on benchmark/ground_truth/markdown/ being present or
    unchanged)."""

    def test_four_consistent_level_three_jobs(self, locator):
        text = (
            "# Nilesh Shivram Magare\n"
            "\n"
            "## Sr. Engineer\n"
            "\n"
            "## Summary\n"
            "Detail-oriented engineer.\n"
            "\n"
            "## Experience\n"
            "\n"
            "### Senior Engineer in Customer Support and Service\n"
            "**Ambetronics Engineers Pvt. Ltd**\n"
            "09/2021 - Present | Andheri\n"
            "- Managing a team\n"
            "\n"
            "### Maintenance & Service Engineer\n"
            "**Amit Engineering & Construction Works**\n"
            "10/2019 - 08/2021 | Sewree\n"
            "- Managed operations\n"
            "\n"
            "### Graduate Apprentice\n"
            "**Bharat Petroleum Corp. Ltd**\n"
            "09/2018 - 08/2019 | Sewree\n"
            "\n"
            "### Automation Engineer\n"
            "**Ecosys Efficiencies Pvt. Ltd**\n"
            "08/2016 - 08/2018 | Goregaon\n"
            "- PLC and HMI programming\n"
            "\n"
            "## Education\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 4
        companies = [span.company for span in spans]
        assert companies == [
            "Ambetronics Engineers Pvt. Ltd",
            "Amit Engineering & Construction Works",
            "Bharat Petroleum Corp. Ltd",
            "Ecosys Efficiencies Pvt. Ltd",
        ]


class TestCurrentEmploymentSectionHeading:
    """Milestone 23.20: '## Current Employment' — confirmed real,
    current heading on AMAN_CV_BCA.md. Previously invisible to the
    locator entirely (no _EXPERIENCE_SECTION_NAMES entry matched),
    so WEV-016/017/018 could never null this job's unsupported
    employment_type/location."""

    def test_current_employment_heading_is_recognized(self, locator):
        text = (
            "## Current Employment\n"
            "### Asha Tech (Asha Learningogy)\n"
            "Full Stack Developer - Full Time\n"
            "Feb 2025 - Present\n"
            "- Successfully completed a 6-month internship\n"
            "\n"
            "## Projects\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 1
        # "Asha Tech (Asha Learningogy)" contains no recognized
        # company-suffix keyword (see _COMPANY_SUFFIX_KEYWORDS), so —
        # exactly like every other job whose company name isn't
        # detectable this way — it's read as `designation`, not
        # `company`; company stays None rather than guessing (the
        # module's own "if detectable" contract, unrelated to and
        # unaffected by this milestone's change).
        assert spans[0].designation == "Asha Tech (Asha Learningogy)"
        assert spans[0].company is None

    def test_current_employment_section_still_ends_at_projects(self, locator):
        """Confirmed real pattern: AMAN_CV_BCA.md's 'Projects' section
        immediately follows 'Current Employment' — 'project' is already
        a known non-experience keyword, so this must still correctly
        exclude Projects-section content from the located span."""
        text = (
            "## Current Employment\n"
            "### Asha Tech\n"
            "Feb 2025 - Present\n"
            "- Did real work\n"
            "\n"
            "## Projects\n"
            "### SoWhat - Shopify E-commerce Platform\n"
            "- A side project, not a job\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 1
        assert "SoWhat" not in spans[0].markdown_text


class TestRepeatedSectionNameAsJobHeading:
    """Milestone 23.20: some resumes repeat the section's own name as
    each individual job's heading — 'Work Experience (A)', 'Work
    Experience (B)', ... one per job — instead of one section heading
    followed by consistently-leveled per-job sub-headings. Confirmed
    real, current pattern: AshishKumarSingh_RBM_Pune_Stonelam.md has
    five such headings at inconsistent levels (##, #, #, ###, ###)
    presenting each job as a one-row markdown table, which the
    pre-existing single-level _find_block_markers tier cannot
    reconcile at all (0 spans found before this milestone)."""

    _ASHISHKUMARSINGH_STYLE_TEXT = (
        "# KUMAR ASHISH\n"
        "\n"
        "## PROFILE SUMMARY\n"
        "Sales professional.\n"
        "\n"
        "## WORK EXPERIENCE (A)\n"
        "| Company Name | Designation | Area of Expertise | Total Experience |\n"
        "|---------------|-------------|------------------|------------------|\n"
        "| Knauf India Pvt. Ltd. | Area Sales Manager | Projects & Channel Sales | 2.8 Years |\n"
        "\n"
        "## Job Profile\n"
        "- Responsible for sales in Pune\n"
        "\n"
        "# WORK EXPERIENCE (B)\n"
        "\n"
        "| Company Name | Designation | Working Experience |\n"
        "|---------------|-------------|----------------------------------------------|\n"
        "| CLASSIC FLOORINGS AND INTERIORS PVT. LTD. | MANAGER - SALES | 27th July 2015 to 11th August 2023 |\n"
        "\n"
        "## Job Profile\n"
        "- Managed key accounts\n"
        "\n"
        "### Work Experience (C)\n"
        "| Company Name | Designation | Working Experience |\n"
        "|---------------|-------------|----------------------------------------------|\n"
        "| GLAZE ROCKS PVT. LTD. | BUSINESS DEVELOPMENT MANAGER | 07th November 2014 to 01st June 2015 |\n"
        "\n"
        "## Job Profile\n"
        "- Developed new business\n"
        "\n"
        "## Personal Information\n"
    )

    def test_repeated_heading_produces_one_span_per_job(self, locator):
        spans = locator.locate(self._ASHISHKUMARSINGH_STYLE_TEXT)

        assert len(spans) == 3

    def test_repeated_headings_at_inconsistent_levels_are_all_recognized(self, locator):
        """The three job headings above are level 2, 1, and 3
        respectively — a single-level filter would keep only one of
        them; this pattern must recognize all three regardless."""
        spans = locator.locate(self._ASHISHKUMARSINGH_STYLE_TEXT)

        assert len(spans) == 3
        assert "Knauf India Pvt. Ltd." in spans[0].markdown_text
        assert "CLASSIC FLOORINGS AND INTERIORS PVT. LTD." in spans[1].markdown_text
        assert "GLAZE ROCKS PVT. LTD." in spans[2].markdown_text

    def test_section_ends_at_first_non_experience_heading_after_the_last_repeated_marker(
        self, locator
    ):
        spans = locator.locate(self._ASHISHKUMARSINGH_STYLE_TEXT)

        assert len(spans) == 3
        assert "Personal Information" not in spans[-1].markdown_text

    def test_date_range_is_still_extracted_from_a_table_row(self, locator):
        """_extract_date_range searches the whole block's raw text, so
        a job's date range inside a table cell is still found even
        though structured company/designation extraction is not (see
        the table-row tests below)."""
        spans = locator.locate(self._ASHISHKUMARSINGH_STYLE_TEXT)

        assert spans[1].date_range == "27th July 2015 to 11th August 2023"
        assert spans[2].date_range == "07th November 2014 to 01st June 2015"

    def test_company_and_designation_are_none_for_a_table_formatted_job_not_guessed_wrong(
        self, locator
    ):
        """The marker heading itself ('WORK EXPERIENCE (A)') carries no
        job data, and the real data sits inside a markdown table this
        locator does not parse into cells — company/designation must be
        None, never a garbled guess (e.g. the table's own header row),
        per the module's 'if detectable' contract. The real company
        name is still recoverable by match_jobs_to_spans()'s
        substring-presence fallback, proven separately in
        TestRealDatasetSamples-style integration below."""
        spans = locator.locate(self._ASHISHKUMARSINGH_STYLE_TEXT)

        for span in spans:
            assert span.company is None
            assert span.designation is None

    def test_single_occurrence_of_the_section_name_is_not_this_pattern(self, locator):
        """Exactly one 'Work Experience (A)'-style heading (no second
        occurrence) must fall through to the ordinary single-section
        path completely unaffected by this milestone — this pattern
        requires 2+ repeats. The ordinary path's own exact-match
        _find_section_start was deliberately NOT changed to also strip
        a trailing parenthetical suffix (that stripping is scoped only
        to _find_repeated_section_name_markers), so a single such
        heading is recognized by neither path — the same (lack of)
        behavior as before this milestone, not a regression, since this
        exact shape was never supported and no real resume in this
        project's dataset has ever shown a single, non-repeated
        occurrence of this pattern (the one confirmed real resume,
        AshishKumarSingh, repeats it five times)."""
        text = (
            "## WORK EXPERIENCE (A)\n"
            "### Senior Engineer\n"
            "**Acme Corp**\n"
            "Jan 2020 - Present\n"
            "\n"
            "## Education\n"
        )

        spans = locator.locate(text)

        assert spans == []

    def test_real_jobs_still_match_their_table_formatted_spans_via_substring_presence(
        self, locator
    ):
        """End-to-end proof this pattern is actually useful: even with
        company/designation both None on the span, a real extracted job
        still matches confidently, because its own company/designation
        text appears verbatim inside the table row."""
        jobs = [
            _job(
                company="Knauf India Pvt. Ltd.",
                designation="Area Sales Manager",
                start_date=None,
                end_date=None,
            ),
            _job(
                company="CLASSIC FLOORINGS AND INTERIORS PVT. LTD.",
                designation="MANAGER - SALES",
                start_date="Jul 2015",
                end_date="Aug 2023",
            ),
            _job(
                company="GLAZE ROCKS PVT. LTD.",
                designation="BUSINESS DEVELOPMENT MANAGER",
                start_date="Nov 2014",
                end_date="Jun 2015",
            ),
        ]
        spans = locator.locate(self._ASHISHKUMARSINGH_STYLE_TEXT)

        matches = locator.match_jobs_to_spans(jobs, spans)

        assert set(matches) == {0, 1, 2}
        assert matches[0].confidence >= 0.5
        assert matches[1].confidence >= 0.5
        assert matches[2].confidence >= 0.5


class TestRepeatedSectionNameRequiresIdenticalStrippedText:
    """Milestone 23.28: narrows TestRepeatedSectionNameAsJobHeading's own
    mechanism. Milestone 23.27's dataset-wide census found the original
    trigger ("2+ headings anywhere in the document each independently
    belong to _EXPERIENCE_SECTION_NAMES") over-fires when a resume has
    two *different* legitimate section-name synonyms, not the same
    heading repeated — confirmed real, current cases:
    Rima_Karmakar_CV.md ("Professional Experience" then, after a page-
    break-derived restart, "Work Experience") and
    Ankush Mehta_Data Scientist_Intileo.md ("Experience" used twice
    across a page break). Both merged 3-4 real jobs into one oversized
    span per marker, so only the first job under each heading could
    ever be confidently matched. The fix requires the repeated
    headings' *stripped* text to be identical — AshishKumarSingh's own
    pattern (identical text after parenthetical-suffix stripping) is
    unaffected; see the class above, still passing unmodified."""

    def test_different_recognized_synonyms_are_not_treated_as_repeated_markers(self, locator):
        """The exact Rima_Karmakar_CV-shaped case: 'Professional
        Experience' (with 3 real sub-jobs) and, later, 'Work
        Experience' (with 3 more) — two different, both-legitimate
        _EXPERIENCE_SECTION_NAMES members, not one repeated marker."""
        text = (
            "# Rima Karmakar\n"
            "\n"
            "## Professional Experience\n"
            "\n"
            "### QualityKiosk Technologies Pvt Ltd – Assistant Test Manager (April 2025 – Present)\n"
            "- Lead QA engineers.\n"
            "\n"
            "### Blenheim Chalcot IT Services – Senior Test Engineer (September 2024 – April 2025)\n"
            "- Performed REST API testing.\n"
            "\n"
            "### Explo Solutions – QA Lead (December 2021 – September 2024)\n"
            "- Managed QA team activities.\n"
            "\n"
            "## Earlier Experience\n"
            "\n"
            "# Work Experience\n"
            "\n"
            "- **Crossasyst Technologies - QA Automation Engineer** (Dec 2020 – Dec 2021)\n"
            "- **IFL Finance Ltd - Senior Test Engineer** (Sept 2018 – Dec 2020)\n"
            "- **Media.net - QA Engineer** (Oct 2015 – Sept 2018)\n"
            "\n"
            "# Education & Certifications\n"
        )

        markers = locator._find_repeated_section_name_markers(text.splitlines())

        assert markers == []

    def test_professional_experience_and_work_experience_strip_to_different_text(self, locator):
        """Direct assertion of the specific stripped-text values
        Milestone 23.27 §2/§7 confirmed differ for Rima_Karmakar_CV."""
        assert (
            locator._strip_trailing_parenthetical_suffix("Professional Experience").lower()
            != locator._strip_trailing_parenthetical_suffix("Work Experience").lower()
        )
        assert locator._strip_trailing_parenthetical_suffix("Professional Experience").lower() == (
            "professional experience"
        )
        assert locator._strip_trailing_parenthetical_suffix("Work Experience").lower() == "work experience"

    def test_same_synonym_repeated_verbatim_across_a_page_break_still_qualifies(self, locator):
        """The exact Ankush Mehta_Data Scientist_Intileo-shaped case:
        'Experience' used twice (once per page-derived block), i.e. the
        SAME stripped text repeated — this still meets the (narrowed)
        trigger condition, since Milestone 23.27 confirmed this specific
        resume's problem is not fixed by requiring identical text (both
        headings already are identical) — this test documents that
        fact directly rather than asserting a fix Milestone 23.28 does
        not, and per its own investigation cannot safely, provide for
        this exact case. See
        docs/MILESTONE_23_28_WORK_EXPERIENCE_LOCATOR_PRECISION_FIXES_RESULTS.md
        for the full explanation."""
        text = (
            "# Ankush Mehta\n"
            "\n"
            "## Experience\n"
            "### AI Engineer, Isourse – Delhi, India\n"
            "**August 2025 - Present**\n"
            "- Built a RAG system.\n"
            "\n"
            "### AI Developer Intern, Buyogo – Noida, UP, India\n"
            "- Developed features.\n"
            "\n"
            "# Experience\n"
            "\n"
            "## Data Science Intern, Tata Power DDL – Delhi, India (July 2024 – August 2024)\n"
            "- Developed time series models.\n"
            "\n"
            "## Projects\n"
        )

        markers = locator._find_repeated_section_name_markers(text.splitlines())

        # Both headings strip to the identical "experience" -- this is
        # the SAME stripped text repeated, exactly like
        # AshishKumarSingh's own pattern, so it still qualifies under
        # the narrowed rule. This is a known, documented limitation of
        # Candidate A as specified, not a bug in this implementation.
        assert markers == [2, 10]

    def test_ashishkumarsingh_pattern_is_unaffected_by_the_narrower_rule(self, locator):
        """Regression control: the original Milestone 23.20 beneficiary
        (identical stripped text, parenthetical-suffix-stripped) must
        continue to trigger the repeated-marker path exactly as before."""
        spans = locator.locate(self._ASHISHKUMARSINGH_STYLE_TEXT_FOR_2328)

        assert len(spans) == 3

    _ASHISHKUMARSINGH_STYLE_TEXT_FOR_2328 = (
        TestRepeatedSectionNameAsJobHeading._ASHISHKUMARSINGH_STYLE_TEXT
    )

    def test_parenthetical_suffix_stripping_still_treats_repeats_as_identical(self, locator):
        """'WORK EXPERIENCE (A)' and 'WORK EXPERIENCE (B)' must still be
        recognized as the identical stripped text 'work experience' —
        the existing parenthetical-suffix stripping this rule depends
        on is preserved exactly, not reimplemented."""
        stripped_a = locator._strip_trailing_parenthetical_suffix("WORK EXPERIENCE (A)").lower()
        stripped_b = locator._strip_trailing_parenthetical_suffix("WORK EXPERIENCE (B)").lower()

        assert stripped_a == stripped_b == "work experience"

    def test_case_normalization_is_unchanged(self, locator):
        """Case-insensitivity (already-existing '.lower()' behavior)
        continues to apply consistently: 'WORK EXPERIENCE' and 'Work
        Experience' repeated together still qualify as identical."""
        text = (
            "## WORK EXPERIENCE\n"
            "### Senior Engineer\n"
            "**Acme Corp**\n"
            "Jan 2020 - Present\n"
            "\n"
            "# Work Experience\n"
            "### Junior Engineer\n"
            "**Beta Corp**\n"
            "Jan 2018 - Dec 2019\n"
            "\n"
            "## Education\n"
        )

        markers = locator._find_repeated_section_name_markers(text.splitlines())

        assert markers == [0, 5]

    def test_non_experience_heading_never_becomes_a_repeated_marker(self, locator):
        """A heading that isn't in _EXPERIENCE_SECTION_NAMES at all,
        repeated any number of times, must never trigger this path —
        unaffected by Milestone 23.28's change, verified directly."""
        text = (
            "## Projects\n"
            "### First Project\n"
            "\n"
            "## Projects\n"
            "### Second Project\n"
            "\n"
            "## Education\n"
        )

        markers = locator._find_repeated_section_name_markers(text.splitlines())

        assert markers == []


class TestEmploymentOutlineSectionHeading:
    """Milestone 23.28 (Candidate B): '## Employment Outline' —
    confirmed real, current heading on
    Nikhil Khanna_Data Visualization Analyst-Power BI Developer_Intileo.md.
    Previously invisible to the locator entirely (locate() returned
    zero spans for this document at all — both of its two real jobs
    were unreachable), the same category of gap Milestone 23.20's
    'current employment' addition already closed once, just a
    different literal string. See
    docs/MILESTONE_23_27_WORK_EXPERIENCE_LOCATOR_NO_SPAN_CENSUS.md
    §6/§7."""

    def test_employment_outline_heading_is_recognized(self, locator):
        text = (
            "# Nikhil Khanna\n"
            "\n"
            "## Employment Outline\n"
            "### Teleperformance Global (Data Analyst) | April 2025 - Present\n"
            "- Partnered with finance and operations.\n"
            "\n"
            "### CGM Aria, Noida (Data Analyst) | Aug 2022 - April 2025\n"
            "- Gathered business requirements.\n"
            "\n"
            "## Education\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 2
        assert "Teleperformance Global" in spans[0].markdown_text
        assert "CGM Aria" in spans[1].markdown_text

    def test_employment_outline_case_normalization_matches_existing_mechanism(self, locator):
        """Case-insensitivity works identically to every other
        _EXPERIENCE_SECTION_NAMES entry — no new normalization logic."""
        text = (
            "## EMPLOYMENT OUTLINE\n"
            "### Senior Engineer\n"
            "**Acme Corp**\n"
            "Jan 2020 - Present\n"
            "\n"
            "## Education\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 1

    def test_employment_outline_inside_body_text_is_not_a_section_heading(self, locator):
        """The phrase appearing in ordinary prose (not as a markdown
        heading line) must never be mistaken for a section boundary —
        recognition is heading-anchored (_HEADING_PATTERN-gated), same
        as every other entry in the set."""
        text = (
            "## Summary\n"
            "This resume includes a full employment outline below.\n"
            "\n"
            "## Education\n"
            "### B.Tech\n"
        )

        spans = locator.locate(text)

        assert spans == []

    def test_existing_recognized_headings_still_behave_identically(self, locator):
        """Negative control: adding one new frozenset member does not
        change recognition of any pre-existing synonym."""
        text = (
            "## Work Experience\n"
            "### Senior Engineer\n"
            "**Acme Corp**\n"
            "Jan 2020 - Present\n"
            "\n"
            "## Education\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 1


class TestIntermediateHeadingFallback:
    """Milestone 23.30 (Candidate C3, per
    docs/MILESTONE_23_29_INTERMEDIATE_HEADING_FALLBACK_SAFETY_INVESTIGATION.md's
    own validated rule): when the primary heading tier's single marker
    level yields exactly one marker, but real per-job headings sit one
    level deeper (an intermediate, non-job heading swallowing them —
    confirmed real on Dattatray S. Phad_Desktop Support Engineer_Intileo.md),
    recover the deeper headings as job markers — but ONLY when every one
    of them independently has its own date range in its own immediate
    block. Two real negative controls (CA Ishika Sharma Resume,
    Krishankant Soni_GAD-UK_GT-Intileo — department/subsection headings
    of one single job, no dates) must remain completely unaffected."""

    _DATTATRAY_STYLE_TEXT = (
        "# Dattatray S. Phad\n"
        "\n"
        "### Professional Experience\n"
        "\n"
        "**Lauren Information Technologies Pvt. Ltd**\n"
        "Mumbai, India | Jun 2025 – Present\n"
        "- IT support to VIP staff.\n"
        "\n"
        "# IT Support Experience\n"
        "\n"
        "## IT HUB Computers\n"
        "**Desktop Support Engineer**\n"
        "Mumbai, India | June 2023 – March 2025\n"
        "- Delivered desktop support.\n"
        "\n"
        "## Nex-Z Solutions (Freelancer)\n"
        "**IT Support Specialist**\n"
        "Mumbai, India | January 2021 – May 2023\n"
        "- Freelance support work.\n"
        "\n"
        "## Vsquare Networks\n"
        "**Network Support Technician**\n"
        "Mumbai, India | September 2019 – September 2021\n"
        "- Network support tasks.\n"
        "\n"
        "## Crystal Solutions Ltd\n"
        "**Hardware Assistant Trainee**\n"
        "Mumbai, India | July 2019 – September 2019\n"
        "- Hardware assistance.\n"
        "\n"
        "## EDUCATION\n"
    )

    _CA_ISHIKA_STYLE_TEXT = (
        "# CA Ishika Sharma\n"
        "\n"
        "## Work Experience\n"
        "\n"
        "### Articleship Experience - Gupta Shiv & Co.\n"
        "\n"
        "#### Audit & Assurance\n"
        "- Conducted statutory audits.\n"
        "\n"
        "#### Direct Taxation\n"
        "- Prepared tax returns.\n"
        "\n"
        "#### Indirect Taxation\n"
        "- GST compliance work.\n"
        "\n"
        "#### Accounting\n"
        "- Bookkeeping support.\n"
        "\n"
        "## Technical & Soft Skills\n"
    )

    _KRISHANKANT_STYLE_TEXT = (
        "# Krishankant Soni\n"
        "\n"
        "## Work Experience\n"
        "\n"
        "### Articleship Firm – PSD and Associates\n"
        "\n"
        "#### Domain & Key Client\n"
        "- Statutory Audit clients.\n"
        "\n"
        "#### Performance Areas\n"
        "- Internal Audit work.\n"
        "\n"
        "## Education\n"
    )

    def test_candidate_c_positive_case_dattatray_style_recovers_four_job_markers(self, locator):
        """(1) Positive case: the intermediate 'IT Support Experience'
        heading (level 1) must be bypassed in favor of its 4 real,
        date-confirmed level-2 job headings. 5 spans total: the leading
        Lauren block (pre-existing, unrelated '_build_spans' behavior —
        any non-blank content before the first marker is its own span,
        here absorbing the intermediate heading line harmlessly into
        its own tail) plus the 4 newly-recovered job spans."""
        spans = locator.locate(self._DATTATRAY_STYLE_TEXT)

        assert len(spans) == 5
        assert spans[0].company == "Lauren Information Technologies Pvt. Ltd"
        assert spans[1].designation == "IT HUB Computers"
        assert spans[2].company == "Nex-Z Solutions (Freelancer)"
        assert spans[3].designation == "Vsquare Networks"
        assert spans[4].company == "Crystal Solutions Ltd"

    def test_ca_ishika_negative_control_practice_area_headings_never_become_jobs(self, locator):
        """(2) CA Ishika Sharma negative control: 4 department/practice-
        area headings, none with a nearby date range — the fallback
        MUST NOT fire; the section must remain exactly one span."""
        spans = locator.locate(self._CA_ISHIKA_STYLE_TEXT)

        assert len(spans) == 1
        assert spans[0].company == "Articleship Experience - Gupta Shiv & Co."

    def test_krishankant_negative_control_subsection_headings_never_become_jobs(self, locator):
        """(3) Krishankant Soni negative control: 2 descriptive
        subsection headings of one job, no nearby dates — the fallback
        MUST NOT fire."""
        spans = locator.locate(self._KRISHANKANT_STYLE_TEXT)

        assert len(spans) == 1
        assert spans[0].company == "Articleship Firm – PSD and Associates"

    def test_fewer_than_two_lower_level_headings_does_not_fall_back(self, locator):
        """(4) A single deeper heading (ordinary sub-content, e.g. one
        "Responsibilities" label) must not trigger the fallback."""
        text = (
            "## Experience\n"
            "\n"
            "# Senior Role\n"
            "\n"
            "## Responsibilities\n"
            "- Did things.\n"
            "Jan 2020 - Dec 2021\n"
            "\n"
            "## Education\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 1

    def test_mixed_lower_level_heading_levels_does_not_fall_back(self, locator):
        """(5) Deeper headings spanning more than one distinct level
        must not trigger the fallback at all -- no silent filtering to
        a same-level subset (that would be a broader, unvalidated rule
        than the one Milestone 23.29 actually tested)."""
        text = (
            "## Experience\n"
            "\n"
            "# Senior Role\n"
            "\n"
            "## First Sub Corp\n"
            "Jan 2020 - Dec 2021\n"
            "\n"
            "### Second Sub Corp\n"
            "Jan 2018 - Dec 2019\n"
            "\n"
            "## Education\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 1

    def test_one_lower_level_heading_without_a_date_range_blocks_the_fallback(self, locator):
        """(6) If even one of two-or-more otherwise-qualifying deeper
        headings lacks its own date range, the fallback must decline
        entirely -- not partially fire for the others."""
        text = (
            "## Experience\n"
            "\n"
            "# Senior Role\n"
            "\n"
            "## Alpha Corp\n"
            "No date stated here at all.\n"
            "\n"
            "## Beta Corp\n"
            "Jan 2018 - Dec 2019\n"
            "\n"
            "## Education\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 1

    def test_multiple_lower_level_headings_where_one_lacks_a_date_range(self, locator):
        """(7) Same as (6) but with 3 candidates, only 2 of which have
        their own date -- still must not fall back."""
        text = (
            "## Experience\n"
            "\n"
            "# Senior Role\n"
            "\n"
            "## Alpha Corp\n"
            "Jan 2022 - Dec 2022\n"
            "\n"
            "## Beta Corp\n"
            "No date stated here at all.\n"
            "\n"
            "## Gamma Corp\n"
            "Jan 2018 - Dec 2019\n"
            "\n"
            "## Education\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 1

    def test_date_belonging_to_a_sibling_block_does_not_count(self, locator):
        """(8) A date range that exists in the document, but only in a
        *different* heading's own block, must not satisfy condition D
        for a heading that has none of its own -- the check is local
        to each candidate's own immediate block, never pooled across
        the whole section."""
        text = (
            "## Experience\n"
            "\n"
            "# Senior Role\n"
            "\n"
            "## Alpha Corp\n"
            "Jan 2022 - Dec 2022\n"
            "\n"
            "## Beta Corp\n"
            "No date in this block at all.\n"
            "\n"
            "## Education\n"
        )

        spans = locator.locate(text)

        # Alpha Corp has its own date; Beta Corp does not -- must not
        # fall back (Beta Corp's missing date is not rescued by
        # Alpha Corp's date belonging to a different block).
        assert len(spans) == 1

    def test_existing_normal_single_heading_experience_is_unaffected(self, locator):
        """(9) The ordinary, already-correct single-job-per-heading
        case must be completely unaffected by this milestone."""
        text = (
            "## Work Experience\n"
            "### Senior Engineer\n"
            "**Acme Corp**\n"
            "Jan 2020 - Present\n"
            "\n"
            "## Education\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 1

    def test_repeated_section_name_path_from_milestone_23_28_is_unaffected(self, locator):
        """(10) The Milestone 23.28 repeated-section-name mechanism
        returns before `_find_block_markers` is ever reached for a
        resume using that path -- AshishKumarSingh's own pattern must
        continue to produce 3 spans, unaffected by this milestone."""
        spans = locator.locate(TestRepeatedSectionNameAsJobHeading._ASHISHKUMARSINGH_STYLE_TEXT)

        assert len(spans) == 3

    def test_employment_outline_heading_from_milestone_23_28_is_unaffected(self, locator):
        """(11) The Milestone 23.28 'employment outline' heading
        addition is unaffected -- a normal single-level-per-job
        structure under that heading still produces one span per job,
        with no intermediate-heading fallback involved at all."""
        text = (
            "## Employment Outline\n"
            "### Senior Engineer\n"
            "**Acme Corp**\n"
            "Jan 2020 - Present\n"
            "\n"
            "## Education\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 1

    def test_candidate_blocks_never_extend_past_the_section_boundary(self, locator):
        """(12) No cross-section leakage: the last candidate's
        'immediate block' must stop at the section's own end boundary
        (the next non-experience heading), never absorbing content
        from a following section."""
        text = (
            "## Experience\n"
            "\n"
            "# Senior Role\n"
            "\n"
            "## Alpha Corp\n"
            "Jan 2022 - Dec 2022\n"
            "\n"
            "## Beta Corp\n"
            "Jan 2018 - Dec 2019\n"
            "\n"
            "## Education\n"
            "### B.Tech\n"
            "2014 - 2018\n"
        )

        spans = locator.locate(text)

        # 3 spans: the leading 'Senior Role' stub (pre-existing,
        # unrelated '_build_spans' leading-content behavior) plus the
        # 2 real, newly-recovered job spans. The actual property this
        # test verifies is the last real job span's own boundary.
        assert len(spans) == 3
        assert spans[-1].company == "Beta Corp"
        assert "B.Tech" not in spans[-1].markdown_text
        assert "2014 - 2018" not in spans[-1].markdown_text

    def test_all_five_dattatray_jobs_match_correctly_end_to_end(self, locator):
        """End-to-end proof, mirroring Milestone 23.29's own simulation:
        with real job JSON, all 5 jobs (the pre-existing correct match
        plus the 3 newly-recovered ones) match their own correct
        span identity, none cross-associated."""
        jobs = [
            _job(company="Lauren Information Technologies Pvt. Ltd.", designation="IT Desktop Support Engineer"),
            _job(company="IT HUB Computers", designation="IT Desktop Support Engineer"),
            _job(company="Nex-Z Solutions (Freelancer)", designation="IT Support Specialist"),
            _job(company="Vsquare Networks", designation="Network Support Technician"),
            _job(company="Crystal Solutions Ltd", designation="Hardware Assistant Trainee"),
        ]

        spans = locator.locate(self._DATTATRAY_STYLE_TEXT)
        matches = locator.match_jobs_to_spans(jobs, spans)

        assert set(matches) == {0, 1, 2, 3, 4}
        assert "Lauren" in matches[0].markdown_text
        assert matches[2].company == "Nex-Z Solutions (Freelancer)"
        assert matches[3].designation == "Vsquare Networks"
        assert matches[4].company == "Crystal Solutions Ltd"


class TestPipeDelimitedHeadingIsNotMistakenForATableRow:
    """Milestone 23.20 negative control: a heading using ' | ' as an
    inline separator between role/company/location must NOT be
    mistaken for a markdown table row just because it contains pipes —
    confirmed real, current pattern: Anushka_ (1).md uses
    '### Software Development Engineer | The HR Ally | Remote (April
    2025 - Present)'. Only a line whose first non-whitespace character
    is itself '|' (a real table row) is affected."""

    def test_pipe_delimited_heading_keeps_prior_extraction_behavior(self, locator):
        text = (
            "## Experience\n"
            "### Software Development Engineer | The HR Ally | Remote (April 2025 - Present)\n"
            "- Built things\n"
            "\n"
            "## Education\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 1
        # Not a table row (doesn't start with '|'), so the existing
        # best-effort heuristic still runs — company/designation are
        # not None, even though this heuristic can't cleanly separate
        # role/company/location from one pipe-joined heading line.
        assert spans[0].designation is not None

    def test_genuine_table_row_starting_with_pipe_is_still_recognized_as_a_table(
        self, locator
    ):
        text = (
            "## Experience\n"
            "### Engineer\n"
            "| Company | Role |\n"
            "|---|---|\n"
            "| Acme Corp | Senior Engineer |\n"
            "\n"
            "## Education\n"
        )

        spans = locator.locate(text)

        assert len(spans) == 1
        # The heading line itself ("### Engineer") is still used —
        # this control only proves a genuine table row occurring
        # *after* a normal heading doesn't crash or corrupt extraction;
        # the heading's own text was already a clean designation before
        # this milestone's change and remains so.
        assert spans[0].designation == "Engineer"


class TestMatchJobsToSpansConfidenceMatching:
    """Milestone 17.9: WorkExperienceLocator.match_jobs_to_spans().

    Every JobSpan here is constructed directly (via the `_span()`
    helper), not produced by locate(), so each test controls exactly
    which signals are and aren't available — the same approach
    tests/test_work_experience_validator.py already uses for its own
    job/span fixtures.
    """

    def test_identical_companies_match_with_high_confidence(self, locator):
        job = _job(company="Analytical Engines Ltd", designation="Mathematician")
        span = _span(
            company="Analytical Engines Ltd",
            designation="Mathematician",
            markdown_text="Analytical Engines Ltd\nMathematician",
        )

        matches = locator.match_jobs_to_spans([job], [span])

        assert 0 in matches
        assert matches[0].company == "Analytical Engines Ltd"
        assert matches[0].confidence >= 0.9

    def test_repeated_employer_disambiguated_by_date_overlap(self, locator):
        """Two jobs at the same company, with different, non-overlapping
        date ranges that each cleanly line up with one of two
        same-named spans — the date signal must correctly tell them
        apart even though the company signal ties."""
        job_first_stint = _job(
            company="Acme Corp", designation="Junior Engineer",
            start_date="Jan 2018", end_date="Dec 2019",
        )
        job_second_stint = _job(
            company="Acme Corp", designation="Senior Engineer",
            start_date="Jan 2022", end_date="Dec 2023",
        )
        span_first_stint = _span(
            company="Acme Corp", designation="Junior Engineer", date_range="Jan 2018 - Dec 2019",
            markdown_text="Acme Corp\nJunior Engineer\nJan 2018 - Dec 2019",
        )
        span_second_stint = _span(
            company="Acme Corp", designation="Senior Engineer", date_range="Jan 2022 - Dec 2023",
            markdown_text="Acme Corp\nSenior Engineer\nJan 2022 - Dec 2023",
        )

        matches = locator.match_jobs_to_spans(
            [job_first_stint, job_second_stint], [span_first_stint, span_second_stint]
        )

        assert matches[0].date_range == "Jan 2018 - Dec 2019"
        assert matches[1].date_range == "Jan 2022 - Dec 2023"

    def test_missing_company_still_matches_via_designation_and_date(self, locator):
        job = _job(
            company=None, designation="AI Engineer", start_date="Aug 2025", end_date=None,
        )
        span = _span(
            company=None,
            designation="AI Engineer, Isourse - Delhi, India",
            date_range="August 2025 - Present",
            markdown_text=(
                "AI Engineer, Isourse - Delhi, India\nAugust 2025 - Present\n"
                "- Built a RAG system."
            ),
        )

        matches = locator.match_jobs_to_spans([job], [span])

        assert 0 in matches
        assert matches[0].confidence > 0.5

    def test_designation_only_match_with_no_usable_dates(self, locator):
        """Neither side has a usable company or a parseable date — only
        designation similarity (plus the weak order signal) is
        available, and it alone is strong enough to confirm."""
        job = _job(company=None, designation="Senior Data Analyst", start_date=None, end_date=None)
        span = _span(
            company=None,
            designation="Senior Data Analyst",
            date_range=None,
            markdown_text="Senior Data Analyst\n- Built dashboards.",
        )

        matches = locator.match_jobs_to_spans([job], [span])

        assert 0 in matches
        assert matches[0].confidence > 0.5

    def test_date_only_match_with_no_usable_company_or_designation(self, locator):
        """Company and designation are both unusable (missing, or too
        dissimilar to help) — only the date range can identify the
        right span among several candidates."""
        job = _job(
            company=None, designation=None, start_date="Jan 2020", end_date="Dec 2020",
        )
        wrong_span = _span(
            company=None, designation=None, date_range="Jan 2015 - Dec 2016",
            markdown_text="Some other role\nJan 2015 - Dec 2016",
        )
        right_span = _span(
            company=None, designation=None, date_range="Jan 2020 - Dec 2020",
            markdown_text="Some role\nJan 2020 - Dec 2020",
        )

        matches = locator.match_jobs_to_spans([job], [wrong_span, right_span])

        assert 0 in matches
        assert matches[0].date_range == "Jan 2020 - Dec 2020"

    def test_reordered_jobs_are_matched_correctly_despite_order_signal_disagreeing(self, locator):
        """The JSON lists the newer role first, but the document lists
        the older role first — company, designation, and date overlap
        must all still correctly identify each job's true span despite
        the (here, actively wrong) occurrence-order signal."""
        newer_job = _job(
            company="Beta Inc", designation="Lead Engineer",
            start_date="Jan 2022", end_date="Dec 2023",
        )
        older_job = _job(
            company="Alpha LLC", designation="Junior Engineer",
            start_date="Jan 2018", end_date="Dec 2019",
        )
        # Document order is chronological (older role's span first),
        # the opposite of the JSON's [newer, older] order above.
        older_span = _span(
            company="Alpha LLC", designation="Junior Engineer", date_range="Jan 2018 - Dec 2019",
            markdown_text="Alpha LLC\nJunior Engineer\nJan 2018 - Dec 2019",
        )
        newer_span = _span(
            company="Beta Inc", designation="Lead Engineer", date_range="Jan 2022 - Dec 2023",
            markdown_text="Beta Inc\nLead Engineer\nJan 2022 - Dec 2023",
        )

        matches = locator.match_jobs_to_spans([newer_job, older_job], [older_span, newer_span])

        assert matches[0].company == "Beta Inc"
        assert matches[1].company == "Alpha LLC"

    def test_overlapping_concurrent_jobs_disambiguated_by_company_and_designation(self, locator):
        """Two genuinely concurrent jobs (fully overlapping date
        ranges — a legitimate pattern per docs/work_experience_design.md)
        must still each match their own, correct span via company and
        designation, even though the date signal alone can't tell them
        apart."""
        job_a = _job(
            company="Day Company", designation="Consultant",
            start_date="Jan 2021", end_date="Dec 2022",
        )
        job_b = _job(
            company="Night Company", designation="Freelancer",
            start_date="Jan 2021", end_date="Dec 2022",
        )
        span_a = _span(
            company="Day Company", designation="Consultant", date_range="Jan 2021 - Dec 2022",
            markdown_text="Day Company\nConsultant\nJan 2021 - Dec 2022",
        )
        span_b = _span(
            company="Night Company", designation="Freelancer", date_range="Jan 2021 - Dec 2022",
            markdown_text="Night Company\nFreelancer\nJan 2021 - Dec 2022",
        )

        matches = locator.match_jobs_to_spans([job_a, job_b], [span_a, span_b])

        assert matches[0].company == "Day Company"
        assert matches[1].company == "Night Company"

    def test_ambiguous_match_is_left_unmatched(self, locator):
        """Two candidate spans are equally (weakly) plausible for the
        same job — identical designation, no company or date signal to
        break the tie, and (by construction) an equal occurrence-order
        distance too — so the job must be left unmatched rather than
        guessed at. A second, unrelated "filler" job/span pair is
        included to confirm the ambiguity doesn't leak into an
        unrelated, perfectly clear match elsewhere in the same
        document."""
        filler_job = _job(company="Umbrella Corp", designation="Director")
        ambiguous_job = _job(company=None, designation="Analyst", start_date=None, end_date=None)
        candidate_a = _span(
            company=None, designation="Analyst", date_range=None,
            markdown_text="Analyst\n- Did analysis A.",
        )
        filler_span = _span(
            company="Umbrella Corp", designation="Director",
            markdown_text="Umbrella Corp\nDirector",
        )
        candidate_b = _span(
            company=None, designation="Analyst", date_range=None,
            markdown_text="Analyst\n- Did analysis B.",
        )

        # ambiguous_job (index 1) sits exactly one position away from
        # BOTH candidate_a (index 0) and candidate_b (index 2), so the
        # weak occurrence-order signal ties too, not just designation.
        matches = locator.match_jobs_to_spans(
            [filler_job, ambiguous_job], [candidate_a, filler_span, candidate_b]
        )

        assert matches[0].company == "Umbrella Corp"
        assert 1 not in matches

    def test_two_jobs_uniquely_preferring_the_same_span_leaves_the_loser_unmatched(self, locator):
        """A different flavor of ambiguity: two DIFFERENT jobs each
        uniquely prefer the same single span (e.g. a bad JSON company
        value that happens to closely resemble only one real span).
        The higher-scoring job gets it; the other is left unmatched
        rather than reassigned to some worse, unrelated span."""
        strong_job = _job(company="Acme Corporation", designation="Engineer")
        weak_job = _job(company="Acme Corp", designation="Manager")
        only_span = _span(
            company="Acme Corporation", designation="Engineer",
            markdown_text="Acme Corporation\nEngineer",
        )

        matches = locator.match_jobs_to_spans([strong_job, weak_job], [only_span])

        assert 0 in matches
        assert 1 not in matches

    def test_low_confidence_pair_is_left_unmatched(self, locator):
        """Completely unrelated company/designation text and no date
        information at all — the combined score must fall below the
        confidence threshold, leaving the job unmatched."""
        job = _job(company="Zephyr Dynamics", designation="Astrophysicist")
        span = _span(
            company="Totally Different Enterprises",
            designation="Warehouse Associate",
            markdown_text="Totally Different Enterprises\nWarehouse Associate",
        )

        matches = locator.match_jobs_to_spans([job], [span])

        assert matches == {}

    def test_confidence_field_is_a_real_score_in_zero_to_one_range(self, locator):
        job = _job(company="Acme Corp", designation="Engineer")
        span = _span(company="Acme Corp", designation="Engineer")

        matches = locator.match_jobs_to_spans([job], [span])

        assert 0.0 <= matches[0].confidence <= 1.0

    def test_located_spans_default_confidence_to_zero(self, locator):
        """A JobSpan straight from locate() — never matched against any
        job — always has confidence=0.0 (nothing to score yet)."""
        text = "## Experience\n### Acme Corp\nJan 2020 - Present\n- Did work.\n"

        spans = locator.locate(text)

        assert spans[0].confidence == 0.0

    def test_empty_jobs_or_empty_spans_returns_no_matches(self, locator):
        job = _job()
        span = _span()

        assert locator.match_jobs_to_spans([], [span]) == {}
        assert locator.match_jobs_to_spans([job], []) == {}
        assert locator.match_jobs_to_spans([], []) == {}

    def test_job_with_no_identifying_information_at_all_is_never_matched_by_order_alone(
        self, locator
    ):
        """A job with no company, no designation, and no parseable
        dates must never be matched purely because its position lines
        up with a span — occurrence order alone is not enough."""
        job = _job(company=None, designation=None, start_date=None, end_date=None)
        span = _span(
            company=None, designation=None, date_range=None, markdown_text="Something.",
        )

        matches = locator.match_jobs_to_spans([job], [span])

        assert matches == {}
