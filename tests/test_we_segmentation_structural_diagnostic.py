"""
tests/test_we_segmentation_structural_diagnostic.py

Milestone T8/Phase 2A -- Phase E adversarial structural analysis.

Tests the EXISTING, UNMODIFIED `WorkExperienceLocator`
(services/work_experience_locator.py) and `WorkExperienceValidator`
(services/work_experience_validator.py) against the 12 required
structural scenarios. This milestone is diagnostic only -- no
production code is changed. These tests document real, measured
behavior (including real, confirmed limitations) rather than assert a
desired behavior that doesn't yet exist.

Real-corpus cases use real, already-committed markdown/output data,
clearly labeled. All other cases are explicitly SYNTHETIC fixtures
built to isolate one structural scenario.
"""
from __future__ import annotations

import pytest

import json
from pathlib import Path

from services.work_experience_locator import WorkExperienceLocator
from services.work_experience_validator import WorkExperienceValidator


def _locator() -> WorkExperienceLocator:
    return WorkExperienceLocator()


@pytest.mark.local_data('corpus_markdown', 'module_outputs')
class TestScenario1_WorkExperienceFollowedByProjects:
    """#1 -- REAL (resume 001, AMAN_CV_BCA). Confirms, via direct
    execution of the current, unmodified locator, that the deterministic
    section boundary between a real `## Current Employment` and a
    following `## Projects` section is correctly identified -- 1 span,
    ending before the Projects heading."""

    def test_locator_finds_exactly_one_real_span_ending_before_projects(self):
        markdown = Path("output/markdown/AMAN_CV_BCA.md").read_text(encoding="utf-8")
        spans = _locator().locate(markdown)
        assert len(spans) == 1
        assert "Asha Tech" in spans[0].markdown_text
        assert "SoWhat" not in spans[0].markdown_text
        assert "Adiroha" not in spans[0].markdown_text


class TestScenario2_ProjectHeadingResemblesJobTitle:
    """#2 -- SYNTHETIC. A Project sub-heading phrased like a job title
    (a role name, no company) must not be picked up as a new Experience
    span once the Projects section boundary has been passed."""

    MARKDOWN = """# Resume

## Experience

### Real Company Inc
**Software Engineer**
*Jan 2020 - Dec 2022*

- Did real work.

## Projects

### Full-Stack Developer (Freelance Showcase)
*2021*
A personal portfolio project, not an employer.
"""

    def test_project_subheading_shaped_like_a_job_title_not_captured_as_a_span(self):
        spans = _locator().locate(self.MARKDOWN)
        for span in spans:
            assert "Full-Stack Developer (Freelance Showcase)" not in span.markdown_text


class TestScenario3_ProjectHeadingContainsTechnologyOrClientName:
    """#3 -- SYNTHETIC. A Project heading naming a real-sounding client
    or technology must not leak into the preceding job's span."""

    MARKDOWN = """# Resume

## Experience

### Acme Corp
**Analyst**
*2019 - 2021*

- Did real analyst work.

## Projects

### Acme Client Portal - React/Node Migration
*2022*
- Migrated a client portal.
"""

    def test_project_heading_with_client_name_not_folded_into_prior_job(self):
        spans = _locator().locate(self.MARKDOWN)
        assert len(spans) == 1
        assert "Migration" not in spans[0].markdown_text


@pytest.mark.local_data('corpus_markdown', 'module_outputs')
class TestScenario4_WorkExperienceWithZeroBullets:
    """#4 -- REAL (resume 001, job[0], "Asha Tech"). Confirms the
    located span for a real job with zero content bullets in the
    source is still found, and its own fabricated field values (pulled
    from elsewhere) are correctly stripped by the existing, unmodified
    WorkExperienceValidator (WEV-015/017/018), demonstrating the
    deterministic layer already reduces -- without fully solving -- the
    resume-001 defect."""

    def test_zero_bullet_job_span_found_and_hallucinated_fields_stripped(self):
        markdown = Path("output/markdown/AMAN_CV_BCA.md").read_text(encoding="utf-8")
        extracted = json.loads(
            Path("output/modules/AMAN_CV_BCA/work_experience.json").read_text(encoding="utf-8")
        )
        jobs = extracted["work_experience"]
        validator = WorkExperienceValidator()
        validated, report = validator.validate(markdown, jobs)

        # job[0] ("Asha Tech...") is matched to the one real span.
        rule_ids_for_job0 = {i.rule_id for i in report.issues if i.job_index == 0}
        assert "WEV-017" in rule_ids_for_job0  # location stripped -- not in the real span
        assert "WEV-018" in rule_ids_for_job0  # duration stripped -- not in the real span
        assert validated[0]["location"] is None
        assert validated[0]["duration"] is None


@pytest.mark.local_data('corpus_markdown', 'module_outputs')
class TestScenario5_WorkExperienceWithManyBullets:
    """#5 -- REAL (resume 013, RoboAIAPaths -- 6 real bullets). A job
    with many real bullets is located as a single span containing all
    of them, undivided."""

    def test_many_bullet_job_located_as_one_span(self):
        markdown = Path(
            "output/markdown/Kishan Yadav_Data Scientist_Intileo.md"
        ).read_text(encoding="utf-8")
        spans = _locator().locate(markdown)
        matched = [s for s in spans if "RoboAIAPaths" in s.markdown_text or (
            s.designation and "AI Engineer" in (s.designation or "")
        )]
        assert len(matched) >= 1
        # the real source has 6 real bullets under this job
        bullet_count = matched[0].markdown_text.count("\n- ")
        assert bullet_count >= 5


@pytest.mark.local_data('corpus_markdown', 'module_outputs')
class TestScenario6_ProjectsContainingDates:
    """#6 -- REAL (resume 001's Projects, each with an explicit date
    range). Confirms dated Project sub-headings are not mistaken for
    a second Experience entity once the section boundary is passed."""

    def test_dated_project_subheadings_do_not_extend_experience_span(self):
        markdown = Path("output/markdown/AMAN_CV_BCA.md").read_text(encoding="utf-8")
        spans = _locator().locate(markdown)
        assert len(spans) == 1
        assert "Aug 2023 - Oct 2025" not in spans[0].markdown_text  # SoWhat's own date range


class TestScenario7_ProjectsWithoutDates:
    """#7 -- SYNTHETIC. An undated Project sub-heading is equally not
    mistaken for an Experience entity -- date presence/absence is not
    what the section boundary check depends on."""

    MARKDOWN = """# Resume

## Experience

### Real Company
**Engineer**
*2018 - 2020*

- Real work.

## Projects

### Undated Side Project
No date given for this one at all.
- Built something.
"""

    def test_undated_project_subheading_does_not_extend_span(self):
        spans = _locator().locate(self.MARKDOWN)
        assert len(spans) == 1
        assert "Undated Side Project" not in spans[0].markdown_text


@pytest.mark.local_data('corpus_markdown', 'module_outputs')
class TestScenario8_LegitimateWorkExperienceMentioningProjects:
    """#8 -- REAL (resume 013, "Global Source Technology" job, whose
    own real bullet legitimately mentions two Project names by name:
    "applying concepts practiced in academic projects like EcoVision
    and MockMate"). Confirms this legitimate in-bullet mention does not
    cause any span/boundary corruption -- the mention is real, internal
    job content, not a section-boundary signal."""

    def test_legitimate_in_bullet_project_mention_does_not_break_span(self):
        markdown = Path(
            "output/markdown/Kishan Yadav_Data Scientist_Intileo.md"
        ).read_text(encoding="utf-8")
        spans = _locator().locate(markdown)
        matching = [s for s in spans if "Global Source Technology" in s.markdown_text]
        assert len(matching) == 1
        assert "EcoVision and MockMate" in matching[0].markdown_text


@pytest.mark.local_data('corpus_markdown', 'module_outputs')
class TestScenario9_IdenticalOrSimilarTextInWorkExperienceAndProjects:
    """#9 -- REAL (resume 013, RoboAIAPaths job vs. ASB project -- see
    docs/MILESTONE_T7_PHASE1D... and this milestone's Phase B/C for the
    full evidence: this is genuine, intentional, source-level
    duplication, not a parsing defect). Confirms the located Experience
    span for RoboAIAPaths is entirely separate from (does not include)
    the later, distinct `# Projects` section content."""

    def test_intentional_duplicate_content_still_yields_separate_correct_spans(self):
        markdown = Path(
            "output/markdown/Kishan Yadav_Data Scientist_Intileo.md"
        ).read_text(encoding="utf-8")
        spans = _locator().locate(markdown)
        matching = [s for s in spans if "RoboAIAPaths" in s.markdown_text]
        assert len(matching) == 1
        # The Experience span must not reach into the later Projects
        # section's own (textually similar) condensed highlight bullets.
        assert "ASB – Full-Stack AI Numerology Platform" not in matching[0].markdown_text


@pytest.mark.local_data('corpus_markdown', 'module_outputs')
class TestScenario10_MultipleWorkExperienceEntriesFollowedByProjects:
    """#10 -- REAL (resume 006, AshishKumarSingh -- 5 real jobs, no
    Projects section at all, included here as the "many entries"
    baseline) plus a SYNTHETIC multi-job-then-Projects fixture to
    directly test the required scenario shape."""

    def test_real_multi_job_resume_all_five_spans_found(self):
        markdown = Path(
            "output/markdown/AshishKumarSingh_RBM_Pune_Stonelam.md"
        ).read_text(encoding="utf-8")
        spans = _locator().locate(markdown)
        assert len(spans) == 5

    MARKDOWN = """# Resume

## Experience

### Company A
**Role A**
*2015 - 2017*
- Work A.

### Company B
**Role B**
*2017 - 2019*
- Work B.

### Company C
**Role C**
*2019 - 2021*
- Work C.

## Projects

### Side Project X
- Built X.
"""

    def test_synthetic_three_jobs_then_projects_yields_three_spans_only(self):
        spans = _locator().locate(self.MARKDOWN)
        assert len(spans) == 3
        for span in spans:
            assert "Side Project X" not in span.markdown_text


class TestScenario11_ProjectsAppearingBeforeWorkExperience:
    """#11 -- SYNTHETIC. When Projects precedes Experience in reading
    order, the located Experience span must start at the real Experience
    heading, not be confused by the earlier Projects content."""

    MARKDOWN = """# Resume

## Projects

### Early Side Project
- Built something before the job history section.

## Experience

### Real Employer
**Engineer**
*2020 - 2022*
- Real employment content.
"""

    def test_projects_before_experience_does_not_corrupt_the_located_span(self):
        spans = _locator().locate(self.MARKDOWN)
        assert len(spans) == 1
        assert "Early Side Project" not in spans[0].markdown_text
        assert "Real Employer" in spans[0].markdown_text


@pytest.mark.local_data('corpus_markdown', 'module_outputs')
class TestScenario12_NestedProjectHeadingsSubHeadings:
    """#12 -- REAL (resume 001's actual structure: each Project entry
    is nested two heading levels deep, `### Full-Stack` / `#### SoWhat -
    Shopify E-commerce Platform`). Confirms the located Experience span
    ends at the FIRST following heading of any level under `##
    Projects`, not just a same-level heading -- i.e. nested sub-headings
    are correctly treated as still being inside/after the Projects
    section, not accidentally re-extending the Experience span."""

    def test_nested_project_subheadings_all_excluded_from_experience_span(self):
        markdown = Path("output/markdown/AMAN_CV_BCA.md").read_text(encoding="utf-8")
        spans = _locator().locate(markdown)
        assert len(spans) == 1
        # "Full-Stack" is deliberately excluded from this check: it is a
        # real coincidental substring of the genuine job's own prose
        # ("...transitioned to full-time Full-Stack Developer.") -- not
        # leakage from the nested "### Full-Stack" Project sub-heading.
        # Confirmed by direct inspection before asserting.
        for nested_heading_text in ("SoWhat", "Asha Learningy", "Adiroha"):
            assert nested_heading_text not in spans[0].markdown_text


@pytest.mark.local_data('corpus_markdown', 'module_outputs')
class TestFalsePositiveRiskOfUnmatchedSpanAsFabricationSignal:
    """Decisive counter-evidence for Phase D/F: "a job that cannot be
    confidently matched to any located span" is NOT a safe, general
    signal for "this job is a fabricated entity" -- it also fires on a
    REAL, GT-confirmed genuine job (resume 006/AshishKumarSingh's first
    job, "Knauf India Pvt. Ltd."), for an unrelated, already-disclosed
    locator limitation (its own "repeated-marker structure", named in
    docs/MILESTONE_V2_4E_WE_SECTION_BOUNDARY_PRECISION.md as an
    unaddressed pre-existing limitation), not because that job is
    illegitimate."""

    def test_real_genuine_first_job_on_006_has_no_confident_span_match(self):
        markdown = Path(
            "output/markdown/AshishKumarSingh_RBM_Pune_Stonelam.md"
        ).read_text(encoding="utf-8")
        extracted = json.loads(
            Path("output/modules/AshishKumarSingh_RBM_Pune_Stonelam/work_experience.json")
            .read_text(encoding="utf-8")
        )
        jobs = extracted["work_experience"]
        locator = _locator()
        spans = locator.locate(markdown)
        span_by_index = locator.match_jobs_to_spans(jobs, spans)

        assert jobs[0]["company"] == "Knauf India Pvt. Ltd."
        assert span_by_index.get(0) is None  # real job, unmatched -- a real false-positive risk
