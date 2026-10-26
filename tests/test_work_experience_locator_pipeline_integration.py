"""
tests/test_work_experience_locator_pipeline_integration.py

Integration tests for Milestone 17.7 and Milestone 17.8, proving

    markdown -> WorkExperienceLocator -> WorkExperienceValidator -> ParserPipeline

works end to end, using the same source markdown across every step.

Milestone 17.7 only established the API — WorkExperienceLocator's
output wasn't yet consumed by WorkExperienceValidator's rule logic, so
"works end to end" there meant only that every component accepted the
same real markdown consistently. Milestone 17.8 changed that: WEV-015
now genuinely uses WorkExperienceLocator internally (via
WorkExperienceValidator's own `locator` attribute) to remove leaked
technologies, matching jobs to spans by normalized company name only.
TestFullChainRemovesLeakedTechnologies below is the 17.8-specific
proof that the full chain — real markdown, read once by
ParserPipeline, handed to the real WorkExperienceValidator, which
internally calls the real WorkExperienceLocator — actually removes a
technology that only appears in the resume's Skills section (never in
the job's own text) by the time it reaches the pipeline's final merged
output.

Milestone 17.9 built WorkExperienceLocator.match_jobs_to_spans() — a
confidence-based matcher combining company similarity, designation
similarity, date overlap, and occurrence order — as standalone
infrastructure, not yet consumed anywhere. Milestone 17.10 wires it
into WEV-015 in place of the old normalized-company-name-only lookup.
TestFullChainImprovedMatchingViaConfidenceScoring below is the
17.10-specific proof that this measurably improves real coverage
through the *full pipeline* (not just the standalone validator): a job
whose heading combines designation, company, and location on one line
(a real pattern found in Milestone 17.8's own investigation) is one
the locator never extracts a `company` for at all — the old matcher
could therefore never match it — but the new confidence-based matcher
still confidently matches it via designation and date overlap, and
WEV-015's unchanged technology-provenance check then runs correctly
against it, with the winning match's confidence score propagated all
the way to the final ValidationReport.

The five non-work_experience module extractors are mocked (as in
tests/test_parser_pipeline.py) so this stays a fast unit-level
integration test; WorkExperienceExtractor is also mocked
(returning data matching the fixture markdown) since exercising the
real model is what test_parser_pipeline.py's own
test_real_pipeline_end_to_end_integration (marked @pytest.mark.integration)
already covers, and duplicating a real-Ollama call here would just slow
this test down for no new coverage. WorkExperienceLocator and
WorkExperienceValidator themselves run for real (they're pure,
deterministic, in-memory logic — no reason to mock either).
"""

import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from config import OUTPUT_DIR
from pipeline.parser_pipeline import ParserPipeline
from services.contact_information_extractor import ContactInformationExtractor
from services.education_extractor import EducationExtractor
from services.languages_extractor import LanguagesExtractor
from services.personal_information_extractor import PersonalInformationExtractor
from services.professional_summary_extractor import ProfessionalSummaryExtractor
from services.project_extractor import ProjectExtractor
from services.skills_extractor import SkillsExtractor
from services.work_experience_extractor import WorkExperienceExtractor
from services.work_experience_locator import WorkExperienceLocator
from services.work_experience_validator import WorkExperienceValidator

FIXTURE_MARKDOWN = (
    "# Ada Lovelace\n"
    "\n"
    "## Experience\n"
    "### Mathematician\n"
    "**Analytical Engines Ltd**\n"
    "Full-time | Jan 2020 - Present\n"
    "- Wrote the first published algorithm\n"
    "\n"
    "### Junior Analyst\n"
    "**Babbage & Co**\n"
    "Full-time | Jan 2018 - Dec 2019\n"
    "- Assisted with early calculations\n"
    "\n"
    "## Education\n"
    "### Mathematics\n"
)
# Milestone 23.15: "Full-time" is now stated directly in each job's own
# span above (not merely implied), so both DIRTY_MATHEMATICIAN_JOB's and
# CLEAN_ANALYST_JOB's employment_type is genuinely source-supported and
# WEV-016 leaves it untouched — keeping these tests' original,
# WEV-001-only intent intact rather than picking up an unrelated new
# WEV-016 side effect from an employment_type claim the fixture never
# actually stated.

# This job's end_date deliberately violates WEV-001 (a literal "Present"
# string) so the test can confirm the validator genuinely still applies
# Phase 1 corrections when called as part of the full pipeline, using
# markdown supplied from the same fixture the locator also parses.
DIRTY_MATHEMATICIAN_JOB = {
    "company": "Analytical Engines Ltd",
    "designation": "Mathematician",
    "employment_type": "Full-time",
    "location": None,
    "start_date": "Jan 2020",
    "end_date": "Present",
    "currently_working": False,
    "duration": None,
    "responsibilities": ["Wrote the first published algorithm"],
    "technologies": [],
}
CLEAN_ANALYST_JOB = {
    "company": "Babbage & Co",
    "designation": "Junior Analyst",
    "employment_type": "Full-time",
    "location": None,
    "start_date": "Jan 2018",
    "end_date": "Dec 2019",
    "currently_working": False,
    "duration": None,
    "responsibilities": ["Assisted with early calculations"],
    "technologies": [],
}

# The non-work_experience modules only need to be schema-valid, not
# realistic — resume_schema_v1.json requires personal_information,
# contact_information, professional_summary, skills, languages,
# work_experience, and education as its seven top-level keys.
PERSONAL_INFO_JSON = {
    "personal_information": {
        "full_name": "Ada Lovelace",
        "first_name": "Ada",
        "last_name": "Lovelace",
        "current_location": None,
    }
}
CONTACT_INFO_JSON = {
    "contact_information": {
        "email": ["ada@example.com"],
        "phone": [],
        "linkedin": None,
        "github": None,
        "portfolio": None,
    }
}
SKILLS_JSON = {"skills": ["Mathematics"]}
SUMMARY_JSON = {"professional_summary": "Mathematician and writer."}
LANGUAGES_JSON = {"languages": ["English"]}
EDUCATION_JSON = {"education": []}


def _module_result(json_object):
    return SimpleNamespace(json_object=json_object, processing_time=0.01)


def _mock_project_extractor():
    """Mock for the ProjectExtractor that Milestone T5 registered as the
    pipeline's eighth, default-constructed extractor. Without this, every
    ParserPipeline built below with only the seven legacy extractors
    mocked would construct a real ProjectExtractor and make a live
    Ollama call inside the deterministic suite (fails with
    ModelNotFoundError when the model is absent, silently spends an LLM
    call when it is present). Projects are not under test here."""
    extractor = MagicMock(spec=ProjectExtractor)
    extractor.extract.return_value = _module_result({"projects": []})
    return extractor


class TestLocatorStandaloneOnTheFixtureMarkdown:
    """Confirms the locator, run directly against the same markdown the
    pipeline test below uses, finds both jobs correctly — establishing
    that the fixture is realistic and the locator handles it, even
    though this milestone doesn't yet connect its output to anything."""

    def test_locator_finds_both_jobs(self):
        spans = WorkExperienceLocator().locate(FIXTURE_MARKDOWN)

        assert len(spans) == 2
        assert spans[0].company == "Analytical Engines Ltd"
        assert spans[0].designation == "Mathematician"
        assert spans[0].date_range == "Jan 2020 - Present"
        assert spans[1].company == "Babbage & Co"
        assert spans[1].designation == "Junior Analyst"


class TestValidatorStandaloneWithTheSameMarkdown:
    """Confirms the validator accepts the same real markdown text (not
    just an empty placeholder) alongside the JSON, and still applies
    Phase 1 corrections — the markdown parameter is inert but must not
    interfere."""

    def test_validator_corrects_dirty_job_given_the_real_markdown(self):
        validator = WorkExperienceValidator()

        validated, report = validator.validate(
            FIXTURE_MARKDOWN, [DIRTY_MATHEMATICIAN_JOB, CLEAN_ANALYST_JOB]
        )

        assert validated[0]["end_date"] is None
        assert validated[0]["currently_working"] is True
        assert validated[1] == CLEAN_ANALYST_JOB
        assert len(report.issues) == 1
        assert report.issues[0].rule_id == "WEV-001"


@pytest.fixture
def output_cleanup():
    resume_names = []
    yield resume_names
    for resume_name in resume_names:
        shutil.rmtree(OUTPUT_DIR / "modules" / resume_name, ignore_errors=True)
        (OUTPUT_DIR / "json_final" / f"{resume_name}.json").unlink(missing_ok=True)


class TestFullChainMarkdownLocatorValidatorParser:
    """The end-to-end proof: the same markdown text flows through the
    locator (standalone), the validator (via the real, wired
    ParserPipeline), and the parser's final output — all consistent
    with one another."""

    def test_markdown_flows_through_locator_and_through_the_real_pipeline_consistently(
        self, tmp_path, output_cleanup
    ):
        resume_name = "test_resume_locator_pipeline_integration"
        output_cleanup.append(resume_name)
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text(FIXTURE_MARKDOWN, encoding="utf-8")

        # Step 1: the locator, run directly against the markdown file's
        # text (independent of the pipeline).
        spans = WorkExperienceLocator().locate(markdown_path.read_text(encoding="utf-8"))
        assert len(spans) == 2

        # Step 2: the same markdown, through the real, wired
        # ParserPipeline — the five non-work_experience module
        # extractors are mocked (fast, no Ollama); WorkExperienceExtractor
        # is mocked to return the dirty fixture data; WorkExperienceValidator
        # runs for real and must receive the pipeline's own markdown read.
        mock_personal_information_extractor = MagicMock(spec=PersonalInformationExtractor)
        mock_personal_information_extractor.extract.return_value = _module_result(
            PERSONAL_INFO_JSON
        )
        mock_contact_information_extractor = MagicMock(spec=ContactInformationExtractor)
        mock_contact_information_extractor.extract.return_value = _module_result(
            CONTACT_INFO_JSON
        )
        mock_skills_extractor = MagicMock(spec=SkillsExtractor)
        mock_skills_extractor.extract.return_value = _module_result(SKILLS_JSON)
        mock_professional_summary_extractor = MagicMock(spec=ProfessionalSummaryExtractor)
        mock_professional_summary_extractor.extract.return_value = _module_result(SUMMARY_JSON)
        mock_languages_extractor = MagicMock(spec=LanguagesExtractor)
        mock_languages_extractor.extract.return_value = _module_result(LANGUAGES_JSON)
        mock_education_extractor = MagicMock(spec=EducationExtractor)
        mock_education_extractor.extract.return_value = _module_result(EDUCATION_JSON)
        mock_work_experience_extractor = MagicMock(spec=WorkExperienceExtractor)
        mock_work_experience_extractor.extract.return_value = _module_result(
            {"work_experience": [DIRTY_MATHEMATICIAN_JOB, CLEAN_ANALYST_JOB]}
        )

        pipeline = ParserPipeline(
            personal_information_extractor=mock_personal_information_extractor,
            contact_information_extractor=mock_contact_information_extractor,
            skills_extractor=mock_skills_extractor,
            professional_summary_extractor=mock_professional_summary_extractor,
            languages_extractor=mock_languages_extractor,
            work_experience_extractor=mock_work_experience_extractor,
            education_extractor=mock_education_extractor,
            project_extractor=_mock_project_extractor(),
            # work_experience_validator: real (default) — this is the
            # piece under test.
        )

        result = pipeline.run(markdown_path)

        # Step 3: the parser's final output reflects the validator's
        # Phase 1 correction (WEV-001) — the same correction the
        # standalone validator test above confirmed on the same data.
        corrected_jobs = result.merged_json["work_experience"]
        assert corrected_jobs[0]["end_date"] is None
        assert corrected_jobs[0]["currently_working"] is True
        assert corrected_jobs[1] == CLEAN_ANALYST_JOB

        # And the validation report on the result matches what running
        # the validator standalone (Step 2's sibling test) found.
        assert result.work_experience_validation_report is not None
        assert len(result.work_experience_validation_report.issues) == 1
        assert result.work_experience_validation_report.issues[0].rule_id == "WEV-001"

        # Consistency check: the number of jobs the locator found in the
        # raw markdown (2) matches the number of work_experience entries
        # the full pipeline produced (2) — the same source document,
        # examined two different ways, agrees on job count for this
        # well-structured fixture.
        assert len(spans) == len(corrected_jobs)


# Milestone 17.8: a markdown fixture whose Experience section only ever
# mentions "Python" for the Mathematician job, plus a Skills section
# (outside the Experience section entirely) that separately lists
# "Kubernetes" — a real, common leakage pattern (docs/WORK_EXPERIENCE_
# AUDIT.md's WE-01 finding) where the whole global skills list gets
# copied onto one job's technologies.
LEAKAGE_FIXTURE_MARKDOWN = (
    "# Ada Lovelace\n"
    "\n"
    "## Experience\n"
    "### Mathematician\n"
    "**Analytical Engines Ltd**\n"
    "Jan 2020 - Present\n"
    "- Wrote the first published algorithm using Python\n"
    "\n"
    "## Skills\n"
    "Python, Kubernetes, Docker\n"
)
DIRTY_MATHEMATICIAN_JOB_WITH_LEAKED_TECH = {
    **DIRTY_MATHEMATICIAN_JOB,
    "end_date": None,
    "currently_working": True,
    # "Python" is genuinely in this job's own span; "Kubernetes" only
    # appears in the Skills section — the whole-skills-list-copied-onto-
    # a-job pattern WEV-015 exists to catch.
    "technologies": ["Python", "Kubernetes"],
}


class TestFullChainRemovesLeakedTechnologies:
    """Milestone 17.8's own explicit deliverable: an integration test
    proving markdown -> locator -> validator -> parser correctly
    removes a leaked technology, through the real, wired pipeline (not
    just the standalone validator — see
    TestWEV015TechnologyProvenance in tests/test_work_experience_validator.py
    for the standalone-validator version of this same proof)."""

    def test_technology_only_present_in_skills_section_is_removed_by_the_full_pipeline(
        self, tmp_path, output_cleanup
    ):
        resume_name = "test_resume_wev015_leakage_integration"
        output_cleanup.append(resume_name)
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text(LEAKAGE_FIXTURE_MARKDOWN, encoding="utf-8")

        # Step 1: the locator, run standalone, confirms the Experience
        # section's own span text does NOT contain "Kubernetes" — it's
        # only present in the (separate, out-of-span) Skills section.
        spans = WorkExperienceLocator().locate(
            markdown_path.read_text(encoding="utf-8")
        )
        assert len(spans) == 1
        assert "kubernetes" not in spans[0].markdown_text.lower()
        assert "python" in spans[0].markdown_text.lower()

        # Step 2: the real, wired ParserPipeline — same mocking pattern
        # as the sibling 17.7 test above, except WorkExperienceExtractor
        # now returns a job whose technologies include the leaked item.
        mock_personal_information_extractor = MagicMock(spec=PersonalInformationExtractor)
        mock_personal_information_extractor.extract.return_value = _module_result(
            PERSONAL_INFO_JSON
        )
        mock_contact_information_extractor = MagicMock(spec=ContactInformationExtractor)
        mock_contact_information_extractor.extract.return_value = _module_result(
            CONTACT_INFO_JSON
        )
        mock_skills_extractor = MagicMock(spec=SkillsExtractor)
        mock_skills_extractor.extract.return_value = _module_result(
            {"skills": ["Python", "Kubernetes", "Docker"]}
        )
        mock_professional_summary_extractor = MagicMock(spec=ProfessionalSummaryExtractor)
        mock_professional_summary_extractor.extract.return_value = _module_result(SUMMARY_JSON)
        mock_languages_extractor = MagicMock(spec=LanguagesExtractor)
        mock_languages_extractor.extract.return_value = _module_result(LANGUAGES_JSON)
        mock_education_extractor = MagicMock(spec=EducationExtractor)
        mock_education_extractor.extract.return_value = _module_result(EDUCATION_JSON)
        mock_work_experience_extractor = MagicMock(spec=WorkExperienceExtractor)
        mock_work_experience_extractor.extract.return_value = _module_result(
            {"work_experience": [DIRTY_MATHEMATICIAN_JOB_WITH_LEAKED_TECH]}
        )

        pipeline = ParserPipeline(
            personal_information_extractor=mock_personal_information_extractor,
            contact_information_extractor=mock_contact_information_extractor,
            skills_extractor=mock_skills_extractor,
            professional_summary_extractor=mock_professional_summary_extractor,
            languages_extractor=mock_languages_extractor,
            work_experience_extractor=mock_work_experience_extractor,
            education_extractor=mock_education_extractor,
            project_extractor=_mock_project_extractor(),
            # work_experience_validator: real (default) — internally
            # uses a real WorkExperienceLocator too.
        )

        result = pipeline.run(markdown_path)

        # Step 3: "Kubernetes" was removed (leaked from Skills), "Python"
        # was kept (genuinely present in the job's own span) — by the
        # time it reaches the pipeline's final merged output.
        corrected_job = result.merged_json["work_experience"][0]
        assert corrected_job["technologies"] == ["Python"]

        report = result.work_experience_validation_report
        assert report is not None
        wev_015_issues = [issue for issue in report.issues if issue.rule_id == "WEV-015"]
        assert len(wev_015_issues) == 1
        assert wev_015_issues[0].auto_fixed is True
        assert "Kubernetes" in wev_015_issues[0].message
        # Milestone 17.10: the confidence-based matcher's winning score
        # is propagated all the way through to the final report.
        assert wev_015_issues[0].confidence is not None
        assert 0.0 <= wev_015_issues[0].confidence <= 1.0

        # The Skills module's own output is untouched — WEV-015 only
        # ever edits work_experience, never the resume-wide skills list.
        assert result.merged_json["skills"] == ["Python", "Kubernetes", "Docker"]


# Milestone 17.10: a real pattern Milestone 17.8's own investigation
# found (docs/TECHNOLOGY_VALIDATION_RESULTS.md, the Ankush Mehta case):
# a job heading that combines designation, company, and location on
# one line ("### AI Engineer, Isourse - Delhi, India"), which the
# locator's company-suffix-keyword heuristic never recognizes as a
# `company` at all (Isourse has no corporate suffix) — so
# `span.company` stays None. The OLD company-name-only matcher
# (Milestone 17.8) could therefore never match this job at all,
# leaving its (genuinely leaked) technology unchecked. The NEW
# confidence-based matcher (Milestone 17.9, wired in this milestone)
# still confidently matches it via designation + date overlap.
IMPROVED_MATCHING_FIXTURE_MARKDOWN = (
    "# Grace Hopper\n"
    "\n"
    "## Experience\n"
    "### AI Engineer, Isourse - Delhi, India\n"
    "**August 2025 - Present**\n"
    "- Built a RAG system using Python.\n"
    "\n"
    "## Skills\n"
    "Python, Kubernetes\n"
)
JOB_WITH_UNEXTRACTABLE_COMPANY = {
    "company": "Isourse",
    "designation": "AI Engineer",
    "employment_type": None,
    "location": None,
    "start_date": "August 2025",
    "end_date": None,
    "currently_working": True,
    "duration": None,
    "responsibilities": ["Built a RAG system using Python."],
    # "Python" is genuine (present in the job's own span); "Kubernetes"
    # is leaked from Skills — WEV-015 should still catch this even
    # though `company` alone can't identify the right span.
    "technologies": ["Python", "Kubernetes"],
}


class TestFullChainImprovedMatchingViaConfidenceScoring:
    """Milestone 17.10's own explicit deliverable: proof that wiring in
    the confidence-based matcher measurably improves real coverage
    through the full pipeline, not just in isolated unit tests — a job
    the old company-name-only approach could never match (because the
    locator itself never extracts a `company` for this heading format)
    is now confidently matched via designation + date overlap, and
    WEV-015's technology check runs against it correctly."""

    def test_job_with_unextractable_company_is_still_matched_and_validated(
        self, tmp_path, output_cleanup
    ):
        resume_name = "test_resume_wev015_improved_matching_integration"
        output_cleanup.append(resume_name)
        markdown_path = tmp_path / f"{resume_name}.md"
        markdown_path.write_text(IMPROVED_MATCHING_FIXTURE_MARKDOWN, encoding="utf-8")

        # Confirms the fixture actually reproduces the real bottleneck:
        # the locator finds exactly one span, but never recognizes a
        # `company` for it at all.
        spans = WorkExperienceLocator().locate(
            markdown_path.read_text(encoding="utf-8")
        )
        assert len(spans) == 1
        assert spans[0].company is None

        mock_personal_information_extractor = MagicMock(spec=PersonalInformationExtractor)
        mock_personal_information_extractor.extract.return_value = _module_result(
            PERSONAL_INFO_JSON
        )
        mock_contact_information_extractor = MagicMock(spec=ContactInformationExtractor)
        mock_contact_information_extractor.extract.return_value = _module_result(
            CONTACT_INFO_JSON
        )
        mock_skills_extractor = MagicMock(spec=SkillsExtractor)
        mock_skills_extractor.extract.return_value = _module_result(
            {"skills": ["Python", "Kubernetes"]}
        )
        mock_professional_summary_extractor = MagicMock(spec=ProfessionalSummaryExtractor)
        mock_professional_summary_extractor.extract.return_value = _module_result(SUMMARY_JSON)
        mock_languages_extractor = MagicMock(spec=LanguagesExtractor)
        mock_languages_extractor.extract.return_value = _module_result(LANGUAGES_JSON)
        mock_education_extractor = MagicMock(spec=EducationExtractor)
        mock_education_extractor.extract.return_value = _module_result(EDUCATION_JSON)
        mock_work_experience_extractor = MagicMock(spec=WorkExperienceExtractor)
        mock_work_experience_extractor.extract.return_value = _module_result(
            {"work_experience": [JOB_WITH_UNEXTRACTABLE_COMPANY]}
        )

        pipeline = ParserPipeline(
            personal_information_extractor=mock_personal_information_extractor,
            contact_information_extractor=mock_contact_information_extractor,
            skills_extractor=mock_skills_extractor,
            professional_summary_extractor=mock_professional_summary_extractor,
            languages_extractor=mock_languages_extractor,
            work_experience_extractor=mock_work_experience_extractor,
            education_extractor=mock_education_extractor,
            project_extractor=_mock_project_extractor(),
            # work_experience_validator: real (default) — internally
            # uses a real WorkExperienceLocator, including its
            # match_jobs_to_spans(), for both locate() and matching.
        )

        result = pipeline.run(markdown_path)

        # Confidently matched (not left unchanged with a Warning) and
        # correctly cleaned: "Kubernetes" removed, "Python" kept.
        corrected_job = result.merged_json["work_experience"][0]
        assert corrected_job["technologies"] == ["Python"]

        report = result.work_experience_validation_report
        wev_015_issues = [issue for issue in report.issues if issue.rule_id == "WEV-015"]
        assert len(wev_015_issues) == 1
        assert wev_015_issues[0].auto_fixed is True
        assert wev_015_issues[0].severity != "Warning"
        assert wev_015_issues[0].confidence is not None
        assert wev_015_issues[0].confidence > 0.5
