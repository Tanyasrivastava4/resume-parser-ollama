"""
tests/test_milestone_v6_6_boundary_correction.py

Targeted, deterministic tests for
scripts/_milestone_v6_6_boundary_correction.py (Milestone V6.6).
Covers exactly the required cases A-E from the milestone brief, using
real corpus source where practical and synthetic fixtures only to
isolate a mechanism not otherwise represented in the real corpus.

No LLM calls.
"""
from pathlib import Path

import pytest

from scripts._milestone_v6_6_boundary_correction import find_skills_section

V5_6_SOURCE_DIR = Path("output/v2_4h_benchmark/stage1/markdown")
M23_67_SOURCE_DIR = Path("output/markdown")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------
# A. Resume 005 real regression: boundary stops before Awards/Reference/
#    Personal Details material.
# ---------------------------------------------------------------------

@pytest.mark.local_data('corpus_markdown', 'v2_4h_markdown')
class TestResume005RealRegression:
    def test_boundary_excludes_awards_reference_personal_details(self):
        text = _read(V5_6_SOURCE_DIR / "Arjun Sharma_Data Architect_GT.md")
        sec = find_skills_section(text)
        assert sec["found"] is True
        body = sec["cleaned_body_text"]
        for forbidden in ("AWARDS AND ACHIVEMENTS", "REFERENCE:", "PERSONAL DETAILS",
                           "hod.mgt@gangainstitute.com", "mailto:"):
            assert forbidden not in body, f"{forbidden!r} leaked into the Skills body"

    def test_boundary_terminated_by_bare_text_next_section_title(self):
        text = _read(V5_6_SOURCE_DIR / "Arjun Sharma_Data Architect_GT.md")
        sec = find_skills_section(text)
        assert sec["end_reason"] == "bare_text_next_section_title"

    def test_real_skills_content_still_present(self):
        text = _read(V5_6_SOURCE_DIR / "Arjun Sharma_Data Architect_GT.md")
        sec = find_skills_section(text)
        assert "Vertical Exposure" in sec["cleaned_body_text"]
        assert "AzureDevops" in sec["cleaned_body_text"]


# ---------------------------------------------------------------------
# Newly-discovered-in-this-milestone regressions on the same lineage:
# 013, 016, 021 also overran under the v2_4h_benchmark lineage.
# ---------------------------------------------------------------------

@pytest.mark.local_data('corpus_markdown', 'v2_4h_markdown')
class TestOtherV5_6ResumesAlsoCorrected:
    def test_resume_013_no_longer_runs_into_hyperlink_annotations(self):
        text = _read(V5_6_SOURCE_DIR / "Kishan Yadav_Data Scientist_Intileo.md")
        sec = find_skills_section(text)
        assert "drive.google.com" not in sec["cleaned_body_text"]
        assert "Page 3:" not in sec["cleaned_body_text"]

    def test_resume_016_no_longer_runs_into_hyperlink_annotations(self):
        text = _read(V5_6_SOURCE_DIR / "Mukul Kanojia_Full Stack Developer_Intileo.md")
        sec = find_skills_section(text)
        assert "github.com" not in sec["cleaned_body_text"]
        assert "linkedin.com" not in sec["cleaned_body_text"]

    def test_resume_021_no_longer_runs_into_work_experience(self):
        text = _read(V5_6_SOURCE_DIR / "PRAGNESH S. SHAH_Sr. Associate-IT Ops_Intileo.md")
        sec = find_skills_section(text)
        assert "Name of Organisation" not in sec["cleaned_body_text"]
        assert "Designation:" not in sec["cleaned_body_text"]

    def test_resume_023_already_correct_unchanged(self):
        text = _read(V5_6_SOURCE_DIR / "Ritika Jain_AM2-Global Audit Delivery-Australia Region_GT-Intileo.md")
        sec = find_skills_section(text)
        assert sec["end_reason"] == "end_of_document"
        assert len(sec["cleaned_body_text"]) == 371  # unchanged from pre-fix

    def test_resume_001_still_correctly_not_found_unrelated_defect(self):
        # A different, already-documented Stage-1 defect (column-bleed
        # fusing "Summary" and "Technical Skills") -- must remain
        # unaffected by this milestone's fix.
        text = _read(V5_6_SOURCE_DIR / "AMAN_CV_BCA.md")
        sec = find_skills_section(text)
        assert sec["found"] is False


# ---------------------------------------------------------------------
# B. Resume 006: existing invalid/unsafe boundary remains invalid.
# ---------------------------------------------------------------------

@pytest.mark.local_data('corpus_markdown', 'v2_4h_markdown')
class TestResume006RemainsUnsafe:
    def test_still_flagged_implausible(self):
        text = _read(M23_67_SOURCE_DIR / "AshishKumarSingh_RBM_Pune_Stonelam.md")
        sec = find_skills_section(text)
        assert sec["found"] is True
        assert sec["boundary_content_implausible"] is True


# ---------------------------------------------------------------------
# C. Resume 017: NOT_FOUND behavior remains unchanged.
# ---------------------------------------------------------------------

@pytest.mark.local_data('corpus_markdown', 'v2_4h_markdown')
class TestResume017StillNotFound:
    def test_still_not_found(self):
        text = _read(M23_67_SOURCE_DIR / "Naukri_NileshShivramMagare[6y_6m].md")
        sec = find_skills_section(text)
        assert sec["found"] is False


# ---------------------------------------------------------------------
# D. Valid Skills layouts with nested category/subheading lines --
#    legitimate nested headings must not be treated as section ends.
# ---------------------------------------------------------------------

@pytest.mark.local_data('corpus_markdown', 'v2_4h_markdown')
class TestNestedSubheadingsPreserved:
    def test_real_multi_subheading_resume_013_unaffected(self):
        # 013 (output/markdown lineage) has many '##'/'###' category
        # subheadings (Languages, Tools & Platforms, Cloud & Big Data,
        # ...) all deeper than or equal to its own heading level --
        # none should be mistaken for a document-ending next-section
        # title.
        text = _read(M23_67_SOURCE_DIR / "Kishan Yadav_Data Scientist_Intileo.md")
        sec = find_skills_section(text)
        assert sec["found"] is True
        assert "Tools & Platforms" in sec["cleaned_body_text"]
        assert "Cloud & Big Data" in sec["cleaned_body_text"]

    def test_synthetic_bare_text_category_subheadings_not_terminated(self):
        # Synthetic: isolates the mechanism (bare-text CATEGORY labels
        # that are NOT in the next-section vocabulary at all) from real
        # corpus noise.
        md = (
            "Skills\n"
            "Programming\n"
            "- Python\n"
            "- SQL\n"
            "Tools\n"
            "- Git\n"
            "- Docker\n"
        )
        sec = find_skills_section(md)
        assert sec["found"] is True
        assert "Python" in sec["cleaned_body_text"]
        assert "Docker" in sec["cleaned_body_text"]

    def test_legitimate_programming_languages_subheading_still_preserved(self):
        # Confirms the V5.9 nested-Languages/denylist mechanism still
        # works correctly and is not short-circuited by the new
        # bare-text check (which deliberately excludes "Languages" from
        # its own vocabulary for exactly this reason).
        md = "Skills\n### Languages\nJavaScript, TypeScript, Python\n### Tools\nGit\n"
        sec = find_skills_section(md)
        assert sec["found"] is True
        assert sec["excluded_nonskills_spans"] == []
        assert "JavaScript" in sec["cleaned_body_text"]


# ---------------------------------------------------------------------
# E. A non-markdown section transition: a genuine following section
#    title without '#' heading syntax must terminate Skills.
# ---------------------------------------------------------------------

class TestBareTextSectionTransition:
    def test_bare_education_title_terminates_section(self):
        md = "Skills\n- Python\n- SQL\nEducation\n- B.Tech, 2020\n"
        sec = find_skills_section(md)
        assert sec["found"] is True
        assert "Python" in sec["cleaned_body_text"]
        assert "B.Tech" not in sec["cleaned_body_text"]
        assert sec["end_reason"] == "bare_text_next_section_title"

    def test_compound_title_with_and_terminates_section(self):
        # Real corpus pattern: "Awards and Achievements" / "Awards & Recognition".
        md = "Skills\n- Python\nAwards and Achievements\n- Employee of the year\n"
        sec = find_skills_section(md)
        assert "Employee of the year" not in sec["cleaned_body_text"]
        assert sec["end_reason"] == "bare_text_next_section_title"

    def test_long_prose_line_containing_vocab_word_does_not_terminate(self):
        # A real Skills bullet that happens to CONTAIN a vocab word
        # (e.g. "Awards") deep in a long sentence must NOT be mistaken
        # for a section title -- the <=40-char guard exists precisely
        # for this.
        md = (
            "Skills\n"
            "- Python\n"
            "- Received multiple internal awards for outstanding project delivery and mentorship\n"
            "- SQL\n"
        )
        sec = find_skills_section(md)
        assert "SQL" in sec["cleaned_body_text"]
        assert "outstanding project delivery" in sec["cleaned_body_text"]

    def test_bullet_line_containing_vocab_word_never_terminates(self):
        # A bulleted line is never itself treated as a section title,
        # regardless of content.
        md = "Skills\n- Awards\n- Python\n"
        sec = find_skills_section(md)
        assert "Python" in sec["cleaned_body_text"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
