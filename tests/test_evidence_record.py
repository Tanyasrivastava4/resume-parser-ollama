"""
tests/test_evidence_record.py

Tests for services/evidence_record.py, with particular focus on
Milestone V2.9's review-priority annotation for Work Experience:

- zero mutation of extracted data (the mandatory Step 5/6 requirement
  from docs/MILESTONE_V2_9_WE_REVIEW_PRIORITIZATION.md)
- the three review_priority states (NONE/STANDARD/ELEVATED) fire on
  exactly the evidence combinations they're documented to
- Education entries are never assigned a review_priority or
  company_equals_designation value (both stay None) -- this milestone
  is Work Experience only, by explicit non-goal.

Pure in-memory logic, no external dependencies (no Ollama, no real
pipeline output) -- mirrors tests/test_education_validator.py's own
structure.
"""

import copy

import pytest

from services.education_validator import EducationValidator
from services.evidence_record import (
    build_contact_information_evidence,
    build_education_evidence,
    build_languages_evidence,
    build_personal_information_evidence,
    build_professional_summary_evidence,
    build_work_experience_evidence,
)
from services.work_experience_validator import WorkExperienceValidator


def _job(**overrides):
    """A minimal, otherwise-valid work_experience entry with all ten
    schema fields, overridden as needed per test."""
    base = {
        "company": "Acme Corp",
        "designation": "Software Engineer",
        "employment_type": None,
        "location": None,
        "start_date": "Jan 2020",
        "end_date": "Dec 2021",
        "currently_working": False,
        "duration": None,
        "responsibilities": [],
        "technologies": [],
    }
    base.update(overrides)
    return base


def _education_entry(**overrides):
    base = {
        "institution": "State University",
        "degree": "B.Tech",
        "specialization": None,
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
def wev():
    return WorkExperienceValidator()


@pytest.fixture
def edv():
    return EducationValidator()


class TestZeroMutation:
    """The mandatory Milestone V2.9 requirement: building an evidence
    record, with or without review-priority computation, must never
    change a single extracted field."""

    def test_work_experience_entries_unchanged_after_evidence_build(self, wev):
        jobs = [_job(company="Textile (X)", designation="Textile (X)")]
        markdown = "## Work Experience\n### Software Engineer at Acme Corp\nJan 2020 - Dec 2021\n"

        validated, report = wev.validate(markdown, jobs)
        before = copy.deepcopy(validated)

        build_work_experience_evidence(validated, markdown, report)

        assert validated == before

    def test_calling_build_evidence_twice_is_idempotent_and_non_mutating(self, wev):
        jobs = [_job()]
        markdown = "## Work Experience\n### Software Engineer at Acme Corp\nJan 2020 - Dec 2021\n"
        validated, report = wev.validate(markdown, jobs)
        before = copy.deepcopy(validated)

        rec1 = build_work_experience_evidence(validated, markdown, report)
        rec2 = build_work_experience_evidence(validated, markdown, report)

        assert validated == before
        assert rec1 == rec2

    def test_education_entries_unchanged_after_evidence_build(self, edv):
        entries = [_education_entry()]
        markdown = "## Education\n| Degree | Institution | Year |\n|---|---|---|\n| B.Tech | State University | 2020 |\n"
        validated, report = edv.validate(entries, markdown)
        before = copy.deepcopy(validated)

        build_education_evidence(validated, markdown, report)

        assert validated == before


class TestReviewPriorityStates:
    """Milestone V2.9: review_priority must reflect exactly the
    evidence-based rules it's documented to, never more."""

    def test_confidently_matched_job_gets_priority_none(self, wev):
        jobs = [_job()]
        markdown = "## Work Experience\n### Software Engineer at Acme Corp\nJan 2020 - Dec 2021\n- Did things\n"
        validated, report = wev.validate(markdown, jobs)

        rec = build_work_experience_evidence(validated, markdown, report)

        assert rec.entries[0].locator_matched is True
        assert rec.entries[0].review_priority == "NONE"

    def test_unmatched_job_with_company_equals_designation_gets_elevated(self, wev):
        jobs = [_job(company="Textile (Pratapsons)", designation="Textile (Pratapsons)")]
        markdown = "## Something Else Entirely\nNo work experience section at all.\n"
        validated, report = wev.validate(markdown, jobs)

        rec = build_work_experience_evidence(validated, markdown, report)

        assert rec.entries[0].locator_matched is False
        assert rec.entries[0].company_equals_designation is True
        assert rec.entries[0].review_priority == "ELEVATED"

    def test_unmatched_job_with_distinct_company_and_designation_gets_standard(self, wev):
        jobs = [_job(company="Acme Corp", designation="Software Engineer")]
        markdown = "## Something Else Entirely\nNo work experience section at all.\n"
        validated, report = wev.validate(markdown, jobs)

        rec = build_work_experience_evidence(validated, markdown, report)

        assert rec.entries[0].locator_matched is False
        assert rec.entries[0].company_equals_designation is False
        assert rec.entries[0].review_priority == "STANDARD"

    def test_company_equals_designation_is_case_and_whitespace_insensitive(self, wev):
        jobs = [_job(company="  Textile (X)  ", designation="TEXTILE (X)")]
        markdown = "## Something Else Entirely\nNo work experience section at all.\n"
        validated, report = wev.validate(markdown, jobs)

        rec = build_work_experience_evidence(validated, markdown, report)

        assert rec.entries[0].company_equals_designation is True

    def test_no_markdown_supplied_leaves_review_priority_none_not_guessed(self, wev):
        jobs = [_job(company="Textile (X)", designation="Textile (X)")]
        validated, report = wev.validate("", jobs)

        rec = build_work_experience_evidence(validated, markdown_text=None, validation_report=report)

        assert rec.entries[0].locator_matched is None
        assert rec.entries[0].review_priority is None

    def test_education_never_gets_review_priority_or_company_equals_designation(self, edv):
        entries = [_education_entry()]
        markdown = "## Education\n| Degree | Institution | Year |\n|---|---|---|\n| B.Tech | State University | 2020 |\n"
        validated, report = edv.validate(entries, markdown)

        rec = build_education_evidence(validated, markdown, report)

        assert rec.entries[0].review_priority is None
        assert rec.entries[0].company_equals_designation is None


def _make_pdf(tmp_path, links):
    """A minimal one-page PDF with the given [(rect_tuple, uri), ...]
    link annotations, for testing build_contact_information_evidence's
    PDF-hyperlink-annotation path without touching any real corpus PDF.
    """
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    for rect, uri in links:
        page.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(*rect), "uri": uri})
    pdf_path = tmp_path / "synthetic.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _ci(**overrides):
    """A minimal, otherwise-empty contact_information object, all 5
    schema keys always present."""
    base = {"email": [], "phone": [], "linkedin": None, "github": None, "portfolio": None}
    base.update(overrides)
    return base


class TestContactInformationEvidence:
    """Milestone V3.8 -- Contact Information Evidence Framework.

    Covers every case Step 4 explicitly requires: visible-text-only,
    annotation-only, both agreeing, sources disagreeing, annotation
    absence, non-contact annotations, idempotency, and zero mutation.
    """

    def test_zero_mutation_of_extracted_output(self, tmp_path):
        ci = _ci(email=["ada@example.com"], linkedin="https://linkedin.com/in/ada")
        ci_snapshot = copy.deepcopy(ci)
        markdown = "Ada Lovelace\nada@example.com\nlinkedin.com/in/ada\n"

        build_contact_information_evidence(ci, markdown_text=markdown)

        assert ci == ci_snapshot

    def test_idempotency_calling_twice_yields_identical_result(self, tmp_path):
        ci = _ci(email=["ada@example.com"])
        markdown = "Ada Lovelace\nada@example.com\n"

        rec1 = build_contact_information_evidence(ci, markdown_text=markdown)
        rec2 = build_contact_information_evidence(ci, markdown_text=markdown)

        assert rec1 == rec2

    def test_visible_text_only_evidence(self):
        ci = _ci(email=["ada@example.com"])
        markdown = "Ada Lovelace\nContact: ada@example.com\n"

        rec = build_contact_information_evidence(ci, markdown_text=markdown)
        field = next(f for f in rec.fields if f.field_name == "email")

        assert field.visible_text_checked is True
        assert field.visible_text_matches == ["ada@example.com"]
        assert field.annotations_checked is None  # no PDF supplied -- never guessed
        assert field.evidence_summary == "FULLY_CORROBORATED"

    def test_annotation_only_evidence(self, tmp_path):
        ci = _ci(linkedin="https://www.linkedin.com/in/ada")
        # Markdown body has no visible URL at all -- only a bare label,
        # matching the real corpus pattern V3.6/V3.7 found (e.g. AMAN_CV_BCA).
        markdown = "Ada Lovelace\nlinkedin\n"
        pdf_path = _make_pdf(tmp_path, [((50, 40, 150, 60), "https://www.linkedin.com/in/ada")])

        rec = build_contact_information_evidence(ci, markdown_text=markdown, pdf_path=pdf_path)
        field = next(f for f in rec.fields if f.field_name == "linkedin")

        assert field.visible_text_checked is True
        assert field.visible_text_matches == []
        assert field.annotations_checked is True
        assert field.annotation_matches == ["https://www.linkedin.com/in/ada"]
        assert field.evidence_summary == "FULLY_CORROBORATED"

    def test_both_sources_agree(self, tmp_path):
        ci = _ci(linkedin="https://www.linkedin.com/in/ada")
        markdown = "Ada Lovelace\nhttps://www.linkedin.com/in/ada\n"
        pdf_path = _make_pdf(tmp_path, [((50, 40, 150, 60), "https://www.linkedin.com/in/ada")])

        rec = build_contact_information_evidence(ci, markdown_text=markdown, pdf_path=pdf_path)
        field = next(f for f in rec.fields if f.field_name == "linkedin")

        assert field.visible_text_matches == ["https://www.linkedin.com/in/ada"]
        assert field.annotation_matches == ["https://www.linkedin.com/in/ada"]
        assert field.evidence_summary == "FULLY_CORROBORATED"

    def test_sources_disagree_with_no_corroboration_at_all(self, tmp_path):
        """Synthetic case: neither the visible text nor the annotation
        agree with the extracted value, and a same-type annotation with
        a DIFFERENT value exists -- this is the one state the real
        22-resume corpus never actually reaches (every real V3.7
        mismatch happened to have a corroborating visible-text match),
        so this test exercises it directly rather than leaving it
        unverified."""
        ci = _ci(linkedin="https://www.linkedin.com/in/wrong-value")
        markdown = "Ada Lovelace\nNo linkedin text visible anywhere.\n"
        pdf_path = _make_pdf(tmp_path, [((50, 40, 150, 60), "https://www.linkedin.com/in/ada")])

        rec = build_contact_information_evidence(ci, markdown_text=markdown, pdf_path=pdf_path)
        field = next(f for f in rec.fields if f.field_name == "linkedin")

        assert field.visible_text_matches == []
        assert field.annotation_matches == []
        assert field.unmatched_same_type_annotations == ["https://www.linkedin.com/in/ada"]
        assert field.evidence_summary == "EVIDENCE_DISAGREEMENT"
        # The disagreement must never touch the extracted value itself.
        assert ci["linkedin"] == "https://www.linkedin.com/in/wrong-value"

    def test_known_bad_annotation_cannot_override_a_visible_text_corroborated_value(self, tmp_path):
        """Direct regression test for V3.7's two confirmed false-positive
        cases (placeholder mailto:, stale linkedin annotation): when
        visible text already corroborates the extracted value, a
        disagreeing same-type annotation must be recorded transparently
        but must NOT flip evidence_summary away from FULLY_CORROBORATED."""
        ci = _ci(email=["real@example.com"])
        markdown = "Ada Lovelace\nreal@example.com\n"
        pdf_path = _make_pdf(tmp_path, [((50, 40, 150, 60), "mailto:placeholder@example.com")])

        rec = build_contact_information_evidence(ci, markdown_text=markdown, pdf_path=pdf_path)
        field = next(f for f in rec.fields if f.field_name == "email")

        assert field.evidence_summary == "FULLY_CORROBORATED"
        assert field.unmatched_same_type_annotations == ["placeholder@example.com"]
        assert ci["email"] == ["real@example.com"]

    def test_annotation_absence_is_none_not_false(self):
        """No pdf_path supplied at all -- annotations_checked must be
        None (unknown), never False (which would misrepresent "checked,
        found none" as a fact this function never established)."""
        ci = _ci(email=["ada@example.com"])
        markdown = "Ada Lovelace\nada@example.com\n"

        rec = build_contact_information_evidence(ci, markdown_text=markdown, pdf_path=None)
        field = next(f for f in rec.fields if f.field_name == "email")

        assert field.annotations_checked is None
        assert field.annotation_matches == []
        assert field.evidence_summary == "FULLY_CORROBORATED"  # visible text alone is enough

    def test_unopenable_pdf_leaves_annotations_checked_none(self, tmp_path):
        ci = _ci(email=["ada@example.com"])
        bad_pdf = tmp_path / "not_a_real_pdf.pdf"
        bad_pdf.write_bytes(b"this is not a pdf")

        rec = build_contact_information_evidence(ci, markdown_text=None, pdf_path=bad_pdf)
        field = next(f for f in rec.fields if f.field_name == "email")

        assert field.annotations_checked is None
        assert field.evidence_summary == "NO_EVIDENCE_SOURCES_AVAILABLE"

    def test_non_contact_annotations_never_become_contact_values(self, tmp_path):
        """A GitHub repo/project link (deep path) and an unrelated
        external link must never be treated as evidence for ANY field
        -- not matched, not even shown as a disagreement, since they
        were never plausibly claiming to be contact information at
        all. Mirrors V3.7's confirmed corpus finding (~30% of real
        annotations are non-contact)."""
        ci = _ci(github="https://github.com/ada")
        markdown = "Ada Lovelace\nhttps://github.com/ada\n"
        pdf_path = _make_pdf(
            tmp_path,
            [
                ((50, 40, 150, 60), "https://github.com/ada"),  # real profile -- corroborates
                ((50, 80, 150, 100), "https://github.com/ada/some-project"),  # deep path -- excluded
                ((50, 120, 150, 140), "https://vendor.example.com/login"),  # unrelated -- excluded
            ],
        )

        rec = build_contact_information_evidence(ci, markdown_text=markdown, pdf_path=pdf_path)
        github_field = next(f for f in rec.fields if f.field_name == "github")
        portfolio_field = next(f for f in rec.fields if f.field_name == "portfolio")

        assert github_field.annotation_matches == ["https://github.com/ada"]
        assert github_field.unmatched_same_type_annotations == []  # repo link silently excluded, not flagged
        assert github_field.evidence_summary == "FULLY_CORROBORATED"
        # Neither non-contact link leaks into the empty portfolio field.
        assert portfolio_field.extracted_values == []
        assert portfolio_field.evidence_summary == "NO_EXTRACTED_VALUE"

    def test_no_extracted_value_is_its_own_state_regardless_of_annotations(self, tmp_path):
        ci = _ci()  # every field empty/null
        pdf_path = _make_pdf(tmp_path, [((50, 40, 150, 60), "https://www.linkedin.com/in/someone")])

        rec = build_contact_information_evidence(ci, markdown_text="No contact info here.", pdf_path=pdf_path)

        for f in rec.fields:
            assert f.extracted_values == []
            assert f.evidence_summary == "NO_EXTRACTED_VALUE"

    def test_all_five_fields_always_present_in_schema_order(self):
        ci = _ci(email=["a@example.com"])
        rec = build_contact_information_evidence(ci)

        assert [f.field_name for f in rec.fields] == [
            "email", "phone", "linkedin", "github", "portfolio",
        ]

    def test_multi_value_phone_array_partial_corroboration(self):
        ci = _ci(phone=["+91-8097305823", "+91-9819008544"])
        markdown = "Ada Lovelace\n+91-8097305823\n"  # only the first number is visible

        rec = build_contact_information_evidence(ci, markdown_text=markdown)
        field = next(f for f in rec.fields if f.field_name == "phone")

        assert field.visible_text_matches == ["+91-8097305823"]
        assert field.evidence_summary == "PARTIALLY_CORROBORATED"
        assert ci["phone"] == ["+91-8097305823", "+91-9819008544"]  # unchanged


class TestLanguagesEvidence:
    """Milestone V4.1 -- Languages Evidence Foundation. Covers the exact
    rule V4.0 demonstrated safe (whole-document, item-level,
    case-insensitive, whole-word text-presence), its documented
    limitation (cannot detect project-description misattribution), and
    the required absence-of-evidence / zero-mutation properties."""

    def test_source_text_present(self):
        rec = build_languages_evidence(["English"], markdown_text="Languages: English, Hindi\n")

        assert rec.values[0].source_text_checked is True
        assert rec.values[0].source_text_present is True
        assert rec.values[0].evidence_summary == "SOURCE_TEXT_PRESENT"

    def test_no_source_text_evidence_pure_hallucination_case(self):
        """Direct regression test for V4.0's confirmed resume-011/019
        pattern: a value with zero textual anchor anywhere in the
        document."""
        rec = build_languages_evidence(["Hindi"], markdown_text="Skilled in SQL, Power BI, and Python.\n")

        assert rec.values[0].source_text_checked is True
        assert rec.values[0].source_text_present is False
        assert rec.values[0].evidence_summary == "NO_SOURCE_TEXT_EVIDENCE"
        # Absence of evidence must never mutate or flag the value for removal.
        assert rec.values[0].value == "Hindi"

    def test_project_description_misattribution_is_not_detected_by_design(self):
        """Direct regression test for V4.0's confirmed resume-004
        pattern: the word IS present in the source text (in a project
        description, not a language-ability statement), so this
        whole-document check reports SOURCE_TEXT_PRESENT -- correctly
        reflecting its own documented, disclosed limitation, not a
        correctness verdict."""
        markdown = (
            "Developed a machine translation model for English to "
            "Hindi Translation using PyTorch.\n"
        )
        rec = build_languages_evidence(["English", "Hindi"], markdown_text=markdown)

        assert all(v.evidence_summary == "SOURCE_TEXT_PRESENT" for v in rec.values)

    def test_no_source_text_available_when_markdown_not_supplied(self):
        rec = build_languages_evidence(["English"], markdown_text=None)

        assert rec.values[0].source_text_checked is None
        assert rec.values[0].source_text_present is None
        assert rec.values[0].evidence_summary == "NO_SOURCE_TEXT_AVAILABLE"

    def test_empty_languages_list_produces_no_records(self):
        rec = build_languages_evidence([], markdown_text="Some resume text.\n")

        assert rec.values == []

    def test_case_insensitive_and_whole_word_matching(self):
        rec = build_languages_evidence(["english"], markdown_text="Fluent in ENGLISH and French.\n")

        assert rec.values[0].evidence_summary == "SOURCE_TEXT_PRESENT"

    def test_substring_is_not_a_false_match(self):
        """'Englishman' must not cause 'English' to register as present
        via a bare substring match -- the check uses word boundaries."""
        rec = build_languages_evidence(["English"], markdown_text="He is an Englishman by birth.\n")

        assert rec.values[0].evidence_summary == "NO_SOURCE_TEXT_EVIDENCE"

    def test_multiple_values_evaluated_independently(self):
        rec = build_languages_evidence(
            ["English", "Hindi"], markdown_text="Languages: English\n"
        )

        assert rec.values[0].evidence_summary == "SOURCE_TEXT_PRESENT"
        assert rec.values[1].evidence_summary == "NO_SOURCE_TEXT_EVIDENCE"

    def test_zero_mutation_of_extracted_list(self):
        languages = ["Marathi", "Hindi", "English"]
        snapshot = list(languages)

        build_languages_evidence(languages, markdown_text="Marathi, Hindi, English\n")

        assert languages == snapshot  # same values, same order, same length

    def test_order_is_preserved_in_the_evidence_record(self):
        languages = ["Marathi", "Hindi", "English"]

        rec = build_languages_evidence(languages, markdown_text="Marathi, Hindi, English\n")

        assert [v.value for v in rec.values] == ["Marathi", "Hindi", "English"]

    def test_idempotency_calling_twice_yields_identical_result(self):
        languages = ["English"]
        markdown = "Languages: English\n"

        rec1 = build_languages_evidence(languages, markdown_text=markdown)
        rec2 = build_languages_evidence(languages, markdown_text=markdown)

        assert rec1 == rec2


class TestProfessionalSummaryEvidence:
    """Milestone V4.5 -- Summary Evidence Foundation. Covers the exact
    signal V4.4 demonstrated safe (whole-document, whitespace-normalized
    verbatim-substring match, plus a raw token-overlap measurement --
    never a calibrated score), its documented limitation (formatting/
    single-character artifacts can break the strict match on genuinely-
    correct content), and the required absence-of-evidence / zero-
    mutation properties. Never reads or accepts a GT parameter -- GT is
    not part of this function's signature at all."""

    def test_source_supported_text(self):
        rec = build_professional_summary_evidence(
            "Full-Stack Web Developer with expertise in React.",
            markdown_text="Summary\nFull-Stack Web Developer with expertise in React.\n",
        )

        assert rec.source_text_checked is True
        assert rec.whole_document_verbatim_match is True
        assert rec.token_overlap_ratio == 1.0
        assert rec.evidence_summary == "SOURCE_TEXT_VERBATIM_MATCH"

    def test_text_without_direct_source_evidence(self):
        """Direct regression test for V4.4's confirmed resume-010
        pattern: a reconstruction that is NOT a verbatim match and
        shows a real (not 1.0) token-overlap gap."""
        rec = build_professional_summary_evidence(
            "Known for a strong technical foundation, a commitment to detail.",
            markdown_text="Known for a strong technical founda on, a en on to detail.\n",
        )

        assert rec.whole_document_verbatim_match is False
        assert rec.evidence_summary == "NO_VERBATIM_MATCH_FOUND"
        assert rec.token_overlap_ratio < 1.0
        # The evidence layer must never itself assert the content is wrong --
        # only the extracted value and the raw signals are returned.
        assert rec.extracted_value == "Known for a strong technical foundation, a commitment to detail."

    def test_unavailable_missing_source_text(self):
        rec = build_professional_summary_evidence("Some summary.", markdown_text=None)

        assert rec.source_text_checked is None
        assert rec.whole_document_verbatim_match is None
        assert rec.token_overlap_ratio is None
        assert rec.evidence_summary == "NO_SOURCE_TEXT_AVAILABLE"

    def test_paraphrase_like_formatting_difference_is_not_treated_as_contradiction(self):
        """Direct regression test for V4.4's confirmed resume-013/020
        pattern: a whole-document substring match fails on pure
        formatting/bullet-marker differences even though content is
        100% present -- the evidence_summary label must stay purely
        descriptive (no verbatim match found), never implying the
        content itself is wrong, and token_overlap_ratio must show the
        full support."""
        rec = build_professional_summary_evidence(
            "– Full-Stack AI Engineer specializing in LLM applications.\n– Experienced in building platforms.",
            markdown_text="Full-Stack AI Engineer specializing in LLM applications. Experienced in building platforms.\n",
        )

        assert rec.whole_document_verbatim_match is False
        assert rec.token_overlap_ratio == 1.0
        assert rec.evidence_summary == "NO_VERBATIM_MATCH_FOUND"
        # Nothing here escalates to a rejection/contradiction label.
        assert rec.evidence_summary not in ("CONTRADICTION", "UNSUPPORTED", "INCORRECT")

    def test_empty_summary(self):
        rec = build_professional_summary_evidence(None, markdown_text="Resume text with no labeled summary.")

        assert rec.extracted_value is None
        assert rec.evidence_summary == "NO_SUMMARY_EXTRACTED"
        assert rec.whole_document_verbatim_match is None
        assert rec.token_overlap_ratio is None

    def test_empty_string_summary_treated_same_as_null(self):
        rec = build_professional_summary_evidence("   ", markdown_text="Some resume text.")

        assert rec.extracted_value is None
        assert rec.evidence_summary == "NO_SUMMARY_EXTRACTED"

    def test_zero_mutation_of_extracted_value(self):
        original = "Full-Stack Web Developer with expertise in React."
        markdown = f"Summary\n{original}\n"

        build_professional_summary_evidence(original, markdown_text=markdown)

        assert original == "Full-Stack Web Developer with expertise in React."

    def test_idempotency_calling_twice_yields_identical_result(self):
        summary = "Full-Stack Web Developer with expertise in React."
        markdown = f"Summary\n{summary}\n"

        rec1 = build_professional_summary_evidence(summary, markdown_text=markdown)
        rec2 = build_professional_summary_evidence(summary, markdown_text=markdown)

        assert rec1 == rec2

    def test_never_reads_or_requires_ground_truth(self):
        """The function signature itself has no GT parameter -- direct
        structural proof it cannot use GT at runtime, per V4.4's own
        explicit finding that GT often diverges from source."""
        import inspect
        sig = inspect.signature(build_professional_summary_evidence)
        assert "ground_truth" not in sig.parameters
        assert "gt" not in sig.parameters


def _pi(**overrides):
    base = {"full_name": None, "first_name": None, "last_name": None, "current_location": None}
    base.update(overrides)
    return base


class TestPersonalInformationEvidence:
    """Milestone V4.8 -- Personal Information Evidence Foundation.
    Covers V4.7's exact two demonstrated signals (name-token
    completeness, location verbatim-presence), their documented
    limitations, and the required absence-of-evidence / zero-mutation
    properties. Never reads or accepts a GT parameter."""

    # --- Name completeness ---

    def test_fully_represented_full_name(self):
        rec = build_personal_information_evidence(
            _pi(full_name="Chirag Sharma", first_name="Chirag", last_name="Sharma")
        )

        assert rec.name_completeness.checked is True
        assert rec.name_completeness.missing_tokens == []
        assert rec.name_completeness.evidence_summary == "NAME_TOKENS_FULLY_REPRESENTED"

    def test_missing_middle_name_token(self):
        """Direct regression test for V4.7's confirmed resume-017/020
        pattern: full_name has a middle token the first/last split
        drops entirely."""
        rec = build_personal_information_evidence(
            _pi(full_name="NILESH SHIVRAM MAGARE", first_name="NILESH", last_name="MAGARE")
        )

        assert rec.name_completeness.missing_tokens == ["shivram"]
        assert rec.name_completeness.evidence_summary == "NAME_TOKENS_MISSING"

    def test_single_letter_middle_initial_is_not_a_false_discrepancy(self):
        """Direct regression test for V4.7's confirmed resume-021
        pattern (the one case that WOULD be a false positive without
        excluding single-character tokens)."""
        rec = build_personal_information_evidence(
            _pi(full_name="PRAGNESH S. SHAH", first_name="PRAGNESH", last_name="SHAH")
        )

        assert rec.name_completeness.missing_tokens == []
        assert rec.name_completeness.evidence_summary == "NAME_TOKENS_FULLY_REPRESENTED"

    def test_title_prefix_and_dropped_given_name(self):
        """Direct regression test for V4.7's confirmed resume-008
        pattern: title kept in first_name, a real name token dropped
        from last_name entirely."""
        rec = build_personal_information_evidence(
            _pi(full_name="CA ISHIKA SHARMA", first_name="CA", last_name="SHARMA")
        )

        assert rec.name_completeness.missing_tokens == ["ishika"]
        assert rec.name_completeness.evidence_summary == "NAME_TOKENS_MISSING"

    def test_missing_full_name_is_not_a_discrepancy(self):
        rec = build_personal_information_evidence(_pi())

        assert rec.name_completeness.checked is None
        assert rec.name_completeness.missing_tokens == []
        assert rec.name_completeness.evidence_summary == "NO_FULL_NAME_TO_CHECK"

    def test_missing_first_and_last_name_reports_missing_tokens_honestly(self):
        rec = build_personal_information_evidence(_pi(full_name="Ada Lovelace"))

        assert rec.name_completeness.checked is True
        assert set(rec.name_completeness.missing_tokens) == {"ada", "lovelace"}
        assert rec.name_completeness.evidence_summary == "NAME_TOKENS_MISSING"

    # --- Location evidence ---

    def test_exact_source_presence(self):
        rec = build_personal_information_evidence(
            _pi(current_location="Mumbai, India"), markdown_text="Ada Lovelace\nMumbai, India\n"
        )

        assert rec.location.verbatim_in_source is True
        assert rec.location.evidence_summary == "SOURCE_TEXT_VERBATIM_MATCH"

    def test_source_absence_pure_fabrication_case(self):
        """Direct regression test for V4.7's confirmed resume-019
        pattern: a location with zero textual anchor anywhere."""
        rec = build_personal_information_evidence(
            _pi(current_location="Noida, India"), markdown_text="Ada Lovelace\nNo location text anywhere.\n"
        )

        assert rec.location.verbatim_in_source is False
        assert rec.location.evidence_summary == "NO_VERBATIM_MATCH_FOUND"

    def test_source_unavailable(self):
        rec = build_personal_information_evidence(_pi(current_location="Mumbai"), markdown_text=None)

        assert rec.location.source_text_checked is None
        assert rec.location.verbatim_in_source is None
        assert rec.location.evidence_summary == "NO_SOURCE_TEXT_AVAILABLE"

    def test_wrong_section_attribution_limitation_represented_honestly(self):
        """Direct regression test for V4.7's confirmed resume-004
        pattern: a location genuinely present in source (in the wrong
        section) is indistinguishable from a correct one to this
        signal -- documented and tested as a known limitation, not
        hidden."""
        rec = build_personal_information_evidence(
            _pi(current_location="Kanpur, UP"),
            markdown_text="Ada Lovelace\nEducation\nBTech, Kanpur, UP\n",
        )

        # The signal reports verbatim presence -- it does NOT and cannot
        # claim this is the candidate's own address.
        assert rec.location.verbatim_in_source is True
        assert rec.location.evidence_summary == "SOURCE_TEXT_VERBATIM_MATCH"

    def test_no_location_extracted(self):
        rec = build_personal_information_evidence(_pi(), markdown_text="Some resume text.")

        assert rec.location.extracted_value is None
        assert rec.location.evidence_summary == "NO_LOCATION_EXTRACTED"

    # --- Mutation / idempotency ---

    def test_zero_mutation_of_extracted_fields(self):
        pi = _pi(full_name="Ada Lovelace", first_name="Ada", last_name="Lovelace", current_location="London")
        snapshot = dict(pi)

        build_personal_information_evidence(pi, markdown_text="Ada Lovelace, London\n")

        assert pi == snapshot

    def test_idempotency_calling_twice_yields_identical_result(self):
        pi = _pi(full_name="Ada Lovelace", first_name="Ada", last_name="Lovelace", current_location="London")
        markdown = "Ada Lovelace, London\n"

        rec1 = build_personal_information_evidence(pi, markdown_text=markdown)
        rec2 = build_personal_information_evidence(pi, markdown_text=markdown)

        assert rec1 == rec2

    def test_never_reads_or_requires_ground_truth(self):
        import inspect
        sig = inspect.signature(build_personal_information_evidence)
        assert "ground_truth" not in sig.parameters
        assert "gt" not in sig.parameters
