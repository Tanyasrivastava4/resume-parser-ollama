"""
tests/test_project_misrouting_actionability.py

Milestone T7/Phase 1D -- adversarial safety analysis of the EXISTING,
UNMODIFIED `detect_project_content_in_work_experience`
(validators/misrouting_detector.py, unchanged since T3). This
milestone does not implement any action mechanism -- these tests exist
to document, with real and synthetic evidence, exactly why the
detector's current design does not (and, this milestone found, should
not yet) support automatic suppression/movement, per this milestone's
own final FLAG-ONLY decision.

Real-corpus cases (#1) use resume 001's actual T4/T5 real extraction
data, verbatim. Every other case is explicitly SYNTHETIC/ADVERSARIAL,
built to test one specific scenario; none is presented as real corpus
evidence.
"""
from __future__ import annotations

from validators.misrouting_detector import _fuzzy_contains, detect_project_content_in_work_experience


def _job(company, **overrides):
    base = {
        "company": company, "designation": "Engineer", "employment_type": "Full-time",
        "location": None, "start_date": "2020", "end_date": None, "currently_working": True,
        "duration": None, "responsibilities": [], "technologies": [],
    }
    base.update(overrides)
    return base


def _project(name, **overrides):
    base = {
        "name": name, "description": None, "role": None, "start_date": None,
        "end_date": None, "highlights": [], "technologies": [], "url": None,
    }
    base.update(overrides)
    return base


class TestRealResume001EvidencePopulation:
    """#1 -- real, verbatim T4/T5 data. Confirms the detector's real
    recall on the ONLY real corpus resume with any Project/Work-
    Experience naming overlap: 1 of 3 genuinely contaminated entries
    caught (see the milestone report's Phase B/C for the full,
    field-level content comparison establishing all three as real
    contamination, not merely name-similar)."""

    _REAL_WE = [
        _job("Asha Tech (Asha Learnology)", responsibilities=[
            "Built high-performance headless e-commerce platform using Next.js 15 & React 19 with Shopify integration",
        ]),
        _job("Asha Learnology", responsibilities=[
            "Developed comprehensive LMS platform with 10+ technical course categories",
        ]),
        _job("Adiroha", responsibilities=[
            "Built modern cybersecurity company website with Next.js 15, showcasing 20+ specialized services",
        ]),
    ]
    _REAL_PROJECTS = [
        _project("SoWhat - Shopify E-commerce Platform"),
        _project("TGES Travel & Transport Management"),
        _project("Asha Learningy - EdTech Platform"),  # note: "Learningy", real spelling variance from WE's "Learnology"
        _project("Adiroha - Cybersecurity Website"),
    ]

    def test_adiroha_is_caught(self):
        flags = detect_project_content_in_work_experience(self._REAL_WE, self._REAL_PROJECTS)
        companies_flagged = {f["work_experience_company"] for f in flags}
        assert "Adiroha" in companies_flagged

    def test_asha_learnology_spelling_variance_is_a_real_false_negative(self):
        # Real, measured: "Asha Learnology" (WE) vs "Asha Learningy -
        # EdTech Platform" (Project) share only the token "asha"
        # (1/2 = 50%, below the detector's own 60% threshold) --
        # despite the underlying bullet content being a near-exact
        # duplicate (confirmed by direct field-content comparison,
        # not by this fuzzy-name check).
        assert _fuzzy_contains("Asha Learnology", "Asha Learningy - EdTech Platform") is False
        flags = detect_project_content_in_work_experience(self._REAL_WE, self._REAL_PROJECTS)
        companies_flagged = {f["work_experience_company"] for f in flags}
        assert "Asha Learnology" not in companies_flagged  # documents the real miss

    def test_hybrid_company_name_contamination_is_structurally_undetectable(self):
        # "Asha Tech (Asha Learnology)" is the entry whose COMPANY name
        # is real (matches GT's genuine employer "Asha Tech") but whose
        # RESPONSIBILITIES are contaminated with a THIRD, unrelated
        # project's content (SoWhat). No company-name-vs-project-name
        # fuzzy match can ever catch this by design -- the company
        # field itself gives no signal.
        assert _fuzzy_contains("Asha Tech (Asha Learnology)", "SoWhat - Shopify E-commerce Platform") is False
        flags = detect_project_content_in_work_experience(self._REAL_WE, self._REAL_PROJECTS)
        companies_flagged = {f["work_experience_company"] for f in flags}
        assert "Asha Tech (Asha Learnology)" not in companies_flagged  # documents the real, structural blind spot

    def test_real_recall_on_this_one_resume_is_one_of_three(self):
        flags = detect_project_content_in_work_experience(self._REAL_WE, self._REAL_PROJECTS)
        # Exactly 1 of the 3 real, content-confirmed contaminated
        # entries is caught -- measured, not estimated.
        assert len(flags) == 1


class TestExactDuplication:
    """#1 (adversarial framing) -- SYNTHETIC. Exact company/name match
    is caught."""

    def test_exact_name_match_is_flagged(self):
        we = [_job("Acme Freelance Project")]
        projects = [_project("Acme Freelance Project")]
        flags = detect_project_content_in_work_experience(we, projects)
        assert len(flags) == 1


class TestProjectTitleMentionedLegitimatelyInWorkExperience:
    """#2 -- SYNTHETIC. A real employer whose own responsibilities
    bullet happens to MENTION a project by name (e.g. "led the Phoenix
    initiative") must not be flagged merely for that textual mention --
    the detector only compares `company` to project `name`, never
    `responsibilities` text, so this is safe by construction. Included
    to explicitly document that boundary."""

    def test_mention_inside_responsibilities_does_not_trigger_company_field_match(self):
        we = [_job("Real Employer Inc.", responsibilities=["Led the Phoenix Initiative rollout across 3 regions"])]
        projects = [_project("Phoenix Initiative")]
        flags = detect_project_content_in_work_experience(we, projects)
        assert flags == []


class TestProjectTechnologyMentionedInWorkExperience:
    """#3 -- SYNTHETIC. Shared technology names between a real job and
    an unrelated project must not trigger a flag -- the detector never
    inspects `technologies` at all."""

    def test_shared_technology_does_not_trigger(self):
        we = [_job("Real Employer Inc.", technologies=["React", "Node.js"])]
        projects = [_project("Unrelated Side Project", technologies=["React", "Node.js"])]
        flags = detect_project_content_in_work_experience(we, projects)
        assert flags == []


class TestSharedEmployerClientTerminology:
    """#4 -- SYNTHETIC. A generic, industry-common term appearing in
    both a real employer name and an unrelated project name (e.g. both
    contain "Tech" or "Solutions") is a real false-positive risk this
    milestone must characterize, not assume away."""

    def test_generic_shared_word_can_false_positive(self):
        we = [_job("Bright Solutions Pvt Ltd")]
        projects = [_project("Home Automation Solutions")]
        flags = detect_project_content_in_work_experience(we, projects)
        # Documents the REAL behavior (whichever it is) rather than
        # assuming safety -- this is exactly the kind of case an
        # automatic-suppression action could not safely act on without
        # a materially stricter match rule than the one that exists.
        # token overlap: {"solutions"} out of the smaller set {"home","automation","solutions"} (3 tokens)
        # 1/3 = 0.33, below 0.6 -- NOT flagged by the current detector.
        assert flags == []


class TestIdenticalBulletText:
    """#5 -- SYNTHETIC (name-level framing; see TestRealResume001EvidencePopulation
    for the real, field-content-verified case). Two entries with
    identical company/name AND identical bullet text -- the clearest
    possible signal, if an action were ever built, this is the
    narrowest case it could safely target."""

    def test_identical_name_and_identical_bullets(self):
        bullets = ["Built a thing.", "Shipped a thing."]
        we = [_job("Widget Co", responsibilities=bullets)]
        projects = [_project("Widget Co", highlights=bullets)]
        flags = detect_project_content_in_work_experience(we, projects)
        assert len(flags) == 1


class TestSimilarButNonIdenticalBulletText:
    """#6 -- SYNTHETIC. Company/name match with SIMILAR but genuinely
    DIFFERENT bullet content -- the detector (company/name only) still
    flags this identically to an exact-duplicate case, because it never
    inspects bullet content at all. This is exactly why "SUPPRESS on
    flag" (Phase D option B) is unsafe without an additional,
    field-level content check this milestone did not build."""

    def test_flag_fires_identically_regardless_of_bullet_similarity(self):
        we = [_job("Widget Co", responsibilities=["Led a completely different, unrelated initiative."])]
        projects = [_project("Widget Co", highlights=["Built the original widget prototype."])]
        flags = detect_project_content_in_work_experience(we, projects)
        assert len(flags) == 1  # same flag as the identical-bullets case -- the detector cannot tell these apart


class TestLegitimateWorkExperienceProjectClientWork:
    """#7 -- SYNTHETIC. A genuinely real, paid client engagement (Work
    Experience) whose company name legitimately resembles an
    unrelated personal project's name must not be conflated -- the
    detector currently has no way to distinguish "this project name
    happens to resemble a real employer" from "this employer entry IS
    the project, misrouted"."""

    def test_coincidental_name_resemblance_between_real_job_and_unrelated_project(self):
        we = [_job("Nova Consulting LLP", responsibilities=["Delivered a client audit engagement."])]
        projects = [_project("Nova - Personal Budgeting App")]
        flags = detect_project_content_in_work_experience(we, projects)
        # token overlap: {"nova"} out of the smaller set {"nova","personal","budgeting","app"} = 1/4 = 0.25 -- not flagged.
        assert flags == []


class TestProjectSectionAbsent:
    """#8 -- real T3-established behavior, re-confirmed here. No
    Projects result at all -- the detector never infers, returns no
    flags unconditionally."""

    def test_none_projects_yields_no_flags(self):
        we = [_job("Whatever Inc")]
        assert detect_project_content_in_work_experience(we, None) == []

    def test_empty_projects_list_yields_no_flags(self):
        we = [_job("Whatever Inc")]
        assert detect_project_content_in_work_experience(we, []) == []


class TestWorkExperienceSectionAbsent:
    """#9 -- SYNTHETIC. No Work Experience entries at all -- nothing to
    flag, regardless of how many real projects exist."""

    def test_empty_work_experience_yields_no_flags(self):
        projects = [_project("Something"), _project("Something Else")]
        assert detect_project_content_in_work_experience([], projects) == []


class TestMultipleProjectsAndMultipleWorkExperienceEntries:
    """#10 -- SYNTHETIC. A larger, mixed population: some genuinely
    distinct pairs, one genuine overlap -- confirms the detector
    checks every (job, project) pair independently and does not stop
    at the first match or miss a later one."""

    def test_only_the_genuinely_overlapping_pair_is_flagged(self):
        we = [
            _job("Genuine Employer One"),
            _job("Genuine Employer Two"),
            _job("Overlap Corp"),
        ]
        projects = [
            _project("Totally Unrelated Project A"),
            _project("Overlap Corp"),
            _project("Totally Unrelated Project B"),
        ]
        flags = detect_project_content_in_work_experience(we, projects)
        assert len(flags) == 1
        assert flags[0]["work_experience_company"] == "Overlap Corp"
