"""
tests/test_t17_deterministic_candidates.py

Milestone T17/Phase 2D -- focused, narrowly-scoped tests for the
research-only deterministic block/candidate-generation code
(scripts/_milestone_t17_blocks.py, scripts/_milestone_t17_candidates.py).
Zero LLM calls. Zero production imports beyond read-only reuse of
WorkExperienceLocator's own regex constants and
validators/misrouting_detector's _fuzzy_contains, neither of which is
modified.
"""
from __future__ import annotations

import pytest

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_t17_blocks import blockify, containing_section_labels  # noqa: E402
from scripts._milestone_t17_candidates import generate_candidates  # noqa: E402


class TestBlocking:
    def test_blank_lines_are_not_their_own_blocks(self):
        text = "# Heading\n\n\nSome text\n"
        blocks = blockify(text)
        assert len(blocks) == 2
        assert blocks[1].blank_lines_before == 2

    def test_page_fence_increments_page_number(self):
        text = "```markdown\n# Page 1\n```\n\n```markdown\n# Page 2\n```\n"
        blocks = blockify(text)
        pages = [b.page_number for b in blocks]
        assert pages[0] == 1  # opening fence itself is page 1
        assert pages[-1] == 2  # second page's content

    def test_heading_classified_correctly(self):
        blocks = blockify("## Experience\n")
        assert blocks[0].block_type == "heading_like"
        assert blocks[0].heading_likeness is True

    def test_date_range_detected(self):
        blocks = blockify("Jan 2020 - Dec 2022\n")
        assert blocks[0].date_range_present is True

    def test_bullet_detected(self):
        blocks = blockify("- Did a thing\n")
        assert blocks[0].block_type == "bullet"

    def test_table_row_detected(self):
        blocks = blockify("| Company | Role |\n")
        assert blocks[0].block_type == "table_row"

    def test_bullet_run_after_lookahead(self):
        blocks = blockify("### Company\n- Bullet one\n")
        assert blocks[0].bullet_run_after is True

    def test_section_distance_resets_at_heading(self):
        # A long, clearly-non-heading-shaped paragraph line (>5 words,
        # ends in a period) so _looks_like_heading correctly returns
        # False and does not itself reset the distance counter.
        blocks = blockify(
            "## Experience\n"
            "This is a genuinely long paragraph sentence with many words.\n"
            "This is another genuinely long paragraph sentence too.\n"
        )
        assert blocks[0].section_distance == 0
        assert blocks[1].section_distance == 1
        assert blocks[2].section_distance == 2

    def test_repeated_pattern_id_groups_identical_feature_tuples(self):
        blocks = blockify("- Bullet one\n- Bullet two\n")
        assert blocks[0].repeated_pattern_id == blocks[1].repeated_pattern_id


class TestSectionLabeling:
    def test_experience_heading_labeled_experience(self):
        blocks = blockify(
            "## Experience\nThis is a genuinely long paragraph sentence with many words.\n"
        )
        labels = containing_section_labels(blocks)
        assert labels[0] == "experience"
        assert labels[1] == "experience"

    def test_disclosed_limitation_a_second_job_heading_overwrites_the_running_label(self):
        # Documents the real, disclosed T17 limitation found during
        # corpus evaluation: a second job-shaped heading within the
        # same real section resets the tracked "most recent heading"
        # to itself, so the running section label is no longer
        # "experience" for content after it -- even though it is still
        # genuinely inside the Experience section. Not fixed in T17
        # (rules are frozen); this test documents the behavior exactly
        # as it exists.
        blocks = blockify("## Experience\n### Acme Corp\nSome content under Acme.\n")
        labels = containing_section_labels(blocks)
        assert labels[0] == "experience"  # the section heading itself
        assert labels[1] != "experience"  # the job heading overwrites it
        assert labels[2] != "experience"  # so does content after it

    def test_non_experience_keyword_labeled_accordingly(self):
        blocks = blockify("## Projects\nSome text\n")
        labels = containing_section_labels(blocks)
        assert labels[0].startswith("non_experience:")

    def test_unrecognized_heading_labeled_unlabeled(self):
        blocks = blockify("## Random Custom Section\nSome text\n")
        labels = containing_section_labels(blocks)
        assert labels[0].startswith("unlabeled:")


class TestCandidateGeneration:
    def test_no_anchors_produces_no_candidates(self):
        blocks = blockify("Just a paragraph.\nAnother paragraph.\n")
        candidates = generate_candidates(blocks)
        assert candidates == []

    def test_single_job_header_produces_one_candidate(self):
        text = "### Acme Corp\n**Engineer**\n*Jan 2020 - Dec 2022*\n- Did a thing\n"
        blocks = blockify(text)
        candidates = generate_candidates(blocks)
        assert len(candidates) == 1
        assert "Acme Corp" in candidates[0].anchor_text

    def test_two_blank_lines_between_jobs_keeps_them_as_separate_candidates(self):
        text = (
            "### Acme Corp\n**Engineer**\n*Jan 2020 - Dec 2022*\n\n\n"
            "### Widget Inc\n**Manager**\n*2018 - 2020*\n"
        )
        blocks = blockify(text)
        candidates = generate_candidates(blocks)
        assert len(candidates) == 2

    def test_disclosed_limitation_one_blank_line_between_jobs_merges_them(self):
        # Documents the real, disclosed T17 limitation found during
        # corpus evaluation (measured: 11 under-segmented candidates
        # across 7 of 22 real resumes): the anchor-group merge rule
        # ("blank_lines_before <= 1") tolerates exactly one blank line,
        # so two distinct jobs separated by only one blank line --
        # a common resume format -- are merged into a single
        # candidate/anchor group instead of two. Not fixed in T17
        # (rules are frozen); this test documents the behavior exactly
        # as it exists, on the exact minimal case that triggers it.
        text = (
            "### Acme Corp\n**Engineer**\n*Jan 2020 - Dec 2022*\n\n"
            "### Widget Inc\n**Manager**\n*2018 - 2020*\n"
        )
        blocks = blockify(text)
        candidates = generate_candidates(blocks)
        assert len(candidates) == 1  # the real, disclosed limitation, not the ideal outcome

    def test_extent_bounded_by_next_anchor(self):
        text = "### Acme Corp\n- Bullet 1\n- Bullet 2\n\n### Widget Inc\n- Bullet 3\n"
        blocks = blockify(text)
        candidates = generate_candidates(blocks)
        assert candidates[0].extent_end < candidates[1].extent_start

    def test_candidate_never_classifies_type(self):
        text = "### Acme Corp\n- Bullet 1\n"
        blocks = blockify(text)
        candidates = generate_candidates(blocks)
        # Candidate has no 'type'/'classification' field of any kind --
        # only structural features and safety flags.
        assert not hasattr(candidates[0], "entity_type")
        assert not hasattr(candidates[0], "classification")

    def test_low_confidence_terminus_flag_on_last_candidate(self):
        text = "### Acme Corp\n- Bullet 1\n"
        blocks = blockify(text)
        candidates = generate_candidates(blocks)
        assert candidates[-1].flags["low_confidence_terminus"] is True

    def test_candidate_cap_is_respected(self):
        # 250 distinct heading-shaped anchors, well above the frozen
        # 200-candidate cap.
        text = "\n\n".join(f"### Company {i}" for i in range(250))
        blocks = blockify(text)
        candidates = generate_candidates(blocks)
        assert len(candidates) <= 200


@pytest.mark.local_data('corpus_markdown')
class TestRealCorpusResume001Forensic:
    """Confirms, using the real, committed resume-001 markdown, that
    Project sub-headings do become candidates (per the mandatory T17
    analysis) -- not filtered out, only flagged."""

    def test_project_subheadings_become_flagged_not_suppressed_candidates(self):
        text = Path("output/markdown/AMAN_CV_BCA.md").read_text(encoding="utf-8")
        blocks = blockify(text)
        candidates = generate_candidates(blocks)
        project_candidates = [c for c in candidates if "Full-Stack" in c.anchor_text]
        assert len(project_candidates) >= 1
        for c in project_candidates:
            assert c.flags["outside_known_experience_vocabulary"] is True
