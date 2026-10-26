"""
tests/test_t20_section_heading_refinement.py

Milestone T20/Phase 2G -- focused tests for the one narrow refinement
in scripts/_milestone_t20_candidates.py (generate_candidates_t20):
a recognized Work-Experience section-title heading never extends its
own anchor group past itself.

Zero LLM calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_t17_blocks import blockify  # noqa: E402
from scripts._milestone_t18_candidates import generate_candidates_v2 as generate_baseline  # noqa: E402
from scripts._milestone_t20_candidates import generate_candidates_t20 as generate_candidate  # noqa: E402


class TestSectionHeadingSingletonFix:
    def test_zero_blank_line_section_heading_no_longer_merges_with_first_job(self):
        text = "# Work Experience\n### Acme Corp\n**Engineer**\n*Jan 2020 - Dec 2022*\n"
        blocks = blockify(text)
        base = generate_baseline(blocks)
        cand = generate_candidate(blocks)
        assert len(base) == 1  # frozen T18 behavior: still merged
        assert base[0].n_blocks_in_extent == 4
        assert len(cand) == 2  # T20 fix: section heading isolated
        assert cand[0].anchor_text == "# Work Experience"
        assert cand[0].n_blocks_in_extent == 1
        assert cand[1].anchor_text == "### Acme Corp"
        assert "Acme Corp" in cand[1].anchor_text

    def test_one_blank_line_section_heading_no_longer_merges_with_first_job(self):
        text = "## Work Experience\n\n### Acme Corp\n**Engineer**\n*Jan 2020 - Dec 2022*\n"
        blocks = blockify(text)
        cand = generate_candidate(blocks)
        assert len(cand) == 2
        assert cand[0].n_blocks_in_extent == 1

    def test_isolated_section_heading_carries_no_safety_flags(self):
        text = "## Work Experience\n### Acme Corp\n**Engineer**\n*Jan 2020 - Dec 2022*\n"
        blocks = blockify(text)
        cand = generate_candidate(blocks)
        assert not any(cand[0].flags.values())

    def test_second_and_later_jobs_are_unaffected(self):
        text = (
            "## Work Experience\n### Acme Corp\n**Engineer**\n*Jan 2020 - Dec 2022*\n\n"
            "### Widget Inc\n**Manager**\n*2018 - 2020*\n"
        )
        blocks = blockify(text)
        base = generate_baseline(blocks)
        cand = generate_candidate(blocks)
        # baseline: [merged section+job1, job2] = 2 candidates
        # candidate: [section, job1, job2] = 3 candidates
        assert len(base) == 2
        assert len(cand) == 3
        assert cand[2].anchor_text == "### Widget Inc"
        # Block indices are absolute (blocks themselves don't move); the
        # extra singleton candidate changes candidate *count/ordering*,
        # not the underlying block index Widget Inc's own extent starts at.
        assert cand[2].extent_start == base[1].extent_start

    def test_no_recall_loss_the_jobs_own_text_is_still_present_somewhere(self):
        text = "## Work Experience\n### Acme Corp\n**Engineer**\n*Jan 2020 - Dec 2022*\n"
        blocks = blockify(text)
        cand = generate_candidate(blocks)
        all_extent_text = " ".join(
            blocks[i].stripped for c in cand for i in range(c.extent_start, c.extent_end + 1)
        )
        assert "Acme Corp" in all_extent_text


class TestManifestationANotFixed:
    """Documents, explicitly, that the T20 refinement does NOT address
    the second, distinct manifestation found in T19's U1 (a section
    heading swallowed as a non-first element of an EARLIER, unrelated
    anchor group's own chain -- e.g. a preceding table_row). This is
    the "second mechanism discovered, not implemented" case recorded
    in the T20 report; this test locks in the current, honest,
    unfixed behavior rather than hiding it."""

    def test_section_heading_still_swallowed_by_a_preceding_unrelated_anchor(self):
        text = (
            "| **Skill A** | **Skill B** |\n"
            "## Work Experience\n"
            "### Acme Corp\n**Engineer**\n*Jan 2020 - Dec 2022*\n"
        )
        blocks = blockify(text)
        cand = generate_candidate(blocks)
        # The table row's own anchor group still absorbs "## Work
        # Experience" AND "### Acme Corp" -- the T20 singleton guard
        # only fires when a section-title heading is about to become
        # its OWN fresh group start, not when it is offered as an
        # extension target to an already-open, unrelated group.
        merged = cand[0]
        assert "Work Experience" in merged.anchor_text or merged.n_blocks_in_extent > 1
        extent_text = " ".join(
            blocks[i].stripped for i in range(merged.extent_start, merged.extent_end + 1)
        )
        assert "Work Experience" in extent_text and "Acme Corp" in extent_text


class TestRegressionAgainstT18Baseline:
    """Confirms the T20 module reuses T18's own Mechanism B unchanged
    for every anchor group that is NOT itself a section-title
    heading."""

    def test_two_blank_lines_between_jobs_still_splits(self):
        text = (
            "### Acme Corp\n**Engineer**\n*Jan 2020 - Dec 2022*\n\n\n"
            "### Widget Inc\n**Manager**\n*2018 - 2020*\n"
        )
        blocks = blockify(text)
        assert len(generate_candidate(blocks)) == 2

    def test_one_blank_line_between_two_dated_jobs_still_splits(self):
        text = (
            "### Acme Corp\n**Engineer**\n*Jan 2020 - Dec 2022*\n\n"
            "### Widget Inc\n**Manager**\n*2018 - 2020*\n"
        )
        blocks = blockify(text)
        assert len(generate_candidate(blocks)) == 2

    def test_no_anchors_produces_no_candidates(self):
        blocks = blockify("Just a paragraph.\nAnother paragraph.\n")
        assert generate_candidate(blocks) == []

    def test_candidate_never_classifies_type(self):
        text = "### Acme Corp\n- Bullet 1\n"
        blocks = blockify(text)
        cand = generate_candidate(blocks)[0]
        assert not hasattr(cand, "entity_type")
        assert not hasattr(cand, "classification")
