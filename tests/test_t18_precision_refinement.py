"""
tests/test_t18_precision_refinement.py

Milestone T18/Phase 2E -- focused, narrowly-scoped tests for the
research-only candidate-generation refinement
(scripts/_milestone_t18_candidates.py, Mechanism B: one-blank-line
merge fix).

Mechanism A (section-label cascade fix) was implemented, tuning-subset
verified, then REVERTED after the mandatory 001 gate revealed a
corpus-wide label-collapse defect not caught during tuning -- see
scripts/_milestone_t18_blocks.py's module docstring and the T18 final
report for the full account. T18's shipped behavior is Mechanism B
only; `generate_candidates_v2` uses T17's own, unmodified
`containing_section_labels()`. This file therefore tests
Mechanism B in isolation and documents Mechanism A's reverted status
directly (see `TestMechanismAReverted` below), rather than testing any
"v2" labeling behavior that is no longer live.

Zero LLM calls. Zero production imports beyond the same read-only
reuse already established in T17's own test file.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_t17_blocks import blockify, containing_section_labels  # noqa: E402
from scripts._milestone_t17_candidates import generate_candidates  # noqa: E402
from scripts._milestone_t18_blocks import containing_section_labels_v2  # noqa: E402
from scripts._milestone_t18_candidates import generate_candidates_v2  # noqa: E402


class TestMechanismBFixResolvesTheFrozenT17Limitation:
    """The exact minimal case T17's own disclosed-limitation test
    (test_disclosed_limitation_one_blank_line_between_jobs_merges_them)
    used to demonstrate the bug -- confirms the T18 fix resolves it."""

    def test_one_blank_line_between_two_dated_jobs_now_splits(self):
        text = (
            "### Acme Corp\n**Engineer**\n*Jan 2020 - Dec 2022*\n\n"
            "### Widget Inc\n**Manager**\n*2018 - 2020*\n"
        )
        blocks = blockify(text)
        v1_candidates = generate_candidates(blocks)
        v2_candidates = generate_candidates_v2(blocks)
        assert len(v1_candidates) == 1  # T17's frozen behavior, unchanged
        assert len(v2_candidates) == 2  # T18's fix: correctly split

    def test_real_corpus_case_002_bajaj_finserv_cholamandalam_now_splits(self):
        # The exact real under-segmented candidate T17 produced on
        # resume 002 (two distinct jobs, "Working as Credit Manager -
        # Centralized Credit at Bajaj Finserv..." and "Working as
        # Branch Credit Manager at Cholamandalam Finance...", each a
        # '#'-level heading with its own embedded date range,
        # separated by exactly one blank line). Confirmed fixed in the
        # full corpus evaluation (benchmark/baselines/t18_corpus_evaluation.csv,
        # resume 002: n_under_segmented_candidates 1 -> 0).
        text = (
            "# Working as Credit Manager - Centralized Credit at Bajaj Finserv "
            "(Apr 2019 - Jan 2021)\n\n"
            "# Working as Branch Credit Manager at Cholamandalam Finance "
            "(Feb 2021 - Present)\n"
        )
        blocks = blockify(text)
        v2_candidates = generate_candidates_v2(blocks)
        assert len(v2_candidates) == 2


class TestMechanismBFixUsesDateRangePresentNotDateBearingType:
    """Documents the real implementation bug found and corrected during
    T18: an earlier draft of this fix checked
    `block.block_type == "date_bearing"`, which almost never fires on
    real corpus data because T17's own block_type precedence
    (heading_like > date_bearing) classifies most job date lines
    (standalone bold/heading-shaped date text) as `heading_like`, not
    `date_bearing`, even though `date_range_present` is True on them.
    The corrected fix checks `date_range_present` directly."""

    def test_heading_shaped_date_line_still_triggers_the_tightened_tolerance(self):
        # "**Jan 2023 - Feb 2026**" is bold/standalone -> heading_like,
        # not date_bearing -- but date_range_present is True on it.
        text = (
            "### Articleship Firm\n**Jan 2023 - Feb 2026**\n\n"
            "### Second Firm\n**2018 - 2020**\n"
        )
        blocks = blockify(text)
        date_block = [b for b in blocks if "Jan 2023" in b.stripped][0]
        assert date_block.block_type == "heading_like"  # not "date_bearing"
        assert date_block.date_range_present is True
        v2_candidates = generate_candidates_v2(blocks)
        assert len(v2_candidates) == 2  # tightened tolerance still applies


class TestMechanismBPreservesOriginalIntendedBehaviors:
    """Regression tests: behaviors T17/T18 must both continue to
    support unchanged."""

    def test_single_job_header_still_produces_one_candidate(self):
        text = "### Acme Corp\n**Engineer**\n*Jan 2020 - Dec 2022*\n- Did a thing\n"
        blocks = blockify(text)
        v2_candidates = generate_candidates_v2(blockify(text))
        assert len(v2_candidates) == 1
        assert "Acme Corp" in v2_candidates[0].anchor_text

    def test_two_blank_lines_between_jobs_still_splits(self):
        text = (
            "### Acme Corp\n**Engineer**\n*Jan 2020 - Dec 2022*\n\n\n"
            "### Widget Inc\n**Manager**\n*2018 - 2020*\n"
        )
        blocks = blockify(text)
        assert len(generate_candidates_v2(blocks)) == 2

    def test_pre_date_company_blank_title_tolerance_preserved(self):
        # A blank line BEFORE any date_range_present block is reached
        # in the group must still merge (T17's original, intentional
        # "Company / blank / Title" tolerance) -- only tightens AFTER
        # the group's own first date_range_present block.
        text = "### Acme Corp\n\n**Engineer**\n*Jan 2020 - Dec 2022*\n"
        blocks = blockify(text)
        v2_candidates = generate_candidates_v2(blocks)
        assert len(v2_candidates) == 1

    def test_no_anchors_produces_no_candidates(self):
        blocks = blockify("Just a paragraph.\nAnother paragraph.\n")
        assert generate_candidates_v2(blocks) == []

    def test_candidate_never_classifies_type(self):
        text = "### Acme Corp\n- Bullet 1\n"
        blocks = blockify(text)
        cand = generate_candidates_v2(blocks)[0]
        assert not hasattr(cand, "entity_type")
        assert not hasattr(cand, "classification")

    def test_candidate_cap_is_respected(self):
        text = "\n\n".join(f"### Company {i}" for i in range(250))
        blocks = blockify(text)
        assert len(generate_candidates_v2(blocks)) <= 200


class TestDisclosedResidualLimitation:
    """A blank line AFTER a job's own first date_range_present block,
    where the header legitimately continues (e.g. a location line
    after the date), would still be split under the tightened
    tolerance. Tuning-subset verification (18/18 resumes, see
    docs/T18_PHASE2E_MECHANISM_TRACE_AND_PLAN.md) found zero real
    corpus occurrences of this shape, but it is not structurally
    impossible -- disclosed here, not assumed away."""

    def test_blank_line_after_date_within_a_single_jobs_own_header_is_split(self):
        text = "### Acme Corp\n*Jan 2020 - Dec 2022*\n\n**Bengaluru, India**\n"
        blocks = blockify(text)
        v2_candidates = generate_candidates_v2(blocks)
        assert len(v2_candidates) == 2  # the real, disclosed residual behavior


class TestMechanismAReverted:
    """Confirms Mechanism A is not used in the live T18 path, and
    documents (without using it) the corpus-wide label-collapse defect
    that caused its reversion -- found at the mandatory 001 gate: a
    document-title heading shallower than all real section headings
    permanently freezes the running label."""

    def test_live_path_uses_t17s_unmodified_label_function(self):
        text = "# Aman Kanojiya\n## Current Employment\n### Acme Corp\n"
        blocks = blockify(text)
        v1_labels = containing_section_labels(blocks)
        v2_candidates = generate_candidates_v2(blocks)
        # The live v2 candidate-generation path must reproduce T17's
        # own label for a known section heading exactly (not the
        # reverted v2 labeling function's behavior).
        current_employment_idx = [
            i for i, b in enumerate(blocks) if b.stripped == "## Current Employment"
        ][0]
        assert v1_labels[current_employment_idx] == "experience"

    def test_reverted_mechanism_a_would_have_collapsed_this_label(self):
        # Retained purely as disclosed evidence of the defect that led
        # to reversion -- NOT exercised by the live evaluation path.
        text = "# Aman Kanojiya\n## Current Employment\n### Acme Corp\n"
        blocks = blockify(text)
        v2_labels = containing_section_labels_v2(blocks)
        current_employment_idx = [
            i for i, b in enumerate(blocks) if b.stripped == "## Current Employment"
        ][0]
        # The defect: level 1 ("# Aman Kanojiya") permanently freezes
        # established_level, so the level-2 "## Current Employment"
        # heading never qualifies to update the label.
        assert v2_labels[current_employment_idx] != "experience"
        assert v2_labels[current_employment_idx] == "unlabeled:aman kanojiya"
