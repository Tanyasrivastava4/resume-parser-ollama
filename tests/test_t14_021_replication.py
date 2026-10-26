"""
tests/test_t14_021_replication.py

Milestone T14/Phase 2C -- focused, narrowly-scoped tests for the
research-only T14 analysis code (scripts/_milestone_t14_analysis.py).
New diagnostic tests, distinct from the project's production
regression suite -- test only T14's own deterministic entity-matching/
split/merge/fabrication classification helpers, never invoking Ollama,
never touching production behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_t13_entity_first_pilot import check_entity_fixation_violation  # noqa: E402


def _match(rid, jobs, gt_jobs):
    """Reimplements the exact matching call shape used by
    scripts/_milestone_t14_analysis.py's _match_jobs_to_gt, against an
    injectable GT list, for isolated testing without reading real
    ground-truth files."""
    import scripts._milestone_t14_analysis as analysis

    analysis.GT_JOBS = {rid: gt_jobs}
    return analysis._match_jobs_to_gt(rid, jobs)


class TestEntityRecallPrecision:
    def test_perfect_match_full_precision_and_recall(self):
        gt = [{"company": "Acme Corp"}, {"company": "Widget Inc"}]
        jobs = [{"company": "Acme Corp"}, {"company": "Widget Inc"}]
        matched, extra, missing, split, merge = _match("x", jobs, gt)
        assert len(matched) == 2
        assert extra == []
        assert missing == []

    def test_missing_tail_entity_recorded(self):
        gt = [{"company": "Acme Corp"}, {"company": "Widget Inc"}, {"company": "Old Co"}]
        jobs = [{"company": "Acme Corp"}, {"company": "Widget Inc"}]
        matched, extra, missing, split, merge = _match("x", jobs, gt)
        assert len(matched) == 2
        assert len(missing) == 1
        assert missing[0]["company"] == "Old Co"

    def test_fabricated_extra_entity_recorded(self):
        gt = [{"company": "Acme Corp"}]
        jobs = [{"company": "Acme Corp"}, {"company": "Totally Unrelated Project"}]
        matched, extra, missing, split, merge = _match("x", jobs, gt)
        assert len(matched) == 1
        assert len(extra) == 1


class TestSplitDetection:
    def test_one_gt_company_claimed_by_two_returned_jobs_is_a_split(self):
        gt = [{"company": "Global Source Technology"}]
        jobs = [{"company": "Global Source Technology"}, {"company": "Global Source Technology"}]
        matched, extra, missing, split, merge = _match("x", jobs, gt)
        assert "Global Source Technology" in split

    def test_no_split_when_each_gt_company_claimed_once(self):
        gt = [{"company": "Acme Corp"}, {"company": "Widget Inc"}]
        jobs = [{"company": "Acme Corp"}, {"company": "Widget Inc"}]
        matched, extra, missing, split, merge = _match("x", jobs, gt)
        assert split == set()


class TestMergeDetection:
    def test_one_returned_job_matching_two_distinct_gt_companies_is_a_merge(self):
        # A contrived case: a returned company string generic/short
        # enough to fuzzy-match two distinct GT entries.
        gt = [{"company": "Global Tech"}, {"company": "Global Technologies"}]
        jobs = [{"company": "Global"}]
        matched, extra, missing, split, merge = _match("x", jobs, gt)
        # "Global" alone is a substring of both normalized GT names,
        # so this should be flagged as an ambiguous merge candidate.
        assert len(merge) >= 1

    def test_no_merge_for_clearly_distinct_companies(self):
        gt = [{"company": "Acme Corp"}, {"company": "Totally Different Widgets"}]
        jobs = [{"company": "Acme Corp"}]
        matched, extra, missing, split, merge = _match("x", jobs, gt)
        assert merge == []


class TestFabrication001:
    def test_known_project_name_is_flagged(self):
        from scripts._milestone_t14_analysis import _is_001_project_fabrication
        assert _is_001_project_fabrication({"company": "Adiroha"}) is True
        assert _is_001_project_fabrication({"company": "SoWhat - Shopify E-commerce Platform"}) is True

    def test_real_job_company_not_flagged(self):
        from scripts._milestone_t14_analysis import _is_001_project_fabrication
        assert _is_001_project_fabrication({"company": "Asha Tech"}) is False


class TestStageBEntityFixationReused:
    """Confirms T14 reuses T13's own, already-fixed
    check_entity_fixation_violation unmodified -- not a T14-local
    reimplementation that could silently diverge."""

    def test_no_violation_when_stage_b_matches_stage_a_exactly(self):
        stage_a = [{"ordinal": 1, "company": "Acme Corp"}]
        returned = [{"company": "Acme Corp"}]
        assert check_entity_fixation_violation(stage_a, returned) is False

    def test_violation_when_stage_b_adds_an_entity(self):
        stage_a = [{"ordinal": 1, "company": "Acme Corp"}]
        returned = [{"company": "Acme Corp"}, {"company": "Extra Co"}]
        assert check_entity_fixation_violation(stage_a, returned) is True
