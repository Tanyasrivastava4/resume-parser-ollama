"""
tests/test_milestone_t2_phase0_failure_taxonomy.py

Milestone T2 / Phase 0 -- focused tests for the deterministic
entity-alignment and date-comparison logic used by Phase A's failure
taxonomy and Phase C's evaluation (both zero-LLM-call, pure-function
research code).
"""
from __future__ import annotations

from scripts._milestone_t2_phase0_failure_taxonomy import (
    _align_entities,
    _dates_equal,
    _fuzzy_contains,
    _normalize,
)


class TestNormalize:
    def test_lowercases_and_strips_punctuation(self):
        assert _normalize("Asha Tech, Ltd.") == "asha tech ltd"

    def test_handles_none(self):
        assert _normalize(None) == ""

    def test_month_abbreviation_canonicalized(self):
        assert _normalize("Jan 2023") == _normalize("January 2023")
        assert _normalize("Sept 2019") == _normalize("September 2019")


class TestFuzzyContains:
    def test_exact_match(self):
        assert _fuzzy_contains("Asha Tech", "Asha Tech")

    def test_substring_match(self):
        assert _fuzzy_contains("Asha Tech", "Asha Tech (Asha Learningogy)")

    def test_token_overlap_match(self):
        assert _fuzzy_contains("KMI Business Technologies Pvt. Ltd.", "KMI Business Technologies")

    def test_unrelated_names_do_not_match(self):
        assert not _fuzzy_contains("Hazel Infotech Ltd.", "SVN Technologies")

    def test_empty_strings_do_not_match(self):
        assert not _fuzzy_contains("", "Asha Tech")
        assert not _fuzzy_contains("Asha Tech", "")


class TestDatesEqual:
    def test_same_date_different_abbreviation(self):
        assert _dates_equal("Jan 2023", "January 2023")

    def test_same_date_iso_vs_long_form(self):
        # The real resume-002 case this milestone found and fixed: ISO
        # vs. long-form English are the SAME real date.
        assert _dates_equal("2017-12-04", "December 04 2017")

    def test_different_dates_not_equal(self):
        assert not _dates_equal("Jan 2023", "Feb 2024")

    def test_none_or_empty_never_equal(self):
        assert not _dates_equal(None, "Jan 2023")
        assert not _dates_equal("Jan 2023", None)
        assert not _dates_equal("", "")

    def test_unparseable_falls_back_to_normalized_string_comparison(self):
        # Neither side parses as a real date -- falls back to
        # word-normalized string equality rather than crashing.
        assert _dates_equal("Present", "Present")
        assert not _dates_equal("Present", "Ongoing")


class TestAlignEntities:
    def test_all_matched_when_names_correspond(self):
        gt = [{"company": "Asha Tech"}, {"company": "Google"}]
        cand = [{"company": "Google Inc."}, {"company": "Asha Tech Pvt Ltd"}]
        matched, unmatched_gt, unmatched_cand = _align_entities(gt, cand, "company", "company")
        assert len(matched) == 2
        assert unmatched_gt == []
        assert unmatched_cand == []

    def test_unmatched_gt_when_candidate_missing_entity(self):
        # The real resume-021 shape: GT has 3 trailing companies the
        # candidate completely omits.
        gt = [{"company": "A"}, {"company": "B"}, {"company": "C"}]
        cand = [{"company": "A"}]
        matched, unmatched_gt, unmatched_cand = _align_entities(gt, cand, "company", "company")
        assert len(matched) == 1
        assert len(unmatched_gt) == 2

    def test_unmatched_cand_when_candidate_over_segments(self):
        # The real resume-001 shape: candidate has extra hallucinated/
        # misrouted entities beyond what GT has.
        gt = [{"company": "Asha Tech"}]
        cand = [{"company": "Asha Tech"}, {"company": "Adiroha"}, {"company": "TGES Travel"}]
        matched, unmatched_gt, unmatched_cand = _align_entities(gt, cand, "company", "company")
        assert len(matched) == 1
        assert len(unmatched_gt) == 0
        assert len(unmatched_cand) == 2

    def test_no_double_matching_same_candidate_twice(self):
        gt = [{"company": "Asha Tech"}, {"company": "Asha Tech"}]
        cand = [{"company": "Asha Tech"}]
        matched, unmatched_gt, unmatched_cand = _align_entities(gt, cand, "company", "company")
        assert len(matched) == 1
        assert len(unmatched_gt) == 1
