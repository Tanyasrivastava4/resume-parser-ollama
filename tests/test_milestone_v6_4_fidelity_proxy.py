"""
tests/test_milestone_v6_4_fidelity_proxy.py

Targeted, deterministic tests for
scripts/_milestone_v6_4_fidelity_proxy.py (Milestone V6.4). All cases
are synthetic/mechanism-focused, not overfit to specific resume IDs --
real historical-case validation is reported separately in
benchmark/baselines/v6_4_validation_results.csv (see the milestone
report), not asserted here as pytest expectations.

No LLM calls.
"""
from scripts._milestone_v6_4_fidelity_proxy import (
    compute_fidelity_proxy,
    item_is_grounded,
)


class TestClearlySupportedContent:
    def test_exact_word_present(self):
        r = item_is_grounded("Python", "Python, SQL, Excel")
        assert r["grounded"] is True

    def test_case_insensitive(self):
        r = item_is_grounded("python", "Python, SQL, Excel")
        assert r["grounded"] is True

    def test_multiword_item_present(self):
        r = item_is_grounded("Power BI", "Power BI, Tableau, Advanced Excel")
        assert r["grounded"] is True


class TestClearlyUnsupportedInventedContent:
    def test_completely_absent_item(self):
        r = item_is_grounded("Kubernetes", "Python, SQL, Excel, Power BI")
        assert r["grounded"] is False

    def test_plausible_but_absent_short_token_not_coincidentally_matched(self):
        # "R" is a real, plausible skill name, but genuinely absent here.
        # Must not falsely match via coincidental substring containment.
        r = item_is_grounded("R", "VS Code, Jupyter Notebook, PyCharm, Spyder, Streamlit Cloud")
        assert r["grounded"] is False


class TestNormalizationOnlyDifferences:
    def test_whitespace_variation(self):
        r = item_is_grounded("Power   BI", "Power BI, Tableau")
        assert r["grounded"] is True

    def test_leading_trailing_whitespace(self):
        r = item_is_grounded("  Excel  ", "Advanced Excel, Word")
        assert r["grounded"] is True


class TestCategoryLabelDifferences:
    def test_bare_item_found_within_labeled_source_line(self):
        # Real V5.6-established pattern: source has a category-prefixed
        # line, and a bare extracted item should still be grounded.
        r = item_is_grounded("Windows 11", "Operating Systems: Windows 11, Windows Server 2012, 2016, 2019 and Linux")
        assert r["grounded"] is True


class TestParentheticalAndPunctuationCases:
    def test_punctuation_leading_token(self):
        r = item_is_grounded("C++", "Skills: C++, Python, Java")
        assert r["grounded"] is True

    def test_punctuation_token_not_confused_with_shorter_relative(self):
        # C++ must not be considered "grounded" merely because bare "C"
        # (a different, real language) is present.
        r = item_is_grounded("C++", "Skills: C, Python, Java")
        assert r["grounded"] is False

    def test_dotted_compound_token(self):
        r = item_is_grounded("Node.js", "Node.js, Express.js, MongoDB")
        assert r["grounded"] is True

    def test_dotted_compound_token_absent(self):
        # Only bare "Node" present, not "Node.js" -- must not match.
        r = item_is_grounded("Node.js", "Node, Express.js, MongoDB")
        assert r["grounded"] is False

    def test_item_embedded_inside_parenthetical_group_in_source(self):
        r = item_is_grounded("Excel", "Microsoft Office (Excel, Word, PowerPoint)")
        assert r["grounded"] is True


class TestRepeatedLabelPrefixPerSplitItem:
    """Real, found-during-validation case (resume 018, V6.1 baseline
    runs 2/5): the model repeats a category label on each item split
    out of one labeled source line (source: "MS Office Suite: Advanced
    Excel, Word, PowerPoint" -> candidate "MS Office Suite: Word"). The
    compound string never appears verbatim (other items sit between
    the label and any one split item), so this must be a SEPARATE,
    clearly-labeled secondary match -- never silently merged into the
    primary direct-match category."""

    def test_label_repeated_per_item_is_grounded_via_secondary_path(self):
        body = "MS Office Suite: Advanced Excel, Word, PowerPoint"
        r = item_is_grounded("MS Office Suite: Word", body)
        assert r["grounded"] is True
        assert r["reason"] == "grounded_after_label_strip"

    def test_direct_match_still_preferred_and_labeled_as_such(self):
        r = item_is_grounded("Python", "Python, SQL")
        assert r["reason"] == "grounded_direct"

    def test_fake_label_does_not_create_a_hallucination_loophole(self):
        # Adding a colon-prefixed fake label must NOT rescue a genuinely
        # absent item -- the remainder must still independently pass.
        body = "MS Office Suite: Advanced Excel, Word, PowerPoint"
        r = item_is_grounded("Fake Category: Kubernetes", body)
        assert r["grounded"] is False

    def test_aggregate_tracks_label_strip_matches_separately(self):
        body = "MS Office Suite: Advanced Excel, Word, PowerPoint"
        result = compute_fidelity_proxy(["MS Office Suite: Word", "Python"], body, boundary_valid=True)
        assert result["n_grounded_after_label_strip"] == 1
        assert result["n_grounded_direct"] == 0  # "Python" isn't in this body at all
        assert result["n_grounded"] == 1


class TestSegmentationAndCompoundItemBehavior:
    def test_compound_source_phrase_supports_full_matching_candidate_item(self):
        r = item_is_grounded(
            "Financial Statement Review under Ind AS, IFRS, and Indian GAAP",
            "Financial Statement Review under Ind AS, IFRS, and Indian GAAP",
        )
        assert r["grounded"] is True

    def test_merged_source_phrase_does_not_grant_bare_item_a_free_pass_unless_present(self):
        # A source phrase like "Hands on finance applications i.e. SAP,
        # Tally, ..." DOES still contain the bare word "SAP" -- this is
        # intentional and correct: grounding checks textual presence,
        # not whether the model produced atomic vs. merged segmentation
        # (that is a SEPARATE, already-documented mechanism -- V5.6/V6.1
        # resume 015 -- not something this proxy is designed to flag).
        r = item_is_grounded("SAP", "Hands on finance applications i.e. SAP, Tally, Quick Books, ZOHO etc.")
        assert r["grounded"] is True


class TestAdversarialLexicalCoincidence:
    """The milestone's own required adversarial question (Section 9):
    could a plausible, semantically-related-but-absent item be
    incorrectly credited? Tests the specific, REAL failure mode found
    and fixed during this milestone's own design work -- a real
    technology name that is a strict substring of a different, related
    real technology name actually present in the source."""

    def test_java_not_confused_with_javascript(self):
        r = item_is_grounded("Java", "HTML, CSS, JavaScript, React.js, Vite.js, Redux")
        assert r["grounded"] is False

    def test_go_not_confused_with_golang(self):
        r = item_is_grounded("Go", "Golang, Python, Java")
        assert r["grounded"] is False

    def test_go_matches_when_genuinely_standalone(self):
        r = item_is_grounded("Go", "Go, Python, Java")
        assert r["grounded"] is True

    def test_semantic_paraphrase_is_not_matched(self):
        # A semantically-related but textually different paraphrase
        # must NOT be credited -- this proxy is lexical, not semantic,
        # by design (see the milestone report's discussion of this
        # being a source of false negatives, not false positives).
        r = item_is_grounded("Image Editing Software", "Adobe Photoshop, Adobe Illustrator, Figma, Sketch")
        assert r["grounded"] is False


class TestAggregateProxyStructuralCases:
    def test_boundary_invalid_short_circuits(self):
        result = compute_fidelity_proxy(["Python"], "irrelevant body", boundary_valid=False)
        assert result["proxy_status"] == "boundary_invalid"
        assert result["grounding_fraction"] is None

    def test_malformed_output_not_a_list(self):
        result = compute_fidelity_proxy("not a list", "Python, SQL", boundary_valid=True)
        assert result["proxy_status"] == "malformed"

    def test_malformed_output_non_string_items(self):
        result = compute_fidelity_proxy([{"skill": "Python"}], "Python, SQL", boundary_valid=True)
        assert result["proxy_status"] == "malformed"

    def test_empty_list_is_not_zero_fidelity(self):
        result = compute_fidelity_proxy([], "Python, SQL", boundary_valid=True)
        assert result["proxy_status"] == "empty"
        assert result["grounding_fraction"] is None  # undefined, not 0

    def test_fully_grounded_case(self):
        result = compute_fidelity_proxy(["Python", "SQL"], "Python, SQL, Excel", boundary_valid=True)
        assert result["proxy_status"] == "evaluated"
        assert result["grounding_fraction"] == 1.0
        assert result["ungrounded_items"] == []

    def test_partially_grounded_case_lists_the_ungrounded_item_explicitly(self):
        result = compute_fidelity_proxy(["Python", "Kubernetes"], "Python, SQL, Excel", boundary_valid=True)
        assert result["proxy_status"] == "evaluated"
        assert result["grounding_fraction"] == 0.5
        assert result["ungrounded_items"] == ["Kubernetes"]

    def test_per_item_evidence_is_explainable(self):
        # A reviewer must be able to see WHY, not just a bare number.
        result = compute_fidelity_proxy(["Python", "Kubernetes"], "Python, SQL", boundary_valid=True)
        assert len(result["per_item_evidence"]) == 2
        assert all("reason" in e for e in result["per_item_evidence"])
