"""
tests/test_milestone_v5_9_signal_corrections.py

Targeted, deterministic tests for
scripts/_milestone_v5_9_signal_corrections.py (Milestone V5.9).

These tests exercise MECHANISMS via small synthetic markdown snippets,
not real resume IDs -- per this milestone's own explicit instruction
not to overfit tests to specific corpus resumes. Real-corpus behavior
(resumes 001, 006, 008, 013, 016, 017, 020, 023) is separately
confirmed by the milestone's full-corpus revalidation script and
reported in the milestone doc with real numbers, not asserted here.

No LLM calls. No production code imported (this module itself avoids
importing services/languages_extractor.py's live import chain -- see
that module's own docstring for why).
"""
import pytest

from scripts._milestone_v5_9_signal_corrections import (
    apparent_comma_item_count,
    find_skills_section,
    in_section_frac,
)


# ---------------------------------------------------------------------
# Problem A: parenthetical commas must not create false top-level items
# ---------------------------------------------------------------------

class TestParentheticalCommaTokenization:
    def test_parenthetical_comma_group_counts_as_one_item(self):
        body = "- Microsoft Office (Excel, Word, PowerPoint)"
        assert apparent_comma_item_count(body) == 1

    def test_parenthetical_group_plus_genuine_top_level_items(self):
        body = "- Microsoft Office (Excel, Word, PowerPoint), Python, SQL"
        # "Microsoft Office (Excel, Word, PowerPoint)" + "Python" + "SQL" = 3
        assert apparent_comma_item_count(body) == 3

    def test_multiple_parenthetical_groups_on_one_line(self):
        body = "- AWS (S3, EC2), Azure (Blob, Functions), GCP"
        assert apparent_comma_item_count(body) == 3

    def test_genuine_top_level_commas_still_split_normally(self):
        body = "- Python, SQL, Pandas, NumPy, Excel"
        assert apparent_comma_item_count(body) == 5

    def test_parenthetical_comma_does_not_leak_across_lines(self):
        body = "- Tool A (X, Y)\n- Tool B, Tool C"
        # line 1: "Tool A (X, Y)" = 1 item; line 2: "Tool B" + "Tool C" = 2
        assert apparent_comma_item_count(body) == 3


# ---------------------------------------------------------------------
# Structural lines (headings, dividers) must not be counted as items
# ---------------------------------------------------------------------

class TestStructuralLineExclusion:
    def test_heading_line_is_not_counted_as_an_item(self):
        body = "### Frontend\n- HTML\n- CSS"
        assert apparent_comma_item_count(body) == 2

    def test_divider_line_is_not_counted_as_an_item(self):
        body = "- Python\n- SQL\n\n---"
        assert apparent_comma_item_count(body) == 2

    def test_empty_body_counts_zero(self):
        assert apparent_comma_item_count("") == 0
        assert apparent_comma_item_count("   \n  \n") == 0


# ---------------------------------------------------------------------
# Problem B: nested non-Skills "Languages" content must not inflate
# the count, but a nested "Languages" heading containing PROGRAMMING
# languages is legitimate Skills content and must be preserved.
# ---------------------------------------------------------------------

class TestNestedLanguagesDisambiguation:
    def test_human_language_subsection_is_excluded(self):
        md = (
            "Skills\n"
            "### Software Proficiency\n"
            "- Tally ERP\n"
            "- MS Excel\n\n"
            "### Languages\n"
            "- English\n"
            "- Hindi\n"
        )
        sec = find_skills_section(md)
        assert sec["found"] is True
        assert sec["excluded_nonskills_spans"], "expected the Languages span to be excluded"
        n = apparent_comma_item_count(sec["cleaned_body_text"])
        assert n == 2  # Tally ERP, MS Excel only

    def test_programming_language_subsection_is_preserved(self):
        md = (
            "Skills\n"
            "### Languages\n"
            "- Python\n"
            "- JavaScript\n\n"
            "### Tools\n"
            "- Git\n"
        )
        sec = find_skills_section(md)
        assert sec["found"] is True
        assert sec["excluded_nonskills_spans"] == [], (
            "a 'Languages' subsection listing programming languages must NOT be excluded"
        )
        n = apparent_comma_item_count(sec["cleaned_body_text"])
        assert n == 3  # Python, JavaScript, Git

    def test_mixed_human_and_programming_language_content_is_not_excluded(self):
        # Deliberately ambiguous: if even one item under "Languages" is
        # NOT a known programming-language token, the span is treated
        # as (probably) human-language content and excluded -- but if
        # ANY item is unrecognized while others are programming
        # languages, this test documents the conservative "all must
        # match" rule's actual behavior rather than assuming it.
        md = "Skills\n### Languages\n- Python\n- French\n"
        sec = find_skills_section(md)
        assert sec["excluded_nonskills_spans"], (
            "mixed content (not ALL items are known programming languages) is excluded, "
            "per the conservative 'all must match' rule"
        )

    def test_no_languages_heading_means_no_exclusion(self):
        md = "Skills\n- Python\n- SQL\n"
        sec = find_skills_section(md)
        assert sec["excluded_nonskills_spans"] == []
        assert sec["cleaned_body_text"] == sec["body_text"]


# ---------------------------------------------------------------------
# Problem C: resume-006-style wrong-section boundary safety
# ---------------------------------------------------------------------

class TestBoundaryContentPlausibility:
    def test_table_like_body_is_flagged_implausible(self):
        md = (
            "Skills\n"
            "| Course | Institute | Year |\n"
            "|--------|-----------|------|\n"
            "| PGDM | XYZ College | 2010 |\n"
            "| MBA | ABC College | 2012 |\n"
        )
        sec = find_skills_section(md)
        assert sec["found"] is True, "a heading WAS matched -- honestly reported, not hidden"
        assert sec["boundary_content_implausible"] is True

    def test_normal_bullet_list_is_not_flagged_implausible(self):
        md = "Skills\n- Python\n- SQL\n- Excel\n"
        sec = find_skills_section(md)
        assert sec["boundary_content_implausible"] is False

    def test_single_stray_pipe_character_does_not_false_trigger(self):
        # A body containing one incidental "|" character (e.g. inside a
        # skill name) must not be flagged as table-like -- the check
        # requires a real separator row AND >=2 pipe-delimited rows.
        md = "Skills\n- Cost | Benefit Analysis\n- Python\n"
        sec = find_skills_section(md)
        assert sec["boundary_content_implausible"] is False


# ---------------------------------------------------------------------
# Legitimate NOT_FOUND behavior must not regress
# ---------------------------------------------------------------------

class TestNotFoundBehavior:
    def test_no_vocabulary_heading_anywhere_is_not_found(self):
        md = "# Resume\n## Summary\nSome text.\n## Experience\n- Did things.\n"
        sec = find_skills_section(md)
        assert sec["found"] is False
        assert sec["boundary_content_implausible"] is False
        assert sec["cleaned_body_text"] == ""


# ---------------------------------------------------------------------
# One normal, clean Skills section end-to-end (boundary + both signals)
# ---------------------------------------------------------------------

class TestCleanSkillsSectionEndToEnd:
    def test_clean_section_full_signal_computation(self):
        md = (
            "# Resume\n"
            "## Summary\n"
            "Some text.\n"
            "## Skills\n"
            "- Python\n"
            "- SQL\n"
            "- Excel\n"
            "## Experience\n"
            "- Did things.\n"
        )
        sec = find_skills_section(md)
        assert sec["found"] is True
        assert sec["boundary_content_implausible"] is False
        assert sec["excluded_nonskills_spans"] == []
        n = apparent_comma_item_count(sec["cleaned_body_text"])
        assert n == 3
        extracted = ["Python", "SQL", "Excel"]
        frac, nf, nt = in_section_frac(extracted, sec["cleaned_body_text"])
        assert frac == 1.0
        assert (nf, nt) == (3, 3)
        fill_ratio = len(extracted) / n
        assert fill_ratio == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
