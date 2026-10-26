"""
tests/test_milestone_v6_5_exploratory_punctuation_insensitive.py

Tests for the EXPLORATORY-ONLY punctuation-insensitive secondary check
(scripts/_milestone_v6_5_exploratory_punctuation_insensitive.py).
This is NOT the frozen V6.4 proxy -- see
tests/test_milestone_v6_4_fidelity_proxy.py for that suite, run
unmodified in this milestone too.

No LLM calls.
"""
from scripts._milestone_v6_5_exploratory_punctuation_insensitive import (
    is_grounded_punctuation_insensitive,
)


class TestRealFalseWarningCasesFromFullCorpusRun:
    def test_email_hyphen_variant(self):
        r = is_grounded_punctuation_insensitive("Email", "Internet and E-mail")
        assert r["grounded_exploratory"] is True

    def test_datafactory_space_variant(self):
        r = is_grounded_punctuation_insensitive("DataFactory", "AWS, AZURE-DataBricks, Data Factory, Analysis Services")
        assert r["grounded_exploratory"] is True

    def test_azure_devops_space_variant(self):
        r = is_grounded_punctuation_insensitive("Azure Devops", "Software Dev.: AzureDevops, ANGULAR 8")
        assert r["grounded_exploratory"] is True

    def test_powerbi_space_variant(self):
        r = is_grounded_punctuation_insensitive("PowerBI", "Visualization Tools: Power BI, Grafana, Tableau")
        assert r["grounded_exploratory"] is True


class TestLengthGuardPreventsCoincidentalMatch:
    def test_short_token_below_guard_never_matches(self):
        # The real risk found during this milestone's own investigation:
        # stripping spaces from a large body can coincidentally contain
        # a short token (e.g. "hive" inside "achivements"). The length
        # guard must block this regardless of body content.
        r = is_grounded_punctuation_insensitive("Hive", "AWARDS AND ACHIVEMENTS: KPMG SPOT Award")
        assert r["grounded_exploratory"] is False
        assert r["reason"] == "below_min_length_guard"

    def test_genuinely_absent_longer_item_still_rejected(self):
        r = is_grounded_punctuation_insensitive("Kubernetes Engine", "Python, SQL, Excel")
        assert r["grounded_exploratory"] is False
        assert r["reason"] == "no_match_even_after_punctuation_strip"


class TestDoesNotOverclaim:
    def test_dotted_token_distinction_preserved(self):
        # Periods are NOT stripped -- "Node.js" and "Node" must remain
        # distinguishable even under this looser secondary check.
        r = is_grounded_punctuation_insensitive("Node.js", "Node, Express.js, MongoDB")
        assert r["grounded_exploratory"] is False
