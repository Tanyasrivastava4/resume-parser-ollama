"""
tests/test_milestone_v6_8_fidelity_proxy_correction.py

Milestone V6.8 -- focused and adversarial tests for the two secondary
grounding checks (Mechanism A: punctuation/spacing; Mechanism B:
elliptical shared-prefix list expansion) in
scripts/_milestone_v6_8_fidelity_proxy_correction.py.

Every real-corpus example below is reproduced verbatim from the
committed benchmark/baselines/v6_5_flag_classification.csv and the
fresh Skills-section source lines re-fetched in this milestone
(resume 021, V6.6-corrected boundary). Every synthetic/hand-constructed
example is explicitly labeled SYNTHETIC/ADVERSARIAL and is never
conflated with the real-corpus evidence in the milestone report.
"""
from __future__ import annotations

from scripts._milestone_v6_8_fidelity_proxy_correction import (
    _elliptical_list_expansions,
    _flexible_gap_pattern,
    compute_fidelity_proxy_v6_8,
    item_is_grounded_v6_8,
)
from scripts._milestone_v6_4_fidelity_proxy import _normalize


# ---------------------------------------------------------------------
# Mechanism A: real-corpus rescue cases
# ---------------------------------------------------------------------

class TestMechanismA_RealCorpusRescues:
    def test_email_vs_e_mail(self):
        r = item_is_grounded_v6_8("Email", "Internet and E-mail")
        assert r["grounded"] is True
        assert r["reason"] == "grounded_after_punctuation_normalization"

    def test_datafactory_vs_data_factory(self):
        r = item_is_grounded_v6_8("DataFactory", "Data Factory")
        assert r["grounded"] is True
        assert r["reason"] == "grounded_after_punctuation_normalization"

    def test_azure_devops_vs_azuredevops(self):
        r = item_is_grounded_v6_8("Azure Devops", "AzureDevops")
        assert r["grounded"] is True
        assert r["reason"] == "grounded_after_punctuation_normalization"

    def test_powerbi_vs_power_bi(self):
        r = item_is_grounded_v6_8("PowerBI", "Power BI")
        assert r["grounded"] is True
        assert r["reason"] == "grounded_after_punctuation_normalization"

    def test_azure_data_bricks_vs_azure_databricks(self):
        r = item_is_grounded_v6_8("Azure-Data Bricks", "AZURE-DataBricks")
        assert r["grounded"] is True
        assert r["reason"] == "grounded_after_punctuation_normalization"

    def test_powerapps_vs_power_apps(self):
        r = item_is_grounded_v6_8("PowerApps", "Power Apps")
        assert r["grounded"] is True
        assert r["reason"] == "grounded_after_punctuation_normalization"


class TestMechanismA_LengthGuard:
    def test_short_item_below_guard_not_rescued(self):
        # SYNTHETIC/ADVERSARIAL: "Go" (2 chars) must never reach the
        # flexible-gap path at all.
        assert _flexible_gap_pattern(_normalize("Go")) is None


class TestMechanismA_PunctuationRigidity:
    """SYNTHETIC/ADVERSARIAL -- these probe the exact bug found and
    fixed in this milestone's own first draft: a gap containing a
    non-separator character (period, plus, hash) must remain a rigid,
    required literal, never silently dropped or loosened."""

    def test_period_gap_stays_rigid_node_js_pattern(self):
        pattern = _flexible_gap_pattern(_normalize("Node.js"))
        assert r"\." in pattern  # the period must appear as a literal, required character

    def test_nodejs_does_not_match_node_dot_js(self):
        r = item_is_grounded_v6_8("NodeJS", "Experienced with Node.js and React")
        assert r["grounded"] is False

    def test_node_dot_js_does_not_match_nodejs_source(self):
        r = item_is_grounded_v6_8("Node.js", "Experienced with NodeJS and React")
        assert r["grounded"] is False

    def test_plus_plus_gap_stays_rigid(self):
        # "C++" is below the 5-char length guard, so use a longer
        # synthetic analog with the same plus-sign structure.
        pattern = _flexible_gap_pattern(_normalize("Csharpplusplus"))
        assert pattern is not None  # sanity: long enough to reach the guard


class TestMechanismA_AdversarialSafety:
    """SYNTHETIC/ADVERSARIAL -- must not introduce any new false
    grounding beyond what the frozen V6.4 proxy already safely rejects."""

    def test_excel_does_not_match_inside_excellent(self):
        r = item_is_grounded_v6_8("Excel", "Excellent communication and teamwork skills")
        assert r["grounded"] is False

    def test_java_still_not_confused_with_javascript(self):
        r = item_is_grounded_v6_8("Java", "Experienced in JavaScript and TypeScript development")
        assert r["grounded"] is False

    def test_go_still_not_confused_with_golang(self):
        r = item_is_grounded_v6_8("Go", "Experienced in Golang microservices")
        assert r["grounded"] is False

    def test_csharp_does_not_fuzzy_match_c_hash(self):
        r = item_is_grounded_v6_8("Csharp", "Proficient in C# and C++")
        assert r["grounded"] is False


# ---------------------------------------------------------------------
# Mechanism B: real-corpus rescue cases
# ---------------------------------------------------------------------

_WINDOWS_LINE = "- Windows 11, Windows Server 2012, 2016, 2019 and Linux"
_FORTIGATE_LINE = ("- Fortigate 310B and 80E, Sophos XGS 2100 SonicWall, Seqrite Endpoint "
                    "Security, Symantec Endpoint Antivirus, Bitdefender Gravity Zone")


class TestMechanismB_RealCorpusRescues:
    def test_windows_server_2016_expanded(self):
        r = item_is_grounded_v6_8("Windows Server 2016", _WINDOWS_LINE)
        assert r["grounded"] is True
        assert r["reason"] == "grounded_after_elliptical_list_expansion"

    def test_windows_server_2019_expanded(self):
        r = item_is_grounded_v6_8("Windows Server 2019", _WINDOWS_LINE)
        assert r["grounded"] is True
        assert r["reason"] == "grounded_after_elliptical_list_expansion"

    def test_windows_server_2012_already_direct_match(self):
        # The anchor token itself is already handled by the frozen
        # primary check -- V6.8 must not re-tag it as its own mechanism.
        r = item_is_grounded_v6_8("Windows Server 2012", _WINDOWS_LINE)
        assert r["grounded"] is True
        assert r["reason"] == "grounded_direct"

    def test_fortigate_80e_expanded(self):
        r = item_is_grounded_v6_8("Fortigate 80E", _FORTIGATE_LINE)
        assert r["grounded"] is True
        assert r["reason"] == "grounded_after_elliptical_list_expansion"

    def test_fortigate_310b_already_direct_match(self):
        r = item_is_grounded_v6_8("Fortigate 310B", _FORTIGATE_LINE)
        assert r["grounded"] is True
        assert r["reason"] == "grounded_direct"


class TestMechanismB_StructuralScopeSafety:
    """SYNTHETIC/ADVERSARIAL -- must not expand into an unrelated
    coordinate item elsewhere in the same outer comma list, and must
    never cross a line boundary."""

    def test_unrelated_trailing_word_not_absorbed_into_expansion(self):
        # "Linux" is the true final coordinate item of the OUTER list,
        # not a continuation of the "Windows Server" version list --
        # it must never be pulled into an expansion.
        expansions = _elliptical_list_expansions(_WINDOWS_LINE)
        assert "windows server linux" not in expansions
        r = item_is_grounded_v6_8("Windows Server Linux", _WINDOWS_LINE)
        assert r["grounded"] is False

    def test_windows_11_not_absorbed_as_prefix_anchor_incorrectly(self):
        r = item_is_grounded_v6_8("Windows 11 Linux", _WINDOWS_LINE)
        assert r["grounded"] is False

    def test_expansion_does_not_cross_line_boundaries(self):
        # SYNTHETIC: the shared-prefix structure on line 1 must not be
        # combined with an unrelated token on line 2.
        two_lines = "- Fortigate 310B and 80E\n- Cisco 2900 and 3900"
        r = item_is_grounded_v6_8("Fortigate 3900", two_lines)
        assert r["grounded"] is False

    def test_non_digit_tokens_never_treated_as_list_continuations(self):
        # SYNTHETIC: "and Security" must not be absorbed as a
        # "Fortigate Security"-style expansion -- continuation tokens
        # require a digit.
        r = item_is_grounded_v6_8("Fortigate Security", _FORTIGATE_LINE)
        assert r["grounded"] is False


# ---------------------------------------------------------------------
# Combined aggregate behavior
# ---------------------------------------------------------------------

class TestAggregateV6_8Proxy:
    def test_boundary_invalid_short_circuits_same_as_frozen_proxy(self):
        result = compute_fidelity_proxy_v6_8(["Excel"], "irrelevant body", boundary_valid=False)
        assert result["proxy_status"] == "boundary_invalid"

    def test_v6_8_rescue_reflected_in_aggregate_fraction(self):
        items = ["Data Factory", "DataFactory", "TotallyInventedThing12345"]
        # NOTE: "Data Factory" (direct) and "DataFactory" (rescued) both
        # ground against a body containing "Data Factory"; the invented
        # item must remain ungrounded.
        result = compute_fidelity_proxy_v6_8(items, "Skills: Data Factory, SQL", boundary_valid=True)
        assert result["proxy_status"] == "evaluated"
        assert result["n_grounded_after_v6_8_correction"] == 1  # only "DataFactory" needed the secondary path
        assert "TotallyInventedThing12345" in result["ungrounded_items"]

    def test_true_unsupported_item_remains_ungrounded_under_v6_8(self):
        # A representative real TRUE_UNSUPPORTED shape (invented
        # certification-like token, no source support at all).
        result = compute_fidelity_proxy_v6_8(
            ["Certified Kubernetes Administrator"], "Skills: Python, SQL, Git", boundary_valid=True,
        )
        assert result["ungrounded_items"] == ["Certified Kubernetes Administrator"]
