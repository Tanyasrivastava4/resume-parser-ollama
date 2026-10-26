"""
tests/test_milestone_v6_9_fidelity_proxy_correction.py

Milestone V6.9 -- focused and adversarial tests for the one new
secondary grounding check adopted this milestone: merged comma-joined
multi-item string segmentation (`_grounded_after_segmentation_split`
in scripts/_milestone_v6_9_fidelity_proxy_correction.py).

Every real-corpus example is reproduced verbatim from the committed
benchmark/baselines/v6_5_flag_classification.csv
(FALSE_WARNING_LABEL_OR_SEGMENTATION rows) and real Skills-section
source text re-fetched in this milestone. Every synthetic/hand-
constructed example is explicitly labeled SYNTHETIC/ADVERSARIAL.
"""
from __future__ import annotations

from scripts._milestone_v6_9_fidelity_proxy_correction import (
    compute_fidelity_proxy_v6_9,
    item_is_grounded_v6_9,
)

_BODY_021 = ("- Fortigate 310B and 80E, Sophos XGS 2100 SonicWall, Seqrite Endpoint "
             "Security, Symantec Endpoint Antivirus, Bitdefender Gravity Zone")

_BODY_005 = (
    "Software Dev.: AzureDevops,ANGULAR 8,JavaScript,Visual studio Core, GitHub, Power Apps,CSS, "
    "Mulesoft, Moogsoft Apex AiOps, AppDynamics,\n"
    "SharePoint Infra.: AWS, AZURE-DataBricks, Data Factory, Analysis Services, ADLS Gen1,2, ELK, "
    "Kubernetes. IAC-Terrafrom, Ansible\n"
    "Languages: Python, R, C, C++, Java, SQL, Pyspark; Database: MySQL, Oracle, Elastic Search, "
    "Ms. Access, Snowflake.\n"
    "Visualization Tools: Power BI, Grafana, Tableau, Kibana, SAP BO. Web Dev. : Flask, Web scraping. "
    "SIEM: Cribl, Splunk, ELK\n"
    "Vertical Exposure: Data Engineering, Software Development and Computer Networking."
)


class TestSegmentationMechanism_RealCorpusRescue:
    def test_021_clean_merged_string_all_parts_verifiable(self):
        item = ("Firewall & Security: Fortigate 310B and 80E, Sophos XGS 2100 SonicWall, "
                "Seqrite Endpoint Security, Bitdefender Gravity Zone")
        r = item_is_grounded_v6_9(item, _BODY_021)
        assert r["grounded"] is True
        assert r["reason"] == "grounded_after_segmentation_split"


class TestSegmentationMechanism_CorrectlyStaysRejected:
    """These two real merged-string items from resume 005 must NOT be
    rescued -- each contains at least one genuinely ungrounded
    sub-part (VSC/Logstash/Linux/Qracle DB/Postman/Terraform in the
    first; ADF/ASA/Hive in the second), confirmed by direct source
    inspection in this milestone. The all-or-nothing rule is the
    safety design under test here."""

    def test_005_merged_string_with_contaminated_parts_item1(self):
        item = ("Python, VSC, SQL, Elastic search, Grafana, Snowflake, MuleSoft, Logstash, "
                "Apex AiOps, AWS, Databricks, DataFactory, Linux, Azure Devops, Cribl, Ansible, "
                "GitHub, Qracle DB, PySpark, BI, Postman, Kubernetes, Terraform")
        r = item_is_grounded_v6_9(item, _BODY_005)
        assert r["grounded"] is False

    def test_005_merged_string_with_contaminated_parts_item2(self):
        item = ("R, PowerBI, Tableau, MySQL, Python, SQL, Azure-Data Bricks, ADF, ASA, Hive, "
                "ELK, Snowflake, PySpark, Access, Moogsoft, Grafana, Kubernetes")
        r = item_is_grounded_v6_9(item, _BODY_005)
        assert r["grounded"] is False


class TestSegmentationMechanism_SafetyAgainstRealTrueUnsupported:
    """SYNTHETIC bodies, but the ITEM TEXT is copied verbatim from real
    V6.5 TRUE_UNSUPPORTED classifications that happen to contain
    commas -- the critical safety population for this mechanism."""

    def test_022_work_experience_bullet_not_rescued(self):
        r = item_is_grounded_v6_9(
            "Own test planning, execution governance, defect management, and reporting",
            "Skills: Python, SQL, Selenium, Git",
        )
        assert r["grounded"] is False

    def test_022_second_work_experience_bullet_not_rescued(self):
        r = item_is_grounded_v6_9(
            "Collaborate with product owners, developers, and stakeholders",
            "Skills: Python, SQL, Selenium, Git",
        )
        assert r["grounded"] is False

    def test_005_education_line_not_rescued(self):
        r = item_is_grounded_v6_9(
            "AISSE (CBSE) 2014, 10th (CBSE) 2012, AISSCE (CBSE) 2013",
            "Skills: Python, SQL, Azure",
        )
        assert r["grounded"] is False


class TestSegmentationMechanism_AdversarialSafety:
    """SYNTHETIC/ADVERSARIAL -- probes the all-or-nothing design and
    known unsafe-relaxation shapes."""

    def test_one_real_one_fabricated_part_not_rescued(self):
        r = item_is_grounded_v6_9("Python, FakeInventedCertificationXYZ", "Skills: Python, SQL, Git")
        assert r["grounded"] is False

    def test_two_fabricated_parts_not_rescued(self):
        r = item_is_grounded_v6_9("FakeThing1, FakeThing2", "Skills: Python, SQL, Git")
        assert r["grounded"] is False

    def test_java_javascript_still_not_confused_within_segmentation(self):
        r = item_is_grounded_v6_9("Java, TypeScript", "Experienced in JavaScript and TypeScript development")
        assert r["grounded"] is False

    def test_single_item_no_comma_does_not_trigger_segmentation(self):
        r = item_is_grounded_v6_9("TotallyInventedThing12345", "Skills: Python, SQL")
        assert r["grounded"] is False
        assert r["reason"] != "grounded_after_segmentation_split"

    def test_empty_middle_part_from_double_comma_handled_safely(self):
        # SYNTHETIC: "Python, , SQL" -- both real non-empty parts ground.
        r = item_is_grounded_v6_9("Python, , SQL", "Skills: Python, SQL")
        assert r["grounded"] is True
        assert r["reason"] == "grounded_after_segmentation_split"


class TestAggregateV6_9Proxy:
    def test_boundary_invalid_short_circuits(self):
        result = compute_fidelity_proxy_v6_9(["Python"], "irrelevant", boundary_valid=False)
        assert result["proxy_status"] == "boundary_invalid"

    def test_segmentation_rescue_reflected_in_aggregate(self):
        # Non-contiguous pairing (skips "Seqrite Endpoint Security" in
        # the source order) so the item cannot ground directly as a
        # whole substring -- exercises the segmentation path, not the
        # frozen direct-match path.
        items = ["Sophos XGS 2100 SonicWall, Bitdefender Gravity Zone", "TotallyFabricated9999"]
        result = compute_fidelity_proxy_v6_9(items, _BODY_021, boundary_valid=True)
        assert result["n_grounded_after_v6_9_correction"] == 1
        assert "TotallyFabricated9999" in result["ungrounded_items"]
