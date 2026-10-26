"""
tests/test_milestone_v6_2_latency_scenarios.py

Tests for scripts/_milestone_v6_2_latency_scenarios.py (Milestone V6.2).
Verifies the blended-latency formula against hand-computed expected
values, and confirms the scenario table's provenance labels are
present for every row (no un-labeled figure). No LLM calls.
"""
from scripts._milestone_v6_2_latency_scenarios import (
    BASELINE_MEAN_SECONDS,
    CANDIDATE_MEAN_SECONDS,
    blended_mean_latency,
    build_scenarios,
)


class TestBlendedMeanLatencyFormula:
    def test_zero_escalation_equals_baseline_only(self):
        assert blended_mean_latency(0.0) == BASELINE_MEAN_SECONDS

    def test_full_escalation_equals_baseline_plus_candidate(self):
        assert blended_mean_latency(1.0) == BASELINE_MEAN_SECONDS + CANDIDATE_MEAN_SECONDS

    def test_hand_computed_20_percent(self):
        # 44.40 + 0.20 * 252.28 = 44.40 + 50.456 = 94.856
        assert round(blended_mean_latency(0.20), 3) == 94.856

    def test_monotonically_increasing_in_escalation_rate(self):
        vals = [blended_mean_latency(p) for p in (0.0, 0.1, 0.2, 0.5, 1.0)]
        assert vals == sorted(vals)
        assert len(set(vals)) == len(vals)  # strictly increasing, no ties


class TestScenarioTable:
    def test_every_row_has_a_provenance_label(self):
        rows = build_scenarios()
        for r in rows:
            assert r["rate_provenance"], f"row {r['escalation_rate_label']} missing provenance"

    def test_20_percent_row_cites_v5_9(self):
        rows = build_scenarios()
        row = next(r for r in rows if r["escalation_rate_label"] == "20%")
        assert "V5.9" in row["rate_provenance"]

    def test_100_percent_row_disclaims_blanket_replacement_equivalence(self):
        rows = build_scenarios()
        row = next(r for r in rows if r["escalation_rate_label"] == "100%")
        assert "NOT equivalent" in row["rate_provenance"]
