"""
scripts/_milestone_v6_2_latency_scenarios.py

Milestone V6.2 -- blended-latency scenario analysis. Uses ONLY real,
already-measured inputs (V6.1's fresh 50-call baseline/candidate mean
latencies). No LLM calls, no invented cost figures.

Architecture-derived formula (not assumed): the V5.9 signal is
computed FROM the baseline model's own output, so escalation is
additive -- an escalated resume pays baseline_latency PLUS
candidate_latency (two sequential calls), never candidate_latency
alone. This is a real, disclosed property of a "detect-then-escalate"
architecture, distinct from (and worse per-escalated-resume than) a
hypothetical architecture that could somehow route directly to the
candidate without first calling the baseline -- no such architecture
has ever been designed, tested, or proposed in this project.

    blended_mean(p) = (1-p) * baseline_mean + p * (baseline_mean + candidate_mean)
                     = baseline_mean + p * candidate_mean

Escalation-rate scenarios are explicitly labeled as hypothetical
except where they cite a real, already-measured corpus rate (V5.9's
20.0% full-22-resume-corpus rate; V5.7's 46.7% deliberately
hard-case-stratified 6-resume rate) -- neither is claimed to be a
production-population estimate.
"""
from __future__ import annotations

import csv
from pathlib import Path

# Measured inputs -- directly from benchmark/baselines/v6_1_per_run_results.csv
BASELINE_MEAN_SECONDS = 44.40
CANDIDATE_MEAN_SECONDS = 252.28

SCENARIOS = [
    ("0%", 0.00, "floor -- no escalation ever fires"),
    ("5%", 0.05, "hypothetical low-escalation scenario"),
    ("10%", 0.10, "hypothetical low-escalation scenario"),
    ("20%", 0.20, "MATCHES V5.9's observed full-22-resume-corpus signal rate under the unmodified 0.9/0.9 policy -- NOT a production-population estimate"),
    ("30%", 0.30, "hypothetical elevated-escalation scenario"),
    ("46.7%", 0.467, "MATCHES V5.7's deliberately hard-case-stratified 6-resume research corpus rate -- NOT representative of general traffic"),
    ("50%", 0.50, "hypothetical high-escalation scenario"),
    ("100%", 1.00, "upper bound -- NOT equivalent to blanket qwen3:8b replacement (V5.6), since this architecture always pays the baseline call too"),
]


def blended_mean_latency(p: float, baseline_mean: float = BASELINE_MEAN_SECONDS,
                          candidate_mean: float = CANDIDATE_MEAN_SECONDS) -> float:
    return baseline_mean + p * candidate_mean


def build_scenarios() -> list[dict]:
    rows = []
    for label, p, note in SCENARIOS:
        blended = blended_mean_latency(p)
        rows.append({
            "escalation_rate_label": label,
            "escalation_rate_fraction": p,
            "blended_mean_latency_seconds": round(blended, 2),
            "latency_vs_baseline_only_multiplier": round(blended / BASELINE_MEAN_SECONDS, 2),
            "latency_vs_global_candidate_only_multiplier": round(blended / CANDIDATE_MEAN_SECONDS, 2),
            "rate_provenance": note,
        })
    return rows


def main() -> None:
    rows = build_scenarios()
    out = Path("benchmark/baselines/v6_2_blended_latency_scenarios.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    print(f"measured ratio (V6.1 fresh) = {round(CANDIDATE_MEAN_SECONDS / BASELINE_MEAN_SECONDS, 2)}x")


if __name__ == "__main__":
    main()
