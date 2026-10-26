"""
scripts/_milestone_v7_0_evidence_map.py

Milestone V7.0 Phase A -- builds the full 22-resume evidence map from
already-committed, real artifacts only: V5.9's full-corpus signal
distribution, V5.8's offline GT-based mechanism reference, V6.5's
population inventory (which also records V5.6/V6.1 real-call
coverage), V6.5's flag classification (fidelity-proxy evidence), and
the mechanism assignments directly published in the V5.6 and V6.2
milestone reports (transcribed here as a literal lookup table, cited
per-resume, not inferred). ZERO new LLM calls -- reads only
already-existing committed CSVs.

Explicit values used throughout, per the governing brief:
CONFIRMED / SUSPECTED / UNKNOWN / NOT_APPLICABLE -- never invented
certainty where no real evidence exists.

Writes: benchmark/baselines/v7_0_evidence_map.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

# Mechanism assignments PUBLISHED and CITED directly in real milestone
# reports (V5.6 Section "Mechanism-level findings", V6.2 Section 5's
# table). Transcribed literally, not re-derived or guessed. Only
# resumes with a directly-published, real-data mechanism classification
# appear here -- every other resume is UNKNOWN, not inferred.
_CONFIRMED_MECHANISM = {
    "013": ("CONFIRMED", "long-list tail-omission",
            "V5.6 (121-GT-item list, 40/41 chronic items recovered under qwen3:8b); "
            "V6.2 re-confirmed from real V6.1 data"),
    "023": ("CONFIRMED", "tail-omission (multi-subheading block)",
            "V5.6 (6/6 chronically-missed 'Domain' subheading items recovered under qwen3:8b)"),
    "005": ("CONFIRMED", "content-substitution (wrong document section)",
            "V5.6 (baseline run3 extracted certifications instead of technical skills, "
            "reproducing a previously-documented V3.4 mechanism); V6.5/V6.8/V6.9 confirmed "
            "28 TRUE_UNSUPPORTED items, boundary-overrun caveat also applies (V6.5/V6.6)"),
    "021": ("CONFIRMED", "label-inclusion/segmentation-convention instability",
            "V5.6 (recall=0.000 runs on BOTH models are category-label-prepended, not missing "
            "content); reproduces at similar rate under both models -- capacity-independent"),
    "018": ("CONFIRMED", "multi-subheading category-omission",
            "V6.2 (baseline: 100% of first subheading, 0% of entire second subheading, "
            "all-or-nothing pattern); V6.1 pilot showed non-overlapping improvement under qwen3:8b"),
    "007": ("CONFIRMED", "minor small-list incompleteness (not length-driven)",
            "V6.2 (misses exactly 1/7 items on a short list); V6.1 pilot showed non-overlapping "
            "improvement under qwen3:8b"),
    "015": ("CONFIRMED", "segmentation-convention mechanism (capacity-independent)",
            "V6.2 (both baseline AND candidate merge multiple GT-atomic items into one shared "
            "descriptive phrase); V6.1 pilot showed NO improvement -- matches 021's mechanism type"),
    "001": ("SUSPECTED", "no material failure on V5.6's own corpus pass; NOT independently re-verified as a treatment case",
            "V5.6 used 001 only as a negative control (never used to judge the primary hypothesis); "
            "candidate improved 0.722->1.000 but this was not part of any predeclared success criterion"),
    "016": ("CONFIRMED", "stable sentinel -- no failure mechanism, used as a regression control",
            "V5.6 and V6.1 both show zero change across every metric, every run, both models"),
    "022": ("SUSPECTED", "content-substitution-adjacent (verbatim Work-Experience bullets presented as Skills)",
            "V6.5 (7 TRUE_UNSUPPORTED items, structurally similar in kind to 005's confirmed mechanism "
            "but never independently investigated as its own dedicated mechanism study)"),
    "019": ("UNKNOWN", "material incompleteness observed (offline fuzzy_recall=0.6923) but signal does not "
                        "flag it (in_section_frac=0.8421<0.9) and no candidate-model evidence exists",
            "V5.8/V5.9 offline reference; V6.5 also found 6 TRUE_UNSUPPORTED items -- an UNRELATED "
            "fidelity finding, never mechanistically connected to the completeness gap"),
    "020": ("CONFIRMED", "GT content not literally present in Skills-section source text at all "
                          "(adjudicated from free-text elsewhere in the resume) -- NOT a model-capacity-"
                          "responsive mechanism, a structural signal-blind-spot",
            "V5.9 Section 9 (direct GT inspection: 'Instrumentation', 'PLC-HMI Programming', "
            "'Validation' do not appear anywhere in the Skills section source text)"),
    "006": ("NOT_APPLICABLE", "boundary structurally implausible (table-like) -- signal unavailable, "
                               "no mechanism can be characterized until the underlying boundary defect is fixed",
            "V5.9 boundary safety check"),
    "017": ("NOT_APPLICABLE", "no Skills-section boundary found at all -- correct abstention, not a mechanism",
            "V5.9 (matches V3.x-era finding, resume genuinely has no Skills-family heading)"),
}


def main() -> None:
    with open("benchmark/baselines/v5_9_full_corpus_signal_distribution.csv", encoding="utf-8") as f:
        signal_rows = {r["resume_id"]: r for r in csv.DictReader(f)}
    with open("benchmark/baselines/v5_8_offline_mechanism_reference.csv", encoding="utf-8") as f:
        offline_rows = {r["resume_id"]: r for r in csv.DictReader(f)}
    with open("benchmark/baselines/v6_5_population_inventory.csv", encoding="utf-8") as f:
        v6_5_rows = {r["resume_id"]: r for r in csv.DictReader(f)}
    with open("benchmark/baselines/v6_5_flag_classification.csv", encoding="utf-8") as f:
        flag_rows = list(csv.DictReader(f))

    from collections import Counter
    flagged_counts = Counter(r["resume_id"] for r in flag_rows)
    true_unsupported_counts = Counter(r["resume_id"] for r in flag_rows if r["category"] == "TRUE_UNSUPPORTED")

    out_rows = []
    for rid in [f"{i:03d}" for i in range(1, 24)]:
        sig = signal_rows.get(rid)
        off = offline_rows.get(rid)
        v65 = v6_5_rows.get(rid)
        mech = _CONFIRMED_MECHANISM.get(rid, ("UNKNOWN", "not investigated / no real mechanism-level finding published", ""))

        if sig is None:
            continue  # no resume numbered this way in the corpus

        boundary_status = "CONFIRMED_NOT_FOUND" if sig["signal_invalid_reason"] == "Skills section not found" else (
            "CONFIRMED_IMPLAUSIBLE" if sig["signal_invalid_reason"] else "CONFIRMED_FOUND_PLAUSIBLE"
        )

        richer = v65["additional_richer_evidence"] if v65 else "none"
        part_v5_6 = "V5.6" in (richer or "")
        part_v6_1 = "V6.1" in (richer or "")

        out_rows.append({
            "resume_id": rid,
            "skills_signal_valid": sig["signal_valid"],
            "would_escalate_v5_7_policy": sig["would_escalate_v5_7_policy"],
            "fill_ratio": sig["fill_ratio"],
            "in_section_frac": sig["in_section_frac"],
            "boundary_status": boundary_status,
            "offline_fuzzy_recall_single_baseline": off["fuzzy_recall"] if off else "",
            "offline_label": off["offline_label"] if off else "",
            "known_mechanism_certainty": mech[0],
            "known_mechanism_description": mech[1],
            "known_mechanism_evidence_source": mech[2],
            "baseline_candidate_model_evidence_exists": part_v5_6 or part_v6_1,
            "part_of_v5_6_real_pilot": part_v5_6,
            "part_of_v6_1_real_pilot": part_v6_1,
            "used_as_control": rid == "016",
            "fidelity_proxy_evidence_exists": rid in v6_5_rows and (v65["eligible"] == "True"),
            "fidelity_proxy_flagged_item_count": flagged_counts.get(rid, 0),
            "fidelity_proxy_true_unsupported_count": true_unsupported_counts.get(rid, 0),
            "known_ambiguity_or_integrity_limitation": (
                "boundary-overrun caveat applies (V6.5/V6.6, v2_4h_benchmark lineage only)" if rid == "005" else
                "boundary structurally implausible, signal/fidelity unavailable" if rid == "006" else
                "no Skills-section boundary found, signal/fidelity unavailable" if rid == "017" else
                "structural signal-blind-spot: GT content not literally present in Skills-section source text" if rid == "020" else
                "content-substitution-adjacent finding never independently investigated as its own mechanism study" if rid == "022" else
                "none identified"
            ),
        })

    out = Path("benchmark/baselines/v7_0_evidence_map.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {out} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
