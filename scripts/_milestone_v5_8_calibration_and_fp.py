"""
scripts/_milestone_v5_8_calibration_and_fp.py

Milestone V5.8 Step 5/6 -- writes:

  benchmark/baselines/v5_8_policy_calibration_assessment.csv
  benchmark/baselines/v5_8_false_positive_analysis.csv

Both derived deterministically from already-written V5.8 CSVs (boundary
validation, full-corpus signal distribution, offline mechanism
reference) plus the manual boundary classifications finalized in
_milestone_v5_8_finalize_review.py. GT (via the offline mechanism
reference) is used ONLY as a retrospective evaluation reference here --
never as a routing-signal input (fill_ratio/in_section_frac are
computed with zero GT access in _milestone_v5_8_full_corpus_validation.py).

ZERO new LLM calls. No production file touched.
"""
from __future__ import annotations

import csv
from pathlib import Path

BASE = Path("benchmark/baselines")


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["resume_id"]: r for r in csv.DictReader(f)}


def main() -> None:
    boundary = load(BASE / "v5_8_boundary_validation.csv")
    signal = load(BASE / "v5_8_full_corpus_signal_distribution.csv")
    offline = load(BASE / "v5_8_offline_mechanism_reference.csv")

    valid_ids = [rid for rid, s in signal.items() if s["signal_valid"] == "True"]

    def is_material(rid):
        fr = offline[rid]["fuzzy_recall"]
        return None if fr == "" else float(fr) < 0.90

    # --------------------------------------------------------------
    # Step 5: policy calibration sweep. A SMALL, interpretable
    # neighborhood around V5.7's provisional 0.9/0.9, not a full grid
    # search chosen to maximize in-sample performance (that would
    # violate this milestone's own explicit anti-overfitting
    # instruction). Values chosen: the provisional value itself, one
    # looser and one tighter fill_ratio step, one looser and one
    # tighter in_section_frac step.
    # --------------------------------------------------------------
    policies = [
        ("provisional_0.90_0.90", 0.90, 0.90),
        ("tighter_fill_0.85_0.90", 0.85, 0.90),
        ("tighter_fill_0.75_0.90", 0.75, 0.90),
        ("looser_fill_0.95_0.90", 0.95, 0.90),
        ("provisional_fill_looser_sec_0.90_0.80", 0.90, 0.80),
        ("exploratory_0.75_0.80", 0.75, 0.80),
    ]

    calib_rows = []
    for name, fill_thr, sec_thr in policies:
        tp = fp = fn = tn = 0
        for rid in valid_ids:
            s = signal[rid]
            fill = float(s["fill_ratio"])
            sec = float(s["in_section_frac"])
            escalate = (fill < fill_thr) and (sec >= sec_thr)
            mat = is_material(rid)
            if mat is None:
                continue
            if escalate and mat:
                tp += 1
            elif escalate and not mat:
                fp += 1
            elif not escalate and mat:
                fn += 1
            else:
                tn += 1
        n_esc = tp + fp
        n_material = tp + fn
        precision = round(tp / n_esc, 3) if n_esc else ""
        recall = round(tp / n_material, 3) if n_material else ""
        calib_rows.append({
            "policy_name": name, "fill_ratio_threshold": fill_thr,
            "in_section_frac_threshold": sec_thr,
            "n_signal_eligible": len(valid_ids),
            "n_escalated": n_esc, "true_positive": tp, "false_positive": fp,
            "false_negative": fn, "true_negative": tn,
            "precision_vs_offline_material_reference": precision,
            "recall_vs_offline_material_reference": recall,
            "note": "",
        })

    calib_rows[0]["note"] = (
        "V5.7's provisional policy, applied unchanged to all 22 resumes. "
        "Precision/recall here are materially worse than V5.7's clean "
        "6-resume story (0 false positives on the stable sentinel) -- "
        "full-corpus validation surfaces real imprecision V5.7's small, "
        "stratified sample could not reveal."
    )
    calib_rows[-1]["note"] = (
        "EXPLORATORY ONLY, NOT A CALIBRATED THRESHOLD. Best precision/recall "
        "combination found in this narrow neighborhood sweep (6 policies "
        "tested, not a full grid search). Based on only 21 signal-eligible "
        "resumes with 9 offline-material-incompleteness positives -- moving "
        "1-2 resumes between categories changes these numbers substantially. "
        "Requires validation on an independent sample before being treated as "
        "a real threshold; explicitly NOT recommended for adoption by this "
        "milestone."
    )

    calib_out = BASE / "v5_8_policy_calibration_assessment.csv"
    with calib_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(calib_rows[0].keys()))
        w.writeheader()
        w.writerows(calib_rows)
    print(f"wrote {calib_out} ({len(calib_rows)} rows)")

    # --------------------------------------------------------------
    # Step 6: false-positive analysis. Every resume the PROVISIONAL
    # 0.9/0.9 policy would escalate, cross-checked against the offline
    # mechanism reference (retrospective evaluation only).
    # --------------------------------------------------------------
    fp_rows = []
    for rid in valid_ids:
        s = signal[rid]
        fill = float(s["fill_ratio"])
        sec = float(s["in_section_frac"])
        escalate = (fill < 0.90) and (sec >= 0.90)
        if not escalate:
            continue
        o = offline[rid]
        b = boundary[rid]
        fuzzy_recall = o["fuzzy_recall"]
        mat = is_material(rid)
        if mat:
            determination = "justified_material_incompleteness_present"
            reason = f"Baseline fuzzy recall {fuzzy_recall} on this run indicates real, non-trivial incompleteness."
        else:
            determination = "likely_false_positive"
            reason = (
                f"Baseline fuzzy recall {fuzzy_recall} on this run is already at or near 1.0 -- "
                "escalation would add latency/cost with no measurable recall gain on this "
                "specific observed run. Cannot rule out this resume being bimodal across "
                "unobserved runs (only 013/023 have V5.6's multi-run data); this is a "
                "single-run false-positive finding, not proof the resume never benefits."
            )
        fp_rows.append({
            "resume_id": rid,
            "fill_ratio": round(fill, 4), "in_section_frac": round(sec, 4),
            "boundary_classification": b["classification"],
            "offline_fuzzy_recall_reference": fuzzy_recall,
            "offline_label_reference": o["offline_label"],
            "determination": determination,
            "reason": reason,
        })

    fp_out = BASE / "v5_8_false_positive_analysis.csv"
    with fp_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fp_rows[0].keys()))
        w.writeheader()
        w.writerows(fp_rows)
    print(f"wrote {fp_out} ({len(fp_rows)} rows)")
    n_fp = sum(1 for r in fp_rows if r["determination"] == "likely_false_positive")
    print(f"{n_fp} of {len(fp_rows)} escalated resumes are likely false positives under the provisional policy")


if __name__ == "__main__":
    main()
