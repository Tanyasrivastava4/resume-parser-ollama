"""
scripts/_milestone_t13_analysis.py

Milestone T13/Phase 2C -- consolidates all real-call checkpoints and
computes the pre-registered entity-level/field-level/relationship
metrics, reusing existing, unmodified deterministic infrastructure as
measurement tools only (misrouting_detector's fuzzy company matcher,
work_experience_validator's technology-presence check).
"""
from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from validators.misrouting_detector import _fuzzy_contains  # noqa: E402

CKPT_DIR = Path(
    "/private/tmp/claude-501/-Users-ruchigupta-Documents-ResumeAIBenchmark/"
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/t13_checkpoints"
)
GT_DIR = Path("benchmark/ground_truth/adjudicated_v1")

POPULATION = {
    "001": "resume001.json",
    "006": "resume006.json",
    "013": "resume013.json",
    "016": "resume016.json",
    "021": "resume021.json",
}

# Known real Project-heading names for resume 001 (source-grounded,
# T7/T8/T9) -- used only to positively identify a fabricated entity's
# origin, not to alter any measurement.
KNOWN_PROJECT_NAMES_001 = ["SoWhat", "TGES", "Asha Learningy", "Adiroha"]

GT_JOBS = {
    rid: json.loads((GT_DIR / f).read_text()).get("work_experience", [])
    for rid, f in POPULATION.items()
}


def _load(rid, condition, run):
    with (CKPT_DIR / f"{rid}__{condition}_{run}.pkl").open("rb") as f:
        return pickle.load(f)


def _match_jobs_to_gt(rid, jobs):
    """Greedy fuzzy matching of returned jobs to GT jobs by company,
    reusing the real, unmodified _fuzzy_contains. Returns
    (matched_pairs, unmatched_jobs, unmatched_gt)."""
    gt_jobs = GT_JOBS[rid]
    gt_used = [False] * len(gt_jobs)
    matched_pairs = []
    unmatched_jobs = []
    for job in jobs:
        company = str(job.get("company") or "")
        match_idx = None
        for i, gt in enumerate(gt_jobs):
            if gt_used[i]:
                continue
            if company and _fuzzy_contains(company, gt.get("company") or ""):
                match_idx = i
                break
        if match_idx is not None:
            gt_used[match_idx] = True
            matched_pairs.append((job, gt_jobs[match_idx]))
        else:
            unmatched_jobs.append(job)
    unmatched_gt = [gt for i, gt in enumerate(gt_jobs) if not gt_used[i]]
    return matched_pairs, unmatched_jobs, unmatched_gt


def _is_001_project_fabrication(job):
    company = str(job.get("company") or "")
    return any(
        proj.lower() in company.lower() or company.lower() in proj.lower()
        for proj in KNOWN_PROJECT_NAMES_001
    )


def _field_match(job, gt, field):
    jv = job.get(field)
    gv = gt.get(field)
    if gv is None:
        return None  # not evaluable
    if jv is None:
        return False
    return str(jv).strip().lower() == str(gv).strip().lower()


def main():
    rows = []
    for rid in POPULATION:
        for condition in ("baseline", "entity_first"):
            for run in range(1, 4):
                d = _load(rid, condition, run)
                jobs = (d.get("json_object") or {}).get("work_experience", []) if d.get("json_object") else []
                gt_jobs = GT_JOBS[rid]

                matched_pairs, unmatched_jobs, unmatched_gt = _match_jobs_to_gt(rid, jobs)

                precision = len(matched_pairs) / len(jobs) if jobs else None
                recall = len(matched_pairs) / len(gt_jobs) if gt_jobs else None
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if precision is not None and recall is not None and (precision + recall) > 0
                    else None
                )

                n_001_fabrications = (
                    sum(1 for j in unmatched_jobs if _is_001_project_fabrication(j)) if rid == "001" else None
                )

                designation_matches = sum(
                    1 for job, gt in matched_pairs if _field_match(job, gt, "designation") is True
                )
                designation_evaluable = sum(
                    1 for job, gt in matched_pairs if _field_match(job, gt, "designation") is not None
                )
                start_date_matches = sum(
                    1 for job, gt in matched_pairs if _field_match(job, gt, "start_date") is True
                )
                start_date_evaluable = sum(
                    1 for job, gt in matched_pairs if _field_match(job, gt, "start_date") is not None
                )

                ef_extra = {}
                if condition == "entity_first":
                    ef_extra = {
                        "n_calls": d.get("n_calls"),
                        "stage_a_n_entities": len(d["stage_a_entities"]) if d.get("stage_a_entities") else None,
                        "entity_fixation_violation": d.get("stage_b_entity_fixation_violation"),
                    }

                rows.append({
                    "resume_id": rid, "condition": condition, "run": run,
                    "status": d.get("status"), "elapsed_s": d.get("elapsed_s"),
                    "n_calls": d.get("n_calls"),
                    "n_jobs": len(jobs), "gt_n_jobs": len(gt_jobs),
                    "n_matched": len(matched_pairs), "n_unmatched_extra": len(unmatched_jobs),
                    "n_missing_gt": len(unmatched_gt),
                    "precision": precision, "recall": recall, "f1": f1,
                    "n_001_project_fabrications": n_001_fabrications,
                    "designation_match_rate": (
                        designation_matches / designation_evaluable if designation_evaluable else None
                    ),
                    "start_date_match_rate": (
                        start_date_matches / start_date_evaluable if start_date_evaluable else None
                    ),
                    **ef_extra,
                })

    out = Path("benchmark/baselines/t13_pilot_results_full.csv")
    fieldnames = [
        "resume_id", "condition", "run", "status", "elapsed_s", "n_calls", "n_jobs", "gt_n_jobs",
        "n_matched", "n_unmatched_extra", "n_missing_gt", "precision", "recall", "f1",
        "n_001_project_fabrications", "designation_match_rate", "start_date_match_rate",
        "stage_a_n_entities", "entity_fixation_violation",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"wrote {out} ({len(rows)} rows)")

    from collections import defaultdict
    summary = defaultdict(list)
    for r in rows:
        summary[(r["resume_id"], r["condition"])].append(r)

    print("\n=== SUMMARY ===")
    for (rid, cond), rs in sorted(summary.items()):
        prec = [r["precision"] for r in rs if r["precision"] is not None]
        rec = [r["recall"] for r in rs if r["recall"] is not None]
        extra = [r["n_unmatched_extra"] for r in rs]
        missing = [r["n_missing_gt"] for r in rs]
        elapsed = [r["elapsed_s"] for r in rs if r["elapsed_s"]]
        print(f"{rid} {cond}: n_jobs={[r['n_jobs'] for r in rs]} gt={rs[0]['gt_n_jobs']} "
              f"precision={[round(p,2) for p in prec]} recall={[round(r,2) for r in rec]} "
              f"extra={extra} missing={missing} elapsed={[round(e,1) for e in elapsed]}")


if __name__ == "__main__":
    main()
