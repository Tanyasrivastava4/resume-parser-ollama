"""
scripts/_milestone_t15_analysis.py

Milestone T15/Phase 2C -- consolidates all real-call checkpoints,
reuses T14's own entity-matching convention unmodified, and builds
the required resume-021 entity-by-entity matrix plus the pre-
registered failure-classification precedence.
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
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/t15_checkpoints"
)
GT_DIR = Path("benchmark/ground_truth/adjudicated_v1")

POPULATION = {
    "021": ("resume021.json", 10),
    "001": ("resume001.json", 3),
    "013": ("resume013.json", 3),
}

KNOWN_PROJECT_NAMES_001 = ["SoWhat", "TGES Travel & Transport Management", "Asha Learningy", "Adiroha"]

GT_JOBS = {
    rid: json.loads((GT_DIR / f).read_text()).get("work_experience", [])
    for rid, (f, _n) in POPULATION.items()
}


def _load(rid, condition, run):
    with (CKPT_DIR / f"{rid}__{condition}_{run}.pkl").open("rb") as f:
        return pickle.load(f)


def _match_jobs_to_gt(rid, jobs):
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

    split_gt_companies = set()
    for gt in gt_jobs:
        claims = sum(
            1 for job in jobs
            if str(job.get("company") or "") and _fuzzy_contains(str(job.get("company") or ""), gt.get("company") or "")
        )
        if claims >= 2:
            split_gt_companies.add(gt.get("company"))

    merge_jobs = []
    for job in jobs:
        company = str(job.get("company") or "")
        if not company:
            continue
        matches = [gt.get("company") for gt in gt_jobs if _fuzzy_contains(company, gt.get("company") or "")]
        if len(matches) >= 2:
            merge_jobs.append((company, matches))

    unmatched_gt = [gt for i, gt in enumerate(gt_jobs) if not gt_used[i]]
    matched_gt_indices = {i for i, used in enumerate(gt_used) if used}
    return matched_pairs, unmatched_jobs, unmatched_gt, split_gt_companies, merge_jobs, matched_gt_indices


def _is_001_project_fabrication(job):
    company = str(job.get("company") or "")
    return any(
        proj.lower() in company.lower() or company.lower() in proj.lower()
        for proj in KNOWN_PROJECT_NAMES_001
    )


def _classify_run(d, jobs, split_gt, merge_jobs, unmatched_jobs, recall):
    """Precedence-ordered classification, per the frozen pre-registration."""
    status = d.get("status")
    if status == "stage_a_parse_failed":
        return "stage_a_structural_failure"
    if status == "stage_b_parse_failed":
        return "stage_b_structural_failure"
    if status == "schema_failed":
        return "schema_failure"
    if split_gt:
        return "entity_split"
    if merge_jobs:
        return "entity_merge"
    if unmatched_jobs:
        return "entity_fabrication"
    if recall is not None and recall < 1.0:
        return "partial_entity_recovery"
    return "successful_complete_entity_structure"


def main():
    rows = []
    matrix_021 = {}  # {run_label: [bool x8]}

    for rid, (_gt_file, n_repeats) in POPULATION.items():
        gt_jobs = GT_JOBS[rid]
        for condition in ("baseline", "entity_first"):
            for run in range(1, n_repeats + 1):
                d = _load(rid, condition, run)
                jobs = (d.get("json_object") or {}).get("work_experience", []) if d.get("json_object") else []

                matched_pairs, unmatched_jobs, unmatched_gt, split_gt, merge_jobs, matched_idx = (
                    _match_jobs_to_gt(rid, jobs)
                )

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
                classification = _classify_run(d, jobs, split_gt, merge_jobs, unmatched_jobs, recall)

                if rid == "021":
                    label = f"{condition}_{run}"
                    matrix_021[label] = [i in matched_idx for i in range(len(gt_jobs))]

                ef_extra = {}
                if condition == "entity_first":
                    ef_extra = {
                        "n_calls": d.get("n_calls"),
                        "stage_a_elapsed_s": d.get("stage_a_elapsed_s"),
                        "stage_b_elapsed_s": d.get("stage_b_elapsed_s"),
                        "stage_a_n_entities": len(d["stage_a_entities"]) if d.get("stage_a_entities") else None,
                        "entity_fixation_violation": d.get("stage_b_entity_fixation_violation"),
                    }

                rows.append({
                    "resume_id": rid, "condition": condition, "run": run,
                    "status": d.get("status"), "classification": classification,
                    "elapsed_s": d.get("elapsed_s"), "n_calls": d.get("n_calls"),
                    "n_jobs": len(jobs), "gt_n_jobs": len(gt_jobs),
                    "n_matched": len(matched_pairs), "n_unmatched_extra": len(unmatched_jobs),
                    "n_missing_gt": len(unmatched_gt),
                    "missing_gt_companies": "; ".join(g.get("company") for g in unmatched_gt),
                    "precision": precision, "recall": recall, "f1": f1,
                    "n_001_project_fabrications": n_001_fabrications,
                    "split_gt_companies": "; ".join(split_gt) if split_gt else "",
                    "merge_jobs": str(merge_jobs) if merge_jobs else "",
                    **ef_extra,
                })

    out = Path("benchmark/baselines/t15_pilot_results_full.csv")
    fieldnames = [
        "resume_id", "condition", "run", "status", "classification", "elapsed_s", "n_calls",
        "n_jobs", "gt_n_jobs", "n_matched", "n_unmatched_extra", "n_missing_gt", "missing_gt_companies",
        "precision", "recall", "f1", "n_001_project_fabrications",
        "split_gt_companies", "merge_jobs",
        "stage_a_elapsed_s", "stage_b_elapsed_s", "stage_a_n_entities", "entity_fixation_violation",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"wrote {out} ({len(rows)} rows)")

    # Resume 021 entity-by-entity matrix
    gt_021 = GT_JOBS["021"]
    matrix_out = Path("benchmark/baselines/t15_021_entity_matrix.csv")
    run_labels = (
        [f"baseline_{i}" for i in range(1, 11)] + [f"entity_first_{i}" for i in range(1, 11)]
    )
    with matrix_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gt_entity"] + run_labels)
        for i, gt in enumerate(gt_021):
            row = [gt.get("company")]
            for label in run_labels:
                row.append("Y" if matrix_021.get(label, [False] * 8)[i] else "N")
            w.writerow(row)
    print(f"wrote {matrix_out}")

    from collections import defaultdict
    summary = defaultdict(list)
    for r in rows:
        summary[(r["resume_id"], r["condition"])].append(r)

    print("\n=== SUMMARY ===")
    for (rid, cond), rs in sorted(summary.items()):
        rec = [r["recall"] for r in rs if r["recall"] is not None]
        prec = [r["precision"] for r in rs if r["precision"] is not None]
        classes = [r["classification"] for r in rs]
        print(f"{rid} {cond}: n={len(rs)} recall={[round(r,2) for r in rec]} "
              f"precision={[round(p,2) for p in prec]}")
        print(f"    classes={classes}")

    print("\n=== 021 ENTITY MATRIX ===")
    for i, gt in enumerate(gt_021):
        b = sum(1 for lbl in [f"baseline_{i}" for i in range(1,11)] if matrix_021.get(lbl,[False]*8)[i])
        ef = sum(1 for lbl in [f"entity_first_{i}" for i in range(1,11)] if matrix_021.get(lbl,[False]*8)[i])
        print(f"  {gt.get('company')}: baseline {b}/10, entity_first {ef}/10")


if __name__ == "__main__":
    main()
