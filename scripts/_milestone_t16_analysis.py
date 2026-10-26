"""
scripts/_milestone_t16_analysis.py

Milestone T16/Phase 2C -- consolidates all real-call checkpoints,
reuses the established entity-matching convention (T13/T14/T15)
unmodified, and applies the frozen T16 failure-classification
precedence (CASE 1-4 retry policy outcomes).
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
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/t16_checkpoints"
)
GT_DIR = Path("benchmark/ground_truth/adjudicated_v1")

GT_FILES = {"021": "resume021.json", "013": "resume013.json"}
RUN_COUNTS = {"021": 10, "013": 5}
TAIL_ENTITIES_021 = ["Hazel Infotech Ltd.", "SVN Technologies", "Kalyani Informix"]

GT_JOBS = {rid: json.loads((GT_DIR / f).read_text()).get("work_experience", []) for rid, f in GT_FILES.items()}


def _load(run_id):
    with (CKPT_DIR / f"{run_id}.pkl").open("rb") as f:
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

    split_gt = set()
    for gt in gt_jobs:
        claims = sum(
            1 for job in jobs
            if str(job.get("company") or "") and _fuzzy_contains(str(job.get("company") or ""), gt.get("company") or "")
        )
        if claims >= 2:
            split_gt.add(gt.get("company"))

    merge_jobs = []
    for job in jobs:
        company = str(job.get("company") or "")
        if not company:
            continue
        matches = [gt.get("company") for gt in gt_jobs if _fuzzy_contains(company, gt.get("company") or "")]
        if len(matches) >= 2:
            merge_jobs.append((company, matches))

    matched_idx = {i for i, used in enumerate(gt_used) if used}
    return matched_pairs, unmatched_jobs, matched_idx, split_gt, merge_jobs


def _final_classification(record, jobs, matched_idx, split_gt, merge_jobs, unmatched_jobs, recall):
    """Applies the frozen T16 precedence: retry-terminal failures first,
    then structural (split/merge/fabrication), then content-vs-structural
    success distinction."""
    if record.get("primary_failure_class"):
        return record["primary_failure_class"]  # STAGE_A_RETRY_FAILURE / STAGE_B_RETRY_FAILURE / ENTITY_FIXATION_FAILURE
    if split_gt:
        return "ENTITY_SPLIT"
    if merge_jobs:
        return "ENTITY_MERGE"
    if unmatched_jobs:
        return "ENTITY_FABRICATION"
    if recall is not None and recall < 1.0:
        return "CONTENT_FAILURE_AFTER_STRUCTURAL_SUCCESS"
    return "SUCCESS"


def main():
    rows = []
    matrix_021 = {}

    for rid, n in RUN_COUNTS.items():
        gt_jobs = GT_JOBS[rid]
        for run_idx in range(1, n + 1):
            run_id = f"{rid}_{run_idx}"
            record = _load(run_id)
            jobs = (
                record["final_json_object"].get("work_experience", [])
                if record.get("final_json_object") else []
            )

            matched_pairs, unmatched_jobs, matched_idx, split_gt, merge_jobs = _match_jobs_to_gt(rid, jobs)
            precision = len(matched_pairs) / len(jobs) if jobs else None
            recall = len(matched_pairs) / len(gt_jobs) if gt_jobs else None

            classification = _final_classification(
                record, jobs, matched_idx, split_gt, merge_jobs, unmatched_jobs, recall
            )

            if rid == "021":
                matrix_021[run_id] = [i in matched_idx for i in range(len(gt_jobs))]
                tail_indices = [i for i, gt in enumerate(gt_jobs) if gt.get("company") in TAIL_ENTITIES_021]
                tail_recovered = [i in matched_idx for i in tail_indices]
            else:
                tail_recovered = None

            stage_a_retried = record.get("stage_a_retry") is not None
            stage_b_retried = record.get("stage_b_retry") is not None
            rescued_by_retry = (
                classification == "SUCCESS" and (stage_a_retried or stage_b_retried)
            )

            rows.append({
                "run_id": run_id, "resume_id": rid, "run": run_idx,
                "total_calls": record["total_calls"], "total_elapsed_s": round(record["total_elapsed_s"], 1),
                "stage_a_retried": stage_a_retried, "stage_b_retried": stage_b_retried,
                "n_jobs": len(jobs), "gt_n_jobs": len(gt_jobs),
                "n_matched": len(matched_pairs), "n_unmatched_extra": len(unmatched_jobs),
                "precision": precision, "recall": recall,
                "full_recall": (recall == 1.0) if recall is not None else False,
                "fabrication": bool(unmatched_jobs), "splitting": bool(split_gt), "merging": bool(merge_jobs),
                "tail_all_recovered": (all(tail_recovered) if tail_recovered else None),
                "tail_recovery_detail": (str(tail_recovered) if tail_recovered is not None else ""),
                "classification": classification,
                "rescued_by_retry": rescued_by_retry,
            })

    out = Path("benchmark/baselines/t16_pilot_results_full.csv")
    fieldnames = list(rows[0].keys())
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")

    # 021 entity matrix
    matrix_out = Path("benchmark/baselines/t16_021_entity_matrix.csv")
    run_labels = [f"021_{i}" for i in range(1, 11)]
    with matrix_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gt_entity"] + run_labels)
        for i, gt in enumerate(GT_JOBS["021"]):
            row = [gt.get("company")] + ["Y" if matrix_021[lbl][i] else "N" for lbl in run_labels]
            w.writerow(row)
    print(f"wrote {matrix_out}")

    print("\n=== SUMMARY ===")
    for r in rows:
        print(r["run_id"], "calls=", r["total_calls"], "recall=", r["recall"],
              "class=", r["classification"], "rescued=", r["rescued_by_retry"])

    print("\n=== 021 TAIL ENTITY RECOVERY ===")
    for i, ent in enumerate(TAIL_ENTITIES_021):
        count = sum(1 for lbl in run_labels if matrix_021[lbl][GT_JOBS["021"].index(
            next(g for g in GT_JOBS["021"] if g.get("company") == ent)
        )])
        print(f"  {ent}: {count}/10")


if __name__ == "__main__":
    main()
