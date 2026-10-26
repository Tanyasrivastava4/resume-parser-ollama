"""
scripts/_milestone_t14_analysis.py

Milestone T14/Phase 2C -- consolidates all real-call checkpoints,
reusing T13's own entity-matching convention (_fuzzy_contains-based
greedy matching) unmodified, plus split/merge/fabrication
classification per the frozen T14 pre-registration.
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
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/t14_checkpoints"
)
GT_DIR = Path("benchmark/ground_truth/adjudicated_v1")

POPULATION = {
    "001": "resume001.json",
    "013": "resume013.json",
    "021": "resume021.json",
}

KNOWN_PROJECT_NAMES_001 = ["SoWhat", "TGES Travel & Transport Management", "Asha Learningy", "Adiroha"]

GT_JOBS = {
    rid: json.loads((GT_DIR / f).read_text()).get("work_experience", [])
    for rid, f in POPULATION.items()
}


def _load(rid, condition, run):
    with (CKPT_DIR / f"{rid}__{condition}_{run}.pkl").open("rb") as f:
        return pickle.load(f)


def _match_jobs_to_gt(rid, jobs):
    """Greedy fuzzy matching -- byte-identical convention to T13's own
    _milestone_t13_analysis.py. Also detects split (one GT company
    claimed by 2+ returned jobs) and merge (one returned job matching
    2+ distinct GT companies) per the frozen T14 protocol."""
    gt_jobs = GT_JOBS[rid]
    gt_used = [False] * len(gt_jobs)
    matched_pairs = []
    unmatched_jobs = []
    split_gt_companies = set()
    gt_claim_count = [0] * len(gt_jobs)

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
            gt_claim_count[match_idx] += 1
            matched_pairs.append((job, gt_jobs[match_idx]))
        else:
            unmatched_jobs.append(job)

    # Second pass (unused-GT-blind to first-match greediness): detect
    # splits by checking, for each GT company, how many jobs in the
    # FULL returned list fuzzy-match it (not just the first).
    for i, gt in enumerate(gt_jobs):
        claims = sum(
            1 for job in jobs
            if str(job.get("company") or "") and _fuzzy_contains(str(job.get("company") or ""), gt.get("company") or "")
        )
        if claims >= 2:
            split_gt_companies.add(gt.get("company"))

    # Merge detection: a single returned job whose company fuzzy-matches 2+ distinct GT companies.
    merge_jobs = []
    for job in jobs:
        company = str(job.get("company") or "")
        if not company:
            continue
        matches = [gt.get("company") for gt in gt_jobs if _fuzzy_contains(company, gt.get("company") or "")]
        if len(matches) >= 2:
            merge_jobs.append((company, matches))

    unmatched_gt = [gt for i, gt in enumerate(gt_jobs) if not gt_used[i]]
    return matched_pairs, unmatched_jobs, unmatched_gt, split_gt_companies, merge_jobs


def _is_001_project_fabrication(job):
    company = str(job.get("company") or "")
    return any(
        proj.lower() in company.lower() or company.lower() in proj.lower()
        for proj in KNOWN_PROJECT_NAMES_001
    )


def main():
    rows = []
    for rid in POPULATION:
        for condition in ("baseline", "entity_first"):
            for run in range(1, 6):
                d = _load(rid, condition, run)
                jobs = (d.get("json_object") or {}).get("work_experience", []) if d.get("json_object") else []
                gt_jobs = GT_JOBS[rid]

                matched_pairs, unmatched_jobs, unmatched_gt, split_gt, merge_jobs = _match_jobs_to_gt(rid, jobs)

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
                    "status": d.get("status"), "elapsed_s": d.get("elapsed_s"),
                    "n_calls": d.get("n_calls"),
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

    out = Path("benchmark/baselines/t14_pilot_results_full.csv")
    fieldnames = [
        "resume_id", "condition", "run", "status", "elapsed_s", "n_calls", "n_jobs", "gt_n_jobs",
        "n_matched", "n_unmatched_extra", "n_missing_gt", "missing_gt_companies",
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

    from collections import defaultdict
    summary = defaultdict(list)
    for r in rows:
        summary[(r["resume_id"], r["condition"])].append(r)

    print("\n=== SUMMARY ===")
    for (rid, cond), rs in sorted(summary.items()):
        rec = [r["recall"] for r in rs if r["recall"] is not None]
        prec = [r["precision"] for r in rs if r["precision"] is not None]
        elapsed = [r["elapsed_s"] for r in rs if r["elapsed_s"]]
        statuses = [r["status"] for r in rs]
        print(f"{rid} {cond}: status={statuses} n_jobs={[r['n_jobs'] for r in rs]} gt={rs[0]['gt_n_jobs']} "
              f"precision={[round(p,2) for p in prec]} recall={[round(r,2) for r in rec]} "
              f"elapsed={[round(e,1) for e in elapsed]}")


if __name__ == "__main__":
    main()
