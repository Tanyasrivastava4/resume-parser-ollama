"""
scripts/_milestone_v6_1_evaluate_pilot.py

Milestone V6.1 Phase 2 -- evaluation of the 50 real pilot results
against real GT, using the DUAL metric methodology frozen in V6.0:
  1. exact-match recall/precision (historical-comparability metric)
  2. fuzzy/normalized recall + content_overlap, via the EXISTING,
     UNMODIFIED evaluate() function from
     scripts/_milestone_v5_8_offline_evaluation.py (the decision-
     grounding metric per V6.0 Phase 3A).

No LLM calls. Reads only the 50 real checkpoint pickles and real GT.
Writes:
  benchmark/baselines/v6_1_per_run_results.csv
  benchmark/baselines/v6_1_resume_condition_summary.csv
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

from scripts._milestone_v5_8_offline_evaluation import evaluate  # noqa: E402 -- REUSED UNCHANGED

CHECKPOINT_DIR = Path(
    "/private/tmp/claude-501/-Users-ruchigupta-Documents-ResumeAIBenchmark/"
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/v6_1_checkpoints"
)
GT_DIR = Path("benchmark/ground_truth/adjudicated_v1")
OUT_DIR = Path("benchmark/baselines")

GT_FILES = {
    "007": "resume007.json", "013": "resume013.json", "015": "resume015.json",
    "016": "resume016.json", "018": "resume018.json",
}


def exact_match_score(extracted: list, gt: list) -> dict:
    """Historical-comparability metric -- identical semantics to
    validators/json_comparator.py's array comparison philosophy
    (case-sensitive, whitespace-sensitive exact string match), applied
    per-item here (not just whole-array pass/fail) to get recall/
    precision, since json_comparator's own _compare_arrays only
    reports a binary PASS/FAIL for the whole array, not per-item
    recall -- this per-item version is necessary to report the same
    kind of number V3.0-V5.9 have always reported for Skills."""
    gt_norm = set(g.strip() for g in gt if isinstance(g, str))
    ext_norm = set(e.strip() for e in extracted if isinstance(e, str))
    if not gt_norm:
        return {"exact_recall": None, "exact_precision": None, "exact_matched": 0}
    matched = len(gt_norm & ext_norm)
    recall = matched / len(gt_norm)
    precision = (matched / len(ext_norm)) if ext_norm else None
    return {"exact_recall": round(recall, 4), "exact_precision": round(precision, 4) if precision is not None else None,
            "exact_matched": matched}


def main() -> None:
    per_run_rows = []
    for rid, gt_file in GT_FILES.items():
        gt = json.loads((GT_DIR / gt_file).read_text(encoding="utf-8")).get("skills", [])
        for condition in ("baseline", "candidate"):
            for run_idx in range(1, 6):
                path = CHECKPOINT_DIR / f"{rid}__{condition}__run{run_idx}.pkl"
                if not path.exists():
                    print(f"MISSING CHECKPOINT: {rid} {condition} run{run_idx}")
                    continue
                with path.open("rb") as f:
                    slot = pickle.load(f)
                if slot["status"] != "ok":
                    per_run_rows.append({
                        "resume_id": rid, "condition": condition, "model": slot["model"],
                        "run_index": run_idx, "status": slot["status"], "n_attempts": slot["n_attempts"],
                        "extracted_count": "", "gt_count": len(gt),
                        "exact_recall": "", "exact_precision": "",
                        "fuzzy_recall": "", "content_overlap": "", "n_fuzzy_missing": "",
                        "elapsed_seconds": "",
                    })
                    continue
                extracted = slot["final"]["skills"] or []
                elapsed = slot["final"]["elapsed_seconds"]
                em = exact_match_score(extracted, gt)
                ev = evaluate(extracted, gt)  # reused, unmodified -- fuzzy_recall, content_overlap, etc.
                per_run_rows.append({
                    "resume_id": rid, "condition": condition, "model": slot["model"],
                    "run_index": run_idx, "status": "ok", "n_attempts": slot["n_attempts"],
                    "extracted_count": len(extracted), "gt_count": len(gt),
                    "exact_recall": em["exact_recall"], "exact_precision": em["exact_precision"],
                    "fuzzy_recall": ev["fuzzy_recall"], "content_overlap": ev["content_overlap"],
                    "n_fuzzy_missing": ev["n_fuzzy_missing"],
                    "elapsed_seconds": elapsed,
                })

    with (OUT_DIR / "v6_1_per_run_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_run_rows[0].keys()))
        w.writeheader()
        w.writerows(per_run_rows)
    print(f"wrote v6_1_per_run_results.csv ({len(per_run_rows)} rows)")

    # --- resume x condition summary ---
    import statistics
    summary_rows = []
    for rid in GT_FILES:
        for condition in ("baseline", "candidate"):
            rows = [r for r in per_run_rows if r["resume_id"] == rid and r["condition"] == condition and r["status"] == "ok"]
            if not rows:
                continue
            exact_recalls = [r["exact_recall"] for r in rows if r["exact_recall"] != ""]
            fuzzy_recalls = [r["fuzzy_recall"] for r in rows if r["fuzzy_recall"] != ""]
            content_overlaps = [r["content_overlap"] for r in rows if r["content_overlap"] != ""]
            elapsed = [r["elapsed_seconds"] for r in rows if r["elapsed_seconds"] != ""]
            summary_rows.append({
                "resume_id": rid, "condition": condition, "model": rows[0]["model"], "n_runs": len(rows),
                "exact_recall_mean": round(statistics.mean(exact_recalls), 4),
                "exact_recall_min": round(min(exact_recalls), 4), "exact_recall_max": round(max(exact_recalls), 4),
                "fuzzy_recall_mean": round(statistics.mean(fuzzy_recalls), 4),
                "fuzzy_recall_min": round(min(fuzzy_recalls), 4), "fuzzy_recall_max": round(max(fuzzy_recalls), 4),
                "content_overlap_mean": round(statistics.mean(content_overlaps), 4),
                "content_overlap_min": round(min(content_overlaps), 4), "content_overlap_max": round(max(content_overlaps), 4),
                "elapsed_median": round(statistics.median(elapsed), 3),
                "elapsed_mean": round(statistics.mean(elapsed), 3),
                "elapsed_min": round(min(elapsed), 3), "elapsed_max": round(max(elapsed), 3),
            })
    with (OUT_DIR / "v6_1_resume_condition_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"wrote v6_1_resume_condition_summary.csv ({len(summary_rows)} rows)")

    print("\n=== SUMMARY (fuzzy_recall mean, baseline -> candidate) ===")
    for rid in GT_FILES:
        b = next((r for r in summary_rows if r["resume_id"] == rid and r["condition"] == "baseline"), None)
        c = next((r for r in summary_rows if r["resume_id"] == rid and r["condition"] == "candidate"), None)
        if b and c:
            print(f"{rid}: fuzzy {b['fuzzy_recall_mean']} [{b['fuzzy_recall_min']}-{b['fuzzy_recall_max']}] -> "
                  f"{c['fuzzy_recall_mean']} [{c['fuzzy_recall_min']}-{c['fuzzy_recall_max']}]  "
                  f"non_overlap={c['fuzzy_recall_min'] > b['fuzzy_recall_max']}")


if __name__ == "__main__":
    main()
