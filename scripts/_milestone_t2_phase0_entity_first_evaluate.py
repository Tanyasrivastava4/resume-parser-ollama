"""
scripts/_milestone_t2_phase0_entity_first_evaluate.py

Milestone T2 / Phase 0 -- Phase C evaluation. Reads the 16 real
checkpoints from scripts/_milestone_t2_phase0_entity_first_prototype.py
(8 baseline, 8 entity-first: 4 resumes x 2 conditions x 2 runs, all
real Ollama calls already made) and scores each against real GT using
the same entity-alignment logic as Phase A
(scripts/_milestone_t2_phase0_failure_taxonomy.py), so Phase A and
Phase C numbers are directly comparable. ZERO new LLM calls.

Writes: benchmark/baselines/t2_phase0_entity_first_results.csv
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

from scripts._milestone_t2_phase0_failure_taxonomy import _align_entities, _dates_equal, _fuzzy_contains  # noqa: E402

CKPT_DIR = Path(
    "/private/tmp/claude-501/-Users-ruchigupta-Documents-ResumeAIBenchmark/"
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/t2_phase0_checkpoints"
)
GT_DIR = Path("benchmark/ground_truth/adjudicated_v1")
GT_FILES = {"021": "resume021.json", "001": "resume001.json", "016": "resume016.json", "013": "resume013.json"}


def score(gt_list: list[dict], cand_list: list[dict]) -> dict:
    matched, unmatched_gt, unmatched_cand = _align_entities(gt_list, cand_list, "company", "company")
    n_title_match = sum(1 for g, c in matched if _fuzzy_contains(g.get("designation", ""), c.get("designation", "")))
    gt_has_start = [(g, c) for g, c in matched if g.get("start_date")]
    n_start_match = sum(1 for g, c in gt_has_start if _dates_equal(g.get("start_date"), c.get("start_date")))
    return {
        "gt_n_jobs": len(gt_list), "cand_n_jobs": len(cand_list),
        "n_matched": len(matched), "n_unmatched_gt": len(unmatched_gt), "n_unmatched_cand": len(unmatched_cand),
        "entity_count_exact_match": len(gt_list) == len(cand_list),
        "entity_recall": round(len(matched) / len(gt_list), 3) if gt_list else None,
        "entity_precision": round(len(matched) / len(cand_list), 3) if cand_list else None,
        "title_match_rate": round(n_title_match / len(matched), 3) if matched else None,
        "start_date_match_rate": round(n_start_match / len(gt_has_start), 3) if gt_has_start else None,
        "unmatched_gt_companies": "; ".join(g.get("company") or "" for g in unmatched_gt),
        "unmatched_cand_companies": "; ".join(c.get("company") or "" for c in unmatched_cand),
    }


def main() -> None:
    rows = []
    for rid, gt_file in GT_FILES.items():
        gt = json.loads((GT_DIR / gt_file).read_text())
        gt_list = gt.get("work_experience") or []

        for condition in ("baseline", "entity_first"):
            for run in (1, 2):
                ckpt_path = CKPT_DIR / f"{rid}__{condition}__run{run}.pkl"
                if not ckpt_path.exists():
                    continue
                with ckpt_path.open("rb") as f:
                    d = pickle.load(f)
                r = d["result"]
                row = {
                    "resume_id": rid, "condition": condition, "run_index": run,
                    "call_status": r.get("status"), "n_calls": r.get("n_calls"),
                    "elapsed_seconds": round(r.get("elapsed_seconds", 0), 1),
                }
                if r.get("status") == "ok":
                    cand_list = r["json_object"].get("work_experience") or []
                    row.update(score(gt_list, cand_list))
                else:
                    row.update({"gt_n_jobs": len(gt_list), "cand_n_jobs": None, "n_matched": None,
                                 "entity_count_exact_match": None, "entity_recall": None,
                                 "entity_precision": None, "title_match_rate": None,
                                 "start_date_match_rate": None, "unmatched_gt_companies": None,
                                 "unmatched_cand_companies": None,
                                 "error_type": r.get("error_type"), "error": r.get("error")})
                rows.append(row)

    out = Path("benchmark/baselines/t2_phase0_entity_first_results.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")

    print("\n=== Per resume, per condition summary ===")
    for rid in GT_FILES:
        for condition in ("baseline", "entity_first"):
            sub = [r for r in rows if r["resume_id"] == rid and r["condition"] == condition and r["call_status"] == "ok"]
            if not sub:
                print(f"{rid} {condition}: no successful runs")
                continue
            recalls = [r["entity_recall"] for r in sub if r["entity_recall"] is not None]
            counts = [r["cand_n_jobs"] for r in sub]
            print(f"{rid} {condition}: n_runs_ok={len(sub)}/2, gt_n={sub[0]['gt_n_jobs']}, "
                  f"cand_n={counts}, entity_recall={recalls}, "
                  f"mean_elapsed={sum(r['elapsed_seconds'] for r in sub)/len(sub):.1f}s")

    print("\n=== Call accounting ===")
    n_calls_baseline = sum(r["n_calls"] or 0 for r in rows if r["condition"] == "baseline")
    n_calls_ef = sum(r["n_calls"] or 0 for r in rows if r["condition"] == "entity_first")
    print(f"Baseline real calls: {n_calls_baseline}")
    print(f"Entity-first real calls: {n_calls_ef}")
    print(f"Total real calls this experiment: {n_calls_baseline + n_calls_ef}")


if __name__ == "__main__":
    main()
