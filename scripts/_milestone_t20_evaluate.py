"""
scripts/_milestone_t20_evaluate.py

Milestone T20/Phase 2G -- paired before(T18)/after(T20) evaluation.

Baseline = T18's own generate_candidates_v2, completely unmodified.
Candidate = T20's generate_candidates_t20 (this milestone's one
refinement), from scripts/_milestone_t20_candidates.py.

Both conditions reuse T17's own blockify(), containing_section_labels(),
_candidate_covers_company(), and every metric definition unchanged --
no metric is redefined for T20.

Two populations, both mandatory:
  A. T19's 5 unseen resumes, using T19's own ground truth
     (reports/t19_unseen_candidate_ground_truth.csv) -- NOT modified,
     only read.
  B. The 4 mandatory historical controls (001, 006, 013, 021), using
     the existing 22-resume benchmark GT
     (benchmark/ground_truth/adjudicated_v1/).

No LLM calls. No rule changes after seeing results.
"""
from __future__ import annotations

import csv
import sys
import json
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_t18_blocks import blockify  # noqa: E402 (re-exported, unchanged, from T17)
from scripts._milestone_t18_candidates import generate_candidates_v2 as generate_baseline  # noqa: E402
from scripts._milestone_t20_candidates import generate_candidates_t20 as generate_candidate  # noqa: E402
from scripts._milestone_t17_evaluate import _candidate_covers_company  # noqa: E402

MARKDOWN_DIR = Path("output/markdown")
T19_GT_CSV = Path("reports/t19_unseen_candidate_ground_truth.csv")
BENCHMARK_GT_DIR = Path("benchmark/ground_truth/adjudicated_v1")
BENCHMARK_MAPPING = BENCHMARK_GT_DIR / "mapping.csv"

UNSEEN_RESUMES = [
    ("U1", "178480991686_Vachan_Premugh_Dicruz_Founder_Office_BTTR.MEN"),
    ("U2", "8th_April_Akasa_Kapoor_Resume"),
    ("U3", "AARTI BAGHOTIYA_Economic Urban Planner_Intileo"),
    ("U4", "AASHISH PARMAR - Email Marketing Executive - Intileo"),
    ("U5", "_Adarsh_Bhagat Resume"),
]
CONTROL_IDS = {"001", "006", "013", "021"}


def _load_t19_gt():
    by_id = defaultdict(list)
    with T19_GT_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_id[row["unseen_id"]].append(row)
    return by_id


def _load_control_population():
    with BENCHMARK_MAPPING.open(encoding="utf-8") as f:
        rows = {r["ResumeID"]: r for r in csv.DictReader(f)}
    return {rid: rows[rid] for rid in CONTROL_IDS}


def _companies_from_gt_rows(gt_rows):
    return [r["company"] for r in gt_rows]


def _companies_from_benchmark_json(gt_file):
    gt_jobs = json.loads((BENCHMARK_GT_DIR / gt_file).read_text(encoding="utf-8")).get("work_experience", [])
    return [g.get("company") or "" for g in gt_jobs]


def _evaluate(gen_fn, blocks, companies):
    candidates = gen_fn(blocks)
    coverage = {}
    for gi, company in enumerate(companies):
        covering = [c.candidate_id for c in candidates if _candidate_covers_company(c, blocks, company)]
        coverage[gi] = covering

    covered_gt = sum(1 for gi in coverage if coverage[gi])
    recall = covered_gt / len(companies) if companies else None

    cand_covers_gt = {c.candidate_id: [] for c in candidates}
    for gi, cids in coverage.items():
        for cid in cids:
            cand_covers_gt[cid].append(gi)

    n_spurious = sum(1 for c in candidates if not cand_covers_gt[c.candidate_id])
    precision = (len(candidates) - n_spurious) / len(candidates) if candidates else None

    over_segmented = [gi for gi, cids in coverage.items() if len(cids) >= 2]
    under_segmented = [cid for cid, gis in cand_covers_gt.items() if len(gis) >= 2]

    unsafe_candidates = [c for c in candidates if any(c.flags.values())]
    unsafe_rate = len(unsafe_candidates) / len(candidates) if candidates else None

    crosses_boundary = [c for c in candidates if c.flags.get("crosses_section_boundary")]
    covering_ids_excl_bleed = set()
    for gi, cids in coverage.items():
        for cid in cids:
            if cid not in {c.candidate_id for c in crosses_boundary}:
                covering_ids_excl_bleed.add(gi)
    bleed_dependent_gt = [gi for gi in coverage if coverage[gi] and gi not in covering_ids_excl_bleed]

    max_span = max((c.extent_end - c.extent_start for c in candidates), default=0)

    return {
        "gt_n_entities": len(companies), "n_candidates": len(candidates),
        "coverage_recall": recall, "covered_gt_count": covered_gt,
        "candidate_precision": precision, "n_spurious_candidates": n_spurious,
        "candidate_entity_ratio": (len(candidates) / len(companies)) if companies else None,
        "n_over_segmented_gt": len(over_segmented),
        "n_under_segmented_candidates": len(under_segmented),
        "n_unsafe_candidates": len(unsafe_candidates), "unsafe_rate": unsafe_rate,
        "n_crosses_section_boundary": len(crosses_boundary),
        "n_bleed_dependent_gt_coverage": len(bleed_dependent_gt),
        "max_candidate_span": max_span, "total_blocks": len(blocks),
        "missing_gt_companies": "; ".join(companies[gi] for gi in coverage if not coverage[gi]),
    }, candidates


def main() -> None:
    t19_gt = _load_t19_gt()
    controls = _load_control_population()

    rows = []

    for uid, stem in UNSEEN_RESUMES:
        blocks = blockify((MARKDOWN_DIR / f"{stem}.md").read_text(encoding="utf-8"))
        companies = _companies_from_gt_rows(t19_gt[uid])
        base_result, base_cands = _evaluate(generate_baseline, blocks, companies)
        cand_result, cand_cands = _evaluate(generate_candidate, blocks, companies)
        for label, result in (("baseline", base_result), ("candidate", cand_result)):
            row = {"population": "unseen", "resume_id": uid, "stem": stem, "condition": label}
            row.update(result)
            rows.append(row)
        print(f"{uid:4} baseline: cand={base_result['n_candidates']:3} recall={base_result['coverage_recall']} prec={base_result['candidate_precision']:.3f}   "
              f"candidate: cand={cand_result['n_candidates']:3} recall={cand_result['coverage_recall']} prec={cand_result['candidate_precision']:.3f}")

    for rid, row_meta in controls.items():
        stem = Path(row_meta["Markdown"]).stem
        gt_file = row_meta["JSON"]
        blocks = blockify((MARKDOWN_DIR / f"{stem}.md").read_text(encoding="utf-8"))
        companies = _companies_from_benchmark_json(gt_file)
        base_result, base_cands = _evaluate(generate_baseline, blocks, companies)
        cand_result, cand_cands = _evaluate(generate_candidate, blocks, companies)
        for label, result in (("baseline", base_result), ("candidate", cand_result)):
            row = {"population": "control", "resume_id": rid, "stem": stem, "condition": label}
            row.update(result)
            rows.append(row)
        print(f"{rid:4} baseline: cand={base_result['n_candidates']:3} recall={base_result['coverage_recall']} prec={base_result['candidate_precision']:.3f}   "
              f"candidate: cand={cand_result['n_candidates']:3} recall={cand_result['coverage_recall']} prec={cand_result['candidate_precision']:.3f}")

    out = Path("reports/t20_section_heading_metrics.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
