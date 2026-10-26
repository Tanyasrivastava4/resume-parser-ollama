"""
scripts/_milestone_t19_evaluate.py

Milestone T19/Phase 2F -- Phase D: compute unseen-batch metrics using
T17/T18's own, unmodified coverage-matching convention
(_candidate_covers_company, imported directly, not redefined) against
the manually-created ground truth in
reports/t19_unseen_candidate_ground_truth.csv.

Metric definitions are IDENTICAL to T17/T18's own
(coverage recall, candidate precision, over/under-segmentation,
crosses_section_boundary, unsafe rate, bleed-dependent coverage) --
none are redefined for this milestone.

No LLM calls. No rule changes.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_t18_blocks import blockify  # noqa: E402
from scripts._milestone_t18_candidates import generate_candidates_v2  # noqa: E402
from scripts._milestone_t17_evaluate import _candidate_covers_company  # noqa: E402

MARKDOWN_DIR = Path("output/markdown")
GT_CSV = Path("reports/t19_unseen_candidate_ground_truth.csv")

UNSEEN_RESUMES = [
    ("U1", "178480991686_Vachan_Premugh_Dicruz_Founder_Office_BTTR.MEN"),
    ("U2", "8th_April_Akasa_Kapoor_Resume"),
    ("U3", "AARTI BAGHOTIYA_Economic Urban Planner_Intileo"),
    ("U4", "AASHISH PARMAR - Email Marketing Executive - Intileo"),
    ("U5", "_Adarsh_Bhagat Resume"),
]


def _load_gt():
    by_resume = defaultdict(list)
    with GT_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_resume[row["unseen_id"]].append(row)
    return by_resume


def evaluate_one(uid: str, stem: str, gt_rows):
    markdown_text = (MARKDOWN_DIR / f"{stem}.md").read_text(encoding="utf-8")
    blocks = blockify(markdown_text)
    candidates = generate_candidates_v2(blocks)

    coverage = {}
    for gi, gt in enumerate(gt_rows):
        company = gt["company"]
        covering = [c.candidate_id for c in candidates if _candidate_covers_company(c, blocks, company)]
        coverage[gi] = covering

    covered_gt = sum(1 for gi in coverage if coverage[gi])
    recall = covered_gt / len(gt_rows) if gt_rows else None

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

    result = {
        "unseen_id": uid, "stem": stem,
        "gt_n_entities": len(gt_rows), "n_candidates": len(candidates),
        "coverage_recall": recall, "covered_gt_count": covered_gt,
        "candidate_precision": precision, "n_spurious_candidates": n_spurious,
        "candidate_entity_ratio": (len(candidates) / len(gt_rows)) if gt_rows else None,
        "n_over_segmented_gt": len(over_segmented),
        "n_under_segmented_candidates": len(under_segmented),
        "n_unsafe_candidates": len(unsafe_candidates), "unsafe_rate": unsafe_rate,
        "n_crosses_section_boundary": len(crosses_boundary),
        "n_bleed_dependent_gt_coverage": len(bleed_dependent_gt),
        "missing_gt_entities": "; ".join(
            f"{gt_rows[gi]['company']} / {gt_rows[gi]['title']}" for gi in coverage if not coverage[gi]
        ),
    }
    return result, blocks, candidates, coverage, gt_rows


def main() -> None:
    gt = _load_gt()
    rows = []
    print(f"{'ID':4} {'stem':50} {'gt':4} {'cand':6} {'recall':8} {'precision':10}")
    for uid, stem in UNSEEN_RESUMES:
        result, blocks, candidates, coverage, gt_rows = evaluate_one(uid, stem, gt[uid])
        rows.append(result)
        print(f"{uid:4} {stem[:50]:50} {result['gt_n_entities']:4} {result['n_candidates']:6} "
              f"{result['coverage_recall']!s:8} {result['candidate_precision']!s:10}")
        if result["missing_gt_entities"]:
            print(f"     MISSING: {result['missing_gt_entities']}")

    out = Path("reports/t19_unseen_candidate_metrics.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")

    def _agg(rs, key):
        vals = [r[key] for r in rs if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    print("\n=== AGGREGATE (5 unseen resumes) ===")
    print("total GT entities:", sum(r["gt_n_entities"] for r in rows))
    print("total candidates:", sum(r["n_candidates"] for r in rows))
    print("mean coverage_recall:", _agg(rows, "coverage_recall"))
    print("mean candidate_precision:", _agg(rows, "candidate_precision"))
    print("mean candidate_entity_ratio:", _agg(rows, "candidate_entity_ratio"))
    print("mean unsafe_rate:", _agg(rows, "unsafe_rate"))
    print("total under-segmented:", sum(r["n_under_segmented_candidates"] for r in rows))
    print("total over-segmented:", sum(r["n_over_segmented_gt"] for r in rows))
    print("total crosses_section_boundary:", sum(r["n_crosses_section_boundary"] for r in rows))
    print("total bleed-dependent coverage:", sum(r["n_bleed_dependent_gt_coverage"] for r in rows))
    print("resumes with zero recall:", sum(1 for r in rows if r["coverage_recall"] == 0))
    print("resumes with recall < 1.0:", sum(1 for r in rows if r["coverage_recall"] is not None and r["coverage_recall"] < 1.0))
    print("resumes with zero precision:", sum(1 for r in rows if r["candidate_precision"] == 0))


if __name__ == "__main__":
    main()
