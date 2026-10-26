"""
scripts/_milestone_t18_evaluate.py

Milestone T18/Phase 2E -- offline, zero-LLM evaluation harness,
comparing the T18-refined candidate generator against the frozen T17
baseline. Reuses T17's own evaluation/matching convention
(_fuzzy_contains, coverage/precision/segmentation definitions)
unchanged -- no evaluation definition is redefined.

No LLM calls. Does not modify T17's own files or any production file,
existing benchmark output, or GT/PDF source.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_t18_blocks import blockify  # noqa: E402 (re-exported, unchanged, from T17)
from scripts._milestone_t18_candidates import generate_candidates_v2  # noqa: E402
from scripts._milestone_t17_evaluate import _candidate_covers_company  # noqa: E402 (unchanged matching convention)

MARKDOWN_DIR = Path("output/markdown")
GT_DIR = Path("benchmark/ground_truth/adjudicated_v1")
MAPPING_CSV = GT_DIR / "mapping.csv"

HELD_OUT_IDS = {"001", "006", "013", "021"}


def _load_population():
    with MAPPING_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate_resume_v2(rid: str, stem: str, gt_file: str):
    markdown_text = (MARKDOWN_DIR / f"{stem}.md").read_text(encoding="utf-8")
    blocks = blockify(markdown_text)
    candidates = generate_candidates_v2(blocks)

    gt_path = GT_DIR / gt_file
    gt_jobs = json.loads(gt_path.read_text(encoding="utf-8")).get("work_experience", [])

    coverage = {}
    for gi, gt in enumerate(gt_jobs):
        company = gt.get("company") or ""
        covering = [c.candidate_id for c in candidates if _candidate_covers_company(c, blocks, company)]
        coverage[gi] = covering

    covered_gt = sum(1 for gi in coverage if coverage[gi])
    recall = covered_gt / len(gt_jobs) if gt_jobs else None

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

    return {
        "resume_id": rid, "stem": stem,
        "gt_n_jobs": len(gt_jobs), "n_candidates": len(candidates),
        "coverage_recall": recall, "covered_gt_count": covered_gt,
        "candidate_precision": precision, "n_spurious_candidates": n_spurious,
        "candidate_entity_ratio": (len(candidates) / len(gt_jobs)) if gt_jobs else None,
        "n_over_segmented_gt": len(over_segmented),
        "n_under_segmented_candidates": len(under_segmented),
        "n_unsafe_candidates": len(unsafe_candidates), "unsafe_rate": unsafe_rate,
        "n_crosses_section_boundary": len(crosses_boundary),
        "n_bleed_dependent_gt_coverage": len(bleed_dependent_gt),
        "missing_gt_companies": "; ".join(
            gt_jobs[gi].get("company") or "" for gi in coverage if not coverage[gi]
        ),
    }, blocks, candidates, coverage, gt_jobs


def main() -> None:
    population = _load_population()
    rows = []

    for row in population:
        rid = row["ResumeID"]
        stem = Path(row["Markdown"]).stem
        gt_file = row["JSON"]
        result, blocks, candidates, coverage, gt_jobs = evaluate_resume_v2(rid, stem, gt_file)
        rows.append(result)
        print(f"{rid} ({stem}): candidates={result['n_candidates']} "
              f"recall={result['coverage_recall']} precision={result['candidate_precision']}")

    out = Path("benchmark/baselines/t18_corpus_evaluation.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")

    tuning = [r for r in rows if r["resume_id"] not in HELD_OUT_IDS]
    held_out = [r for r in rows if r["resume_id"] in HELD_OUT_IDS]

    def _agg(rs, key):
        vals = [r[key] for r in rs if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    print("\n=== AGGREGATE (all 22) ===")
    print("mean coverage_recall:", _agg(rows, "coverage_recall"))
    print("mean candidate_precision:", _agg(rows, "candidate_precision"))
    print("mean candidate_entity_ratio:", _agg(rows, "candidate_entity_ratio"))
    print("mean unsafe_rate:", _agg(rows, "unsafe_rate"))
    print("total under-segmented:", sum(r["n_under_segmented_candidates"] for r in rows))
    print("total over-segmented:", sum(r["n_over_segmented_gt"] for r in rows))
    print("total candidates:", sum(r["n_candidates"] for r in rows))
    print("total crosses_section_boundary:", sum(r["n_crosses_section_boundary"] for r in rows))
    print("total bleed-dependent coverage:", sum(r["n_bleed_dependent_gt_coverage"] for r in rows))

    print("\n=== HELD-OUT SUBSET (001/006/013/021) ===")
    print("mean coverage_recall:", _agg(held_out, "coverage_recall"))
    print("mean candidate_precision:", _agg(held_out, "candidate_precision"))
    for r in held_out:
        print(" ", r)


if __name__ == "__main__":
    main()
