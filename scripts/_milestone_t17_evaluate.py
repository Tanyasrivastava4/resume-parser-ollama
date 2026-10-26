"""
scripts/_milestone_t17_evaluate.py

Milestone T17/Phase 2D -- offline, zero-LLM evaluation harness, per
docs/T17_PHASE2D_CANDIDATE_GENERATION_PRE_REGISTRATION.md.

Runs the frozen blocking + candidate-generation + safety-flagging
passes across the full 22-resume corpus, computes all required
metrics, and writes CSV artifacts. No LLM calls anywhere. Does not
modify any production file or existing benchmark output.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_t17_blocks import blockify  # noqa: E402
from scripts._milestone_t17_candidates import generate_candidates  # noqa: E402
from validators.misrouting_detector import _fuzzy_contains  # noqa: E402

MARKDOWN_DIR = Path("output/markdown")
GT_DIR = Path("benchmark/ground_truth/adjudicated_v1")
MAPPING_CSV = GT_DIR / "mapping.csv"

HELD_OUT_IDS = {"001", "006", "013", "021"}


def _load_population():
    with MAPPING_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _candidate_covers_company(cand, blocks, company: str) -> bool:
    if _fuzzy_contains(cand.anchor_text, company):
        return True
    for i in range(cand.extent_start, cand.extent_end + 1):
        if _fuzzy_contains(blocks[i].stripped, company):
            return True
    return False


def evaluate_resume(rid: str, stem: str, gt_file: str):
    markdown_text = (MARKDOWN_DIR / f"{stem}.md").read_text(encoding="utf-8")
    blocks = blockify(markdown_text)
    candidates = generate_candidates(blocks)

    gt_path = GT_DIR / gt_file
    gt_jobs = json.loads(gt_path.read_text(encoding="utf-8")).get("work_experience", [])

    coverage = {}  # gt_index -> list of candidate_ids covering it
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
    # Coverage loss attributable to cross-section-bleed candidates:
    # would any GT entity become uncovered if these candidates were
    # removed? (Measured, not assumed.)
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
        "over_segmented_gt_companies": "; ".join(gt_jobs[gi].get("company") or "" for gi in over_segmented),
        "n_under_segmented_candidates": len(under_segmented),
        "n_unsafe_candidates": len(unsafe_candidates), "unsafe_rate": unsafe_rate,
        "n_crosses_section_boundary": len(crosses_boundary),
        "n_bleed_dependent_gt_coverage": len(bleed_dependent_gt),
        "missing_gt_companies": "; ".join(
            gt_jobs[gi].get("company") or "" for gi in coverage if not coverage[gi]
        ),
    }, blocks, candidates, coverage, gt_jobs


def main():
    population = _load_population()
    rows = []
    per_resume_data = {}

    for row in population:
        rid = row["ResumeID"]
        stem = Path(row["Markdown"]).stem
        gt_file = row["JSON"]
        result, blocks, candidates, coverage, gt_jobs = evaluate_resume(rid, stem, gt_file)
        rows.append(result)
        per_resume_data[rid] = (blocks, candidates, coverage, gt_jobs)
        print(f"{rid} ({stem}): candidates={result['n_candidates']} "
              f"recall={result['coverage_recall']} precision={result['candidate_precision']} "
              f"unsafe_rate={result['unsafe_rate']}")

    out = Path("benchmark/baselines/t17_corpus_evaluation.csv")
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
    print("resumes with recall < 1.0:", [r["resume_id"] for r in rows if r["coverage_recall"] is not None and r["coverage_recall"] < 1.0])

    print("\n=== TUNING SUBSET (18) ===")
    print("mean coverage_recall:", _agg(tuning, "coverage_recall"))
    print("mean candidate_precision:", _agg(tuning, "candidate_precision"))

    print("\n=== HELD-OUT SUBSET (001/006/013/021) ===")
    print("mean coverage_recall:", _agg(held_out, "coverage_recall"))
    print("mean candidate_precision:", _agg(held_out, "candidate_precision"))
    for r in held_out:
        print(" ", r["resume_id"], r)

    return per_resume_data


if __name__ == "__main__":
    main()
