"""
scripts/_milestone_t22_evaluate.py

Milestone T22/Phase 2J -- evaluates the deterministic-assembly entities
against GT, reusing T17/T19/T21's own unmodified _candidate_covers_company
matching convention (never redefined here).

ENTITY-TO-GT MATCHING DEFINITION (frozen before inspecting results):
an entity "covers" a GT Work Experience company if ANY of its member
candidates individually covers that company (via
_candidate_covers_company, applied per-candidate exactly as T17-T21
always have -- an entity's coverage is the union of its members'
coverage, no new matching logic is introduced).

  Entity recall = (# GT entities covered by >=1 entity) / (# GT entities)
  Entity precision = (# entities covering >=1 real GT company) / (# entities)
  Over-segmentation = # GT entities covered by >=2 DIFFERENT entities
                       (the same real job represented as 2+ separate
                       final entities -- an assembly under-merge)
  Under-segmentation (merge error) = # entities covering >=2 DIFFERENT
                       GT companies (two distinct real jobs incorrectly
                       combined into one final entity -- an assembly
                       over-merge)

Candidate-escape and candidate-adjudication limitations (a real GT job
with no accepted candidate at all) are explicitly NOT counted as
assembly failures -- see the failure-taxonomy classification.

Zero LLM calls.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_t17_evaluate import _candidate_covers_company  # noqa: E402
from scripts._milestone_t21_evaluate import STEMS, _load_t19_gt, _load_control_gt, UNSEEN  # noqa: E402
from scripts._milestone_t22_assemble import load_t21b_decisions, run_resume  # noqa: E402


def evaluate_resume(rid: str, stem: str, companies, blocks, candidates, accepted, entities):
    id_to_candidate = {f"C{c.candidate_id}": c for c in candidates}

    coverage = {}  # gi -> [entity_id, ...]
    for gi, company in enumerate(companies):
        covering = []
        for e in entities:
            if any(_candidate_covers_company(id_to_candidate[cid], blocks, company) for cid in e.candidate_ids):
                covering.append(e.entity_id)
        coverage[gi] = covering

    covered = sum(1 for gi in coverage if coverage[gi])
    recall = covered / len(companies) if companies else None

    entity_covers_gt = {e.entity_id: [] for e in entities}
    for gi, eids in coverage.items():
        for eid in eids:
            entity_covers_gt[eid].append(gi)

    n_spurious_entities = sum(1 for e in entities if not entity_covers_gt[e.entity_id])
    precision = (len(entities) - n_spurious_entities) / len(entities) if entities else None

    over_segmented = [gi for gi, eids in coverage.items() if len(eids) >= 2]  # same job, 2+ final entities
    under_merged = [e.entity_id for e in entities if len(entity_covers_gt[e.entity_id]) >= 2]  # 2+ real jobs, 1 entity

    accepted_ids = {f"C{c.candidate_id}" for c in accepted}
    entity_ids_used = {cid for e in entities for cid in e.candidate_ids}
    unassigned = accepted_ids - entity_ids_used
    duplicated = len(entity_ids_used) != sum(len(e.candidate_ids) for e in entities)

    multi_candidate_entities = sum(1 for e in entities if len(e.candidate_ids) > 1)
    single_candidate_entities = sum(1 for e in entities if len(e.candidate_ids) == 1)

    return {
        "resume_id": rid, "stem": stem, "population": "unseen" if rid in UNSEEN else "control",
        "gt_n_entities": len(companies), "accepted_candidates": len(accepted),
        "final_entities": len(entities),
        "candidates_per_entity": (len(accepted) / len(entities)) if entities else None,
        "entity_recall": recall, "covered_gt_count": covered,
        "entity_precision": precision, "n_spurious_entities": n_spurious_entities,
        "n_over_segmented_gt": len(over_segmented),
        "n_under_merged_entities": len(under_merged),
        "multi_candidate_entities": multi_candidate_entities,
        "single_candidate_entities": single_candidate_entities,
        "n_unassigned_accepted_candidates": len(unassigned),
        "unassigned_candidate_ids": ";".join(sorted(unassigned, key=lambda s: int(s[1:]))),
        "duplicate_assignment_detected": duplicated,
        "missing_gt_companies": "; ".join(companies[gi] for gi in coverage if not coverage[gi]),
    }, coverage


def main():
    t19_gt = _load_t19_gt()
    control_gt = _load_control_gt()
    gt_by_id = {**t19_gt, **control_gt}
    decisions_by_resume = load_t21b_decisions()

    resume_rows = []
    trace_rows = []
    for rid, stem in STEMS.items():
        blocks, candidates, accepted, entities = run_resume(rid, stem, decisions_by_resume)
        companies = gt_by_id[rid]
        result, coverage = evaluate_resume(rid, stem, companies, blocks, candidates, accepted, entities)
        resume_rows.append(result)
        print(f"{rid:4} accepted={result['accepted_candidates']:3} entities={result['final_entities']:3} "
              f"recall={result['entity_recall']} prec={result['entity_precision']} "
              f"over_seg={result['n_over_segmented_gt']} under_merge={result['n_under_merged_entities']} "
              f"unassigned={result['n_unassigned_accepted_candidates']}")
        if result["missing_gt_companies"]:
            print(f"     NOT covered by any entity: {result['missing_gt_companies']}")

        for e in entities:
            trace_rows.append({
                "resume_id": rid, "entity_id": e.entity_id,
                "candidate_ids": ";".join(e.candidate_ids),
                "n_candidates": len(e.candidate_ids),
                "extent_start": e.extent_start, "extent_end": e.extent_end,
                "company_evidence": " | ".join(e.company_evidence),
                "date_evidence": " | ".join(e.date_evidence),
            })

    out = Path("reports/t22_per_resume_metrics.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(resume_rows[0].keys()))
        w.writeheader()
        w.writerows(resume_rows)
    print(f"\nwrote {out}")

    out2 = Path("reports/t22_entity_traceability.csv")
    with out2.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
        w.writeheader()
        w.writerows(trace_rows)
    print(f"wrote {out2}")

    total_gt = sum(r["gt_n_entities"] for r in resume_rows)
    total_covered = sum(r["covered_gt_count"] for r in resume_rows)
    total_entities = sum(r["final_entities"] for r in resume_rows)
    total_spurious = sum(r["n_spurious_entities"] for r in resume_rows)
    total_over = sum(r["n_over_segmented_gt"] for r in resume_rows)
    total_under_merge = sum(r["n_under_merged_entities"] for r in resume_rows)
    total_unassigned = sum(r["n_unassigned_accepted_candidates"] for r in resume_rows)
    total_accepted = sum(r["accepted_candidates"] for r in resume_rows)

    print("\n=== AGGREGATE ===")
    print(f"total GT entities: {total_gt}")
    print(f"total accepted candidates: {total_accepted}")
    print(f"total final entities: {total_entities}")
    print(f"aggregate entity recall: {total_covered/total_gt:.4f}")
    print(f"aggregate entity precision: {(total_entities-total_spurious)/total_entities:.4f}")
    print(f"total over-segmentation (1 GT job, 2+ entities): {total_over}")
    print(f"total under-merge / merge errors (1 entity, 2+ GT jobs): {total_under_merge}")
    print(f"total unassigned accepted candidates: {total_unassigned} (assignment coverage: "
          f"{(total_accepted-total_unassigned)/total_accepted:.4f})")


if __name__ == "__main__":
    main()
