"""
scripts/_milestone_t23_evaluate.py

Milestone T23/Phase 2K -- evaluates T23's hybrid entities against GT,
reusing T17's own unmodified _candidate_covers_company (identical
definition to T22, never redefined), and produces a paired T22-vs-T23
comparison per resume.
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
from scripts._milestone_t22_assemble import load_t21b_decisions, run_resume as run_resume_t22  # noqa: E402
from scripts._milestone_t18_blocks import blockify  # noqa: E402
from scripts._milestone_t20_candidates import generate_candidates_t20  # noqa: E402
from scripts._milestone_t22_assemble import is_accepted  # noqa: E402
from scripts._milestone_t23_ambiguity_detection import classify_junction  # noqa: E402
from scripts._milestone_t22_assemble import Entity  # noqa: E402

MARKDOWN_DIR = Path("output/markdown")


def _load_saved_decisions():
    """Reuses reports/t23_adjudication_decisions.csv -- the decisions
    already produced (and LLM-called for) by scripts/_milestone_t23_run.py
    -- rather than invoking the LLM a second time for evaluation."""
    path = Path("reports/t23_adjudication_decisions.csv")
    saved = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                saved[row["ambiguity_id"]] = row
    return saved


def assemble_entities_from_saved_decisions(rid: str, blocks, accepted, saved_decisions):
    """Byte-for-byte same grouping logic as assemble_entities_hybrid,
    but consumes ALREADY-MADE LLM decisions instead of calling the LLM
    again -- zero additional LLM calls."""
    entities = []
    if not accepted:
        return entities
    junctions = {}
    for i in range(len(accepted) - 1):
        a, b = accepted[i], accepted[i + 1]
        if b.extent_start == a.extent_end + 1:
            junctions[i] = classify_junction(blocks, a, b)

    ambiguity_counter = 0
    entities.append(Entity(entity_id="E0", candidate_ids=[f"C{accepted[0].candidate_id}"],
                            extent_start=accepted[0].extent_start, extent_end=accepted[0].extent_end))

    for i in range(1, len(accepted)):
        cur = accepted[i]
        j = junctions.get(i - 1)
        should_merge = False
        if j is not None:
            if j.classification == "CLEAR_MERGE":
                should_merge = True
            elif j.classification == "CLEAR_SEPARATE":
                should_merge = False
            elif j.classification == "SEMANTICALLY_AMBIGUOUS":
                ambiguity_counter += 1
                ambiguity_id = f"{rid}_AMB_{ambiguity_counter:03d}"
                saved = saved_decisions.get(ambiguity_id)
                should_merge = saved is not None and saved["decision"] == "MERGE_SPECIFIED"
        if should_merge:
            entities[-1].candidate_ids.append(f"C{cur.candidate_id}")
            entities[-1].extent_end = cur.extent_end
        else:
            entities.append(Entity(entity_id=f"E{len(entities)}", candidate_ids=[f"C{cur.candidate_id}"],
                                    extent_start=cur.extent_start, extent_end=cur.extent_end))

    for e in entities:
        for cid in e.candidate_ids:
            cand = next(c for c in accepted if f"C{c.candidate_id}" == cid)
            e.company_evidence.append(cand.anchor_text)
    return entities


def run_resume_t23_from_saved(rid, stem, decisions_by_resume, saved_decisions):
    blocks = blockify((MARKDOWN_DIR / f"{stem}.md").read_text(encoding="utf-8"))
    candidates = generate_candidates_t20(blocks)
    dmap = decisions_by_resume.get(rid, {})
    accepted = sorted([c for c in candidates if is_accepted(dmap.get(f"C{c.candidate_id}"))],
                       key=lambda c: c.extent_start)
    entities = assemble_entities_from_saved_decisions(rid, blocks, accepted, saved_decisions)
    return blocks, candidates, accepted, entities


def evaluate_entities(companies, blocks, candidates, accepted, entities):
    id_to_candidate = {f"C{c.candidate_id}": c for c in candidates}
    coverage = {}
    for gi, company in enumerate(companies):
        covering = [e.entity_id for e in entities
                    if any(_candidate_covers_company(id_to_candidate[cid], blocks, company) for cid in e.candidate_ids)]
        coverage[gi] = covering
    covered = sum(1 for gi in coverage if coverage[gi])
    recall = covered / len(companies) if companies else None

    entity_covers_gt = {e.entity_id: [] for e in entities}
    for gi, eids in coverage.items():
        for eid in eids:
            entity_covers_gt[eid].append(gi)
    n_spurious = sum(1 for e in entities if not entity_covers_gt[e.entity_id])
    precision = (len(entities) - n_spurious) / len(entities) if entities else None

    over_seg = [gi for gi, eids in coverage.items() if len(eids) >= 2]
    under_merge = [e.entity_id for e in entities if len(entity_covers_gt[e.entity_id]) >= 2]

    accepted_ids = {f"C{c.candidate_id}" for c in accepted}
    used_ids = {cid for e in entities for cid in e.candidate_ids}
    unassigned = accepted_ids - used_ids
    duplicated = len(used_ids) != sum(len(e.candidate_ids) for e in entities)

    return {
        "gt_n_entities": len(companies), "accepted_candidates": len(accepted), "final_entities": len(entities),
        "entity_recall": recall, "covered_gt_count": covered,
        "entity_precision": precision, "n_spurious_entities": n_spurious,
        "n_over_segmented_gt": len(over_seg), "n_under_merged_entities": len(under_merge),
        "n_unassigned_accepted_candidates": len(unassigned), "duplicate_assignment_detected": duplicated,
        "missing_gt_companies": "; ".join(companies[gi] for gi in coverage if not coverage[gi]),
    }


def main():
    t19_gt = _load_t19_gt()
    control_gt = _load_control_gt()
    gt_by_id = {**t19_gt, **control_gt}
    decisions = load_t21b_decisions()
    saved_decisions = _load_saved_decisions()
    # Per-resume LLM call count, read from the call log already produced
    # by scripts/_milestone_t23_run.py -- no LLM is invoked again here.
    llm_calls_by_resume = {}
    call_log_path = Path("reports/t23_call_log.csv")
    if call_log_path.exists():
        with call_log_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                llm_calls_by_resume[row["resume_id"]] = llm_calls_by_resume.get(row["resume_id"], 0) + 1

    comparison_rows = []
    trace_rows = []
    total_llm_calls = sum(llm_calls_by_resume.values())

    for rid, stem in STEMS.items():
        companies = gt_by_id[rid]

        # T22 baseline (deterministic-only, reused verbatim)
        blocks22, cands22, accepted22, entities22 = run_resume_t22(rid, stem, decisions)
        t22_result = evaluate_entities(companies, blocks22, cands22, accepted22, entities22)

        # T23 hybrid -- reconstructed from ALREADY-SAVED LLM decisions,
        # zero additional LLM calls made here.
        blocks23, cands23, accepted23, entities23 = run_resume_t23_from_saved(rid, stem, decisions, saved_decisions)
        t23_result = evaluate_entities(companies, blocks23, cands23, accepted23, entities23)
        n_calls_this_resume = llm_calls_by_resume.get(rid, 0)

        row = {"resume_id": rid, "stem": stem, "population": "unseen" if rid in UNSEEN else "control",
               "llm_calls_this_resume": n_calls_this_resume}
        for k, v in t22_result.items():
            row[f"t22_{k}"] = v
        for k, v in t23_result.items():
            row[f"t23_{k}"] = v
        comparison_rows.append(row)

        print(f"{rid:4} T22: entities={t22_result['final_entities']:2} recall={t22_result['entity_recall']} "
              f"prec={t22_result['entity_precision']}  |  T23: entities={t23_result['final_entities']:2} "
              f"recall={t23_result['entity_recall']} prec={t23_result['entity_precision']}  llm_calls={n_calls_this_resume}")

        id_to_candidate23 = {f"C{c.candidate_id}": c for c in cands23}
        for e in entities23:
            trace_rows.append({
                "resume_id": rid, "entity_id": e.entity_id, "candidate_ids": ";".join(e.candidate_ids),
                "n_candidates": len(e.candidate_ids), "extent_start": e.extent_start, "extent_end": e.extent_end,
                "company_evidence": " | ".join(e.company_evidence)[:200],
            })

    out = Path("reports/t23_entity_comparison.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        w.writeheader()
        w.writerows(comparison_rows)
    print(f"\nwrote {out}")

    out2 = Path("reports/t23_metrics.csv")
    with out2.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
        w.writeheader()
        w.writerows(trace_rows)
    print(f"wrote {out2}")

    total_gt = sum(r["t23_gt_n_entities"] for r in comparison_rows)
    t22_covered = sum(r["t22_covered_gt_count"] for r in comparison_rows)
    t23_covered = sum(r["t23_covered_gt_count"] for r in comparison_rows)
    t22_entities = sum(r["t22_final_entities"] for r in comparison_rows)
    t23_entities = sum(r["t23_final_entities"] for r in comparison_rows)
    t22_spurious = sum(r["t22_n_spurious_entities"] for r in comparison_rows)
    t23_spurious = sum(r["t23_n_spurious_entities"] for r in comparison_rows)
    t22_under = sum(r["t22_n_under_merged_entities"] for r in comparison_rows)
    t23_under = sum(r["t23_n_under_merged_entities"] for r in comparison_rows)
    t22_unassigned = sum(r["t22_n_unassigned_accepted_candidates"] for r in comparison_rows)
    t23_unassigned = sum(r["t23_n_unassigned_accepted_candidates"] for r in comparison_rows)

    print("\n=== AGGREGATE ===")
    print(f"total GT entities: {total_gt}")
    print(f"T22 recall: {t22_covered/total_gt:.4f}   T23 recall: {t23_covered/total_gt:.4f}")
    print(f"T22 precision: {(t22_entities-t22_spurious)/t22_entities:.4f}   "
          f"T23 precision: {(t23_entities-t23_spurious)/t23_entities:.4f}")
    print(f"T22 under-merge total: {t22_under}   T23 under-merge total: {t23_under}")
    print(f"T22 unassigned: {t22_unassigned}   T23 unassigned: {t23_unassigned}")
    print(f"total LLM calls across corpus: {total_llm_calls}")


if __name__ == "__main__":
    main()
