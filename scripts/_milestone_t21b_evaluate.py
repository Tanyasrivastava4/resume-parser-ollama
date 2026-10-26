"""
scripts/_milestone_t21b_evaluate.py

Milestone T21B/Phase 2I -- candidate-level evaluation.

CANDIDATE TP/FP/FN/TN DEFINITION (frozen before inspecting results):
A candidate is "GT-relevant" if its own extent (via T17/T19/T21's own
unmodified _candidate_covers_company, the same convention used
throughout T17-T21 -- not redefined here) fuzzy-matches at least one
real GT Work Experience company for that resume.

A candidate is "LLM-accepted" if its decision is KEEP with
classification WORK_EXPERIENCE, OR decision CLASSIFY with
classification WORK_EXPERIENCE (both are valid ways under the T21B
contract to mark a candidate as Work-Experience-relevant; this single
rule is frozen here, before any result was inspected).

  TP = GT-relevant AND LLM-accepted
  FP = NOT GT-relevant AND LLM-accepted
  FN = GT-relevant AND NOT LLM-accepted (includes DISCARD, CLASSIFY as
       non-WE, and unaddressed/missing candidates)
  TN = NOT GT-relevant AND NOT LLM-accepted

SECONDARY METRIC (clearly labeled, not entity precision): "GT entity
coverage via accepted candidates" -- for each GT Work Experience
entity, whether AT LEAST ONE LLM-accepted candidate covers it. No
entity construction, no merge, no split.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_t18_blocks import blockify  # noqa: E402
from scripts._milestone_t20_candidates import generate_candidates_t20  # noqa: E402
from scripts._milestone_t17_evaluate import _candidate_covers_company  # noqa: E402
from scripts._milestone_t21_evaluate import STEMS, _load_t19_gt, _load_control_gt, UNSEEN, CONTROLS  # noqa: E402

MARKDOWN_DIR = Path("output/markdown")

ACCEPTED_DECISIONS = {"KEEP", "CLASSIFY"}
ACCEPTED_CLASSIFICATION = "WORK_EXPERIENCE"


def _is_accepted(d) -> bool:
    return d is not None and d["decision"] in ACCEPTED_DECISIONS and d.get("classification") == ACCEPTED_CLASSIFICATION


def evaluate_resume(rid: str, stem: str, companies):
    blocks = blockify((MARKDOWN_DIR / f"{stem}.md").read_text(encoding="utf-8"))
    candidates = generate_candidates_t20(blocks)
    valid_ids = {f"C{c.candidate_id}" for c in candidates}
    id_to_candidate = {f"C{c.candidate_id}": c for c in candidates}

    raw = json.loads(Path(f"reports/t21b_raw_llm_response_{rid}.json").read_text(encoding="utf-8"))
    parsed = raw.get("parsed") or {}
    decisions = {d["candidate_id"]: d for d in parsed.get("candidate_decisions", [])}

    relevant_ids = set()
    company_covering_ids = {}
    for company in companies:
        covering = {f"C{c.candidate_id}" for c in candidates if _candidate_covers_company(c, blocks, company)}
        company_covering_ids[company] = covering
        relevant_ids |= covering

    tp = fp = fn = tn = 0
    section_heading_correct = section_heading_total = 0
    per_candidate_rows = []
    for cid in sorted(valid_ids, key=lambda s: int(s[1:])):
        d = decisions.get(cid)
        is_relevant = cid in relevant_ids
        accepted = _is_accepted(d)
        if is_relevant and accepted:
            tp += 1; outcome = "TP"
        elif is_relevant and not accepted:
            fn += 1; outcome = "FN"
        elif not is_relevant and accepted:
            fp += 1; outcome = "FP"
        else:
            tn += 1; outcome = "TN"

        c = id_to_candidate[cid]
        is_section_heading_singleton = (c.n_blocks_in_extent == 1 and c.containing_section_label == "experience")
        if is_section_heading_singleton:
            section_heading_total += 1
            if not accepted:
                section_heading_correct += 1

        per_candidate_rows.append({
            "resume_id": rid, "candidate_id": cid, "is_gt_relevant": is_relevant,
            "llm_decision": d["decision"] if d else "UNADDRESSED",
            "llm_classification": d.get("classification") if d else "",
            "accepted": accepted, "outcome": outcome,
        })

    n_missing = len(valid_ids) - len(decisions)
    coverage = n_missing == 0 and len(decisions) == len(valid_ids)  # completeness, not GT recall
    # entity coverage via accepted candidates
    covered_companies = [c for c in companies if company_covering_ids[c] & {cid for cid in valid_ids if _is_accepted(decisions.get(cid))}]
    entity_coverage_recall = len(covered_companies) / len(companies) if companies else None

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else None

    invented = sum(1 for d in parsed.get("candidate_decisions", []) if d.get("candidate_id") not in valid_ids)
    seen = [d.get("candidate_id") for d in parsed.get("candidate_decisions", [])]
    duplicates = len(seen) - len(set(seen))

    result = {
        "resume_id": rid, "stem": stem, "population": "unseen" if rid in UNSEEN else "control",
        "n_candidates": len(valid_ids), "n_decisions_returned": len(decisions),
        "n_missing_decisions": n_missing, "decision_coverage": len(decisions) / len(valid_ids) if valid_ids else None,
        "n_invented": invented, "n_duplicates": duplicates,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "candidate_precision": precision, "candidate_recall": recall, "candidate_f1": f1,
        "gt_n_entities": len(companies),
        "gt_entity_coverage_via_accepted_candidates": entity_coverage_recall,
        "section_heading_singletons_total": section_heading_total,
        "section_heading_singletons_correctly_discarded": section_heading_correct,
        "missing_gt_companies": "; ".join(c for c in companies if c not in covered_companies),
    }
    return result, per_candidate_rows


def main():
    t19_gt = _load_t19_gt()
    control_gt = _load_control_gt()
    gt_by_id = {**t19_gt, **control_gt}

    resume_rows = []
    all_candidate_rows = []
    for rid, stem in STEMS.items():
        result, cand_rows = evaluate_resume(rid, stem, gt_by_id[rid])
        resume_rows.append(result)
        all_candidate_rows.extend(cand_rows)
        print(f"{rid:4} coverage={result['decision_coverage']:.3f} TP={result['TP']:2} FP={result['FP']:2} "
              f"FN={result['FN']:2} TN={result['TN']:2} prec={result['candidate_precision']} rec={result['candidate_recall']} "
              f"entity_coverage={result['gt_entity_coverage_via_accepted_candidates']}")
        if result["missing_gt_companies"]:
            print(f"     NOT covered by any accepted candidate: {result['missing_gt_companies']}")

    out = Path("reports/t21b_per_resume_metrics.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(resume_rows[0].keys()))
        w.writeheader()
        w.writerows(resume_rows)
    print(f"\nwrote {out}")

    out2 = Path("reports/t21b_candidate_metrics.csv")
    with out2.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_candidate_rows[0].keys()))
        w.writeheader()
        w.writerows(all_candidate_rows)
    print(f"wrote {out2}")

    total_tp = sum(r["TP"] for r in resume_rows)
    total_fp = sum(r["FP"] for r in resume_rows)
    total_fn = sum(r["FN"] for r in resume_rows)
    total_tn = sum(r["TN"] for r in resume_rows)
    total_invented = sum(r["n_invented"] for r in resume_rows)
    total_duplicates = sum(r["n_duplicates"] for r in resume_rows)
    total_missing = sum(r["n_missing_decisions"] for r in resume_rows)
    total_returned = sum(r["n_decisions_returned"] for r in resume_rows)
    total_candidates = sum(r["n_candidates"] for r in resume_rows)

    agg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else None
    agg_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else None
    agg_f1 = (2 * agg_precision * agg_recall / (agg_precision + agg_recall)) if (agg_precision and agg_recall) else None

    print("\n=== AGGREGATE ===")
    print(f"total candidates: {total_candidates}")
    print(f"total decisions returned: {total_returned}")
    print(f"total missing decisions: {total_missing}  (missing rate: {total_missing/total_candidates:.3f})")
    print(f"total invented candidate IDs: {total_invented}  (invention rate: {total_invented/total_returned:.4f})")
    print(f"total duplicate decisions: {total_duplicates}  (duplicate rate: {total_duplicates/total_returned:.4f})")
    print(f"TP={total_tp} FP={total_fp} FN={total_fn} TN={total_tn}")
    print(f"aggregate candidate precision: {agg_precision}")
    print(f"aggregate candidate recall: {agg_recall}")
    print(f"aggregate candidate F1: {agg_f1}")
    mean_entity_coverage = sum(r["gt_entity_coverage_via_accepted_candidates"] for r in resume_rows) / len(resume_rows)
    print(f"mean GT entity coverage via accepted candidates: {mean_entity_coverage}")


if __name__ == "__main__":
    main()
