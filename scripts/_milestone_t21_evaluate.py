"""
scripts/_milestone_t21_evaluate.py

Milestone T21/Phase 2H -- computes adjudicated-vs-baseline metrics,
invention rate, candidate-escape rate, and the four gates (001/006/
013/021), independently re-verifying the candidate-only constraint
(never trusting the model's own claim).

"Confirmed entities" = ONLY what appears in parsed["confirmed_entities"]
for a successful adjudication. A candidate present in the frozen
inventory but never mentioned in candidate_decisions (an LLM
incompleteness failure, NOT a discard, NOT an invention) is reported
explicitly as UNADDRESSED -- it is not silently treated as KEPT or
DISCARDED.

Reuses T17's own _candidate_covers_company unchanged for GT matching,
identical to T18/T19/T20's own convention.
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

MARKDOWN_DIR = Path("output/markdown")
T19_GT_CSV = Path("reports/t19_unseen_candidate_ground_truth.csv")
BENCHMARK_GT_DIR = Path("benchmark/ground_truth/adjudicated_v1")
BENCHMARK_MAPPING = BENCHMARK_GT_DIR / "mapping.csv"

UNSEEN = {"U1", "U2", "U3", "U4", "U5"}
CONTROLS = {"001", "006", "013", "021"}

STEMS = {
    "U1": "178480991686_Vachan_Premugh_Dicruz_Founder_Office_BTTR.MEN",
    "U2": "8th_April_Akasa_Kapoor_Resume",
    "U3": "AARTI BAGHOTIYA_Economic Urban Planner_Intileo",
    "U4": "AASHISH PARMAR - Email Marketing Executive - Intileo",
    "U5": "_Adarsh_Bhagat Resume",
    "001": "AMAN_CV_BCA",
    "006": "AshishKumarSingh_RBM_Pune_Stonelam",
    "013": "Kishan Yadav_Data Scientist_Intileo",
    "021": "PRAGNESH S. SHAH_Sr. Associate-IT Ops_Intileo",
}


def _load_t19_gt():
    from collections import defaultdict
    by_id = defaultdict(list)
    with T19_GT_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_id[row["unseen_id"]].append(row["company"])
    return by_id


def _load_control_gt():
    with BENCHMARK_MAPPING.open(encoding="utf-8") as f:
        mapping = {r["ResumeID"]: r for r in csv.DictReader(f)}
    out = {}
    for rid in CONTROLS:
        gt_file = mapping[rid]["JSON"]
        gt_jobs = json.loads((BENCHMARK_GT_DIR / gt_file).read_text(encoding="utf-8")).get("work_experience", [])
        out[rid] = [g.get("company") or "" for g in gt_jobs]
    return out


def _baseline_metrics(blocks, candidates, companies):
    coverage = {}
    for gi, company in enumerate(companies):
        coverage[gi] = [c.candidate_id for c in candidates if _candidate_covers_company(c, blocks, company)]
    covered = sum(1 for gi in coverage if coverage[gi])
    recall = covered / len(companies) if companies else None
    cand_covers = {c.candidate_id: [] for c in candidates}
    for gi, cids in coverage.items():
        for cid in cids:
            cand_covers[cid].append(gi)
    n_spurious = sum(1 for c in candidates if not cand_covers[c.candidate_id])
    precision = (len(candidates) - n_spurious) / len(candidates) if candidates else None
    return {
        "n_candidates": len(candidates), "recall": recall, "precision": precision,
        "ratio": (len(candidates) / len(companies)) if companies else None,
    }, coverage


def _adjudicated_metrics(blocks, candidates, companies, parsed, valid_ids):
    """Independently verify candidate-only constraint and compute
    adjudicated recall/precision/invention using ONLY confirmed_entities."""
    id_to_candidate = {f"C{c.candidate_id}": c for c in candidates}
    decided_ids = {d.get("candidate_id") for d in (parsed.get("candidate_decisions") or [])}
    unaddressed = sorted(valid_ids - decided_ids, key=lambda s: int(s[1:]))

    confirmed = parsed.get("confirmed_entities") or []
    invented_entities = []
    entity_records = []
    for e in confirmed:
        cids = e.get("candidate_ids") or []
        invented_refs = [cid for cid in cids if cid not in valid_ids]
        real_cids = [cid for cid in cids if cid in valid_ids]
        if invented_refs:
            invented_entities.append({"entity_id": e.get("entity_id"), "invented_ids": invented_refs})
            continue  # an entity referencing any invented ID cannot be trusted at all
        if not real_cids:
            continue
        # Union of extent text across all real candidate_ids in this entity.
        text_parts = []
        for cid in real_cids:
            c = id_to_candidate[cid]
            text_parts.extend(blocks[i].stripped for i in range(c.extent_start, c.extent_end + 1))
        entity_records.append({
            "entity_id": e.get("entity_id"), "candidate_ids": real_cids,
            "classification": e.get("classification"), "text": " ".join(text_parts),
        })

    coverage = {}
    for gi, company in enumerate(companies):
        covering = [
            rec["entity_id"] for rec in entity_records
            if rec["classification"] == "WORK_EXPERIENCE" and any(
                _candidate_covers_company(id_to_candidate[cid], blocks, company) for cid in rec["candidate_ids"]
            )
        ]
        coverage[gi] = covering
    covered = sum(1 for gi in coverage if coverage[gi])
    recall = covered / len(companies) if companies else None

    n_we_entities = sum(1 for rec in entity_records if rec["classification"] == "WORK_EXPERIENCE")
    entity_covers = {rec["entity_id"]: 0 for rec in entity_records if rec["classification"] == "WORK_EXPERIENCE"}
    for gi, eids in coverage.items():
        for eid in eids:
            entity_covers[eid] = entity_covers.get(eid, 0) + 1
    false_positive_entities = sum(1 for eid, n in entity_covers.items() if n == 0)
    precision = ((n_we_entities - false_positive_entities) / n_we_entities) if n_we_entities else None

    return {
        "n_confirmed_entities": len(entity_records),
        "n_confirmed_we_entities": n_we_entities,
        "recall": recall, "precision": precision,
        "n_invented_entities": len(invented_entities),
        "invented_detail": invented_entities,
        "n_unaddressed_candidates": len(unaddressed),
        "unaddressed_candidate_ids": unaddressed,
        "missing_gt_companies": [companies[gi] for gi in coverage if not coverage[gi]],
    }, coverage


def main():
    t19_gt = _load_t19_gt()
    control_gt = _load_control_gt()
    gt_by_id = {**t19_gt, **control_gt}

    rows = []
    for rid, stem in STEMS.items():
        blocks = blockify((MARKDOWN_DIR / f"{stem}.md").read_text(encoding="utf-8"))
        candidates = generate_candidates_t20(blocks)
        companies = gt_by_id[rid]
        valid_ids = {f"C{c.candidate_id}" for c in candidates}

        base, base_cov = _baseline_metrics(blocks, candidates, companies)

        raw_path = Path(f"reports/t21_raw_llm_response_{rid}.json")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        parsed = raw.get("parsed")

        if parsed is None:
            adj = {"n_confirmed_entities": 0, "n_confirmed_we_entities": 0, "recall": None,
                   "precision": None, "n_invented_entities": 0, "invented_detail": [],
                   "n_unaddressed_candidates": len(valid_ids),
                   "unaddressed_candidate_ids": sorted(valid_ids), "missing_gt_companies": companies}
        else:
            adj, adj_cov = _adjudicated_metrics(blocks, candidates, companies, parsed, valid_ids)

        row = {
            "resume_id": rid, "stem": stem, "population": "unseen" if rid in UNSEEN else "control",
            "gt_n_entities": len(companies),
            "baseline_n_candidates": base["n_candidates"], "baseline_recall": base["recall"],
            "baseline_precision": base["precision"], "baseline_ratio": base["ratio"],
            "adjudicated_n_confirmed_entities": adj["n_confirmed_entities"],
            "adjudicated_n_confirmed_we_entities": adj["n_confirmed_we_entities"],
            "adjudicated_recall": adj["recall"], "adjudicated_precision": adj["precision"],
            "n_invented_entities": adj["n_invented_entities"],
            "n_unaddressed_candidates": adj["n_unaddressed_candidates"],
            "unaddressed_candidate_ids": ";".join(adj["unaddressed_candidate_ids"]),
            "missing_gt_companies": "; ".join(adj["missing_gt_companies"]),
        }
        rows.append(row)
        print(f"{rid:4} baseline: cand={base['n_candidates']:3} recall={base['recall']} prec={base['precision']:.3f}  |  "
              f"adjudicated: entities={adj['n_confirmed_entities']:2} recall={adj['recall']} "
              f"invented={adj['n_invented_entities']} unaddressed={adj['n_unaddressed_candidates']}")
        if adj["missing_gt_companies"]:
            print(f"     MISSING under adjudication: {adj['missing_gt_companies']}")

    out = Path("reports/t21_candidate_adjudication_metrics.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
