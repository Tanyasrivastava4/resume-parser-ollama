"""
scripts/_milestone_t21_run.py

Milestone T21/Phase 2H -- main runner.

Phase 1: generate and SAVE the deterministic baseline (T20's
generate_candidates_t20, unmodified) for all 9 resumes, BEFORE any LLM
call is made.

Phase 2: run exactly one batched LLM adjudication call per resume (9
planned calls total), with at most one bounded structural retry per
resume if the first response is not valid JSON / violates the
candidate-only schema.

Phase 3: write raw per-candidate-decision output and per-call
accounting to reports/t21_candidate_adjudication_raw_output.csv and
reports/t21_candidate_adjudication_decisions.csv.

No semantic retry anywhere. No GT is shown to the model at any point.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from services.ollama_client import OllamaClient  # noqa: E402
from scripts._milestone_t18_blocks import blockify  # noqa: E402
from scripts._milestone_t20_candidates import generate_candidates_t20  # noqa: E402
from scripts._milestone_t21_adjudicate import (  # noqa: E402
    MODEL,
    adjudicate_resume,
    build_candidate_inventory,
)

MARKDOWN_DIR = Path("output/markdown")

POPULATION = [
    ("U1", "178480991686_Vachan_Premugh_Dicruz_Founder_Office_BTTR.MEN"),
    ("U2", "8th_April_Akasa_Kapoor_Resume"),
    ("U3", "AARTI BAGHOTIYA_Economic Urban Planner_Intileo"),
    ("U4", "AASHISH PARMAR - Email Marketing Executive - Intileo"),
    ("U5", "_Adarsh_Bhagat Resume"),
    ("001", "AMAN_CV_BCA"),
    ("006", "AshishKumarSingh_RBM_Pune_Stonelam"),
    ("013", "Kishan Yadav_Data Scientist_Intileo"),
    ("021", "PRAGNESH S. SHAH_Sr. Associate-IT Ops_Intileo"),
]


def phase1_baseline():
    """Generate and save the deterministic candidate baseline BEFORE
    any LLM call. Returns {resume_id: (blocks, candidates, inventory)}."""
    baseline = {}
    rows = []
    for rid, stem in POPULATION:
        blocks = blockify((MARKDOWN_DIR / f"{stem}.md").read_text(encoding="utf-8"))
        candidates = generate_candidates_t20(blocks)
        inventory = build_candidate_inventory(blocks, candidates)
        baseline[rid] = (blocks, candidates, inventory)
        for item in inventory:
            row = dict(item)
            row["resume_id"] = rid
            row["stem"] = stem
            rows.append(row)
        print(f"[baseline] {rid:4} ({stem[:45]:45}) candidates={len(candidates)}")

    out = Path("reports/t21_baseline_candidate_inventory.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["resume_id", "stem", "candidate_id", "anchor_text",
                      "section_label", "n_blocks", "flags", "text_preview"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            r["flags"] = ";".join(r["flags"])
            w.writerow({k: r[k] for k in fieldnames})
    print(f"\nwrote {out} ({len(rows)} candidate rows) -- BASELINE SAVED BEFORE ANY LLM CALL\n")
    return baseline


def phase2_adjudicate(baseline):
    client = OllamaClient()
    call_log = []
    decision_rows = []

    for rid, stem in POPULATION:
        print(f"[adjudicate] {rid} ...", end=" ", flush=True)
        result = adjudicate_resume(rid, stem, client)
        status = "OK" if result.parsed is not None else f"FAILED: {result.error}"
        print(f"{status}  latency={result.latency_seconds:.1f}s  "
              f"structural_retry={result.structural_retry_used}  "
              f"validation_errors={len(result.validation_errors)}")

        call_log.append({
            "resume_id": rid, "stem": stem, "n_candidates": result.n_candidates,
            "success": result.parsed is not None, "error": result.error or "",
            "structural_retry_used": result.structural_retry_used,
            "latency_seconds": round(result.latency_seconds, 3),
            "validation_error_count": len(result.validation_errors),
            "validation_errors": " | ".join(result.validation_errors),
            "unrepresentable_by_candidates": bool(
                result.parsed and result.parsed.get("unrepresentable_by_candidates")
            ),
        })

        if result.parsed:
            for d in result.parsed.get("candidate_decisions", []):
                decision_rows.append({
                    "resume_id": rid, "stem": stem,
                    "candidate_id": d.get("candidate_id"),
                    "decision": d.get("decision"),
                    "classification": d.get("classification"),
                    "merge_group": ";".join(d.get("merge_group") or []),
                })
            for e in result.parsed.get("confirmed_entities", []):
                decision_rows.append({
                    "resume_id": rid, "stem": stem,
                    "candidate_id": "ENTITY:" + str(e.get("entity_id")),
                    "decision": "CONFIRMED_ENTITY",
                    "classification": e.get("classification"),
                    "merge_group": ";".join(e.get("candidate_ids") or []),
                })

        # Save raw response regardless of success/failure, for audit.
        raw_out = Path(f"reports/t21_raw_llm_response_{rid}.json")
        raw_out.write_text(json.dumps({
            "resume_id": rid, "stem": stem, "raw_response": result.raw_response,
            "parsed": result.parsed, "error": result.error,
            "validation_errors": result.validation_errors,
            "structural_retry_used": result.structural_retry_used,
            "latency_seconds": result.latency_seconds,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    out_calls = Path("reports/t21_call_log.csv")
    with out_calls.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(call_log[0].keys()))
        w.writeheader()
        w.writerows(call_log)
    print(f"\nwrote {out_calls}")

    out_decisions = Path("reports/t21_candidate_adjudication_decisions.csv")
    with out_decisions.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["resume_id", "stem", "candidate_id", "decision", "classification", "merge_group"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(decision_rows)
    print(f"wrote {out_decisions}")

    return call_log, decision_rows


def main():
    print(f"=== Model: {MODEL} ===\n")
    baseline = phase1_baseline()
    call_log, decision_rows = phase2_adjudicate(baseline)

    planned = len(POPULATION)
    actual = sum(1 + int(c["structural_retry_used"]) for c in call_log)
    successful = sum(1 for c in call_log if c["success"])
    retries = sum(1 for c in call_log if c["structural_retry_used"])
    failed = sum(1 for c in call_log if not c["success"])

    print("\n=== CALL ACCOUNTING ===")
    print(f"planned calls (1 per resume): {planned}")
    print(f"actual HTTP calls made: {actual}")
    print(f"successful (valid, schema-conformant) adjudications: {successful}")
    print(f"structural retries used: {retries}")
    print(f"semantic retries: 0 (never performed, by design)")
    print(f"failed (could not produce a valid response even after retry): {failed}")
    print(f"mean latency: {sum(c['latency_seconds'] for c in call_log)/len(call_log):.2f}s")


if __name__ == "__main__":
    main()
