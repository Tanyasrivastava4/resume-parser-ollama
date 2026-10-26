"""
scripts/_milestone_t23_run.py

Milestone T23/Phase 2K -- main runner. Runs the hybrid deterministic +
narrow-LLM assembly across all 9 resumes, calling the LLM only for
SEMANTICALLY_AMBIGUOUS junctions (pre-registered: exactly 2, both in
U2). Writes call log and decision CSVs.
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
from scripts._milestone_t22_assemble import load_t21b_decisions  # noqa: E402
from scripts._milestone_t23_assemble import run_resume_hybrid  # noqa: E402

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


def main():
    client = OllamaClient()
    decisions_by_resume = load_t21b_decisions()

    call_log_rows = []
    adjudication_rows = []
    all_entities = {}

    for rid, stem in STEMS.items():
        blocks, candidates, accepted, entities, llm_calls = run_resume_hybrid(rid, stem, decisions_by_resume, client)
        all_entities[rid] = (blocks, candidates, accepted, entities)
        print(f"[{rid}] accepted={len(accepted)} entities={len(entities)} llm_calls={len(llm_calls)}")

        for r in llm_calls:
            actual_calls = 1 + int(r.structural_retry_used)
            call_log_rows.append({
                "resume_id": rid, "ambiguity_id": r.ambiguity_id,
                "candidate_ids": ";".join(r.candidate_ids),
                "success": r.parsed is not None, "error": r.error or "",
                "structural_retry_used": r.structural_retry_used,
                "actual_calls": actual_calls,
                "latency_seconds": round(r.latency_seconds, 3),
                "validation_errors": " | ".join(r.validation_errors),
            })
            adjudication_rows.append({
                "resume_id": rid, "ambiguity_id": r.ambiguity_id,
                "candidate_ids": ";".join(r.candidate_ids),
                "decision": (r.parsed or {}).get("decision", "FAILED"),
                "merge_groups": json.dumps((r.parsed or {}).get("merge_groups", [])),
            })

            raw_out = Path(f"reports/t23_raw_llm_response_{r.ambiguity_id}.json")
            raw_out.write_text(json.dumps({
                "ambiguity_id": r.ambiguity_id, "candidate_ids": r.candidate_ids,
                "raw_response": r.raw_response, "retry_raw_response": r.retry_raw_response,
                "parsed": r.parsed, "error": r.error, "validation_errors": r.validation_errors,
                "structural_retry_used": r.structural_retry_used, "latency_seconds": r.latency_seconds,
            }, indent=2, ensure_ascii=False), encoding="utf-8")

    out_calls = Path("reports/t23_call_log.csv")
    if call_log_rows:
        with out_calls.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(call_log_rows[0].keys()))
            w.writeheader()
            w.writerows(call_log_rows)
    else:
        out_calls.write_text("resume_id,ambiguity_id,candidate_ids,success,error,structural_retry_used,actual_calls,latency_seconds,validation_errors\n")
    print(f"\nwrote {out_calls} ({len(call_log_rows)} rows)")

    out_adj = Path("reports/t23_adjudication_decisions.csv")
    if adjudication_rows:
        with out_adj.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(adjudication_rows[0].keys()))
            w.writeheader()
            w.writerows(adjudication_rows)
    else:
        out_adj.write_text("resume_id,ambiguity_id,candidate_ids,decision,merge_groups\n")
    print(f"wrote {out_adj} ({len(adjudication_rows)} rows)")

    total_calls = sum(r["actual_calls"] for r in call_log_rows)
    print(f"\n=== CALL ACCOUNTING ===")
    print(f"total ambiguities requiring LLM: {len(call_log_rows)}")
    print(f"total actual HTTP calls: {total_calls}")
    print(f"structural retries: {sum(1 for r in call_log_rows if r['structural_retry_used'])}")
    print(f"failures: {sum(1 for r in call_log_rows if not r['success'])}")
    if call_log_rows:
        print(f"mean latency: {sum(r['latency_seconds'] for r in call_log_rows)/len(call_log_rows):.2f}s")


if __name__ == "__main__":
    main()
