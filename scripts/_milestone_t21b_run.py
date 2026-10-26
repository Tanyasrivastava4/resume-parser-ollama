"""
scripts/_milestone_t21b_run.py

Milestone T21B/Phase 2I -- main runner.

Phase A: build and display the exact frozen population, candidate
inventory hashes, prompt, and output contract -- BEFORE any LLM call.

Phase B: (only when --execute is passed) run exactly one batched
candidate-decision-only call per resume (9 planned), with at most one
bounded structural retry per resume.

Phase C: write raw outputs and decisions to reports/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from services.ollama_client import OllamaClient  # noqa: E402
from scripts._milestone_t18_blocks import blockify  # noqa: E402
from scripts._milestone_t20_candidates import generate_candidates_t20  # noqa: E402
from scripts._milestone_t21b_adjudicate import (  # noqa: E402
    MODEL,
    adjudicate_resume,
    build_candidate_inventory,
    build_prompt,
)

MARKDOWN_DIR = Path("output/markdown")
INPUT_DIR = Path("input/resumes")

POPULATION = [
    ("U1", "178480991686_Vachan_Premugh_Dicruz_Founder_Office_BTTR.MEN",
     "178480991686_Vachan_Premugh_Dicruz_Founder_Office_BTTR.MEN.pdf"),
    ("U2", "8th_April_Akasa_Kapoor_Resume", "8th_April_Akasa_Kapoor_Resume.pdf"),
    ("U3", "AARTI BAGHOTIYA_Economic Urban Planner_Intileo",
     "AARTI BAGHOTIYA_Economic Urban Planner_Intileo.pdf"),
    ("U4", "AASHISH PARMAR - Email Marketing Executive - Intileo",
     "AASHISH PARMAR - Email Marketing Executive - Intileo.pdf"),
    ("U5", "_Adarsh_Bhagat Resume", "_Adarsh_Bhagat Resume.pdf"),
    ("001", "AMAN_CV_BCA", "AMAN_CV_BCA.pdf"),
    ("006", "AshishKumarSingh_RBM_Pune_Stonelam", "AshishKumarSingh_RBM_Pune_Stonelam.pdf"),
    ("013", "Kishan Yadav_Data Scientist_Intileo", "Kishan Yadav_Data Scientist_Intileo.pdf"),
    ("021", "PRAGNESH S. SHAH_Sr. Associate-IT Ops_Intileo",
     "PRAGNESH S. SHAH_Sr. Associate-IT Ops_Intileo.pdf"),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase_a_freeze_and_display():
    print("=" * 70)
    print("T21B PHASE A -- FROZEN STATE (before any LLM call)")
    print("=" * 70)

    print(f"\nModel: {MODEL}")
    print("Planned calls: 9 (1 per resume)")
    print("Max calls if every resume needs 1 structural retry: 18")

    print("\n--- Population (9 resumes, frozen, identical to T21) ---")
    total_candidates = 0
    inventories = {}
    for rid, stem, pdf_name in POPULATION:
        pdf_hash = _sha256(INPUT_DIR / pdf_name) if (INPUT_DIR / pdf_name).exists() else "N/A (control, not re-hashed here; see T21B report for benchmark-manifest hash)"
        blocks = blockify((MARKDOWN_DIR / f"{stem}.md").read_text(encoding="utf-8"))
        candidates = generate_candidates_t20(blocks)
        inventory = build_candidate_inventory(blocks, candidates)
        inventories[rid] = inventory
        total_candidates += len(candidates)
        print(f"  {rid:4} {stem[:45]:45} candidates={len(candidates):3}  input_sha256={pdf_hash[:16]}...")

    print(f"\nTotal candidates across 9 resumes: {total_candidates} (T21 frozen inventory: 151)")
    assert total_candidates == 151, "candidate inventory does not match T21's frozen 151 -- STOP"

    # Cross-check against T21's own frozen CSV artifact byte-for-byte.
    frozen_rows = list(csv.DictReader(open("reports/t21_baseline_candidate_inventory.csv")))
    frozen_by_resume = {}
    for r in frozen_rows:
        frozen_by_resume.setdefault(r["resume_id"], []).append(r)
    mismatches = 0
    for rid, inventory in inventories.items():
        frozen = frozen_by_resume[rid]
        for item, frow in zip(inventory, frozen):
            flags_str = ";".join(item["flags"])
            if (item["candidate_id"] != frow["candidate_id"] or item["anchor_text"] != frow["anchor_text"]
                    or str(item["section_label"]) != frow["section_label"] or str(item["n_blocks"]) != frow["n_blocks"]
                    or flags_str != frow["flags"] or item["text_preview"] != frow["text_preview"]):
                mismatches += 1
    print(f"Byte-for-byte cross-check against T21's frozen reports/t21_baseline_candidate_inventory.csv: "
          f"{'IDENTICAL' if mismatches == 0 else f'{mismatches} MISMATCHES -- STOP'}")
    assert mismatches == 0, "candidate inventory diverges from T21's frozen artifact -- STOP"

    print("\n--- Frozen prompt (identical for all 9 resumes except the candidate list) ---")
    example_prompt = build_prompt(inventories["U3"])  # smallest resume, for display
    print(example_prompt[:3000])
    print(f"... [full prompt length for U3: {len(example_prompt)} chars; "
          f"varies per resume by candidate-list length only]")

    print("\n--- Output contract ---")
    print(json.dumps({
        "candidate_decisions": [
            {"candidate_id": "<id>", "decision": "KEEP|DISCARD|CLASSIFY",
             "classification": "WORK_EXPERIENCE|PROJECT|EDUCATION|OTHER_NON_EXPERIENCE|null"}
        ]
    }, indent=2))
    print("\nNO confirmed_entities field. NO entity IDs. NO merge groups. NO spans.")

    return inventories


def phase_b_execute():
    client = OllamaClient()
    call_log = []
    decision_rows = []

    for rid, stem, _pdf in POPULATION:
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
        })

        if result.parsed:
            for d in result.parsed.get("candidate_decisions", []):
                decision_rows.append({
                    "resume_id": rid, "stem": stem,
                    "candidate_id": d.get("candidate_id"),
                    "decision": d.get("decision"),
                    "classification": d.get("classification"),
                })

        raw_out = Path(f"reports/t21b_raw_llm_response_{rid}.json")
        raw_out.write_text(json.dumps({
            "resume_id": rid, "stem": stem,
            "raw_response": result.raw_response,
            "retry_raw_response": result.retry_raw_response,
            "parsed": result.parsed, "error": result.error,
            "validation_errors": result.validation_errors,
            "structural_retry_used": result.structural_retry_used,
            "latency_seconds": result.latency_seconds,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    out_calls = Path("reports/t21b_call_log.csv")
    with out_calls.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(call_log[0].keys()))
        w.writeheader()
        w.writerows(call_log)
    print(f"\nwrote {out_calls}")

    out_decisions = Path("reports/t21b_candidate_decisions.csv")
    with out_decisions.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["resume_id", "stem", "candidate_id", "decision", "classification"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(decision_rows)
    print(f"wrote {out_decisions}")

    planned = len(POPULATION)
    actual = sum(1 + int(c["structural_retry_used"]) for c in call_log)
    successful = sum(1 for c in call_log if c["success"])
    retries = sum(1 for c in call_log if c["structural_retry_used"])
    failed = sum(1 for c in call_log if not c["success"])

    print("\n=== CALL ACCOUNTING ===")
    print(f"planned calls: {planned}")
    print(f"actual HTTP calls made: {actual}")
    print(f"successful: {successful}")
    print(f"structural retries used: {retries}")
    print(f"semantic retries: 0 (never performed, by design)")
    print(f"failed: {failed}")
    print(f"mean latency: {sum(c['latency_seconds'] for c in call_log)/len(call_log):.2f}s")


if __name__ == "__main__":
    phase_a_freeze_and_display()
    if "--execute" in sys.argv:
        print("\n" + "=" * 70)
        print("PROCEEDING TO PHASE B (LLM CALLS)")
        print("=" * 70 + "\n")
        phase_b_execute()
    else:
        print("\n(Phase A only -- pass --execute to make the frozen LLM calls.)")
