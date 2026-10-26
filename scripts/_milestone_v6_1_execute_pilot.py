"""
scripts/_milestone_v6_1_execute_pilot.py

Milestone V6.1 -- EXECUTION of the frozen V6.0 protocol. Makes REAL
Ollama calls (qwen3:1.7b baseline, qwen3:8b candidate). This script
does not design, tune, or reinterpret anything -- every parameter below
is copied directly from the frozen V6.0 protocol
(docs/MILESTONE_V6_0_SKILLS_SELECTIVE_ESCALATION_PROSPECTIVE_PILOT_PROTOCOL.md
and its committed CSVs), not re-derived.

Population (frozen, V6.0 Phase 1): 007, 013, 015, 016, 018 -- exactly
5 resumes, no addition or removal.
Conditions (frozen, V6.0 Phase 2): baseline=qwen3:1.7b,
candidate=qwen3:8b; 5 runs each; 50 planned calls; 60 hard ceiling.
Source (frozen, V6.0 Phase 2): output/markdown/ (Milestone-23.67
lineage) -- NOT output/v2_4h_benchmark/stage1/markdown/.
Extraction contract (frozen, V6.0 Phase 2): SkillsExtractor.extract(
    markdown_path, model_name=...), unmodified, direct call -- NOT the
full ParserPipeline, so output/json_final/ and output/modules/ are
never touched.

Every attempt (success or failure) is checkpointed to a per-slot pickle
file BEFORE moving to the next slot, so an interruption loses at most
the in-flight call. Re-running this script skips any slot whose
checkpoint already exists -- it does NOT re-run successful slots for
convenience, and does NOT silently discard a prior failed attempt (the
failed attempt is preserved inside the slot's own record).

Retry policy (frozen, V6.0 Phase 2): up to 2 attempts per slot (the
extractor's own internal OLLAMA_MAX_RETRIES=3 already retries
transient errors within a single .extract() call; this script adds
exactly one additional outer retry if the whole call still raises,
mirroring V5.6's own disclosed "preserve both attempts" convention).
Both attempts are always recorded in the slot's `attempts` list,
whichever succeeds or not.
"""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_v5_8_full_corpus_validation import (  # noqa: E402
    MARKDOWN_DIR,
    load_mapping,
)
from services.skills_extractor import SkillsExtractor  # noqa: E402

CHECKPOINT_DIR = Path(
    "/private/tmp/claude-501/-Users-ruchigupta-Documents-ResumeAIBenchmark/"
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/v6_1_checkpoints"
)
# NOTE (remediation CI gate fix, 2026-10-17): CHECKPOINT_DIR is this script's original,
# machine-specific scratch location, kept verbatim so the recorded milestone
# stays reproducible on the machine it ran on. It used to be created HERE, at
# import time, which made merely importing this module fail on any other
# machine (PermissionError on /private on the Linux CI runner). It is now
# created lazily, immediately before the first checkpoint path is used.
# Nothing else about the script, its results or its tests changes.

# --- FROZEN, copied verbatim from the V6.0 protocol -- do not edit here ---
POPULATION = ["007", "013", "015", "016", "018"]
CONDITIONS = [("baseline", "qwen3:1.7b"), ("candidate", "qwen3:8b")]
RUNS_PER_CONDITION = 5
MAX_TOTAL_ATTEMPTS = 60
PLANNED_CALLS = len(POPULATION) * len(CONDITIONS) * RUNS_PER_CONDITION  # 50
# ---------------------------------------------------------------------------


def slot_path(rid: str, condition: str, run_idx: int) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"{rid}__{condition}__run{run_idx}.pkl"


def load_existing_slots() -> list[dict]:
    slots = []
    for p in sorted(CHECKPOINT_DIR.glob("*.pkl")):
        with p.open("rb") as f:
            slots.append(pickle.load(f))
    return slots


def total_attempts_so_far() -> int:
    return sum(s["n_attempts"] for s in load_existing_slots())


def write_environment_metadata() -> None:
    meta_path = CHECKPOINT_DIR / "_environment_metadata.json"
    if meta_path.exists():
        return  # captured once, at the start -- never overwritten mid-run
    ollama_list = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=15).stdout
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {
        "ollama_list": ollama_list,
        "python_version": sys.version,
        "git_head_at_start": git_head,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "planned_calls": PLANNED_CALLS,
        "max_total_attempts": MAX_TOTAL_ATTEMPTS,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Environment metadata captured: {meta_path}")


def run_slot(rid: str, condition: str, model: str, run_idx: int, md_path: Path, extractor: SkillsExtractor) -> dict:
    path = slot_path(rid, condition, run_idx)
    if path.exists():
        with path.open("rb") as f:
            existing = pickle.load(f)
        print(f"SKIP (already complete): {rid} {condition} run{run_idx} status={existing['status']}")
        return existing

    attempts = []
    final = None
    status = "failed"
    for attempt_num in (1, 2):
        t0 = time.perf_counter()
        try:
            result = extractor.extract(md_path, model_name=model)
            elapsed = time.perf_counter() - t0
            obj = result.json_object
            skills = obj.get("skills", []) if isinstance(obj, dict) else obj
            attempts.append({
                "attempt": attempt_num, "status": "ok", "elapsed_seconds": round(elapsed, 3),
                "skills": skills, "error": None,
            })
            final = {"skills": skills, "elapsed_seconds": round(elapsed, 3)}
            status = "ok"
            break
        except Exception as e:  # noqa: BLE001 -- deliberately broad: any extractor failure must be recorded, not crash the run
            elapsed = time.perf_counter() - t0
            attempts.append({
                "attempt": attempt_num, "status": "error", "elapsed_seconds": round(elapsed, 3),
                "skills": None, "error": f"{type(e).__name__}: {e}",
            })
            status = "failed"
            print(f"  ATTEMPT {attempt_num} FAILED: {rid} {condition} run{run_idx} -- {type(e).__name__}: {e}")
            continue

    slot = {
        "resume_id": rid, "markdown_filename": md_path.name, "condition": condition,
        "model": model, "run_index": run_idx, "n_attempts": len(attempts),
        "attempts": attempts, "final": final, "status": status,
    }
    with path.open("wb") as f:
        pickle.dump(slot, f)
    print(f"DONE: {rid} {condition} run{run_idx} status={status} n_attempts={len(attempts)}"
          + (f" recall_items={len(final['skills'])} elapsed={final['elapsed_seconds']}s" if final else ""))
    return slot


def main() -> None:
    write_environment_metadata()
    mapping = {r["ResumeID"]: r for r in load_mapping()}
    for rid in POPULATION:
        if rid not in mapping:
            print(f"HARD STOP: resume {rid} not found in mapping -- protocol violation, aborting.")
            return
        md_path = MARKDOWN_DIR / mapping[rid]["Markdown"]
        if not md_path.exists():
            print(f"HARD STOP: source markdown missing for {rid}: {md_path} -- protocol violation, aborting.")
            return

    extractor = SkillsExtractor()

    for rid in POPULATION:
        md_path = MARKDOWN_DIR / mapping[rid]["Markdown"]
        for condition, model in CONDITIONS:
            for run_idx in range(1, RUNS_PER_CONDITION + 1):
                current_total = total_attempts_so_far()
                if not slot_path(rid, condition, run_idx).exists() and current_total >= MAX_TOTAL_ATTEMPTS:
                    print(f"HARD CAP ({MAX_TOTAL_ATTEMPTS}) REACHED -- stopping before {rid} {condition} run{run_idx}.")
                    return
                run_slot(rid, condition, model, run_idx, md_path, extractor)

    all_slots = load_existing_slots()
    n_ok = sum(1 for s in all_slots if s["status"] == "ok")
    n_failed = sum(1 for s in all_slots if s["status"] == "failed")
    print(f"\n=== EXECUTION COMPLETE: {len(all_slots)}/{PLANNED_CALLS} slots, "
          f"{n_ok} ok, {n_failed} failed, {total_attempts_so_far()} total attempts ===")


if __name__ == "__main__":
    main()
