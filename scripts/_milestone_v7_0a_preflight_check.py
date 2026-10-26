"""
scripts/_milestone_v7_0a_preflight_check.py

Milestone V7.0A -- supersedes scripts/_milestone_v7_0_preflight_check.py
for the future, NOT-YET-AUTHORIZED Skills selective-escalation
regression-corpus-extension experiment, after V7.0A's own protocol-
closure investigation:

  1. Resume 022 is REMOVED from the population (V7.0A Part B:
     classification INSUFFICIENT_EVIDENCE, not CONFIRMED -- excluded
     per a rule frozen before any new-data examination; see the
     milestone report).
  2. A genuine gap in V7.0's own pre-flight script is closed: that
     script computed and PRINTED a source-file hash but had nothing to
     compare it against (no prior frozen value existed), so it could
     not actually detect drift. This script compares against
     `benchmark/baselines/v7_0a_frozen_source_file_hashes.csv`, frozen
     in this milestone, closing that gap GOING FORWARD.
  3. A disclosed, NOT fully resolved limitation carries forward from
     V7.0A Part A: `output/` is git-ignored (confirmed via
     `git check-ignore`), so no cryptographic chain exists proving
     resume 005's current source file is byte-identical to what V5.6
     actually read at its own execution time. This script can only
     detect drift SINCE this milestone's own freeze point, not before
     it -- this is stated explicitly, not hidden.

This script makes ZERO LLM/model calls -- read-only, deterministic
checks only. RESEARCH-ONLY -- must never be imported by, or copied
into, `services/` or `pipeline/`.

Exit code 0 = all checks passed. Exit code 1 = at least one check failed.
"""
from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

FROZEN_PROMPT_SHA256 = "1b16785169cde3e370c62c57ed2ba5a3604d21074e0ab99af5f1932d3ccc608a"
FROZEN_SCHEMA_SHA256 = "994b555f90c80578a269e897e816ef1027754b38b1f7c4668d8652a1b880d218"
FROZEN_BASELINE_MODEL = "qwen3:1.7b"
FROZEN_CANDIDATE_MODEL = "qwen3:8b"

# FINAL frozen population per V7.0A closure -- 022 removed.
REQUIRED_RESUMES = {
    "016": {"stem": "Mukul Kanojia_Full Stack Developer_Intileo", "lineage": "output/markdown"},
    "005": {"stem": "Arjun Sharma_Data Architect_GT", "lineage": "output/v2_4h_benchmark/stage1/markdown"},
}
RUNS_PER_CONDITION = 5
CONDITIONS = 2
PLANNED_CALLS = len(REQUIRED_RESUMES) * CONDITIONS * RUNS_PER_CONDITION  # 20
HARD_CALL_CAP = 30  # 20 planned + 10 buffer, matching V6.0/V6.1's own +10 precedent


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_production_manifest() -> bool:
    frozen = Path("benchmark/baselines/v7_0a_post_investigation_source_manifest.csv")
    if not frozen.exists():
        print("FAIL: v7_0a_post_investigation_source_manifest.csv missing")
        return False
    with frozen.open() as f:
        frozen_rows = {r["path"]: r["sha256"] for r in csv.DictReader(f)}
    mismatches = []
    for path, expected_hash in frozen_rows.items():
        p = Path(path)
        if not p.exists():
            mismatches.append(f"{path}: MISSING")
            continue
        if sha256_of(p) != expected_hash:
            mismatches.append(f"{path}: HASH CHANGED")
    if mismatches:
        print(f"FAIL: production manifest drift ({len(mismatches)} files):")
        for m in mismatches:
            print(f"  {m}")
        return False
    print(f"PASS: production-source manifest byte-identical ({len(frozen_rows)} files)")
    return True


def check_gt_pdf_manifests() -> bool:
    ok = True

    def check(manifest_path, base_dir, name_col, label):
        nonlocal ok
        rows = list(csv.DictReader(open(manifest_path)))
        missing = mismatch = 0
        for row in rows:
            p = Path(base_dir) / row[name_col]
            if not p.exists():
                missing += 1
                continue
            if sha256_of(p) != row["sha256"]:
                mismatch += 1
        status = "PASS" if (missing == 0 and mismatch == 0) else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"{status}: {label}: {len(rows)} rows, {missing} missing, {mismatch} mismatches")

    check("benchmark/baselines/milestone_23_67_gt_manifest.csv",
          "benchmark/ground_truth/adjudicated_v1", "filename", "GT manifest")
    check("benchmark/baselines/milestone_23_67_input_manifest.csv",
          "input/resumes", "pdf_filename", "PDF manifest")
    return ok


def check_prompt_schema_frozen() -> bool:
    ok = True
    actual_prompt = sha256_of(Path("prompts/modules/skills.md"))
    actual_schema = sha256_of(Path("schemas/modules/skills_schema.json"))
    if actual_prompt != FROZEN_PROMPT_SHA256:
        print(f"FAIL: prompts/modules/skills.md hash changed")
        ok = False
    else:
        print("PASS: prompts/modules/skills.md unchanged")
    if actual_schema != FROZEN_SCHEMA_SHA256:
        print(f"FAIL: schemas/modules/skills_schema.json hash changed")
        ok = False
    else:
        print("PASS: schemas/modules/skills_schema.json unchanged")
    return ok


def check_default_model_unchanged() -> bool:
    import settings
    if settings.DEFAULT_TEXT_MODEL != FROZEN_BASELINE_MODEL:
        print(f"FAIL: settings.DEFAULT_TEXT_MODEL is {settings.DEFAULT_TEXT_MODEL!r}, expected {FROZEN_BASELINE_MODEL!r}")
        return False
    print(f"PASS: settings.DEFAULT_TEXT_MODEL == {FROZEN_BASELINE_MODEL!r} (unchanged)")
    return True


def check_models_installed() -> bool:
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        print("FAIL: `ollama` executable not found")
        return False
    text = out.stdout
    ok = True
    for model in (FROZEN_BASELINE_MODEL, FROZEN_CANDIDATE_MODEL):
        if model not in text:
            print(f"FAIL: {model} not found in `ollama list` output")
            ok = False
        else:
            print(f"PASS: {model} present in `ollama list`")
    return ok


def check_source_artifacts_hash_locked() -> bool:
    """Compares each frozen population resume's CURRENT source-file
    hash against the value frozen in v7_0a_frozen_source_file_hashes.csv
    -- unlike V7.0's own check (which only ever printed a fresh hash
    with nothing to compare against), this can actually detect drift
    SINCE this milestone's own freeze point.

    Disclosed limitation, not silently hidden: this CANNOT detect
    drift BEFORE this freeze point (e.g. whether resume 005's file
    differs from what V5.6 originally read) -- output/ is git-ignored
    (confirmed via `git check-ignore`) and no hash of this file was
    ever recorded at V5.6's own execution time.
    """
    frozen_path = Path("benchmark/baselines/v7_0a_frozen_source_file_hashes.csv")
    if not frozen_path.exists():
        print("FAIL: v7_0a_frozen_source_file_hashes.csv missing")
        return False
    with frozen_path.open() as f:
        frozen = {r["resume_id"]: r for r in csv.DictReader(f)}

    ok = True
    for rid, info in REQUIRED_RESUMES.items():
        p = Path(info["lineage"]) / f"{info['stem']}.md"
        if not p.exists():
            print(f"FAIL: source artifact missing for {rid}: {p}")
            ok = False
            continue
        actual = sha256_of(p)
        expected = frozen.get(rid, {}).get("sha256")
        if expected is None:
            print(f"FAIL: no frozen hash recorded for {rid}")
            ok = False
        elif actual != expected:
            print(f"FAIL: {rid} source file hash CHANGED since V7.0A freeze "
                  f"(expected {expected[:12]}..., got {actual[:12]}...)")
            ok = False
        else:
            print(f"PASS: {rid} source file hash-locked, unchanged since V7.0A freeze ({actual[:12]}...)")
    return ok


def check_signal_population_unchanged() -> bool:
    from scripts._milestone_v5_8_full_corpus_validation import MARKDOWN_DIR, load_baseline_skills, load_mapping
    from scripts._milestone_v5_9_signal_corrections import apparent_comma_item_count, find_skills_section, in_section_frac

    escalate_now = set()
    for row in load_mapping():
        rid = row["ResumeID"]
        md_path = MARKDOWN_DIR / row["Markdown"]
        if not md_path.exists():
            continue
        baseline = load_baseline_skills(row["Markdown"])
        if baseline["status"] != "ok":
            continue
        extracted = baseline["actual"] or []
        text = md_path.read_text(encoding="utf-8", errors="replace")
        sec = find_skills_section(text)
        if not sec["found"] or sec["boundary_content_implausible"]:
            continue
        n = apparent_comma_item_count(sec["cleaned_body_text"])
        frac, _, _ = in_section_frac(extracted, sec["cleaned_body_text"])
        if not n or frac is None:
            continue
        fill_ratio = len(extracted) / n
        if fill_ratio < 0.9 and frac >= 0.9:
            escalate_now.add(rid)

    expected = {"007", "013", "015", "018"}
    if escalate_now != expected:
        print(f"FAIL: signal population drifted. Expected {sorted(expected)}, got {sorted(escalate_now)}")
        return False
    print(f"PASS: signal population unchanged: {sorted(escalate_now)}")
    return True


def check_call_budget() -> bool:
    print(f"Planned calls (final frozen population, 016+005 only, 022 excluded): {PLANNED_CALLS}")
    print(f"Hard call cap: {HARD_CALL_CAP}")
    if PLANNED_CALLS > HARD_CALL_CAP:
        print("FAIL: planned calls exceed hard cap")
        return False
    print("PASS: planned call count within hard cap")
    return True


def check_output_directory_safety() -> bool:
    n_files = sum(1 for _ in Path("output").rglob("*") if _.is_file())
    print(f"output/ file count: {n_files} (future execution session must re-check this is UNCHANGED after every call batch)")
    return True


def main() -> int:
    print("=== Milestone V7.0A pre-flight check (zero LLM calls, DESIGN VALIDATION ONLY) ===\n")
    print("Supersedes V7.0's own pre-flight script: 022 removed from population; "
          "source-file hash-lock now has a real comparison target.\n")
    print("DISCLOSED, NOT FULLY RESOLVED LIMITATION: output/ is git-ignored; resume 005's "
          "byte-identity to what V5.6 originally read cannot be cryptographically proven, only "
          "supported by strong circumstantial content-continuity evidence (see milestone report). "
          "This script detects drift SINCE this milestone's freeze point only.\n")
    checks = [
        check_production_manifest(),
        check_gt_pdf_manifests(),
        check_prompt_schema_frozen(),
        check_default_model_unchanged(),
        check_models_installed(),
        check_source_artifacts_hash_locked(),
        check_signal_population_unchanged(),
        check_call_budget(),
        check_output_directory_safety(),
    ]
    print(f"\n{sum(checks)}/{len(checks)} checks passed.")
    if all(checks):
        print("ALL CHECKS PASSED. (This does not authorize execution.)")
        return 0
    print("AT LEAST ONE CHECK FAILED. Do not proceed to execution.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
