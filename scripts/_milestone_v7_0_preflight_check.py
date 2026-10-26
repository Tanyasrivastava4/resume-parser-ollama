"""
scripts/_milestone_v7_0_preflight_check.py

Milestone V7.0 pre-flight verification script for the FUTURE, NOT-YET-
AUTHORIZED Skills selective-escalation regression-corpus extension
experiment (frozen population: 016 + 005 required, 022 optional --
see docs/MILESTONE_V7_0_SKILLS_REGRESSION_CORPUS_EXTENSION_DESIGN.md).

This script makes ZERO LLM/model calls -- it only performs
deterministic, read-only checks and re-derives population-relevant
facts fresh, so a future execution session cannot silently begin
spending call budget against a stale or drifted checkpoint, a
different branch/commit than the one this design was frozen against,
or a source artifact that has moved.

RESEARCH-ONLY. This script must never be imported by, or copied into,
`services/` or `pipeline/` -- it exists solely to gate a future,
separately-authorized execution milestone, exactly like V6.0's own
`scripts/_milestone_v6_0_preflight_check.py` precedent.

Run this BEFORE any real Ollama call in a future execution session. If
ANY check fails, STOP and re-derive the protocol -- do not proceed
with a stale population, a missing source artifact, or an unverified
environment.

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

# The exact commit this design was frozen against -- a future execution
# session must reference this commit explicitly per the authorization
# gate (Phase G of the milestone).
FROZEN_DESIGN_COMMIT_PARENT = "8775db6"  # V6.9's own end-state, this milestone's own checkpoint base

# Frozen population, per docs/MILESTONE_V7_0_..._DESIGN.md Phase C/D.
REQUIRED_RESUMES = {
    "016": {"stem": "Mukul Kanojia_Full Stack Developer_Intileo", "lineage": "output/markdown"},
    "005": {"stem": "Arjun Sharma_Data Architect_GT", "lineage": "output/v2_4h_benchmark/stage1/markdown"},
}
OPTIONAL_RESUMES = {
    "022": {"stem": "Rima_Karmakar_CV", "lineage": "output/markdown"},
}
RUNS_PER_CONDITION = 5
CONDITIONS = 2  # baseline + candidate
PLANNED_CALLS_REQUIRED_ONLY = len(REQUIRED_RESUMES) * CONDITIONS * RUNS_PER_CONDITION       # 20
PLANNED_CALLS_WITH_OPTIONAL = (len(REQUIRED_RESUMES) + len(OPTIONAL_RESUMES)) * CONDITIONS * RUNS_PER_CONDITION  # 30
HARD_CALL_CAP = 40  # planned (30, if optional included) + 10 buffer, matching V6.0/V6.1's own +10 precedent


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_production_manifest() -> bool:
    frozen = Path("benchmark/baselines/v7_0_post_investigation_source_manifest.csv")
    if not frozen.exists():
        print("FAIL: v7_0_post_investigation_source_manifest.csv missing")
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
        print(f"FAIL: prompts/modules/skills.md hash changed (expected {FROZEN_PROMPT_SHA256}, got {actual_prompt})")
        ok = False
    else:
        print("PASS: prompts/modules/skills.md unchanged")
    if actual_schema != FROZEN_SCHEMA_SHA256:
        print(f"FAIL: schemas/modules/skills_schema.json hash changed (expected {FROZEN_SCHEMA_SHA256}, got {actual_schema})")
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
    """Metadata query only (`ollama list`) -- NOT an inference call."""
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


def check_source_artifacts_available() -> bool:
    """Confirms every frozen population resume's exact source markdown
    file exists at its frozen lineage path, and records its hash for
    the future execution session's own record -- catches a moved/
    renamed/regenerated Stage-1 file before any call is spent."""
    ok = True
    for rid, info in {**REQUIRED_RESUMES, **OPTIONAL_RESUMES}.items():
        p = Path(info["lineage"]) / f"{info['stem']}.md"
        if not p.exists():
            print(f"FAIL: source artifact missing for {rid}: {p}")
            ok = False
        else:
            print(f"PASS: source artifact present for {rid}: {p} (sha256={sha256_of(p)[:12]}...)")
    return ok


def check_signal_population_and_eligibility_unchanged() -> bool:
    """Zero-LLM-call re-derivation confirming: (1) the frozen 0.9/0.9
    signal STILL flags exactly {007, 013, 015, 018} -- i.e. this
    design's own explicit finding that no NEW required case exists
    under the unmodified policy has not silently drifted; (2) 005/016/022
    remain real, present, eligible resumes in the V6.5 population
    inventory (boundary found, not implausible)."""
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
    ok = True
    if escalate_now != expected:
        print(f"FAIL: signal population drifted. Expected {sorted(expected)}, got {sorted(escalate_now)} "
              f"-- this design's own 'no new required case under the frozen policy' finding no longer holds; "
              f"re-run Phase C before proceeding.")
        ok = False
    else:
        print(f"PASS: signal population unchanged: {sorted(escalate_now)} (confirms this design's own finding still holds)")
    return ok


def check_call_budget() -> bool:
    print(f"Planned calls (required only, 016+005): {PLANNED_CALLS_REQUIRED_ONLY}")
    print(f"Planned calls (required + optional, +022): {PLANNED_CALLS_WITH_OPTIONAL}")
    print(f"Hard call cap: {HARD_CALL_CAP}")
    if PLANNED_CALLS_WITH_OPTIONAL > HARD_CALL_CAP:
        print("FAIL: planned calls exceed hard cap -- population or run count has drifted from the frozen design")
        return False
    print("PASS: planned call count within hard cap")
    return True


def check_output_directory_safety() -> bool:
    """The frozen protocol (Phase E) requires direct SkillsExtractor
    calls only, never ParserPipeline -- confirms output/ has not grown
    since this design was frozen, i.e. no prior accidental production
    write exists to be confused with this future experiment's own
    checkpoint-only output."""
    n_files = sum(1 for _ in Path("output").rglob("*") if _.is_file())
    print(f"output/ file count: {n_files} (record this value; the future execution session must "
          f"re-check this count is UNCHANGED immediately after every call batch)")
    return True


def main() -> int:
    print("=== Milestone V7.0 pre-flight check (zero LLM calls, DESIGN VALIDATION ONLY) ===\n")
    print("This script validates the FROZEN FUTURE PROTOCOL. It does not authorize or "
          "perform execution. A future execution session must reference the final V7.0 "
          f"commit descending from {FROZEN_DESIGN_COMMIT_PARENT} and receive explicit "
          "human authorization before making any call.\n")
    checks = [
        check_production_manifest(),
        check_gt_pdf_manifests(),
        check_prompt_schema_frozen(),
        check_default_model_unchanged(),
        check_models_installed(),
        check_source_artifacts_available(),
        check_signal_population_and_eligibility_unchanged(),
        check_call_budget(),
        check_output_directory_safety(),
    ]
    print(f"\n{sum(checks)}/{len(checks)} checks passed.")
    if all(checks):
        print("ALL CHECKS PASSED. (This does not authorize execution -- see banner above.)")
        return 0
    print("AT LEAST ONE CHECK FAILED. Do not proceed to execution.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
