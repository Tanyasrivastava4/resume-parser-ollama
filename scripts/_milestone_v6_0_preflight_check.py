"""
scripts/_milestone_v6_0_preflight_check.py

Milestone V6.0 pre-flight verification script for the Skills selective-
escalation prospective pilot (Milestone V6.1, NOT executed by this
milestone). This script makes ZERO LLM/model calls -- it only performs
deterministic, read-only checks and re-derives the current signal
population fresh, so that a future execution session cannot silently
begin spending call budget against a stale or drifted checkpoint.

Run this BEFORE any real Ollama call in a future V6.1 execution
session. If ANY check fails, STOP and re-derive the protocol -- do not
proceed with a stale population or an unverified environment.

Exit code 0 = all checks passed. Exit code 1 = at least one check
failed (see printed detail).
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

# The exact population this protocol predeclared, derived from the
# real V5.9 committed artifacts (see the milestone report §Phase 1).
# Re-derived fresh below, not hardcoded trust -- this constant is only
# the EXPECTED value to compare the fresh re-derivation against.
EXPECTED_ESCALATE = {"007", "013", "015", "018"}
EXPECTED_NOT_ESCALATE_CONTROL = {"016"}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_production_manifest() -> bool:
    """Re-verify the 58-file production-source manifest is still
    byte-identical to the one frozen at V6.0's own checkpoint."""
    frozen = Path("benchmark/baselines/v6_0_pre_investigation_source_manifest.csv")
    if not frozen.exists():
        print("FAIL: v6_0_pre_investigation_source_manifest.csv missing")
        return False
    with frozen.open(newline="", encoding="utf-8") as f:
        frozen_rows = {r["path"]: r["sha256"] for r in csv.DictReader(f)}
    mismatches = []
    for path, expected_hash in frozen_rows.items():
        p = Path(path)
        if not p.exists():
            mismatches.append(f"{path}: MISSING")
            continue
        actual_hash = sha256_of(p)
        if actual_hash != expected_hash:
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
        print(f"FAIL: prompts/modules/skills.md hash changed since V6.0 freeze "
              f"(expected {FROZEN_PROMPT_SHA256}, got {actual_prompt})")
        ok = False
    else:
        print("PASS: prompts/modules/skills.md unchanged since V6.0 freeze")
    if actual_schema != FROZEN_SCHEMA_SHA256:
        print(f"FAIL: schemas/modules/skills_schema.json hash changed since V6.0 freeze "
              f"(expected {FROZEN_SCHEMA_SHA256}, got {actual_schema})")
        ok = False
    else:
        print("PASS: schemas/modules/skills_schema.json unchanged since V6.0 freeze")
    return ok


def check_default_model_unchanged() -> bool:
    import settings
    if settings.DEFAULT_TEXT_MODEL != FROZEN_BASELINE_MODEL:
        print(f"FAIL: settings.DEFAULT_TEXT_MODEL is {settings.DEFAULT_TEXT_MODEL!r}, "
              f"expected {FROZEN_BASELINE_MODEL!r}")
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


def check_signal_population_unchanged() -> bool:
    """Zero-LLM-call re-derivation of the current signal-qualified
    population using V5.9's own corrected, committed logic -- confirms
    the population this protocol was designed around has not silently
    drifted (e.g. from an unrelated markdown/checkpoint change) before
    any call budget is spent."""
    from scripts._milestone_v5_8_full_corpus_validation import (
        MARKDOWN_DIR, load_baseline_skills, load_mapping,
    )
    from scripts._milestone_v5_9_signal_corrections import (
        apparent_comma_item_count, find_skills_section, in_section_frac,
    )

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

    if escalate_now != EXPECTED_ESCALATE:
        print(f"FAIL: signal population drifted. Expected {sorted(EXPECTED_ESCALATE)}, "
              f"got {sorted(escalate_now)}")
        return False
    print(f"PASS: signal population unchanged: {sorted(escalate_now)}")
    return True


def main() -> int:
    print("=== Milestone V6.0 pre-flight check (zero LLM calls) ===\n")
    checks = [
        check_production_manifest(),
        check_gt_pdf_manifests(),
        check_prompt_schema_frozen(),
        check_default_model_unchanged(),
        check_models_installed(),
        check_signal_population_unchanged(),
    ]
    print()
    if all(checks):
        print("ALL PRE-FLIGHT CHECKS PASSED. Population and environment match the frozen protocol.")
        print("This script does NOT authorize execution by itself -- separate, explicit human")
        print("authorization referencing this milestone's commit is still required before any call.")
        return 0
    print("PRE-FLIGHT CHECK FAILED. Do not proceed. Re-derive the protocol before any call.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
