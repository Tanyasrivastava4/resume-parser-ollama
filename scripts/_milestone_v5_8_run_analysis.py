"""
scripts/_milestone_v5_8_run_analysis.py

Milestone V5.8 executable driver. Runs the full-corpus (22-resume)
boundary validation and signal computation defined in
scripts/_milestone_v5_8_full_corpus_validation.py, and writes:

  benchmark/baselines/v5_8_pre_investigation_source_manifest.csv
  benchmark/baselines/v5_8_boundary_validation.csv
  benchmark/baselines/v5_8_full_corpus_signal_distribution.csv

ZERO new LLM calls. Reads only already-existing artifacts (see the
module docstring in _milestone_v5_8_full_corpus_validation.py for the
exact provenance of each one).

This script does NOT write v5_8_policy_calibration_assessment.csv or
v5_8_false_positive_analysis.csv -- those require the human-authored
manual boundary review (Step 2's explicit "manually inspect all
ambiguous or questionable boundary cases" instruction) and the offline
GT-based discrimination analysis, both done separately after reviewing
this script's own output, and written directly as part of the V5.8
report-authoring step, not auto-generated from this script's signals
alone.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_v5_8_full_corpus_validation import (  # noqa: E402
    GT_MAPPING,
    MARKDOWN_DIR,
    PDF_DIR,
    apparent_comma_item_count,
    find_skills_section,
    in_section_frac,
    load_baseline_skills,
    load_mapping,
)

OUT_DIR = Path("benchmark/baselines")

PRODUCTION_SOURCE_FILES = [
    "config.py",
    "settings.py",
    "pipeline/parser_pipeline.py",
    "evaluation/parser_evaluator.py",
    "validators/json_comparator.py",
    "prompts/modules/skills.md",
]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_source_manifest() -> None:
    rows = []
    for rel in PRODUCTION_SOURCE_FILES:
        p = Path(rel)
        if p.exists():
            rows.append({"path": rel, "sha256": sha256_of(p), "size": p.stat().st_size, "present": True})
        else:
            rows.append({"path": rel, "sha256": "", "size": "", "present": False})
    out = OUT_DIR / "v5_8_pre_investigation_source_manifest.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "sha256", "size", "present"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


def main() -> None:
    write_source_manifest()

    mapping_rows = load_mapping()
    print(f"Loaded {len(mapping_rows)} rows from {GT_MAPPING}")

    boundary_rows = []
    signal_rows = []

    for row in mapping_rows:
        resume_id = row["ResumeID"]
        pdf_name = row["PDF"]
        md_name = row["Markdown"]
        gt_json_name = row["JSON"]

        md_path = MARKDOWN_DIR / md_name
        pdf_path = PDF_DIR / pdf_name

        md_present = md_path.exists()
        pdf_present = pdf_path.exists()

        baseline = load_baseline_skills(md_name)
        baseline_status = baseline["status"]
        extracted = baseline["actual"] or []
        gt_skills = baseline["expected"] or []

        if not md_present:
            boundary_rows.append({
                "resume_id": resume_id, "pdf": pdf_name, "markdown": md_name,
                "markdown_present": md_present, "pdf_present": pdf_present,
                "found": False, "is_fallback": False, "heading_matched": "",
                "start_line": "", "end_line": "", "body_char_count": 0,
                "apparent_comma_item_count": "", "multiple_candidates": "",
                "note": "markdown file missing", "classification": "NOT_FOUND",
            })
            signal_rows.append({
                "resume_id": resume_id, "baseline_status": baseline_status,
                "extracted_count": len(extracted), "gt_count_OFFLINE_ONLY": len(gt_skills),
                "apparent_comma_item_count": "", "fill_ratio": "",
                "in_section_frac": "", "in_section_found": "", "in_section_total": "",
                "boundary_classification": "NOT_FOUND",
                "signal_valid": False, "signal_invalid_reason": "markdown source missing",
                "would_escalate_v5_7_policy": "",
            })
            continue

        text = md_path.read_text(encoding="utf-8", errors="replace")
        section = find_skills_section(text)

        n_apparent = apparent_comma_item_count(section["body_text"]) if section["found"] else None
        frac, nf, nt = in_section_frac(extracted, section["body_text"]) if section["found"] else (None, 0, 0)

        boundary_rows.append({
            "resume_id": resume_id, "pdf": pdf_name, "markdown": md_name,
            "markdown_present": md_present, "pdf_present": pdf_present,
            "found": section["found"], "is_fallback": section["is_fallback"],
            "heading_matched": section["heading_matched"] or "",
            "start_line": section["start_line"] if section["start_line"] is not None else "",
            "end_line": section["end_line"] if section["end_line"] is not None else "",
            "body_char_count": len(section["body_text"]),
            "apparent_comma_item_count": n_apparent if n_apparent is not None else "",
            "multiple_candidates": "; ".join(section["multiple_candidates"]),
            "note": section["note"],
            "classification": "",  # filled in by manual review pass, see report
        })

        if baseline_status != "ok":
            signal_valid = False
            invalid_reason = f"baseline checkpoint status={baseline_status}"
        elif not section["found"]:
            signal_valid = False
            invalid_reason = "Skills section not found"
        elif not n_apparent:
            signal_valid = False
            invalid_reason = "apparent_comma_item_count is 0 (fill_ratio undefined, division by zero)"
        elif frac is None:
            signal_valid = False
            invalid_reason = "extracted_count is 0 (in_section_frac undefined)"
        else:
            signal_valid = True
            invalid_reason = ""

        fill_ratio = (len(extracted) / n_apparent) if (signal_valid and n_apparent) else None
        would_escalate = ""
        if signal_valid:
            would_escalate = bool(fill_ratio < 0.9 and frac is not None and frac >= 0.9)

        signal_rows.append({
            "resume_id": resume_id,
            "baseline_status": baseline_status,
            "extracted_count": len(extracted),
            "gt_count_OFFLINE_ONLY": len(gt_skills),
            "apparent_comma_item_count": n_apparent if n_apparent is not None else "",
            "fill_ratio": round(fill_ratio, 4) if fill_ratio is not None else "",
            "in_section_frac": round(frac, 4) if frac is not None else "",
            "in_section_found": nf,
            "in_section_total": nt,
            "boundary_classification": "",  # filled in by manual review pass
            "signal_valid": signal_valid,
            "signal_invalid_reason": invalid_reason,
            "would_escalate_v5_7_policy": would_escalate,
        })

    boundary_out = OUT_DIR / "v5_8_boundary_validation.csv"
    with boundary_out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(boundary_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(boundary_rows)
    print(f"wrote {boundary_out} ({len(boundary_rows)} rows)")

    signal_out = OUT_DIR / "v5_8_full_corpus_signal_distribution.csv"
    with signal_out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(signal_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(signal_rows)
    print(f"wrote {signal_out} ({len(signal_rows)} rows)")

    n_eligible = sum(1 for r in signal_rows if r["signal_valid"])
    n_escalate = sum(1 for r in signal_rows if r["would_escalate_v5_7_policy"] is True)
    print(f"\nSUMMARY: {len(signal_rows)} resumes total, {n_eligible} signal-eligible, "
          f"{n_escalate} would escalate under V5.7's provisional 0.9/0.9 policy "
          f"({100*n_escalate/n_eligible:.1f}% of eligible)" if n_eligible else "no eligible resumes")


if __name__ == "__main__":
    main()
