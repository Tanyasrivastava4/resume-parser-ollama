"""
scripts/_milestone_v5_9_full_corpus_revalidation.py

Milestone V5.9 Phase 5 -- re-run the CORRECTED boundary/signal
measurement logic (scripts/_milestone_v5_9_signal_corrections.py)
across all 22 resumes, using the exact same Milestone-23.67 frozen
checkpoints and output/markdown/ source V5.8 used (zero new LLM
calls), and compare against V5.8's own committed results.

The ORIGINAL, UNMODIFIED V5.7 policy (fill_ratio < 0.9 AND
in_section_frac >= 0.9) is used as the ONLY comparison instrument --
this script does not tune, adjust, or recommend a new threshold. Any
exploratory threshold analysis is out of scope for this script by
design (kept separate, clearly labeled, in the milestone report only).

Writes:
  benchmark/baselines/v5_9_boundary_validation.csv
  benchmark/baselines/v5_9_full_corpus_signal_distribution.csv
  benchmark/baselines/v5_9_before_after_comparison.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_v5_8_full_corpus_validation import (  # noqa: E402
    MARKDOWN_DIR,
    PDF_DIR,
    load_baseline_skills,
    load_mapping,
)
from scripts._milestone_v5_9_signal_corrections import (  # noqa: E402
    apparent_comma_item_count,
    find_skills_section,
    in_section_frac,
)

OUT_DIR = Path("benchmark/baselines")
FILL_RATIO_THRESHOLD = 0.9   # UNCHANGED from V5.7's original policy
IN_SECTION_THRESHOLD = 0.9   # UNCHANGED from V5.7's original policy


def load_v58_signal_csv() -> dict:
    with (OUT_DIR / "v5_8_full_corpus_signal_distribution.csv").open(newline="", encoding="utf-8") as f:
        return {r["resume_id"]: r for r in csv.DictReader(f)}


def load_v58_boundary_csv() -> dict:
    with (OUT_DIR / "v5_8_boundary_validation.csv").open(newline="", encoding="utf-8") as f:
        return {r["resume_id"]: r for r in csv.DictReader(f)}


def main() -> None:
    v58_signal = load_v58_signal_csv()
    v58_boundary = load_v58_boundary_csv()

    boundary_rows = []
    signal_rows = []
    compare_rows = []

    for row in load_mapping():
        resume_id = row["ResumeID"]
        pdf_name = row["PDF"]
        md_name = row["Markdown"]

        md_path = MARKDOWN_DIR / md_name
        pdf_path = PDF_DIR / pdf_name
        md_present = md_path.exists()
        pdf_present = pdf_path.exists()

        baseline = load_baseline_skills(md_name)
        baseline_status = baseline["status"]
        extracted = baseline["actual"] or []

        if not md_present:
            boundary_rows.append({
                "resume_id": resume_id, "found": False, "boundary_content_implausible": "",
                "excluded_nonskills_spans": "", "note": "markdown missing",
            })
            signal_rows.append({
                "resume_id": resume_id, "baseline_status": baseline_status,
                "extracted_count": len(extracted), "apparent_comma_item_count": "",
                "fill_ratio": "", "in_section_frac": "", "signal_valid": False,
                "signal_invalid_reason": "markdown source missing",
                "would_escalate_v5_7_policy": "",
            })
            continue

        text = md_path.read_text(encoding="utf-8", errors="replace")
        sec = find_skills_section(text)

        boundary_rows.append({
            "resume_id": resume_id,
            "found": sec["found"],
            "heading_matched": sec.get("heading_matched") or "",
            "boundary_content_implausible": sec.get("boundary_content_implausible", ""),
            "excluded_nonskills_spans": str(sec.get("excluded_nonskills_spans", [])),
            "body_char_count_original": len(sec.get("body_text", "")),
            "body_char_count_cleaned": len(sec.get("cleaned_body_text", "")),
            "note": sec.get("note", ""),
        })

        if not sec["found"]:
            signal_rows.append({
                "resume_id": resume_id, "baseline_status": baseline_status,
                "extracted_count": len(extracted), "apparent_comma_item_count": "",
                "fill_ratio": "", "in_section_frac": "", "signal_valid": False,
                "signal_invalid_reason": "Skills section not found",
                "would_escalate_v5_7_policy": "",
            })
            continue

        if sec["boundary_content_implausible"]:
            signal_rows.append({
                "resume_id": resume_id, "baseline_status": baseline_status,
                "extracted_count": len(extracted), "apparent_comma_item_count": "",
                "fill_ratio": "", "in_section_frac": "", "signal_valid": False,
                "signal_invalid_reason": "boundary content structurally implausible (table-like) -- V5.9 new safety check",
                "would_escalate_v5_7_policy": "",
            })
            continue

        n_apparent = apparent_comma_item_count(sec["cleaned_body_text"])
        frac, nf, nt = in_section_frac(extracted, sec["cleaned_body_text"])

        if baseline_status != "ok":
            signal_valid, reason = False, f"baseline checkpoint status={baseline_status}"
        elif not n_apparent:
            signal_valid, reason = False, "apparent_comma_item_count is 0"
        elif frac is None:
            signal_valid, reason = False, "extracted_count is 0"
        else:
            signal_valid, reason = True, ""

        fill_ratio = (len(extracted) / n_apparent) if (signal_valid and n_apparent) else None
        would_escalate = ""
        if signal_valid:
            would_escalate = bool(fill_ratio < FILL_RATIO_THRESHOLD and frac is not None and frac >= IN_SECTION_THRESHOLD)

        signal_rows.append({
            "resume_id": resume_id, "baseline_status": baseline_status,
            "extracted_count": len(extracted),
            "apparent_comma_item_count": n_apparent,
            "fill_ratio": round(fill_ratio, 4) if fill_ratio is not None else "",
            "in_section_frac": round(frac, 4) if frac is not None else "",
            "signal_valid": signal_valid,
            "signal_invalid_reason": reason,
            "would_escalate_v5_7_policy": would_escalate,
        })

        # --- before/after comparison against V5.8's own committed results ---
        v58s = v58_signal.get(resume_id, {})
        v58b = v58_boundary.get(resume_id, {})
        compare_rows.append({
            "resume_id": resume_id,
            "v58_apparent_comma_item_count": v58s.get("apparent_comma_item_count", ""),
            "v59_apparent_comma_item_count": n_apparent,
            "v58_fill_ratio": v58s.get("fill_ratio", ""),
            "v59_fill_ratio": round(fill_ratio, 4) if fill_ratio is not None else "",
            "v58_would_escalate": v58s.get("would_escalate_v5_7_policy", ""),
            "v59_would_escalate": would_escalate,
            "escalation_decision_changed": str(v58s.get("would_escalate_v5_7_policy", "")) != str(would_escalate),
            "v58_boundary_classification": v58b.get("classification", ""),
            "v59_boundary_content_implausible": sec.get("boundary_content_implausible", ""),
        })

    with (OUT_DIR / "v5_9_boundary_validation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(boundary_rows[0].keys()))
        w.writeheader()
        w.writerows(boundary_rows)
    print(f"wrote v5_9_boundary_validation.csv ({len(boundary_rows)} rows)")

    with (OUT_DIR / "v5_9_full_corpus_signal_distribution.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(signal_rows[0].keys()))
        w.writeheader()
        w.writerows(signal_rows)
    print(f"wrote v5_9_full_corpus_signal_distribution.csv ({len(signal_rows)} rows)")

    with (OUT_DIR / "v5_9_before_after_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(compare_rows[0].keys()))
        w.writeheader()
        w.writerows(compare_rows)
    print(f"wrote v5_9_before_after_comparison.csv ({len(compare_rows)} rows)")

    n_eligible = sum(1 for r in signal_rows if r["signal_valid"])
    n_escalate = sum(1 for r in signal_rows if r["would_escalate_v5_7_policy"] is True)
    n_changed = sum(1 for r in compare_rows if r["escalation_decision_changed"] == "True" or r["escalation_decision_changed"] is True)
    print(f"\nSUMMARY: {len(signal_rows)} resumes total, {n_eligible} signal-eligible, "
          f"{n_escalate} would escalate under the ORIGINAL 0.9/0.9 policy "
          f"({100*n_escalate/n_eligible:.1f}% of eligible)" if n_eligible else "no eligible")
    print(f"Escalation decision changed vs V5.8 for {n_changed} resumes")


if __name__ == "__main__":
    main()
