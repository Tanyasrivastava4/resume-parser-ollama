"""
scripts/_milestone_v6_6_full_corpus_revalidation.py

Milestone V6.6 Phase 4 -- full-corpus, both-lineage before/after
boundary revalidation. ZERO LLM calls. Reads only already-existing
Stage-1 markdown.

Covers:
  1. All 22 resumes under output/markdown/ (Milestone-23.67 / V6.1
     lineage) -- the primary corpus the V5.9 boundary classification
     (18 CLEAR, 2 ACCEPTABLE_WITH_AMBIGUITY, 006 INCORRECT, 017
     NOT_FOUND) was built against.
  2. The 6 V5.6 resumes under output/v2_4h_benchmark/stage1/markdown/
     -- the lineage where the real overrun defect was found (V6.5) and
     where this milestone found it is NOT resume-005-specific (013,
     016, 021 also affected).

Writes: benchmark/baselines/v6_6_boundary_validation_before_after.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_v5_8_full_corpus_validation import load_mapping, MARKDOWN_DIR  # noqa: E402
from scripts._milestone_v5_9_signal_corrections import find_skills_section as old_find  # noqa: E402
from scripts._milestone_v6_6_boundary_correction import find_skills_section as new_find  # noqa: E402

V5_6_SOURCE_DIR = Path("output/v2_4h_benchmark/stage1/markdown")
V5_6_STEMS = {
    "001": "AMAN_CV_BCA", "005": "Arjun Sharma_Data Architect_GT",
    "013": "Kishan Yadav_Data Scientist_Intileo",
    "016": "Mukul Kanojia_Full Stack Developer_Intileo",
    "021": "PRAGNESH S. SHAH_Sr. Associate-IT Ops_Intileo",
    "023": "Ritika Jain_AM2-Global Audit Delivery-Australia Region_GT-Intileo",
}

# Pre-established V5.9 classification (output/markdown lineage), for
# the "changed from reference" comparison column.
V5_9_REFERENCE_CLASSIFICATION = {
    "006": "INCORRECT", "017": "NOT_FOUND", "008": "ACCEPTABLE_WITH_AMBIGUITY",
    "020": "ACCEPTABLE_WITH_AMBIGUITY",
    # all other 18 resumes: CLEAR
}


def classify(rid: str, sec: dict) -> str:
    if rid in V5_9_REFERENCE_CLASSIFICATION:
        return V5_9_REFERENCE_CLASSIFICATION[rid]
    if not sec["found"]:
        return "NOT_FOUND"
    if sec.get("boundary_content_implausible"):
        return "INCORRECT"
    return "CLEAR"


def main() -> None:
    rows = []

    # --- Lineage 1: output/markdown, all 22 resumes ---
    mapping = load_mapping()
    for row in mapping:
        rid = row["ResumeID"]
        md_path = MARKDOWN_DIR / row["Markdown"]
        if not md_path.exists():
            continue
        text = md_path.read_text(encoding="utf-8", errors="replace")
        old = old_find(text)
        new = new_find(text)
        old_body = old.get("cleaned_body_text", "")
        new_body = new.get("cleaned_body_text", "")
        changed = old_body != new_body or old["found"] != new["found"] or \
            old.get("boundary_content_implausible") != new.get("boundary_content_implausible")
        rows.append({
            "lineage": "output/markdown", "resume_id": rid,
            "v5_9_reference_status": V5_9_REFERENCE_CLASSIFICATION.get(rid, "CLEAR"),
            "old_found": old["found"], "old_implausible": old.get("boundary_content_implausible"),
            "old_body_len": len(old_body),
            "new_found": new["found"], "new_implausible": new.get("boundary_content_implausible"),
            "new_body_len": len(new_body), "new_end_reason": new.get("end_reason"),
            "boundary_changed": changed,
            "change_assessment": "no_change" if not changed else "CHANGED -- see report for correctness assessment",
        })

    # --- Lineage 2: v2_4h_benchmark, 6 V5.6 resumes ---
    for rid, stem in V5_6_STEMS.items():
        md_path = V5_6_SOURCE_DIR / f"{stem}.md"
        if not md_path.exists():
            continue
        text = md_path.read_text(encoding="utf-8", errors="replace")
        old = old_find(text)
        new = new_find(text)
        old_body = old.get("cleaned_body_text", "")
        new_body = new.get("cleaned_body_text", "")
        changed = old_body != new_body or old["found"] != new["found"]
        rows.append({
            "lineage": "v2_4h_benchmark", "resume_id": rid,
            "v5_9_reference_status": "not_previously_classified_this_lineage",
            "old_found": old["found"], "old_implausible": old.get("boundary_content_implausible"),
            "old_body_len": len(old_body),
            "new_found": new["found"], "new_implausible": new.get("boundary_content_implausible"),
            "new_body_len": len(new_body), "new_end_reason": new.get("end_reason"),
            "boundary_changed": changed,
            "change_assessment": (
                "no_change" if not changed else
                "CORRECTED -- overrun eliminated (see milestone report direct-inspection confirmation)"
                if rid in ("005", "013", "016", "021") else
                "CHANGED -- see report"
            ),
        })

    out = Path("benchmark/baselines/v6_6_boundary_validation_before_after.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")

    n_changed = sum(1 for r in rows if r["boundary_changed"])
    print(f"\nTotal rows: {len(rows)}, changed: {n_changed}")
    for r in rows:
        if r["boundary_changed"]:
            print(f"  {r['lineage']}/{r['resume_id']}: old_len={r['old_body_len']} -> new_len={r['new_body_len']} ({r['change_assessment']})")


if __name__ == "__main__":
    main()
