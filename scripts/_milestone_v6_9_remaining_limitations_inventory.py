"""
scripts/_milestone_v6_9_remaining_limitations_inventory.py

Milestone V6.9 Phase A -- builds the exact remaining-limitations
population: every item still flagged (ungrounded) after applying the
V6.8-corrected proxy (which already fixed Mechanisms A and B) to the
same full-corpus population V6.5/V6.8 used. Cross-references each
remaining item against its ORIGINAL V6.5 classification (never
reclassifying historical evidence -- only adding a preliminary V6.9
mechanism hypothesis column). ZERO new LLM calls -- reads only the
already-committed benchmark/baselines/v6_8_full_corpus_item_results_before_after.csv
(itself built from real checkpoints/markdown, zero new calls) and the
already-committed benchmark/baselines/v6_5_flag_classification.csv.

Writes: benchmark/baselines/v6_9_remaining_limitations_inventory.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

# Preliminary mechanism hypothesis per ORIGINAL V6.5 category -- a
# hypothesis only, refined by direct source inspection in Phase B, not
# a reclassification of the original category.
_PRELIM_HYPOTHESIS = {
    "TRUE_UNSUPPORTED": "genuine unsupported content (safety baseline, not a correction target)",
    "FALSE_WARNING_PARAPHRASE_OR_TYPO": "single-character-edit-distance mismatch (direction varies: source-side typo corrected by model, or model-introduced typo of real content)",
    "FALSE_WARNING_LABEL_OR_SEGMENTATION": "candidate item is a merged, comma-joined multi-technology string where each individual constituent is separately verifiable against source",
    "BOUNDARY_OR_SOURCE_AMBIGUITY": "item found verbatim in an adjacent, same-level section outside the (correctly) detected Skills boundary -- not a proxy defect",
    "UNDETERMINED": "insufficient evidence in original V6.5 classification to determine genuine-vs-fabricated; no new evidence found in this milestone",
}


def main() -> None:
    with open("benchmark/baselines/v6_8_full_corpus_item_results_before_after.csv", encoding="utf-8") as f:
        v6_8_rows = list(csv.DictReader(f))
    with open("benchmark/baselines/v6_5_flag_classification.csv", encoding="utf-8") as f:
        v6_5_rows = list(csv.DictReader(f))
    v6_5_lookup = {
        (r["resume_id"], r["data_source"], r["condition"] + r["run_index"], r["item"]): r
        for r in v6_5_rows
    }

    still_flagged = [r for r in v6_8_rows if r["after_grounded"] == "False"]

    out_rows = []
    unmatched = []
    for r in still_flagged:
        key = (r["resume_id"], r["data_source"], r["condition"] + r["run_index"], r["item"])
        v6_5_row = v6_5_lookup.get(key)
        if v6_5_row is None:
            unmatched.append(key)
            original_category = "NOT_IN_V6_5_CLASSIFICATION -- requires manual investigation"
            evidence_note = ""
        else:
            original_category = v6_5_row["category"]
            evidence_note = v6_5_row["evidence_note"]

        out_rows.append({
            "resume_id": r["resume_id"],
            "data_source": r["data_source"],
            "condition": r["condition"],
            "run_index": r["run_index"],
            "candidate_item": r["item"],
            "original_v6_5_category": original_category,
            "original_v6_5_evidence_note": evidence_note,
            "preliminary_v6_9_mechanism_hypothesis": _PRELIM_HYPOTHESIS.get(original_category, "UNKNOWN -- new category, investigate"),
            "in_scope_for_v6_9_correction_attempt": original_category not in ("TRUE_UNSUPPORTED",),
        })

    out = Path("benchmark/baselines/v6_9_remaining_limitations_inventory.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print(f"wrote {out} ({len(out_rows)} rows)")
    print(f"Unmatched (not found in V6.5 classification -- would require manual investigation): {len(unmatched)}")
    for u in unmatched:
        print("  ", u)

    from collections import Counter
    print("\nBreakdown by original V6.5 category:")
    for cat, count in Counter(r["original_v6_5_category"] for r in out_rows).most_common():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
