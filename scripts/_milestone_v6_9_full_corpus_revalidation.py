"""
scripts/_milestone_v6_9_full_corpus_revalidation.py

Milestone V6.9 -- full-corpus before/after re-evaluation of the ONE new
V6.9 secondary check (merged comma-joined multi-item string
segmentation, in scripts/_milestone_v6_9_fidelity_proxy_correction.py)
against the SAME population, source lineages, and V6.6-corrected
boundary detector used by V6.5/V6.8's own full-corpus work. ZERO new
LLM calls -- reads only already-existing checkpoint pickles and
already-existing Stage-1 markdown.

"Before" = the V6.8-corrected proxy (compute_fidelity_proxy_v6_8,
Mechanisms A+B already applied) so the comparison isolates exactly
what V6.9 adds on top of V6.8, not what V6.8 already contributed.
"After" = the V6.9-corrected proxy (compute_fidelity_proxy_v6_9).

Writes:
  benchmark/baselines/v6_9_full_corpus_item_results_before_after.csv
  benchmark/baselines/v6_9_true_unsupported_safety_check.csv
  benchmark/baselines/v6_9_mechanism_level_analysis.csv
"""
from __future__ import annotations

import csv
import pickle
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_v5_8_full_corpus_validation import (  # noqa: E402
    MARKDOWN_DIR,
    load_baseline_skills,
    load_mapping,
)
from scripts._milestone_v6_6_boundary_correction import find_skills_section  # noqa: E402 -- V6.6-corrected
from scripts._milestone_v6_8_fidelity_proxy_correction import compute_fidelity_proxy_v6_8  # noqa: E402
from scripts._milestone_v6_9_fidelity_proxy_correction import compute_fidelity_proxy_v6_9  # noqa: E402

V5_6_CKPT = Path(
    "/private/tmp/claude-501/-Users-ruchigupta-Documents-ResumeAIBenchmark/"
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/v5_6_checkpoints"
)
V6_1_CKPT = Path(
    "/private/tmp/claude-501/-Users-ruchigupta-Documents-ResumeAIBenchmark/"
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/v6_1_checkpoints"
)
V5_6_SOURCE_DIR = Path("output/v2_4h_benchmark/stage1/markdown")
V6_1_SOURCE_DIR = Path("output/markdown")

V5_6_STEMS = {
    "001": "AMAN_CV_BCA", "005": "Arjun Sharma_Data Architect_GT",
    "013": "Kishan Yadav_Data Scientist_Intileo", "016": "Mukul Kanojia_Full Stack Developer_Intileo",
    "021": "PRAGNESH S. SHAH_Sr. Associate-IT Ops_Intileo",
    "023": "Ritika Jain_AM2-Global Audit Delivery-Australia Region_GT-Intileo",
}
V6_1_STEMS = {
    "007": "Bhavita Jain_GAD_GT", "013": "Kishan Yadav_Data Scientist_Intileo",
    "015": "Monika Yathamshetti_AM-GAD_GT", "016": "Mukul Kanojia_Full Stack Developer_Intileo",
    "018": "Naukri_ShubhangiBagla[4y_0m]",
}


def _body_and_validity(stem: str, source_dir: Path) -> tuple[str, bool]:
    md_path = source_dir / f"{stem}.md"
    md = md_path.read_text(encoding="utf-8", errors="replace")
    sec = find_skills_section(md)
    valid = bool(sec["found"]) and not sec.get("boundary_content_implausible", False)
    return sec.get("cleaned_body_text", ""), valid


def _load_v6_5_eligible_resumes() -> set[str]:
    eligible = set()
    with open("benchmark/baselines/v6_5_population_inventory.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["eligible"] == "True":
                eligible.add(row["resume_id"])
    return eligible


def _load_true_unsupported() -> set[tuple[str, str, str, str]]:
    keys = set()
    with open("benchmark/baselines/v6_5_flag_classification.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["category"] == "TRUE_UNSUPPORTED":
                keys.add((row["resume_id"], row["data_source"], row["condition"] + row["run_index"], row["item"]))
    return keys


def main() -> None:
    eligible = _load_v6_5_eligible_resumes()
    true_unsupported = _load_true_unsupported()
    mapping = {r["ResumeID"]: r for r in load_mapping()}

    before_after_rows = []

    def _eval_and_record(data_source: str, rid: str, condition: str, run_idx: int, model: str,
                          items: list[str], body_text: str, boundary_valid: bool) -> None:
        before = compute_fidelity_proxy_v6_8(items, body_text, boundary_valid)
        after = compute_fidelity_proxy_v6_9(items, body_text, boundary_valid)
        before_ev = {e["item"]: e for e in (before.get("per_item_evidence") or [])}
        after_ev = {e["item"]: e for e in (after.get("per_item_evidence") or [])}
        for item in items:
            b = before_ev.get(item, {"grounded": None, "reason": None})
            a = after_ev.get(item, {"grounded": None, "reason": None})
            is_true_unsupported = (rid, data_source, condition + str(run_idx), item) in true_unsupported
            before_after_rows.append({
                "data_source": data_source, "resume_id": rid, "condition": condition, "run_index": run_idx,
                "model": model, "item": item,
                "before_v6_8_grounded": b["grounded"], "before_v6_8_reason": b["reason"],
                "after_v6_9_grounded": a["grounded"], "after_v6_9_reason": a["reason"],
                "changed": b["grounded"] != a["grounded"],
                "is_known_true_unsupported": is_true_unsupported,
                "SAFETY_VIOLATION": is_true_unsupported and (b["grounded"] is False) and (a["grounded"] is True),
            })

    for rid in sorted(eligible):
        stem = Path(mapping[rid]["Markdown"]).stem
        body_text, boundary_valid = _body_and_validity(stem, MARKDOWN_DIR)
        baseline = load_baseline_skills(mapping[rid]["Markdown"])
        _eval_and_record("milestone_23_67_single_baseline", rid, "baseline", 1, "qwen3:1.7b",
                          baseline["actual"], body_text, boundary_valid)

    for rid, stem in V5_6_STEMS.items():
        body_text, boundary_valid = _body_and_validity(stem, V5_6_SOURCE_DIR)
        for condition, model in (("baseline", "qwen3:1.7b"), ("candidate", "qwen3:8b")):
            for run_idx in range(1, 6):
                p = V5_6_CKPT / f"{rid}__{condition}__run{run_idx}.pkl"
                if not p.exists():
                    continue
                with p.open("rb") as f:
                    slot = pickle.load(f)
                if not slot.get("final"):
                    continue
                _eval_and_record("v5_6_checkpoint", rid, condition, run_idx, model,
                                  slot["final"]["skills"], body_text, boundary_valid)

    for rid, stem in V6_1_STEMS.items():
        body_text, boundary_valid = _body_and_validity(stem, V6_1_SOURCE_DIR)
        for condition, model in (("baseline", "qwen3:1.7b"), ("candidate", "qwen3:8b")):
            for run_idx in range(1, 6):
                p = V6_1_CKPT / f"{rid}__{condition}__run{run_idx}.pkl"
                if not p.exists():
                    continue
                with p.open("rb") as f:
                    slot = pickle.load(f)
                if slot["status"] != "ok":
                    continue
                _eval_and_record("v6_1_checkpoint", rid, condition, run_idx, model,
                                  slot["final"]["skills"], body_text, boundary_valid)

    out1 = Path("benchmark/baselines/v6_9_full_corpus_item_results_before_after.csv")
    with out1.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(before_after_rows[0].keys()))
        w.writeheader()
        w.writerows(before_after_rows)
    print(f"wrote {out1} ({len(before_after_rows)} rows)")

    safety_rows = [r for r in before_after_rows if r["is_known_true_unsupported"]]
    violations = [r for r in safety_rows if r["SAFETY_VIOLATION"]]
    out2 = Path("benchmark/baselines/v6_9_true_unsupported_safety_check.csv")
    with out2.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(safety_rows[0].keys()) if safety_rows else
                            list(before_after_rows[0].keys()))
        w.writeheader()
        w.writerows(safety_rows)
    print(f"wrote {out2} ({len(safety_rows)} rows, {len(violations)} SAFETY VIOLATIONS)")

    changed = [r for r in before_after_rows if r["changed"]]
    mech_counts = Counter(r["after_v6_9_reason"] for r in changed)
    out3 = Path("benchmark/baselines/v6_9_mechanism_level_analysis.csv")
    with out3.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(before_after_rows[0].keys()))
        w.writeheader()
        w.writerows(changed)
    print(f"wrote {out3} ({len(changed)} rows)")

    print("\n=== SUMMARY ===")
    print(f"Total items evaluated: {len(before_after_rows)}")
    print(f"Total changed (grounded flipped False->True): {len(changed)}")
    for reason, count in mech_counts.most_common():
        print(f"  {reason}: {count}")
    print(f"\nKnown TRUE_UNSUPPORTED items checked: {len(safety_rows)}")
    print(f"SAFETY VIOLATIONS (TRUE_UNSUPPORTED rescued): {len(violations)}")
    if violations:
        for v in violations:
            print(f"  !!! {v['resume_id']} {v['data_source']} {v['condition']}{v['run_index']} {v['item']!r} "
                  f"-> {v['after_v6_9_reason']}")


if __name__ == "__main__":
    main()
