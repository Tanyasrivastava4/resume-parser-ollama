"""
scripts/_milestone_v6_5_full_corpus_run.py

Milestone V6.5 -- runs the FROZEN, UNCHANGED V6.4 fidelity proxy
(scripts/_milestone_v6_4_fidelity_proxy.py -- imported, not
reimplemented, not modified) against every existing, already-committed
Skills output for the full 22-resume benchmark corpus:

  1. The single Milestone-23.67 frozen qwen3:1.7b baseline checkpoint
     for every resume where a Skills boundary can be safely located
     (output/markdown/ lineage) -- NEW coverage for 14 resumes never
     proxy-validated before (002,003,004,008,009,010,011,014,019,020,
     022, plus 006/017 correctly excluded as ineligible).
  2. The 60 real V5.6 baseline+candidate checkpoints (001,005,013,016,
     021,023 -- 5 runs each condition), scored against V5.6's OWN
     correct source lineage (output/v2_4h_benchmark/stage1/markdown/)
     -- NEW coverage for 001, 005, 021, 023.
  3. The 50 real V6.1 baseline+candidate checkpoints (007,013,015,016,
     018 -- 5 runs each condition), scored against output/markdown/
     (the same lineage V6.1 itself used) -- reproduces V6.4's own
     already-reported findings for these 5 resumes, using the same
     frozen proxy, for a single consistent full-corpus artifact.

ZERO new LLM calls anywhere -- reads only already-existing checkpoint
pickles and already-existing Stage-1 markdown.

Writes:
  benchmark/baselines/v6_5_population_inventory.csv
  benchmark/baselines/v6_5_full_corpus_item_results.csv
"""
from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_v5_8_full_corpus_validation import (  # noqa: E402
    MARKDOWN_DIR,
    load_baseline_skills,
    load_mapping,
)
from scripts._milestone_v5_9_signal_corrections import find_skills_section  # noqa: E402
from scripts._milestone_v6_4_fidelity_proxy import compute_fidelity_proxy  # noqa: E402 -- FROZEN, unmodified

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


def build_population_inventory() -> list[dict]:
    rows = []
    for row in load_mapping():
        rid = row["ResumeID"]
        stem = Path(row["Markdown"]).stem
        md_path = MARKDOWN_DIR / row["Markdown"]
        md_present = md_path.exists()
        baseline = load_baseline_skills(row["Markdown"]) if md_present else {"status": "no_markdown", "actual": None}
        status = baseline["status"]
        n_extracted = len(baseline["actual"]) if baseline["actual"] else 0

        if not md_present:
            rows.append({"resume_id": rid, "stem": stem, "source_lineage": "output/markdown",
                         "md_present": False, "baseline_status": status, "n_extracted": 0,
                         "boundary_found": "", "boundary_implausible": "", "eligible": False,
                         "exclusion_reason": "markdown source file missing"})
            continue

        text = md_path.read_text(encoding="utf-8", errors="replace")
        sec = find_skills_section(text)
        found = sec["found"]
        implausible = sec.get("boundary_content_implausible", False)

        if status != "ok":
            eligible, reason = False, f"baseline checkpoint status={status}"
        elif not found:
            eligible, reason = False, "no Skills-section boundary found (V5.9 corrected detector)"
        elif implausible:
            eligible, reason = False, "boundary found but structurally implausible (V5.9 table-like check)"
        elif n_extracted == 0:
            eligible, reason = False, "baseline extracted zero Skills items"
        else:
            eligible, reason = True, ""

        extra_evidence = []
        if rid in V5_6_STEMS:
            extra_evidence.append("V5.6 (5 baseline + 5 candidate runs, output/v2_4h_benchmark lineage)")
        if rid in V6_1_STEMS:
            extra_evidence.append("V6.1 (5 baseline + 5 candidate runs, output/markdown lineage)")

        rows.append({
            "resume_id": rid, "stem": stem, "source_lineage": "output/markdown",
            "md_present": True, "baseline_status": status, "n_extracted": n_extracted,
            "boundary_found": found, "boundary_implausible": implausible,
            "eligible": eligible, "exclusion_reason": reason,
            "additional_richer_evidence": "; ".join(extra_evidence) if extra_evidence else "none",
        })
    return rows


def main() -> None:
    pop_rows = build_population_inventory()
    with open("benchmark/baselines/v6_5_population_inventory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pop_rows[0].keys()))
        w.writeheader()
        w.writerows(pop_rows)
    n_eligible = sum(1 for r in pop_rows if r["eligible"])
    print(f"Population inventory: {len(pop_rows)} total, {n_eligible} eligible, {len(pop_rows) - n_eligible} excluded")
    for r in pop_rows:
        if not r["eligible"]:
            print(f"  EXCLUDED {r['resume_id']}: {r['exclusion_reason']}")

    item_rows = []

    # --- 1. Milestone-23.67 single-run baseline, all eligible resumes ---
    mapping = {r["ResumeID"]: r for r in load_mapping()}
    for prow in pop_rows:
        if not prow["eligible"]:
            continue
        rid = prow["resume_id"]
        stem = prow["stem"]
        body_text, boundary_valid = _body_and_validity(stem, MARKDOWN_DIR)
        baseline = load_baseline_skills(mapping[rid]["Markdown"])
        result = compute_fidelity_proxy(baseline["actual"], body_text, boundary_valid)
        for e in result["per_item_evidence"]:
            item_rows.append({
                "data_source": "milestone_23_67_single_baseline", "resume_id": rid, "condition": "baseline",
                "run_index": 1, "model": "qwen3:1.7b", "item": e["item"], "grounded": e["grounded"],
                "reason": e["reason"],
            })

    # --- 2. V5.6 real 60-slot data (own correct source lineage) ---
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
                items = slot["final"]["skills"]
                result = compute_fidelity_proxy(items, body_text, boundary_valid)
                for e in result["per_item_evidence"]:
                    item_rows.append({
                        "data_source": "v5_6_checkpoint", "resume_id": rid, "condition": condition,
                        "run_index": run_idx, "model": model, "item": e["item"], "grounded": e["grounded"],
                        "reason": e["reason"],
                    })

    # --- 3. V6.1 real 50-slot data (reproduces V6.4's own findings, same frozen proxy) ---
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
                items = slot["final"]["skills"]
                result = compute_fidelity_proxy(items, body_text, boundary_valid)
                for e in result["per_item_evidence"]:
                    item_rows.append({
                        "data_source": "v6_1_checkpoint", "resume_id": rid, "condition": condition,
                        "run_index": run_idx, "model": model, "item": e["item"], "grounded": e["grounded"],
                        "reason": e["reason"],
                    })

    with open("benchmark/baselines/v6_5_full_corpus_item_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(item_rows[0].keys()))
        w.writeheader()
        w.writerows(item_rows)

    n_total = len(item_rows)
    n_grounded = sum(1 for r in item_rows if r["grounded"])
    n_ungrounded = n_total - n_grounded
    print(f"\nTotal items evaluated: {n_total}")
    print(f"Grounded: {n_grounded} ({100*n_grounded/n_total:.1f}%)")
    print(f"Ungrounded (flagged): {n_ungrounded} ({100*n_ungrounded/n_total:.1f}%)")

    from collections import Counter
    print("\nBy data source:")
    for src in ("milestone_23_67_single_baseline", "v5_6_checkpoint", "v6_1_checkpoint"):
        rows = [r for r in item_rows if r["data_source"] == src]
        g = sum(1 for r in rows if r["grounded"])
        print(f"  {src}: n={len(rows)} grounded={g} ({100*g/len(rows):.1f}%)" if rows else f"  {src}: n=0")

    print("\nFlagged (ungrounded) items by resume:")
    flagged = [r for r in item_rows if not r["grounded"]]
    by_resume = Counter(r["resume_id"] for r in flagged)
    for rid, count in sorted(by_resume.items()):
        print(f"  {rid}: {count} flagged")
    print(f"\nTotal unique flagged (item, resume, condition) surface for manual classification: {len(flagged)}")


if __name__ == "__main__":
    main()
