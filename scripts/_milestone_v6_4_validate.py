"""
scripts/_milestone_v6_4_validate.py

Milestone V6.4 -- runs the fidelity proxy (_milestone_v6_4_fidelity_proxy.py)
against real historical data:

  1. All 50 real V6.1 pilot checkpoints (013, 007, 015, 016, 018 --
     baseline + candidate, 5 runs each), using the SAME Stage-1 source
     lineage V6.1 itself used (output/markdown/, the Milestone-23.67
     lineage) -- the correct, matching source for this data.
  2. The real, known V5.6 content-substitution case (005, baseline
     run3, recall=0.000, confirmed content-substitution since V3.4/
     V3.5), using V5.6's OWN correct Stage-1 source lineage
     (output/v2_4h_benchmark/stage1/markdown/) -- deliberately NOT
     output/markdown/, to avoid a source-lineage mismatch confound
     (the same mismatch V6.0/V6.1 already disclosed and avoided for
     the pilot itself).

ZERO LLM calls -- reads only already-existing checkpoint pickles and
already-existing Stage-1 markdown.

Writes: benchmark/baselines/v6_4_validation_results.csv
"""
from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_v5_9_signal_corrections import find_skills_section  # noqa: E402
from scripts._milestone_v6_4_fidelity_proxy import compute_fidelity_proxy  # noqa: E402

V6_1_CKPT = Path(
    "/private/tmp/claude-501/-Users-ruchigupta-Documents-ResumeAIBenchmark/"
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/v6_1_checkpoints"
)
V5_6_CKPT = Path(
    "/private/tmp/claude-501/-Users-ruchigupta-Documents-ResumeAIBenchmark/"
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/v5_6_checkpoints"
)
V6_1_SOURCE_DIR = Path("output/markdown")
V5_6_SOURCE_DIR = Path("output/v2_4h_benchmark/stage1/markdown")

V6_1_STEMS = {
    "007": "Bhavita Jain_GAD_GT", "013": "Kishan Yadav_Data Scientist_Intileo",
    "015": "Monika Yathamshetti_AM-GAD_GT", "016": "Mukul Kanojia_Full Stack Developer_Intileo",
    "018": "Naukri_ShubhangiBagla[4y_0m]",
}

# Real, direct-inspection mechanism labels this project already
# established (V5.6/V6.1/V6.2) -- reused for reporting only, NEVER used
# as an input to the proxy computation itself.
KNOWN_MECHANISM = {
    "007": "minor_small_list_gap", "013": "long_list_tail_omission",
    "015": "segmentation_convention_no_benefit", "016": "stable_control",
    "018": "multi_subheading_category_omission",
}


def _section_body_and_validity(stem: str, source_dir: Path) -> tuple[str, bool]:
    md_path = source_dir / f"{stem}.md"
    md = md_path.read_text(encoding="utf-8")
    sec = find_skills_section(md)
    boundary_valid = bool(sec["found"]) and not sec.get("boundary_content_implausible", False)
    return sec.get("cleaned_body_text", ""), boundary_valid


def main() -> None:
    rows = []

    # --- 1. Real V6.1 pilot data: 50 real checkpoints ---
    for rid, stem in V6_1_STEMS.items():
        body_text, boundary_valid = _section_body_and_validity(stem, V6_1_SOURCE_DIR)
        for condition in ("baseline", "candidate"):
            for run_idx in range(1, 6):
                p = V6_1_CKPT / f"{rid}__{condition}__run{run_idx}.pkl"
                if not p.exists():
                    continue
                with p.open("rb") as f:
                    slot = pickle.load(f)
                if slot["status"] != "ok":
                    rows.append({
                        "source": "v6_1_pilot", "resume_id": rid, "condition": condition,
                        "run_index": run_idx, "known_mechanism": KNOWN_MECHANISM[rid],
                        "boundary_valid": boundary_valid, "proxy_status": "call_failed",
                        "n_items": "", "n_grounded": "", "n_ungrounded": "",
                        "grounding_fraction": "", "ungrounded_items": "",
                    })
                    continue
                items = slot["final"]["skills"]
                result = compute_fidelity_proxy(items, body_text, boundary_valid)
                rows.append({
                    "source": "v6_1_pilot", "resume_id": rid, "condition": condition,
                    "run_index": run_idx, "known_mechanism": KNOWN_MECHANISM[rid],
                    "boundary_valid": boundary_valid, "proxy_status": result["proxy_status"],
                    "n_items": result["n_items"], "n_grounded": result["n_grounded"],
                    "n_ungrounded": result["n_ungrounded"],
                    "grounding_fraction": result["grounding_fraction"],
                    "ungrounded_items": "; ".join(result["ungrounded_items"]),
                })

    # --- 2. Real, known V5.6 content-substitution case (005, baseline run3) ---
    body_text, boundary_valid = _section_body_and_validity("Arjun Sharma_Data Architect_GT", V5_6_SOURCE_DIR)
    with (V5_6_CKPT / "005__baseline__run3.pkl").open("rb") as f:
        slot = pickle.load(f)
    result = compute_fidelity_proxy(slot["final"]["skills"], body_text, boundary_valid)
    rows.append({
        "source": "v5_6_known_content_substitution", "resume_id": "005", "condition": "baseline",
        "run_index": 3, "known_mechanism": "content_substitution_confirmed_v3_4_v3_5_v5_6",
        "boundary_valid": boundary_valid, "proxy_status": result["proxy_status"],
        "n_items": result["n_items"], "n_grounded": result["n_grounded"],
        "n_ungrounded": result["n_ungrounded"], "grounding_fraction": result["grounding_fraction"],
        "ungrounded_items": "; ".join(result["ungrounded_items"]),
    })

    out = Path("benchmark/baselines/v6_4_validation_results.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")

    # --- summary ---
    import statistics
    print("\n=== Mean grounding_fraction by resume x condition (real V6.1 data) ===")
    for rid in V6_1_STEMS:
        for condition in ("baseline", "candidate"):
            vals = [r["grounding_fraction"] for r in rows
                    if r["resume_id"] == rid and r["condition"] == condition and r["grounding_fraction"] != ""]
            if vals:
                print(f"{rid} ({KNOWN_MECHANISM[rid]}) {condition}: mean={statistics.mean(vals):.3f} "
                      f"range=[{min(vals):.3f}-{max(vals):.3f}] n={len(vals)}")
    subst = [r for r in rows if r["source"] == "v5_6_known_content_substitution"][0]
    print(f"\n005 known content-substitution case: grounding_fraction={subst['grounding_fraction']} "
          f"({subst['n_grounded']}/{subst['n_items']} grounded)")


if __name__ == "__main__":
    main()
