"""
scripts/_milestone_v6_6_fidelity_proxy_impact.py

Milestone V6.6 Phase 5 -- re-runs the FROZEN, UNMODIFIED V6.4/V6.5
fidelity proxy against the same already-existing checkpoint data used
in V6.5, but with the V6.6-corrected Skills-section boundary instead
of the pre-fix V5.9 boundary. ZERO new LLM calls -- reads only
already-existing checkpoint pickles and already-existing Stage-1
markdown.

Only the 4 resumes whose boundary actually changed (Phase 4: 005, 013,
016, 021, all under the v2_4h_benchmark/V5.6 lineage) can possibly show
a different fidelity result -- every other resume's boundary is
provably identical (Phase 4's own before/after comparison), so this
script only recomputes those 4, for the full set of real V5.6
checkpoints available for them.

Writes: benchmark/baselines/v6_6_fidelity_proxy_impact.csv
"""
from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_v5_9_signal_corrections import find_skills_section as old_find  # noqa: E402
from scripts._milestone_v6_4_fidelity_proxy import compute_fidelity_proxy  # noqa: E402 -- FROZEN, unmodified
from scripts._milestone_v6_6_boundary_correction import find_skills_section as new_find  # noqa: E402

V5_6_CKPT = Path(
    "/private/tmp/claude-501/-Users-ruchigupta-Documents-ResumeAIBenchmark/"
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/v5_6_checkpoints"
)
V5_6_SOURCE_DIR = Path("output/v2_4h_benchmark/stage1/markdown")
CHANGED_RESUMES = {
    "005": "Arjun Sharma_Data Architect_GT",
    "013": "Kishan Yadav_Data Scientist_Intileo",
    "016": "Mukul Kanojia_Full Stack Developer_Intileo",
    "021": "PRAGNESH S. SHAH_Sr. Associate-IT Ops_Intileo",
}


def _body_and_validity(sec: dict) -> tuple[str, bool]:
    body = sec.get("cleaned_body_text", "")
    valid = bool(sec["found"]) and not sec.get("boundary_content_implausible", False)
    return body, valid


def main() -> None:
    rows = []
    for rid, stem in CHANGED_RESUMES.items():
        text = (V5_6_SOURCE_DIR / f"{stem}.md").read_text(encoding="utf-8", errors="replace")
        old_sec = old_find(text)
        new_sec = new_find(text)
        old_body, old_valid = _body_and_validity(old_sec)
        new_body, new_valid = _body_and_validity(new_sec)

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

                old_result = compute_fidelity_proxy(items, old_body, old_valid)
                new_result = compute_fidelity_proxy(items, new_body, new_valid)

                old_frac = old_result.get("grounding_fraction")
                new_frac = new_result.get("grounding_fraction")
                old_ungrounded = set(old_result.get("ungrounded_items") or [])
                new_ungrounded = set(new_result.get("ungrounded_items") or [])
                newly_unsupported = new_ungrounded - old_ungrounded
                newly_supported = old_ungrounded - new_ungrounded

                rows.append({
                    "resume_id": rid, "condition": condition, "model": model, "run_index": run_idx,
                    "old_proxy_status": old_result["proxy_status"], "new_proxy_status": new_result["proxy_status"],
                    "old_n_items": old_result.get("n_items"), "new_n_items": new_result.get("n_items"),
                    "old_grounding_fraction": old_frac, "new_grounding_fraction": new_frac,
                    "fraction_changed": old_frac != new_frac,
                    "items_newly_flagged_unsupported": "; ".join(sorted(newly_unsupported)),
                    "items_newly_recognized_supported": "; ".join(sorted(newly_supported)),
                })

    out = Path("benchmark/baselines/v6_6_fidelity_proxy_impact.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")

    n_changed = sum(1 for r in rows if r["fraction_changed"])
    print(f"\nRows where grounding_fraction changed: {n_changed} / {len(rows)}")
    for r in rows:
        if r["fraction_changed"]:
            print(f"  {r['resume_id']} {r['condition']} run{r['run_index']}: "
                  f"{r['old_grounding_fraction']} -> {r['new_grounding_fraction']} "
                  f"(newly_flagged={r['items_newly_flagged_unsupported']!r}, "
                  f"newly_supported={r['items_newly_recognized_supported']!r})")


if __name__ == "__main__":
    main()
