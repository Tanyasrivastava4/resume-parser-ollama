"""
scripts/_milestone_t20_raw_output.py

Milestone T20/Phase 2G -- dumps full per-candidate detail for both
conditions (baseline T18, candidate T20) across the full T20
population, for reports/t20_section_heading_raw_output.csv.

No LLM calls.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_t18_blocks import blockify  # noqa: E402
from scripts._milestone_t18_candidates import generate_candidates_v2 as generate_baseline  # noqa: E402
from scripts._milestone_t20_candidates import generate_candidates_t20 as generate_candidate  # noqa: E402

MARKDOWN_DIR = Path("output/markdown")

POPULATION = [
    ("unseen", "U1", "178480991686_Vachan_Premugh_Dicruz_Founder_Office_BTTR.MEN"),
    ("unseen", "U2", "8th_April_Akasa_Kapoor_Resume"),
    ("unseen", "U3", "AARTI BAGHOTIYA_Economic Urban Planner_Intileo"),
    ("unseen", "U4", "AASHISH PARMAR - Email Marketing Executive - Intileo"),
    ("unseen", "U5", "_Adarsh_Bhagat Resume"),
    ("control", "001", "AMAN_CV_BCA"),
    ("control", "006", "AshishKumarSingh_RBM_Pune_Stonelam"),
    ("control", "013", "Kishan Yadav_Data Scientist_Intileo"),
    ("control", "021", "PRAGNESH S. SHAH_Sr. Associate-IT Ops_Intileo"),
]


def main() -> None:
    rows = []
    for population, rid, stem in POPULATION:
        blocks = blockify((MARKDOWN_DIR / f"{stem}.md").read_text(encoding="utf-8"))
        for condition, gen in (("baseline", generate_baseline), ("candidate", generate_candidate)):
            candidates = gen(blocks)
            for c in candidates:
                flags_on = [k for k, v in c.flags.items() if v]
                rows.append({
                    "population": population, "resume_id": rid, "stem": stem,
                    "condition": condition, "candidate_id": c.candidate_id,
                    "anchor_text": c.anchor_text, "extent_start": c.extent_start,
                    "extent_end": c.extent_end, "n_blocks_in_extent": c.n_blocks_in_extent,
                    "containing_section_label": c.containing_section_label,
                    "flags_on": ";".join(flags_on),
                })

    out = Path("reports/t20_section_heading_raw_output.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
