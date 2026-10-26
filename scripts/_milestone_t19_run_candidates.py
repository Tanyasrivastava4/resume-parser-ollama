"""
scripts/_milestone_t19_run_candidates.py

Milestone T19/Phase 2F -- Phase B: run the CURRENT, UNCHANGED T18
deterministic candidate-generation implementation against the 6 (5
usable PDF, 1 excluded .docx set aside) genuinely unseen resumes.

Imports blockify() and generate_candidates_v2() directly from T18's
own modules -- not re-typed, not modified, not tuned. Same convention
T18 itself used when importing T17's own functions.

No LLM calls in this script. No rule changes. No retries. No manual
intervention beyond running this script once and reading its output.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_t18_blocks import blockify  # noqa: E402 (unchanged, re-exported from T17)
from scripts._milestone_t18_candidates import generate_candidates_v2  # noqa: E402

MARKDOWN_DIR = Path("output/markdown")

UNSEEN_RESUMES = [
    ("U1", "178480991686_Vachan_Premugh_Dicruz_Founder_Office_BTTR.MEN"),
    ("U2", "8th_April_Akasa_Kapoor_Resume"),
    ("U3", "AARTI BAGHOTIYA_Economic Urban Planner_Intileo"),
    ("U4", "AASHISH PARMAR - Email Marketing Executive - Intileo"),
    ("U5", "_Adarsh_Bhagat Resume"),
]


def run_one(uid: str, stem: str):
    markdown_text = (MARKDOWN_DIR / f"{stem}.md").read_text(encoding="utf-8")
    blocks = blockify(markdown_text)
    candidates = generate_candidates_v2(blocks)
    return blocks, candidates


def main() -> None:
    rows = []
    print(f"{'ID':4} {'stem':55} {'blocks':7} {'candidates':11}")
    for uid, stem in UNSEEN_RESUMES:
        blocks, candidates = run_one(uid, stem)
        print(f"{uid:4} {stem[:55]:55} {len(blocks):7} {len(candidates):11}")
        for c in candidates:
            extent_blocks = blocks[c.extent_start : c.extent_end + 1]
            flags_on = [k for k, v in c.flags.items() if v]
            rows.append({
                "unseen_id": uid,
                "stem": stem,
                "candidate_id": c.candidate_id,
                "anchor_text": c.anchor_text,
                "extent_start": c.extent_start,
                "extent_end": c.extent_end,
                "n_blocks_in_extent": c.n_blocks_in_extent,
                "containing_section_label": c.containing_section_label,
                "contains_bullet_run": c.contains_bullet_run,
                "flags_on": ";".join(flags_on),
                "extent_line_numbers": f"{extent_blocks[0].line_number}-{extent_blocks[-1].line_number}",
                "extent_preview": " | ".join(b.stripped[:40] for b in extent_blocks[:4]),
            })

    out = Path("reports/t19_unseen_candidate_raw_output.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} candidate rows across {len(UNSEEN_RESUMES)} resumes)")


if __name__ == "__main__":
    main()
