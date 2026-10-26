"""
scripts/_milestone_t6_phase1c_corpus_regression.py

Milestone T6/Phase 1C -- Phase E corpus regression. Applies the NEW
`_ground_project_urls` fix to T4/Phase 1A's own REAL, already-captured
Project extraction checkpoints (8 resumes: 001, 005, 006, 013, 016,
017, 018, 021) against their REAL source markdown, and compares every
resulting url value to the PRE-fix (T4's own original) value.

ZERO new LLM calls -- reuses T4's own 16 real checkpoints (8 of which
are the "projects" module results this script needs).

Writes: benchmark/baselines/t6_phase1c_corpus_regression.csv
"""
from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from services.project_extractor import _ground_project_urls  # noqa: E402 -- the NEW fix, unit-under-test

T4_CKPT_DIR = Path(
    "/private/tmp/claude-501/-Users-ruchigupta-Documents-ResumeAIBenchmark/"
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/t4_phase1a_checkpoints"
)
MARKDOWN_DIR = Path("output/markdown")


def _load_population() -> list[dict]:
    with open("benchmark/baselines/t4_phase1a_frozen_validation_population.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    population = _load_population()
    rows = []

    for row in population:
        rid, stem = row["resume_id"], row["stem"]
        ckpt_path = T4_CKPT_DIR / f"{rid}__projects.pkl"
        with ckpt_path.open("rb") as f:
            ckpt = pickle.load(f)
        assert ckpt["result"]["status"] == "ok", f"{rid}: checkpoint not ok"
        pre_fix_json = ckpt["result"]["json_object"]
        pre_fix_projects = pre_fix_json.get("projects", [])

        markdown_text = (MARKDOWN_DIR / f"{stem}.md").read_text(encoding="utf-8")
        post_fix_json = _ground_project_urls(pre_fix_json, markdown_text)
        post_fix_projects = post_fix_json.get("projects", [])

        assert len(pre_fix_projects) == len(post_fix_projects), f"{rid}: entity count changed -- must never happen"

        for i, (pre, post) in enumerate(zip(pre_fix_projects, post_fix_projects)):
            pre_url, post_url = pre.get("url"), post.get("url")
            changed = pre_url != post_url
            # Every other field must be byte-identical -- this fix must
            # only ever touch `url`.
            other_fields_changed = [
                k for k in pre if k != "url" and pre.get(k) != post.get(k)
            ]
            rows.append({
                "resume_id": rid, "project_index": i, "project_name": pre.get("name"),
                "pre_fix_url": pre_url, "post_fix_url": post_url, "url_changed": changed,
                "other_fields_changed": "; ".join(other_fields_changed) if other_fields_changed else "none",
                "classification": (
                    "no_change" if not changed else
                    "known_url_association_defect_corrected" if rid == "016" and pre.get("name") == "Bewakoof.com Clone" else
                    "UNEXPECTED_CHANGE_REQUIRES_INVESTIGATION"
                ),
            })

    out = Path("benchmark/baselines/t6_phase1c_corpus_regression.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")

    changed_rows = [r for r in rows if r["url_changed"]]
    unexpected = [r for r in rows if r["classification"] == "UNEXPECTED_CHANGE_REQUIRES_INVESTIGATION"]
    other_field_changes = [r for r in rows if r["other_fields_changed"] != "none"]
    print(f"\nTotal project entities checked: {len(rows)}")
    print(f"URL changed: {len(changed_rows)}")
    for r in changed_rows:
        print(f"  {r['resume_id']} [{r['project_index']}] {r['project_name']!r}: "
              f"{r['pre_fix_url']!r} -> {r['post_fix_url']!r} ({r['classification']})")
    print(f"\nUnexpected changes (require investigation): {len(unexpected)}")
    print(f"Other-field changes (must be zero -- fix must only touch url): {len(other_field_changes)}")


if __name__ == "__main__":
    main()
