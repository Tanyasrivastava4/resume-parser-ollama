"""
scripts/_milestone_v5_8_offline_evaluation.py

Milestone V5.8 Step 5/6 support -- OFFLINE-ONLY reference evaluation of
the Milestone 23.67 frozen baseline's Skills extraction against real
adjudicated GT, for all 22 resumes. This is used SOLELY to assess, after
the fact, whether the V5.7 routing signals (fill_ratio, in_section_frac)
correlate with genuine incompleteness vs. other mechanisms -- GT is
NEVER fed into fill_ratio or in_section_frac themselves (see
_milestone_v5_8_full_corpus_validation.py, which computes those two
signals with zero GT access).

Reuses the exact length-guarded fuzzy-match definition V5.7 itself
found and fixed (naive substring containment falsely matched short
strings like a lone extracted token 'R' against 'IAC-Terrafrom'; fixed
by requiring the shorter normalized string to be >=4 characters) --
not reinvented here.

ZERO new LLM calls. Reads only the already-existing Milestone 23.67
checkpoints and the already-existing adjudicated GT files.

Output: benchmark/baselines/v5_8_offline_mechanism_reference.csv
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_v5_8_full_corpus_validation import (  # noqa: E402
    GT_MAPPING,
    load_baseline_skills,
    load_mapping,
)

GT_DIR = Path("benchmark/ground_truth/adjudicated_v1")
OUT = Path("benchmark/baselines/v5_8_offline_mechanism_reference.csv")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def fuzzy_eq(a: str, b: str) -> bool:
    a, b = norm(a), norm(b)
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) < 4:
        return False
    return shorter in longer


def evaluate(extracted: list, gt: list) -> dict:
    if not gt:
        return {
            "strict_recall": "", "fuzzy_recall": "", "content_overlap": "",
            "n_fuzzy_missing": "", "tail_concentration": "",
            "offline_label": "gt_empty_not_applicable",
        }
    gt_norm = [norm(g) for g in gt]
    ext = [e for e in extracted if isinstance(e, str)]
    ext_norm_set = set(norm(e) for e in ext)

    strict_matched = sum(1 for g in gt_norm if g in ext_norm_set)
    strict_recall = strict_matched / len(gt)

    fuzzy_missing_idx = []
    fuzzy_matched = 0
    for i, g in enumerate(gt):
        hit = any(fuzzy_eq(g, e) for e in ext)
        if hit:
            fuzzy_matched += 1
        else:
            fuzzy_missing_idx.append(i)
    fuzzy_recall = fuzzy_matched / len(gt)

    if ext:
        ext_with_match = sum(1 for e in ext if any(fuzzy_eq(e, g) for g in gt))
        content_overlap = ext_with_match / len(ext)
    else:
        content_overlap = None

    n = len(gt)
    tail_start = int(n * 0.75)
    tail_count = sum(1 for i in fuzzy_missing_idx if i >= tail_start)
    tail_concentration = (tail_count / len(fuzzy_missing_idx)) if fuzzy_missing_idx else None

    # Offline mechanism label -- descriptive only, reusing V5.6/V5.7's
    # own established mechanism categories. NOT used as a routing
    # input anywhere; used only to check, after the fact, what kind of
    # resume the provisional signal did or did not flag.
    if fuzzy_recall >= 0.90:
        label = "stable_no_material_failure"
    elif content_overlap is not None and content_overlap < 0.5:
        label = "likely_content_substitution"
    elif tail_concentration is not None and tail_concentration >= 0.60:
        label = "likely_tail_omission"
    else:
        label = "material_incompleteness_other_pattern"

    return {
        "strict_recall": round(strict_recall, 4),
        "fuzzy_recall": round(fuzzy_recall, 4),
        "content_overlap": round(content_overlap, 4) if content_overlap is not None else "",
        "n_fuzzy_missing": len(fuzzy_missing_idx),
        "tail_concentration": round(tail_concentration, 4) if tail_concentration is not None else "",
        "offline_label": label,
    }


def main() -> None:
    rows = []
    for row in load_mapping():
        resume_id = row["ResumeID"]
        md_name = row["Markdown"]
        gt_json_name = row["JSON"]
        gt_path = GT_DIR / gt_json_name

        baseline = load_baseline_skills(md_name)
        if baseline["status"] != "ok":
            rows.append({"resume_id": resume_id, "baseline_status": baseline["status"],
                         "strict_recall": "", "fuzzy_recall": "", "content_overlap": "",
                         "n_fuzzy_missing": "", "tail_concentration": "",
                         "offline_label": "baseline_not_ok"})
            continue
        if not gt_path.exists():
            rows.append({"resume_id": resume_id, "baseline_status": baseline["status"],
                         "strict_recall": "", "fuzzy_recall": "", "content_overlap": "",
                         "n_fuzzy_missing": "", "tail_concentration": "",
                         "offline_label": "gt_file_missing"})
            continue

        gt_obj = json.loads(gt_path.read_text(encoding="utf-8"))
        gt_skills = gt_obj.get("skills", [])
        ev = evaluate(baseline["actual"] or [], gt_skills)
        rows.append({"resume_id": resume_id, "baseline_status": baseline["status"], **ev})

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")
    from collections import Counter
    print(Counter(r["offline_label"] for r in rows))


if __name__ == "__main__":
    main()
