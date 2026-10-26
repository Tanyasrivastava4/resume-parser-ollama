"""
scripts/_milestone_t2_phase0_failure_taxonomy.py

Milestone T2 / Phase 0 -- Phase A (Failure Taxonomy) and supporting
data for Phase C's resume-selection. ZERO LLM/model calls -- reads
only already-existing, already-committed baseline module output
(output/modules/<resume>/{work_experience,education,professional_summary}.json,
the Milestone-23.67 single-baseline checkpoint reused throughout the
V5.8-V7.0A research line) and real GT
(benchmark/ground_truth/adjudicated_v1/*.json).

Computes real, measured entity-level and field-level comparison
metrics for Work Experience and Education across all 22 corpus
resumes -- a systematic failure-taxonomy pass that has not been done
in this form before (prior milestones' WE/Education coverage was
narrower/manual). Skills is NOT re-derived here -- the existing
V3.0-V6.9 Skills research (46 real milestones' worth of evidence) is
reused BY REFERENCE, not reproduced, per this project's own established
discipline against redundant re-investigation of already-settled
questions.

RESEARCH-ONLY. Not imported by, and does not modify, services/ or
pipeline/.

Writes:
  benchmark/baselines/t2_phase0_we_entity_comparison.csv
  benchmark/baselines/t2_phase0_education_entity_comparison.csv
  benchmark/baselines/t2_phase0_summary_presence_check.csv
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from dateutil import parser as _dateutil_parser

MAPPING_PATH = Path("benchmark/ground_truth/adjudicated_v1/mapping.csv")
GT_DIR = Path("benchmark/ground_truth/adjudicated_v1")
MODULES_DIR = Path("output/modules")


_MONTH_ABBREV = {
    "jan": "january", "feb": "february", "mar": "march", "apr": "april",
    "jun": "june", "jul": "july", "aug": "august", "sep": "september",
    "sept": "september", "oct": "october", "nov": "november", "dec": "december",
}


def _normalize(text) -> str:
    """Word-level normalization PLUS month-abbreviation canonicalization
    (e.g. "Jan 2023" == "January 2023") -- added after this milestone's
    own spot-check found the raw word-normalizer was flagging semantically
    identical dates as mismatches purely due to abbreviation, exactly the
    "format variant vs. real error" distinction this project's Skills
    research (V6.4 fuzzy_eq) already established matters. Does NOT
    attempt general semantic date parsing (e.g. "01/2023" vs "Jan 2023")
    -- that gap is disclosed, not silently fixed."""
    if not text:
        return ""
    norm = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
    words = [_MONTH_ABBREV.get(w, w) for w in norm.split()]
    return " ".join(words)


def _dates_equal(a, b) -> bool:
    """Real date-object comparison (via dateutil), not string
    normalization -- so "2017-12-04" and "December 04 2017" (same real
    date, different formats) correctly compare equal. Added after this
    milestone's own spot-check found the word-normalizer conflating a
    genuine cross-entity field-confusion bug (resume 016: the same
    literal start_date value repeated across different jobs) with pure
    format-representation differences (resume 002: ISO vs. long-form
    English, semantically identical). Falls back to word-normalized
    string comparison if either value fails to parse as a date (e.g.
    partial dates like "2023" alone, or non-date text)."""
    if not a or not b:
        return False
    try:
        da = _dateutil_parser.parse(str(a), fuzzy=True, default=None)
        db = _dateutil_parser.parse(str(b), fuzzy=True, default=None)
        return da.date() == db.date()
    except Exception:
        return _normalize(a) == _normalize(b)


def _fuzzy_contains(a: str, b: str) -> bool:
    """Loose containment match on normalized text -- either string
    contains a materially-overlapping token run of the other. Used
    only for entity-alignment (finding "which GT job does this
    candidate job most likely correspond to"), not as a correctness
    metric in itself."""
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    # token-overlap fallback for reordered/abbreviated names
    ta, tb = set(na.split()), set(nb.split())
    if not ta or not tb:
        return False
    overlap = len(ta & tb) / min(len(ta), len(tb))
    return overlap >= 0.6


def load_mapping() -> list[dict]:
    with MAPPING_PATH.open() as f:
        return list(csv.DictReader(f))


def _align_entities(gt_list: list[dict], cand_list: list[dict], gt_key: str, cand_key: str):
    """Greedy best-match alignment by the primary identity field
    (company for WE, institution for Education). Returns
    (matched_pairs, unmatched_gt, unmatched_cand)."""
    matched = []
    used_cand = set()
    unmatched_gt = []
    for gt_item in gt_list:
        best_idx, best_score = None, 0.0
        for i, cand_item in enumerate(cand_list):
            if i in used_cand:
                continue
            if _fuzzy_contains(gt_item.get(gt_key, ""), cand_item.get(cand_key, "")):
                score = 1.0
                if best_idx is None or score > best_score:
                    best_idx, best_score = i, score
        if best_idx is not None:
            matched.append((gt_item, cand_list[best_idx]))
            used_cand.add(best_idx)
        else:
            unmatched_gt.append(gt_item)
    unmatched_cand = [c for i, c in enumerate(cand_list) if i not in used_cand]
    return matched, unmatched_gt, unmatched_cand


def analyze_work_experience(rid: str, gt: dict, module_dir: Path) -> dict | None:
    we_path = module_dir / "work_experience.json"
    if not we_path.exists():
        return None
    try:
        cand = json.loads(we_path.read_text())
    except Exception:
        return {"resume_id": rid, "status": "malformed_output"}
    cand_list = cand.get("work_experience") or []
    gt_list = gt.get("work_experience") or []

    matched, unmatched_gt, unmatched_cand = _align_entities(gt_list, cand_list, "company", "company")
    n_title_match = sum(1 for g, c in matched if _fuzzy_contains(g.get("designation", ""), c.get("designation", "")))
    # Denominator excludes matched pairs where GT itself has no value for
    # the field -- otherwise a resume where GT legitimately has no start
    # date would show a misleading 0% "match rate" for a field that was
    # never wrong, just absent from GT. Caught and fixed in this
    # milestone (spot-checked against resume 016, a known-good control).
    gt_has_start = [(g, c) for g, c in matched if g.get("start_date")]
    n_start_date_match = sum(1 for g, c in gt_has_start if _dates_equal(g.get("start_date"), c.get("start_date")))
    gt_has_end = [(g, c) for g, c in matched if g.get("end_date")]
    n_end_date_match = sum(1 for g, c in gt_has_end if _dates_equal(g.get("end_date"), c.get("end_date")))
    n_currently_working_match = sum(1 for g, c in matched if g.get("currently_working") == c.get("currently_working"))

    return {
        "resume_id": rid, "status": "ok",
        "gt_n_jobs": len(gt_list), "cand_n_jobs": len(cand_list),
        "n_matched_entities": len(matched),
        "n_unmatched_gt_entities": len(unmatched_gt),
        "n_unmatched_cand_entities": len(unmatched_cand),
        "entity_count_exact_match": len(gt_list) == len(cand_list),
        "entity_recall": len(matched) / len(gt_list) if gt_list else None,
        "entity_precision": len(matched) / len(cand_list) if cand_list else None,
        "title_match_rate_among_matched": n_title_match / len(matched) if matched else None,
        "start_date_match_rate_among_gt_with_start_date": n_start_date_match / len(gt_has_start) if gt_has_start else None,
        "n_gt_matched_with_start_date": len(gt_has_start),
        "end_date_match_rate_among_gt_with_end_date": n_end_date_match / len(gt_has_end) if gt_has_end else None,
        "n_gt_matched_with_end_date": len(gt_has_end),
        "currently_working_match_rate_among_matched": n_currently_working_match / len(matched) if matched else None,
        "unmatched_gt_companies": "; ".join(g.get("company") or "" for g in unmatched_gt),
        "unmatched_cand_companies": "; ".join(c.get("company") or "" for c in unmatched_cand),
    }


def analyze_education(rid: str, gt: dict, module_dir: Path) -> dict | None:
    edu_path = module_dir / "education.json"
    if not edu_path.exists():
        return None
    try:
        cand = json.loads(edu_path.read_text())
    except Exception:
        return {"resume_id": rid, "status": "malformed_output"}
    cand_list = cand.get("education") or []
    gt_list = gt.get("education") or []

    matched, unmatched_gt, unmatched_cand = _align_entities(gt_list, cand_list, "institution", "institution")
    n_degree_match = sum(1 for g, c in matched if _fuzzy_contains(g.get("degree", ""), c.get("degree", "")))
    gt_has_start = [(g, c) for g, c in matched if g.get("start_date")]
    n_start_date_match = sum(1 for g, c in gt_has_start if _dates_equal(g.get("start_date"), c.get("start_date")))
    n_currently_studying_match = sum(1 for g, c in matched if g.get("currently_studying") == c.get("currently_studying"))

    return {
        "resume_id": rid, "status": "ok",
        "gt_n_records": len(gt_list), "cand_n_records": len(cand_list),
        "n_matched_entities": len(matched),
        "n_unmatched_gt_entities": len(unmatched_gt),
        "n_unmatched_cand_entities": len(unmatched_cand),
        "entity_count_exact_match": len(gt_list) == len(cand_list),
        "entity_recall": len(matched) / len(gt_list) if gt_list else None,
        "entity_precision": len(matched) / len(cand_list) if cand_list else None,
        "degree_match_rate_among_matched": n_degree_match / len(matched) if matched else None,
        "start_date_match_rate_among_gt_with_start_date": n_start_date_match / len(gt_has_start) if gt_has_start else None,
        "n_gt_matched_with_start_date": len(gt_has_start),
        "currently_studying_match_rate_among_matched": n_currently_studying_match / len(matched) if matched else None,
        "unmatched_gt_institutions": "; ".join(g.get("institution") or "" for g in unmatched_gt),
        "unmatched_cand_institutions": "; ".join(c.get("institution") or "" for c in unmatched_cand),
    }


def analyze_summary(rid: str, gt: dict, module_dir: Path) -> dict | None:
    s_path = module_dir / "professional_summary.json"
    if not s_path.exists():
        return None
    try:
        cand = json.loads(s_path.read_text())
    except Exception:
        return {"resume_id": rid, "status": "malformed_output"}
    cand_text = cand.get("professional_summary") or ""
    gt_text = (gt.get("professional_summary") or {})
    if isinstance(gt_text, dict):
        gt_text = gt_text.get("professional_summary") or ""
    return {
        "resume_id": rid, "status": "ok",
        "gt_present": bool(gt_text), "cand_present": bool(cand_text),
        "gt_len_chars": len(gt_text) if gt_text else 0,
        "cand_len_chars": len(cand_text) if cand_text else 0,
        "presence_match": bool(gt_text) == bool(cand_text),
    }


def main() -> None:
    we_rows, edu_rows, summary_rows = [], [], []
    for row in load_mapping():
        rid = row["ResumeID"]
        gt_path = GT_DIR / row["JSON"]
        module_dir = MODULES_DIR / Path(row["Markdown"]).stem
        if not gt_path.exists() or not module_dir.exists():
            continue
        gt = json.loads(gt_path.read_text())

        we = analyze_work_experience(rid, gt, module_dir)
        if we:
            we_rows.append(we)
        edu = analyze_education(rid, gt, module_dir)
        if edu:
            edu_rows.append(edu)
        summ = analyze_summary(rid, gt, module_dir)
        if summ:
            summary_rows.append(summ)

    for rows, path in (
        (we_rows, "benchmark/baselines/t2_phase0_we_entity_comparison.csv"),
        (edu_rows, "benchmark/baselines/t2_phase0_education_entity_comparison.csv"),
        (summary_rows, "benchmark/baselines/t2_phase0_summary_presence_check.csv"),
    ):
        if not rows:
            continue
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path} ({len(rows)} rows)")

    print("\n=== Work Experience summary ===")
    ok = [r for r in we_rows if r["status"] == "ok"]
    print(f"{len(ok)}/{len(we_rows)} resumes with valid WE output")
    exact_count = sum(1 for r in ok if r["entity_count_exact_match"])
    print(f"Entity count exact match: {exact_count}/{len(ok)}")
    recalls = [r["entity_recall"] for r in ok if r["entity_recall"] is not None]
    print(f"Mean entity recall: {sum(recalls)/len(recalls):.3f}" if recalls else "n/a")

    print("\n=== Education summary ===")
    ok_e = [r for r in edu_rows if r["status"] == "ok"]
    print(f"{len(ok_e)}/{len(edu_rows)} resumes with valid Education output")
    exact_count_e = sum(1 for r in ok_e if r["entity_count_exact_match"])
    print(f"Entity count exact match: {exact_count_e}/{len(ok_e)}")
    recalls_e = [r["entity_recall"] for r in ok_e if r["entity_recall"] is not None]
    print(f"Mean entity recall: {sum(recalls_e)/len(recalls_e):.3f}" if recalls_e else "n/a")

    print("\n=== Summary presence ===")
    ok_s = [r for r in summary_rows if r["status"] == "ok"]
    mismatches = [r for r in ok_s if not r["presence_match"]]
    print(f"{len(ok_s)} resumes checked, {len(mismatches)} presence mismatches")


if __name__ == "__main__":
    main()
