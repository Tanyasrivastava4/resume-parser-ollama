"""
scripts/_milestone_t12_analysis.py

Milestone T12/Phase 2C -- consolidates all 45 real-call checkpoints
and computes the pre-registered metrics, reusing existing, unmodified
deterministic infrastructure as measurement tools only
(WorkExperienceLocator, WorkExperienceValidator's technology-presence
check, misrouting_detector's fuzzy company matcher) -- no production
logic changed.
"""
from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from services.work_experience_locator import WorkExperienceLocator  # noqa: E402
from services.work_experience_validator import _technology_present_in_span  # noqa: E402
from validators.misrouting_detector import _fuzzy_contains  # noqa: E402

CKPT_DIR = Path(
    "/private/tmp/claude-501/-Users-ruchigupta-Documents-ResumeAIBenchmark/"
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/t12_checkpoints"
)
MARKDOWN_DIR = Path("output/markdown")
GT_DIR = Path("benchmark/ground_truth/adjudicated_v1")

POPULATION = {
    "001": ("AMAN_CV_BCA", "resume001.json"),
    "013": ("Kishan Yadav_Data Scientist_Intileo", "resume013.json"),
    "016": ("Mukul Kanojia_Full Stack Developer_Intileo", "resume016.json"),
}

GT_COMPANIES = {
    rid: [j["company"] for j in json.loads((GT_DIR / gt_file).read_text()).get("work_experience", [])]
    for rid, (_, gt_file) in POPULATION.items()
}
GT_RESP_001 = "Successfully completed a 6-month internship and transitioned to full-time Full-Stack Developer."


def _load(rid, condition, run):
    with (CKPT_DIR / f"{rid}__{condition}_{run}.pkl").open("rb") as f:
        return pickle.load(f)


def _is_hallucinated_entity(rid, job):
    company = str(job.get("company") or "")
    if not company:
        return False
    for gt_company in GT_COMPANIES.get(rid, []):
        if _fuzzy_contains(company, gt_company):
            return False
    return True


def _classify_001_responsibility(resp_list):
    if not resp_list:
        return "EMPTY"
    joined = " ".join(resp_list).strip().rstrip(".")
    gt_norm = GT_RESP_001.strip().rstrip(".")
    if joined.lower() == gt_norm.lower():
        return "EXACT"
    if gt_norm.lower() in joined.lower() or joined.lower() in gt_norm.lower():
        return "NEAR/PARTIAL"
    # check if all real words are present, just resegmented
    return "OTHER"


def _duplication_check(rid, jobs):
    """True if 2+ entities share a fuzzy-matching company (same real
    entity duplicated), among entities that are NOT hallucinated."""
    real_companies = [
        str(j.get("company") or "") for j in jobs if not _is_hallucinated_entity(rid, j)
    ]
    for i in range(len(real_companies)):
        for j in range(i + 1, len(real_companies)):
            if _fuzzy_contains(real_companies[i], real_companies[j]):
                return True
    return False


def _tech_contamination(rid, stem, jobs, locator, markdown_text):
    spans = locator.locate(markdown_text)
    span_by_index = locator.match_jobs_to_spans(jobs, spans)
    total, contaminated = 0, 0
    for i, job in enumerate(jobs):
        techs = job.get("technologies") or []
        span = span_by_index.get(i)
        for t in techs:
            total += 1
            if span is not None and not _technology_present_in_span(t, span.markdown_text):
                contaminated += 1
    return total, contaminated


def main():
    locator = WorkExperienceLocator()
    rows = []

    for rid, (stem, _gt) in POPULATION.items():
        markdown_text = (MARKDOWN_DIR / f"{stem}.md").read_text(encoding="utf-8")
        for condition in ("baseline", "span", "marker"):
            for run in range(1, 6):
                d = _load(rid, condition, run)
                jobs = (d["json_object"] or {}).get("work_experience", []) if d["json_object"] else []
                entity_count = len(jobs)
                hallucinated = [j for j in jobs if _is_hallucinated_entity(rid, j)]
                duplicated = _duplication_check(rid, jobs)
                total_tech, contam_tech = _tech_contamination(rid, stem, jobs, locator, markdown_text)

                resp_classification = ""
                if rid == "001":
                    real_job = next((j for j in jobs if not _is_hallucinated_entity(rid, j)), None)
                    resp_classification = _classify_001_responsibility(
                        real_job.get("responsibilities", []) if real_job else []
                    )

                rows.append({
                    "resume_id": rid, "condition": condition, "run": run,
                    "status": d["status"], "attempts": d["attempts"], "elapsed_s": d["elapsed_s"],
                    "input_chars": d["input_chars"],
                    "entity_count": entity_count, "gt_entity_count": len(GT_COMPANIES[rid]),
                    "hallucinated_count": len(hallucinated),
                    "hallucinated_companies": "; ".join(str(j.get("company")) for j in hallucinated),
                    "duplicated_real_entity": duplicated,
                    "total_technologies": total_tech, "contaminated_technologies": contam_tech,
                    "contamination_rate": (contam_tech / total_tech) if total_tech else 0.0,
                    "resp_classification_001": resp_classification,
                })

    out = Path("benchmark/baselines/t12_pilot_results_full.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")

    from collections import defaultdict
    summary = defaultdict(list)
    for r in rows:
        summary[(r["resume_id"], r["condition"])].append(r)

    print("\n=== SUMMARY ===")
    for (rid, cond), rs in sorted(summary.items()):
        hall = sum(1 for r in rs if r["hallucinated_count"] > 0)
        dup = sum(1 for r in rs if r["duplicated_real_entity"])
        avg_contam = sum(r["contamination_rate"] for r in rs) / len(rs)
        avg_elapsed = sum(r["elapsed_s"] for r in rs if r["elapsed_s"]) / len(rs)
        entity_counts = [r["entity_count"] for r in rs]
        resp = [r["resp_classification_001"] for r in rs] if rid == "001" else None
        print(f"{rid} {cond}: hallucination {hall}/5, duplication {dup}/5, "
              f"avg_tech_contam={avg_contam:.2f}, avg_elapsed={avg_elapsed:.1f}s, "
              f"entities={entity_counts}" + (f", resp={resp}" if resp else ""))


if __name__ == "__main__":
    main()
