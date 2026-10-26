"""
scripts/_milestone_t10_phase2c_analysis.py

Milestone T10/Phase 2C -- consolidates all 30 real-call checkpoints
into the final results CSV and computes the pre-registered primary/
secondary metrics, reusing existing, unmodified deterministic
infrastructure as measurement tools only (WorkExperienceLocator,
WorkExperienceValidator's own technology-presence check) -- no new
production logic, no modification of either module.
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
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/t10_phase2c_checkpoints"
)
MARKDOWN_DIR = Path("output/markdown")
GT_DIR = Path("benchmark/ground_truth/adjudicated_v1")

POPULATION = {
    "001": ("AMAN_CV_BCA", "resume001.json"),
    "013": ("Kishan Yadav_Data Scientist_Intileo", "resume013.json"),
    "016": ("Mukul Kanojia_Full Stack Developer_Intileo", "resume016.json"),
}

# Known real Project heading names per resume (source-grounded, from
# T7/T8/T9's own direct source-markdown reading) -- used only to
# detect whether a WE entity's company/designation corresponds to a
# Project heading (entity hallucination), not to alter any output.
KNOWN_PROJECT_NAMES = {
    "001": ["SoWhat", "TGES", "Asha Learningy", "Adiroha"],
}

GT_COMPANIES = {
    rid: [j["company"] for j in json.loads((GT_DIR / gt_file).read_text()).get("work_experience", [])]
    for rid, (_, gt_file) in POPULATION.items()
}


def _load(rid: str, condition: str, run: int):
    path = CKPT_DIR / f"{rid}__{condition}_{run}.pkl"
    with path.open("rb") as f:
        return pickle.load(f)


def _is_hallucinated_entity(rid: str, job: dict) -> bool:
    """An entity is hallucinated if its company does not fuzzy-match
    (the same real, unmodified _fuzzy_contains T3/T7 already use in
    production) any real GT company for this resume -- e.g. "Asha Tech
    (Asha Learningogy)" fuzzy-matches GT's "Asha Tech" and is correctly
    NOT flagged; "Adiroha"/"SoWhat"/"TGES"/"Asha Learningy" do not
    fuzzy-match any GT company and ARE flagged."""
    company = str(job.get("company") or "")
    if not company:
        return False
    for gt_company in GT_COMPANIES.get(rid, []):
        if _fuzzy_contains(company, gt_company):
            return False
    return True


def _technology_contamination(rid: str, stem: str, jobs: list, locator: WorkExperienceLocator, markdown_text: str):
    """Returns (total_tech_items, contaminated_count) using the real,
    unmodified WEV-015 technology-presence check against each job's own
    matched locator span."""
    spans = locator.locate(markdown_text)
    span_by_index = locator.match_jobs_to_spans(jobs, spans)
    total, contaminated = 0, 0
    for i, job in enumerate(jobs):
        techs = job.get("technologies") or []
        span = span_by_index.get(i)
        for t in techs:
            total += 1
            if span is None:
                continue  # cannot judge without a matched span -- not counted as contaminated
            if not _technology_present_in_span(t, span.markdown_text):
                contaminated += 1
    return total, contaminated


def main() -> None:
    locator = WorkExperienceLocator()
    rows = []

    for rid, (stem, _gt_file) in POPULATION.items():
        markdown_text = (MARKDOWN_DIR / f"{stem}.md").read_text(encoding="utf-8")

        for condition in ("baseline", "candidate"):
            for run in range(1, 6):
                d = _load(rid, condition, run)
                jobs = (d["json_object"] or {}).get("work_experience", []) if d["json_object"] else []
                entity_count = len(jobs)
                hallucinated = [j for j in jobs if _is_hallucinated_entity(rid, j)]
                gt_count = len(GT_COMPANIES[rid])
                omitted = max(0, gt_count - (entity_count - len(hallucinated)))
                total_tech, contaminated_tech = _technology_contamination(
                    rid, stem, jobs, locator, markdown_text
                )
                rows.append({
                    "resume_id": rid, "condition": condition, "run": run,
                    "status": d["status"], "attempts": d["attempts"], "elapsed_s": d["elapsed_s"],
                    "input_chars": d["input_chars"],
                    "entity_count": entity_count, "gt_entity_count": gt_count,
                    "hallucinated_entity_count": len(hallucinated),
                    "hallucinated_companies": "; ".join(str(j.get("company")) for j in hallucinated),
                    "entity_omission_count": omitted,
                    "total_technologies": total_tech,
                    "contaminated_technologies": contaminated_tech,
                    "contamination_rate": (contaminated_tech / total_tech) if total_tech else 0.0,
                })

    out = Path("benchmark/baselines/t10_phase2c_pilot_results_full.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")

    # Print per-resume/condition summary
    from collections import defaultdict
    summary = defaultdict(list)
    for r in rows:
        summary[(r["resume_id"], r["condition"])].append(r)

    print("\n=== SUMMARY ===")
    for (rid, cond), rs in sorted(summary.items()):
        hall_runs = sum(1 for r in rs if r["hallucinated_entity_count"] > 0)
        omit_runs = sum(1 for r in rs if r["entity_omission_count"] > 0)
        avg_contam = sum(r["contamination_rate"] for r in rs) / len(rs)
        avg_elapsed = sum(r["elapsed_s"] for r in rs if r["elapsed_s"]) / len(rs)
        print(f"{rid} {cond}: hallucination {hall_runs}/5, omission {omit_runs}/5, "
              f"avg_tech_contam_rate={avg_contam:.2f}, avg_elapsed={avg_elapsed:.1f}s, "
              f"entity_counts={[r['entity_count'] for r in rs]}")


if __name__ == "__main__":
    main()
