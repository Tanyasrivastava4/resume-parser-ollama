"""
scripts/_milestone_t8_phase2a_corpus_structural_audit.py

Milestone T8/Phase 2A -- Phase C corpus-wide structural audit.

ZERO new LLM calls. Uses only:
  - the real, committed Stage-1 markdown (output/markdown/<stem>.md)
  - the real, current, UNMODIFIED WorkExperienceLocator
    (services/work_experience_locator.py)
  - the real, already-committed WE extraction output
    (output/modules/<stem>/work_experience.json)
  - the real ground truth (benchmark/ground_truth/adjudicated_v1/)

For each of the 22 real corpus resumes, records:
  - GT work_experience entity count
  - WorkExperienceLocator span count (deterministic, current code)
  - real committed WE-extraction entity count
  - whether a Projects-shaped heading exists in the markdown, and
    whether it sits after the Experience section
  - whether any real WE company name fuzzy-overlaps a heading-line
    found under a detected Projects section (the resume-001 pattern)

Writes: benchmark/baselines/t8_phase2a_corpus_structural_audit.csv
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

from services.work_experience_locator import WorkExperienceLocator  # noqa: E402
from validators.misrouting_detector import _fuzzy_contains  # noqa: E402

MAPPING_CSV = Path("benchmark/ground_truth/adjudicated_v1/mapping.csv")
MARKDOWN_DIR = Path("output/markdown")
MODULES_DIR = Path("output/modules")
GT_DIR = Path("benchmark/ground_truth/adjudicated_v1")

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_PROJECT_SECTION_WORDS = re.compile(r"\bproject", re.IGNORECASE)
_EXPERIENCE_SECTION_WORDS = re.compile(
    r"\b(experience|employment|work history)\b", re.IGNORECASE
)


def _load_population() -> list[dict]:
    with MAPPING_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _headings(markdown_text: str) -> list[tuple[int, int, str]]:
    """Return (line_index, level, text) for every markdown heading line."""
    out = []
    for i, line in enumerate(markdown_text.splitlines()):
        m = _HEADING_RE.match(line)
        if m:
            out.append((i, len(m.group(1)), m.group(2).strip()))
    return out


def _find_projects_heading(headings: list[tuple[int, int, str]]) -> tuple[int, int, str] | None:
    for h in headings:
        if _PROJECT_SECTION_WORDS.search(h[2]) and not _EXPERIENCE_SECTION_WORDS.search(h[2]):
            return h
    return None


def _find_experience_heading(headings: list[tuple[int, int, str]]) -> tuple[int, int, str] | None:
    for h in headings:
        if _EXPERIENCE_SECTION_WORDS.search(h[2]) or h[2].strip().lower() in (
            "experience", "work experience", "current employment", "professional experience",
        ):
            return h
    return None


def main() -> None:
    population = _load_population()
    locator = WorkExperienceLocator()
    rows = []

    for row in population:
        rid = row["ResumeID"]
        stem = Path(row["Markdown"]).stem
        gt_path = GT_DIR / row["JSON"]
        md_path = MARKDOWN_DIR / f"{stem}.md"
        modules_path = MODULES_DIR / stem / "work_experience.json"

        if not md_path.exists() or not gt_path.exists():
            rows.append({"resume_id": rid, "stem": stem, "status": "MISSING_INPUT"})
            continue

        markdown_text = md_path.read_text(encoding="utf-8")
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        gt_jobs = gt.get("work_experience", [])
        gt_companies = [j.get("company") for j in gt_jobs if isinstance(j, dict)]

        # Real, current, unmodified deterministic locator.
        spans = locator.locate(markdown_text)

        # Real, already-committed WE extraction output (if present).
        if modules_path.exists():
            we_extracted = json.loads(modules_path.read_text(encoding="utf-8"))
            extracted_jobs = we_extracted.get("work_experience", [])
        else:
            extracted_jobs = None

        headings = _headings(markdown_text)
        proj_heading = _find_projects_heading(headings)
        exp_heading = _find_experience_heading(headings)

        projects_after_experience = None
        if proj_heading and exp_heading:
            projects_after_experience = proj_heading[0] > exp_heading[0]

        # Resume-001-shaped check: does any real GT company NOT
        # fuzzy-match, while any EXTRACTED (non-GT) company DOES
        # fuzzy-match a heading-line found under the Projects section?
        false_entity_candidates = []
        if extracted_jobs is not None and proj_heading:
            proj_section_start = proj_heading[0]
            # collect heading-shaped lines strictly after the Projects heading
            sub_heading_texts = [h[2] for h in headings if h[0] > proj_section_start]
            for job in extracted_jobs:
                if not isinstance(job, dict):
                    continue
                company = job.get("company")
                if not company:
                    continue
                is_gt = any(
                    isinstance(gtc, str) and gtc.strip().lower() == str(company).strip().lower()
                    for gtc in gt_companies
                )
                if is_gt:
                    continue
                for sub in sub_heading_texts:
                    if _fuzzy_contains(company, sub):
                        false_entity_candidates.append((company, sub))
                        break

        rows.append({
            "resume_id": rid,
            "stem": stem,
            "status": "ok",
            "gt_job_count": len(gt_jobs),
            "locator_span_count": len(spans),
            "extracted_job_count": (len(extracted_jobs) if extracted_jobs is not None else None),
            "locator_matches_gt_count": len(spans) == len(gt_jobs),
            "extracted_matches_gt_count": (
                len(extracted_jobs) == len(gt_jobs) if extracted_jobs is not None else None
            ),
            "has_projects_heading": proj_heading is not None,
            "projects_heading_text": proj_heading[2] if proj_heading else None,
            "projects_after_experience": projects_after_experience,
            "false_entity_candidates": (
                "; ".join(f"{c!r}<-{s!r}" for c, s in false_entity_candidates)
                if false_entity_candidates else "none"
            ),
        })

    out = Path("benchmark/baselines/t8_phase2a_corpus_structural_audit.csv")
    fieldnames = [
        "resume_id", "stem", "status", "gt_job_count", "locator_span_count",
        "extracted_job_count", "locator_matches_gt_count", "extracted_matches_gt_count",
        "has_projects_heading", "projects_heading_text", "projects_after_experience",
        "false_entity_candidates",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"wrote {out} ({len(rows)} rows)")

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    print(f"\nResumes with 'false_entity_candidates' != none:")
    for r in ok_rows:
        if r["false_entity_candidates"] != "none":
            print(f"  {r['resume_id']} ({r['stem']}): {r['false_entity_candidates']}")

    print(f"\nlocator_matches_gt_count: {sum(1 for r in ok_rows if r['locator_matches_gt_count'])}/{len(ok_rows)}")
    print(f"extracted_matches_gt_count: {sum(1 for r in ok_rows if r['extracted_matches_gt_count'])}/{len(ok_rows)}")


if __name__ == "__main__":
    main()
