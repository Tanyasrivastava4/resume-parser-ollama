"""
scripts/_milestone_t4_phase1a_evaluate.py

Milestone T4 / Phase 1A -- evaluation of the 16 real checkpoints from
scripts/_milestone_t4_phase1a_validation_run.py. ZERO new LLM calls.

Computes, per real result:
  1. Entity count (structural/entity-detection signal only -- no GT
     comparison, since none exists for these modules).
  2. Source-grounding, reusing the FROZEN, UNMODIFIED V6.4 fidelity-
     proxy word-boundary lexical check (scripts/_milestone_v6_4_fidelity_proxy.py)
     against each entity's key text fields.
  3. T3's own misrouting detectors (validators/misrouting_detector.py,
     unmodified), applied in both directions:
     - certification_content_in_skills, run against the REAL baseline
       Skills output for the same resume (output/modules/<stem>/skills.json)
     - project_content_in_work_experience, run against the REAL
       baseline Work Experience output for the same resume, using this
       run's own real Projects result as the cross-reference (not a
       hand-built fixture this time -- a real one exists).

Writes: benchmark/baselines/t4_phase1a_evaluation_results.csv
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

from scripts._milestone_v6_4_fidelity_proxy import item_is_grounded  # noqa: E402 -- FROZEN, unmodified
from scripts._milestone_v6_6_boundary_correction import find_skills_section  # noqa: E402
from validators.misrouting_detector import (  # noqa: E402
    detect_certification_content_in_skills,
    detect_project_content_in_work_experience,
)

CKPT_DIR = Path(
    "/private/tmp/claude-501/-Users-ruchigupta-Documents-ResumeAIBenchmark/"
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/t4_phase1a_checkpoints"
)
MODULES_DIR = Path("output/modules")
MARKDOWN_DIR = Path("output/markdown")


def _load_population() -> list[dict]:
    with open("benchmark/baselines/t4_phase1a_frozen_validation_population.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _full_markdown_body(stem: str) -> str:
    """Whole-document body used for grounding checks -- Certifications/
    Projects content can legitimately sit outside the narrow Skills-
    section boundary, so (unlike the Skills fidelity proxy's own scoped
    body) grounding here is checked against the FULL document text."""
    return (MARKDOWN_DIR / f"{stem}.md").read_text(encoding="utf-8", errors="replace")


def main() -> None:
    population = _load_population()
    rows = []

    for row in population:
        rid, stem = row["resume_id"], row["stem"]
        body = _full_markdown_body(stem)

        we_path = MODULES_DIR / stem / "work_experience.json"
        skills_path = MODULES_DIR / stem / "skills.json"
        real_we = json.loads(we_path.read_text())["work_experience"] if we_path.exists() else []
        real_skills = json.loads(skills_path.read_text())["skills"] if skills_path.exists() else []

        for module, key, name_field in (("certifications", "certifications", "name"), ("projects", "projects", "name")):
            ckpt = CKPT_DIR / f"{rid}__{module}.pkl"
            with ckpt.open("rb") as f:
                d = pickle.load(f)
            r = d["result"]
            if r["status"] != "ok":
                rows.append({"resume_id": rid, "module": module, "status": r["status"],
                             "n_entities": None, "n_grounded": None, "n_ungrounded": None,
                             "ungrounded_names": r.get("error"), "misrouting_flags": None})
                continue

            entities = r["json_object"][key]
            grounded_flags = []
            for e in entities:
                text = e.get(name_field) or ""
                g = item_is_grounded(text, body)
                grounded_flags.append((text, g["grounded"], g["reason"]))

            n_grounded = sum(1 for _, g, _ in grounded_flags if g)
            ungrounded = [t for t, g, _ in grounded_flags if not g]

            misrouting_flags = []
            if module == "certifications":
                misrouting_flags = detect_certification_content_in_skills(real_skills)
            else:
                misrouting_flags = detect_project_content_in_work_experience(real_we, entities)

            rows.append({
                "resume_id": rid, "module": module, "status": "ok",
                "n_entities": len(entities), "n_grounded": n_grounded,
                "n_ungrounded": len(ungrounded), "ungrounded_names": "; ".join(ungrounded),
                "misrouting_flags": len(misrouting_flags),
                "misrouting_detail": "; ".join(str(f) for f in misrouting_flags) if misrouting_flags else "",
            })

    out = Path("benchmark/baselines/t4_phase1a_evaluation_results.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")

    for r in rows:
        print(f"{r['resume_id']:5} {r['module']:15} status={r['status']:8} n={r['n_entities']} "
              f"grounded={r['n_grounded']} ungrounded={r['n_ungrounded']} "
              f"misroute_flags={r['misrouting_flags']}")
        if r["n_ungrounded"] and int(r["n_ungrounded"]) > 0:
            print(f"      UNGROUNDED: {r['ungrounded_names']}")


if __name__ == "__main__":
    main()
