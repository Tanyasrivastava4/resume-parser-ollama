"""
scripts/_milestone_t5_phase1b_integration_verification.py

Milestone T5 / Phase 1B -- verifies the newly-wired Project module
integration against the T4 frozen 8-resume population, ZERO new LLM
calls, ZERO file writes to output/.

For each of the 8 resumes, combines:
  - the REAL, already-committed baseline module_results for the 7
    pre-existing modules (output/modules/<resume>/*.json, the frozen
    Milestone-23.67 reference corpus -- read-only, never touched)
  - T4/Phase 1A's own REAL Project extraction checkpoint result
    (scripts/_milestone_t4_phase1a_validation_run.py's real
    output, 16 real Ollama calls already made and disclosed in that
    milestone)

...then runs them through the REAL, unmodified ParserMerger.merge()
and REAL, unmodified SchemaValidator.validate() (against the T3-
extended resume_schema_v1.json) -- entirely in-memory, confirming the
exact behavior a live pipeline run would produce for these 8 resumes,
without spending any new call budget and without writing to (or
risking corruption of) the frozen output/ reference corpus.

This is a deliberate, disclosed methodology choice: a live pipeline
run against a real frozen-corpus resume would overwrite that resume's
own reference output/modules/*.json and output/json_final/*.json files
-- files every other milestone's own integrity checks depend on
remaining stable. One genuinely live, non-simulated, end-to-end
pipeline run WAS separately made this milestone (the self-cleaning
synthetic-resume integration test,
tests/test_parser_pipeline.py::test_real_pipeline_end_to_end_integration,
8 real calls, verified to leave output/ byte-for-byte unchanged
afterward) -- this script provides the broader, real-data-grounded
8-resume coverage the frozen corpus itself cannot safely be re-run
against.

Writes: benchmark/baselines/t5_phase1b_integration_verification.csv
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

from services.parser_merger import ParserMerger  # noqa: E402 -- REAL, unmodified
from validators.misrouting_detector import detect_project_content_in_work_experience  # noqa: E402
from validators.schema_validator import SchemaValidationError, SchemaValidator  # noqa: E402 -- REAL, unmodified

MODULES_DIR = Path("output/modules")
T4_CKPT_DIR = Path(
    "/private/tmp/claude-501/-Users-ruchigupta-Documents-ResumeAIBenchmark/"
    "0db0bd62-77c0-4395-8aab-ca1cf5bdaeea/scratchpad/t4_phase1a_checkpoints"
)
PRE_EXISTING_MODULE_KEYS = (
    "personal_information", "contact_information", "skills",
    "professional_summary", "languages", "work_experience", "education",
)


def _load_population() -> list[dict]:
    with open("benchmark/baselines/t4_phase1a_frozen_validation_population.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    population = _load_population()
    merger = ParserMerger()
    validator = SchemaValidator()
    rows = []

    for row in population:
        rid, stem = row["resume_id"], row["stem"]
        module_dir = MODULES_DIR / stem

        # Real, pre-existing, frozen module_results (read-only).
        module_results = {}
        for key in PRE_EXISTING_MODULE_KEYS:
            p = module_dir / f"{key}.json"
            module_results[key] = json.loads(p.read_text())

        # T4's real Project checkpoint for this resume.
        ckpt_path = T4_CKPT_DIR / f"{rid}__projects.pkl"
        with ckpt_path.open("rb") as f:
            ckpt = pickle.load(f)
        assert ckpt["result"]["status"] == "ok", f"{rid}: T4 project checkpoint was not 'ok'"
        module_results["projects"] = ckpt["result"]["json_object"]

        # Real merge + real schema validation, in-memory only.
        merged = merger.merge(list(module_results.values()))
        try:
            validator.validate(merged)
            schema_status = "valid"
        except SchemaValidationError as e:
            schema_status = f"INVALID: {e}"

        # Backward-compatibility check: the 7 pre-existing modules'
        # values in the merged result must be byte-identical to their
        # own real, frozen source files -- confirms adding "projects"
        # changed nothing else.
        unexpected_diffs = []
        for key in PRE_EXISTING_MODULE_KEYS:
            if merged.get(key) != module_results[key].get(key):
                unexpected_diffs.append(key)

        # Misrouting-safety re-check (T3 detector, unmodified) using
        # the real WE output and the real T4 Project result together.
        we_list = module_results["work_experience"].get("work_experience", [])
        proj_list = module_results["projects"].get("projects", [])
        misrouting_flags = detect_project_content_in_work_experience(we_list, proj_list)

        rows.append({
            "resume_id": rid, "schema_status": schema_status,
            "n_projects": len(proj_list),
            "n_work_experience_entries": len(we_list),
            "unexpected_pre_existing_module_diffs": "; ".join(unexpected_diffs) if unexpected_diffs else "none",
            "misrouting_flags": len(misrouting_flags),
            "misrouting_detail": "; ".join(str(f) for f in misrouting_flags) if misrouting_flags else "",
            "has_certifications_key": "certifications" in merged,
        })

    out = Path("benchmark/baselines/t5_phase1b_integration_verification.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")

    for r in rows:
        print(f"{r['resume_id']:5} schema={r['schema_status']:8} n_projects={r['n_projects']} "
              f"diffs={r['unexpected_pre_existing_module_diffs']} misroute_flags={r['misrouting_flags']} "
              f"has_cert_key={r['has_certifications_key']}")


if __name__ == "__main__":
    main()
