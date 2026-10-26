"""
scripts/_milestone_t19_stage1_prep.py

Milestone T19/Phase 2F -- INPUT PREPARATION ONLY, not part of the T19
diagnostic itself.

The T19 diagnostic (deterministic candidate generation, T17/T18's own
unmodified blockify()/generate_candidates_v2()) operates on
output/markdown/<stem>.md, exactly as it did for the original 22-resume
corpus. None of the 6 unseen PDF resumes selected for T19 have this
file yet. Producing it requires running the REAL, UNMODIFIED production
Stage-1 pipeline (services/document_extractor.py), which uses a vision
model (qwen2.5vl:3b, "production's own Stage-1 OCR" model per
docs/MILESTONE_V5_6_SKILLS_CROSS_MODEL_COMPARISON.md) -- one call per
non-blank page, identical to how every one of the 22 original benchmark
resumes' markdown was produced.

This script makes exactly those vision-model calls, once per resume,
with NO other behavior -- it does not call any text-based extraction
module (personal_information, skills, work_experience, etc.), so no
LLM call from this script touches anything the T19 diagnostic itself
measures. Per explicit user direction, these calls are disclosed
separately and explicitly from the diagnostic's own 0-LLM-call count
in the T19 report -- nothing is hidden.

Uses:
- model_name: "qwen2.5vl:3b" (production's own Stage-1 OCR model)
- prompt_id: "resume/extract_markdown" (production's own Stage-1 prompt)
- strategy: None -> resolves to settings.OCR_STRATEGY ("single_page",
  the production default)

Does not modify services/document_extractor.py or any other production
file -- calls it exactly as BatchProcessor/benchmark/run_benchmark.py
already do.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from services.document_extractor import DocumentExtractor  # noqa: E402

MODEL_NAME = "qwen2.5vl:3b"
PROMPT_ID = "resume/extract_markdown"

UNSEEN_PDF_RESUMES = [
    "178480991686_Vachan_Premugh_Dicruz_Founder_Office_BTTR.MEN.pdf",
    "8th_April_Akasa_Kapoor_Resume.pdf",
    "AARTI BAGHOTIYA_Economic Urban Planner_Intileo.pdf",
    "AASHISH PARMAR - Email Marketing Executive - Intileo.pdf",
    "_Adarsh_Bhagat Resume.pdf",
]


def main() -> None:
    extractor = DocumentExtractor()
    total_pages = 0
    for fname in UNSEEN_PDF_RESUMES:
        pdf_path = Path("input/resumes") / fname
        assert pdf_path.exists(), f"missing: {pdf_path}"
        result = extractor.extract(pdf_path, model_name=MODEL_NAME, prompt_id=PROMPT_ID)
        total_pages += result.page_count
        print(f"{fname}: page_count={result.page_count} -> {result.markdown_path}")
    print(f"\nTotal Stage-1 vision-model calls (1 per non-blank page, blank pages "
          f"skip the model per Milestone 15B.5): reported per-resume above. "
          f"Total pages rendered across all 5 resumes: {total_pages} "
          f"(upper bound on call count; blank pages contribute 0 calls each).")


if __name__ == "__main__":
    main()
