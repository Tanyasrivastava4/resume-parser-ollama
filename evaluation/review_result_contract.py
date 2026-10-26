"""
evaluation/review_result_contract.py

Evidence Review Result Contract (Milestone V5.2).

Milestone V5.1 (evaluation/human_review_evidence_report.py) already
produces a plain, JSON-serializable, per-resume human-review report
from an already-built ParserPipelineResult. This module does NOT
reimplement or reinterpret any part of that report -- it exists solely
to package V5.1's own output alongside the existing, unmodified
extracted-result contract into one stable, serializable structure a
future consumer (a frontend, an external service, a CLI) can retrieve
without needing to understand ParserPipelineResult, ModuleEvidenceRecord,
or any other internal Python evidence type.

Step 1 of this milestone found NO existing REST API, web framework,
response-model layer, or frontend anywhere in this repository (no
fastapi/flask/django/uvicorn dependency in requirements.txt; no
app.route/@app/APIRouter/Blueprint anywhere in the codebase). The real,
already-established product boundary today is:
  (a) ParserPipeline.run() -> ParserPipelineResult, the Python service
      interface every real caller (scripts/run_parser_pipeline.py, this
      project's only production CLI entry point) already uses directly;
  (b) the on-disk output/json_final/<resume>.json file ParserPipeline
      itself writes -- byte-for-byte the same merged_json this module
      reads, per pipeline/parser_pipeline.py's own _save_final_json().
This is Outcome B (per this milestone's own decision gate): no API
exists, so this module defines a lightweight, serializable contract
function at that existing boundary -- no network infrastructure, no new
web framework, nothing that did not already exist as a plain Python
call.

This module does NOT:
  - modify pipeline/parser_pipeline.py, ParserPipelineResult, or any
    extraction service -- confirmed by this milestone's own empty
    `git diff --stat` (see docs/MILESTONE_V5_2_EVIDENCE_REVIEW_API_
    CONTRACT.md);
  - change what ParserPipeline.run() saves to disk, or where -- the
    existing output/json_final/<resume>.json contract is completely
    untouched; a consumer who never calls this module's functions sees
    no difference whatsoever;
  - embed review evidence inside `parsed_result` -- the two are always
    separate top-level keys in the structure this module returns, never
    merged, exactly mirroring V5.1's/V3.9's own "evidence is additive,
    never embedded in extracted output" principle;
  - reconstruct, reinterpret, or duplicate any explanation text --
    `review_report` is exactly evaluation.human_review_evidence_report.
    build_resume_evidence_report()'s own return value, passed through
    unmodified. V5.1 remains the single source of truth for every
    human-review explanation; this module never writes its own;
  - invent a numeric confidence score, a severity score, a cross-module
    ranking, or ACCEPT/REJECT/ESCALATE semantics -- none of that exists
    in V5.1's report, and this module adds nothing to it beyond the
    packaging described above;
  - call an LLM, run the pipeline itself, or require one to be
    available -- every function here takes an already-completed
    ParserPipelineResult as input, exactly like V5.1's own
    build_resume_evidence_report().

Milestone V5.3 adds write_review_result_contract() and
review_report_output_path() below -- a purely additive, opt-in
persistence step for this same contract. This does NOT touch
pipeline/parser_pipeline.py's own output-writing (_save_module_json()/
_save_final_json(), both unmodified -- output/json_final/<resume>.json
is written exactly as before, whether or not a caller ever persists a
review contract). The new artifact is written to
<output_root>/review_reports/<stem>.json -- a sibling directory of
output/json_final/, named and structured to mirror that existing
convention exactly (same stem-derived filename, same
mkdir(parents=True, exist_ok=True) + plain write_text() pattern
_save_final_json() itself already uses -- this codebase has no atomic
temp-file+rename write pattern anywhere to be consistent with, so none
is introduced here). Persisting a review contract never re-runs
extraction and never calls an LLM -- it only serializes an
already-completed ParserPipelineResult, exactly like
build_review_result_contract() itself already does.
"""

import json
from pathlib import Path
from typing import Any, Dict, Union

from evaluation.human_review_evidence_report import build_resume_evidence_report

# Sibling of "json_final" and "modules" under the same OUTPUT_DIR root
# ParserPipeline._save_final_json()/_save_module_json() already write
# to -- see pipeline/parser_pipeline.py, unmodified by this milestone.
REVIEW_REPORTS_SUBDIR = "review_reports"

# Bumped only if this contract's own top-level shape changes in a
# backward-incompatible way (e.g. a key is renamed or removed) -- an
# additive change (a new optional key) does not require a bump. A
# future consumer can use this to detect a shape it doesn't understand
# yet, without this project needing any actual API-versioning
# infrastructure (routing, content negotiation, etc.) to provide it.
CONTRACT_VERSION = "1.0"

# Modules this project's parser extracts but has no evidence-based
# review conclusion for (Milestones V4.2/V4.3 closed the Skills
# evidence investigation, NO-GO -- decomposition-based extraction was
# found to actively degrade recovery). Listed explicitly in every
# contract this module builds so a consumer can tell "no review
# evidence exists for this module" apart from "review evidence exists
# and found nothing" -- the same "absence is not silence" discipline
# V5.1's own report already applies to individual fields, extended here
# to whole modules.
MODULES_WITHOUT_REVIEW_EVIDENCE = ("skills",)


def build_review_result_contract(resume_id: str, stem: str, result: Any) -> Dict[str, Any]:
    """Build the full, serializable review-result contract for one
    resume from an already-completed ParserPipelineResult.

    This function never mutates `result` (or any evidence object it
    holds) and never calls ParserPipeline.run() itself -- see this
    module's own top docstring.

    Args:
        resume_id: Caller-supplied identifier for this resume (e.g. the
            mapping.csv ResumeID, or any other stable string).
        stem: Caller-supplied filename stem for this resume.
        result: An already-produced ParserPipelineResult.

    Returns:
        A plain, JSON-serializable dict:
            {
                "contract_version": "1.0",
                "resume_id": ..., "stem": ...,
                "parsed_result": result.merged_json,
                "review_report": build_resume_evidence_report(...),
                "modules_without_review_evidence": ["skills"],
            }
        `parsed_result` is the EXACT SAME object
        ParserPipeline.run() already produces and already saves to
        output/json_final/<resume>.json -- never a copy, never altered,
        never merged with `review_report`. An existing consumer reading
        only `parsed_result` (or the equivalent on-disk file) receives
        identical data to before this milestone existed.
    """
    return {
        "contract_version": CONTRACT_VERSION,
        "resume_id": resume_id,
        "stem": stem,
        "parsed_result": result.merged_json,
        "review_report": build_resume_evidence_report(resume_id, stem, result),
        "modules_without_review_evidence": list(MODULES_WITHOUT_REVIEW_EVIDENCE),
    }


def serialize_review_result_contract(contract: Dict[str, Any]) -> str:
    """Serialize a contract dict (from build_review_result_contract())
    to a JSON string. A thin, explicit wrapper over json.dumps -- kept
    as its own function so the exact serialization contract (key
    ordering not guaranteed, UTF-8, indent=2) is documented and tested
    in one place rather than left to each caller's own json.dumps call.
    Pure -- never mutates `contract`."""
    return json.dumps(contract, indent=2, ensure_ascii=False)


def deserialize_review_result_contract(payload: str) -> Dict[str, Any]:
    """Parse a JSON string produced by serialize_review_result_contract()
    (or any equivalent JSON encoding of the same shape) back into a
    plain dict. A thin wrapper over json.loads, kept as its own function
    for the same reason as serialize_review_result_contract()."""
    return json.loads(payload)


def review_report_output_path(output_root: Union[str, Path], stem: str) -> Path:
    """The stable, deterministic path a review-result contract for
    `stem` is written to under `output_root` -- always
    <output_root>/review_reports/<stem>.json, mirroring
    ParserPipeline._save_final_json()'s own
    <output_root>/json_final/<stem>.json convention exactly (same
    stem-derived filename, same flat single-file-per-resume layout).

    Args:
        output_root: The root output directory (e.g. config.OUTPUT_DIR
            -- the same root json_final/ and modules/ already live
            under). This function does not default or guess this path
            itself; the caller supplies it explicitly, the same way
            ParserPipeline itself hardcodes OUTPUT_DIR rather than
            accepting a caller-supplied override (see
            pipeline/parser_pipeline.py's own module docstring).
        stem: The resume's filename stem (markdown_path.stem) -- the
            same identifier _save_final_json() already uses for
            output/json_final/<stem>.json.

    Returns:
        A Path, not yet created on disk.
    """
    return Path(output_root) / REVIEW_REPORTS_SUBDIR / f"{stem}.json"


def write_review_result_contract(
    resume_id: str, stem: str, result: Any, output_root: Union[str, Path]
) -> Path:
    """Build, serialize, and persist a review-result contract for one
    resume to <output_root>/review_reports/<stem>.json (Milestone
    V5.3).

    This is a purely additive, opt-in step: it never modifies
    `result`, never touches output/json_final/<stem>.json or
    output/modules/<stem>/*.json (ParserPipeline's own, separate output
    -- completely untouched, unmodified, and unaware this function
    exists), never re-runs extraction, and never calls an LLM -- it
    only reads an already-completed ParserPipelineResult, exactly like
    build_review_result_contract() itself.

    Args:
        resume_id: Passed through to build_review_result_contract().
        stem: Passed through to build_review_result_contract() and used
            to derive the output path via review_report_output_path().
        result: An already-produced ParserPipelineResult.
        output_root: The root output directory review reports are
            written under (see review_report_output_path()).

    Returns:
        The Path the contract was written to.

    Raises:
        OSError: If the target directory cannot be created or the file
            cannot be written (e.g. a permissions error) -- propagated
            unmodified, exactly as any other filesystem write in this
            codebase already does (see _save_final_json()'s own
            docstring; no special-case handling is added here that
            doesn't already exist for the analogous existing write).
    """
    contract = build_review_result_contract(resume_id, stem, result)
    payload = serialize_review_result_contract(contract)
    path = review_report_output_path(output_root, stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path
