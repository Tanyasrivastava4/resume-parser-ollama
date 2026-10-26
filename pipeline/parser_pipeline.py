"""
pipeline/parser_pipeline.py

Parser Pipeline (Milestone 14.7; per-module retry added in Milestone
14.9.2; work_experience wired in as a sixth module in Milestone 17.3;
projects wired in as an eighth module in Milestone T5/Phase 1B).

Wires together every existing modular parsing service into one full
markdown-to-JSON pipeline:

    markdown
        |
        v
    PersonalInformationExtractor  -> output/modules/<resume>/personal_information.json
    ContactInformationExtractor   -> output/modules/<resume>/contact_information.json
                                      (+ observational evidence, Milestone V3.9 --
                                       see ParserPipelineResult.contact_information_evidence;
                                       never alters the .json file above)
    SkillsExtractor                -> output/modules/<resume>/skills.json
    ProfessionalSummaryExtractor   -> output/modules/<resume>/professional_summary.json
    LanguagesExtractor              -> output/modules/<resume>/languages.json
    WorkExperienceExtractor         -> output/modules/<resume>/work_experience.json
    EducationExtractor               -> output/modules/<resume>/education.json
    ProjectExtractor                  -> output/modules/<resume>/projects.json
                                      (Milestone T5/Phase 1B -- no dedicated
                                       validator/evidence-record, per the
                                       constructor's own docstring; T3 built
                                       and T4 real-call-validated this
                                       extractor before this wiring)
        |
        v
    ParserMerger                   -> combined resume dict
        |
        v
    SchemaValidator (resume_schema_v1.json)
        |
        v
    output/json_final/<resume>.json

Output root (remediation R-08, 2026-10-17): every path above is relative
to `output_root`, an optional constructor argument. When it is omitted
(every pre-R-08 caller) the root is config.OUTPUT_DIR, i.e. exactly the
fixed output/modules/ and output/json_final/ locations this pipeline has
always written to. Passing a different root lets a benchmark or a second
concurrent run keep its output apart without subclassing (which is what
services/isolated_parser_pipeline.py previously had to do).

This pipeline starts from an already-rendered markdown file. It does not
render PDFs (no PDFLoader, no DocumentExtractor), does not scan
input/resumes/ itself, and does not benchmark anything — it is pure
orchestration. Every step's actual logic lives in the service it calls;
this module only decides the order to call them in, what to pass between
them, and where to save each stage's output.

The module extractors run sequentially, in the order the milestone
specifies. If any single step fails (markdown missing, a bad JSON
response, a schema violation, a merge conflict), the whole run() call
raises immediately — this is a single-document pipeline, not a resilient
multi-document batch processor (that's BatchProcessor, a separate,
unrelated component).

Milestone 17.3: WorkExperienceExtractor (built and tested since
Milestone 15.2, but previously never wired into this pipeline — see
docs/PARSER_ARCHITECTURE.md, Milestone 17.2) is added as a sixth module
extractor, using the exact same pattern as the five that already existed
(constructor-injected, appended to _module_extractors, covered by the
same per-step retry and save-to-disk logic with no special-casing). No
other step, and no shared infrastructure (ParserMerger, the retry logic
below, SchemaValidator), required any change to support it.

Milestone 17.6: WorkExperienceValidator (designed in
docs/WORK_EXPERIENCE_VALIDATION.md, Milestone 17.5) runs immediately
after WorkExperienceExtractor.extract() returns, and before its result
is added to module_results / handed to ParserMerger. This IS a
special case in the otherwise-generic module loop — no other module has
a validation step, because the audit that motivated this
(docs/WORK_EXPERIENCE_AUDIT.md, Milestone 17.4) only covered
work_experience. The validator is deterministic (no model call, no
retry) and only ever nulls out an already-extracted field or flips
currently_working/end_date to be internally consistent — it never adds,
removes, or reorders job entries.

Milestone 17.7: this pipeline now reads the resume's markdown text once
(right before the work_experience step, not at the top of run() — every
earlier step has already successfully read markdown_path by then, so
this can't introduce a new "missing markdown" failure path) and passes
it to WorkExperienceValidator.validate(), establishing the API future
markdown-aware rules will use (services/work_experience_locator.py,
built standalone in this same milestone, is the reusable span-finding
infrastructure those future rules will need — it is not yet wired into
the validator). Phase 1's five rules ignore the markdown text; behavior
is unchanged from Milestone 17.6.

Milestone 18.3: EducationExtractor (built in Milestone 18.2, but
previously never wired into this pipeline — see
docs/EDUCATION_EXTRACTOR_RESULTS.md) is added as a seventh module
extractor, immediately after work_experience, using the exact same
pattern Milestone 17.3 established for work_experience itself
(constructor-injected, appended to _module_extractors, covered by the
same per-step retry and save-to-disk logic with no special-casing).
At that point education had no validator step. No other step, and no
shared infrastructure (ParserMerger, the retry logic below,
SchemaValidator), required any change to support it.

Milestone 18.10: EducationValidator (designed in
docs/EDUCATION_VALIDATION.md, Milestone 18.9) runs immediately after
EducationExtractor.extract() returns, and before its result is added
to module_results / handed to ParserMerger — the same special-cased
position in the module loop WorkExperienceValidator already
established for work_experience in Milestone 17.6, now with a second
instance, `if module_key == "education": ...`. The validator is
deterministic (no model call, no retry, no markdown, no cross-module
input for this Phase 1) and only ever nulls out an already-extracted
field, splits a date-range string into its two already-nullable
fields, or flips currently_studying to be internally consistent with a
stated grade — it never adds, removes, or reorders education entries.

Milestone 14.9.2: small local models occasionally produce a bad response
for one step (InvalidJSONError) or one that doesn't conform to that
step's own schema (SchemaValidationError), and the same resume often
succeeds on an immediate retry. Each call that re-invokes the model —
every module extractor's extract() — is retried in place, up to 2 extra
times (3 attempts total), before the error is allowed to propagate.
Nothing else is retried: not the whole resume, and not the final
schema_validator.validate(merged_json) call, since that call takes
already-fixed data and retrying it with the same input would just fail
the same way every time.

Milestone V3.9: ContactInformationExtractor's raw result now also feeds
build_contact_information_evidence() (services/evidence_record.py,
Milestone V3.8) immediately after extraction — the same special-cased
position in the module loop work_experience/education validation already
use, now with a third branch, `if module_key == "contact_information":`.
Unlike the work_experience/education branches, this one is PURELY
OBSERVATIONAL: json_object is never reassigned, so there is no code path
through which it can alter contact_information's own returned value —
this is verified directly by a byte-for-byte corpus regression proof, not
merely asserted (see docs/MILESTONE_V3_9_CONTACT_INFORMATION_EVIDENCE_
INTEGRATION.md). The resulting ContactInformationEvidenceRecord is
exposed only via the new, optional ParserPipelineResult.contact_
information_evidence field — never consumed by ParserMerger or
SchemaValidator, and never influences ACCEPT/REJECT/REPAIR/ESCALATE
routing, since no such routing exists for this module. run() also gained
a new, optional `pdf_path` parameter (default None, fully backward
compatible) so PDF hyperlink-annotation evidence can be supplied when
available; when omitted, annotation evidence is simply unavailable
(never guessed, never derived from markdown_path).

Milestone 23.7: JsonDraftGenerator — a separate, whole-document,
single-model-call draft step that used to run before this loop — has
been retired entirely (see docs/JSON_DRAFT_GENERATOR_ARCHITECTURE_
REVIEW.md, docs/JSON_DRAFT_GENERATOR_RETIREMENT_READINESS.md, and
docs/JSON_DRAFT_GENERATOR_RETIREMENT_IMPLEMENTATION_PLAN.md for the
investigation and plan this implements). Its output (draft_json) was
never consumed by ParserMerger or any module extractor — merged_json
has always been, and remains, built exclusively from module_results.
ParserPipelineResult.draft_json is kept as a field, permanently None,
for backward compatibility with existing callers/tests that construct
or read a ParserPipelineResult, but nothing produces a non-None value
for it anymore. services/json_draft_generator.py and
prompts/parser/json_draft.md no longer exist.

Milestone V5.0: the four other already-built, standalone observational
evidence functions -- build_education_evidence()/build_work_experience_
evidence() (Milestone V2.7) and build_languages_evidence()/build_
professional_summary_evidence()/build_personal_information_evidence()
(Milestones V4.1/V4.5/V4.8) -- are now wired into this loop using the
EXACT SAME non-reassigning pattern Milestone V3.9 established for
Contact Information (Skills has no evidence function and remains
untouched):
    - `if module_key == "personal_information":` builds personal_
      information_evidence from the module's own raw json_object and
      this resume's markdown text.
    - `elif module_key == "professional_summary":` builds professional_
      summary_evidence the same way.
    - `elif module_key == "languages":` builds languages_evidence the
      same way.
    - `elif module_key == "work_experience":` (pre-existing mutating
      branch, unchanged) now ALSO calls build_work_experience_evidence()
      immediately afterward, using the POST-validation job list and the
      SAME WorkExperienceValidationReport the branch already produced --
      an additional, non-reassigning statement appended to an existing
      branch, not a new code path that could alter json_object.
    - `elif module_key == "education":` (pre-existing mutating branch,
      unchanged) gains the analogous additional call to build_education_
      evidence().
Every one of these five new calls is purely additive: none of them
reassigns json_object, so none of them can alter any module's own
extracted value -- verified directly by a byte-for-byte corpus
regression proof, not merely asserted (see docs/MILESTONE_V5_0_PARSER_
EVIDENCE_INTEGRATION.md). Each new evidence record is exposed only via
its own new, optional ParserPipelineResult field -- never consumed by
ParserMerger or SchemaValidator, and never influencing any ACCEPT/
REJECT/REPAIR/ESCALATE routing, since no such routing exists for any of
these modules.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

from config import OUTPUT_DIR
from services.base_module_extractor import InvalidJSONError
from services.contact_information_extractor import ContactInformationExtractor
from services.education_extractor import EducationExtractor
from services.education_validator import EducationValidator
from services.education_validator import ValidationReport as EducationValidationReport
from services.evidence_record import (
    ContactInformationEvidenceRecord,
    LanguagesEvidenceRecord,
    ModuleEvidenceRecord,
    PersonalInformationEvidenceRecord,
    ProfessionalSummaryEvidence,
    build_contact_information_evidence,
    build_education_evidence,
    build_languages_evidence,
    build_personal_information_evidence,
    build_professional_summary_evidence,
    build_work_experience_evidence,
)
from services.languages_extractor import LanguagesExtractor
from services.parser_merger import ParserMerger
from services.personal_information_extractor import PersonalInformationExtractor
from services.professional_summary_extractor import ProfessionalSummaryExtractor
from services.project_extractor import ProjectExtractor
from services.skills_extractor import SkillsExtractor
from services.work_experience_extractor import WorkExperienceExtractor
from services.work_experience_validator import ValidationReport, WorkExperienceValidator
from validators.schema_validator import SchemaValidationError, SchemaValidator

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Total attempts allowed for a single step (the initial attempt plus up
# to this many extra retries): 1 initial + 2 retries = 3 attempts.
MAX_ATTEMPTS = 3

# Only these are worth retrying — a fresh model call might just come out
# differently next time. Anything else (missing markdown/prompt/model,
# Ollama being unreachable, etc.) is a configuration problem that a
# retry can't fix.
RETRYABLE_EXCEPTIONS = (InvalidJSONError, SchemaValidationError)


class ParserPipelineError(Exception):
    """Base exception for all Parser Pipeline failures."""


@dataclass
class ParserPipelineResult:
    """Result of running the full parser pipeline over one resume.

    Attributes:
        draft_json: Always None (Milestone 23.7). Previously the raw
            JSON draft produced by the now-retired JsonDraftGenerator
            step; kept as a field, permanently None, for backward
            compatibility with existing callers/tests that construct or
            read a ParserPipelineResult. Was never consumed by
            ParserMerger or any module extractor even when it was
            populated — see this module's own docstring.
        module_results: Mapping of module name to that module's raw
            json_object exactly as its extractor returned it (e.g.
            {"personal_information": {"personal_information": {...}},
            "contact_information": {"contact_information": {...}}, ...}),
            one entry per module extractor.
        merged_json: The final, schema-validated combined resume JSON
            produced by ParserMerger — the shape schemas/resume_schema_v1
            .json expects.
        processing_time: Wall-clock time, in seconds, for the whole
            pipeline run (draft generation, every module, merging, and
            validation).
        retry_count: Total number of retries used across every module
            extractor — 0 if every step succeeded on its first attempt. See
            RETRYABLE_EXCEPTIONS for which failures count.
        work_experience_validation_report: The ValidationReport
            WorkExperienceValidator produced for this resume's
            work_experience array (Milestone 17.6), or None if the
            work_experience module itself failed before validation
            could run.
        education_validation_report: The ValidationReport
            EducationValidator produced for this resume's education
            array (Milestone 18.10), or None if the education module
            itself failed before validation could run.
        contact_information_evidence: The purely observational
            ContactInformationEvidenceRecord built from this resume's
            already-extracted contact_information (Milestone V3.9), or
            None if the contact_information module itself failed before
            evidence-building could run. This is NEVER used to modify,
            delete, replace, or autofill any value in module_results
            or merged_json — see services/evidence_record.py's own
            docstring and docs/MILESTONE_V3_9_CONTACT_INFORMATION_
            EVIDENCE_INTEGRATION.md for the corpus proof that
            attaching it changes nothing about the pipeline's returned
            contact_information value.
        work_experience_evidence: The purely observational
            ModuleEvidenceRecord built from this resume's already-
            validated work_experience array and the SAME
            WorkExperienceValidationReport already produced for
            work_experience_validation_report above (Milestone V5.0),
            or None if the work_experience module itself failed before
            evidence-building could run. NEVER used to modify, delete,
            replace, or autofill any value — see services/evidence_
            record.py's own docstring.
        education_evidence: The purely observational ModuleEvidenceRecord
            built from this resume's already-validated education array
            and the SAME EducationValidationReport already produced for
            education_validation_report above (Milestone V5.0), or None
            if the education module itself failed before evidence-
            building could run. NEVER used to modify, delete, replace,
            or autofill any value.
        languages_evidence: The purely observational
            LanguagesEvidenceRecord built from this resume's already-
            extracted languages array (Milestone V5.0), or None if the
            languages module itself failed before evidence-building
            could run. NEVER used to modify, delete, replace, or
            autofill any value.
        professional_summary_evidence: The purely observational
            ProfessionalSummaryEvidence built from this resume's
            already-extracted professional_summary value (Milestone
            V5.0), or None if the professional_summary module itself
            failed before evidence-building could run. NEVER used to
            modify, delete, replace, or autofill any value.
        personal_information_evidence: The purely observational
            PersonalInformationEvidenceRecord built from this resume's
            already-extracted personal_information object (Milestone
            V5.0), or None if the personal_information module itself
            failed before evidence-building could run. NEVER used to
            modify, delete, replace, or autofill any value.
    """

    draft_json: Any
    module_results: Dict[str, Any] = field(default_factory=dict)
    merged_json: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    retry_count: int = 0
    work_experience_validation_report: Optional[ValidationReport] = None
    education_validation_report: Optional[EducationValidationReport] = None
    contact_information_evidence: Optional[ContactInformationEvidenceRecord] = None
    work_experience_evidence: Optional[ModuleEvidenceRecord] = None
    education_evidence: Optional[ModuleEvidenceRecord] = None
    languages_evidence: Optional[LanguagesEvidenceRecord] = None
    professional_summary_evidence: Optional[ProfessionalSummaryEvidence] = None
    personal_information_evidence: Optional[PersonalInformationEvidenceRecord] = None


class ParserPipeline:
    """Wires together every modular parsing service into one full,
    markdown-to-JSON pipeline.

    Every dependency is constructor-injected — this class holds no
    module-level or global state, and each instance owns its own set of
    collaborators (or the defaults each one already builds from
    settings.py/config.py).
    """

    def __init__(
        self,
        personal_information_extractor: Optional[PersonalInformationExtractor] = None,
        contact_information_extractor: Optional[ContactInformationExtractor] = None,
        skills_extractor: Optional[SkillsExtractor] = None,
        professional_summary_extractor: Optional[ProfessionalSummaryExtractor] = None,
        languages_extractor: Optional[LanguagesExtractor] = None,
        work_experience_extractor: Optional[WorkExperienceExtractor] = None,
        work_experience_validator: Optional[WorkExperienceValidator] = None,
        education_extractor: Optional[EducationExtractor] = None,
        education_validator: Optional[EducationValidator] = None,
        project_extractor: Optional[ProjectExtractor] = None,
        parser_merger: Optional[ParserMerger] = None,
        schema_validator: Optional[SchemaValidator] = None,
        output_root: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            personal_information_extractor: Defaults to a new
                PersonalInformationExtractor().
            contact_information_extractor: Defaults to a new
                ContactInformationExtractor().
            skills_extractor: Defaults to a new SkillsExtractor().
            professional_summary_extractor: Defaults to a new
                ProfessionalSummaryExtractor().
            languages_extractor: Defaults to a new LanguagesExtractor().
            work_experience_extractor: Defaults to a new
                WorkExperienceExtractor() (Milestone 17.3 — the
                extractor itself was built in Milestone 15.2).
            work_experience_validator: Deterministic post-extraction
                validator run on work_experience only (Milestone
                17.6). Defaults to a new WorkExperienceValidator().
            education_extractor: Defaults to a new EducationExtractor()
                (Milestone 18.3 — the extractor itself was built in
                Milestone 18.2).
            education_validator: Deterministic post-extraction
                validator run on education only (Milestone 18.10).
                Defaults to a new EducationValidator().
            project_extractor: Defaults to a new ProjectExtractor()
                (Milestone T5/Phase 1B -- the extractor itself was
                built, schema-designed, and real-call-validated in
                T3/Phase 1 and T4/Phase 1A; this is its first wiring
                into the live pipeline). No dedicated per-module
                validator or evidence-builder exists for this module
                (matching the "skills"/"professional_summary"/
                "languages" modules' own established minimal-branch
                pattern, not the heavier work_experience/education
                pattern) -- T4's validation found no evidenced need for
                one, and this milestone's own explicit scope is the
                smallest additive integration the existing pipeline
                abstraction already supports.
            parser_merger: Combines every module's result into one
                resume dict. Defaults to a new ParserMerger().
            schema_validator: Validates the merged resume JSON. Defaults
                to a new SchemaValidator(), which validates against
                schemas/resume_schema_v1.json (the full resume schema —
                distinct from any single module's own schema).
            output_root: Remediation R-08. Directory under which
                modules/<resume>/<module>.json and json_final/<resume>.json
                are written. None (the default) means config.OUTPUT_DIR,
                resolved at save time, which is byte-for-byte the
                pre-R-08 behavior. Nothing else about the run changes
                with this value.
        """
        self.personal_information_extractor = (
            personal_information_extractor or PersonalInformationExtractor()
        )
        self.contact_information_extractor = (
            contact_information_extractor or ContactInformationExtractor()
        )
        self.skills_extractor = skills_extractor or SkillsExtractor()
        self.professional_summary_extractor = (
            professional_summary_extractor or ProfessionalSummaryExtractor()
        )
        self.languages_extractor = languages_extractor or LanguagesExtractor()
        self.work_experience_extractor = (
            work_experience_extractor or WorkExperienceExtractor()
        )
        self.work_experience_validator = (
            work_experience_validator or WorkExperienceValidator()
        )
        self.education_extractor = education_extractor or EducationExtractor()
        self.education_validator = education_validator or EducationValidator()
        self.project_extractor = project_extractor or ProjectExtractor()
        self.parser_merger = parser_merger or ParserMerger()
        self.schema_validator = schema_validator or SchemaValidator()
        self.output_root: Optional[Path] = Path(output_root) if output_root is not None else None

        # (module result key, extractor) pairs, in the pipeline's
        # required run order. "projects" (Milestone T5/Phase 1B) is
        # appended LAST, after every pre-existing module -- the
        # smallest-possible insertion point: it cannot alter any
        # existing module's own run order or introduce any new
        # cross-module dependency, and (like "skills") needs no special
        # per-module branch below (no validator, no evidence record --
        # see the constructor's own docstring for why).
        self._module_extractors: List[Tuple[str, Any]] = [
            ("personal_information", self.personal_information_extractor),
            ("contact_information", self.contact_information_extractor),
            ("skills", self.skills_extractor),
            ("professional_summary", self.professional_summary_extractor),
            ("languages", self.languages_extractor),
            ("work_experience", self.work_experience_extractor),
            ("education", self.education_extractor),
            ("projects", self.project_extractor),
        ]

    def run(
        self,
        markdown_path: Union[str, Path],
        pdf_path: Optional[Union[str, Path]] = None,
    ) -> ParserPipelineResult:
        """Run the full parser pipeline over one markdown file.

        Args:
            markdown_path: Path to the source markdown file, e.g.
                output/markdown/resume.md.
            pdf_path: Optional path to the resume's original source PDF
                (Milestone V3.9). Purely additive and opt-in — defaults
                to None, which is fully backward compatible with every
                existing caller: contact_information_evidence's PDF
                hyperlink-annotation fields simply stay unavailable
                (annotations_checked=None, never guessed or derived
                from markdown_path) exactly as build_contact_information
                _evidence() already documents. This pipeline never
                derives a PDF path on its own — no filename/stem
                guessing — since this pipeline has never read from
                input/resumes/ or known about PDFs at all (see this
                module's own top docstring), and guessing wrongly would
                silently attribute the wrong PDF's annotations to this
                resume.

        Returns:
            A ParserPipelineResult describing every stage's output.

        Raises:
            MarkdownNotFoundError: If `markdown_path` does not exist or
                is not a file (from services.base_module_extractor).
            PromptNotFoundError: If any module's prompt id has no
                matching prompt file (from services.prompt_manager).
            InvalidJSONError: If any module extractor's response is
                still not syntactically valid JSON after MAX_ATTEMPTS
                tries.
            SchemaValidationError: If any module's result still doesn't
                conform to its own schema after MAX_ATTEMPTS tries, or
                if the final merged resume JSON doesn't conform to
                resume_schema_v1.json (from validators.schema_validator
                — the merged-JSON check is not retried, since it
                validates already-fixed data).
            MergeConflictError: If two modules' results define the same
                top-level key (from services.parser_merger).
            ModelNotFoundError: If a required model is not installed in
                Ollama (from services.ollama_client).
            OllamaNotRunningError: If the Ollama server cannot be reached
                (from services.ollama_client).
            OllamaConnectionError: If a different connection-level error
                occurs while calling Ollama (from services.ollama_client).
        """
        start_time = time.perf_counter()
        markdown_path = Path(markdown_path)
        resume_name = markdown_path.stem

        logger.info("Running parser pipeline for %s", markdown_path)

        total_retry_count = 0

        # Milestone 23.7: JsonDraftGenerator has been retired — see this
        # module's own docstring. draft_json is kept, always None, only
        # for ParserPipelineResult's backward-compatible field shape.
        draft_json: Any = None

        module_results: Dict[str, Any] = {}
        work_experience_validation_report: Optional[ValidationReport] = None
        education_validation_report: Optional[EducationValidationReport] = None
        contact_information_evidence: Optional[ContactInformationEvidenceRecord] = None
        work_experience_evidence: Optional[ModuleEvidenceRecord] = None
        education_evidence: Optional[ModuleEvidenceRecord] = None
        languages_evidence: Optional[LanguagesEvidenceRecord] = None
        professional_summary_evidence: Optional[ProfessionalSummaryEvidence] = None
        personal_information_evidence: Optional[PersonalInformationEvidenceRecord] = None
        for module_key, extractor in self._module_extractors:
            module_result, module_retries = self._run_with_retry(
                extractor.__class__.__name__,
                lambda extractor=extractor: extractor.extract(markdown_path),
            )
            json_object = module_result.json_object
            total_retry_count += module_retries

            if module_key == "personal_information":
                # Milestone V5.0: purely observational, same non-
                # reassigning pattern the contact_information branch
                # below already established (Milestone V3.9) -- read
                # here rather than depending on any other branch's
                # markdown_text local, per this loop's own established
                # "don't depend on run order" convention.
                markdown_text = markdown_path.read_text(encoding="utf-8")
                personal_information_evidence = build_personal_information_evidence(
                    json_object.get("personal_information", {}),
                    markdown_text=markdown_text,
                )
            elif module_key == "contact_information":
                # Milestone V3.9: purely observational -- json_object is
                # never reassigned in this branch (unlike the
                # work_experience/education branches below), so there is
                # no code path through which this could alter the
                # extracted contact_information value. build_contact_
                # information_evidence() is itself read-only (Milestone
                # V3.8) and is called with the SAME markdown_text every
                # other markdown-aware step in this loop already reads.
                markdown_text = markdown_path.read_text(encoding="utf-8")
                contact_information_evidence = build_contact_information_evidence(
                    json_object.get("contact_information", {}),
                    markdown_text=markdown_text,
                    pdf_path=pdf_path,
                )
            elif module_key == "professional_summary":
                # Milestone V5.0: purely observational, same pattern as
                # the personal_information branch above. GT is never a
                # parameter to build_professional_summary_evidence() --
                # see that function's own docstring.
                markdown_text = markdown_path.read_text(encoding="utf-8")
                professional_summary_evidence = build_professional_summary_evidence(
                    json_object.get("professional_summary"),
                    markdown_text=markdown_text,
                )
            elif module_key == "languages":
                # Milestone V5.0: purely observational, same pattern as
                # the personal_information branch above.
                markdown_text = markdown_path.read_text(encoding="utf-8")
                languages_evidence = build_languages_evidence(
                    json_object.get("languages", []),
                    markdown_text=markdown_text,
                )
            elif module_key == "work_experience":
                # Read here, not at the top of run(): every earlier
                # extractor has already successfully read markdown_path
                # by this point, so this can't introduce a new "missing
                # markdown" failure path that didn't already exist.
                markdown_text = markdown_path.read_text(encoding="utf-8")
                json_object, work_experience_validation_report = self._validate_work_experience(
                    markdown_text, json_object
                )
                # Milestone V5.0: an additional, non-reassigning
                # statement appended AFTER the existing mutating branch
                # above -- build_work_experience_evidence() itself never
                # mutates its arguments (Milestone V2.7), and json_object
                # is not reassigned again here, so this cannot introduce
                # any further change to the already-validated value.
                # Uses the SAME post-validation job list and the SAME
                # ValidationReport already produced immediately above.
                work_experience_evidence = build_work_experience_evidence(
                    json_object.get("work_experience", []),
                    markdown_text,
                    work_experience_validation_report,
                )
            elif module_key == "education":
                # Milestone V2.6D: same read-here reasoning as the
                # work_experience branch above -- by this point in the
                # loop every earlier extractor (including
                # work_experience) has already successfully read
                # markdown_path, so this cannot introduce a new
                # "missing markdown" failure path. Read explicitly here
                # rather than relying on work_experience's own
                # markdown_text local surviving into this branch, so
                # this doesn't silently depend on work_experience
                # running earlier in _module_extractors' order.
                markdown_text = markdown_path.read_text(encoding="utf-8")
                json_object, education_validation_report = self._validate_education(
                    json_object, markdown_text
                )
                # Milestone V5.0: same additional, non-reassigning
                # statement pattern as the work_experience branch above.
                education_evidence = build_education_evidence(
                    json_object.get("education", []),
                    markdown_text,
                    education_validation_report,
                )

            module_results[module_key] = json_object
            self._save_module_json(resume_name, module_key, json_object)

        merged_json = self.parser_merger.merge(list(module_results.values()))
        self.schema_validator.validate(merged_json)

        final_path = self._save_final_json(resume_name, merged_json)
        processing_time = time.perf_counter() - start_time

        logger.info(
            "Parser pipeline complete for %s in %.2fs (%d retr(y/ies)) -> %s",
            markdown_path,
            processing_time,
            total_retry_count,
            final_path,
        )

        return ParserPipelineResult(
            draft_json=draft_json,
            module_results=module_results,
            merged_json=merged_json,
            processing_time=processing_time,
            retry_count=total_retry_count,
            work_experience_validation_report=work_experience_validation_report,
            education_validation_report=education_validation_report,
            contact_information_evidence=contact_information_evidence,
            work_experience_evidence=work_experience_evidence,
            education_evidence=education_evidence,
            languages_evidence=languages_evidence,
            professional_summary_evidence=professional_summary_evidence,
            personal_information_evidence=personal_information_evidence,
        )

    def _validate_work_experience(
        self, markdown_text: str, json_object: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], ValidationReport]:
        """Run WorkExperienceValidator (Milestone 17.6; markdown-aware
        API added in Milestone 17.7) on WorkExperienceExtractor's raw
        result.

        Args:
            markdown_text: The resume's full source markdown, read from
                markdown_path. Threaded through to the validator per
                Milestone 17.7's API — Phase 1's rules don't use it yet.
            json_object: The raw `{"work_experience": [...]}` dict
                WorkExperienceExtractor.extract() returned.

        Returns:
            A (validated_json_object, report) tuple. validated_json_object
            has the same `{"work_experience": [...]}` shape, with any
            auto-fixable issues corrected; report is WorkExperienceValidator's
            full ValidationReport for this resume.
        """
        jobs = json_object.get("work_experience", [])
        validated_jobs, report = self.work_experience_validator.validate(markdown_text, jobs)
        return {"work_experience": validated_jobs}, report

    def _validate_education(
        self, json_object: Dict[str, Any], markdown_text: str
    ) -> Tuple[Dict[str, Any], EducationValidationReport]:
        """Run EducationValidator (Milestone 18.10; markdown_text plumbed
        through as of Milestone V2.6D) on EducationExtractor's raw
        result.

        Args:
            json_object: The raw `{"education": [...]}` dict
                EducationExtractor.extract() returned.
            markdown_text: The resume's full source markdown (Milestone
                V2.6D). Phase 1's Quick Win rules (EDV-001/008/009/010/
                011) still need no source markdown and are unaffected;
                only EDV-012 (purely additive corroboration evidence,
                never a repair or rejection) consumes it.

        Returns:
            A (validated_json_object, report) tuple. validated_json_object
            has the same `{"education": [...]}` shape, with any
            auto-fixable issues corrected; report is EducationValidator's
            full ValidationReport for this resume.
        """
        entries = json_object.get("education", [])
        validated_entries, report = self.education_validator.validate(entries, markdown_text)
        return {"education": validated_entries}, report

    @staticmethod
    def _run_with_retry(step_name: str, func: Callable[[], _T]) -> Tuple[_T, int]:
        """Call `func`, retrying up to MAX_ATTEMPTS total attempts if it
        raises a RETRYABLE_EXCEPTIONS error — the same failure mode a
        fresh model call might not repeat.

        Args:
            step_name: Name to use in log messages (e.g. the calling
                extractor's class name).
            func: A zero-argument callable to invoke (typically a
                lambda wrapping `some_extractor.extract(markdown_path)`).

        Returns:
            A (result, retries_used) tuple. retries_used is 0 if `func`
            succeeded on the first attempt.

        Raises:
            Whatever `func` raised on its final attempt, if every
            attempt failed with a retryable exception — same exception
            type and original message, chained via `from exc`, but with
            the failing step's name prepended (e.g. "SkillsExtractor
            failed: ..."), so a caller retrying the whole resume (see
            pipeline/resume_retry_queue.py) can report which step
            caused it. Any non-retryable exception propagates
            immediately and unmodified, with no retry at all.
        """
        attempt = 1
        while True:
            try:
                return func(), attempt - 1
            except RETRYABLE_EXCEPTIONS as exc:
                if attempt >= MAX_ATTEMPTS:
                    logger.error(
                        "%s failed after %d attempt(s); giving up. Reason: %s",
                        step_name,
                        attempt,
                        type(exc).__name__,
                    )
                    raise type(exc)(f"{step_name} failed: {exc}") from exc
                attempt += 1
                logger.warning(
                    "Retrying %s — attempt %d/%d — reason: %s",
                    step_name,
                    attempt,
                    MAX_ATTEMPTS,
                    type(exc).__name__,
                )

    def _resolved_output_root(self) -> Path:
        """The directory module/final JSON is written under: the
        constructor's output_root if one was given, otherwise
        config.OUTPUT_DIR (looked up at call time, so the pre-R-08
        behavior -- including any test that patches this module's
        OUTPUT_DIR -- is unchanged)."""
        return self.output_root if self.output_root is not None else OUTPUT_DIR

    def _save_module_json(self, resume_name: str, module_key: str, json_object: Any) -> Path:
        """Save one module's raw result to
        <output_root>/modules/<resume_name>/<module_key>.json."""
        module_dir = self._resolved_output_root() / "modules" / resume_name
        module_dir.mkdir(parents=True, exist_ok=True)
        module_path = module_dir / f"{module_key}.json"
        module_path.write_text(json.dumps(json_object, indent=2), encoding="utf-8")
        return module_path

    def _save_final_json(self, resume_name: str, merged_json: Dict[str, Any]) -> Path:
        """Save the merged, validated resume JSON to
        <output_root>/json_final/<resume_name>.json."""
        final_dir = self._resolved_output_root() / "json_final"
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / f"{resume_name}.json"
        final_path.write_text(json.dumps(merged_json, indent=2), encoding="utf-8")
        return final_path
