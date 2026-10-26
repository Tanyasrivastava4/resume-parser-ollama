"""
services/project_extractor.py

Project Extractor (Milestone T3/Phase 1 -- schema foundation; wired
into the live pipeline in Milestone T5/Phase 1B; URL field-fidelity
fix added in Milestone T6/Phase 1C).

    output/markdown/<resume>.md
                |
                v
    prompts/modules/projects.md + markdown, in one prompt
                |
                v
    OllamaClient.generate_text (settings.DEFAULT_TEXT_MODEL, default "qwen3:1.7b")
                |
                v
    parsed {"projects": [...]}, validated against
    schemas/modules/projects_schema.json
                |
                v
    _ground_project_urls()  -- Milestone T6/Phase 1C, see below
                |
                v
    returned in memory

All pipeline mechanics (reading markdown, prompting, parsing, validating,
timing) live in services/base_module_extractor.py and are not duplicated
here.

Per this milestone's Part C relationship design (docs/MILESTONE_T3_PHASE1_SCHEMA_FOUNDATION.md):
this extractor's own `technologies` output must NEVER be automatically
copied into the canonical `skills` array by any assembler/merger logic.
No such copying exists anywhere in this codebase (confirmed by direct
reading, T3 Part A) -- services/parser_merger.py performs a pure key
union with no cross-module field derivation of any kind. This docstring
note exists so a future change does not silently introduce that
coupling without a fresh evidence-based decision.

Milestone T6/Phase 1C (docs/MILESTONE_T6_PHASE1C_PROJECT_URL_FIELD_FIDELITY.md):
T4/T5's own real validation found a real, confirmed field-level defect
on resume 016 -- the "Bewakoof.com Clone" project entry was returned
with `url: "https://github.com/Mukulkanojia123"`, which is the
candidate's own general GitHub PROFILE link from the resume's header
(appearing before "## EDUCATION", two full sections before "##
PROJECTS" even begins), not a link specific to that project. Direct
inspection of the real source confirmed "Bewakoof.com Clone"'s own
text span (from its own heading to the next section boundary) contains
NO url-shaped text at all -- the model fabricated an association
between two textually and structurally unconnected parts of the
document (unsupported inference / field extraction error, not a
segmentation or source-ambiguity problem -- the entity boundaries
themselves were already correct).

`_ground_project_urls()` is a narrow, deterministic, Project-specific
post-extraction correction: for each project entity with a non-null
`url`, it locates that SPECIFIC project's own source-text span (from
its own `name` occurrence in the markdown to the next project's `name`
occurrence, or the next markdown heading, or end of document) and
checks whether the returned `url` string appears literally within THAT
span. If not, `url` is corrected to `None` -- never guessed at, never
reassigned to a different value. If a project's own `name` cannot be
located in the source at all, that project's `url` is left untouched
(the check requires a located span to be safely applied at all; a
failed lookup is not treated as evidence of anything).

This intentionally does NOT use the shared `preprocess_json()` hook
(services/base_module_extractor.py) -- that hook's existing signature
(`preprocess_json(self, json_object)`) has no access to the source
markdown text at all, and broadening it would change every other
module extractor's own hook contract (WorkExperienceExtractor,
EducationExtractor), a materially broader change than this milestone's
own explicit "Project-specific, no broad refactor" scope. Overriding
`extract()` itself -- calling `super().extract()` for the complete,
already-tested, already-validated base behavior, then applying this
one additional, narrowly-scoped, read-only-of-markdown correction --
is the smallest change that stays entirely local to this file.
Correcting a string url to `None` can only make the (already-passed)
schema validation MORE permissive (`url` is `["string", "null"]`), so
this never risks a new, post-hoc validation failure.
"""

import re
from pathlib import Path
from typing import Any, List, Optional, Union

from config import SCHEMAS_DIR
from settings import DEFAULT_TEXT_MODEL
from services.base_module_extractor import BaseModuleExtractor, BaseModuleResult

ProjectResult = BaseModuleResult

__all__ = ["ProjectExtractor", "ProjectResult"]

DEFAULT_PROMPT_ID = "modules/projects"
DEFAULT_SCHEMA_PATH = SCHEMAS_DIR / "modules" / "projects_schema.json"

# A bare markdown heading line (1-6 '#' characters followed by
# whitespace), used only as a fallback span terminator for the LAST
# project in a list (where there is no "next project name" to bound
# the span). Deliberately simple and self-contained -- this module
# does not import the research-layer boundary-detection scripts under
# scripts/ (per this project's own established discipline that
# services/ never imports research-layer code).
_HEADING_LINE_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)


def _project_span(markdown_text: str, projects: List[dict], index: int) -> Optional[str]:
    """Returns the source-text span this specific project entity (at
    `index` in `projects`) plausibly occupies, or None if its own
    `name` cannot be located in `markdown_text` at all (in which case
    no correction should be attempted for it -- a failed lookup is not
    evidence of anything).

    Span start: the position of this project's own `name` text.
    Span end: the position of the NEXT project's own `name` text (if
    one exists and is found after this one), else the next markdown
    heading line after this project's name, else the end of the
    document.
    """
    name = projects[index].get("name")
    if not name or not isinstance(name, str):
        return None
    start = markdown_text.find(name)
    if start == -1:
        return None
    search_from = start + len(name)

    end = len(markdown_text)
    if index + 1 < len(projects):
        next_name = projects[index + 1].get("name")
        if next_name and isinstance(next_name, str):
            next_pos = markdown_text.find(next_name, search_from)
            if next_pos != -1:
                end = next_pos
    if end == len(markdown_text):
        heading_match = _HEADING_LINE_RE.search(markdown_text, search_from)
        if heading_match:
            end = heading_match.start()

    return markdown_text[start:end]


def _ground_project_urls(json_object: Any, markdown_text: str) -> Any:
    """Milestone T6/Phase 1C: returns `json_object` with any project's
    `url` corrected to `None` if that url string is not literally
    present within that SAME project's own source-text span (see
    `_project_span`). Never invents or reassigns a url to a different
    value -- only ever null-corrects an ungrounded one. Never mutates
    `json_object` or any project dict in place; returns a new
    structure only if at least one correction was needed.

    Defensive by design: if `json_object` isn't the expected shape,
    it is returned completely unchanged -- diagnosing a malformed
    shape is SchemaValidator.validate()'s job (already run, by this
    point, inside the base class's own extract()), not this
    function's.
    """
    if not isinstance(json_object, dict):
        return json_object
    projects = json_object.get("projects")
    if not isinstance(projects, list):
        return json_object

    new_projects: List[Any] = []
    changed = False
    for i, project in enumerate(projects):
        if not isinstance(project, dict):
            new_projects.append(project)
            continue
        url = project.get("url")
        if not url or not isinstance(url, str):
            new_projects.append(project)
            continue

        span = _project_span(markdown_text, projects, i)
        if span is None:
            # This project's own name couldn't be located -- do not
            # attempt a correction without a reliable span.
            new_projects.append(project)
            continue

        if url in span:
            new_projects.append(project)
            continue

        # Ungrounded within this project's own span -- correct to
        # None, never to a different (guessed) value.
        new_projects.append({**project, "url": None})
        changed = True

    if not changed:
        return json_object
    return {**json_object, "projects": new_projects}


class ProjectExtractor(BaseModuleExtractor):
    """Extracts independent projects from a resume's markdown.

    All pipeline mechanics come from BaseModuleExtractor. This class
    answers the three questions the base class needs (which prompt,
    which schema, which model) via the standard hooks, and additionally
    overrides `extract()` itself (Milestone T6/Phase 1C) to apply the
    narrow, deterministic, Project-specific URL-grounding correction
    described in this module's own docstring -- see that docstring for
    why `preprocess_json()` could not be used instead.
    """

    def _get_prompt_id(self) -> str:
        return DEFAULT_PROMPT_ID

    def _get_schema_path(self) -> Path:
        return DEFAULT_SCHEMA_PATH

    def _get_model_name(self) -> str:
        return DEFAULT_TEXT_MODEL

    def extract(
        self,
        markdown_path: Union[str, Path],
        model_name: Optional[str] = None,
        prompt_id: Optional[str] = None,
    ) -> BaseModuleResult:
        """Exactly BaseModuleExtractor.extract()'s own contract and
        behavior (same arguments, same return type, same exceptions),
        plus one additional, narrowly-scoped correction (Milestone
        T6/Phase 1C): any project `url` not literally present within
        that specific project's own source-text span is corrected to
        `None`. See this module's own docstring for the full rationale
        and why this could not be implemented via the shared
        `preprocess_json()` hook instead.
        """
        result = super().extract(markdown_path, model_name=model_name, prompt_id=prompt_id)
        markdown_text = Path(markdown_path).read_text(encoding="utf-8")
        corrected_json_object = _ground_project_urls(result.json_object, markdown_text)
        if corrected_json_object is result.json_object:
            return result
        return BaseModuleResult(json_object=corrected_json_object, processing_time=result.processing_time)
