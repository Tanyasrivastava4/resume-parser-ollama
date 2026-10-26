"""
tests/test_project_url_field_fidelity.py

Milestone T6/Phase 1C -- adversarial and regression tests for
services/project_extractor.py's `_ground_project_urls` /
`_project_span` URL field-fidelity correction.

Real-corpus regression case (#1) uses the actual, unmodified source
markdown for resume 016 (output/markdown/Mukul Kanojia_Full Stack Developer_Intileo.md)
and the real T4/Phase 1A extraction result for that resume, verbatim.
Every other case (#2-#8) is explicitly SYNTHETIC/ADVERSARIAL, built to
test a specific structural scenario the repository's own real corpus
does not happen to otherwise exercise -- none of these synthetic
markdown snippets are presented as real resume content.

ZERO LLM calls -- `_ground_project_urls`/`_project_span` are pure,
deterministic functions operating on already-produced JSON + real or
synthetic markdown text.
"""
from __future__ import annotations

import pytest

from pathlib import Path

from services.project_extractor import _ground_project_urls, _project_span

_RESUME_016_PATH = Path("output/markdown/Mukul Kanojia_Full Stack Developer_Intileo.md")


def _project(name, url, **overrides):
    base = {
        "name": name, "description": None, "role": None, "start_date": None,
        "end_date": None, "highlights": [], "technologies": [], "url": url,
    }
    base.update(overrides)
    return base


@pytest.mark.local_data('corpus_markdown')
class TestRealResume016Regression:
    """#1 -- the exact real historical failure, using the real,
    unmodified source markdown and the real T4 extraction result."""

    def test_bewakoof_url_is_corrected_to_none(self):
        markdown_text = _RESUME_016_PATH.read_text(encoding="utf-8")
        json_object = {
            "projects": [
                _project("Senraa (https://senraa.com/)", "https://senraa.com/"),
                _project("Bewakoof.com Clone", "https://github.com/Mukulkanojia123"),
            ]
        }
        result = _ground_project_urls(json_object, markdown_text)
        projects = result["projects"]
        assert projects[0]["url"] == "https://senraa.com/"  # legitimate, self-contained -- preserved
        assert projects[1]["url"] is None  # misattributed profile link -- corrected

    def test_real_source_confirms_bewakoof_span_has_no_url(self):
        # Direct confirmation of the root-cause evidence: "Bewakoof.com
        # Clone"'s own span genuinely contains no url-shaped text.
        markdown_text = _RESUME_016_PATH.read_text(encoding="utf-8")
        projects = [
            {"name": "Senraa (https://senraa.com/)"},
            {"name": "Bewakoof.com Clone"},
        ]
        span = _project_span(markdown_text, projects, 1)
        assert "http" not in span
        assert "Mukulkanojia123" not in span


class TestSingleExplicitUrl:
    """#2 -- SYNTHETIC. A project with one explicit, self-contained URL."""

    def test_url_within_own_span_is_preserved(self):
        markdown_text = (
            "## PROJECTS\n"
            "### Widget App (https://widgetapp.example.com)\n"
            "- Built a widget management tool.\n"
        )
        json_object = {"projects": [_project("Widget App (https://widgetapp.example.com)", "https://widgetapp.example.com")]}
        result = _ground_project_urls(json_object, markdown_text)
        assert result["projects"][0]["url"] == "https://widgetapp.example.com"


class TestMultipleProjectsDifferentUrls:
    """#3 -- SYNTHETIC. Multiple projects, each with its own distinct,
    correctly-associated URL -- confirms the fix does not cross-
    contaminate between adjacent, both-legitimate entries."""

    def test_each_project_keeps_its_own_url(self):
        markdown_text = (
            "## PROJECTS\n"
            "### Alpha (https://alpha.example.com)\n"
            "- First project.\n"
            "### Beta (https://beta.example.com)\n"
            "- Second project.\n"
            "### Gamma (https://gamma.example.com)\n"
            "- Third project.\n"
        )
        json_object = {"projects": [
            _project("Alpha (https://alpha.example.com)", "https://alpha.example.com"),
            _project("Beta (https://beta.example.com)", "https://beta.example.com"),
            _project("Gamma (https://gamma.example.com)", "https://gamma.example.com"),
        ]}
        result = _ground_project_urls(json_object, markdown_text)
        urls = [p["url"] for p in result["projects"]]
        assert urls == ["https://alpha.example.com", "https://beta.example.com", "https://gamma.example.com"]


class TestProjectWithNoUrl:
    """#4 -- SYNTHETIC. A project with url already None -- must remain
    untouched (nothing to correct, no span lookup needed to matter)."""

    def test_none_url_passes_through_unchanged(self):
        markdown_text = "## PROJECTS\n### Solo Project\n- No link mentioned.\n"
        json_object = {"projects": [_project("Solo Project", None)]}
        result = _ground_project_urls(json_object, markdown_text)
        assert result is json_object  # unchanged object identity -- no correction was needed
        assert result["projects"][0]["url"] is None


class TestUnrelatedGithubUrlsElsewhere:
    """#5 -- SYNTHETIC, directly modeled on the real resume-016
    mechanism. A resume-header GitHub URL, structurally distant from a
    project with no URL of its own, must NOT be adopted."""

    def test_header_github_url_not_adopted_by_url_less_project(self):
        markdown_text = (
            "# Jane Doe\n"
            "[github.com/janedoe](https://github.com/janedoe)\n\n"
            "## EDUCATION\n- Some University\n\n"
            "## PROJECTS\n"
            "### Inventory Tracker\n"
            "- Built a stock tracking dashboard.\n"
        )
        json_object = {"projects": [_project("Inventory Tracker", "https://github.com/janedoe")]}
        result = _ground_project_urls(json_object, markdown_text)
        assert result["projects"][0]["url"] is None


class TestUrlBelongingToWorkExperience:
    """#6 -- SYNTHETIC. A URL that genuinely belongs to a Work
    Experience entry (e.g. an employer's site) must not be treated as
    grounded for an adjacent, URL-less Project."""

    def test_work_experience_url_not_adopted_by_project(self):
        markdown_text = (
            "## PROJECTS\n"
            "### Portfolio Site\n"
            "- Personal portfolio, no public link listed.\n\n"
            "## EXPERIENCE\n"
            "### Acme Corp (https://acmecorp.example.com)\n"
            "- Backend engineer.\n"
        )
        json_object = {"projects": [_project("Portfolio Site", "https://acmecorp.example.com")]}
        result = _ground_project_urls(json_object, markdown_text)
        assert result["projects"][0]["url"] is None


class TestUrlBelongingToEducation:
    """#7 -- SYNTHETIC. A URL belonging to an Education entry (e.g. a
    university's own site) must not be treated as grounded for an
    adjacent, URL-less Project."""

    def test_education_url_not_adopted_by_project(self):
        markdown_text = (
            "## EDUCATION\n"
            "### State University (https://stateuniversity.example.edu)\n"
            "- B.Sc. Computer Science\n\n"
            "## PROJECTS\n"
            "### Capstone Tool\n"
            "- Final-year capstone project, no link listed.\n"
        )
        json_object = {"projects": [_project("Capstone Tool", "https://stateuniversity.example.edu")]}
        result = _ground_project_urls(json_object, markdown_text)
        assert result["projects"][0]["url"] is None


class TestAdjacentProjectsMisleadingProximity:
    """#8 -- SYNTHETIC. Two adjacent projects where the FIRST has a
    real URL and the SECOND (immediately following, no blank-line
    separation beyond the heading) does not -- proximity alone must
    not cause the first project's URL to leak into the second's span."""

    def test_first_projects_url_does_not_leak_into_second(self):
        markdown_text = (
            "## PROJECTS\n"
            "### Real Deal (https://realdeal.example.com)\n"
            "- Has its own real link.\n"
            "### No Deal\n"
            "- Immediately adjacent, no link of its own.\n"
        )
        json_object = {"projects": [
            _project("Real Deal (https://realdeal.example.com)", "https://realdeal.example.com"),
            _project("No Deal", "https://realdeal.example.com"),  # hypothetical leaked/misattributed value
        ]}
        result = _ground_project_urls(json_object, markdown_text)
        assert result["projects"][0]["url"] == "https://realdeal.example.com"
        assert result["projects"][1]["url"] is None


class TestProjectNameNotLocatable:
    """SYNTHETIC, safety-boundary case: if a project's own `name`
    cannot be found in the source at all, no correction is attempted
    (never guess without a located span)."""

    def test_unlocatable_name_leaves_url_untouched(self):
        markdown_text = "## PROJECTS\n### Something Else Entirely\n- Text.\n"
        json_object = {"projects": [_project("A Name Not In The Source", "https://example.com/whatever")]}
        result = _ground_project_urls(json_object, markdown_text)
        # Conservative default: left as-is, not corrected, not guessed.
        assert result["projects"][0]["url"] == "https://example.com/whatever"
