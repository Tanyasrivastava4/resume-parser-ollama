"""
tests/conftest.py

Shared pytest hooks for this repository.

`local_data` marker (added 2026-10-17, remediation item R-02)
--------------------------------------------------------------
A number of tests are forensic checks against this project's REAL run
artifacts -- the benchmark corpus's Stage-1 markdown under
`output/markdown/`, per-module parser output under `output/modules/`,
and the V2.4H routed-Stage-1 markdown under
`output/v2_4h_benchmark/stage1/markdown/`. Those directories are
deliberately git-ignored (they are generated outputs of local runs on
real candidate resumes), so a fresh clone or a CI runner does not have
them, and those tests used to FAIL there with FileNotFoundError (35
failures across 10 files on a clean checkout of afb68e6).

A test or class marked `@pytest.mark.local_data(<sentinel>, ...)` is
SKIPPED, with an explicit reason, when the artifacts it needs are not
present in this checkout. When they are present (the development
machine), the marker is inert and the test runs exactly as before.
Nothing about what the tests assert has changed.

Sentinels (each maps to a concrete, cheap presence check below):

    corpus_markdown   every Markdown file named in
                      benchmark/ground_truth/json/mapping.csv exists
                      under output/markdown/
    unseen_markdown   the 5 T19 unseen-resume markdown files exist under
                      output/markdown/
    v2_4h_markdown    every Markdown file named in
                      benchmark/ground_truth/adjudicated_v1/mapping.csv
                      (the 22-resume population V2.4H actually ran;
                      resume012 was never part of it) exists under
                      output/v2_4h_benchmark/stage1/markdown/
    module_outputs    output/modules/<stem>/work_experience.json exists
                      for every mapping.csv resume
    any_corpus_markdown
                      at least ONE *.md file exists under output/markdown/.
                      Use this -- not corpus_markdown -- for tests whose
                      assertion IS "every mapped file exists": with the
                      stricter sentinel such a test could only ever skip,
                      never fail, when a mapped file went missing on the
                      development machine (independent review finding,
                      2026-10-17). With this sentinel it skips only on a
                      checkout that has no corpus markdown at all.
"""
import csv
from pathlib import Path

import pytest

from config import BASE_DIR, GROUND_TRUTH_DIR, MARKDOWN_DIR, OUTPUT_DIR

_V2_4H_MARKDOWN_DIR = OUTPUT_DIR / "v2_4h_benchmark" / "stage1" / "markdown"
_MODULES_DIR = OUTPUT_DIR / "modules"

# The 5 unseen resumes frozen by Milestone T19 (scripts/_milestone_t19_stage1_prep.py).
_UNSEEN_STEMS = (
    "178480991686_Vachan_Premugh_Dicruz_Founder_Office_BTTR.MEN",
    "8th_April_Akasa_Kapoor_Resume",
    "AARTI BAGHOTIYA_Economic Urban Planner_Intileo",
    "AASHISH PARMAR - Email Marketing Executive - Intileo",
    "_Adarsh_Bhagat Resume",
)


_ADJUDICATED_GROUND_TRUTH_DIR = GROUND_TRUTH_DIR.parent / "adjudicated_v1"


def _mapping_markdown_names(ground_truth_dir=GROUND_TRUTH_DIR):
    mapping = ground_truth_dir / "mapping.csv"
    if not mapping.is_file():
        return []
    with mapping.open(newline="", encoding="utf-8") as f:
        return [row["Markdown"] for row in csv.DictReader(f)]


def _missing_for(sentinel):
    """Return the list of required paths that are absent for `sentinel`
    (empty list == artifacts present)."""
    if sentinel == "corpus_markdown":
        required = [MARKDOWN_DIR / name for name in _mapping_markdown_names()]
    elif sentinel == "unseen_markdown":
        required = [MARKDOWN_DIR / f"{stem}.md" for stem in _UNSEEN_STEMS]
    elif sentinel == "v2_4h_markdown":
        required = [
            _V2_4H_MARKDOWN_DIR / name
            for name in _mapping_markdown_names(_ADJUDICATED_GROUND_TRUTH_DIR)
        ]
    elif sentinel == "any_corpus_markdown":
        if MARKDOWN_DIR.is_dir() and any(MARKDOWN_DIR.glob("*.md")):
            return []
        return [str(MARKDOWN_DIR.relative_to(BASE_DIR)) + "/*.md (none present)"]
    elif sentinel == "module_outputs":
        required = [
            _MODULES_DIR / Path(name).stem / "work_experience.json"
            for name in _mapping_markdown_names()
        ]
    else:
        raise pytest.UsageError(f"unknown local_data sentinel: {sentinel!r}")
    if not required:
        return [str(sentinel) + " (mapping.csv unreadable)"]
    return [str(p.relative_to(BASE_DIR)) for p in required if not p.is_file()]


def pytest_runtest_setup(item):
    for marker in item.iter_markers(name="local_data"):
        sentinels = marker.args or ("corpus_markdown",)
        missing = []
        for sentinel in sentinels:
            missing.extend(_missing_for(sentinel))
        if missing:
            shown = ", ".join(missing[:3]) + (" ..." if len(missing) > 3 else "")
            pytest.skip(
                "requires local, git-ignored run artifacts not present in this "
                f"checkout ({len(missing)} missing, e.g. {shown}); see tests/conftest.py"
            )
