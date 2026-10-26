"""
scripts/benchmark_suites.py

Milestone 22.4: Fast Benchmark Infrastructure.

Resolves a named benchmark suite (configs/benchmark_suites.yaml) into
the concrete mapping.csv rows it covers, by looking each suite member
up in benchmark/ground_truth/json/mapping.csv — never by hardcoding a
filename anywhere in this module or its callers. Shared by
scripts/run_parser_pipeline.py and scripts/run_parser_evaluation.py so
both tools resolve a suite name exactly the same way.

This module is pure I/O plumbing: it does not run the parser, does not
import ParserPipeline or ParserEvaluator, and does not decide what a
suite *means* for either script (which files to process, where to
write reports) — each caller decides that for itself. It also does not
modify pipeline/parser_pipeline.py, evaluation/parser_evaluator.py, or
any file this milestone's scope excludes.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from config import CONFIGS_DIR, GROUND_TRUTH_DIR

logger = logging.getLogger(__name__)

DEFAULT_SUITES_PATH = CONFIGS_DIR / "benchmark_suites.yaml"
MAPPING_CSV_FILENAME = "mapping.csv"

# The literal suite value (configs/benchmark_suites.yaml's "full: - '*'")
# meaning "every resume in mapping.csv, unfiltered."
FULL_SUITE_WILDCARD = "*"

# The conventional name of the suite equivalent to "no suite at all."
# Callers use this to decide whether a --suite flag should change
# behavior at all (see each script's own handling) — this module
# itself treats "full" as an ordinary suite name like any other and
# resolves it the same way (via the wildcard row above).
FULL_SUITE_NAME = "full"


class BenchmarkSuiteError(Exception):
    """Base exception for all benchmark-suite resolution failures."""


class SuiteFileNotFoundError(BenchmarkSuiteError):
    """Raised when the benchmark suites YAML file doesn't exist."""


class MappingFileNotFoundError(BenchmarkSuiteError):
    """Raised when ground_truth_dir has no mapping.csv to resolve against."""


class SuiteNotFoundError(BenchmarkSuiteError):
    """Raised when a requested suite name isn't defined in the YAML file."""


class SuiteResumeNotFoundError(BenchmarkSuiteError):
    """Raised when a suite lists a resume name mapping.csv has no row
    for — mapping.csv is the single source of truth for filenames, so
    this is treated as a configuration error, never a silent skip."""


def load_suites(suites_path: Path = DEFAULT_SUITES_PATH) -> Dict[str, List[str]]:
    """Load every suite defined in `suites_path`, as written.

    Args:
        suites_path: Path to the benchmark suites YAML file. Defaults
            to config.CONFIGS_DIR / "benchmark_suites.yaml".

    Returns:
        Mapping of suite name -> list of resume names (mapping.csv
        Markdown-column stems, or the single-element wildcard list
        `["*"]`), exactly as written in the YAML file. `{}` if the
        file is empty.

    Raises:
        SuiteFileNotFoundError: If `suites_path` doesn't exist.
    """
    if not suites_path.is_file():
        raise SuiteFileNotFoundError(f"Benchmark suites file not found: {suites_path}")
    with suites_path.open("r", encoding="utf-8") as suites_file:
        suites = yaml.safe_load(suites_file)
    return suites or {}


def resolve_suite(
    suite_name: str,
    suites_path: Path = DEFAULT_SUITES_PATH,
    ground_truth_dir: Path = GROUND_TRUTH_DIR,
) -> List[Dict[str, str]]:
    """Resolve `suite_name` into the mapping.csv rows it covers.

    Args:
        suite_name: A key in `suites_path` (e.g. "smoke", "regression",
            "full").
        suites_path: Path to the benchmark suites YAML file. Defaults
            to config.CONFIGS_DIR / "benchmark_suites.yaml".
        ground_truth_dir: Directory containing mapping.csv (the same
            directory evaluation.parser_evaluator.ParserEvaluator reads
            it from) — the only source of truth for resolving a suite
            member's name to a real (ResumeID, PDF, Markdown, JSON)
            row. No filename is ever hardcoded in this module.

    Returns:
        The mapping.csv rows (ResumeID/PDF/Markdown/JSON dicts) the
        suite covers, in mapping.csv's own row order — deduplicated if
        a suite happens to list the same resume twice. A suite whose
        value is the literal wildcard list `["*"]` (see
        FULL_SUITE_WILDCARD) resolves to every row in mapping.csv.

    Raises:
        SuiteFileNotFoundError: If `suites_path` doesn't exist.
        MappingFileNotFoundError: If `ground_truth_dir` has no
            mapping.csv.
        SuiteNotFoundError: If `suite_name` isn't a key in `suites_path`.
        SuiteResumeNotFoundError: If the suite lists a resume name with
            no matching row in mapping.csv.
    """
    suites = load_suites(suites_path)
    if suite_name not in suites:
        available = ", ".join(sorted(suites)) or "(none defined)"
        raise SuiteNotFoundError(
            f"Unknown benchmark suite '{suite_name}'. Defined suites: "
            f"{available} (see {suites_path})."
        )

    resume_names = suites[suite_name] or []
    all_rows = _load_mapping_rows(ground_truth_dir)

    if resume_names == [FULL_SUITE_WILDCARD]:
        return all_rows

    rows_by_stem = {Path(row["Markdown"]).stem: row for row in all_rows}
    resolved_rows = []
    seen_stems = set()
    for name in resume_names:
        if name not in rows_by_stem:
            mapping_path = ground_truth_dir / MAPPING_CSV_FILENAME
            raise SuiteResumeNotFoundError(
                f"Suite '{suite_name}' lists '{name}', which has no matching "
                f"row in {mapping_path}."
            )
        if name not in seen_stems:
            seen_stems.add(name)
            resolved_rows.append(rows_by_stem[name])
    return resolved_rows


def _load_mapping_rows(ground_truth_dir: Path) -> List[Dict[str, str]]:
    """Read every row of `ground_truth_dir`/mapping.csv, in order."""
    mapping_path = ground_truth_dir / MAPPING_CSV_FILENAME
    if not mapping_path.is_file():
        raise MappingFileNotFoundError(f"mapping.csv not found: {mapping_path}")
    with mapping_path.open(newline="", encoding="utf-8") as mapping_file:
        return list(csv.DictReader(mapping_file))


def is_full_suite(suite_name: Optional[str]) -> bool:
    """True if `suite_name` is None or names the "full" suite — the two
    cases every caller in this milestone treats as "identical to not
    filtering at all," per Requirement 5 (preserve existing behavior
    when no suite is specified)."""
    return suite_name is None or suite_name == FULL_SUITE_NAME
