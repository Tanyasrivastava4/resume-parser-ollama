"""
config.py

Static, non-secret configuration for the Resume AI Benchmark project.

This module holds values that are the same for every developer and every
environment: folder locations, pipeline identifiers, and the shared output
schema name. It should never contain secrets, API keys, or anything that
differs between machines — that belongs in settings.py instead.

Nothing in this file executes parsing logic. It only defines constants that
other modules (parsers, benchmark, validators) will import later.
"""

from pathlib import Path

# Project root, resolved relative to this file so it works regardless of
# the current working directory the app is launched from.
BASE_DIR = Path(__file__).resolve().parent

# Core folder locations, used by parsers/benchmark/validators to read and
# write files consistently instead of hardcoding paths in multiple places.
INPUT_DIR = BASE_DIR / "input"
RESUMES_INPUT_DIR = INPUT_DIR / "resumes"
OUTPUT_DIR = BASE_DIR / "output"
BENCHMARK_DIR = BASE_DIR / "benchmark"
COMPARISON_DIR = BENCHMARK_DIR / "comparison"
GROUND_TRUTH_DIR = BENCHMARK_DIR / "ground_truth" / "json"

# Remediation R-07 (2026-10-17): the human-adjudicated ground truth produced
# by Milestone 23.66 and used by the canonical frozen baseline (Milestone
# 23.67). It lives beside the legacy GT above and has its own mapping.csv
# (22 rows -- resume012, a job description, is permanently excluded).
# GROUND_TRUTH_DIR deliberately still points at the legacy set so that no
# historical command or report silently changes meaning; callers opt into
# the adjudicated set explicitly (scripts/run_parser_evaluation.py
# --ground-truth adjudicated_v1).
ADJUDICATED_GROUND_TRUTH_DIR = BENCHMARK_DIR / "ground_truth" / "adjudicated_v1"

# Named ground-truth selections accepted by --ground-truth. "legacy" is the
# pre-R-07 default and MUST stay first / the default.
GROUND_TRUTH_DIRS = {
    "legacy": GROUND_TRUTH_DIR,
    "adjudicated_v1": ADJUDICATED_GROUND_TRUTH_DIR,
}
DEFAULT_GROUND_TRUTH_NAME = "legacy"
MARKDOWN_DIR = OUTPUT_DIR / "markdown"

# Manually verified reference markdown (Milestone 15A.3), mirroring
# GROUND_TRUTH_DIR's json/ layout — used only by benchmark/markdown_benchmark.py
# to score Stage 1 (PDF -> Markdown) accuracy, independently of JSON accuracy.
MARKDOWN_GROUND_TRUTH_DIR = BENCHMARK_DIR / "ground_truth" / "markdown"

# Where MarkdownBenchmark writes its own benchmark_report.md (Milestone 15A.3).
MARKDOWN_BENCHMARK_OUTPUT_DIR = OUTPUT_DIR / "benchmark" / "markdown"
PROMPTS_DIR = BASE_DIR / "prompts"
SCHEMAS_DIR = BASE_DIR / "schemas"
VALIDATORS_DIR = BASE_DIR / "validators"
SCRIPTS_DIR = BASE_DIR / "scripts"
LOGS_DIR = BASE_DIR / "logs"
TESTS_DIR = BASE_DIR / "tests"
PARSERS_DIR = BASE_DIR / "parsers"

# Milestone 22.4: Fast Benchmark Infrastructure — holds
# benchmark_suites.yaml, the developer-tooling config that names named
# resume subsets (smoke/regression/full) for --suite.
CONFIGS_DIR = BASE_DIR / "configs"

# Identifiers for the three benchmarked pipelines, as named in
# PROJECT_PLAN.md. Kept as constants so pipeline names are typed once and
# reused consistently across parsers, benchmark, and logs.
PIPELINE_A = "pipeline_a_qwen_vl_direct_json"
PIPELINE_B = "pipeline_b_qwen_vl_markdown_json"
PIPELINE_C = "pipeline_c_docling_markdown_qwen_json"

PIPELINES = (PIPELINE_A, PIPELINE_B, PIPELINE_C)

# Every pipeline must produce output conforming to this single shared JSON
# schema, so benchmarking can compare them apples-to-apples. The actual
# schema definition will live under validators/ once implemented.
RESUME_JSON_SCHEMA_NAME = "resume_schema_v1"
