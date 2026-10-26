"""
tests/test_run_markdown_benchmark.py

Tests for scripts/run_markdown_benchmark.py (Milestone 15A.3).

MarkdownBenchmark is always mocked — no test here invokes the script's
real, argument-free defaults, since those point at the project's real
benchmark/ground_truth/markdown/ and output/markdown/. This exercises
only the script's own CLI/orchestration logic (calling run(), printing
the result), not MarkdownBenchmark itself.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from benchmark.markdown_benchmark import MarkdownBenchmark, MarkdownBenchmarkResult, MarkdownComparisonResult
from scripts.run_markdown_benchmark import (
    _parse_args,
    print_benchmark_result,
    run_markdown_benchmark_cli,
)

SAMPLE_COMPARISON = MarkdownComparisonResult(
    resume_name="alice",
    reference_path="ref/alice.md",
    generated_path="gen/alice.md",
    character_accuracy=0.95,
    word_accuracy=0.90,
    line_accuracy=0.85,
    heading_accuracy=1.0,
    missing_lines=1,
    additional_lines=0,
    total_differences=1,
)

SAMPLE_RESULT = MarkdownBenchmarkResult(
    total_resumes=2,
    successful_resumes=1,
    failed_resumes=1,
    overall_character_accuracy=0.95,
    overall_word_accuracy=0.90,
    overall_line_accuracy=0.85,
    overall_heading_accuracy=1.0,
    total_missing_lines=1,
    total_additional_lines=0,
    total_differences=1,
    processing_time=0.5,
    comparisons={"alice": SAMPLE_COMPARISON},
    failures={"bob": "Generated markdown file not found: gen/bob.md"},
)


@pytest.fixture
def mock_benchmark():
    benchmark = MagicMock(spec=MarkdownBenchmark)
    benchmark.run.return_value = SAMPLE_RESULT
    benchmark.output_dir = Path("/fake/output/benchmark/markdown")
    return benchmark


class TestRunMarkdownBenchmarkCli:
    def test_calls_run_and_returns_result(self, mock_benchmark):
        result = run_markdown_benchmark_cli(benchmark=mock_benchmark)

        mock_benchmark.run.assert_called_once()
        assert result is SAMPLE_RESULT

    def test_prints_summary(self, mock_benchmark, capsys):
        run_markdown_benchmark_cli(benchmark=mock_benchmark)

        captured = capsys.readouterr()
        assert "Total resumes:    2" in captured.out
        assert "Successful:       1" in captured.out
        assert "Failed:           1" in captured.out

    def test_prints_overall_accuracy(self, mock_benchmark, capsys):
        run_markdown_benchmark_cli(benchmark=mock_benchmark)

        captured = capsys.readouterr()
        assert "Overall accuracy" in captured.out
        assert "Character: 95.00%" in captured.out
        assert "Word:      90.00%" in captured.out
        assert "Line:      85.00%" in captured.out
        assert "Heading:   100.00%" in captured.out

    def test_prints_line_differences(self, mock_benchmark, capsys):
        run_markdown_benchmark_cli(benchmark=mock_benchmark)

        captured = capsys.readouterr()
        assert "Missing lines:    1" in captured.out
        assert "Additional lines: 0" in captured.out
        assert "Total differences: 1" in captured.out

    def test_prints_per_resume_results(self, mock_benchmark, capsys):
        run_markdown_benchmark_cli(benchmark=mock_benchmark)

        captured = capsys.readouterr()
        assert "Per resume results" in captured.out
        assert "alice" in captured.out

    def test_prints_failures_when_present(self, mock_benchmark, capsys):
        run_markdown_benchmark_cli(benchmark=mock_benchmark)

        captured = capsys.readouterr()
        assert "Failures" in captured.out
        assert "bob" in captured.out

    def test_omits_failures_section_when_there_are_none(self, mock_benchmark, capsys):
        mock_benchmark.run.return_value = MarkdownBenchmarkResult(
            total_resumes=1,
            successful_resumes=1,
            failed_resumes=0,
            overall_character_accuracy=1.0,
            overall_word_accuracy=1.0,
            overall_line_accuracy=1.0,
            overall_heading_accuracy=1.0,
            total_missing_lines=0,
            total_additional_lines=0,
            total_differences=0,
        )

        run_markdown_benchmark_cli(benchmark=mock_benchmark)

        captured = capsys.readouterr()
        assert "Failures" not in captured.out

    def test_prints_report_location(self, mock_benchmark, capsys):
        run_markdown_benchmark_cli(benchmark=mock_benchmark)

        captured = capsys.readouterr()
        assert "Report saved to: /fake/output/benchmark/markdown" in captured.out


class TestPrintBenchmarkResult:
    def test_output_contains_all_expected_sections(self, capsys):
        print_benchmark_result(SAMPLE_RESULT, Path("/fake/output/benchmark/markdown"))

        captured = capsys.readouterr()
        for heading in (
            "Markdown benchmark summary",
            "Overall accuracy",
            "Line differences",
            "Per resume results",
            "Failures",
        ):
            assert heading in captured.out


class TestParseArgs:
    def test_no_arguments_is_valid(self):
        args = _parse_args([])
        assert vars(args) == {}

    def test_help_exits_cleanly(self):
        with pytest.raises(SystemExit) as exc_info:
            _parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_rejects_unexpected_argument(self):
        with pytest.raises(SystemExit) as exc_info:
            _parse_args(["--bogus"])
        assert exc_info.value.code != 0
