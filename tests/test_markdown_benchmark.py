"""
tests/test_markdown_benchmark.py

Tests for benchmark/markdown_benchmark.py (Milestone 15A.3).

Every test here is pure text comparison or filesystem I/O against
tmp_path — no Ollama calls, no network access, and no randomness.
MarkdownComparator and MarkdownBenchmark are both fully deterministic
(difflib only), so tests assert exact values rather than ranges or
"close enough" approximations wherever the expected value is knowable.

MarkdownBenchmark never touches the project's real benchmark/ or
output/ directories — every test builds its own temporary reference_dir/
generated_dir/output_dir.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from benchmark.markdown_benchmark import (
    MarkdownBenchmark,
    MarkdownBenchmarkResult,
    MarkdownComparator,
    MarkdownComparisonResult,
    ReferenceMarkdownDirectoryNotFoundError,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def dataset(tmp_path):
    """Empty reference_dir/generated_dir pair; each test populates the
    specific files it needs."""
    reference_dir = tmp_path / "reference"
    generated_dir = tmp_path / "generated"
    reference_dir.mkdir()
    generated_dir.mkdir()
    return SimpleNamespace(reference_dir=reference_dir, generated_dir=generated_dir)


# ---------------------------------------------------------------------------
# MarkdownComparator — pure comparison logic, no filesystem involved.
# ---------------------------------------------------------------------------


class TestMarkdownComparatorIdentical:
    def test_identical_text_scores_perfect_across_all_metrics(self):
        comparator = MarkdownComparator()
        text = "# Heading\n\nSome content here.\nAnother line.\n"

        result = comparator.compare("resume", text, text, "ref.md", "gen.md")

        assert result.character_accuracy == 1.0
        assert result.word_accuracy == 1.0
        assert result.line_accuracy == 1.0
        assert result.heading_accuracy == 1.0
        assert result.missing_lines == 0
        assert result.additional_lines == 0
        assert result.total_differences == 0

    def test_both_empty_scores_perfect(self):
        comparator = MarkdownComparator()

        result = comparator.compare("resume", "", "", "ref.md", "gen.md")

        assert result.character_accuracy == 1.0
        assert result.word_accuracy == 1.0
        assert result.line_accuracy == 1.0
        assert result.heading_accuracy == 1.0
        assert result.missing_lines == 0
        assert result.additional_lines == 0
        assert result.total_differences == 0

    def test_result_records_the_given_paths_and_resume_name(self):
        comparator = MarkdownComparator()

        result = comparator.compare("alice", "text", "text", "ref/alice.md", "gen/alice.md")

        assert result.resume_name == "alice"
        assert result.reference_path == "ref/alice.md"
        assert result.generated_path == "gen/alice.md"


class TestMarkdownComparatorCharacterAccuracy:
    def test_single_character_lost_reduces_character_accuracy(self):
        comparator = MarkdownComparator()

        result = comparator.compare(
            "resume", "8799791143", "879979143", "ref.md", "gen.md"
        )

        assert 0.0 < result.character_accuracy < 1.0

    def test_completely_different_text_has_zero_character_accuracy(self):
        comparator = MarkdownComparator()

        result = comparator.compare("resume", "aaaaaaaaaa", "zzzzzzzzzz", "ref.md", "gen.md")

        assert result.character_accuracy == 0.0

    def test_empty_reference_nonempty_generated_is_zero_accuracy(self):
        comparator = MarkdownComparator()

        result = comparator.compare("resume", "", "some text", "ref.md", "gen.md")

        assert result.character_accuracy == 0.0


class TestMarkdownComparatorWordAccuracy:
    def test_single_word_change_reduces_word_accuracy(self):
        comparator = MarkdownComparator()
        reference = "The quick brown fox"
        generated = "The quick brown wolf"

        result = comparator.compare("resume", reference, generated, "ref.md", "gen.md")

        assert result.word_accuracy < 1.0

    def test_identical_words_score_perfect_word_accuracy(self):
        comparator = MarkdownComparator()
        text = "Python SQL Docker AWS"

        result = comparator.compare("resume", text, text, "ref.md", "gen.md")

        assert result.word_accuracy == 1.0

    def test_whitespace_only_differences_still_reflected_in_word_accuracy(self):
        """Word accuracy tokenizes on whitespace, so extra spaces between
        otherwise-identical words don't change the word list."""
        comparator = MarkdownComparator()
        reference = "Python  SQL   Docker"
        generated = "Python SQL Docker"

        result = comparator.compare("resume", reference, generated, "ref.md", "gen.md")

        assert result.word_accuracy == 1.0


class TestMarkdownComparatorLineAccuracyAndDiffCounts:
    def test_missing_line_is_counted(self):
        comparator = MarkdownComparator()
        reference = "Line one\nLine two\nLine three\n"
        generated = "Line one\nLine three\n"

        result = comparator.compare("resume", reference, generated, "ref.md", "gen.md")

        assert result.missing_lines == 1
        assert result.additional_lines == 0
        assert result.total_differences == 1
        assert result.line_accuracy < 1.0

    def test_additional_line_is_counted(self):
        comparator = MarkdownComparator()
        reference = "Line one\nLine two\n"
        generated = "Line one\nLine two\nExtra line\n"

        result = comparator.compare("resume", reference, generated, "ref.md", "gen.md")

        assert result.missing_lines == 0
        assert result.additional_lines == 1
        assert result.total_differences == 1

    def test_replaced_line_counts_as_both_missing_and_additional(self):
        comparator = MarkdownComparator()
        reference = "Sep 2025\n"
        generated = "Sep 2023\n"

        result = comparator.compare("resume", reference, generated, "ref.md", "gen.md")

        assert result.missing_lines == 1
        assert result.additional_lines == 1
        assert result.total_differences == 2

    def test_total_differences_always_equals_missing_plus_additional(self):
        comparator = MarkdownComparator()
        reference = "A\nB\nC\nD\n"
        generated = "A\nX\nC\nY\nZ\n"

        result = comparator.compare("resume", reference, generated, "ref.md", "gen.md")

        assert result.total_differences == result.missing_lines + result.additional_lines

    def test_empty_reference_all_generated_lines_are_additional(self):
        comparator = MarkdownComparator()
        generated = "Line one\nLine two\n"

        result = comparator.compare("resume", "", generated, "ref.md", "gen.md")

        assert result.missing_lines == 0
        assert result.additional_lines == 2

    def test_empty_generated_all_reference_lines_are_missing(self):
        comparator = MarkdownComparator()
        reference = "Line one\nLine two\nLine three\n"

        result = comparator.compare("resume", reference, "", "ref.md", "gen.md")

        assert result.missing_lines == 3
        assert result.additional_lines == 0


class TestMarkdownComparatorHeadingAccuracy:
    def test_identical_headings_score_perfect(self):
        comparator = MarkdownComparator()
        text = "# Name\n\n## Experience\n\nSome body text.\n"

        result = comparator.compare("resume", text, text, "ref.md", "gen.md")

        assert result.heading_accuracy == 1.0

    def test_dropped_heading_reduces_heading_accuracy(self):
        comparator = MarkdownComparator()
        reference = "# Name\n\n## Experience\n\n## Skills\n"
        generated = "# Name\n\n## Experience\n"

        result = comparator.compare("resume", reference, generated, "ref.md", "gen.md")

        assert result.heading_accuracy < 1.0

    def test_body_text_changes_do_not_affect_heading_accuracy(self):
        comparator = MarkdownComparator()
        reference = "# Name\n\nOriginal body paragraph text right here.\n"
        generated = "# Name\n\nCompletely different body paragraph wording.\n"

        result = comparator.compare("resume", reference, generated, "ref.md", "gen.md")

        assert result.heading_accuracy == 1.0

    def test_no_headings_in_either_document_scores_perfect(self):
        comparator = MarkdownComparator()
        reference = "Just plain text.\nNo headings here.\n"
        generated = "Just plain text.\nDifferent plain text.\n"

        result = comparator.compare("resume", reference, generated, "ref.md", "gen.md")

        assert result.heading_accuracy == 1.0


class TestMarkdownComparatorDeterminism:
    def test_repeated_comparison_of_same_input_is_identical(self):
        comparator = MarkdownComparator()
        reference = "# Name\n\nSome content.\n"
        generated = "# Name\n\nSome different content.\n"

        first = comparator.compare("resume", reference, generated, "ref.md", "gen.md")
        second = comparator.compare("resume", reference, generated, "ref.md", "gen.md")

        assert first == second


# ---------------------------------------------------------------------------
# MarkdownBenchmark — orchestration, filesystem I/O against tmp_path.
# ---------------------------------------------------------------------------


class TestMarkdownBenchmarkHappyPath:
    def test_two_matching_resumes_are_compared(self, dataset, tmp_path):
        _write(dataset.reference_dir / "alice.md", "# Alice\n\nSkills.\n")
        _write(dataset.generated_dir / "alice.md", "# Alice\n\nSkills.\n")
        _write(dataset.reference_dir / "bob.md", "# Bob\n\nExperience.\n")
        _write(dataset.generated_dir / "bob.md", "# Bob\n\nDifferent.\n")

        benchmark = MarkdownBenchmark(
            reference_dir=dataset.reference_dir,
            generated_dir=dataset.generated_dir,
            output_dir=tmp_path / "report",
        )

        result = benchmark.run()

        assert isinstance(result, MarkdownBenchmarkResult)
        assert result.total_resumes == 2
        assert result.successful_resumes == 2
        assert result.failed_resumes == 0
        assert set(result.comparisons.keys()) == {"alice", "bob"}
        assert result.comparisons["alice"].character_accuracy == 1.0
        assert result.comparisons["bob"].character_accuracy < 1.0

    def test_result_metrics_are_averaged_correctly(self, dataset, tmp_path):
        _write(dataset.reference_dir / "alice.md", "same text")
        _write(dataset.generated_dir / "alice.md", "same text")
        _write(dataset.reference_dir / "bob.md", "aaaa")
        _write(dataset.generated_dir / "bob.md", "zzzz")

        benchmark = MarkdownBenchmark(
            reference_dir=dataset.reference_dir,
            generated_dir=dataset.generated_dir,
            output_dir=tmp_path / "report",
        )

        result = benchmark.run()

        assert result.overall_character_accuracy == pytest.approx(0.5)

    def test_total_line_diff_counts_summed_across_resumes(self, dataset, tmp_path):
        _write(dataset.reference_dir / "alice.md", "A\nB\n")
        _write(dataset.generated_dir / "alice.md", "A\n")
        _write(dataset.reference_dir / "bob.md", "X\n")
        _write(dataset.generated_dir / "bob.md", "X\nY\n")

        benchmark = MarkdownBenchmark(
            reference_dir=dataset.reference_dir,
            generated_dir=dataset.generated_dir,
            output_dir=tmp_path / "report",
        )

        result = benchmark.run()

        assert result.total_missing_lines == 1
        assert result.total_additional_lines == 1
        assert result.total_differences == 2

    def test_processing_time_is_recorded(self, dataset, tmp_path):
        _write(dataset.reference_dir / "alice.md", "# Alice\n")
        _write(dataset.generated_dir / "alice.md", "# Alice\n")

        benchmark = MarkdownBenchmark(
            reference_dir=dataset.reference_dir,
            generated_dir=dataset.generated_dir,
            output_dir=tmp_path / "report",
        )

        result = benchmark.run()

        assert result.processing_time >= 0


class TestMarkdownBenchmarkMissingGenerated:
    def test_missing_generated_file_recorded_as_failure_others_continue(
        self, dataset, tmp_path
    ):
        _write(dataset.reference_dir / "alice.md", "# Alice\n")
        _write(dataset.generated_dir / "alice.md", "# Alice\n")
        _write(dataset.reference_dir / "missing.md", "# Missing\n")
        # No corresponding output file written for "missing.md".

        benchmark = MarkdownBenchmark(
            reference_dir=dataset.reference_dir,
            generated_dir=dataset.generated_dir,
            output_dir=tmp_path / "report",
        )

        result = benchmark.run()

        assert result.successful_resumes == 1
        assert result.failed_resumes == 1
        assert "missing" in result.failures
        assert "not found" in result.failures["missing"].lower()


class TestMarkdownBenchmarkReferenceDirMissing:
    def test_missing_reference_dir_raises(self, tmp_path):
        benchmark = MarkdownBenchmark(
            reference_dir=tmp_path / "does_not_exist",
            generated_dir=tmp_path / "generated",
            output_dir=tmp_path / "report",
        )

        with pytest.raises(ReferenceMarkdownDirectoryNotFoundError):
            benchmark.run()


class TestMarkdownBenchmarkEmptyReferenceDir:
    def test_empty_reference_dir_produces_zero_not_misleading_full_accuracy(
        self, dataset, tmp_path
    ):
        benchmark = MarkdownBenchmark(
            reference_dir=dataset.reference_dir,
            generated_dir=dataset.generated_dir,
            output_dir=tmp_path / "report",
        )

        result = benchmark.run()

        assert result.total_resumes == 0
        assert result.successful_resumes == 0
        assert result.overall_character_accuracy == 0.0
        assert result.overall_word_accuracy == 0.0
        assert result.overall_line_accuracy == 0.0
        assert result.overall_heading_accuracy == 0.0


class TestMarkdownBenchmarkIgnoresNonMarkdownFiles:
    def test_non_md_files_in_reference_dir_are_ignored(self, dataset, tmp_path):
        _write(dataset.reference_dir / "alice.md", "# Alice\n")
        _write(dataset.generated_dir / "alice.md", "# Alice\n")
        _write(dataset.reference_dir / "notes.txt", "not a resume")

        benchmark = MarkdownBenchmark(
            reference_dir=dataset.reference_dir,
            generated_dir=dataset.generated_dir,
            output_dir=tmp_path / "report",
        )

        result = benchmark.run()

        assert result.total_resumes == 1
        assert "notes" not in result.comparisons


class TestMarkdownBenchmarkResumeNaming:
    def test_resume_name_uses_markdown_stem_including_spaces(self, dataset, tmp_path):
        filename = "Anandu Chandran_Area Credit Manager.md"
        _write(dataset.reference_dir / filename, "# Anandu\n")
        _write(dataset.generated_dir / filename, "# Anandu\n")

        benchmark = MarkdownBenchmark(
            reference_dir=dataset.reference_dir,
            generated_dir=dataset.generated_dir,
            output_dir=tmp_path / "report",
        )

        result = benchmark.run()

        assert "Anandu Chandran_Area Credit Manager" in result.comparisons


class TestMarkdownBenchmarkReportGeneration:
    def test_benchmark_report_is_written(self, dataset, tmp_path):
        _write(dataset.reference_dir / "alice.md", "# Alice\n")
        _write(dataset.generated_dir / "alice.md", "# Alice\n")
        output_dir = tmp_path / "report"

        MarkdownBenchmark(
            reference_dir=dataset.reference_dir,
            generated_dir=dataset.generated_dir,
            output_dir=output_dir,
        ).run()

        assert (output_dir / "benchmark_report.md").is_file()

    def test_report_contains_required_sections(self, dataset, tmp_path):
        _write(dataset.reference_dir / "alice.md", "# Alice\n")
        _write(dataset.generated_dir / "alice.md", "# Alice\n")
        output_dir = tmp_path / "report"

        MarkdownBenchmark(
            reference_dir=dataset.reference_dir,
            generated_dir=dataset.generated_dir,
            output_dir=output_dir,
        ).run()

        report_text = (output_dir / "benchmark_report.md").read_text(encoding="utf-8")
        for heading in (
            "# Markdown Benchmark Report",
            "## Overall Accuracy",
            "## Summary",
            "## Per Resume Results",
            "## Worst Resumes",
            "## Recommendations",
        ):
            assert heading in report_text, heading

    def test_per_resume_results_lists_every_compared_resume(self, dataset, tmp_path):
        _write(dataset.reference_dir / "alice.md", "# Alice\n")
        _write(dataset.generated_dir / "alice.md", "# Alice\n")
        _write(dataset.reference_dir / "bob.md", "# Bob\n")
        _write(dataset.generated_dir / "bob.md", "# Bobby\n")
        output_dir = tmp_path / "report"

        MarkdownBenchmark(
            reference_dir=dataset.reference_dir,
            generated_dir=dataset.generated_dir,
            output_dir=output_dir,
        ).run()

        report_text = (output_dir / "benchmark_report.md").read_text(encoding="utf-8")
        per_resume_section = report_text.split("## Per Resume Results")[1].split(
            "## Worst Resumes"
        )[0]
        assert "alice" in per_resume_section
        assert "bob" in per_resume_section

    def test_worst_resumes_section_identifies_lowest_scoring_resume(
        self, dataset, tmp_path
    ):
        _write(dataset.reference_dir / "good.md", "same text")
        _write(dataset.generated_dir / "good.md", "same text")
        _write(dataset.reference_dir / "bad.md", "aaaa")
        _write(dataset.generated_dir / "bad.md", "zzzz")
        output_dir = tmp_path / "report"

        MarkdownBenchmark(
            reference_dir=dataset.reference_dir,
            generated_dir=dataset.generated_dir,
            output_dir=output_dir,
        ).run()

        report_text = (output_dir / "benchmark_report.md").read_text(encoding="utf-8")
        worst_section = report_text.split("## Worst Resumes")[1].split(
            "## Recommendations"
        )[0]
        assert "bad" in worst_section

    def test_recommendations_flag_low_character_accuracy(self, dataset, tmp_path):
        _write(dataset.reference_dir / "bad.md", "aaaa")
        _write(dataset.generated_dir / "bad.md", "zzzz")
        output_dir = tmp_path / "report"

        MarkdownBenchmark(
            reference_dir=dataset.reference_dir,
            generated_dir=dataset.generated_dir,
            output_dir=output_dir,
        ).run()

        report_text = (output_dir / "benchmark_report.md").read_text(encoding="utf-8")
        recommendations_section = report_text.split("## Recommendations")[1]
        assert "Character accuracy" in recommendations_section

    def test_recommendations_report_no_issues_when_perfect(self, dataset, tmp_path):
        _write(dataset.reference_dir / "alice.md", "# Alice\n\nSame content.\n")
        _write(dataset.generated_dir / "alice.md", "# Alice\n\nSame content.\n")
        output_dir = tmp_path / "report"

        MarkdownBenchmark(
            reference_dir=dataset.reference_dir,
            generated_dir=dataset.generated_dir,
            output_dir=output_dir,
        ).run()

        report_text = (output_dir / "benchmark_report.md").read_text(encoding="utf-8")
        recommendations_section = report_text.split("## Recommendations")[1]
        assert "No significant accuracy issues" in recommendations_section

    def test_failures_section_present_only_when_failures_exist(self, dataset, tmp_path):
        _write(dataset.reference_dir / "missing.md", "# Missing\n")
        output_dir = tmp_path / "report"

        MarkdownBenchmark(
            reference_dir=dataset.reference_dir,
            generated_dir=dataset.generated_dir,
            output_dir=output_dir,
        ).run()

        report_text = (output_dir / "benchmark_report.md").read_text(encoding="utf-8")
        assert "## Failures" in report_text
        assert "missing" in report_text

    def test_failures_section_omitted_when_no_failures(self, dataset, tmp_path):
        _write(dataset.reference_dir / "alice.md", "# Alice\n")
        _write(dataset.generated_dir / "alice.md", "# Alice\n")
        output_dir = tmp_path / "report"

        MarkdownBenchmark(
            reference_dir=dataset.reference_dir,
            generated_dir=dataset.generated_dir,
            output_dir=output_dir,
        ).run()

        report_text = (output_dir / "benchmark_report.md").read_text(encoding="utf-8")
        assert "## Failures" not in report_text

    def test_output_dir_created_if_missing(self, dataset, tmp_path):
        _write(dataset.reference_dir / "alice.md", "# Alice\n")
        _write(dataset.generated_dir / "alice.md", "# Alice\n")
        output_dir = tmp_path / "nested" / "report" / "dir"
        assert not output_dir.exists()

        MarkdownBenchmark(
            reference_dir=dataset.reference_dir,
            generated_dir=dataset.generated_dir,
            output_dir=output_dir,
        ).run()

        assert output_dir.is_dir()
        assert (output_dir / "benchmark_report.md").is_file()


class TestMarkdownBenchmarkDependencyInjection:
    def test_custom_markdown_comparator_is_used(self, dataset, tmp_path):
        mock_comparator = MagicMock(spec=MarkdownComparator)
        mock_comparator.compare.return_value = MarkdownComparisonResult(
            resume_name="alice",
            reference_path="ref.md",
            generated_path="gen.md",
            character_accuracy=1.0,
            word_accuracy=1.0,
            line_accuracy=1.0,
            heading_accuracy=1.0,
            missing_lines=0,
            additional_lines=0,
            total_differences=0,
        )
        _write(dataset.reference_dir / "alice.md", "# Alice\n")
        _write(dataset.generated_dir / "alice.md", "# Alice\n")

        benchmark = MarkdownBenchmark(
            markdown_comparator=mock_comparator,
            reference_dir=dataset.reference_dir,
            generated_dir=dataset.generated_dir,
            output_dir=tmp_path / "report",
        )
        benchmark.run()

        mock_comparator.compare.assert_called_once()

    def test_defaults_build_a_real_markdown_comparator(self):
        benchmark = MarkdownBenchmark()

        assert isinstance(benchmark.markdown_comparator, MarkdownComparator)
