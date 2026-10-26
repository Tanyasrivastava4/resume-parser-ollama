"""
tests/test_run_parser_evaluation.py

Tests for scripts/run_parser_evaluation.py (Milestone 14.8.1).

ParserEvaluator is always mocked or otherwise avoided — no test here ever
invokes the script's real, argument-free defaults, since those point at
the project's real benchmark/ground_truth/ and output/markdown/ and would
run a real evaluation against real data. This exercises only the
script's own CLI/orchestration logic (calling evaluate(), printing the
result), not ParserEvaluator itself.
"""

import csv
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evaluation.parser_evaluator import ParserEvaluationResult, ParserEvaluator
from config import ADJUDICATED_GROUND_TRUTH_DIR, COMPARISON_DIR, GROUND_TRUTH_DIR
from scripts.run_parser_evaluation import (
    _parse_args,
    _suite_scoped_evaluator,
    print_evaluation_result,
    resolve_ground_truth_dir,
    run_parser_evaluation_cli,
)

SAMPLE_RESULT = ParserEvaluationResult(
    total_resumes=2,
    successful_resumes=1,
    failed_resumes=1,
    overall_accuracy=0.875,
    module_accuracy={"skills": 1.0, "personal_information": 0.75},
    field_accuracy={"skills": 1.0, "personal_information.full_name": 0.5},
    resume_accuracy={"alice": 0.875},
    processing_time=1.2345,
    failures={"bob": "Markdown file not found: bob.md"},
)


@pytest.fixture
def mock_evaluator():
    evaluator = MagicMock(spec=ParserEvaluator)
    evaluator.evaluate.return_value = SAMPLE_RESULT
    evaluator.output_dir = Path("/fake/comparison")
    return evaluator


class TestRunParserEvaluationCli:
    def test_calls_evaluate_and_returns_result(self, mock_evaluator):
        result = run_parser_evaluation_cli(evaluator=mock_evaluator)

        mock_evaluator.evaluate.assert_called_once()
        assert result is SAMPLE_RESULT

    def test_prints_summary(self, mock_evaluator, capsys):
        run_parser_evaluation_cli(evaluator=mock_evaluator)

        captured = capsys.readouterr()
        assert "Total resumes:    2" in captured.out
        assert "Successful:       1" in captured.out
        assert "Failed:           1" in captured.out
        assert "Overall accuracy: 87.50%" in captured.out

    def test_prints_module_and_field_and_resume_accuracy(self, mock_evaluator, capsys):
        run_parser_evaluation_cli(evaluator=mock_evaluator)

        captured = capsys.readouterr()
        assert "Module accuracy" in captured.out
        assert "skills: 100.00%" in captured.out
        assert "Field accuracy" in captured.out
        assert "personal_information.full_name: 50.00%" in captured.out
        assert "Resume accuracy" in captured.out
        assert "alice: 87.50%" in captured.out

    def test_prints_report_location(self, mock_evaluator, capsys):
        run_parser_evaluation_cli(evaluator=mock_evaluator)

        captured = capsys.readouterr()
        assert "Reports saved to: /fake/comparison" in captured.out

    def test_prints_failures_when_present(self, mock_evaluator, capsys):
        run_parser_evaluation_cli(evaluator=mock_evaluator)

        captured = capsys.readouterr()
        assert "Failures" in captured.out
        assert "bob: Markdown file not found: bob.md" in captured.out

    def test_omits_failures_section_when_there_are_none(self, mock_evaluator, capsys):
        mock_evaluator.evaluate.return_value = ParserEvaluationResult(
            total_resumes=1,
            successful_resumes=1,
            failed_resumes=0,
            overall_accuracy=1.0,
        )

        run_parser_evaluation_cli(evaluator=mock_evaluator)

        captured = capsys.readouterr()
        assert "Failures" not in captured.out


class TestPrintEvaluationResult:
    def test_output_contains_all_expected_sections(self, capsys):
        print_evaluation_result(SAMPLE_RESULT, Path("/fake/comparison"))

        captured = capsys.readouterr()
        for heading in (
            "Parser evaluation summary",
            "Module accuracy",
            "Field accuracy",
            "Resume accuracy",
            "Failures",
        ):
            assert heading in captured.out


class TestParseArgs:
    def test_no_arguments_is_valid(self):
        """Milestone 22.4 added one new, optional argument (--suite);
        `vars(args) == {}` no longer holds now that argparse has
        something to report even when nothing was passed on the
        command line — but the value it reports, None, is exactly the
        "do nothing different" default, preserving the pre-22.4
        behavior this test originally existed to confirm."""
        args = _parse_args([])
        # Remediation R-07 added --ground-truth / --ground-truth-dir; both
        # default to None, which resolve_ground_truth_dir maps to the
        # legacy set -- still "do nothing different".
        assert vars(args) == {"suite": None, "ground_truth": None, "ground_truth_dir": None}

    def test_suite_argument(self):
        args = _parse_args(["--suite", "smoke"])
        assert args.suite == "smoke"

    def test_help_exits_cleanly(self):
        with pytest.raises(SystemExit) as exc_info:
            _parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_rejects_unexpected_argument(self):
        with pytest.raises(SystemExit) as exc_info:
            _parse_args(["--bogus"])
        assert exc_info.value.code != 0


class TestSuiteScopedEvaluator:
    """Milestone 22.4. config.GROUND_TRUTH_DIR/COMPARISON_DIR and
    scripts.benchmark_suites.resolve_suite are monkeypatched throughout
    so these tests never touch the real project's
    benchmark/ground_truth/json/ or benchmark/comparison/ — see
    tests/test_benchmark_suites.py for resolve_suite's own tests."""

    @pytest.fixture
    def real_ground_truth_dir(self, tmp_path):
        """A stand-in for the real GROUND_TRUTH_DIR: two ground-truth
        JSON files _suite_scoped_evaluator should symlink into its
        temporary, suite-scoped directory."""
        directory = tmp_path / "real_ground_truth"
        directory.mkdir()
        (directory / "resume001.json").write_text('{"skills": []}', encoding="utf-8")
        (directory / "resume002.json").write_text('{"skills": ["SQL"]}', encoding="utf-8")
        return directory

    def _patch(self, monkeypatch, rows, real_ground_truth_dir, comparison_dir):
        monkeypatch.setattr(
            "scripts.run_parser_evaluation.resolve_suite", lambda suite: rows
        )
        monkeypatch.setattr(
            "scripts.run_parser_evaluation.GROUND_TRUTH_DIR", real_ground_truth_dir
        )
        monkeypatch.setattr(
            "scripts.run_parser_evaluation.COMPARISON_DIR", comparison_dir
        )

    def test_builds_evaluator_with_filtered_mapping_csv(
        self, tmp_path, monkeypatch, real_ground_truth_dir
    ):
        rows = [
            {"ResumeID": "001", "PDF": "a.pdf", "Markdown": "a.md", "JSON": "resume001.json"},
            {"ResumeID": "002", "PDF": "b.pdf", "Markdown": "b.md", "JSON": "resume002.json"},
        ]
        self._patch(monkeypatch, rows, real_ground_truth_dir, tmp_path / "comparison")

        with _suite_scoped_evaluator("smoke") as evaluator:
            mapping_path = evaluator.ground_truth_dir / "mapping.csv"
            with mapping_path.open(newline="", encoding="utf-8") as mapping_file:
                written_rows = list(csv.DictReader(mapping_file))
            assert [r["Markdown"] for r in written_rows] == ["a.md", "b.md"]

    def test_ground_truth_json_files_are_symlinked_and_readable(
        self, tmp_path, monkeypatch, real_ground_truth_dir
    ):
        rows = [{"ResumeID": "001", "PDF": "a.pdf", "Markdown": "a.md", "JSON": "resume001.json"}]
        self._patch(monkeypatch, rows, real_ground_truth_dir, tmp_path / "comparison")

        with _suite_scoped_evaluator("smoke") as evaluator:
            linked = evaluator.ground_truth_dir / "resume001.json"
            assert linked.is_symlink()
            assert linked.read_text(encoding="utf-8") == '{"skills": []}'

    def test_output_dir_is_scoped_by_suite_name(
        self, tmp_path, monkeypatch, real_ground_truth_dir
    ):
        rows = [{"ResumeID": "001", "PDF": "a.pdf", "Markdown": "a.md", "JSON": "resume001.json"}]
        comparison_dir = tmp_path / "comparison"
        self._patch(monkeypatch, rows, real_ground_truth_dir, comparison_dir)

        with _suite_scoped_evaluator("smoke") as evaluator:
            assert evaluator.output_dir == comparison_dir / "smoke"

    def test_temporary_ground_truth_dir_is_cleaned_up_after_the_context_exits(
        self, tmp_path, monkeypatch, real_ground_truth_dir
    ):
        rows = [{"ResumeID": "001", "PDF": "a.pdf", "Markdown": "a.md", "JSON": "resume001.json"}]
        self._patch(monkeypatch, rows, real_ground_truth_dir, tmp_path / "comparison")

        with _suite_scoped_evaluator("smoke") as evaluator:
            temp_dir = evaluator.ground_truth_dir
            assert temp_dir.exists()

        assert not temp_dir.exists()

    def test_real_ground_truth_files_are_never_modified(
        self, tmp_path, monkeypatch, real_ground_truth_dir
    ):
        """The symlink is one-directional -- nothing about building or
        tearing down a suite-scoped evaluator may touch the real
        ground-truth files it links to."""
        rows = [{"ResumeID": "001", "PDF": "a.pdf", "Markdown": "a.md", "JSON": "resume001.json"}]
        self._patch(monkeypatch, rows, real_ground_truth_dir, tmp_path / "comparison")
        original_content = (real_ground_truth_dir / "resume001.json").read_text(encoding="utf-8")

        with _suite_scoped_evaluator("smoke"):
            pass

        assert (real_ground_truth_dir / "resume001.json").read_text(encoding="utf-8") == original_content


class TestRunParserEvaluationCliSuite:
    """Milestone 22.4: an explicitly-injected `evaluator=` always wins
    over `suite=`, and `suite="full"`/`suite=None` never build a
    suite-scoped evaluator at all -- both preserve every existing
    caller's behavior unchanged (requirement 5)."""

    def test_explicit_evaluator_ignores_suite(self, mock_evaluator):
        result = run_parser_evaluation_cli(evaluator=mock_evaluator, suite="smoke")

        mock_evaluator.evaluate.assert_called_once()
        assert result is SAMPLE_RESULT

    def test_suite_full_does_not_build_a_suite_scoped_evaluator(self, monkeypatch, capsys):
        """Requirement 5: --suite full must be indistinguishable from
        omitting --suite -- confirmed here by asserting the
        suite-resolution machinery is never even invoked."""
        sentinel_evaluator = MagicMock(spec=ParserEvaluator)
        sentinel_evaluator.evaluate.return_value = SAMPLE_RESULT
        sentinel_evaluator.output_dir = Path("/fake/comparison")
        monkeypatch.setattr(
            "scripts.run_parser_evaluation.ParserEvaluator", lambda: sentinel_evaluator
        )

        def _fail_if_called(suite):
            raise AssertionError("resolve_suite must not be called for suite='full'")

        monkeypatch.setattr("scripts.run_parser_evaluation.resolve_suite", _fail_if_called)

        result = run_parser_evaluation_cli(suite="full")

        assert result is SAMPLE_RESULT


# ---------------------------------------------------------------------------
# Remediation R-07 (2026-10-17): explicit ground-truth selection.
# ---------------------------------------------------------------------------


class TestGroundTruthSelectionParseArgs:
    def test_named_selection(self):
        args = _parse_args(["--ground-truth", "adjudicated_v1"])
        assert args.ground_truth == "adjudicated_v1"
        assert args.ground_truth_dir is None

    def test_explicit_directory(self, tmp_path):
        args = _parse_args(["--ground-truth-dir", str(tmp_path)])
        assert args.ground_truth_dir == tmp_path
        assert args.ground_truth is None

    def test_named_and_directory_are_mutually_exclusive(self, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            _parse_args(["--ground-truth", "legacy", "--ground-truth-dir", str(tmp_path)])
        assert exc_info.value.code != 0

    def test_unknown_selection_is_rejected(self):
        with pytest.raises(SystemExit) as exc_info:
            _parse_args(["--ground-truth", "nonsense"])
        assert exc_info.value.code != 0


class TestResolveGroundTruthDir:
    def test_default_is_the_legacy_directory(self):
        assert resolve_ground_truth_dir() == (GROUND_TRUTH_DIR, "legacy")
        assert resolve_ground_truth_dir(None, None) == (GROUND_TRUTH_DIR, "legacy")

    def test_named_adjudicated_selection(self):
        assert resolve_ground_truth_dir("adjudicated_v1") == (
            ADJUDICATED_GROUND_TRUTH_DIR,
            "adjudicated_v1",
        )

    def test_explicit_directory_is_labelled_custom(self, tmp_path):
        assert resolve_ground_truth_dir(ground_truth_dir=tmp_path) == (tmp_path, "custom")

    def test_both_arguments_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            resolve_ground_truth_dir("legacy", tmp_path)

    def test_unknown_name_rejected(self):
        with pytest.raises(ValueError):
            resolve_ground_truth_dir("nonsense")


class TestRealGroundTruthDirectories:
    """Both tracked GT sets exist with a mapping.csv; the adjudicated set
    is the 22-resume population (resume012, the job description, was
    permanently excluded in Milestone 23.67 §4)."""

    @staticmethod
    def _ids(directory):
        with (directory / "mapping.csv").open(newline="", encoding="utf-8") as f:
            return [row["ResumeID"] for row in csv.DictReader(f)]

    def test_legacy_mapping_has_23_rows_including_resume012(self):
        ids = self._ids(GROUND_TRUTH_DIR)
        assert len(ids) == 23
        assert "012" in ids

    def test_adjudicated_mapping_has_22_rows_without_resume012(self):
        ids = self._ids(ADJUDICATED_GROUND_TRUTH_DIR)
        assert len(ids) == 22
        assert "012" not in ids
        assert set(ids) == set(self._ids(GROUND_TRUTH_DIR)) - {"012"}


class TestPrintEvaluationResultGroundTruthLine:
    def test_omitted_ground_truth_prints_no_line(self, capsys):
        print_evaluation_result(SAMPLE_RESULT, Path("/fake/comparison"))
        assert "Ground truth:" not in capsys.readouterr().out

    def test_ground_truth_line_names_directory_and_row_count(self, tmp_path, capsys):
        with (tmp_path / "mapping.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["ResumeID", "PDF", "Markdown", "JSON"])
            writer.writeheader()
            writer.writerow({"ResumeID": "001", "PDF": "a.pdf", "Markdown": "a.md", "JSON": "a.json"})
            writer.writerow({"ResumeID": "002", "PDF": "b.pdf", "Markdown": "b.md", "JSON": "b.json"})
        print_evaluation_result(SAMPLE_RESULT, Path("/fake/comparison"), tmp_path)
        out = capsys.readouterr().out
        assert f"Ground truth:     {tmp_path} (2 resumes in mapping.csv)" in out

    def test_unreadable_mapping_does_not_fail_printing(self, tmp_path, capsys):
        print_evaluation_result(SAMPLE_RESULT, Path("/fake/comparison"), tmp_path / "missing")
        assert "mapping.csv unreadable" in capsys.readouterr().out


class TestRunParserEvaluationCliGroundTruth:
    """The default selection must keep the exact pre-R-07 code path (a
    bare `ParserEvaluator()`); a non-default selection must build the
    evaluator with that directory and a selection-scoped output_dir."""

    def test_default_selection_builds_a_bare_evaluator(self, monkeypatch):
        calls = []

        def factory(*args, **kwargs):
            calls.append((args, kwargs))
            evaluator = MagicMock(spec=ParserEvaluator)
            evaluator.evaluate.return_value = SAMPLE_RESULT
            evaluator.output_dir = Path("/fake/comparison")
            evaluator.ground_truth_dir = GROUND_TRUTH_DIR
            return evaluator

        monkeypatch.setattr("scripts.run_parser_evaluation.ParserEvaluator", factory)
        run_parser_evaluation_cli()
        assert calls == [((), {})]

    def test_adjudicated_selection_builds_evaluator_with_that_directory(self, monkeypatch):
        calls = []

        def factory(*args, **kwargs):
            calls.append((args, kwargs))
            evaluator = MagicMock(spec=ParserEvaluator)
            evaluator.evaluate.return_value = SAMPLE_RESULT
            evaluator.output_dir = kwargs["output_dir"]
            evaluator.ground_truth_dir = kwargs["ground_truth_dir"]
            return evaluator

        monkeypatch.setattr("scripts.run_parser_evaluation.ParserEvaluator", factory)
        run_parser_evaluation_cli(ground_truth="adjudicated_v1")
        assert calls == [
            (
                (),
                {
                    "ground_truth_dir": ADJUDICATED_GROUND_TRUTH_DIR,
                    "output_dir": COMPARISON_DIR / "adjudicated_v1",
                },
            )
        ]

    def test_summary_names_the_selected_ground_truth(self, monkeypatch, capsys):
        def factory(*args, **kwargs):
            evaluator = MagicMock(spec=ParserEvaluator)
            evaluator.evaluate.return_value = SAMPLE_RESULT
            evaluator.output_dir = kwargs["output_dir"]
            evaluator.ground_truth_dir = kwargs["ground_truth_dir"]
            return evaluator

        monkeypatch.setattr("scripts.run_parser_evaluation.ParserEvaluator", factory)
        run_parser_evaluation_cli(ground_truth="adjudicated_v1")
        out = capsys.readouterr().out
        assert f"Ground truth:     {ADJUDICATED_GROUND_TRUTH_DIR} (22 resumes in mapping.csv)" in out

    def test_suite_with_non_default_ground_truth_symlinks_from_that_directory(
        self, tmp_path, monkeypatch
    ):
        other_gt = tmp_path / "other_gt"
        other_gt.mkdir()
        (other_gt / "resume001.json").write_text('{"skills": ["R"]}', encoding="utf-8")
        rows = [{"ResumeID": "001", "PDF": "a.pdf", "Markdown": "a.md", "JSON": "resume001.json"}]
        seen = {}

        def fake_resolve_suite(suite, **kwargs):
            seen.update(kwargs)
            return rows

        monkeypatch.setattr("scripts.run_parser_evaluation.resolve_suite", fake_resolve_suite)
        comparison_dir = tmp_path / "comparison"
        monkeypatch.setattr("scripts.run_parser_evaluation.COMPARISON_DIR", comparison_dir)

        with _suite_scoped_evaluator(
            "smoke", ground_truth_dir=other_gt, output_dir=comparison_dir / "custom" / "smoke"
        ) as evaluator:
            linked = evaluator.ground_truth_dir / "resume001.json"
            assert linked.is_symlink()
            assert linked.resolve() == (other_gt / "resume001.json").resolve()
            assert evaluator.output_dir == comparison_dir / "custom" / "smoke"
        assert seen == {"ground_truth_dir": other_gt}

    def test_suite_with_default_ground_truth_calls_resolve_suite_unchanged(
        self, tmp_path, monkeypatch
    ):
        """The pre-R-07 call shape -- resolve_suite(suite) with no
        keyword -- is preserved for the default selection."""
        real_gt = tmp_path / "gt"
        real_gt.mkdir()
        (real_gt / "resume001.json").write_text("{}", encoding="utf-8")
        rows = [{"ResumeID": "001", "PDF": "a.pdf", "Markdown": "a.md", "JSON": "resume001.json"}]
        monkeypatch.setattr("scripts.run_parser_evaluation.resolve_suite", lambda suite: rows)
        monkeypatch.setattr("scripts.run_parser_evaluation.GROUND_TRUTH_DIR", real_gt)
        monkeypatch.setattr("scripts.run_parser_evaluation.COMPARISON_DIR", tmp_path / "comparison")

        with _suite_scoped_evaluator("smoke", ground_truth_dir=real_gt) as evaluator:
            assert evaluator.output_dir == tmp_path / "comparison" / "smoke"
