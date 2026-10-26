"""
tests/test_run_parser_pipeline.py

Tests for scripts/run_parser_pipeline.py (Milestone 14.8.1;
ResumeRetryQueue integration added in Milestone 18.7).

ParserPipeline is always mocked or otherwise avoided — no test here ever
invokes the script's real, argument-free defaults, since those point at
the project's real output/markdown/ and would run real extraction against
real data. This exercises only the script's own CLI/orchestration logic
(finding files, continuing past failures, printing a summary, and — since
Milestone 18.7 — queuing failures for ResumeRetryQueue and reporting the
three-way initial/retried/permanent-failure breakdown), not ParserPipeline
or ResumeRetryQueue themselves (each has its own dedicated test file).
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from config import OUTPUT_DIR
from pipeline.parser_pipeline import ParserPipeline
from scripts.run_parser_pipeline import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    _parse_args,
    find_markdown_files,
    find_suite_markdown_files,
    run_parser_pipeline_cli,
)
from services.base_module_extractor import InvalidJSONError


def _fake_result(processing_time: float = 0.1):
    return SimpleNamespace(
        draft_json={}, module_results={}, merged_json={}, processing_time=processing_time
    )


@pytest.fixture
def mock_pipeline():
    return MagicMock(spec=ParserPipeline)


class TestFindMarkdownFiles:
    def test_finds_md_files_recursively_sorted(self, tmp_path):
        (tmp_path / "b.md").write_text("# B", encoding="utf-8")
        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / "a.md").write_text("# A", encoding="utf-8")
        (tmp_path / "not_markdown.txt").write_text("ignore me", encoding="utf-8")

        result = find_markdown_files(tmp_path)

        assert result == sorted(result)
        assert {path.name for path in result} == {"a.md", "b.md"}

    def test_returns_empty_list_for_missing_directory(self, tmp_path):
        assert find_markdown_files(tmp_path / "does_not_exist") == []

    def test_returns_empty_list_for_empty_directory(self, tmp_path):
        assert find_markdown_files(tmp_path) == []


class TestFindSuiteMarkdownFiles:
    """Milestone 22.4. scripts.benchmark_suites.resolve_suite is
    monkeypatched here so these tests never touch the real project's
    configs/benchmark_suites.yaml or mapping.csv — see
    tests/test_benchmark_suites.py for resolve_suite's own tests."""

    def test_resolves_suite_rows_to_sorted_paths_under_input_dir(self, tmp_path, monkeypatch):
        rows = [
            {"ResumeID": "002", "PDF": "bob.pdf", "Markdown": "bob.md", "JSON": "resume002.json"},
            {"ResumeID": "001", "PDF": "alice.pdf", "Markdown": "alice.md", "JSON": "resume001.json"},
        ]
        monkeypatch.setattr(
            "scripts.run_parser_pipeline.resolve_suite", lambda suite: rows
        )

        result = find_suite_markdown_files(tmp_path, "smoke")

        assert result == sorted(result)
        assert result == [tmp_path / "alice.md", tmp_path / "bob.md"]

    def test_includes_path_even_if_file_does_not_exist_on_disk(self, tmp_path, monkeypatch):
        """A suite member with no markdown file yet is still returned —
        ParserPipeline.run() raising MarkdownNotFoundError for it is
        handled by the existing per-resume failure handling, the same
        as any other missing file."""
        rows = [{"ResumeID": "001", "PDF": "x.pdf", "Markdown": "missing.md", "JSON": "x.json"}]
        monkeypatch.setattr(
            "scripts.run_parser_pipeline.resolve_suite", lambda suite: rows
        )

        result = find_suite_markdown_files(tmp_path, "smoke")

        assert result == [tmp_path / "missing.md"]

    def test_propagates_resolve_suite_errors(self, tmp_path, monkeypatch):
        from scripts.benchmark_suites import SuiteNotFoundError

        def _raise(suite):
            raise SuiteNotFoundError("nope")

        monkeypatch.setattr("scripts.run_parser_pipeline.resolve_suite", _raise)

        with pytest.raises(SuiteNotFoundError):
            find_suite_markdown_files(tmp_path, "doesnotexist")


class TestRunParserPipelineCli:
    def test_calls_pipeline_for_every_markdown_file(self, tmp_path, mock_pipeline, capsys):
        (tmp_path / "alice.md").write_text("# Alice", encoding="utf-8")
        (tmp_path / "bob.md").write_text("# Bob", encoding="utf-8")
        mock_pipeline.run.return_value = _fake_result()

        run_parser_pipeline_cli(tmp_path, DEFAULT_OUTPUT_DIR, pipeline=mock_pipeline)

        assert mock_pipeline.run.call_count == 2
        called_paths = {call.args[0] for call in mock_pipeline.run.call_args_list}
        assert called_paths == {tmp_path / "alice.md", tmp_path / "bob.md"}

    def test_continues_after_one_resume_fails(self, tmp_path, mock_pipeline, capsys):
        (tmp_path / "alice.md").write_text("# Alice", encoding="utf-8")
        (tmp_path / "bob.md").write_text("# Bob", encoding="utf-8")

        def _run(markdown_path):
            if Path(markdown_path).stem == "alice":
                raise RuntimeError("model exploded")
            return _fake_result()

        mock_pipeline.run.side_effect = _run

        run_parser_pipeline_cli(tmp_path, DEFAULT_OUTPUT_DIR, pipeline=mock_pipeline)

        # bob (1) + alice's initial attempt (1) + alice's ResumeRetryQueue
        # attempt (1) — RuntimeError isn't one of ResumeRetryQueue's own
        # retryable exception types, so it passes straight through
        # without ResumeRetryQueue retrying it a second time itself.
        assert mock_pipeline.run.call_count == 3
        captured = capsys.readouterr()
        assert "Total:            2" in captured.out
        assert "Success:          1" in captured.out
        assert "Failed:           1" in captured.out
        assert "Initial successes:    1" in captured.out
        assert "Retried successes:    0" in captured.out
        assert "Permanent failures:   1" in captured.out

    def test_prints_average_processing_time(self, tmp_path, mock_pipeline, capsys):
        (tmp_path / "alice.md").write_text("# Alice", encoding="utf-8")
        (tmp_path / "bob.md").write_text("# Bob", encoding="utf-8")
        mock_pipeline.run.side_effect = [_fake_result(2.0), _fake_result(4.0)]

        run_parser_pipeline_cli(tmp_path, DEFAULT_OUTPUT_DIR, pipeline=mock_pipeline)

        captured = capsys.readouterr()
        assert "Average time:     3.0000s" in captured.out

    def test_empty_input_dir_prints_zero_summary(self, tmp_path, mock_pipeline, capsys):
        run_parser_pipeline_cli(tmp_path, DEFAULT_OUTPUT_DIR, pipeline=mock_pipeline)

        mock_pipeline.run.assert_not_called()
        captured = capsys.readouterr()
        assert "Total:            0" in captured.out
        assert "Average time:     0.0000s" in captured.out
        # No failures at all -> the retry queue phase never runs.
        assert "Retry Queue" not in captured.out
        assert "Initial successes:    0" in captured.out
        assert "Retried successes:    0" in captured.out
        assert "Permanent failures:   0" in captured.out


class TestReviewReportPersistence:
    """Milestone V5.3: --persist-review-reports / persist_review_reports.
    evaluation.review_result_contract.write_review_result_contract is
    monkeypatched throughout -- this exercises only this script's own
    orchestration logic (whether/how it is called), not the contract-
    building/serialization/file-writing logic itself, which has its own
    dedicated tests in tests/test_review_result_contract.py. Matches
    this file's own established philosophy (see module docstring)."""

    def test_default_off_never_calls_write_review_result_contract(
        self, tmp_path, mock_pipeline, monkeypatch
    ):
        (tmp_path / "alice.md").write_text("# Alice", encoding="utf-8")
        mock_pipeline.run.return_value = _fake_result()
        write_mock = MagicMock()
        monkeypatch.setattr("scripts.run_parser_pipeline.write_review_result_contract", write_mock)

        run_parser_pipeline_cli(tmp_path, DEFAULT_OUTPUT_DIR, pipeline=mock_pipeline)

        write_mock.assert_not_called()

    def test_enabled_calls_write_review_result_contract_for_each_success(
        self, tmp_path, mock_pipeline, monkeypatch
    ):
        (tmp_path / "alice.md").write_text("# Alice", encoding="utf-8")
        (tmp_path / "bob.md").write_text("# Bob", encoding="utf-8")
        result = _fake_result()
        mock_pipeline.run.return_value = result
        write_mock = MagicMock()
        monkeypatch.setattr("scripts.run_parser_pipeline.write_review_result_contract", write_mock)

        run_parser_pipeline_cli(
            tmp_path, DEFAULT_OUTPUT_DIR, pipeline=mock_pipeline, persist_review_reports=True
        )

        assert write_mock.call_count == 2
        called_stems = {call.args[0] for call in write_mock.call_args_list}
        assert called_stems == {"alice", "bob"}
        # resume_id and stem are the same value; result and OUTPUT_DIR
        # (the real, hardcoded root -- never args.output_dir) are passed
        # through unchanged -- see evaluation/review_result_contract.py's
        # own review_report_output_path() docstring for why.
        for call in write_mock.call_args_list:
            resume_id, stem, passed_result, output_root = call.args
            assert resume_id == stem
            assert passed_result is result
            assert output_root == OUTPUT_DIR

    def test_never_called_for_a_failed_resume(self, tmp_path, mock_pipeline, monkeypatch):
        (tmp_path / "broken.md").write_text("# Broken", encoding="utf-8")
        mock_pipeline.run.side_effect = RuntimeError("model exploded")
        write_mock = MagicMock()
        monkeypatch.setattr("scripts.run_parser_pipeline.write_review_result_contract", write_mock)

        run_parser_pipeline_cli(
            tmp_path, DEFAULT_OUTPUT_DIR, pipeline=mock_pipeline, persist_review_reports=True
        )

        write_mock.assert_not_called()

    def test_called_for_a_resume_that_only_succeeds_on_retry(
        self, tmp_path, mock_pipeline, monkeypatch
    ):
        (tmp_path / "flaky.md").write_text("# Flaky", encoding="utf-8")
        call_count = {"n": 0}

        def _run(markdown_path):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise InvalidJSONError("bad json on the first try")
            return _fake_result()

        mock_pipeline.run.side_effect = _run
        write_mock = MagicMock()
        monkeypatch.setattr("scripts.run_parser_pipeline.write_review_result_contract", write_mock)

        run_parser_pipeline_cli(
            tmp_path, DEFAULT_OUTPUT_DIR, pipeline=mock_pipeline, persist_review_reports=True
        )

        write_mock.assert_called_once()
        assert write_mock.call_args.args[0] == "flaky"

    def test_persistence_failure_does_not_fail_the_resume(
        self, tmp_path, mock_pipeline, monkeypatch, capsys
    ):
        """A resume that parsed successfully must still be reported as
        a success even if the separate, optional review-artifact write
        fails."""
        (tmp_path / "alice.md").write_text("# Alice", encoding="utf-8")
        mock_pipeline.run.return_value = _fake_result()
        write_mock = MagicMock(side_effect=OSError("disk full"))
        monkeypatch.setattr("scripts.run_parser_pipeline.write_review_result_contract", write_mock)

        run_parser_pipeline_cli(
            tmp_path, DEFAULT_OUTPUT_DIR, pipeline=mock_pipeline, persist_review_reports=True
        )

        write_mock.assert_called_once()
        captured = capsys.readouterr()
        assert "Success:          1" in captured.out
        assert "Failed:           0" in captured.out
        assert "Permanent failures:   0" in captured.out

    def test_review_reports_directory_printed_only_when_enabled(
        self, tmp_path, mock_pipeline, monkeypatch, capsys
    ):
        (tmp_path / "alice.md").write_text("# Alice", encoding="utf-8")
        mock_pipeline.run.return_value = _fake_result()
        monkeypatch.setattr("scripts.run_parser_pipeline.write_review_result_contract", MagicMock())

        run_parser_pipeline_cli(tmp_path, DEFAULT_OUTPUT_DIR, pipeline=mock_pipeline)
        captured_off = capsys.readouterr()
        assert "Review reports directory" not in captured_off.out

        run_parser_pipeline_cli(
            tmp_path, DEFAULT_OUTPUT_DIR, pipeline=mock_pipeline, persist_review_reports=True
        )
        captured_on = capsys.readouterr()
        assert "Review reports directory" in captured_on.out


class TestRunParserPipelineCliSuite:
    """Milestone 22.4: --suite restricts run_parser_pipeline_cli to a
    resolved subset instead of scanning input_dir. resolve_suite is
    monkeypatched throughout — see tests/test_benchmark_suites.py for
    its own tests, and TestFindSuiteMarkdownFiles above."""

    def test_suite_processes_only_resolved_files_not_the_whole_directory(
        self, tmp_path, mock_pipeline, monkeypatch
    ):
        (tmp_path / "alice.md").write_text("# Alice", encoding="utf-8")
        (tmp_path / "bob.md").write_text("# Bob", encoding="utf-8")
        (tmp_path / "carol.md").write_text("# Carol", encoding="utf-8")
        mock_pipeline.run.return_value = _fake_result()
        rows = [{"ResumeID": "001", "PDF": "a.pdf", "Markdown": "alice.md", "JSON": "a.json"}]
        monkeypatch.setattr(
            "scripts.run_parser_pipeline.resolve_suite", lambda suite: rows
        )

        run_parser_pipeline_cli(
            tmp_path, DEFAULT_OUTPUT_DIR, pipeline=mock_pipeline, suite="smoke"
        )

        # Only alice.md, resolved via the (mocked) suite -- bob.md and
        # carol.md exist on disk but are not part of the suite, and a
        # full directory scan (find_markdown_files) is never called.
        mock_pipeline.run.assert_called_once_with(tmp_path / "alice.md")

    def test_suite_full_behaves_identically_to_no_suite(
        self, tmp_path, mock_pipeline, capsys
    ):
        (tmp_path / "alice.md").write_text("# Alice", encoding="utf-8")
        (tmp_path / "bob.md").write_text("# Bob", encoding="utf-8")
        mock_pipeline.run.return_value = _fake_result()

        run_parser_pipeline_cli(
            tmp_path, DEFAULT_OUTPUT_DIR, pipeline=mock_pipeline, suite="full"
        )

        # Both files processed via the ordinary directory scan --
        # resolve_suite is never invoked for "full" (requirement 5:
        # preserve existing behavior when the suite is effectively
        # "everything").
        assert mock_pipeline.run.call_count == 2
        called_paths = {call.args[0] for call in mock_pipeline.run.call_args_list}
        assert called_paths == {tmp_path / "alice.md", tmp_path / "bob.md"}

    def test_no_suite_argument_is_unaffected(self, tmp_path, mock_pipeline):
        """suite=None (the default) takes the exact same code path as
        --suite full -- confirming Milestone 22.4 added nothing that
        changes default behavior."""
        (tmp_path / "alice.md").write_text("# Alice", encoding="utf-8")
        mock_pipeline.run.return_value = _fake_result()

        run_parser_pipeline_cli(tmp_path, DEFAULT_OUTPUT_DIR, pipeline=mock_pipeline)

        mock_pipeline.run.assert_called_once_with(tmp_path / "alice.md")


class TestResumeRetryQueueIntegration:
    """Milestone 18.7: resumes that fail their initial ParserPipeline
    attempt are queued, not retried immediately, and handed to a
    ResumeRetryQueue only after every batch has been attempted at
    least once — ResumeRetryQueue's own retry logic (not modified by
    this milestone) then decides whether a queued resume ultimately
    succeeds or permanently fails."""

    def test_resume_that_fails_once_then_succeeds_on_the_retry_queue(
        self, tmp_path, mock_pipeline, capsys
    ):
        (tmp_path / "flaky.md").write_text("# Flaky", encoding="utf-8")
        (tmp_path / "steady.md").write_text("# Steady", encoding="utf-8")
        call_count = {"flaky": 0}

        def _run(markdown_path):
            if Path(markdown_path).stem == "flaky":
                call_count["flaky"] += 1
                if call_count["flaky"] == 1:
                    raise InvalidJSONError("bad json on the first try")
                return _fake_result()
            return _fake_result()

        mock_pipeline.run.side_effect = _run

        run_parser_pipeline_cli(tmp_path, DEFAULT_OUTPUT_DIR, pipeline=mock_pipeline)

        # steady (1) + flaky's initial failed attempt (1) + flaky's
        # ResumeRetryQueue attempt, which succeeds immediately (1).
        assert mock_pipeline.run.call_count == 3
        captured = capsys.readouterr()
        assert "Retry Queue (1 resume(s))" in captured.out
        assert "Initial successes:    1" in captured.out
        assert "Retried successes:    1" in captured.out
        assert "Permanent failures:   0" in captured.out
        assert "Permanently failed resumes:" not in captured.out

    def test_resume_that_fails_the_entire_retry_queue_is_a_permanent_failure(
        self, tmp_path, mock_pipeline, capsys
    ):
        (tmp_path / "broken.md").write_text("# Broken", encoding="utf-8")
        mock_pipeline.run.side_effect = InvalidJSONError("always invalid")

        run_parser_pipeline_cli(tmp_path, DEFAULT_OUTPUT_DIR, pipeline=mock_pipeline)

        # Initial attempt (1) + ResumeRetryQueue's own MAX_RESUME_ATTEMPTS
        # (2: one fresh attempt, one internal retry) — InvalidJSONError
        # is retryable, so ResumeRetryQueue genuinely retries it once
        # itself before giving up.
        assert mock_pipeline.run.call_count == 3
        captured = capsys.readouterr()
        assert "Initial successes:    0" in captured.out
        assert "Retried successes:    0" in captured.out
        assert "Permanent failures:   1" in captured.out
        assert "Permanently failed resumes:" in captured.out
        assert "broken.md" in captured.out
        assert "always invalid" in captured.out

    def test_mixed_outcome_report_reconciles_to_the_total(
        self, tmp_path, mock_pipeline, capsys
    ):
        """One resume succeeds immediately, one succeeds only once
        queued, one never succeeds — the final report's three counts
        must add up to the total number of resumes."""
        (tmp_path / "clean.md").write_text("# Clean", encoding="utf-8")
        (tmp_path / "flaky.md").write_text("# Flaky", encoding="utf-8")
        (tmp_path / "broken.md").write_text("# Broken", encoding="utf-8")
        flaky_calls = {"n": 0}

        def _run(markdown_path):
            stem = Path(markdown_path).stem
            if stem == "clean":
                return _fake_result()
            if stem == "flaky":
                flaky_calls["n"] += 1
                if flaky_calls["n"] == 1:
                    raise InvalidJSONError("flaky failure")
                return _fake_result()
            raise InvalidJSONError("broken is always broken")

        mock_pipeline.run.side_effect = _run

        run_parser_pipeline_cli(tmp_path, DEFAULT_OUTPUT_DIR, pipeline=mock_pipeline)

        captured = capsys.readouterr()
        assert "Total resumes:        3" in captured.out
        assert "Initial successes:    1" in captured.out
        assert "Retried successes:    1" in captured.out
        assert "Permanent failures:   1" in captured.out
        assert "broken.md" in captured.out
        assert "flaky.md" not in captured.out.split("Permanently failed resumes:")[-1]

    def test_retry_queue_reuses_the_same_pipeline_instance(
        self, tmp_path, mock_pipeline, capsys
    ):
        """ResumeRetryQueue must be built around the exact same
        ParserPipeline instance the initial pass used, not a fresh
        default one — confirmed indirectly, since the retry attempt
        only shows up as a further call on `mock_pipeline` itself."""
        (tmp_path / "broken.md").write_text("# Broken", encoding="utf-8")
        mock_pipeline.run.side_effect = InvalidJSONError("bad json")

        run_parser_pipeline_cli(tmp_path, DEFAULT_OUTPUT_DIR, pipeline=mock_pipeline)

        assert mock_pipeline.run.call_count == 3


class TestParseArgs:
    def test_defaults(self):
        args = _parse_args([])

        assert args.input_dir == DEFAULT_INPUT_DIR
        assert args.output_dir == DEFAULT_OUTPUT_DIR
        assert args.suite is None
        assert args.persist_review_reports is False
        assert args.output_root is None  # remediation R-08

    def test_suite_argument(self):
        args = _parse_args(["--suite", "smoke"])

        assert args.suite == "smoke"

    def test_persist_review_reports_flag(self):
        args = _parse_args(["--persist-review-reports"])
        assert args.persist_review_reports is True

    def test_custom_input_and_output_dir(self, tmp_path):
        custom_input = tmp_path / "custom_input"
        custom_output = tmp_path / "custom_output"

        args = _parse_args(
            ["--input-dir", str(custom_input), "--output-dir", str(custom_output)]
        )

        assert args.input_dir == custom_input
        assert args.output_dir == custom_output

    def test_rejects_unexpected_argument(self):
        with pytest.raises(SystemExit):
            _parse_args(["--bogus"])



class TestOutputRoot:
    """Remediation R-08 (2026-10-17): --output-root redirects every
    artifact; omitting it preserves the pre-R-08 behavior exactly."""

    def test_output_root_argument(self, tmp_path):
        args = _parse_args(["--output-root", str(tmp_path / "root")])
        assert args.output_root == tmp_path / "root"

    def test_default_builds_a_bare_pipeline(self, tmp_path, monkeypatch, capsys):
        calls = []

        def factory(*args, **kwargs):
            calls.append((args, kwargs))
            pipeline = MagicMock(spec=ParserPipeline)
            pipeline.run.return_value = _fake_result()
            return pipeline

        monkeypatch.setattr("scripts.run_parser_pipeline.ParserPipeline", factory)
        (tmp_path / "alice.md").write_text("# Alice", encoding="utf-8")

        run_parser_pipeline_cli(tmp_path, DEFAULT_OUTPUT_DIR)

        assert calls == [((), {})]
        assert f"Output directory: {DEFAULT_OUTPUT_DIR}" in capsys.readouterr().out

    def test_output_root_builds_pipeline_with_that_root_and_reports_it(
        self, tmp_path, monkeypatch, capsys
    ):
        calls = []

        def factory(*args, **kwargs):
            calls.append((args, kwargs))
            pipeline = MagicMock(spec=ParserPipeline)
            pipeline.run.return_value = _fake_result()
            return pipeline

        monkeypatch.setattr("scripts.run_parser_pipeline.ParserPipeline", factory)
        (tmp_path / "alice.md").write_text("# Alice", encoding="utf-8")
        root = tmp_path / "run_root"

        run_parser_pipeline_cli(tmp_path, DEFAULT_OUTPUT_DIR, output_root=root)

        assert calls == [((), {"output_root": root})]
        assert f"Output directory: {root / 'json_final'}" in capsys.readouterr().out

    def test_output_root_does_not_warn_about_output_dir(self, tmp_path, mock_pipeline, caplog):
        mock_pipeline.run.return_value = _fake_result()
        (tmp_path / "alice.md").write_text("# Alice", encoding="utf-8")

        with caplog.at_level("WARNING", logger="scripts.run_parser_pipeline"):
            run_parser_pipeline_cli(
                tmp_path, DEFAULT_OUTPUT_DIR, pipeline=mock_pipeline, output_root=tmp_path / "r"
            )

        assert "only affects what this summary reports" not in caplog.text

    def test_review_reports_follow_the_output_root(self, tmp_path, mock_pipeline, monkeypatch, capsys):
        (tmp_path / "alice.md").write_text("# Alice", encoding="utf-8")
        result = _fake_result()
        mock_pipeline.run.return_value = result
        write_mock = MagicMock()
        monkeypatch.setattr("scripts.run_parser_pipeline.write_review_result_contract", write_mock)
        root = tmp_path / "run_root"

        run_parser_pipeline_cli(
            tmp_path,
            DEFAULT_OUTPUT_DIR,
            pipeline=mock_pipeline,
            persist_review_reports=True,
            output_root=root,
        )

        assert write_mock.call_count == 1
        assert write_mock.call_args.args == ("alice", "alice", result, root)
        assert f"Review reports directory: {root / 'review_reports'}" in capsys.readouterr().out

    def test_without_output_root_review_reports_still_use_the_fixed_root(
        self, tmp_path, mock_pipeline, monkeypatch
    ):
        (tmp_path / "alice.md").write_text("# Alice", encoding="utf-8")
        mock_pipeline.run.return_value = _fake_result()
        write_mock = MagicMock()
        monkeypatch.setattr("scripts.run_parser_pipeline.write_review_result_contract", write_mock)

        run_parser_pipeline_cli(
            tmp_path, DEFAULT_OUTPUT_DIR, pipeline=mock_pipeline, persist_review_reports=True
        )

        assert write_mock.call_args.args[3] == OUTPUT_DIR
