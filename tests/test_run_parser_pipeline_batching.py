"""
tests/test_run_parser_pipeline_batching.py

Tests for the --batch-size behavior added to scripts/run_parser_pipeline
.py in Milestone 14.9.4.

ParserPipeline is always mocked — no test here invokes the script's real
defaults. These focus specifically on batching mechanics: splitting into
the right number/size of batches, completing one batch before starting
the next, continuing past failures within and across batches, and
printing "Batch N/M" progress plus per-batch statistics.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pipeline.parser_pipeline import ParserPipeline
from scripts.run_parser_pipeline import _batchify, _parse_args, run_parser_pipeline_cli


def _fake_result(processing_time: float = 0.1):
    return SimpleNamespace(
        draft_json={}, module_results={}, merged_json={}, processing_time=processing_time
    )


@pytest.fixture
def mock_pipeline():
    return MagicMock(spec=ParserPipeline)


def _make_markdown_files(tmp_path, names):
    for name in names:
        (tmp_path / f"{name}.md").write_text(f"# {name}", encoding="utf-8")


class TestBatchify:
    def test_splits_into_equal_batches(self):
        paths = [Path(f"{i}.md") for i in range(6)]

        batches = _batchify(paths, batch_size=2)

        assert batches == [paths[0:2], paths[2:4], paths[4:6]]

    def test_last_batch_may_be_smaller(self):
        paths = [Path(f"{i}.md") for i in range(5)]

        batches = _batchify(paths, batch_size=2)

        assert [len(b) for b in batches] == [2, 2, 1]

    def test_none_batch_size_returns_one_batch(self):
        paths = [Path(f"{i}.md") for i in range(5)]

        batches = _batchify(paths, batch_size=None)

        assert batches == [paths]

    def test_non_positive_batch_size_returns_one_batch(self):
        paths = [Path(f"{i}.md") for i in range(3)]

        assert _batchify(paths, batch_size=0) == [paths]
        assert _batchify(paths, batch_size=-1) == [paths]

    def test_empty_input_returns_no_batches(self):
        assert _batchify([], batch_size=3) == []

    def test_batch_size_larger_than_input_returns_one_batch(self):
        paths = [Path(f"{i}.md") for i in range(3)]

        assert _batchify(paths, batch_size=100) == [paths]


class TestRunParserPipelineCliBatching:
    def test_completes_one_batch_before_starting_next(
        self, tmp_path, mock_pipeline, capsys
    ):
        """Every resume in batch 1 must be processed before any resume
        in batch 2 starts."""
        _make_markdown_files(tmp_path, ["a", "b", "c", "d"])
        call_order = []

        def _run(markdown_path):
            call_order.append(Path(markdown_path).stem)
            return _fake_result()

        mock_pipeline.run.side_effect = _run

        run_parser_pipeline_cli(tmp_path, tmp_path, pipeline=mock_pipeline, batch_size=2)

        assert call_order == ["a", "b", "c", "d"]
        assert mock_pipeline.run.call_count == 4

    def test_displays_batch_progress_headers(self, tmp_path, mock_pipeline, capsys):
        _make_markdown_files(tmp_path, ["a", "b", "c", "d", "e"])
        mock_pipeline.run.return_value = _fake_result()

        run_parser_pipeline_cli(tmp_path, tmp_path, pipeline=mock_pipeline, batch_size=2)

        captured = capsys.readouterr()
        assert "Batch 1/3" in captured.out
        assert "Batch 2/3" in captured.out
        assert "Batch 3/3" in captured.out

    def test_prints_statistics_after_every_batch(self, tmp_path, mock_pipeline, capsys):
        _make_markdown_files(tmp_path, ["a", "b", "c", "d"])
        mock_pipeline.run.return_value = _fake_result()

        run_parser_pipeline_cli(tmp_path, tmp_path, pipeline=mock_pipeline, batch_size=2)

        captured = capsys.readouterr()
        assert "Batch 1/2 summary" in captured.out
        assert "Batch 2/2 summary" in captured.out
        # Each batch of 2 should report its own total of 2.
        assert captured.out.count("Total:            2") == 2

    def test_continues_past_failures_within_and_across_batches(
        self, tmp_path, mock_pipeline, capsys
    ):
        _make_markdown_files(tmp_path, ["a", "b", "c", "d"])

        def _run(markdown_path):
            if Path(markdown_path).stem in ("b", "d"):
                raise RuntimeError("model exploded")
            return _fake_result()

        mock_pipeline.run.side_effect = _run

        run_parser_pipeline_cli(tmp_path, tmp_path, pipeline=mock_pipeline, batch_size=2)

        # a, b, c, d initial attempts (4) + b's and d's ResumeRetryQueue
        # attempts (2 more) — RuntimeError passes straight through
        # ResumeRetryQueue without an internal second retry (it's not
        # one of ResumeRetryQueue's own retryable exception types).
        assert mock_pipeline.run.call_count == 6
        captured = capsys.readouterr()
        assert "Success:          1" in captured.out  # once per batch
        assert "Failed:           1" in captured.out
        # Overall summary at the end still reflects the grand total.
        assert "Total:            4" in captured.out
        assert "Success:          2" in captured.out
        assert "Failed:           2" in captured.out
        # Both b and d were queued and retried, but RuntimeError isn't
        # retryable by ResumeRetryQueue, so both remain permanent failures.
        assert "Initial successes:    2" in captured.out
        assert "Retried successes:    0" in captured.out
        assert "Permanent failures:   2" in captured.out

    def test_no_batch_size_processes_everything_as_one_batch(
        self, tmp_path, mock_pipeline, capsys
    ):
        _make_markdown_files(tmp_path, ["a", "b", "c"])
        mock_pipeline.run.return_value = _fake_result()

        run_parser_pipeline_cli(tmp_path, tmp_path, pipeline=mock_pipeline)

        captured = capsys.readouterr()
        assert "Batch 1/1" in captured.out
        assert mock_pipeline.run.call_count == 3

    def test_final_summary_reflects_grand_total_across_all_batches(
        self, tmp_path, mock_pipeline, capsys
    ):
        _make_markdown_files(tmp_path, ["a", "b", "c", "d", "e"])
        mock_pipeline.run.side_effect = [
            _fake_result(1.0),
            _fake_result(2.0),
            _fake_result(3.0),
            _fake_result(4.0),
            _fake_result(5.0),
        ]

        run_parser_pipeline_cli(tmp_path, tmp_path, pipeline=mock_pipeline, batch_size=2)

        captured = capsys.readouterr()
        assert "Parser pipeline summary" in captured.out
        assert "Total:            5" in captured.out
        assert "Success:          5" in captured.out
        assert "Average time:     3.0000s" in captured.out  # mean of 1..5

    def test_empty_input_dir_with_batch_size_prints_zero_summary(
        self, tmp_path, mock_pipeline, capsys
    ):
        run_parser_pipeline_cli(tmp_path, tmp_path, pipeline=mock_pipeline, batch_size=5)

        mock_pipeline.run.assert_not_called()
        captured = capsys.readouterr()
        assert "Total:            0" in captured.out
        assert "Batch" not in captured.out  # no files -> no batches at all


class TestParseArgsBatchSize:
    def test_defaults_to_none(self):
        args = _parse_args([])
        assert args.batch_size is None

    def test_accepts_batch_size(self):
        args = _parse_args(["--batch-size", "10"])
        assert args.batch_size == 10

    def test_rejects_non_integer_batch_size(self):
        with pytest.raises(SystemExit):
            _parse_args(["--batch-size", "not-a-number"])
