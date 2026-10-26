"""
tests/test_d8_strict_excluding_optional_keys.py

Decision D8 (2026-10-17): every benchmark report shows (1) the raw strict
accuracy -- comparator rules and historical meaning unchanged -- and
(2) "strict accuracy excluding optional keys", a separately named
diagnostic that re-counts the same outcomes with optional top-level
schema keys (certifications, projects) left out. These tests cover
optional-key handling, genuine strict mismatches, backward compatibility
of every pre-D8 output, report/run identification, and a regression
check against the two frozen HEAD baselines.
"""
import csv
import glob
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from config import BASE_DIR
from evaluation.parser_evaluator import ParserEvaluationResult, ParserEvaluator
from pipeline.parser_pipeline import ParserPipeline
from scripts.run_parser_evaluation import print_evaluation_result
from validators.json_comparator import (
    COMPARATOR_VERSION,
    ComparisonResult,
    JsonComparator,
    load_optional_top_level_keys,
    strict_excluding_optional_keys,
)

GT = {"skills": ["Python"], "languages": ["English"], "professional_summary": "x"}


def _compare(expected, actual):
    return JsonComparator().compare(expected, actual)


class TestOptionalKeysFromSchema:
    def test_real_schema_declares_exactly_certifications_and_projects_optional(self):
        assert load_optional_top_level_keys() == ["certifications", "projects"]

    def test_reads_from_the_given_schema_file(self, tmp_path):
        schema = {"properties": {"a": {}, "b": {}, "c": {}}, "required": ["a"]}
        p = tmp_path / "s.json"
        p.write_text(json.dumps(schema), encoding="utf-8")
        assert load_optional_top_level_keys(p) == ["b", "c"]

    def test_comparator_version_is_the_d8_one(self):
        assert COMPARATOR_VERSION == "1.1-d8"


class TestStrictExcludingOptionalKeys:
    def test_extra_optional_key_is_excluded_and_raw_is_untouched(self):
        result = _compare(GT, {**GT, "projects": []})
        # Raw strict (unchanged comparator): projects is EXTRA -> 3/4.
        assert (result.total_fields, result.matched_fields, result.extra_fields) == (4, 3, 1)
        assert result.accuracy == pytest.approx(3 / 4)
        ex = strict_excluding_optional_keys(result, ["certifications", "projects"])
        assert (ex.total_fields, ex.matched_fields, ex.excluded_fields) == (3, 3, 1)
        assert ex.accuracy == pytest.approx(1.0)
        assert ex.optional_keys == ["certifications", "projects"]

    def test_genuine_required_key_mismatch_counts_in_both_metrics(self):
        result = _compare(GT, {**GT, "skills": ["Rust"], "projects": []})
        assert result.accuracy == pytest.approx(2 / 4)  # skills FAIL + projects EXTRA
        ex = strict_excluding_optional_keys(result, ["projects"])
        assert (ex.total_fields, ex.matched_fields) == (3, 2)  # skills still FAIL
        assert ex.accuracy == pytest.approx(2 / 3)

    def test_optional_key_present_and_matching_in_both_is_credited_raw_but_excluded_diag(self):
        gt = {**GT, "projects": [{"name": "P"}]}
        result = _compare(gt, gt)
        assert result.accuracy == pytest.approx(1.0)
        ex = strict_excluding_optional_keys(result, ["projects"])
        assert ex.excluded_fields == 1
        assert (ex.total_fields, ex.matched_fields) == (3, 3)

    def test_optional_key_present_and_wrong_in_both_is_penalised_raw_but_excluded_diag(self):
        gt = {**GT, "projects": [{"name": "P"}]}
        result = _compare(gt, {**GT, "projects": [{"name": "Q"}]})
        assert result.accuracy == pytest.approx(3 / 4)
        ex = strict_excluding_optional_keys(result, ["projects"])
        assert ex.accuracy == pytest.approx(1.0)
        assert ex.excluded_fields == 1

    def test_missing_optional_key_is_also_excluded(self):
        gt = {**GT, "certifications": []}
        result = _compare(gt, GT)
        assert result.missing_fields == 1
        ex = strict_excluding_optional_keys(result, ["certifications"])
        assert (ex.total_fields, ex.matched_fields, ex.excluded_fields) == (3, 3, 1)

    def test_no_optional_keys_present_gives_identical_counts(self):
        result = _compare(GT, {**GT, "skills": ["Rust"]})
        ex = strict_excluding_optional_keys(result, ["certifications", "projects"])
        assert (ex.total_fields, ex.matched_fields, ex.excluded_fields) == (
            result.total_fields,
            result.matched_fields,
            0,
        )
        assert ex.accuracy == pytest.approx(result.accuracy)

    def test_only_top_level_segment_is_matched(self):
        """A nested field that merely contains the word 'projects' in a
        deeper segment is not an optional-key field."""
        gt = {"personal_information": {"projects": "ignored"}, "skills": []}
        result = _compare(gt, {**gt, "projects": []})
        ex = strict_excluding_optional_keys(result, ["projects"])
        assert ex.excluded_fields == 1  # only the top-level one
        assert ex.total_fields == 2

    def test_empty_result_is_safe(self):
        empty = ComparisonResult(0, 0, 0, 0, 1.0, [])
        ex = strict_excluding_optional_keys(empty, ["projects"])
        assert ex.accuracy == 1.0 and ex.total_fields == 0

    def test_default_keys_come_from_the_real_schema(self):
        result = _compare(GT, {**GT, "projects": [], "certifications": []})
        ex = strict_excluding_optional_keys(result)
        assert ex.optional_keys == ["certifications", "projects"]
        assert ex.excluded_fields == 2


class TestGenerateReportsBackwardCompatibility:
    def _batch(self):
        comparator = JsonComparator()
        return comparator, {
            "one": comparator.compare(GT, GT),
            "two": comparator.compare(GT, {**GT, "projects": [], "skills": ["Rust"]}),
        }

    def test_three_legacy_csvs_are_unchanged_and_a_fourth_is_added(self, tmp_path):
        comparator, comparisons = self._batch()
        comparator.generate_reports(comparisons, output_dir=tmp_path)
        with (tmp_path / "resume_accuracy.csv").open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == [
                "resume_name", "total_fields", "matched_fields",
                "missing_fields", "extra_fields", "accuracy",
            ]
            rows = {r["resume_name"]: r for r in reader}
        assert rows["two"]["extra_fields"] == "1" and rows["two"]["accuracy"] == "0.5000"
        with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as f:
            summary = next(csv.DictReader(f))
        assert summary["overall_accuracy"] == f"{5 / 7:.4f}"  # raw: 3/3 + 2/4
        with (tmp_path / "field_accuracy.csv").open(newline="", encoding="utf-8") as f:
            assert csv.DictReader(f).fieldnames == ["field_path", "total", "matched", "missing", "extra", "accuracy"]

        with (tmp_path / "resume_accuracy_excluding_optional_keys.csv").open(
            newline="", encoding="utf-8"
        ) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == [
                "resume_name",
                "total_fields_excluding_optional",
                "matched_fields_excluding_optional",
                "excluded_fields",
                "strict_accuracy_excluding_optional_keys",
                "optional_keys",
                "comparator_version",
            ]
            ex_rows = {r["resume_name"]: r for r in reader}
        assert ex_rows["two"]["excluded_fields"] == "1"
        assert ex_rows["two"]["strict_accuracy_excluding_optional_keys"] == f"{2 / 3:.4f}"
        assert ex_rows["__all__"]["strict_accuracy_excluding_optional_keys"] == f"{5 / 6:.4f}"
        assert ex_rows["__all__"]["optional_keys"] == "certifications|projects"
        assert ex_rows["__all__"]["comparator_version"] == COMPARATOR_VERSION


class TestEvaluatorResultAndReports:
    """ParserEvaluator against a tiny dataset where one parse emits an
    extra optional key: raw and diagnostic figures must differ exactly
    as D8 describes, and every report must identify comparator + GT."""

    @pytest.fixture
    def dataset(self, tmp_path):
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        md_dir = tmp_path / "md"
        md_dir.mkdir()
        base = {
            "personal_information": {"full_name": "A", "first_name": "A", "last_name": "B", "current_location": None},
            "contact_information": {"email": [], "phone": [], "linkedin": None, "github": None, "portfolio": None},
            "professional_summary": None, "skills": ["Python"], "languages": [],
            "work_experience": [], "education": [],
        }
        for stem in ("alice", "bob"):
            (md_dir / f"{stem}.md").write_text(f"# {stem}", encoding="utf-8")
            (gt_dir / f"{stem}.json").write_text(json.dumps(base), encoding="utf-8")
        with (gt_dir / "mapping.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["ResumeID", "PDF", "Markdown", "JSON"])
            w.writeheader()
            w.writerow({"ResumeID": "001", "PDF": "alice.pdf", "Markdown": "alice.md", "JSON": "alice.json"})
            w.writerow({"ResumeID": "002", "PDF": "bob.pdf", "Markdown": "bob.md", "JSON": "bob.json"})
        pipeline = MagicMock(spec=ParserPipeline)

        def _run(markdown_path):
            stem = Path(markdown_path).stem
            merged = dict(base)
            if stem == "bob":
                merged = {**base, "projects": []}  # extra optional key
            return SimpleNamespace(draft_json={}, module_results={}, merged_json=merged, processing_time=0.01)

        pipeline.run.side_effect = _run
        return SimpleNamespace(gt_dir=gt_dir, md_dir=md_dir, pipeline=pipeline, base=base)

    def test_raw_and_diagnostic_figures(self, dataset, tmp_path):
        evaluator = ParserEvaluator(
            parser_pipeline=dataset.pipeline, json_comparator=JsonComparator(),
            ground_truth_dir=dataset.gt_dir, markdown_dir=dataset.md_dir, output_dir=tmp_path / "out",
        )
        result = evaluator.evaluate()
        # alice: 14/14; bob: 14/15 raw (projects EXTRA) -> 28/29 raw, 28/28 diag.
        assert result.overall_accuracy == pytest.approx(28 / 29)
        assert result.overall_accuracy_excluding_optional_keys == pytest.approx(1.0)
        assert result.resume_accuracy["bob"] == pytest.approx(14 / 15)
        assert result.resume_accuracy_excluding_optional_keys["bob"] == pytest.approx(1.0)
        assert result.optional_keys_excluded == ["certifications", "projects"]
        assert result.comparator_version == COMPARATOR_VERSION
        assert result.ground_truth_dir == str(dataset.gt_dir)
        assert result.ground_truth_row_count == 2
        assert len(result.ground_truth_mapping_sha256) == 64

    def test_report_and_run_identification_files(self, dataset, tmp_path):
        out = tmp_path / "out"
        ParserEvaluator(
            parser_pipeline=dataset.pipeline, json_comparator=JsonComparator(),
            ground_truth_dir=dataset.gt_dir, markdown_dir=dataset.md_dir, output_dir=out,
        ).evaluate()
        report = (out / "benchmark_report.md").read_text(encoding="utf-8")
        assert "| Overall accuracy | 96.55% |" in report
        assert "| Strict accuracy excluding optional keys (diagnostic, D8) | 100.00% |" in report
        assert "## Run identification" in report
        assert f"| Comparator version | {COMPARATOR_VERSION} |" in report
        assert f"| Ground truth directory | {dataset.gt_dir} |" in report
        assert "| Ground truth mapping rows | 2 |" in report
        assert "| Resume | Accuracy | Excluding optional keys (D8) |" in report
        assert "| bob | 93.33% | 100.00% |" in report
        ident = json.loads((out / "run_identification.json").read_text(encoding="utf-8"))
        assert ident["comparator_version"] == COMPARATOR_VERSION
        assert ident["ground_truth_row_count"] == 2
        assert ident["optional_keys_excluded"] == ["certifications", "projects"]
        assert ident["strict_accuracy_raw"] == pytest.approx(28 / 29)
        assert ident["strict_accuracy_excluding_optional_keys"] == pytest.approx(1.0)
        assert (out / "resume_accuracy_excluding_optional_keys.csv").is_file()

    def test_pre_d8_result_construction_still_works(self):
        """Every pre-D8 caller builds ParserEvaluationResult without the
        new fields; they must default sensibly."""
        r = ParserEvaluationResult(total_resumes=1, successful_resumes=1, failed_resumes=0, overall_accuracy=0.5)
        assert r.overall_accuracy_excluding_optional_keys == 0.0
        assert r.resume_accuracy_excluding_optional_keys == {}
        assert r.optional_keys_excluded == []
        assert r.comparator_version == COMPARATOR_VERSION
        assert r.ground_truth_dir == "" and r.ground_truth_row_count == 0

    def test_cli_summary_prints_both_figures_and_comparator_version(self, capsys):
        r = ParserEvaluationResult(
            total_resumes=1, successful_resumes=1, failed_resumes=0, overall_accuracy=0.5,
            overall_accuracy_excluding_optional_keys=0.75, optional_keys_excluded=["projects"],
        )
        print_evaluation_result(r, Path("/x"))
        out = capsys.readouterr().out
        assert "Overall accuracy: 50.00%" in out
        assert "Strict accuracy excluding optional keys (diagnostic, D8): 75.00% [excluded: projects]" in out
        assert f"Comparator version: {COMPARATOR_VERSION}" in out


class TestFrozenHeadBaselinesRescored:
    """Regression guard against the two frozen, git-tracked HEAD baselines
    (benchmark/baselines/head_2c75781_pass{1,2}/checkpoints/*.pkl). The
    numbers below are the ones published in their READMEs and in the
    remediation report §7; this test re-derives them from the pickled
    ComparisonResults with the D8 helper. Raw figures must stay what the
    unchanged comparator produced; diagnostic figures must equal the
    hand-computed 'excluding projects' values."""

    @staticmethod
    def _rescore(run_dir):
        raw_m = raw_t = ex_m = ex_t = excluded = n = 0
        for pkl in sorted(glob.glob(str(run_dir / "checkpoints" / "*.pkl"))):
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            if data["status"] != "ok":
                continue
            c = data["comparison"]
            ex = strict_excluding_optional_keys(c)
            raw_m += c.matched_fields
            raw_t += c.total_fields
            ex_m += ex.matched_fields
            ex_t += ex.total_fields
            excluded += ex.excluded_fields
            n += 1
        return n, raw_m, raw_t, ex_m, ex_t, excluded

    @pytest.mark.parametrize(
        "run_dir, expected",
        [
            ("head_2c75781_pass1", (21, 207, 315, 207, 294, 21)),
            ("head_2c75781_pass2", (22, 211, 330, 211, 308, 22)),
        ],
    )
    def test_published_numbers_are_reproduced(self, run_dir, expected):
        path = BASE_DIR / "benchmark" / "baselines" / run_dir
        assert path.is_dir(), f"frozen baseline missing: {path}"
        n, raw_m, raw_t, ex_m, ex_t, excluded = self._rescore(path)
        assert (n, raw_m, raw_t, ex_m, ex_t, excluded) == expected
        assert raw_m / raw_t == pytest.approx({"head_2c75781_pass1": 0.6571, "head_2c75781_pass2": 0.6394}[run_dir], abs=5e-5)
        assert ex_m / ex_t == pytest.approx({"head_2c75781_pass1": 0.7041, "head_2c75781_pass2": 0.6851}[run_dir], abs=5e-5)
