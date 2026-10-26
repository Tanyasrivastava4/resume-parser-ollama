"""
validators/json_comparator.py

JSON Comparator (Milestone 13.3).

Compares a ground-truth resume JSON against a generated one, field by
field, and reports how accurate the generated JSON is. This is an
accuracy measurement tool: it does not judge schema conformance (that's
validators/schema_validator.py) and does not modify either JSON — it only
reads them and reports differences.

Nested objects are flattened into dot-notation field paths for
comparison (e.g. "personal_information.full_name"). Arrays are compared
as unordered sets of their elements, so reordering "skills" doesn't count
as a difference, but adding, removing, or changing an element does.
"""

import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import COMPARISON_DIR, SCHEMAS_DIR

logger = logging.getLogger(__name__)

# Field comparison outcomes.
# Decision D8 (2026-10-17). Identifies the scoring rules a report was produced
# with. "1.0" is every comparator behaviour that existed before D8 -- compare(),
# compare_diagnostic() and the three CSVs are unchanged by D8 and still mean
# exactly what every historical report means. "1.1-d8" adds, additively, the
# diagnostic metric strict_excluding_optional_keys() and a fourth CSV. Bump
# this whenever a scoring rule changes so no two reports can be compared
# without noticing.
COMPARATOR_VERSION = "1.1-d8"

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_MISSING = "MISSING"
STATUS_EXTRA = "EXTRA"

# Entry-level outcomes used only by the diagnostic mode (Milestone 20.3,
# see compare_diagnostic() below). Named distinctly from STATUS_* above —
# these describe whether an array *entry* (e.g. one work_experience job)
# was found at all and how well it matched, not whether a scalar field
# passed — so they are never compared against or confused with STATUS_PASS
# et al.
ENTRY_STATUS_FULL_MATCH = "FULL_MATCH"
ENTRY_STATUS_PARTIAL_MATCH = "PARTIAL_MATCH"
ENTRY_STATUS_MISSING = "MISSING_ENTRY"
ENTRY_STATUS_EXTRA = "EXTRA_ENTRY"


@dataclass
class FieldComparison:
    """The comparison outcome for a single field path.

    Attributes:
        field_path: Dot-notation path to the field, e.g.
            "personal_information.full_name", or "skills" for a
            top-level array field.
        expected: The ground-truth value at this path (None if the field
            doesn't exist in the ground truth — an EXTRA field).
        actual: The generated value at this path (None if the field
            doesn't exist in the generated JSON — a MISSING field).
        status: One of PASS, FAIL, MISSING, EXTRA.
    """

    field_path: str
    expected: Any
    actual: Any
    status: str


@dataclass
class ComparisonResult:
    """Aggregate result of comparing one ground-truth/generated JSON pair.

    Attributes:
        total_fields: Total number of field paths compared (the union of
            every field path present in either JSON).
        matched_fields: Number of fields with status PASS.
        missing_fields: Number of fields present in the ground truth but
            absent from the generated JSON (status MISSING).
        extra_fields: Number of fields present in the generated JSON but
            absent from the ground truth (status EXTRA).
        accuracy: matched_fields / total_fields (1.0 if there were no
            fields to compare at all).
        field_results: Every field's individual FieldComparison, in the
            order fields were visited.
    """

    total_fields: int
    matched_fields: int
    missing_fields: int
    extra_fields: int
    accuracy: float
    field_results: List[FieldComparison] = field(default_factory=list)


@dataclass
class OptionalKeyExclusion:
    """Decision D8 diagnostic: a ComparisonResult re-counted with every
    top-level field under an OPTIONAL schema key left out.

    Attributes:
        total_fields / matched_fields: the raw ComparisonResult counts
            minus the excluded fields.
        excluded_fields: how many FieldComparison rows were left out.
        optional_keys: the top-level keys treated as optional (sorted).
        accuracy: matched_fields / total_fields (1.0 if nothing left).
    """

    total_fields: int
    matched_fields: int
    excluded_fields: int
    optional_keys: List[str]
    accuracy: float


def load_optional_top_level_keys(
    schema_path: Path = SCHEMAS_DIR / "resume_schema_v1.json",
) -> List[str]:
    """Top-level keys the final resume schema declares but does not
    require -- today `certifications` and `projects` (Milestone T3). Read
    from the schema file so the list cannot drift from the contract."""
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    properties = set(schema.get("properties", {}))
    required = set(schema.get("required", []))
    return sorted(properties - required)


def strict_excluding_optional_keys(
    result: ComparisonResult, optional_keys: Optional[List[str]] = None
) -> OptionalKeyExclusion:
    """Decision D8 (2026-10-17): the SAME strict field outcomes as
    `result.field_results`, re-counted with every field whose top-level
    key is optional in the schema left out of both numerator and
    denominator.

    Why: since Milestone T5 the parser emits a top-level `projects`
    array. The adjudicated ground truth (Milestone 23.66) has no
    `projects` key, so compare() -- correctly, by its own rules --
    scores it as one EXTRA field on every resume, lowering strict
    accuracy by about 1/15 per resume irrespective of parser quality.
    This helper does NOT change compare(): the raw strict figure keeps
    its historical meaning, and this is reported beside it as a
    separately named diagnostic. It excludes optional keys whether they
    are EXTRA, MISSING, PASS or FAIL, so the figure stays comparable
    once `projects` ground truth exists (planned with corpus_v2 GT
    authoring); at that point the raw strict figure is the one that
    credits/penalises `projects`.
    """
    keys = sorted(optional_keys) if optional_keys is not None else load_optional_top_level_keys()
    key_set = set(keys)
    kept = [fr for fr in result.field_results if fr.field_path.split(".", 1)[0] not in key_set]
    excluded = len(result.field_results) - len(kept)
    matched = sum(1 for fr in kept if fr.status == STATUS_PASS)
    total = len(kept)
    return OptionalKeyExclusion(
        total_fields=total,
        matched_fields=matched,
        excluded_fields=excluded,
        optional_keys=keys,
        accuracy=(matched / total) if total else 1.0,
    )


@dataclass
class EntryFieldComparison:
    """Diagnostic-mode-only: one sub-field's outcome within a matched array
    entry (e.g. "employment_type" within one work_experience entry).

    Unlike FieldComparison, `field_name` is just the key within the entry
    (not a dot-notation path from the document root) — the entry it belongs
    to is identified by the EntryDiagnostic that holds it.
    """

    field_name: str
    expected: Any
    actual: Any
    matched: bool


@dataclass
class EntryDiagnostic:
    """Diagnostic-mode-only: the outcome for one array entry.

    Attributes:
        expected_index: This entry's index in the ground-truth array, or
            None if it has no counterpart there (status EXTRA_ENTRY).
        actual_index: This entry's index in the generated array, or None
            if nothing in the generated array was matched to it (status
            MISSING_ENTRY).
        status: One of ENTRY_STATUS_FULL_MATCH (every sub-field matched),
            ENTRY_STATUS_PARTIAL_MATCH (aligned to a counterpart, but at
            least one sub-field differs), ENTRY_STATUS_MISSING (a
            ground-truth entry with no matching generated entry), or
            ENTRY_STATUS_EXTRA (a generated entry with no matching
            ground-truth entry).
        match_score: Fraction of sub-fields that matched (0.0 for MISSING
            and EXTRA, since there is no counterpart to score against).
        field_comparisons: Per-sub-field detail, empty for MISSING/EXTRA.
    """

    expected_index: Optional[int]
    actual_index: Optional[int]
    status: str
    match_score: float
    field_comparisons: List[EntryFieldComparison] = field(default_factory=list)


@dataclass
class ArrayFieldDiagnostic:
    """Diagnostic-mode-only: the full entry-matching breakdown for one
    array-of-objects field (e.g. "work_experience").

    Attributes:
        field_path: The field's name (top-level only today; see
            compare_diagnostic()'s docstring for why).
        expected_count: Number of entries in the ground-truth array.
        actual_count: Number of entries in the generated array.
        entries: Every EntryDiagnostic — matched entries (full or
            partial) first in ground-truth order, then MISSING entries,
            then EXTRA entries.
        entry_accuracy: Fraction of ground-truth entries that were found
            at all (status FULL_MATCH or PARTIAL_MATCH), regardless of
            how many of their sub-fields were right. 1.0 if
            expected_count is 0.
        field_accuracy: Fraction of sub-fields, summed only over matched
            (full or partial) entries, that were correct. 1.0 if there
            were no matched entries to check (nothing to be wrong about).
    """

    field_path: str
    expected_count: int
    actual_count: int
    entries: List[EntryDiagnostic] = field(default_factory=list)
    entry_accuracy: float = 1.0
    field_accuracy: float = 1.0


@dataclass
class DiagnosticResult:
    """Diagnostic-mode-only: the full diagnostic breakdown for one
    ground-truth/generated JSON pair (Milestone 20.3).

    This is entirely additive to, and never derived by modifying,
    ComparisonResult — `strict_result` embeds the exact, unmodified output
    of JsonComparator.compare() so the strict score this carries is
    guaranteed identical to what a non-diagnostic caller would get for the
    same two documents. See docs/JSON_COMPARATOR_DIAGNOSTIC.md.

    Attributes:
        strict_accuracy: Same value as strict_result.accuracy — repeated
            here for convenience so callers of compare_diagnostic() don't
            need to reach into strict_result for the headline number.
        entry_accuracy: Entry-level accuracy aggregated across every
            array-of-objects field (see ArrayFieldDiagnostic.entry_accuracy),
            weighted by each field's own ground-truth entry count. 1.0 if
            there were no array-of-objects fields to check.
        field_accuracy: Sub-field-level accuracy aggregated across every
            array-of-objects field's matched entries, weighted by
            sub-field count. 1.0 if there were no matched entries.
        array_diagnostics: Per-array-field breakdown, keyed by field name.
        mismatch_by_field: Dot-notation "<array_field>.<sub_field>" ->
            {"total": int, "matched": int} counts, aggregated only over
            matched entries — the diagnostic-mode equivalent of
            field_accuracy.csv, but at array-sub-field granularity.
        strict_result: The unmodified ComparisonResult from compare().
    """

    strict_accuracy: float
    entry_accuracy: float
    field_accuracy: float
    array_diagnostics: Dict[str, ArrayFieldDiagnostic] = field(default_factory=dict)
    mismatch_by_field: Dict[str, Dict[str, int]] = field(default_factory=dict)
    strict_result: Optional[ComparisonResult] = None


class JsonComparator:
    """Compares ground-truth vs. generated resume JSON, field by field."""

    def compare(self, expected_json: Any, actual_json: Any) -> ComparisonResult:
        """Compare `expected_json` (ground truth) against `actual_json`
        (generated), field by field.

        Comparison rules:
            - null == null is a PASS.
            - Arrays are compared as unordered sets of their elements.
            - Nested objects are compared recursively.
            - A field present only in `expected_json` is MISSING (FAIL).
            - A field present only in `actual_json` is EXTRA (FAIL).
            - A field present in both with a different value is FAIL.

        Args:
            expected_json: The ground-truth JSON value (typically a dict).
            actual_json: The generated JSON value to check against it.

        Returns:
            A ComparisonResult summarizing every field's outcome.
        """
        field_results: List[FieldComparison] = []
        self._compare_value("", expected_json, actual_json, field_results)

        matched = sum(1 for r in field_results if r.status == STATUS_PASS)
        missing = sum(1 for r in field_results if r.status == STATUS_MISSING)
        extra = sum(1 for r in field_results if r.status == STATUS_EXTRA)
        total = len(field_results)
        accuracy = (matched / total) if total else 1.0

        return ComparisonResult(
            total_fields=total,
            matched_fields=matched,
            missing_fields=missing,
            extra_fields=extra,
            accuracy=accuracy,
            field_results=field_results,
        )

    def generate_reports(
        self,
        comparisons: Dict[str, ComparisonResult],
        output_dir: Path = COMPARISON_DIR,
    ) -> None:
        """Write field_accuracy.csv, resume_accuracy.csv, and summary.csv
        for a batch of comparisons.

        Args:
            comparisons: Mapping of resume name to its ComparisonResult
                (typically one entry per resume compared in a benchmark
                run).
            output_dir: Directory to write the three CSVs into. Defaults
                to benchmark/comparison/.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._write_resume_accuracy_csv(comparisons, output_dir / "resume_accuracy.csv")
        self._write_field_accuracy_csv(comparisons, output_dir / "field_accuracy.csv")
        self._write_summary_csv(comparisons, output_dir / "summary.csv")
        # Decision D8: a FOURTH, additive file. The three above are
        # byte-for-byte what they were before D8.
        self._write_resume_accuracy_excluding_optional_keys_csv(
            comparisons, output_dir / "resume_accuracy_excluding_optional_keys.csv"
        )
        logger.info("Wrote comparison reports to %s", output_dir)

    def compare_diagnostic(self, expected_json: Any, actual_json: Any) -> DiagnosticResult:
        """Diagnostic-mode comparison (Milestone 20.3).

        This is a separate, additive analysis layered on top of compare()
        — it never changes, and is never used to compute, the strict
        result. It exists because compare()'s array scoring
        (_compare_arrays: a whole array is one PASS/FAIL field, see that
        method's docstring) makes a single wrong sub-field on one entry
        indistinguishable from every entry being completely wrong, which
        is accurate as a strict pass/fail signal but not diagnostically
        useful (see docs/JSON_COMPARATOR_AUDIT.md, which this milestone
        implements the recommendations of).

        For every top-level field that is an array of objects in either
        document (e.g. "work_experience", "education" — arrays of
        scalars like "skills"/"languages" have nothing for this mode to
        add over compare()'s own whole-array result, so they're left
        alone), this aligns each ground-truth entry to its best-matching
        generated entry (by fraction of identical sub-fields — no
        assumption is made about which sub-field is an "identifying" one
        like `company`; see _entry_similarity()), then scores each
        matched pair's sub-fields individually. Ground-truth entries with
        no acceptable match (similarity 0) are MISSING; generated entries
        left over are EXTRA.

        Only top-level array-of-object fields are given this treatment —
        the real schema (resume_schema_v1.json) never nests one array of
        objects inside another entry's fields, so this covers every case
        that exists today without needing a general recursive matcher.

        Args:
            expected_json: The ground-truth JSON value (typically a dict).
            actual_json: The generated JSON value to check against it.

        Returns:
            A DiagnosticResult. Its strict_result/strict_accuracy are
            compare()'s own unmodified output.
        """
        strict_result = self.compare(expected_json, actual_json)

        array_diagnostics: Dict[str, ArrayFieldDiagnostic] = {}
        if isinstance(expected_json, dict) and isinstance(actual_json, dict):
            for key in sorted(set(expected_json) | set(actual_json)):
                expected_value = expected_json.get(key)
                actual_value = actual_json.get(key)
                if not isinstance(expected_value, list) or not isinstance(actual_value, list):
                    continue
                if self._is_object_array(expected_value) or self._is_object_array(actual_value):
                    array_diagnostics[key] = self._diagnose_array(key, expected_value, actual_value)

        entry_accuracy = self._aggregate_entry_accuracy(array_diagnostics)
        field_accuracy = self._aggregate_field_accuracy_diag(array_diagnostics)
        mismatch_by_field = self._aggregate_mismatch_by_field(array_diagnostics)

        return DiagnosticResult(
            strict_accuracy=strict_result.accuracy,
            entry_accuracy=entry_accuracy,
            field_accuracy=field_accuracy,
            array_diagnostics=array_diagnostics,
            mismatch_by_field=mismatch_by_field,
            strict_result=strict_result,
        )

    def generate_diagnostic_reports(
        self,
        diagnostics: Dict[str, DiagnosticResult],
        output_dir: Path = COMPARISON_DIR,
    ) -> None:
        """Write diagnostic_summary.csv, diagnostic_mismatch_by_field.csv,
        and diagnostic_report.md for a batch of diagnostic comparisons.

        Entirely additive: these are new filenames alongside (never
        replacing) generate_reports()'s resume_accuracy.csv/
        field_accuracy.csv/summary.csv, so calling both against the same
        output_dir leaves every existing file byte-for-byte as
        generate_reports() alone would have written it.

        Args:
            diagnostics: Mapping of resume name to its DiagnosticResult.
            output_dir: Directory to write the three files into. Defaults
                to benchmark/comparison/.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._write_diagnostic_summary_csv(diagnostics, output_dir / "diagnostic_summary.csv")
        self._write_diagnostic_mismatch_csv(
            diagnostics, output_dir / "diagnostic_mismatch_by_field.csv"
        )
        self._write_diagnostic_report_md(diagnostics, output_dir / "diagnostic_report.md")
        logger.info("Wrote diagnostic reports to %s", output_dir)

    def _compare_value(
        self,
        field_path: str,
        expected: Any,
        actual: Any,
        results: List[FieldComparison],
    ) -> None:
        """Dispatch a single value comparison to the right strategy based
        on type, appending outcome(s) to `results`."""
        if isinstance(expected, dict) and isinstance(actual, dict):
            self._compare_objects(field_path, expected, actual, results)
        elif isinstance(expected, list) and isinstance(actual, list):
            self._compare_arrays(field_path, expected, actual, results)
        else:
            status = STATUS_PASS if expected == actual else STATUS_FAIL
            results.append(FieldComparison(field_path, expected, actual, status))

    def _compare_objects(
        self,
        prefix: str,
        expected: dict,
        actual: dict,
        results: List[FieldComparison],
    ) -> None:
        """Recursively compare two dicts, one field per key in their union."""
        all_keys = sorted(set(expected.keys()) | set(actual.keys()))
        for key in all_keys:
            field_path = f"{prefix}.{key}" if prefix else key
            if key not in actual:
                results.append(
                    FieldComparison(field_path, expected[key], None, STATUS_MISSING)
                )
            elif key not in expected:
                results.append(
                    FieldComparison(field_path, None, actual[key], STATUS_EXTRA)
                )
            else:
                self._compare_value(field_path, expected[key], actual[key], results)

    @staticmethod
    def _compare_arrays(
        field_path: str,
        expected: list,
        actual: list,
        results: List[FieldComparison],
    ) -> None:
        """Compare two arrays as unordered sets of their elements."""

        def _as_set(items: list) -> frozenset:
            return frozenset(json.dumps(item, sort_keys=True, default=str) for item in items)

        status = STATUS_PASS if _as_set(expected) == _as_set(actual) else STATUS_FAIL
        results.append(FieldComparison(field_path, expected, actual, status))

    # -- Diagnostic mode (Milestone 20.3) -----------------------------
    #
    # Everything below this point is used only by compare_diagnostic() /
    # generate_diagnostic_reports() above. None of it is called by, or
    # able to affect, compare() / generate_reports() / _compare_arrays()
    # above, which is how "diagnostic mode must never affect the existing
    # benchmark score" (Milestone 20.3, requirement 3) is guaranteed —
    # not by convention, but because there is no shared code path.

    @staticmethod
    def _is_object_array(items: list) -> bool:
        """True if at least one element of `items` is a dict — i.e. this
        looks like an array-of-objects field (work_experience, education)
        rather than an array of scalars (skills, languages). An empty
        list, or a list that turns out to hold only scalars, is not an
        object array; diagnostic entry-matching has nothing useful to add
        there over compare()'s own whole-array result."""
        return any(isinstance(item, dict) for item in items)

    @staticmethod
    def _diagnostic_values_equal(expected: Any, actual: Any) -> bool:
        """Equality used only for diagnostic sub-field scoring.

        Behaves like compare()'s own rules (null == null passes, dicts
        recurse) with one deliberate difference: list-valued sub-fields
        (e.g. a work_experience entry's `technologies`/`responsibilities`)
        are compared as unordered sets at every nesting level, not just
        at the top of the document the way _compare_arrays does. This
        closes the ordering gap documented in
        docs/JSON_COMPARATOR_AUDIT.md section 1 — a real but separate
        finding from the whole-array-as-one-field issue this milestone's
        entry-matching addresses. It only affects diagnostic-mode
        scoring; compare()'s own order-sensitivity for nested lists is
        unchanged.
        """
        if isinstance(expected, list) and isinstance(actual, list):
            dumped_expected = frozenset(
                json.dumps(item, sort_keys=True, default=str) for item in expected
            )
            dumped_actual = frozenset(
                json.dumps(item, sort_keys=True, default=str) for item in actual
            )
            return dumped_expected == dumped_actual
        if isinstance(expected, dict) and isinstance(actual, dict):
            keys = set(expected.keys()) | set(actual.keys())
            return all(
                JsonComparator._diagnostic_values_equal(expected.get(k), actual.get(k))
                for k in keys
            )
        return expected == actual

    @staticmethod
    def _entry_similarity(expected_entry: Any, actual_entry: Any) -> float:
        """Fraction of sub-fields two array entries agree on, used to
        decide which generated entry best matches which ground-truth
        entry. No sub-field is treated as a special "identifying" key
        (e.g. `company`) — every key contributes equally, so an entry
        with the right company but every other field wrong scores the
        same as it would matching on any other single field, and an
        entry that's merely missing its company but right on everything
        else is not penalized for that alone.

        Non-dict entries (should not occur for a real work_experience/
        education array, but not assumed impossible) fall back to plain
        equality: 1.0 if identical, 0.0 otherwise.
        """
        if not isinstance(expected_entry, dict) or not isinstance(actual_entry, dict):
            return 1.0 if expected_entry == actual_entry else 0.0
        keys = set(expected_entry.keys()) | set(actual_entry.keys())
        if not keys:
            return 1.0
        matched = sum(
            1
            for k in keys
            if JsonComparator._diagnostic_values_equal(expected_entry.get(k), actual_entry.get(k))
        )
        return matched / len(keys)

    @staticmethod
    def _compare_entry_fields(
        expected_entry: Any, actual_entry: Any
    ) -> List[EntryFieldComparison]:
        """Per-sub-field comparison for one matched (expected, actual)
        entry pair, used to populate EntryDiagnostic.field_comparisons."""
        if not isinstance(expected_entry, dict) or not isinstance(actual_entry, dict):
            matched = expected_entry == actual_entry
            return [EntryFieldComparison("", expected_entry, actual_entry, matched)]
        keys = sorted(set(expected_entry.keys()) | set(actual_entry.keys()))
        return [
            EntryFieldComparison(
                field_name=key,
                expected=expected_entry.get(key),
                actual=actual_entry.get(key),
                matched=JsonComparator._diagnostic_values_equal(
                    expected_entry.get(key), actual_entry.get(key)
                ),
            )
            for key in keys
        ]

    @classmethod
    def _diagnose_array(
        cls, field_path: str, expected_list: list, actual_list: list
    ) -> ArrayFieldDiagnostic:
        """Align expected_list's entries to actual_list's (best-match, by
        _entry_similarity) and score each aligned pair's sub-fields.

        Matching is greedy: every possible (expected_index, actual_index)
        pair with similarity > 0 is a candidate; candidates are consumed
        highest-similarity-first, each index used at most once. A pair
        with similarity 0 (nothing in common) is never matched — an
        expected entry with literally no sub-field in common with any
        remaining actual entry is indistinguishable from "not extracted"
        and is reported as MISSING rather than arbitrarily paired with
        whatever actual entry happens to be left (see
        docs/JSON_COMPARATOR_DIAGNOSTIC.md for the rationale and its
        trade-offs).
        """
        expected_count = len(expected_list)
        actual_count = len(actual_list)

        candidates = []
        for i in range(expected_count):
            for j in range(actual_count):
                score = cls._entry_similarity(expected_list[i], actual_list[j])
                if score > 0:
                    candidates.append((score, i, j))
        # Highest similarity first; ties broken by (i, j) for determinism.
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

        matched_expected: Dict[int, int] = {}
        used_actual = set()
        for score, i, j in candidates:
            if i in matched_expected or j in used_actual:
                continue
            matched_expected[i] = j
            used_actual.add(j)

        entries: List[EntryDiagnostic] = []
        for i in range(expected_count):
            if i not in matched_expected:
                continue
            j = matched_expected[i]
            score = cls._entry_similarity(expected_list[i], actual_list[j])
            field_comparisons = cls._compare_entry_fields(expected_list[i], actual_list[j])
            status = (
                ENTRY_STATUS_FULL_MATCH if score == 1.0 else ENTRY_STATUS_PARTIAL_MATCH
            )
            entries.append(EntryDiagnostic(i, j, status, score, field_comparisons))
        for i in range(expected_count):
            if i not in matched_expected:
                entries.append(EntryDiagnostic(i, None, ENTRY_STATUS_MISSING, 0.0, []))
        for j in range(actual_count):
            if j not in used_actual:
                entries.append(EntryDiagnostic(None, j, ENTRY_STATUS_EXTRA, 0.0, []))

        found = [
            e
            for e in entries
            if e.status in (ENTRY_STATUS_FULL_MATCH, ENTRY_STATUS_PARTIAL_MATCH)
        ]
        entry_accuracy = (len(found) / expected_count) if expected_count else 1.0

        total_subfields = sum(len(e.field_comparisons) for e in found)
        matched_subfields = sum(
            1 for e in found for fc in e.field_comparisons if fc.matched
        )
        field_accuracy = (matched_subfields / total_subfields) if total_subfields else 1.0

        return ArrayFieldDiagnostic(
            field_path=field_path,
            expected_count=expected_count,
            actual_count=actual_count,
            entries=entries,
            entry_accuracy=entry_accuracy,
            field_accuracy=field_accuracy,
        )

    @staticmethod
    def _aggregate_entry_accuracy(array_diagnostics: Dict[str, ArrayFieldDiagnostic]) -> float:
        """Entry accuracy across every array field, weighted by each
        field's own ground-truth entry count (not a plain average of the
        per-field accuracies, so a field with 8 entries counts 8x as much
        as one with 1)."""
        total = sum(d.expected_count for d in array_diagnostics.values())
        if not total:
            return 1.0
        found = sum(
            sum(
                1
                for e in d.entries
                if e.status in (ENTRY_STATUS_FULL_MATCH, ENTRY_STATUS_PARTIAL_MATCH)
            )
            for d in array_diagnostics.values()
        )
        return found / total

    @staticmethod
    def _aggregate_field_accuracy_diag(array_diagnostics: Dict[str, ArrayFieldDiagnostic]) -> float:
        """Sub-field accuracy across every array field's matched entries,
        weighted by sub-field count."""
        total = 0
        matched = 0
        for d in array_diagnostics.values():
            for e in d.entries:
                if e.status not in (ENTRY_STATUS_FULL_MATCH, ENTRY_STATUS_PARTIAL_MATCH):
                    continue
                total += len(e.field_comparisons)
                matched += sum(1 for fc in e.field_comparisons if fc.matched)
        return (matched / total) if total else 1.0

    @staticmethod
    def _aggregate_mismatch_by_field(
        array_diagnostics: Dict[str, ArrayFieldDiagnostic],
    ) -> Dict[str, Dict[str, int]]:
        """"<array_field>.<sub_field>" -> {"total", "matched"} counts,
        over matched entries only (a MISSING/EXTRA entry has no sub-field
        values to attribute a mismatch to any specific field)."""
        breakdown: Dict[str, Dict[str, int]] = {}
        for array_field, diag in array_diagnostics.items():
            for entry in diag.entries:
                if entry.status not in (ENTRY_STATUS_FULL_MATCH, ENTRY_STATUS_PARTIAL_MATCH):
                    continue
                for fc in entry.field_comparisons:
                    key = f"{array_field}.{fc.field_name}" if fc.field_name else array_field
                    counts = breakdown.setdefault(key, {"total": 0, "matched": 0})
                    counts["total"] += 1
                    if fc.matched:
                        counts["matched"] += 1
        return breakdown

    @staticmethod
    def _write_resume_accuracy_csv(
        comparisons: Dict[str, ComparisonResult], csv_path: Path
    ) -> None:
        """One row per resume: its field counts and overall accuracy."""
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "resume_name",
                    "total_fields",
                    "matched_fields",
                    "missing_fields",
                    "extra_fields",
                    "accuracy",
                ]
            )
            for resume_name, result in comparisons.items():
                writer.writerow(
                    [
                        resume_name,
                        result.total_fields,
                        result.matched_fields,
                        result.missing_fields,
                        result.extra_fields,
                        f"{result.accuracy:.4f}",
                    ]
                )

    @staticmethod
    def _write_field_accuracy_csv(
        comparisons: Dict[str, ComparisonResult], csv_path: Path
    ) -> None:
        """One row per field path, aggregated across every resume in the
        batch — which fields are reliable vs. consistently wrong."""
        totals: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "matched": 0, "missing": 0, "extra": 0}
        )
        for result in comparisons.values():
            for field_result in result.field_results:
                counts = totals[field_result.field_path]
                counts["total"] += 1
                if field_result.status == STATUS_PASS:
                    counts["matched"] += 1
                elif field_result.status == STATUS_MISSING:
                    counts["missing"] += 1
                elif field_result.status == STATUS_EXTRA:
                    counts["extra"] += 1

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["field_path", "total", "matched", "missing", "extra", "accuracy"])
            for field_path in sorted(totals):
                counts = totals[field_path]
                accuracy = (counts["matched"] / counts["total"]) if counts["total"] else 0.0
                writer.writerow(
                    [
                        field_path,
                        counts["total"],
                        counts["matched"],
                        counts["missing"],
                        counts["extra"],
                        f"{accuracy:.4f}",
                    ]
                )

    @staticmethod
    def _write_resume_accuracy_excluding_optional_keys_csv(
        comparisons: Dict[str, ComparisonResult], csv_path: Path
    ) -> None:
        """Decision D8: one row per resume, re-counted with optional
        top-level keys excluded (see strict_excluding_optional_keys), plus
        one final `__all__` row with the batch totals. The header names
        the comparator version and the excluded keys so the file is
        self-describing."""
        optional_keys = load_optional_top_level_keys()
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "resume_name",
                    "total_fields_excluding_optional",
                    "matched_fields_excluding_optional",
                    "excluded_fields",
                    "strict_accuracy_excluding_optional_keys",
                    "optional_keys",
                    "comparator_version",
                ]
            )
            sum_total = sum_matched = sum_excluded = 0
            for resume_name, result in comparisons.items():
                ex = strict_excluding_optional_keys(result, optional_keys)
                sum_total += ex.total_fields
                sum_matched += ex.matched_fields
                sum_excluded += ex.excluded_fields
                writer.writerow(
                    [
                        resume_name,
                        ex.total_fields,
                        ex.matched_fields,
                        ex.excluded_fields,
                        f"{ex.accuracy:.4f}",
                        "|".join(optional_keys),
                        COMPARATOR_VERSION,
                    ]
                )
            writer.writerow(
                [
                    "__all__",
                    sum_total,
                    sum_matched,
                    sum_excluded,
                    f"{(sum_matched / sum_total) if sum_total else 1.0:.4f}",
                    "|".join(optional_keys),
                    COMPARATOR_VERSION,
                ]
            )

    @staticmethod
    def _write_summary_csv(
        comparisons: Dict[str, ComparisonResult], csv_path: Path
    ) -> None:
        """One row summarizing the whole batch."""
        total_resumes = len(comparisons)
        total_fields = sum(result.total_fields for result in comparisons.values())
        matched_fields = sum(result.matched_fields for result in comparisons.values())
        missing_fields = sum(result.missing_fields for result in comparisons.values())
        extra_fields = sum(result.extra_fields for result in comparisons.values())
        overall_accuracy = (matched_fields / total_fields) if total_fields else 1.0
        average_resume_accuracy = (
            sum(result.accuracy for result in comparisons.values()) / total_resumes
            if total_resumes
            else 0.0
        )

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "total_resumes",
                    "total_fields",
                    "matched_fields",
                    "missing_fields",
                    "extra_fields",
                    "overall_accuracy",
                    "average_resume_accuracy",
                ]
            )
            writer.writerow(
                [
                    total_resumes,
                    total_fields,
                    matched_fields,
                    missing_fields,
                    extra_fields,
                    f"{overall_accuracy:.4f}",
                    f"{average_resume_accuracy:.4f}",
                ]
            )

    @staticmethod
    def _write_diagnostic_summary_csv(
        diagnostics: Dict[str, "DiagnosticResult"], csv_path: Path
    ) -> None:
        """One row per resume: strict accuracy alongside the diagnostic
        entry/field accuracies, so the gap between them is visible
        resume-by-resume, not just in aggregate."""
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                ["resume_name", "strict_accuracy", "entry_accuracy", "field_accuracy"]
            )
            for resume_name, diag in diagnostics.items():
                writer.writerow(
                    [
                        resume_name,
                        f"{diag.strict_accuracy:.4f}",
                        f"{diag.entry_accuracy:.4f}",
                        f"{diag.field_accuracy:.4f}",
                    ]
                )

    @staticmethod
    def _write_diagnostic_mismatch_csv(
        diagnostics: Dict[str, "DiagnosticResult"], csv_path: Path
    ) -> None:
        """Mismatch breakdown by field (Milestone 20.3, requirement 4),
        aggregated across every resume in the batch — one row per
        "<array_field>.<sub_field>" path, e.g. "work_experience.duration",
        so which specific sub-fields drive the remaining error is visible
        at a glance."""
        totals: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "matched": 0})
        for diag in diagnostics.values():
            for field_path, counts in diag.mismatch_by_field.items():
                totals[field_path]["total"] += counts["total"]
                totals[field_path]["matched"] += counts["matched"]

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["field_path", "total", "matched", "mismatched", "accuracy"])
            for field_path in sorted(totals):
                counts = totals[field_path]
                mismatched = counts["total"] - counts["matched"]
                accuracy = (counts["matched"] / counts["total"]) if counts["total"] else 0.0
                writer.writerow(
                    [
                        field_path,
                        counts["total"],
                        counts["matched"],
                        mismatched,
                        f"{accuracy:.4f}",
                    ]
                )

    @staticmethod
    def _write_diagnostic_report_md(
        diagnostics: Dict[str, "DiagnosticResult"], report_path: Path
    ) -> None:
        """Human-readable Markdown summary: aggregate strict/entry/field
        accuracy, an entry-level rollup per array field, and the
        mismatch-by-field breakdown — the same four things
        generate_diagnostic_reports()'s CSVs carry, in one report."""
        total_resumes = len(diagnostics)

        total_strict_fields = sum(
            diag.strict_result.total_fields for diag in diagnostics.values() if diag.strict_result
        )
        matched_strict_fields = sum(
            diag.strict_result.matched_fields
            for diag in diagnostics.values()
            if diag.strict_result
        )
        aggregate_strict_accuracy = (
            (matched_strict_fields / total_strict_fields) if total_strict_fields else 1.0
        )

        array_field_totals: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"expected": 0, "actual": 0, "found": 0, "sub_total": 0, "sub_matched": 0}
        )
        for diag in diagnostics.values():
            for field_path, array_diag in diag.array_diagnostics.items():
                totals = array_field_totals[field_path]
                totals["expected"] += array_diag.expected_count
                totals["actual"] += array_diag.actual_count
                totals["found"] += sum(
                    1
                    for e in array_diag.entries
                    if e.status in (ENTRY_STATUS_FULL_MATCH, ENTRY_STATUS_PARTIAL_MATCH)
                )
                for e in array_diag.entries:
                    if e.status not in (ENTRY_STATUS_FULL_MATCH, ENTRY_STATUS_PARTIAL_MATCH):
                        continue
                    totals["sub_total"] += len(e.field_comparisons)
                    totals["sub_matched"] += sum(1 for fc in e.field_comparisons if fc.matched)

        total_expected_entries = sum(t["expected"] for t in array_field_totals.values())
        total_found_entries = sum(t["found"] for t in array_field_totals.values())
        aggregate_entry_accuracy = (
            (total_found_entries / total_expected_entries) if total_expected_entries else 1.0
        )
        total_sub = sum(t["sub_total"] for t in array_field_totals.values())
        total_sub_matched = sum(t["sub_matched"] for t in array_field_totals.values())
        aggregate_field_accuracy = (total_sub_matched / total_sub) if total_sub else 1.0

        mismatch_totals: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "matched": 0}
        )
        for diag in diagnostics.values():
            for field_path, counts in diag.mismatch_by_field.items():
                mismatch_totals[field_path]["total"] += counts["total"]
                mismatch_totals[field_path]["matched"] += counts["matched"]

        lines = [
            "# Diagnostic Comparison Report",
            "",
            "Diagnostic mode never changes the strict score above it — see "
            "docs/JSON_COMPARATOR_DIAGNOSTIC.md for how the two relate.",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Resumes | {total_resumes} |",
            f"| Strict accuracy (whole-array, all-or-nothing) | {aggregate_strict_accuracy:.2%} |",
            f"| Entry accuracy (was this entry found at all?) | {aggregate_entry_accuracy:.2%} |",
            f"| Field accuracy (sub-fields of found entries) | {aggregate_field_accuracy:.2%} |",
            "",
            "## Entry-Level Matching by Array Field",
            "",
            "| Field | GT entries | Generated entries | Found | Missing | Extra | Entry accuracy | Field accuracy |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for field_path in sorted(array_field_totals):
            totals = array_field_totals[field_path]
            missing = totals["expected"] - totals["found"]
            extra = totals["actual"] - totals["found"]
            entry_acc = (totals["found"] / totals["expected"]) if totals["expected"] else 1.0
            field_acc = (totals["sub_matched"] / totals["sub_total"]) if totals["sub_total"] else 1.0
            lines.append(
                f"| {field_path} | {totals['expected']} | {totals['actual']} | "
                f"{totals['found']} | {missing} | {extra} | {entry_acc:.2%} | {field_acc:.2%} |"
            )

        lines += ["", "## Mismatch Breakdown by Field", "", "| Field | Total | Matched | Mismatched | Accuracy |", "|---|---|---|---|---|"]
        for field_path in sorted(mismatch_totals):
            counts = mismatch_totals[field_path]
            mismatched = counts["total"] - counts["matched"]
            accuracy = (counts["matched"] / counts["total"]) if counts["total"] else 0.0
            lines.append(
                f"| {field_path} | {counts['total']} | {counts['matched']} | {mismatched} | {accuracy:.2%} |"
            )
        if not mismatch_totals:
            lines.append("| _(no array-of-object fields found in this batch)_ | | | | |")

        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
