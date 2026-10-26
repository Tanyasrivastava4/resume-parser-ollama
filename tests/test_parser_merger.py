"""
tests/test_parser_merger.py

Tests for services/parser_merger.py (Milestone 14.7 prerequisite).

Pure in-memory logic, no external dependencies — every test runs
directly against ParserMerger with no mocking.
"""

import pytest

from services.parser_merger import MergeConflictError, ParserMerger


@pytest.fixture
def merger():
    return ParserMerger()


class TestMerge:
    def test_combines_multiple_module_results(self, merger):
        result = merger.merge(
            [
                {"personal_information": {"full_name": "Ada Lovelace"}},
                {"skills": ["Mathematics"]},
                {"languages": ["English"]},
            ]
        )

        assert result == {
            "personal_information": {"full_name": "Ada Lovelace"},
            "skills": ["Mathematics"],
            "languages": ["English"],
        }

    def test_empty_list_returns_empty_dict(self, merger):
        assert merger.merge([]) == {}

    def test_single_module_result_passes_through_unchanged(self, merger):
        module_object = {"skills": ["Python", "SQL"]}

        result = merger.merge([module_object])

        assert result == module_object
        assert result is not module_object  # merged into a new dict, not the same object

    def test_values_are_not_modified(self, merger):
        """No enrichment/normalization: values must come through exactly
        as given, including null and nested structures."""
        result = merger.merge(
            [
                {
                    "personal_information": {
                        "full_name": "Ada Lovelace",
                        "current_location": None,
                    }
                },
                {"professional_summary": None},
            ]
        )

        assert result["personal_information"]["current_location"] is None
        assert result["professional_summary"] is None

    def test_conflicting_key_raises_merge_conflict_error(self, merger):
        with pytest.raises(MergeConflictError):
            merger.merge(
                [
                    {"skills": ["Python"]},
                    {"skills": ["SQL"]},
                ]
            )

    def test_all_five_module_results_merge_into_full_resume_shape(self, merger):
        """Mirrors the real pipeline's five module extractors merging
        into the shape schemas/resume_schema_v1.json expects."""
        result = merger.merge(
            [
                {
                    "personal_information": {
                        "full_name": "Ada Lovelace",
                        "first_name": "Ada",
                        "last_name": "Lovelace",
                        "current_location": None,
                    }
                },
                {
                    "contact_information": {
                        "email": ["ada@example.com"],
                        "phone": [],
                        "linkedin": None,
                        "github": None,
                        "portfolio": None,
                    }
                },
                {"skills": ["Mathematics"]},
                {"professional_summary": "Mathematician and writer."},
                {"languages": ["English"]},
            ]
        )

        assert set(result.keys()) == {
            "personal_information",
            "contact_information",
            "skills",
            "professional_summary",
            "languages",
        }

    def test_six_module_results_merge_into_full_resume_shape(self, merger):
        """Milestone 17.3: work_experience became a sixth module merged
        into the real pipeline's output. ParserMerger itself needed no
        change to support this — confirmed here with a sixth input added
        to the same shape the test above already covers."""
        result = merger.merge(
            [
                {"personal_information": {"full_name": "Ada Lovelace"}},
                {"contact_information": {"email": ["ada@example.com"]}},
                {"skills": ["Mathematics"]},
                {"professional_summary": "Mathematician and writer."},
                {"languages": ["English"]},
                {
                    "work_experience": [
                        {
                            "company": "Analytical Engines Ltd",
                            "designation": "Mathematician",
                        }
                    ]
                },
            ]
        )

        assert set(result.keys()) == {
            "personal_information",
            "contact_information",
            "skills",
            "professional_summary",
            "languages",
            "work_experience",
        }
        assert result["work_experience"] == [
            {"company": "Analytical Engines Ltd", "designation": "Mathematician"}
        ]

    def test_seven_module_results_merge_into_full_resume_shape(self, merger):
        """Milestone 18.3: education became a seventh module merged into
        the real pipeline's output. ParserMerger itself needed no
        change to support this — confirmed here with a seventh input
        added to the same shape the tests above already cover."""
        result = merger.merge(
            [
                {"personal_information": {"full_name": "Ada Lovelace"}},
                {"contact_information": {"email": ["ada@example.com"]}},
                {"skills": ["Mathematics"]},
                {"professional_summary": "Mathematician and writer."},
                {"languages": ["English"]},
                {
                    "work_experience": [
                        {
                            "company": "Analytical Engines Ltd",
                            "designation": "Mathematician",
                        }
                    ]
                },
                {
                    "education": [
                        {"institution": "Acme University", "degree": "B.Sc."}
                    ]
                },
            ]
        )

        assert set(result.keys()) == {
            "personal_information",
            "contact_information",
            "skills",
            "professional_summary",
            "languages",
            "work_experience",
            "education",
        }
        assert result["education"] == [
            {"institution": "Acme University", "degree": "B.Sc."}
        ]
