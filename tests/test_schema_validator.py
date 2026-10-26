"""
tests/test_schema_validator.py

Tests for validators/schema_validator.py (Milestone 13.2).

Uses the real schemas/resume_schema_v1.json (no need to fake it — schema
validation is a cheap, deterministic, local operation), so these tests
also double as a regression check on the schema file itself.
"""

import copy
import json

import pytest

from validators.schema_validator import SchemaValidationError, SchemaValidator

VALID_RESUME = {
    "personal_information": {
        "full_name": "Ada Lovelace",
        "first_name": "Ada",
        "last_name": "Lovelace",
        "current_location": None,
    },
    "contact_information": {
        "email": ["ada@example.com"],
        "phone": [],
        "linkedin": None,
        "github": None,
        "portfolio": None,
    },
    "professional_summary": None,
    "skills": ["Mathematics", "Analytical writing"],
    "languages": ["English", "French"],
    "work_experience": [
        {
            "company": "Analytical Engines Ltd",
            "designation": "Mathematician",
            "employment_type": "Full-time",
            "location": None,
            "start_date": "1842",
            "end_date": None,
            "currently_working": True,
            "duration": None,
            "responsibilities": ["Wrote the first published algorithm"],
            "technologies": [],
        }
    ],
    "education": [
        {
            "institution": "Royal Institution",
            "degree": "Self-taught",
            "specialization": "Mathematics",
            "location": None,
            "start_date": None,
            "end_date": None,
            "currently_studying": False,
            "grade": None,
            "description": None,
        }
    ],
}


@pytest.fixture
def validator():
    return SchemaValidator()


class TestValidate:
    def test_valid_resume_returns_true(self, validator):
        assert validator.validate(VALID_RESUME) is True

    def test_missing_top_level_required_property_raises(self, validator):
        data = copy.deepcopy(VALID_RESUME)
        del data["skills"]

        with pytest.raises(SchemaValidationError):
            validator.validate(data)

    def test_missing_nested_required_property_raises(self, validator):
        data = copy.deepcopy(VALID_RESUME)
        del data["personal_information"]["full_name"]

        with pytest.raises(SchemaValidationError):
            validator.validate(data)

    def test_wrong_data_type_for_scalar_field_raises(self, validator):
        data = copy.deepcopy(VALID_RESUME)
        data["professional_summary"] = 12345  # must be string or null

        with pytest.raises(SchemaValidationError):
            validator.validate(data)

    def test_array_field_given_as_non_array_raises(self, validator):
        data = copy.deepcopy(VALID_RESUME)
        data["skills"] = "Mathematics"  # must be an array, not a string

        with pytest.raises(SchemaValidationError):
            validator.validate(data)

    def test_invalid_array_item_type_raises(self, validator):
        data = copy.deepcopy(VALID_RESUME)
        data["skills"] = ["Mathematics", 42]  # every item must be a string

        with pytest.raises(SchemaValidationError):
            validator.validate(data)

    def test_unexpected_extra_property_raises(self, validator):
        data = copy.deepcopy(VALID_RESUME)
        data["unexpected_field"] = "surprise"

        with pytest.raises(SchemaValidationError):
            validator.validate(data)

    def test_non_dict_input_raises(self, validator):
        with pytest.raises(SchemaValidationError):
            validator.validate(["not", "a", "resume"])


class TestSchemaCaching:
    def test_schema_is_not_loaded_until_first_validate_call(self, validator):
        assert validator._schema is None

        validator.validate(VALID_RESUME)

        assert validator._schema is not None

    def test_schema_object_is_reused_across_calls(self, validator):
        validator.validate(VALID_RESUME)
        first_schema = validator._schema

        validator.validate(VALID_RESUME)
        second_schema = validator._schema

        assert first_schema is second_schema

    def test_schema_file_is_valid_json(self, validator):
        with validator.schema_path.open(encoding="utf-8") as schema_file:
            schema = json.load(schema_file)

        assert schema["type"] == "object"
        assert "personal_information" in schema["properties"]

    def test_work_experience_is_a_required_top_level_property(self, validator):
        """Milestone 17.3: work_experience must be required, not merely
        permitted, so a pipeline regression that stops producing it is
        caught by schema validation rather than silently passing."""
        with validator.schema_path.open(encoding="utf-8") as schema_file:
            schema = json.load(schema_file)

        assert "work_experience" in schema["properties"]
        assert "work_experience" in schema["required"]

    def test_education_is_a_required_top_level_property(self, validator):
        """Milestone 18.3: education must be required, not merely
        permitted, so a pipeline regression that stops producing it is
        caught by schema validation rather than silently passing."""
        with validator.schema_path.open(encoding="utf-8") as schema_file:
            schema = json.load(schema_file)

        assert "education" in schema["properties"]
        assert "education" in schema["required"]
