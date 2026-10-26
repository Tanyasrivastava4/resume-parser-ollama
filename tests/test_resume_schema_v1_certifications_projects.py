"""
tests/test_resume_schema_v1_certifications_projects.py

Milestone T3/Phase 1 -- backward-compatibility and new-capability tests
for the extended schemas/resume_schema_v1.json (certifications/projects
added as OPTIONAL top-level keys) and the two new module schemas
(certifications_schema.json, projects_schema.json). ZERO LLM calls --
uses only already-existing, already-committed real production output
(output/json_final/*.json) and hand-built schema-shaped fixtures.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from validators.schema_validator import SchemaValidationError, SchemaValidator

_JSON_FINAL_DIR = Path("output/json_final")
_REAL_JSON_FINAL_FILES = sorted(_JSON_FINAL_DIR.glob("*.json")) if _JSON_FINAL_DIR.exists() else []


class TestBackwardCompatibility_AllRealExistingOutputs:
    """Every one of the real, already-committed production output files
    (produced before this milestone, none of which populate
    certifications/projects) must remain valid against the extended
    schema -- this milestone must not silently break existing valid
    outputs (Part F)."""

    @pytest.mark.skipif(not _REAL_JSON_FINAL_FILES, reason="output/json_final not present in this environment")
    def test_every_real_existing_output_file_still_validates(self):
        validator = SchemaValidator()
        failures = []
        for p in _REAL_JSON_FINAL_FILES:
            data = json.loads(p.read_text())
            try:
                validator.validate(data)
            except SchemaValidationError as e:
                failures.append((p.name, str(e)))
        assert failures == [], f"Real existing outputs broke backward compatibility: {failures}"

    @pytest.mark.skipif(not _REAL_JSON_FINAL_FILES, reason="output/json_final not present in this environment")
    def test_real_existing_outputs_have_no_certifications_or_projects_key(self):
        # Confirms these are genuinely "existing valid outputs without
        # the new keys" cases, not accidentally already populated.
        for p in _REAL_JSON_FINAL_FILES:
            data = json.loads(p.read_text())
            assert "certifications" not in data
            assert "projects" not in data


class TestNewCapability_PopulatedCertificationsAndProjects:
    """A merged resume JSON WITH populated certifications/projects
    validates -- confirms the new capability actually works, not just
    that its absence is tolerated."""

    @staticmethod
    def _base_resume(**overrides) -> dict:
        base = {
            "personal_information": {
                "full_name": "Ada Lovelace", "first_name": "Ada", "last_name": "Lovelace",
                "current_location": "London",
            },
            "contact_information": {
                "email": ["ada@example.com"], "phone": [], "linkedin": None, "github": None, "portfolio": None,
            },
            "professional_summary": None,
            "skills": ["Python", "SQL"],
            "languages": [],
            "work_experience": [],
            "education": [],
        }
        base.update(overrides)
        return base

    def test_populated_certifications_and_projects_validate(self):
        resume = self._base_resume(
            certifications=[{
                "name": "AWS Certified Solutions Architect", "issuing_organization": "AWS",
                "issue_date": "2023", "expiry_date": None,
            }],
            projects=[{
                "name": "Senraa", "description": "AR fashion platform", "role": None,
                "start_date": None, "end_date": None,
                "highlights": ["Built AR try-on"], "technologies": ["React"], "url": None,
            }],
        )
        SchemaValidator().validate(resume)  # must not raise

    def test_empty_certifications_and_projects_arrays_validate(self):
        resume = self._base_resume(certifications=[], projects=[])
        SchemaValidator().validate(resume)  # must not raise

    def test_invalid_certification_entry_missing_field_is_rejected(self):
        resume = self._base_resume(certifications=[{"name": "PMP"}])  # missing 3 required fields
        with pytest.raises(SchemaValidationError):
            SchemaValidator().validate(resume)

    def test_invalid_project_entry_missing_field_is_rejected(self):
        resume = self._base_resume(projects=[{"name": "Senraa"}])  # missing 7 required fields
        with pytest.raises(SchemaValidationError):
            SchemaValidator().validate(resume)

    def test_certifications_and_skills_coexist_independently(self):
        # Structural confirmation of Part C's independence design: a
        # certification-shaped item in `certifications` and an
        # unrelated real skill in `skills` both validate side by side,
        # with no cross-field constraint linking them.
        resume = self._base_resume(
            skills=["Python", "SQL", "Kubernetes"],
            certifications=[{
                "name": "Certified Kubernetes Administrator", "issuing_organization": "CNCF",
                "issue_date": None, "expiry_date": None,
            }],
        )
        SchemaValidator().validate(resume)  # must not raise -- no forced deduplication/linkage

    def test_projects_and_work_experience_coexist_independently(self):
        resume = self._base_resume(
            work_experience=[{
                "company": "Acme Corp", "designation": "Engineer", "employment_type": "Full-time",
                "location": None, "start_date": "2020", "end_date": None, "currently_working": True,
                "duration": None, "responsibilities": [], "technologies": [],
            }],
            projects=[{
                "name": "Side Project", "description": None, "role": None,
                "start_date": None, "end_date": None, "highlights": [], "technologies": [], "url": None,
            }],
        )
        SchemaValidator().validate(resume)  # must not raise
