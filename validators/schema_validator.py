"""
validators/schema_validator.py

Resume Schema Validation (Milestone 13.2).

Validates a parsed JSON draft (a Python dict) against
schemas/resume_schema_v1.json, using the jsonschema package. This is
purely structural validation — it checks that the JSON Draft Generator's
output has the right shape and types (required fields present, correct
types, no unexpected extra fields). It does not judge whether the
extracted content is itself accurate.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import jsonschema

from config import RESUME_JSON_SCHEMA_NAME, SCHEMAS_DIR

logger = logging.getLogger(__name__)

# Default schema used when no explicit schema_path is given.
DEFAULT_SCHEMA_PATH = SCHEMAS_DIR / f"{RESUME_JSON_SCHEMA_NAME}.json"


class SchemaValidatorError(Exception):
    """Base exception for all schema validator failures."""


class SchemaValidationError(SchemaValidatorError):
    """Raised when a JSON object does not conform to the schema."""


class SchemaValidator:
    """Validates Python dicts (parsed JSON) against a JSON Schema file.

    The schema is read from disk once, the first time it's needed, and
    cached in memory for the lifetime of this instance — repeated
    `validate()` calls never re-read or re-parse the schema file.
    """

    def __init__(self, schema_path: Path = DEFAULT_SCHEMA_PATH) -> None:
        """Initialize the validator.

        Args:
            schema_path: Path to the JSON Schema file to validate
                against. Defaults to schemas/resume_schema_v1.json.
        """
        self.schema_path = Path(schema_path)
        self._schema: Optional[dict] = None

    def validate(self, data: Any) -> bool:
        """Validate `data` against the schema.

        Args:
            data: The parsed JSON value (typically a dict) to validate.

        Returns:
            True if `data` conforms to the schema.

        Raises:
            SchemaValidationError: If `data` does not conform to the
                schema.
        """
        schema = self._load_schema()
        try:
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.ValidationError as exc:
            logger.error("Schema validation failed: %s", exc.message)
            raise SchemaValidationError(exc.message) from exc
        return True

    def _load_schema(self) -> dict:
        """Load and cache the schema from disk, on first use only."""
        if self._schema is None:
            logger.debug("Loading JSON schema from %s", self.schema_path)
            with self.schema_path.open("r", encoding="utf-8") as schema_file:
                self._schema = json.load(schema_file)
        return self._schema
