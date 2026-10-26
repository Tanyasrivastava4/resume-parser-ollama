"""
parsers/base_parser.py

Defines the common interface every pipeline's parser must implement.

Per PROJECT_PLAN.md, Pipelines A, B, and C all take a resume file and must
produce output conforming to the same JSON schema, so the benchmark harness
can compare them fairly. This base class is that shared contract: it
declares the shape parsers take (a resume path in, a schema-conformant dict
out) without dictating how any pipeline gets there.

This is intentionally just an interface — no pipeline logic (Qwen2.5-VL,
Docling, markdown conversion, etc.) is implemented here or anywhere yet.
Concrete parsers (e.g. a future parsers/pipeline_a.py) will subclass this.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseParser(ABC):
    """Common interface for all resume-parsing pipelines.

    Subclasses represent one pipeline each (see config.PIPELINES) and must
    return output conforming to the shared resume JSON schema
    (see config.RESUME_JSON_SCHEMA_NAME), so results are comparable across
    pipelines in the benchmark harness.
    """

    #: Identifier matching one of the entries in config.PIPELINES.
    #: Subclasses must set this so benchmark results can be attributed to
    #: the correct pipeline.
    name: str

    @abstractmethod
    def parse(self, resume_path: Path) -> dict[str, Any]:
        """Parse a single resume file and return it as a schema-conformant dict.

        Args:
            resume_path: Path to the source resume file (e.g. a PDF).

        Returns:
            A dict conforming to the shared resume JSON schema.
        """
        raise NotImplementedError
