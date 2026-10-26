"""
services/isolated_parser_pipeline.py

Milestone V2.4 -- evaluation-only convenience. NOT called by
scripts/run_parser_pipeline.py or services/batch_processor.py.

History: pipeline/parser_pipeline.py's _save_module_json()/_save_final_json()
used to be @staticmethod and hardcoded to config.OUTPUT_DIR, so this
subclass had to override both to redirect output. Remediation R-08
(2026-10-17) gave ParserPipeline its own `output_root` constructor
argument, which is exactly what this class provided; it is now a thin
alias kept so the existing V2.4 runner scripts (and any future one
written against the same name) keep working unchanged. A resume
processed through this class is parsed exactly as ParserPipeline would
parse it; only the save destination differs.
"""

from pathlib import Path

from pipeline.parser_pipeline import ParserPipeline


class IsolatedParserPipeline(ParserPipeline):
    """ParserPipeline that writes its module/final JSON output under a
    caller-supplied root directory instead of output/modules/ and
    output/json_final/. Equivalent to ParserPipeline(output_root=...)."""

    def __init__(self, *args, output_root: Path, **kwargs):
        super().__init__(*args, output_root=Path(output_root), **kwargs)

    @property
    def _output_root(self) -> Path:
        # Pre-R-08 attribute name, kept for anything that inspected it.
        return self.output_root
