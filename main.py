"""
main.py

Single entry point for the Resume AI Benchmark application.

This will eventually wire together: reading resumes from input/, running
them through the parsers in parsers/ (each implementing base_parser's
interface), validating output against the shared schema via validators/,
and recording results through benchmark/.

For now this is a placeholder only — no pipeline is implemented or called
yet. It exists so the project has a single, obvious entry point from the
start, per clean-architecture conventions (entry point separate from
business logic).
"""


def main() -> None:
    """Entry point. Intentionally does nothing yet — pipelines are not implemented."""
    print("Resume AI Benchmark: project scaffolding only, no pipelines implemented yet.")


if __name__ == "__main__":
    main()
