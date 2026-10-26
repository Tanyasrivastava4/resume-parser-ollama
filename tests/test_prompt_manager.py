"""
tests/test_prompt_manager.py

Tests for services/prompt_manager.py (Milestone 9).

Each test builds its own isolated prompts directory under pytest's
`tmp_path`, so these tests never touch the project's real prompts/
directory and don't depend on any actual prompt content existing there.
"""

from pathlib import Path

import pytest

from services.prompt_manager import (
    PromptManager,
    PromptNotFoundError,
    PromptRenderError,
)


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    """A temporary prompts/ tree with a flat prompt and nested ones."""
    (tmp_path / "greeting.md").write_text("Hello {{name}}!", encoding="utf-8")

    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    (resume_dir / "extract_markdown.md").write_text(
        "Extract a resume for {{candidate_name}} as of {{today}}.",
        encoding="utf-8",
    )

    invoice_dir = tmp_path / "invoice"
    invoice_dir.mkdir()
    (invoice_dir / "extract_json.md").write_text(
        "Turn this text into JSON:\n{{resume_text}}", encoding="utf-8"
    )

    return tmp_path


@pytest.fixture
def manager(prompts_dir: Path) -> PromptManager:
    return PromptManager(prompts_dir=prompts_dir)


class TestLoad:
    def test_loads_raw_template_text(self, manager):
        assert manager.load("greeting") == "Hello {{name}}!"

    def test_loads_nested_prompt(self, manager):
        text = manager.load("resume/extract_markdown")

        assert "{{candidate_name}}" in text
        assert "{{today}}" in text

    def test_loads_deeply_nested_prompt(self, manager):
        text = manager.load("invoice/extract_json")

        assert "{{resume_text}}" in text

    def test_missing_prompt_raises_not_found(self, manager):
        with pytest.raises(PromptNotFoundError):
            manager.load("does_not_exist")

    def test_missing_nested_prompt_raises_not_found(self, manager):
        with pytest.raises(PromptNotFoundError):
            manager.load("resume/does_not_exist")

    def test_path_traversal_raises_not_found(self, manager):
        with pytest.raises(PromptNotFoundError):
            manager.load("../outside")


class TestRender:
    def test_replaces_single_variable(self, manager):
        result = manager.render("greeting", {"name": "Ada"})

        assert result == "Hello Ada!"

    def test_replaces_multiple_variables(self, manager):
        result = manager.render(
            "resume/extract_markdown",
            {"candidate_name": "Ada Lovelace", "today": "2026-07-29"},
        )

        assert result == "Extract a resume for Ada Lovelace as of 2026-07-29."

    def test_missing_variable_raises_render_error(self, manager):
        with pytest.raises(PromptRenderError):
            manager.render("greeting", {})

    def test_missing_one_of_several_variables_raises_render_error(self, manager):
        with pytest.raises(PromptRenderError):
            manager.render("resume/extract_markdown", {"candidate_name": "Ada"})

    def test_render_of_missing_prompt_raises_not_found(self, manager):
        with pytest.raises(PromptNotFoundError):
            manager.render("does_not_exist", {"name": "Ada"})


class TestCacheBehaviour:
    def test_second_load_does_not_reread_file(self, manager, prompts_dir):
        first = manager.load("greeting")

        (prompts_dir / "greeting.md").write_text("Changed {{name}}!", encoding="utf-8")
        second = manager.load("greeting")

        assert first == second == "Hello {{name}}!"

    def test_clear_cache_forces_reread(self, manager, prompts_dir):
        manager.load("greeting")

        (prompts_dir / "greeting.md").write_text("Changed {{name}}!", encoding="utf-8")
        manager.clear_cache()
        refreshed = manager.load("greeting")

        assert refreshed == "Changed {{name}}!"

    def test_separate_instances_do_not_share_cache(self, prompts_dir):
        first_manager = PromptManager(prompts_dir=prompts_dir)
        first_manager.load("greeting")

        (prompts_dir / "greeting.md").write_text("Changed {{name}}!", encoding="utf-8")

        second_manager = PromptManager(prompts_dir=prompts_dir)
        assert second_manager.load("greeting") == "Changed {{name}}!"
