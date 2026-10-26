"""
tests/test_pdf_loader.py

Tests for services/pdf_loader.py (Milestone 7).

PDF fixtures are generated on the fly with PyMuPDF itself, so these tests
have no external file dependencies. Each test uses a unique resume name so
it doesn't collide with output from other tests or real project data, and
cleans up the images it writes under output/images/ afterwards.
"""

import shutil

import fitz
import pytest

from config import OUTPUT_DIR
from services import pdf_loader
from services.pdf_loader import (
    InvalidPDFError,
    PDFNotFoundError,
    load_pdf_pages,
)
from settings import PDF_RENDER_DPI


def _make_pdf(path, page_count: int) -> None:
    """Create a minimal valid PDF with `page_count` blank pages at `path`."""
    document = fitz.open()
    for _ in range(page_count):
        document.new_page()
    document.save(path)
    document.close()


@pytest.fixture
def images_cleanup():
    """Track resume names whose output/images/<name>/ folder should be
    removed after the test, regardless of pass/fail."""
    resume_names = []
    yield resume_names
    for resume_name in resume_names:
        shutil.rmtree(OUTPUT_DIR / "images" / resume_name, ignore_errors=True)


def test_load_pdf_pages_returns_one_image_per_page(tmp_path, images_cleanup):
    resume_name = "test_resume_multi_page"
    images_cleanup.append(resume_name)

    pdf_path = tmp_path / f"{resume_name}.pdf"
    _make_pdf(pdf_path, page_count=3)

    result = load_pdf_pages(pdf_path)

    assert result == [
        f"output/images/{resume_name}/page_001.png",
        f"output/images/{resume_name}/page_002.png",
        f"output/images/{resume_name}/page_003.png",
    ]
    for relative_path in result:
        assert (OUTPUT_DIR.parent / relative_path).is_file()


def test_load_pdf_pages_single_page(tmp_path, images_cleanup):
    resume_name = "test_resume_single_page"
    images_cleanup.append(resume_name)

    pdf_path = tmp_path / f"{resume_name}.pdf"
    _make_pdf(pdf_path, page_count=1)

    result = load_pdf_pages(pdf_path)

    assert result == [f"output/images/{resume_name}/page_001.png"]


def test_load_pdf_pages_raises_when_file_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.pdf"

    with pytest.raises(PDFNotFoundError):
        load_pdf_pages(missing_path)


def test_load_pdf_pages_raises_for_non_pdf_extension(tmp_path):
    text_path = tmp_path / "not_a_pdf.txt"
    text_path.write_text("hello")

    with pytest.raises(InvalidPDFError):
        load_pdf_pages(text_path)


def test_load_pdf_pages_raises_for_corrupt_pdf(tmp_path):
    corrupt_path = tmp_path / "corrupt.pdf"
    corrupt_path.write_bytes(b"not actually a pdf")

    with pytest.raises(InvalidPDFError):
        load_pdf_pages(corrupt_path)


def test_load_pdf_pages_respects_custom_dpi(tmp_path, images_cleanup):
    resume_name = "test_resume_dpi"
    images_cleanup.append(resume_name)

    pdf_path = tmp_path / f"{resume_name}.pdf"
    _make_pdf(pdf_path, page_count=1)

    low_dpi_result = load_pdf_pages(pdf_path, dpi=72)
    low_dpi_size = (OUTPUT_DIR.parent / low_dpi_result[0]).stat().st_size

    high_dpi_result = load_pdf_pages(pdf_path, dpi=300)
    high_dpi_size = (OUTPUT_DIR.parent / high_dpi_result[0]).stat().st_size

    assert high_dpi_size > low_dpi_size


def test_default_dpi_comes_from_settings():
    """Milestone 12.5: PDFLoader must read its default DPI from
    settings.PDF_RENDER_DPI instead of a hardcoded value."""
    assert pdf_loader.DEFAULT_DPI == PDF_RENDER_DPI


def test_load_pdf_pages_without_dpi_arg_uses_settings_dpi(tmp_path, images_cleanup):
    default_call_name = "test_resume_default_dpi"
    explicit_call_name = "test_resume_explicit_settings_dpi"
    images_cleanup.append(default_call_name)
    images_cleanup.append(explicit_call_name)

    default_pdf_path = tmp_path / f"{default_call_name}.pdf"
    explicit_pdf_path = tmp_path / f"{explicit_call_name}.pdf"
    _make_pdf(default_pdf_path, page_count=1)
    _make_pdf(explicit_pdf_path, page_count=1)

    default_result = load_pdf_pages(default_pdf_path)
    explicit_result = load_pdf_pages(explicit_pdf_path, dpi=PDF_RENDER_DPI)

    default_size = (OUTPUT_DIR.parent / default_result[0]).stat().st_size
    explicit_size = (OUTPUT_DIR.parent / explicit_result[0]).stat().st_size
    assert default_size == explicit_size
