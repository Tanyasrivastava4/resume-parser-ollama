"""
services/pdf_loader.py

Reusable PDF Loader (Milestone 7).

Rasterizes every page of a PDF into a PNG image using PyMuPDF (fitz), and
saves them under output/images/<resume_name>/. This module is a shared
building block for every pipeline in PROJECT_PLAN.md — Pipelines A, B, and
C all start from "PDF" and need page images (directly, or on the way to
markdown) before any model runs.

This module does no AI / parsing work. It only converts PDF pages to
images and returns their paths.

Example:
    resume1.pdf
        -> output/images/resume1/page_001.png
        -> output/images/resume1/page_002.png
"""

import logging
from pathlib import Path
from typing import List, Union

import fitz  # PyMuPDF

from config import BASE_DIR, OUTPUT_DIR
from settings import PDF_RENDER_DPI

logger = logging.getLogger(__name__)

# The DPI used when the caller does not specify one. Reads from
# settings.PDF_RENDER_DPI so it's configurable per environment without
# touching this module.
DEFAULT_DPI = PDF_RENDER_DPI

# fitz's baseline rendering resolution, used to convert a target DPI into
# a zoom factor for the rendering matrix.
_FITZ_BASE_DPI = 72


class PDFLoaderError(Exception):
    """Base exception for all PDF Loader failures."""


class PDFNotFoundError(PDFLoaderError):
    """Raised when the given PDF path does not exist or is not a file."""


class InvalidPDFError(PDFLoaderError):
    """Raised when the given file is not a readable PDF, or has no pages."""


def load_pdf_pages(pdf_path: Union[str, Path], dpi: int = DEFAULT_DPI) -> List[str]:
    """Render every page of a PDF to a PNG image and return their paths.

    Each page of `pdf_path` is rasterized with PyMuPDF and saved as
    ``output/images/<resume_name>/page_NNN.png``, where ``<resume_name>``
    is the PDF's filename without its extension, and ``NNN`` is the
    1-indexed, zero-padded page number.

    Args:
        pdf_path: Path to the source PDF file.
        dpi: Resolution to render each page at. Higher values produce
            larger, sharper images at the cost of disk space and
            processing time. Defaults to DEFAULT_DPI.

    Returns:
        A list of image paths, one per page in page order, as POSIX-style
        strings relative to the project root. For example::

            [
                "output/images/resume1/page_001.png",
                "output/images/resume1/page_002.png",
            ]

    Raises:
        PDFNotFoundError: If `pdf_path` does not exist or is not a file.
        InvalidPDFError: If the file cannot be opened/read as a PDF, or
            contains zero pages.
        PDFLoaderError: If a page fails to render or save for any other
            reason.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists() or not pdf_path.is_file():
        logger.error("PDF not found: %s", pdf_path)
        raise PDFNotFoundError(f"PDF file not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        logger.error("Not a PDF file: %s", pdf_path)
        raise InvalidPDFError(f"Expected a .pdf file, got: {pdf_path}")

    resume_name = pdf_path.stem
    resume_output_dir = OUTPUT_DIR / "images" / resume_name
    resume_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading PDF %s (dpi=%d)", pdf_path, dpi)

    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        logger.error("Failed to open PDF %s: %s", pdf_path, exc)
        raise InvalidPDFError(f"Could not open PDF file: {pdf_path}") from exc

    try:
        if document.page_count == 0:
            logger.error("PDF has no pages: %s", pdf_path)
            raise InvalidPDFError(f"PDF file has no pages: {pdf_path}")

        zoom = dpi / _FITZ_BASE_DPI
        matrix = fitz.Matrix(zoom, zoom)

        image_paths: List[str] = []

        for page_index in range(document.page_count):
            page_number = page_index + 1
            image_filename = f"page_{page_number:03d}.png"
            image_path = resume_output_dir / image_filename

            try:
                page = document.load_page(page_index)
                pixmap = page.get_pixmap(matrix=matrix)
                pixmap.save(image_path)
            except Exception as exc:
                logger.error(
                    "Failed to render page %d of %s: %s", page_number, pdf_path, exc
                )
                raise PDFLoaderError(
                    f"Failed to render page {page_number} of {pdf_path}"
                ) from exc

            logger.info("Saved page %d -> %s", page_number, image_path)
            image_paths.append(image_path.relative_to(BASE_DIR).as_posix())
    finally:
        document.close()

    logger.info("Loaded %d page(s) from %s", len(image_paths), pdf_path)
    return image_paths
