"""
services/page_quality_classifier.py

Page Quality Classifier — Blank page detection (Milestone 15B.5).

Classifies a rendered page image as Blank or Non-Blank using pixel
statistics only, before it is ever sent to the vision model. This is
the first implemented category from the page-quality classification
framework designed in docs/PAGE_QUALITY_CLASSIFICATION.md (Milestone
15B.4), built specifically to prevent the confirmed hallucination
failure mode documented in docs/HALLUCINATION_INVESTIGATION.md
(Milestone 15B.3): a vision model given a completely blank page
reliably fabricates a generic resume template instead of recognizing
there is nothing to transcribe.

Detection is deterministic and uses no AI or vision model of any kind
— only grayscale pixel statistics computed via PyMuPDF (already a
project dependency, used by services/pdf_loader.py) and numpy. The
same image always produces the same classification.

Calibrated (Milestone 15B.5) against this project's own real data: the
two confirmed-blank pages found in docs/HALLUCINATION_INVESTIGATION.md
(output/images/Bhavita Jain_GAD_GT/page_002.png and
output/images/Dattatray S. Phad_Desktop Support Engineer_Intileo/page_004.png)
both measured an ink ratio of exactly 0.0, while the smallest
confirmed-sparse-but-real page found in that same investigation
(output/images/Rima_Karmakar_CV/page_002.png) measured 0.0089 (0.89%)
— roughly 9x the threshold chosen below, leaving a wide,
empirically-grounded margin between "blank" and "genuinely sparse."

Only Blank vs. Non-Blank is implemented here. Every other category in
docs/PAGE_QUALITY_CLASSIFICATION.md (Sparse, Normal, Poor Scan,
Rotated, Dark, Large Table, Multi-column) is out of scope for this
milestone and is treated as Non-Blank by this module.
"""

import logging
from pathlib import Path
from typing import Union

import fitz
import numpy as np

logger = logging.getLogger(__name__)

# A pixel is counted as "ink" (non-background) if its grayscale value
# is below this threshold, on a 0 (black) - 255 (white) scale.
INK_PIXEL_BRIGHTNESS_THRESHOLD = 250

# A page is classified Blank if its ink ratio (ink pixels / total
# pixels) is below this fraction. Calibrated against this project's
# own confirmed-blank (0.0 ink ratio) and confirmed-sparse (0.0089
# minimum observed) page images — see module docstring. This is a
# configurable constant, not a magic number inlined into the
# comparison below.
BLANK_INK_RATIO_THRESHOLD = 0.001


def is_blank_page(
    image_path: Union[str, Path],
    ink_pixel_brightness_threshold: int = INK_PIXEL_BRIGHTNESS_THRESHOLD,
    blank_ink_ratio_threshold: float = BLANK_INK_RATIO_THRESHOLD,
) -> bool:
    """Classify a rendered page image as Blank or Non-Blank.

    Deterministic, pixel-statistics-only classification — no AI or
    vision model is used. Converts the image to grayscale and computes
    the fraction of pixels darker than `ink_pixel_brightness_threshold`
    (the "ink ratio"); the page is classified Blank if that ratio
    falls below `blank_ink_ratio_threshold`.

    Args:
        image_path: Path to a rendered page image (e.g. a PNG produced
            by services/pdf_loader.py).
        ink_pixel_brightness_threshold: Grayscale value (0-255) below
            which a pixel counts as "ink." Defaults to
            INK_PIXEL_BRIGHTNESS_THRESHOLD; overridable per call
            (primarily for testing specific edge cases without
            touching the module-level constant other callers rely on).
        blank_ink_ratio_threshold: Ink-ratio fraction below which the
            page is classified Blank. Defaults to
            BLANK_INK_RATIO_THRESHOLD; overridable per call for the
            same reason as above.

    Returns:
        True if the page is classified Blank, False otherwise
        (Non-Blank — every other Page Quality Classification category
        from docs/PAGE_QUALITY_CLASSIFICATION.md is treated as
        Non-Blank by this function, since only Blank is implemented in
        this milestone).
    """
    ratio = _ink_ratio(image_path, ink_pixel_brightness_threshold)
    blank = bool(ratio < blank_ink_ratio_threshold)
    if blank:
        logger.info(
            "Detected blank page (ink ratio %.6f < threshold %.6f): %s",
            ratio,
            blank_ink_ratio_threshold,
            image_path,
        )
    return blank


def _ink_ratio(image_path: Union[str, Path], ink_pixel_brightness_threshold: int) -> float:
    """Fraction of pixels in `image_path` darker than
    `ink_pixel_brightness_threshold`, out of every pixel in the image."""
    pixmap = fitz.Pixmap(str(image_path))
    if pixmap.colorspace is not None and pixmap.colorspace.n > 1:
        pixmap = fitz.Pixmap(fitz.csGRAY, pixmap)

    pixels = np.frombuffer(pixmap.samples, dtype=np.uint8)
    pixels = pixels.reshape(pixmap.height, pixmap.width, pixmap.n)[:, :, 0]

    ink_pixel_count = np.count_nonzero(pixels < ink_pixel_brightness_threshold)
    return float(ink_pixel_count / pixels.size)
