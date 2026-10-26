"""
tests/test_page_quality_classifier.py

Tests for services/page_quality_classifier.py (Milestone 15B.5).

Every test builds small synthetic PNG images directly via PyMuPDF
(fitz) and numpy — the same two libraries the classifier itself uses
— so these tests have no dependency on Ollama, any vision model, or
any external image library. Detection is deterministic pixel
statistics only; every assertion here checks an exact, reproducible
value, not a range.

A dedicated regression class at the bottom re-checks the classifier
against the real project images that grounded the threshold choice in
docs/HALLUCINATION_INVESTIGATION.md: the two confirmed-blank pages
(which must classify Blank) and the three confirmed-sparse-but-real
pages (which must not).
"""

from pathlib import Path

import fitz
import numpy as np
import pytest

from services.page_quality_classifier import (
    BLANK_INK_RATIO_THRESHOLD,
    INK_PIXEL_BRIGHTNESS_THRESHOLD,
    is_blank_page,
)


def _save_grayscale_png(path: Path, pixels: np.ndarray) -> None:
    """Save a 2D uint8 numpy array as a single-channel (grayscale) PNG."""
    height, width = pixels.shape
    pixmap = fitz.Pixmap(fitz.csGRAY, width, height, pixels.tobytes(), False)
    pixmap.save(str(path))


def _save_rgb_png(path: Path, pixels: np.ndarray) -> None:
    """Save an (H, W) uint8 numpy array as a 3-channel RGB PNG, by
    repeating it across channels — matching the colorspace
    services/pdf_loader.py actually renders pages in (fitz's default
    RGB pixmap), so this exercises the classifier's grayscale
    conversion branch, not just the already-grayscale path."""
    height, width = pixels.shape
    rgb = np.stack([pixels, pixels, pixels], axis=-1)
    pixmap = fitz.Pixmap(fitz.csRGB, width, height, rgb.tobytes(), False)
    pixmap.save(str(path))


def _blank_pixels(height=100, width=100) -> np.ndarray:
    return np.full((height, width), 255, dtype=np.uint8)


class TestBlankPages:
    def test_all_white_grayscale_page_is_blank(self, tmp_path):
        path = tmp_path / "blank_gray.png"
        _save_grayscale_png(path, _blank_pixels())

        assert is_blank_page(path) is True

    def test_all_white_rgb_page_is_blank(self, tmp_path):
        """Real pages are rendered as RGB by services/pdf_loader.py —
        this exercises the classifier's grayscale-conversion path."""
        path = tmp_path / "blank_rgb.png"
        _save_rgb_png(path, _blank_pixels())

        assert is_blank_page(path) is True

    def test_near_blank_page_below_threshold_is_blank(self, tmp_path):
        """A page with a tiny number of dark pixels — enough that the
        page isn't byte-identical to pure white, but few enough that
        the ink ratio still falls below BLANK_INK_RATIO_THRESHOLD —
        must still classify as Blank."""
        pixels = _blank_pixels(height=200, width=200)  # 40,000 pixels total
        # 30 dark pixels -> ink ratio 0.00075, below the 0.001 default.
        pixels[0, :30] = 0
        path = tmp_path / "near_blank.png"
        _save_grayscale_png(path, pixels)

        assert is_blank_page(path) is True


class TestNonBlankPages:
    def test_sparse_page_above_threshold_is_not_blank(self, tmp_path):
        """Just enough dark pixels to cross BLANK_INK_RATIO_THRESHOLD."""
        pixels = _blank_pixels(height=200, width=200)  # 40,000 pixels total
        # 50 dark pixels -> ink ratio 0.00125, above the 0.001 default.
        pixels[0, :50] = 0
        path = tmp_path / "sparse.png"
        _save_grayscale_png(path, pixels)

        assert is_blank_page(path) is False

    def test_normal_dense_content_page_is_not_blank(self, tmp_path):
        rng_pixels = _blank_pixels(height=200, width=200)
        rng_pixels[20:180, 20:180] = 0  # a large dense block of "text"
        path = tmp_path / "normal.png"
        _save_grayscale_png(path, rng_pixels)

        assert is_blank_page(path) is False

    def test_dark_page_is_not_blank(self, tmp_path):
        """A solid black (or otherwise dark) page is the opposite
        extreme from a blank/white page and must never be classified
        Blank — Blank means 'background-colored,' not merely
        'uniform.'"""
        pixels = np.full((100, 100), 10, dtype=np.uint8)  # near-black
        path = tmp_path / "dark.png"
        _save_grayscale_png(path, pixels)

        assert is_blank_page(path) is False

    def test_dark_rgb_page_is_not_blank(self, tmp_path):
        pixels = np.full((100, 100), 10, dtype=np.uint8)
        path = tmp_path / "dark_rgb.png"
        _save_rgb_png(path, pixels)

        assert is_blank_page(path) is False


class TestThresholdConfigurability:
    def test_custom_ink_ratio_threshold_overrides_default(self, tmp_path):
        """A page just above the default threshold is Non-Blank by
        default, but classifies Blank if the caller passes a looser
        (higher) threshold."""
        pixels = _blank_pixels(height=200, width=200)
        pixels[0, :50] = 0  # ink ratio 0.00125
        path = tmp_path / "custom_threshold.png"
        _save_grayscale_png(path, pixels)

        assert is_blank_page(path) is False
        assert is_blank_page(path, blank_ink_ratio_threshold=0.01) is True

    def test_custom_brightness_threshold_changes_what_counts_as_ink(self, tmp_path):
        """A mid-gray page counts as 'ink' only if the brightness
        threshold is set high enough to include it."""
        pixels = np.full((100, 100), 200, dtype=np.uint8)  # light gray
        path = tmp_path / "light_gray.png"
        _save_grayscale_png(path, pixels)

        # Default threshold (250) counts anything darker than 250 as
        # ink, so this page (all pixels = 200) is 100% ink -> Non-Blank.
        assert is_blank_page(path) is False
        # A stricter threshold that doesn't count light gray as ink at
        # all makes the same page read as 0% ink -> Blank.
        assert is_blank_page(path, ink_pixel_brightness_threshold=100) is True

    def test_module_constants_match_documented_defaults(self):
        """Sanity check that the configurable constants are what the
        module docstring and Milestone 15B.5's calibration describe."""
        assert INK_PIXEL_BRIGHTNESS_THRESHOLD == 250
        assert BLANK_INK_RATIO_THRESHOLD == 0.001


class TestDeterminism:
    def test_repeated_calls_on_the_same_image_agree(self, tmp_path):
        path = tmp_path / "repeat.png"
        _save_grayscale_png(path, _blank_pixels())

        results = [is_blank_page(path) for _ in range(5)]

        assert results == [True] * 5

    def test_accepts_str_and_path(self, tmp_path):
        path = tmp_path / "blank_str.png"
        _save_grayscale_png(path, _blank_pixels())

        assert is_blank_page(path) == is_blank_page(str(path))


class TestLogging:
    def test_logs_detected_blank_page(self, tmp_path, caplog):
        path = tmp_path / "blank_logged.png"
        _save_grayscale_png(path, _blank_pixels())

        with caplog.at_level("INFO", logger="services.page_quality_classifier"):
            is_blank_page(path)

        assert any("Detected blank page" in record.message for record in caplog.records)

    def test_does_not_log_blank_detection_for_non_blank_page(self, tmp_path, caplog):
        pixels = _blank_pixels(height=200, width=200)
        pixels[20:180, 20:180] = 0
        path = tmp_path / "normal_logged.png"
        _save_grayscale_png(path, pixels)

        with caplog.at_level("INFO", logger="services.page_quality_classifier"):
            is_blank_page(path)

        assert not any("Detected blank page" in record.message for record in caplog.records)


class TestRealProjectImageRegression:
    """Re-checks the classifier against the exact real images that
    grounded BLANK_INK_RATIO_THRESHOLD's calibration in
    docs/HALLUCINATION_INVESTIGATION.md. Skips itself if this
    project's real output/images/ tree isn't present (e.g. a fresh
    checkout with no pipeline runs yet), rather than failing."""

    CONFIRMED_BLANK_IMAGES = [
        "output/images/Bhavita Jain_GAD_GT/page_002.png",
        "output/images/Dattatray S. Phad_Desktop Support Engineer_Intileo/page_004.png",
    ]
    CONFIRMED_SPARSE_IMAGES = [
        "output/images/Rima_Karmakar_CV/page_002.png",
        "output/images/Anandu Chandran_Area Credit Manager_SMFG-Intileo/page_004.png",
        "output/images/PRAGNESH S. SHAH_Sr. Associate-IT Ops_Intileo/page_003.png",
    ]

    @pytest.mark.parametrize("image_path", CONFIRMED_BLANK_IMAGES)
    def test_confirmed_blank_pages_classify_blank(self, image_path):
        if not Path(image_path).is_file():
            pytest.skip(f"{image_path} not present in this checkout; skipping.")
        assert is_blank_page(image_path) is True

    @pytest.mark.parametrize("image_path", CONFIRMED_SPARSE_IMAGES)
    def test_confirmed_sparse_pages_classify_non_blank(self, image_path):
        if not Path(image_path).is_file():
            pytest.skip(f"{image_path} not present in this checkout; skipping.")
        assert is_blank_page(image_path) is False
