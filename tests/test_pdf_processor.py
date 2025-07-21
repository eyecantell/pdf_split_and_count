import os
import logging
from pathlib import Path
import pytest
from pdf_split_and_count.pdf_handling import split_double_page_pdf
from pdf_split_and_count.image_processing import deskew_image
from PIL import Image

# Configure logging
logger = logging.getLogger("pdf_split_and_count.image_processing")
logger.setLevel(logging.DEBUG)

@pytest.fixture
def tmp_path(tmp_path):
    return tmp_path

def test_orientation_detection(tmp_path, caplog):
    """Test that deskew_image correctly detects and corrects orientation for various rotations."""
    test_cases = [
        ("cap_rotated_90.pdf", "~270", None, 1.0),  # Expected -90-degree (270-degree) rotation
        ("cap_rotated_270.pdf", "~90", None, 1.0),  # Expected -270-degree (90-degree) rotation
        ("pray_portrait_two_pages.pdf", "~270", 0.771, 0.771),  # Expected ~270-degree rotation, aspect unchanged if rotation fails
        ("pray_landscape_two_pages.pdf", "~0", 1.297, 1.297),  # Expected ~0-degree rotation
    ]
    
    for pdf_name, expected_angle_str, original_aspect, deskewed_aspect in test_cases:
        caplog.clear()
        pdf_path = os.path.join("tests", "pdfs_for_test", pdf_name)
        assert os.path.exists(pdf_path), f"PDF file not found at {pdf_path}"
        
        # Load the first page as an image
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=1)
        assert images, f"Failed to convert {pdf_name} to image"
        original_image = images[0]
        
        # Deskew the image and capture logs
        caplog.set_level(logging.DEBUG, logger="pdf_split_and_count.image_processing")
        deskewed_image = deskew_image(original_image)
        
        # Verify orientation correction
        original_width, original_height = original_image.size
        deskewed_width, deskewed_height = deskewed_image.size
        logger.debug(f"Original size for {pdf_name}: {original_width}x{original_height}, Deskewed size: {deskewed_width}x{deskewed_height}")
        
        original_aspect_actual = original_width / original_height if original_aspect is None else original_aspect
        deskewed_aspect_actual = deskewed_width / deskewed_height
        logger.debug(f"Original aspect for {pdf_name}: {original_aspect_actual}, Deskewed aspect: {deskewed_aspect_actual}")
        
        # Check log for angle detection or fallback
        log_text = caplog.text.lower()
        assert any(msg in log_text for msg in ["tesseract osd angle", "fallback detected"]), \
               f"No orientation angle or fallback logged for {pdf_name}"
        angle_lines = [line for line in log_text.splitlines() if "tesseract osd angle" in line or "fallback detected" in line]
        if angle_lines:
            detected_angle = float(angle_lines[0].split(": ")[1].split()[0]) if "angle" in angle_lines[0] else \
                           (float(angle_lines[0].split(": ")[1]) if "fallback detected angle" in angle_lines[0] else 0.0)
            logger.debug(f"Detected angle for {pdf_name}: {detected_angle}")
            tolerance = 20.0 if abs(float(expected_angle_str.lstrip("~"))) != 90 else 10.0
            assert abs(detected_angle - float(expected_angle_str.lstrip("~"))) < tolerance, \
                   f"Expected angle {expected_angle_str} not detected for {pdf_name}, got {detected_angle}"
        
        # Expect aspect ratio to change only if rotation is applied
        if abs(detected_angle) > 10.0:  # Significant rotation
            assert abs(deskewed_aspect_actual - 1.0 / original_aspect_actual) < 0.1, \
                   f"Expected inverted aspect ratio for {pdf_name}, got {deskewed_aspect_actual}"
        else:
            assert abs(deskewed_aspect_actual - original_aspect_actual) < 0.1, \
                   f"Expected unchanged aspect ratio for {pdf_name}, got {deskewed_aspect_actual}"

def test_count_pages_in_ten_things_pdf(tmp_path):
    """Test that count_pages_in_pdf correctly counts pages in a multi-page PDF."""
    pdf_path = os.path.join("tests", "pdfs_for_test", "TenThingsDevLearnFull.pdf")
    assert os.path.exists(pdf_path), f"PDF file not found at {pdf_path}"
    page_count = count_pages_in_pdf(pdf_path)
    assert page_count == 9, f"Expected 9 pages, got {page_count}"

def test_count_pages_in_single_page_pdf(tmp_path):
    """Test that count_pages_in_pdf correctly counts pages in a single-page PDF."""
    pdf_path = os.path.join("tests", "pdfs_for_test", "single_page.pdf")
    assert os.path.exists(pdf_path), f"PDF file not found at {pdf_path}"
    page_count = count_pages_in_pdf(pdf_path)
    assert page_count == 1, f"Expected 1 page, got {page_count}"

def test_split_single_page_pdf(tmp_path):
    """Test that split_double_page_pdf does not split a single-page PDF."""
    pdf_path = os.path.join("tests", "pdfs_for_test", "single_page.pdf")
    assert os.path.exists(pdf_path), f"PDF file not found at {pdf_path}"
    output_dir = tmp_path / "split_output"
    split_paths = split_double_page_pdf(pdf_path, output_dir)
    assert len(split_paths) == 1, f"Expected 1 output image, got {len(split_paths)}"

def test_split_ten_things_pdf(tmp_path, caplog):
    """Test that split_double_page_pdf handles a multi-page PDF with two-column layout."""
    pdf_path = os.path.join("tests", "pdfs_for_test", "TenThingsDevLearnFull.pdf")
    assert os.path.exists(pdf_path), f"PDF file not found at {pdf_path}"
    output_dir = tmp_path / "split_output"
    caplog.set_level(logging.DEBUG, logger="pdf_split_and_count.image_processing")
    split_paths = split_double_page_pdf(pdf_path, output_dir)
    assert len(split_paths) == 9, f"Expected 9 output images, got {len(split_paths)}"

def test_split_portrait_double_page_pdf(tmp_path, caplog):
    """Test that split_double_page_pdf splits a portrait double-page PDF into two pages."""
    pdf_path = os.path.join("tests", "pdfs_for_test", "pray_portrait_two_pages.pdf")
    assert os.path.exists(pdf_path), f"PDF file not found at {pdf_path}"
    output_dir = tmp_path / "split_output"
    caplog.set_level(logging.DEBUG, logger="pdf_split_and_count.image_processing")
    split_paths = split_double_page_pdf(pdf_path, output_dir)
    assert len(split_paths) == 2, f"Expected 2 output images, got {len(split_paths)}"

def test_split_landscape_double_page_pdf(tmp_path, caplog):
    """Test that split_double_page_pdf splits a landscape double-page PDF into two pages."""
    pdf_path = os.path.join("tests", "pdfs_for_test", "pray_landscape_two_pages.pdf")
    assert os.path.exists(pdf_path), f"PDF file not found at {pdf_path}"
    output_dir = tmp_path / "split_output"
    caplog.set_level(logging.DEBUG, logger="pdf_split_and_count.image_processing")
    split_paths = split_double_page_pdf(pdf_path, output_dir)
    assert len(split_paths) == 2, f"Expected 2 output images, got {len(split_paths)}"