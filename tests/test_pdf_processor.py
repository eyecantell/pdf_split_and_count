import os
import logging
from pathlib import Path
import pytest
from pdf_split_and_count import split_double_page_pdf, count_pages_in_pdf
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
from pdf_orientation_corrector.main import detect_and_correct_orientation
from pdf2image import convert_from_path
import pytesseract

logger = logging.getLogger("pdf_split_and_count.image_processing")
logger.setLevel(logging.DEBUG)

@pytest.fixture
def tmp_path(tmp_path):
    return tmp_path

def test_orientation_detection(tmp_path, caplog):
    """Test that detect_and_correct_orientation correctly orients PDFs."""
    test_cases = [
        ("cap_rotated_90.pdf", "~270", 1.7778, 0.5625),
        ("cap_rotated_270.pdf", "~90", 0.5625, 1.7778),
        ("TenThingsDevLearnFull_first_two_sheets_portrait_on_one_page.pdf", "~90", 1.296, 0.771),
        ("TenThingsDevLearnFull_upside_down_first_two_sheets_on_one_page.pdf", "~180", 1.297, 1.297),
        ("pray_portrait_two_pages.pdf", "~90", 0.7727, 1.2941),
        ("pray_landscape_two_pages.pdf", "~90", 1.2941, 0.7727),  # Adjust if different
    ]
    
    for pdf_name, expected_angle_str, original_aspect, deskewed_aspect in test_cases:
        caplog.clear()
        pdf_path = os.path.join("tests", "pdfs_for_test", pdf_name)
        output_pdf = tmp_path / f"{pdf_name}_corrected.pdf"
        assert os.path.exists(pdf_path), f"PDF file not found at {pdf_path}"
        
        # Correct orientation
        caplog.set_level(logging.DEBUG, logger="pdf_orientation_corrector")
        detect_and_correct_orientation(pdf_path, output_pdf, dpi=300, batch_size=20, verbose=True)
        
        # Verify output
        images = convert_from_path(output_pdf, dpi=300, first_page=1, last_page=1)
        assert images, f"Failed to convert {pdf_name} to image"
        deskewed_image = images[0]
        
        # Verify aspect ratio
        deskewed_width, deskewed_height = deskewed_image.size
        deskewed_aspect_actual = deskewed_width / deskewed_height
        logger.debug(f"Deskewed size for {pdf_name}: {deskewed_width}x{deskewed_height}, Aspect: {deskewed_aspect_actual:.4f}")
        assert abs(deskewed_aspect_actual - deskewed_aspect) < 0.1, \
               f"Expected deskewed aspect {deskewed_aspect} for {pdf_name}, got {deskewed_aspect_actual}"
        
        # Verify text is right-side-up
        data = pytesseract.image_to_data(deskewed_image, config='--psm 6 -c textord_tabfind_find_tables=0', output_type=pytesseract.Output.DICT)
        top_texts = [text for i, text in enumerate(data['text']) if text.strip() and data['top'][i] < deskewed_height // 4]
        top_conf = sum(float(c) for i, c in enumerate(data['conf']) if data['text'][i].strip() and data['top'][i] < deskewed_height // 4 and float(c) > 0)
        assert len(top_texts) > 2 and top_conf > 100, \
               f"Text not right-side-up for {pdf_name}: insufficient top text (count: {len(top_texts)}, confidence: {top_conf:.2f})"

def test_count_pages_in_ten_things_pdf(tmp_path):
    """Test that count_pages_in_pdf correctly counts pages in a multi-page PDF."""
    pdf_path = os.path.join("tests", "pdfs_for_test", "TenThingsDevLearnFull.pdf")
    assert os.path.exists(pdf_path), f"PDF file not found at {pdf_path}"
    page_count = count_pages_in_pdf(pdf_path)
    assert page_count == 9, f"Expected 9 pages, got {page_count}"

def test_count_pages_in_single_page_pdf(tmp_path):
    """Test that count_pages_in_pdf correctly counts pages in a single-page PDF."""
    pdf_path = os.path.join("tests", "pdfs_for_test", "single_page_of_six_stages_of_debugging.pdf")
    assert os.path.exists(pdf_path), f"PDF file not found at {pdf_path}"
    page_count = count_pages_in_pdf(pdf_path)
    assert page_count == 1, f"Expected 1 page, got {page_count}"

def test_split_single_page_pdf(tmp_path):
    """Test that split_double_page_pdf does not split a single-page PDF."""
    pdf_path = os.path.join("tests", "pdfs_for_test", "single_page_of_six_stages_of_debugging.pdf")
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
    pdf_path = os.path.join("tests", "pdfs_for_test", "TenThingsDevLearnFull_first_two_sheets_portrait_on_one_page.pdf")
    assert os.path.exists(pdf_path), f"PDF file not found at {pdf_path}"
    output_dir = tmp_path / "split_output"
    caplog.set_level(logging.DEBUG, logger="pdf_split_and_count.image_processing")
    split_paths = split_double_page_pdf(pdf_path, output_dir)
    assert len(split_paths) == 2, f"Expected 2 output images, got {len(split_paths)}"

def test_split_upside_down_double_page_pdf(tmp_path, caplog):
    """Test that split_double_page_pdf splits an upside-down double-page PDF into two pages."""
    pdf_path = os.path.join("tests", "pdfs_for_test", "TenThingsDevLearnFull_upside_down_first_two_sheets_on_one_page.pdf")
    assert os.path.exists(pdf_path), f"PDF file not found at {pdf_path}"
    output_dir = tmp_path / "split_output"
    caplog.set_level(logging.DEBUG, logger="pdf_split_and_count.image_processing")
    split_paths = split_double_page_pdf(pdf_path, output_dir)
    assert len(split_paths) == 2, f"Expected 2 output images, got {len(split_paths)}"

def test_split_landscape_double_page_pdf(tmp_path, caplog):
    """Test that split_double_page_pdf splits a landscape double-page PDF into two pages per page."""
    pdf_path = os.path.join("tests", "pdfs_for_test", "TenThingsDevLearnFull_two_sheets_per_page.pdf")
    assert os.path.exists(pdf_path), f"PDF file not found at {pdf_path}"
    output_dir = tmp_path / "split_output"
    caplog.set_level(logging.DEBUG, logger="pdf_split_and_count.image_processing")
    split_paths = split_double_page_pdf(pdf_path, output_dir)
    assert len(split_paths) == 10, f"Expected 10 output images, got {len(split_paths)}"