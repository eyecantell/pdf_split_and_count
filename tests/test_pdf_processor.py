import os
import logging
from pathlib import Path
import pytest
from pdf_split_and_count import split_double_page_pdf, count_pages_in_pdf
import warnings
from pdf2image import convert_from_path
# Suppress PyPDF2 deprecation and syntax warnings from pdf_orientation_corrector
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pdf_orientation_corrector.main")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pdf_orientation_corrector.main")
from pdf_orientation_corrector.main import detect_and_correct_orientation

logger = logging.getLogger("pdf_split_and_count.image_processing")
logger.setLevel(logging.DEBUG)

@pytest.fixture
def tmp_path(tmp_path):
    return tmp_path

def test_orientation_detection(tmp_path, caplog, capsys):
    """Test that detect_and_correct_orientation correctly orients PDFs by checking stdout."""
    test_cases = [
        ("cap_rotated_90.pdf", ["Page 0 needs -90 degrees rotation"]),
        ("cap_rotated_270.pdf", ["Page 0 needs 90 degrees rotation"]),
        ("TenThingsDevLearnFull_first_two_sheets_portrait_on_one_page.pdf", ["Page 0 needs 90 degrees rotation"]),
        ("TenThingsDevLearnFull_upside_down_first_two_sheets_on_one_page.pdf", ["Page 0 is upside down. Needs 180 degrees rotation"]),
        ("pray_portrait_two_pages.pdf", ["Page 0 needs 90 degrees rotation"]),
        ("pray_landscape_two_pages.pdf", ["Rotation angle detected: 0 degrees"]),
    ]
    
    for pdf_name, expected_messages in test_cases:
        caplog.clear()
        pdf_path = os.path.join("tests", "pdfs_for_test", pdf_name)
        output_pdf = tmp_path / f"{pdf_name}_corrected.pdf"
        assert os.path.exists(pdf_path), f"PDF file not found at {pdf_path}"
        
        # Correct orientation
        caplog.set_level(logging.DEBUG, logger="pdf_orientation_corrector")
        detect_and_correct_orientation(pdf_path, output_pdf, dpi=300)
        
        # Capture stdout and verify the expected rotation messages
        captured = capsys.readouterr()
        logger.info(f"STDOUT for {pdf_name} is: {captured.out}\n--")
        for message in expected_messages:
            assert message in captured.out, (
                f"For {pdf_name}, expected stdout to contain '{message}', "
                f"but got: {captured.out}"
            )
        

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