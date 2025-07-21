import pytest
import os
from pdf_split_and_count import count_pages_in_pdf, split_double_page_pdf, deskew_image
from pypdf import PdfReader
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def test_count_pages_in_ten_things_pdf():
    """Test that count_pages_in_pdf correctly counts 9 pages in TenThingsDevLearnFull.pdf."""
    pdf_path = os.path.join("tests", "pdfs_for_test", "TenThingsDevLearnFull.pdf")
    assert os.path.exists(pdf_path), f"PDF file not found at {pdf_path}"
    page_count = count_pages_in_pdf(pdf_path)
    assert page_count == 9, f"Expected 9 pages, but got {page_count}"

def test_count_pages_in_single_page_pdf():
    """Test that count_pages_in_pdf correctly counts 1 page in single_page_of_six_stages_of_debugging.pdf."""
    pdf_path = os.path.join("tests", "pdfs_for_test", "single_page_of_six_stages_of_debugging.pdf")
    assert os.path.exists(pdf_path), f"PDF file not found at {pdf_path}"
    page_count = count_pages_in_pdf(pdf_path)
    assert page_count == 1, f"Expected 1 page, but got {page_count}"

def test_split_single_page_pdf(tmp_path):
    """Test that split_double_page_pdf does not split a single-page PDF."""
    pdf_path = os.path.join("tests", "pdfs_for_test", "single_page_of_six_stages_of_debugging.pdf")
    assert os.path.exists(pdf_path), f"PDF file not found at {pdf_path}"
    output_dir = tmp_path / "split_output"
    split_paths = split_double_page_pdf(pdf_path, output_dir)
    assert len(split_paths) == 1, f"Expected 1 output image, got {len(split_paths)}"
    assert split_paths[0].name == "single_page_of_six_stages_of_debugging_page_1.png", \
        f"Expected single_page_of_six_stages_of_debugging_page_1.png, got {split_paths[0].name}"

def test_split_ten_things_pdf(tmp_path):
    """Test that split_double_page_pdf does not split a multi-page single-page PDF."""
    pdf_path = os.path.join("tests", "pdfs_for_test", "TenThingsDevLearnFull.pdf")
    assert os.path.exists(pdf_path), f"PDF file not found at {pdf_path}"
    output_dir = tmp_path / "split_output"
    split_paths = split_double_page_pdf(pdf_path, output_dir)
    assert len(split_paths) == 9, f"Expected 9 output images, got {len(split_paths)}"
    for i, path in enumerate(split_paths, 1):
        assert path.name == f"TenThingsDevLearnFull_page_{i}.png", \
            f"Expected TenThingsDevLearnFull_page_{i}.png, got {path.name}"

def test_split_double_page_pdf(tmp_path):
    """Test that split_double_page_pdf splits a double-page PDF into two pages."""
    pdf_path = os.path.join("tests", "pdfs_for_test", "pray_landsacpe_two_pages.pdf")
    assert os.path.exists(pdf_path), f"PDF file not found at {pdf_path}"
    output_dir = tmp_path / "split_output"
    split_paths = split_double_page_pdf(pdf_path, output_dir)
    assert len(split_paths) == 2, f"Expected 2 output images, got {len(split_paths)}"
    assert split_paths[0].name == "pray_landsacpe_two_pages_page_1.png", \
        f"Expected pray_landsacpe_two_pages_page_1.png, got {split_paths[0].name}"
    assert split_paths[1].name == "pray_landsacpe_two_pages_page_2.png", \
        f"Expected pray_landsacpe_two_pages_page_2.png, got {split_paths[1].name}"