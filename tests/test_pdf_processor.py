import pytest
import os
from pdf_split_and_count import count_pages_in_pdf

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