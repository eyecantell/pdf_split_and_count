import os
from pypdf import PdfReader, PdfWriter
from pdf2image import convert_from_path
from PIL import Image
from pathlib import Path
from .image_processing import clean_image, detect_double_page

def count_pages_in_pdf(pdf_path):
    """Count total pages in a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf = PdfReader(file)
            return len(pdf.pages)
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return 0

def split_double_page_pdf(pdf_path, output_dir):
    """Split PDFs with two pages per sheet into individual pages if double-page layout is detected."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert PDF pages to images
    images = convert_from_path(pdf_path)
    split_pages = []
    
    for i, image in enumerate(images):
        # Check if the page is double-page using OCR
        if detect_double_page(image):
            width, height = image.size
            # Split into two equal parts (left and right)
            left_page = image.crop((0, 0, width // 2, height))
            right_page = image.crop((width // 2, 0, width, height))
            
            # Clean both pages
            left_page = clean_image(left_page)
            right_page = clean_image(right_page)
            
            # Save split pages
            left_path = output_dir / f"{Path(pdf_path).stem}_page_{i*2+1}.png"
            right_path = output_dir / f"{Path(pdf_path).stem}_page_{i*2+2}.png"
            left_page.save(left_path, quality=95)
            right_page.save(right_path, quality=95)
            split_pages.extend([left_path, right_path])
        else:
            # Single-page layout, process as is
            cleaned_page = clean_image(image)
            page_path = output_dir / f"{Path(pdf_path).stem}_page_{i+1}.png"
            cleaned_page.save(page_path, quality=95)
            split_pages.append(page_path)
    
    return split_pages

def merge_pages_to_pdf(image_paths, output_pdf_path):
    """Merge images back into a single PDF."""
    if not image_paths:
        return
    
    # Load images and convert to PDF-compatible format
    images = [Image.open(path).convert('RGB') for path in image_paths]
    
    # Create a new PDF
    pdf_writer = PdfWriter()
    
    # Convert each image to PDF page
    for image in images:
        # Save image to a temporary PDF
        temp_pdf = f"temp_{id(image)}.pdf"
        image.save(temp_pdf, "PDF", resolution=100.0)
        
        # Read the temporary PDF and add its pages to the writer
        temp_reader = PdfReader(temp_pdf)
        for page in temp_reader.pages:
            pdf_writer.add_page(page)
        
        # Clean up temporary PDF
        os.remove(temp_pdf)
    
    # Save the final PDF
    with open(output_pdf_path, 'wb') as output_file:
        pdf_writer.write(output_file)
    print(f"Merged PDF saved to {output_pdf_path}")