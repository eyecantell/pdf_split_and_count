import os
from pathlib import Path
import warnings
from pypdf import PdfReader, PdfWriter
from pdf2image import convert_from_path
from PIL import Image
from .image_processing import detect_double_page, clean_image
from .utils import suppress_output
import logging

logger = logging.getLogger(__name__)

# Suppress PyPDF2 deprecation and syntax warnings from pdf_orientation_corrector
warnings.filterwarnings("ignore", category=DeprecationWarning, module="PyPDF2")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pdf_orientation_corrector")
from pdf_orientation_corrector.main import detect_and_correct_orientation

def count_pages_in_pdf(pdf_path):
    """Count the number of pages in a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            return len(pdf_reader.pages)
    except Exception as e:
        logger.error(f"Error reading {pdf_path}: {e}")
        return 0

def split_double_page_pdf(pdf_path, output_dir):
    """Split a PDF with double-page spreads into single pages."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_image_paths = []
    
    # Correct orientation first, suppressing pdf_orientation_corrector output
    temp_pdf = output_dir / "temp_corrected.pdf"
    with suppress_output(suppress_logging_for="pdf_orientation_corrector"):
        detect_and_correct_orientation(pdf_path, temp_pdf, dpi=300, batch_size=20)
    
    # Convert corrected PDF to images
    try:
        images = convert_from_path(temp_pdf, dpi=300)
    except Exception as e:
        logger.error(f"Error converting {temp_pdf} to images: {e}")
        return []
    
    for i, image in enumerate(images):
        # Detect if the page is a double-page spread
        is_double_page, split_info = detect_double_page(image)
        if is_double_page:
            if split_info == "horizontal":
                left_half = image.crop((0, 0, image.width // 2, image.height))
                right_half = image.crop((image.width // 2, 0, image.width, image.height))
                split_images = [left_half, right_half]
            else:  # vertical
                top_half = image.crop((0, 0, image.width, image.height // 2))
                bottom_half = image.crop((0, image.height // 2, image.width, image.height))
                split_images = [top_half, bottom_half]
        else:
            split_images = [image]
        
        # Clean and save split images
        for j, split_image in enumerate(split_images):
            cleaned_image = split_image #clean_image(split_image)
            output_path = output_dir / f"{Path(pdf_path).stem}_page_{i+1}_{j+1}.png"
            cleaned_image.save(output_path)
            split_image_paths.append(output_path)
    
    # Clean up temporary PDF
    if temp_pdf.exists():
        os.remove(temp_pdf)
    
    return split_image_paths

def merge_pages_to_pdf(image_paths, output_pdf):
    """Merge a list of image files into a single PDF."""
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create a new PDF writer
        pdf_writer = PdfWriter()
        
        for img_path in sorted(image_paths):  # Sort to maintain order
            logger.info(f"Adding image {img_path} to PDF")
            try:
                # Open image with PIL
                img = Image.open(img_path)
                # Convert to RGB if necessary (PDF doesn't support RGBA)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                # Save image to a temporary PDF
                temp_pdf = output_pdf.parent / f"temp_{Path(img_path).name}.pdf"
                img.save(temp_pdf, format='PDF', resolution=300)
                
                # Read the temporary PDF and add its page to the writer
                with open(temp_pdf, 'rb') as temp_file:
                    temp_reader = PdfReader(temp_file)
                    pdf_writer.add_page(temp_reader.pages[0])
                
                # Clean up temporary PDF
                os.remove(temp_pdf)
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {e}")
                continue
        
        # Write the final PDF
        with open(output_pdf, 'wb') as output_file:
            pdf_writer.write(output_file)
        logger.info(f"Merged PDF saved to {output_pdf}")
        
    except Exception as e:
        logger.error(f"Error merging images to PDF {output_pdf}: {e}")
        return
    
    return output_pdf