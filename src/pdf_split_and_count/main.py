from pathlib import Path
from .pdf_handling import split_double_page_pdf, merge_pages_to_pdf
from .reporting import generate_page_count_report
from prepdir import configure_logging
import logging

logger = logging.getLogger(__name__)
configure_logging(logger, level="INFO")

def process_pdfs_in_folder(folder_path, output_dir="processed_pdfs"):
    """Process all PDFs in the folder: count pages, split, clean, and merge."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate page count report
    generate_page_count_report(folder_path)
    
    # Process each PDF
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                print(f"Processing {pdf_path}...")
                
                # Split double-page PDF
                split_image_paths = split_double_page_pdf(pdf_path, output_dir / Path(pdf_path).stem)
                
                # Merge back into a single PDF
                output_pdf = output_dir / f"{Path(pdf_path).stem}_processed.pdf"
                merge_pages_to_pdf(split_image_paths, output_pdf)
                
                # Clean up temporary image files
                for img_path in split_image_paths:
                    os.remove(img_path)

if __name__ == "__main__":
    folder_path = input("Enter the folder path containing PDFs: ")
    process_pdfs_in_folder(folder_path)