import os
import csv
from pypdf import PdfReader, PdfWriter
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np
from pathlib import Path

def count_pages_in_pdf(pdf_path):
    """Count total pages in a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf = PdfReader(file)
            return len(pdf.pages)
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return 0

def generate_page_count_report(folder_path, output_csv="page_count_report.csv"):
    """Recursively scan folder and generate CSV with page counts."""
    page_counts = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                page_count = count_pages_in_pdf(pdf_path)
                page_counts.append((pdf_path, page_count))
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File Path', 'Page Count'])
        for pdf_path, count in page_counts:
            writer.writerow([pdf_path, count])
    print(f"Page count report saved to {output_csv}")

def clean_image(image):
    """Remove punch holes, borders, and artifacts from an image."""
    # Convert PIL image to OpenCV format
    img_array = np.array(image)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Remove punch holes (detect circles and fill them)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                              param1=50, param2=30, minRadius=5, maxRadius=20)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (255, 255, 255), -1)
    
    # Remove dark borders and shadows (thresholding)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(thresh)
    cleaned = cv2.bitwise_and(img, img, mask=mask)
    
    # Convert back to PIL image
    cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cleaned_rgb)

def split_double_page_pdf(pdf_path, output_dir):
    """Split PDFs with two pages per sheet into individual pages."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert PDF pages to images
    images = convert_from_path(pdf_path)
    split_pages = []
    
    for i, image in enumerate(images):
        width, height = image.size
        # Split image into two equal parts (left and right)
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