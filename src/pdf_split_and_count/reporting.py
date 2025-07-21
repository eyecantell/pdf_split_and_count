import os
import csv
from .pdf_handling import count_pages_in_pdf

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