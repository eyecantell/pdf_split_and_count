from .pdf_processor import process_pdfs_in_folder

if __name__ == "__main__":
    folder_path = input("Enter the folder path containing PDFs: ")
    process_pdfs_in_folder(folder_path)