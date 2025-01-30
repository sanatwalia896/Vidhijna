import os
from PyPDF2 import PdfMerger


def merge_pdfs_in_directory(directory, output_filename="merged.pdf"):
    """
    Merges all PDF files in a specified directory into a single PDF.

    Args:
        directory (str): The directory containing the PDF files.
        output_filename (str): The name of the output merged PDF file.

    Returns:
        str: Path to the merged PDF file.
    """
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return None

    # Create a PdfMerger object
    merger = PdfMerger()

    # Get all PDF files in the directory
    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith(".pdf")]

    # Ensure there are PDFs to merge
    if not pdf_files:
        print("Error: No PDF files found in the directory.")
        return None

    # Sort files numerically (if filenames contain numbers)
    pdf_files.sort(
        key=lambda f: (
            int("".join(filter(str.isdigit, f))) if any(c.isdigit() for c in f) else f
        )
    )

    # Merge PDFs
    for pdf in pdf_files:
        pdf_path = os.path.join(directory, pdf)
        try:
            merger.append(pdf_path)
            print(f"Added: {pdf_path}")
        except Exception as e:
            print(f"Error adding {pdf_path}: {e}")

    # Write the merged PDF to an output file
    output_path = os.path.join(directory, output_filename)
    try:
        merger.write(output_path)
        merger.close()
        print(f"Merged PDF saved as: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error saving merged PDF: {e}")
        return None


# Example usage
directory = "/Users/sanatwalia/Desktop/Assignments_applications/Vidhijna/commercial_cases"  # Replace with your directory path
output_file = "merged_output_cases.pdf"
merge_pdfs_in_directory(directory, output_file)
