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
    # Create a PdfMerger object
    merger = PdfMerger()

    # Get all PDF files in the directory
    pdf_files = [f for f in os.listdir(directory) if f.endswith(".pdf")]

    # Sort files to ensure a consistent merge order
    pdf_files.sort()

    # Loop through and append each PDF to the merger
    for pdf in pdf_files:
        pdf_path = os.path.join(directory, pdf)
        merger.append(pdf_path)
        print(f"Added: {pdf_path}")

    # Write the merged PDF to an output file
    output_path = os.path.join(directory, output_filename)
    merger.write(output_path)
    merger.close()
    print(f"Merged PDF saved as: {output_path}")

    return output_path


# Example usage
directory = "/Users/sanatwalia/Desktop/Assignments_applications/Vidhijna/commercial_laws"  # Replace with your directory path
output_file = "merged_output_laws.pdf"
merge_pdfs_in_directory(directory, output_file)
