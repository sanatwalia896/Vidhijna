"""
tools/ocr.py — Document text extraction

Handles:
  - Text PDFs      → pdfplumber
  - Scanned PDFs   → pytesseract (OCR)
  - Images         → pytesseract
  - DOCX           → python-docx
"""

import io
from pathlib import Path


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a text-based PDF using pdfplumber."""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(text.strip())
        return "\n\n".join(p for p in pages if p)
    except Exception as e:
        print(f"[OCR] pdfplumber failed: {e}")
        return ""


def extract_text_from_image(file_bytes: bytes) -> str:
    """Extract text from scanned image using pytesseract."""
    try:
        import pytesseract
        from PIL import Image
        image = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(image, lang="eng")
    except Exception as e:
        print(f"[OCR] pytesseract failed: {e}")
        return ""


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a .docx file."""
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        print(f"[OCR] docx extraction failed: {e}")
        return ""


def extract_text(file_bytes: bytes, filename: str) -> tuple[str, str]:
    """
    Auto-detect file type and extract text.
    Returns (extracted_text, detected_file_type).
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        text = extract_text_from_pdf(file_bytes)
        # If PDF extraction got very little text, try OCR (scanned PDF)
        if len(text.split()) < 50:
            print("[OCR] Low text from pdfplumber — attempting OCR")
            text = _ocr_pdf_pages(file_bytes) or text
        return text, "pdf"

    elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
        return extract_text_from_image(file_bytes), "image"

    elif ext == ".docx":
        return extract_text_from_docx(file_bytes), "docx"

    elif ext == ".txt":
        return file_bytes.decode("utf-8", errors="ignore"), "txt"

    else:
        return "", "unknown"


def _ocr_pdf_pages(file_bytes: bytes) -> str:
    """Convert PDF pages to images and OCR each page."""
    try:
        import pytesseract
        from PIL import Image
        import pdfplumber

        pages_text = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                img = page.to_image(resolution=200).original
                text = pytesseract.image_to_string(img, lang="eng")
                pages_text.append(text)
        return "\n\n".join(pages_text)
    except Exception as e:
        print(f"[OCR] PDF OCR failed: {e}")
        return ""


def detect_document_type(text: str, filename: str) -> str:
    """
    Classify uploaded document type for routing.
    Returns: "contract" | "judgment" | "notice" | "act" | "other"
    """
    text_lower = text[:1000].lower()
    filename_lower = filename.lower()

    if any(w in text_lower for w in ["agreement", "contract", "parties", "whereas", "hereinafter"]):
        return "contract"
    if any(w in text_lower for w in ["judgment", "order", "petitioner", "respondent", "coram"]):
        return "judgment"
    if any(w in text_lower for w in ["notice", "demand", "take notice", "legal notice"]):
        return "notice"
    if any(w in text_lower for w in ["act no.", "be it enacted", "short title", "extent"]):
        return "act"
    return "other"