"""
tools/ocr.py — Document text extraction

Handles:
  - Text PDFs      → pdfplumber
  - Scanned PDFs   → ChatGroq vision LLM (page-by-page)
  - Images         → ChatGroq vision LLM
  - DOCX           → python-docx
"""

import base64
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


def extract_text_from_image(file_bytes: bytes, mime_type: str = "image/png") -> str:
    """Extract text from a scanned image using ChatGroq vision LLM."""
    return _extract_text_via_vision_llm(file_bytes, mime_type)


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
        # If PDF extraction got very little text, fall back to vision LLM (scanned PDF)
        if len(text.split()) < 50:
            print("[OCR] Low text from pdfplumber — attempting vision LLM OCR")
            text = _ocr_pdf_pages(file_bytes) or text
        return text, "pdf"

    elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
        _mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".tiff": "image/tiff",
            ".bmp": "image/bmp",
        }
        return extract_text_from_image(file_bytes, _mime_map.get(ext, "image/png")), "image"

    elif ext == ".docx":
        return extract_text_from_docx(file_bytes), "docx"

    elif ext == ".txt":
        return file_bytes.decode("utf-8", errors="ignore"), "txt"

    else:
        return "", "unknown"


def _extract_text_via_vision_llm(image_bytes: bytes, mime_type: str = "image/png") -> str:
    """Send an image to ChatGroq vision model and return extracted text."""
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64}"

        llm = ChatGroq(model="llama-3.2-11b-vision-preview", temperature=0)
        message = HumanMessage(content=[
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            },
            {
                "type": "text",
                "text": (
                    "Extract all text from this image exactly as it appears. "
                    "Return only the extracted text with no commentary or explanation."
                ),
            },
        ])
        response = llm.invoke([message])
        return response.content or ""
    except Exception as e:
        print(f"[OCR] Vision LLM extraction failed: {e}")
        return ""


def _ocr_pdf_pages(file_bytes: bytes) -> str:
    """Convert each PDF page to an image and extract text via vision LLM."""
    try:
        import pdfplumber

        pages_text = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                img = page.to_image(resolution=200).original
                # Convert PIL image to PNG bytes
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                page_bytes = buf.getvalue()
                print(f"[OCR] Processing page {i + 1} via vision LLM")
                text = _extract_text_via_vision_llm(page_bytes, "image/png")
                pages_text.append(text)
        return "\n\n".join(pages_text)
    except Exception as e:
        print(f"[OCR] PDF vision OCR failed: {e}")
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
