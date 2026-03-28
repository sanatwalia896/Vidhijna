import base64
import io
import gc
import time
from pathlib import Path

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a text-based PDF using pdfplumber."""
    try:
        import pdfplumber
        print(f"[OCR] Starting text extraction with pdfplumber...")
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append(text.strip())
                if i % 5 == 0: print(f"[OCR] Read {i+1} text pages...")
        return "\n\n".join(p for p in pages if p)
    except Exception as e:
        print(f"[OCR] pdfplumber failed: {e}")
        return ""

def extract_text_from_image(file_bytes: bytes, mime_type: str = "image/jpeg") -> str:
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
    """Auto-detect file type and extract text with memory logging."""
    ext = Path(filename).suffix.lower()
    start_time = time.time()
    print(f"\n[OCR START] Processing: {filename} ({len(file_bytes)/1024:.1f} KB)")

    if ext == ".pdf":
        text = extract_text_from_pdf(file_bytes)
        # Fallback for scanned PDFs
        if len(text.split()) < 50:
            print("[OCR] Low text count detected. Switching to Vision LLM (Scanned PDF Mode)...")
            text = _ocr_pdf_pages(file_bytes) or text
        res = text, "pdf"

    elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
        res = extract_text_from_image(file_bytes, "image/jpeg"), "image"

    elif ext == ".docx":
        res = extract_text_from_docx(file_bytes), "docx"

    elif ext == ".txt":
        res = file_bytes.decode("utf-8", errors="ignore"), "txt"
    else:
        res = "", "unknown"

    print(f"[OCR END] Completed in {time.time() - start_time:.2f}s\n")
    return res

def _extract_text_via_vision_llm(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Send image to Groq. Uses JPEG to keep payload small."""
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64}"

        llm = ChatGroq(model="llama-3.2-11b-vision-preview", temperature=0)
        message = HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": "Extract all text from this image. Return only text."}
        ])
        response = llm.invoke([message])
        return response.content or ""
    except Exception as e:
        print(f"[OCR] Vision LLM error: {e}")
        return ""

def _ocr_pdf_pages(file_bytes: bytes) -> str:
    """Memory-optimized PDF scanning."""
    try:
        import pdfplumber
        pages_text = []
        
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            total = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                print(f"[OCR] Processing Page {i+1}/{total}...")
                
                # OPTIMIZATION: 150 DPI JPEG (much lighter than 200 DPI PNG)
                img = page.to_image(resolution=150).original
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=80)
                
                page_bytes = buf.getvalue()
                text = _extract_text_via_vision_llm(page_bytes, "image/jpeg")
                pages_text.append(text)
                
                # MEMORY PURGE: Explicitly clear image objects
                buf.close()
                del img
                del page_bytes
                gc.collect() # Force free RAM
                
        return "\n\n".join(pages_text)
    except Exception as e:
        print(f"[OCR] PDF Vision failed: {e}")
        return ""

def detect_document_type(text: str, filename: str) -> str:
    """Classify document type based on content keywords."""
    text_lower = text[:1500].lower()
    if any(w in text_lower for w in ["agreement", "contract", "whereas"]): return "contract"
    if any(w in text_lower for w in ["judgment", "order", "petitioner"]): return "judgment"
    if any(w in text_lower for w in ["notice", "demand", "legal notice"]): return "notice"
    if any(w in text_lower for w in ["act no.", "enacted"]): return "act"
    return "other"