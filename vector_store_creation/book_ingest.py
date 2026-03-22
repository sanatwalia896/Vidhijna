"""
books_ingest.py — vidhijna-books namespace population

Books are different from legal docs:
  - No strict section hierarchy — chapters, topics, arguments
  - LLM reads preface + TOC + first chapter to understand the book
  - LLM generates doc metadata (one call per book, no per-chunk LLM)
  - Hierarchical split respects chapter/topic boundaries
  - Each chunk links back to legal namespace via covers_acts

Flow per book:
  1. Extract preface + TOC + first chapter (for LLM context)
  2. ONE LLM call → document metadata (book_title, covers_acts, purpose etc.)
  3. Hierarchical split on full text (chapter → topic → paragraph)
  4. Stamp doc metadata + extracted position metadata on every chunk
  5. Upsert to vidhijna-books

One LLM call per book. No per-chunk LLM. Fast.

Run:
  python vector_store_creation/books_ingest.py --file data/legal_books/business_laws_study_notes.pdf
  python vector_store_creation/books_ingest.py
"""

import os
import re
import json
import time
import argparse
from pathlib import Path
from uuid import uuid4
from datetime import datetime

import pdfplumber
from groq import Groq
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────────

GROQ_MODEL      = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PINECONE_INDEX  = os.getenv("PINECONE_INDEX_NAME", "vidhijana-indexes")
NS_BOOKS        = "vidhijna-books"
LOG_DIR         = Path("logs")
CHECKPOINT_DIR  = Path("logs/checkpoints")
BOOK_DIRS       = ["data/legal_books"]
LOG_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

KNOWN_ACTS = [
    "Indian Contract Act, 1872",
    "Companies Act, 2013",
    "Insolvency and Bankruptcy Code, 2016",
    "SEBI Act, 1992",
    "Arbitration and Conciliation Act, 1996",
    "Consumer Protection Act, 2019",
    "Competition Act, 2002",
    "Transfer of Property Act, 1882",
    "Specific Relief Act, 1963",
    "Indian Partnership Act, 1932",
    "Sale of Goods Act, 1930",
    "Negotiable Instruments Act, 1881",
    "Foreign Exchange Management Act, 1999",
    "Constitution of India",
]

# ── Clients ──────────────────────────────────────────────────────────────────────

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

embeddings = HuggingFaceEndpointEmbeddings(
    model=EMBEDDING_MODEL,
    task="feature-extraction",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_TOKEN"),
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
if not pc.has_index(PINECONE_INDEX):
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index        = pc.Index(PINECONE_INDEX)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Book-aware hierarchical splitter
# Books split at chapter → section → topic → paragraph level
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=[
        "\nCHAPTER ",
        "\nChapter ",
        "\nUNIT ",
        "\nUnit ",
        "\nMODULE ",
        "\nSECTION ",
        "\nSection ",
        r"\n\d+\.\d+",     # numbered sections like "3.1", "3.2"
        r"\n\d+\.",         # numbered topics like "1.", "2."
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
    is_separator_regex=True,
)


# ── Checkpoint helpers ───────────────────────────────────────────────────────────

def save_checkpoint(file_path: Path):
    (CHECKPOINT_DIR / f"books_{file_path.stem}.done").touch()

def is_done(file_path: Path) -> bool:
    return (CHECKPOINT_DIR / f"books_{file_path.stem}.done").exists()

def clear_checkpoint(file_path: Path):
    cp = CHECKPOINT_DIR / f"books_{file_path.stem}.done"
    if cp.exists():
        cp.unlink()


# ── Step 1: Extract text ─────────────────────────────────────────────────────────

def extract_text(file_path: Path) -> tuple[str, str]:
    """Returns (preface_toc_first_chapter, full_text)"""
    if file_path.suffix == ".txt":
        full = file_path.read_text(encoding="utf-8", errors="ignore")
        return full[:5000], full

    pages = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text.strip())

    full_text = "\n\n".join(pages)
    # Preface + TOC + first chapter = first 15% of pages or first 5000 chars
    preview_pages = max(10, len(pages) // 7)
    preview_text  = "\n\n".join(pages[:preview_pages])
    return preview_text[:5000], full_text


# ── Step 2: ONE LLM call → book metadata ────────────────────────────────────────

BOOK_METADATA_PROMPT = """You are analysing a legal book or study material for Indian commercial law.

Read the preface, table of contents and opening chapter below.
Generate metadata that will be stamped on EVERY chunk from this book.

Filename: {filename}

Known acts for cross-referencing:
{known_acts}

Preface + TOC + first chapter:
{preview_text}

Return ONLY a valid JSON object:
{{
  "book_title": "full title of this book",
  "authors": "author name(s) if found, else empty string",
  "publisher": "publisher name if found, else empty string",
  "book_type": one of ["study_notes", "commentary", "reasoning", "textbook", "case_digest", "bare_act_explanation"],
  "covers_acts": ["list of act names from the known acts that this book covers"],
  "legal_domain": "comma-separated domains e.g. contract, agency, commercial",
  "purpose": "one sentence — what is this book trying to help the reader understand or do?",
  "difficulty": "basic | intermediate | advanced",
  "importance": "high | medium | low",
  "reasoning_focus": "one of [principle, application, case_analysis, mixed] — what kind of reasoning does this book emphasise?"
}}

No markdown, no explanation. Only JSON.
"""

def generate_book_metadata(file_path: Path, preview_text: str) -> dict:
    print(f"  [LLM] Generating book metadata...")

    resp = None
    for attempt in range(4):
        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": BOOK_METADATA_PROMPT.format(
                    filename=file_path.name,
                    known_acts="\n".join(f"- {a}" for a in KNOWN_ACTS),
                    preview_text=preview_text,
                )}],
                temperature=0.1,
                max_tokens=600,
            )
            break
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = 30 * (attempt + 1)
                print(f"  [rate limit] waiting {wait}s...")
                time.sleep(wait)
            else:
                raise

    if resp is None:
        return _fallback_book_metadata(file_path)

    raw = _clean_json(resp.choices[0].message.content.strip())
    try:
        meta = json.loads(raw)
        print(f"  [LLM] book_title: {meta.get('book_title', '?')}")
        print(f"  [LLM] covers_acts: {meta.get('covers_acts', [])}")
        print(f"  [LLM] book_type: {meta.get('book_type', '?')}")
        return meta
    except json.JSONDecodeError:
        return _fallback_book_metadata(file_path)


def _fallback_book_metadata(file_path: Path) -> dict:
    name = file_path.stem.replace("_", " ").title()
    return {
        "book_title":      name,
        "authors":         "",
        "publisher":       "",
        "book_type":       "study_notes",
        "covers_acts":     [],
        "legal_domain":    "commercial",
        "purpose":         f"Study material: {name}",
        "difficulty":      "intermediate",
        "importance":      "medium",
        "reasoning_focus": "mixed",
    }


# ── Step 3: Hierarchical split ───────────────────────────────────────────────────

def hierarchical_split(full_text: str) -> list[str]:
    chunks = splitter.split_text(full_text)
    valid  = []
    for c in chunks:
        c = c.strip()
        if len(c.split()) < 15:
            continue
        # Skip TOC-style chunks
        lines = [l.strip() for l in c.split("\n") if l.strip()]
        if len(lines) > 6 and sum(1 for l in lines if len(l) < 50) / len(lines) > 0.8:
            continue
        valid.append(c)
    return valid


# ── Step 4: Extract position from chunk text ─────────────────────────────────────

CHAPTER_PAT = re.compile(r"(?:CHAPTER|Chapter|UNIT|Unit)\s+(\w+)\s*[-—]?\s*(.{0,80})", re.IGNORECASE)
SECTION_PAT = re.compile(r"(?:^|\n)\s*(\d+[\.\d]*)\s+([A-Z][^\n]{0,80})", re.MULTILINE)

def extract_book_position(chunk_text: str) -> dict:
    meta = {"chapter": "", "topic_number": ""}
    ch = CHAPTER_PAT.search(chunk_text)
    if ch:
        meta["chapter"] = f"Chapter {ch.group(1)} — {ch.group(2).strip()}"[:100]
    sec = SECTION_PAT.search(chunk_text)
    if sec:
        meta["topic_number"] = sec.group(1)
    return meta


# ── Step 5: Build Documents ──────────────────────────────────────────────────────

def build_documents(
    chunks:    list[str],
    doc_meta:  dict,
    file_path: Path,
) -> list[Document]:
    docs = []
    for chunk_text in chunks:
        position = extract_book_position(chunk_text)

        # Context line for embedding quality
        context = f"{doc_meta.get('book_title', '')} | {position.get('chapter', '')}".strip(" |")
        full_content = f"{context}\n\n{chunk_text}" if context.replace("|","").strip() else chunk_text

        metadata = {
            # Document-level — from LLM (same for all chunks)
            "book_title":       doc_meta.get("book_title", ""),
            "authors":          doc_meta.get("authors", ""),
            "publisher":        doc_meta.get("publisher", ""),
            "book_type":        doc_meta.get("book_type", ""),
            "covers_acts":      ", ".join(doc_meta.get("covers_acts", [])[:5]),
            "legal_domain":     doc_meta.get("legal_domain", ""),
            "book_purpose":     doc_meta.get("purpose", "")[:300],
            "difficulty":       doc_meta.get("difficulty", "intermediate"),
            "importance":       doc_meta.get("importance", "medium"),
            "reasoning_focus":  doc_meta.get("reasoning_focus", "mixed"),
            # Position from regex
            "chapter":          position.get("chapter", ""),
            "topic_number":     position.get("topic_number", ""),
            # Housekeeping
            "source":           file_path.name,
            "namespace":        NS_BOOKS,
            "ingested_at":      datetime.now().isoformat(),
        }

        docs.append(Document(page_content=full_content, metadata=metadata))
    return docs


# ── Step 6: Log ──────────────────────────────────────────────────────────────────

def log_chunks(docs: list[Document], file_path: Path):
    log_file = LOG_DIR / f"{file_path.stem}_books_chunks.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"FILE:         {file_path.name}\n")
        f.write(f"NAMESPACE:    {NS_BOOKS}\n")
        f.write(f"INGESTED AT:  {datetime.now().isoformat()}\n")
        f.write(f"TOTAL CHUNKS: {len(docs)}\n")
        f.write("=" * 70 + "\n\n")
        for i, doc in enumerate(docs, 1):
            f.write(f"CHUNK {i}\n")
            f.write("-" * 40 + "\n")
            f.write("METADATA:\n")
            for k, v in doc.metadata.items():
                f.write(f"  {k:<20}: {v}\n")
            f.write("\nCONTENT:\n")
            f.write(doc.page_content[:400])
            f.write("\n\n" + "=" * 70 + "\n\n")
    print(f"  Log → {log_file}")


# ── Step 7: Upsert ───────────────────────────────────────────────────────────────

def upsert(docs: list[Document]):
    if not docs:
        print("  No chunks to upsert")
        return
    for i in range(0, len(docs), 100):
        batch = docs[i:i+100]
        uuids = [str(uuid4()) for _ in batch]
        vector_store.add_documents(documents=batch, ids=uuids, namespace=NS_BOOKS)
        print(f"  Upserted batch {i//100 + 1} ({len(batch)} chunks)")
        time.sleep(1)


# ── Main pipeline ─────────────────────────────────────────────────────────────────

def process_file(file_path: Path):
    print(f"\n{'='*60}")
    print(f"[BOOKS] {file_path.name}")
    print(f"{'='*60}")

    if is_done(file_path):
        print(f"  Already processed — skipping")
        return

    # Step 1 — Extract
    preview_text, full_text = extract_text(file_path)
    print(f"  Full text: {len(full_text):,} chars")

    # Step 2 — ONE LLM call for book metadata
    doc_meta = generate_book_metadata(file_path, preview_text)
    time.sleep(2)

    # Step 3 — Hierarchical split
    chunks = hierarchical_split(full_text)
    print(f"  Chunks after split: {len(chunks)}")

    # Step 4+5 — Build documents with metadata
    docs = build_documents(chunks, doc_meta, file_path)
    print(f"  Documents built: {len(docs)}")
    log_chunks(docs, file_path)
    upsert(docs)
    save_checkpoint(file_path)
    print(f"  Done: {file_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",      help="Process a single file")
    parser.add_argument("--reprocess", action="store_true")
    args = parser.parse_args()

    if args.file:
        fp = Path(args.file)
        if args.reprocess:
            clear_checkpoint(fp)
        process_file(fp)
        return

    files = []
    for d in BOOK_DIRS:
        p = Path(d)
        if p.exists():
            files.extend(sorted(p.glob("*.pdf")))
            files.extend(sorted(p.glob("*.txt")))

    print(f"Found {len(files)} book files")
    done = sum(1 for f in files if is_done(f))
    print(f"Already done: {done}/{len(files)}\n")

    for f in files:
        process_file(f)

    print("\n" + "="*60)
    print("All books processed!")
    print("="*60)


# ── Helpers ───────────────────────────────────────────────────────────────────────

def _clean_json(raw: str) -> str:
    if "```" in raw:
        parts = raw.split("```")
        raw   = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


if __name__ == "__main__":
    main()