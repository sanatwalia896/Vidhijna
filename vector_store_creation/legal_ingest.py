"""
legal_ingest_v2.py — Fast hierarchical ingestion for vidhijna-legal

Flow per document:
  1. Extract TOC + first 2 pages text
  2. ONE LLM call → generates document-level metadata
  3. Extract full text
  4. Hierarchical splitting (Chapter → Section → Sub-section → Sentence)
  5. Stamp document metadata on every chunk + extract position metadata via regex
  6. Embed + upsert to vidhijna-legal

One LLM call per document. No per-chunk LLM. Fast.

Run:
  python vector_store_creation/legal_ingest_v2.py --file data/legal_docs/indian_contract_act_1872.pdf
  python vector_store_creation/legal_ingest_v2.py
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

# ── Config ─────────────────────────────────────────────────────────────────────

GROQ_MODEL      = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PINECONE_INDEX  = os.getenv("PINECONE_INDEX_NAME", "vidhijana-indexes")
NS_LEGAL        = "vidhijna-legal"
LOG_DIR         = Path("logs")
CHECKPOINT_DIR  = Path("logs/checkpoints")
LEGAL_DIRS      = ["data/legal_docs"]
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
    "Limited Liability Partnership Act, 2008",
    "Constitution of India",
]

# ── Clients ─────────────────────────────────────────────────────────────────────

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

# Hierarchical separators — respects legal document structure
# Tries to split at highest level first, falls back down
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=80,
    separators=[
        "\nCHAPTER ",        # Chapter level
        "\nPART ",           # Part level
        "\nSCHEDULE ",       # Schedule level
        r"\n\n\d+\.",
        r"\n\d+\.",
        r"\n\(\d+\)",
        r"\n\([a-z]\)",
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
    is_separator_regex=True,
)


# ── Checkpoint helpers ──────────────────────────────────────────────────────────

def save_checkpoint(file_path: Path):
    cp = CHECKPOINT_DIR / f"{file_path.stem}.done"
    cp.touch()

def is_done(file_path: Path) -> bool:
    cp = CHECKPOINT_DIR / f"{file_path.stem}.done"
    return cp.exists()

def clear_checkpoint(file_path: Path):
    cp = CHECKPOINT_DIR / f"{file_path.stem}.done"
    if cp.exists():
        cp.unlink()


# ── Step 1: Extract text ────────────────────────────────────────────────────────

def extract_full_text(file_path: Path) -> tuple[str, str]:
    """Returns (toc_and_first_pages, full_text)"""
    if file_path.suffix == ".txt":
        full = file_path.read_text(encoding="utf-8", errors="ignore")
        return full[:3000], full

    pages = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text.strip())

    full_text   = "\n\n".join(pages)
    # TOC is usually in first 3 pages
    toc_preview = "\n\n".join(pages[:3])
    return toc_preview, full_text


# ── Step 2: ONE LLM call → document metadata ───────────────────────────────────

DOC_METADATA_PROMPT = """You are a legal document analyst for Indian commercial law.

Given the filename and TOC/opening pages of a legal document, generate metadata
that will be stamped on EVERY chunk from this document.

Filename: {filename}

TOC and first pages:
{toc_text}

Known acts for cross-referencing:
{known_acts}

Return ONLY a valid JSON object with these fields:
{{
  "act_name": "full official name e.g. Indian Contract Act, 1872",
  "year_enacted": "year as string e.g. 1872",
  "last_amended": "most recent amendment year as string, empty if unknown",
  "legal_domain": "comma-separated domains e.g. contract, agency, commercial",
  "related_acts": ["list of related act names from the known acts list"],
  "total_chapters": "number of chapters as string",
  "importance": "high | medium | low",
  "doc_type": "act | regulation | constitution | code",
  "jurisdiction": "India",
  "summary": "one sentence describing what this act governs"
}}

No markdown, no explanation. Only JSON.
"""

def generate_doc_metadata(file_path: Path, toc_text: str) -> dict:
    print(f"  [LLM] Generating document metadata...")

    resp = None
    for attempt in range(4):
        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": DOC_METADATA_PROMPT.format(
                    filename=file_path.name,
                    toc_text=toc_text[:3000],
                    known_acts="\n".join(f"- {a}" for a in KNOWN_ACTS),
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
        print(f"  [LLM] All retries failed — using filename-based fallback metadata")
        return _fallback_metadata(file_path)

    raw = _clean_json(resp.choices[0].message.content.strip())
    try:
        meta = json.loads(raw)
        print(f"  [LLM] act_name: {meta.get('act_name', '?')}")
        print(f"  [LLM] related_acts: {meta.get('related_acts', [])}")
        return meta
    except json.JSONDecodeError:
        print(f"  [LLM] JSON parse failed — using fallback metadata")
        return _fallback_metadata(file_path)


def _fallback_metadata(file_path: Path) -> dict:
    """Derive basic metadata from filename when LLM fails."""
    name = file_path.stem.replace("_", " ").title()
    year = re.search(r"\d{4}", file_path.name)
    return {
        "act_name":      name,
        "year_enacted":  year.group(0) if year else "",
        "last_amended":  "",
        "legal_domain":  "commercial",
        "related_acts":  [],
        "total_chapters":"",
        "importance":    "high",
        "doc_type":      "act",
        "jurisdiction":  "India",
        "summary":       f"Legal document: {name}",
    }


# ── Step 3: Hierarchical splitting ──────────────────────────────────────────────

def hierarchical_split(full_text: str) -> list[str]:
    """Split using legal document hierarchy via RecursiveCharacterTextSplitter."""
    chunks = splitter.split_text(full_text)
    valid = []
    for c in chunks:
        c = c.strip()
        if len(c.split()) < 15:
            continue
        # Skip TOC-style chunks: many short lines, no real sentences
        lines = [l.strip() for l in c.split("\n") if l.strip()]
        if len(lines) > 6 and sum(1 for l in lines if len(l) < 50) / len(lines) > 0.8:
            continue
        # Skip chunks that are just "ARRANGEMENT OF SECTIONS" style
        if "ARRANGEMENT OF SECTIONS" in c or "arrangement of sections" in c.lower():
            continue
        valid.append(c)
    return valid


# ── Step 4: Extract position metadata from chunk text ──────────────────────────

# Regex patterns for extracting position in legal hierarchy
CHAPTER_PAT    = re.compile(r"CHAPTER\s+([IVXLCDM\d]+)\s*[-—]?\s*(.{0,80})", re.IGNORECASE)
SECTION_PAT    = re.compile(r"(?:^|\n)\s*(\d+[A-Z]?)\.\s+([^\n]{0,80})", re.MULTILINE)
SUBSECTION_PAT = re.compile(r"\((\d+)\)")
CLAUSE_PAT     = re.compile(r"\(([a-z])\)")

def extract_position_metadata(chunk_text: str) -> dict:
    """Extract chapter, section number, section title from chunk text via regex."""
    meta = {
        "chapter":        "",
        "section_number": "",
        "section_title":  "",
    }

    # Chapter
    ch_match = CHAPTER_PAT.search(chunk_text)
    if ch_match:
        meta["chapter"] = f"Chapter {ch_match.group(1)} — {ch_match.group(2).strip()}"[:100]

    # Section number + title (first match in chunk)
    sec_match = SECTION_PAT.search(chunk_text)
    if sec_match:
        meta["section_number"] = sec_match.group(1)
        meta["section_title"]  = sec_match.group(2).strip()[:150]

    return meta


# ── Step 5: Build Documents ─────────────────────────────────────────────────────

def build_documents(
    chunks: list[str],
    doc_meta: dict,
    file_path: Path,
) -> list[Document]:
    docs = []
    for chunk_text in chunks:
        # Position metadata from regex — no LLM needed
        position = extract_position_metadata(chunk_text)

        # Full content = doc summary context + chunk text
        # This helps embedding understand context even for short chunks
        context_line = f"{doc_meta.get('act_name', '')} | {position.get('chapter', '')} | Section {position.get('section_number', '')}".strip(" |")
        full_content = f"{context_line}\n\n{chunk_text}" if context_line.replace("|","").strip() else chunk_text

        metadata = {
            # Document-level (from LLM, same for all chunks)
            "act_name":       doc_meta.get("act_name", ""),
            "year_enacted":   str(doc_meta.get("year_enacted", "")),
            "last_amended":   str(doc_meta.get("last_amended", "")),
            "legal_domain":   doc_meta.get("legal_domain", ""),
            "related_acts":   ", ".join(doc_meta.get("related_acts", [])[:5]),
            "importance":     doc_meta.get("importance", "high"),
            "doc_type":       doc_meta.get("doc_type", "act"),
            "jurisdiction":   doc_meta.get("jurisdiction", "India"),
            "doc_summary":    doc_meta.get("summary", "")[:300],
            # Position-level (from regex, unique per chunk)
            "chapter":        position.get("chapter", ""),
            "section_number": position.get("section_number", ""),
            "section_title":  position.get("section_title", ""),
            # Housekeeping
            "source":         file_path.name,
            "namespace":      NS_LEGAL,
            "ingested_at":    datetime.now().isoformat(),
        }

        docs.append(Document(page_content=full_content, metadata=metadata))
    return docs


# ── Step 6: Log ─────────────────────────────────────────────────────────────────

def log_chunks(docs: list[Document], file_path: Path):
    log_file = LOG_DIR / f"{file_path.stem}_legal_chunks.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"FILE:         {file_path.name}\n")
        f.write(f"NAMESPACE:    {NS_LEGAL}\n")
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


# ── Step 7: Upsert ──────────────────────────────────────────────────────────────

def upsert(docs: list[Document]):
    if not docs:
        print("  No chunks to upsert")
        return
    # Batch into 100s for Pinecone
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        uuids = [str(uuid4()) for _ in batch]
        vector_store.add_documents(documents=batch, ids=uuids, namespace=NS_LEGAL)
        print(f"  Upserted batch {i//batch_size + 1} ({len(batch)} chunks)")
        time.sleep(1)


# ── Main pipeline ───────────────────────────────────────────────────────────────

def process_file(file_path: Path):
    print(f"\n{'='*60}")
    print(f"[LEGAL] {file_path.name}")
    print(f"{'='*60}")

    if is_done(file_path):
        print(f"  Already processed — skipping (delete logs/checkpoints/{file_path.stem}.done to reprocess)")
        return

    # Step 1 — Extract text
    toc_text, full_text = extract_full_text(file_path)
    print(f"  Full text: {len(full_text):,} chars")

    # Step 2 — ONE LLM call for doc metadata
    doc_meta = generate_doc_metadata(file_path, toc_text)
    time.sleep(2)  # brief pause after LLM call

    # Step 3 — Hierarchical split
    chunks = hierarchical_split(full_text)
    print(f"  Chunks after hierarchical split: {len(chunks)}")

    # Step 4+5 — Build documents with metadata
    docs = build_documents(chunks, doc_meta, file_path)
    print(f"  Documents built: {len(docs)}")

    # Step 6 — Log
    log_chunks(docs, file_path)

    # Step 7 — Upsert
    upsert(docs)

    # Mark as done
    save_checkpoint(file_path)
    print(f"  Done: {file_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",    help="Process a single file")
    parser.add_argument("--reprocess", action="store_true",
                        help="Reprocess even if checkpoint exists")
    args = parser.parse_args()

    if args.file:
        fp = Path(args.file)
        if args.reprocess:
            clear_checkpoint(fp)
        process_file(fp)
        return

    files = []
    for d in LEGAL_DIRS:
        p = Path(d)
        if p.exists():
            files.extend(sorted(p.glob("*.pdf")))
            files.extend(sorted(p.glob("*.txt")))

    print(f"Found {len(files)} legal files\n")
    done  = sum(1 for f in files if is_done(f))
    print(f"Already done: {done} / {len(files)}")
    print(f"To process:   {len(files)-done} / {len(files)}\n")

    for f in files:
        process_file(f)

    print("\n" + "="*60)
    print("All legal docs processed!")
    print(f"Chunks in vidhijna-legal → check Pinecone dashboard")
    print("="*60)


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _clean_json(raw: str) -> str:
    if "```" in raw:
        parts = raw.split("```")
        raw   = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


if __name__ == "__main__":
    main()