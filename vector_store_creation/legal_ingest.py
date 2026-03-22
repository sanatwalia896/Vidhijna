"""
legal_ingest.py — vidhijna-legal namespace population

Flow per page:
  1. LLM reads page text → returns char ranges worth indexing + what to skip
  2. Slice text by those ranges
  3. RecursiveCharacterTextSplitter cuts each slice into chunks
  4. LLM extracts metadata for each chunk
  5. Upsert to vidhijna-legal namespace

Run:
  python vector_store_creation/legal_ingest.py --file data/legal_docs/indian_contract_act_1872.pdf
  python vector_store_creation/legal_ingest.py   # all legal docs
"""

import os
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


# ── Checkpoint helpers ──────────────────────────────────────────────────────────

def save_checkpoint(file_path: Path, page_num: int, docs_so_far: int):
    cp = CHECKPOINT_DIR / f"{file_path.stem}.json"
    import json as _json
    _json.dump({"page_num": page_num, "docs": docs_so_far,
                "file": str(file_path)}, open(cp, "w"))

def load_checkpoint(file_path: Path) -> int:
    cp = CHECKPOINT_DIR / f"{file_path.stem}.json"
    if cp.exists():
        import json as _json
        data = _json.load(open(cp))
        page = data.get("page_num", 0)
        docs = data.get("docs", 0)
        print(f"  [checkpoint] Resuming from page {page+1} ({docs} docs already upserted)")
        return page
    return 0

def clear_checkpoint(file_path: Path):
    cp = CHECKPOINT_DIR / f"{file_path.stem}.json"
    if cp.exists():
        cp.unlink()

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

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# ── Step 1: Extract page text ───────────────────────────────────────────────────

def extract_pages(file_path: Path) -> list[dict]:
    if file_path.suffix == ".txt":
        text    = file_path.read_text(encoding="utf-8", errors="ignore")
        batches = [text[i:i+2000] for i in range(0, len(text), 2000)]
        return [{"page_num": i+1, "text": c} for i, c in enumerate(batches)]
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({"page_num": i+1, "text": text.strip()})
    return pages


# ── Step 2: LLM skims page → returns char ranges ───────────────────────────────

PAGE_SKIM_PROMPT = """You are skimming a page from an Indian legal document.

Your job: identify which character ranges contain actual legal content worth indexing.

INCLUDE (legal content):
- Legal provisions and sections
- Definitions
- Penalties and enforcement
- Explanations and illustrations
- Schedules with legal substance

SKIP (noise):
- Section number lists / table of contents lines
- Page numbers, headers, footers
- Blank lines between sections
- Authentication / signature text

Read the page text below. It has character positions 0 to {text_len}.

Return a JSON array of ranges to INCLUDE. Each range:
- start: integer character position (inclusive)
- end: integer character position (exclusive)  
- content_type: one of [provision, definition, penalty, illustration, schedule, preamble]
- section_hint: section number if visible e.g. "73" or "" if not clear

Return ONLY valid JSON array. No markdown, no explanation.
If the entire page is noise, return empty array [].

Page text:
{page_text}
"""

def skim_page(page_text: str, page_num: int) -> list[dict]:
    if len(page_text.strip()) < 50:
        return []

    resp = None
    for attempt in range(4):
        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": PAGE_SKIM_PROMPT.format(
                    text_len=len(page_text),
                    page_text=page_text[:2500],
                )}],
                temperature=0.1,
                max_tokens=800,
            )
            break
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = 30 * (attempt + 1)
                print(f"      [rate limit] waiting {wait}s...")
                time.sleep(wait)
            else:
                raise

    if resp is None:
        print(f"      [skim_page] all retries failed for page {page_num} — skipping")
        return []

    raw = _clean_json(resp.choices[0].message.content.strip())
    try:
        ranges = json.loads(raw)
        if not isinstance(ranges, list):
            return []
        # Validate and clamp ranges to actual text length
        valid = []
        text_len = len(page_text)
        for r in ranges:
            start = max(0, int(r.get("start", 0)))
            end   = min(text_len, int(r.get("end", text_len)))
            if end > start + 30:   # at least 30 chars
                r["start"] = start
                r["end"]   = end
                valid.append(r)
        return valid
    except (json.JSONDecodeError, ValueError):
        # Fallback — include entire page
        return [{"start": 0, "end": len(page_text),
                 "content_type": "provision", "section_hint": ""}]


# ── Step 3: Slice + split ───────────────────────────────────────────────────────

def slice_and_split(page_text: str, ranges: list[dict], page_num: int) -> list[dict]:
    """
    Slice page text by LLM-identified ranges, then run RecursiveCharacterTextSplitter.
    Returns list of {text, content_type, section_hint, page_num}
    """
    results = []
    for r in ranges:
        slice_text = page_text[r["start"]:r["end"]].strip()
        if not slice_text:
            continue

        # RecursiveCharacterTextSplitter on this slice
        sub_chunks = splitter.split_text(slice_text)

        for chunk_text in sub_chunks:
            chunk_text = chunk_text.strip()
            if len(chunk_text.split()) < 15:   # skip very short chunks
                continue
            results.append({
                "text":         chunk_text,
                "content_type": r.get("content_type", "provision"),
                "section_hint": r.get("section_hint", ""),
                "page_num":     page_num,
            })
    return results


# ── Step 4: LLM extracts metadata for a batch of chunks ────────────────────────

METADATA_PROMPT = """You are extracting metadata for legal document chunks going into a RAG vector store.

For each chunk, extract:
- chunk_index: the index number I give you (integer)
- act_name: full official act name e.g. "Indian Contract Act, 1872"
- section_number: exact section number e.g. "73", "2(a)" — empty string if unclear
- section_title: title of this section — empty string if unclear
- chapter: chapter name e.g. "Chapter VI" — empty string if unclear
- doc_type: one of [provision, definition, penalty, schedule, preamble, illustration, explanation]
- legal_concepts: list of 3-5 specific legal concepts e.g. ["breach", "damages", "remoteness"]
- importance: high | medium | low
- related_acts: list of acts from below that this chunk is related to (empty list if none):
{known_acts}
- cross_references: list of specific sections mentioned in this chunk e.g. ["S.74", "S.75"]
- summary: one sentence — what legal question does this chunk answer?

Return a JSON array with one object per chunk. No markdown, no explanation.

Chunks:
{chunks}
"""

def extract_metadata_batch(chunks: list[dict]) -> list[dict]:
    if not chunks:
        return []

    # Format chunks for LLM
    chunks_text = "\n\n".join(
        f"[{i}] {c['text'][:400]}"
        for i, c in enumerate(chunks)
    )

    resp = None
    for attempt in range(4):
        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": METADATA_PROMPT.format(
                    known_acts="\n".join(f"- {a}" for a in KNOWN_ACTS),
                    chunks=chunks_text[:3000],
                )}],
                temperature=0.1,
                max_tokens=2000,
            )
            break
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = 30 * (attempt + 1)
                print(f"      [rate limit] waiting {wait}s...")
                time.sleep(wait)
            else:
                raise

    if resp is None:
        print(f"      [metadata] all retries failed — returning empty metadata")
        return []

    raw = _clean_json(resp.choices[0].message.content.strip())
    try:
        metadata_list = json.loads(raw)
        if not isinstance(metadata_list, list):
            return []
        return metadata_list
    except json.JSONDecodeError:
        return []


# ── Step 5: Build LangChain Documents ──────────────────────────────────────────

def build_documents(
    chunks: list[dict],
    metadata_list: list[dict],
    file_path: Path,
) -> list[Document]:
    docs = []
    meta_by_idx = {m.get("chunk_index", i): m for i, m in enumerate(metadata_list)}

    for i, chunk in enumerate(chunks):
        meta = meta_by_idx.get(i, {})
        text    = chunk["text"].strip()
        summary = meta.get("summary", "").strip()
        act     = meta.get("act_name", "")
        sec     = str(meta.get("section_number", chunk.get("section_hint", "")))

        # Content = summary + "Act — Section X" + verbatim text
        section_ref  = f"{act} — Section {sec}" if act and sec else act
        full_content = "\n\n".join(filter(None, [summary, section_ref, text]))

        metadata = {
            # Level 1 — General
            "act_name":         act,
            "legal_concepts":   ", ".join(meta.get("legal_concepts", [])[:5]),
            "importance":       meta.get("importance", "medium"),
            "doc_type":         meta.get("doc_type", chunk.get("content_type", "provision")),
            # Level 2 — Legal specific
            "section_number":   sec,
            "section_title":    meta.get("section_title", "")[:200],
            "chapter":          meta.get("chapter", ""),
            "related_acts":     ", ".join(meta.get("related_acts", [])[:5]),
            "cross_references": ", ".join(meta.get("cross_references", [])[:8]),
            "summary":          summary[:300],
            # Housekeeping
            "page_num":         str(chunk.get("page_num", "")),
            "source":           file_path.name,
            "namespace":        NS_LEGAL,
            "ingested_at":      datetime.now().isoformat(),
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
            f.write(doc.page_content[:500])
            f.write("\n\n" + "=" * 70 + "\n\n")
    print(f"  Log → {log_file}")


# ── Step 7: Upsert ──────────────────────────────────────────────────────────────

def upsert(docs: list[Document]):
    if not docs:
        print("  No chunks to upsert")
        return
    uuids = [str(uuid4()) for _ in docs]
    vector_store.add_documents(documents=docs, ids=uuids, namespace=NS_LEGAL)
    print(f"  Upserted {len(docs)} chunks → Pinecone [{NS_LEGAL}]")


# ── Main pipeline ───────────────────────────────────────────────────────────────

def process_file(file_path: Path):
    print(f"\n{'='*60}")
    print(f"[LEGAL] {file_path.name}")
    print(f"{'='*60}")

    pages = extract_pages(file_path)
    print(f"  Pages: {len(pages)}")

    # Resume from checkpoint if exists
    resume_from = load_checkpoint(file_path)

    all_docs = []

    for page in pages:
        page_num  = page["page_num"]
        page_text = page["text"]

        # Skip already-processed pages
        if page_num <= resume_from:
            print(f"  [Page {page_num:>3}] already done — skipping")
            continue

        print(f"  [Page {page_num:>3}] Skimming ({len(page_text)} chars)...", end=" ")

        # Step 1 — LLM finds char ranges
        ranges = skim_page(page_text, page_num)
        if not ranges:
            print("skipped")
            save_checkpoint(file_path, page_num, len(all_docs))
            time.sleep(1)
            continue

        # Step 2 — Slice + split
        raw_chunks = slice_and_split(page_text, ranges, page_num)
        if not raw_chunks:
            print("0 chunks")
            save_checkpoint(file_path, page_num, len(all_docs))
            continue

        # Step 3 — Extract metadata + upsert immediately
        metadata_list = extract_metadata_batch(raw_chunks)
        docs = build_documents(raw_chunks, metadata_list, file_path)

        # Upsert this page's chunks immediately — don't wait till end
        upsert(docs)
        all_docs.extend(docs)

        # Save checkpoint after every successful page
        save_checkpoint(file_path, page_num, len(all_docs))

        print(f"{len(docs)} chunks upserted")
        time.sleep(1.5)

    print(f"\n  Total chunks upserted: {len(all_docs)}")
    log_chunks(all_docs, file_path)
    clear_checkpoint(file_path)
    print(f"  Done: {file_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Process a single file")
    args = parser.parse_args()

    if args.file:
        process_file(Path(args.file))
        return

    files = []
    for d in LEGAL_DIRS:
        p = Path(d)
        if p.exists():
            files.extend(sorted(p.glob("*.pdf")))
            files.extend(sorted(p.glob("*.txt")))

    print(f"Found {len(files)} legal files\n")
    for f in files:
        process_file(f)

    print("\nAll done!")


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _clean_json(raw: str) -> str:
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


if __name__ == "__main__":
    main()