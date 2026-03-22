"""
Pinecone_ingest_pipeline.py — Vidhijna complete dual-namespace agentic ingestion

Two namespaces:
  vidhijna-legal  → Acts, sections, constitution (verbatim, rich metadata)
  vidhijna-books  → Commentary, study notes (LLM-rephrased, concise)

Three metadata levels:
  Level 1 — Shared:    act_name, legal_concepts, importance, doc_type
  Level 2 — Legal:     section_number, section_title, chapter, year_enacted,
                        related_acts, cross_references
  Level 3 — Books:     book_title, explains_act, explains_section,
                        difficulty, reasoning_type, related_sections

Three similarity types captured:
  Legal ↔ Legal:   via related_acts + cross_references + shared legal_concepts
  Legal ↔ Books:   via explains_act + explains_section + shared legal_concepts
  Books ↔ Books:   via shared topic + explains_act

Three passes per document:
  Pass 1 — LLM scans page previews → finds useful page ranges
  Pass 2 — LLM deep-reads each range → atomic chunks + full metadata
  Pass 3 — After ALL docs → LLM generates taxonomy.py

Run:
  python vector_store_creation/Pinecone_ingest_pipeline.py --file data/legal_docs/indian_contract_act_1872.pdf
  python vector_store_creation/Pinecone_ingest_pipeline.py
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
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

GROQ_MODEL      = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PINECONE_INDEX  = os.getenv("PINECONE_INDEX_NAME", "vidhijana-indexes")
NS_LEGAL        = "vidhijna-legal"
NS_BOOKS        = "vidhijna-books"
LOG_DIR         = Path("logs")
TAXONOMY_FILE   = Path("vector_store_creation/taxonomy.py")
LOG_DIR.mkdir(exist_ok=True)

LEGAL_DIRS      = ["data/legal_docs"]
BOOK_DIRS       = ["data/legal_books"]

# Known act names for cross-reference resolution
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

# ── Clients ────────────────────────────────────────────────────────────────────

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

# ── Taxonomy accumulator ───────────────────────────────────────────────────────

_seen = {
    "act_names":        set(),
    "legal_concepts":   set(),
    "doc_types_legal":  set(),
    "doc_types_books":  set(),
    "chapters":         set(),
    "book_titles":      set(),
    "topics":           set(),
    "explains_acts":    set(),
    "reasoning_types":  set(),
    "related_acts":     set(),
    "cross_references": set(),
}


# ── Text extraction ────────────────────────────────────────────────────────────

def extract_pages(file_path: Path) -> list[dict]:
    if file_path.suffix == ".txt":
        text    = file_path.read_text(encoding="utf-8", errors="ignore")
        batches = [text[i:i+1500] for i in range(0, len(text), 1500)]
        return [{"page_num": i+1, "text": c} for i, c in enumerate(batches)]
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({"page_num": i+1, "text": text.strip()})
    return pages


def pages_to_preview(pages: list[dict], max_chars: int = 10000) -> str:
    lines, total = [], 0
    for p in pages:
        preview = p["text"][:150].replace("\n", " ")
        line    = f"[Page {p['page_num']}] {preview}"
        lines.append(line)
        total  += len(line)
        if total > max_chars:
            break
    return "\n".join(lines)


def get_page_range_text(pages: list[dict], start: int, end: int) -> str:
    result = []
    for p in pages:
        if start <= p["page_num"] <= end:
            result.append(f"--- Page {p['page_num']} ---\n{p['text']}")
    return "\n\n".join(result)


# ── PASS 1: Find useful page ranges ───────────────────────────────────────────

PASS1_PROMPT = """You are a legal document analyst for Indian commercial law.

Here is a page-by-page preview of a legal document.
Identify which page ranges contain USEFUL content worth indexing.

SKIP (noise):
- Table of contents, arrangement of sections pages
- Blank pages, publisher info, copyright, signatures

INCLUDE (valuable):
- Preamble, definitions, substantive provisions
- Schedules with legal content, penalty sections
- Explanatory notes, illustrations, case analysis

For each useful range return:
- start_page: integer
- end_page: integer
- area_title: what this covers e.g. "Chapter III Contingent Contracts"
- area_type: one of [preamble, definitions, substantive, schedule, penalty, commentary, illustration]
- importance: high | medium | low

Return ONLY a valid JSON array. No markdown, no explanation.

Preview:
{preview}
"""

def pass1_find_ranges(pages: list[dict], filename: str) -> list[dict]:
    print(f"\n  [PASS 1] Scanning: {filename}")
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": PASS1_PROMPT.format(
            preview=pages_to_preview(pages)
        )}],
        temperature=0.1,
        max_tokens=2000,
    )
    raw = _clean_json(resp.choices[0].message.content.strip())
    try:
        ranges    = json.loads(raw)
        important = [r for r in ranges if r.get("importance") in ("high", "medium")]
        print(f"  [PASS 1] {len(important)} useful ranges found")
        for r in important:
            print(f"           Pages {r['start_page']}-{r['end_page']}: {r['area_title']}")
        return important
    except json.JSONDecodeError:
        print(f"  [PASS 1] Parse failed — using full document")
        return [{"start_page": 1, "end_page": len(pages),
                 "area_title": filename, "area_type": "substantive", "importance": "high"}]


# ── PASS 2a: Chunk legal documents ────────────────────────────────────────────

PASS2_LEGAL_PROMPT = """You are chunking an Indian legal document for a RAG vector store.

RULES:
- ONE legal provision per chunk — never mix sections
- Keep text VERBATIM — never rephrase legal language
- Complete sentences only — never cut mid-sentence or mid-clause
- 150 to 500 words per chunk
- Each chunk must make sense standalone

METADATA per chunk:

Level 1 — General:
- act_name: full official name e.g. "Indian Contract Act, 1872"
- legal_concepts: list of 3-6 specific concepts e.g. ["breach", "damages", "remoteness"]
- importance: high | medium | low
- doc_type: one of [provision, definition, penalty, schedule, preamble, illustration, explanation]

Level 2 — Legal specific:
- section_number: exact number e.g. "73", "2(a)", "Schedule II Para 3"
- section_title: title of this section
- chapter: chapter name e.g. "Chapter VI"
- year_enacted: year the act was passed as string
- last_amended: most recent amendment year as string, empty if unknown
- related_acts: list of other acts from this list that overlap with this chunk's topic:
  {known_acts}
- cross_references: list of specific section references mentioned in this chunk e.g. ["S.74", "S.75", "SEBI Act S.11"]

- chunk_text: the verbatim text
- summary: one sentence — what legal question does this chunk answer?

Return ONLY a valid JSON array. No markdown, no explanation.

Document: {filename}
Area: {area_title} (Pages {start_page}-{end_page})

Text:
{text}
"""

def pass2_chunk_legal(range_info: dict, range_text: str, filename: str) -> list[dict]:
    print(f"    [LEGAL] Chunking: {range_info['area_title'][:60]}")
    if len(range_text.strip()) < 100:
        return []

    all_chunks = []
    for batch in _split_batches(range_text, 4000):
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": PASS2_LEGAL_PROMPT.format(
                filename=filename,
                area_title=range_info["area_title"],
                start_page=range_info["start_page"],
                end_page=range_info["end_page"],
                text=batch,
                known_acts="\n".join(f"- {a}" for a in KNOWN_ACTS),
            )}],
            temperature=0.1,
            max_tokens=4000,
        )
        raw = _clean_json(resp.choices[0].message.content.strip())
        try:
            chunks = json.loads(raw)
            if isinstance(chunks, list):
                all_chunks.extend(chunks)
        except json.JSONDecodeError:
            pass
        time.sleep(0.8)

    valid = _filter_chunks(all_chunks)
    print(f"             {len(valid)} valid chunks")
    return valid


# ── PASS 2b: Chunk books ──────────────────────────────────────────────────────

PASS2_BOOKS_PROMPT = """You are chunking a legal commentary/study book for a RAG vector store.

RULES:
- ONE concept per chunk
- REPHRASE and SIMPLIFY — make it concise and clear for retrieval
- Remove redundancy, keep core legal reasoning
- Complete sentences only
- 100 to 400 words per chunk

METADATA per chunk:

Level 1 — General:
- legal_concepts: list of 3-6 specific concepts this explains
- importance: high | medium | low
- doc_type: one of [commentary, illustration, case_analysis, summary, mcq, explanation]

Level 2 — Links to legal namespace:
- explains_act: which act this explains e.g. "Indian Contract Act, 1872" — empty string if general
- explains_section: section number this explains e.g. "73" — empty string if general
- related_sections: list of other section numbers mentioned e.g. ["74", "75", "SEBI Act S.11"]

Level 3 — Books specific:
- book_title: name of the source
- topic: main topic e.g. "Breach of Contract"
- difficulty: basic | intermediate | advanced
- reasoning_type: one of [principle, application, case_example, definition, comparison]
- rephrased_text: your concise rephrased version

Return ONLY a valid JSON array. No markdown, no explanation.

Source: {filename}
Area: {area_title}

Text:
{text}
"""

def pass2_chunk_books(range_info: dict, range_text: str, filename: str) -> list[dict]:
    print(f"    [BOOKS] Chunking: {range_info['area_title'][:60]}")
    if len(range_text.strip()) < 100:
        return []

    all_chunks = []
    for batch in _split_batches(range_text, 4000):
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": PASS2_BOOKS_PROMPT.format(
                filename=filename,
                area_title=range_info["area_title"],
                text=batch,
            )}],
            temperature=0.2,
            max_tokens=4000,
        )
        raw = _clean_json(resp.choices[0].message.content.strip())
        try:
            chunks = json.loads(raw)
            if isinstance(chunks, list):
                all_chunks.extend(chunks)
        except json.JSONDecodeError:
            pass
        time.sleep(0.8)

    valid = _filter_chunks(all_chunks, min_words=15)
    print(f"             {len(valid)} valid chunks")
    return valid


# ── Build Documents ────────────────────────────────────────────────────────────

def build_legal_docs(chunks: list[dict], file_path: Path, range_info: dict) -> list[Document]:
    docs = []
    for chunk in chunks:
        text    = chunk.get("chunk_text", "").strip()
        summary = chunk.get("summary", "").strip()
        if not text or len(text.split()) < 20:
            continue

        section_ref = ""
        if chunk.get("act_name") and chunk.get("section_number"):
            section_ref = f"{chunk['act_name']} — Section {chunk['section_number']}"

        # summary + section ref + verbatim text
        full_content = "\n\n".join(filter(None, [summary, section_ref, text]))

        metadata = {
            # Level 1 — General
            "act_name":         chunk.get("act_name", ""),
            "legal_concepts":   ", ".join(chunk.get("legal_concepts", [])[:6]),
            "importance":       chunk.get("importance", "medium"),
            "doc_type":         chunk.get("doc_type", "provision"),
            # Level 2 — Legal specific
            "section_number":   str(chunk.get("section_number", "")),
            "section_title":    chunk.get("section_title", "")[:200],
            "chapter":          chunk.get("chapter", ""),
            "year_enacted":     str(chunk.get("year_enacted", "")),
            "last_amended":     str(chunk.get("last_amended", "")),
            "related_acts":     ", ".join(chunk.get("related_acts", [])[:5]),
            "cross_references": ", ".join(chunk.get("cross_references", [])[:8]),
            "area_title":       range_info.get("area_title", ""),
            "page_range":       f"{range_info['start_page']}-{range_info['end_page']}",
            # Housekeeping
            "source":           file_path.name,
            "namespace":        NS_LEGAL,
            "ingested_at":      datetime.now().isoformat(),
        }

        docs.append(Document(page_content=full_content, metadata=metadata))

        # Accumulate for taxonomy
        if chunk.get("act_name"):
            _seen["act_names"].add(chunk["act_name"])
        if chunk.get("chapter"):
            _seen["chapters"].add(chunk["chapter"])
        if chunk.get("doc_type"):
            _seen["doc_types_legal"].add(chunk["doc_type"])
        for c in chunk.get("legal_concepts", []):
            _seen["legal_concepts"].add(c)
        for r in chunk.get("related_acts", []):
            _seen["related_acts"].add(r)
        for r in chunk.get("cross_references", []):
            _seen["cross_references"].add(r)

    return docs


def build_book_docs(chunks: list[dict], file_path: Path, range_info: dict) -> list[Document]:
    docs = []
    for chunk in chunks:
        text = chunk.get("rephrased_text", chunk.get("chunk_text", "")).strip()
        if not text or len(text.split()) < 15:
            continue

        topic        = chunk.get("topic", "")
        full_content = "\n\n".join(filter(None, [topic, text]))

        metadata = {
            # Level 1 — General
            "legal_concepts":   ", ".join(chunk.get("legal_concepts", [])[:6]),
            "importance":       chunk.get("importance", "medium"),
            "doc_type":         chunk.get("doc_type", "commentary"),
            # Level 2 — Links to legal namespace
            "explains_act":     chunk.get("explains_act", ""),
            "explains_section": str(chunk.get("explains_section", "")),
            "related_sections": ", ".join(
                [str(s) for s in chunk.get("related_sections", [])[:8]]
            ),
            # Level 3 — Books specific
            "book_title":       chunk.get("book_title",
                                file_path.stem.replace("_", " ").title()),
            "topic":            topic[:200],
            "difficulty":       chunk.get("difficulty", "intermediate"),
            "reasoning_type":   chunk.get("reasoning_type", "explanation"),
            # Housekeeping
            "source":           file_path.name,
            "namespace":        NS_BOOKS,
            "ingested_at":      datetime.now().isoformat(),
        }

        docs.append(Document(page_content=full_content, metadata=metadata))

        # Accumulate for taxonomy
        if chunk.get("book_title"):
            _seen["book_titles"].add(chunk["book_title"])
        if chunk.get("topic"):
            _seen["topics"].add(chunk["topic"])
        if chunk.get("explains_act"):
            _seen["explains_acts"].add(chunk["explains_act"])
        if chunk.get("doc_type"):
            _seen["doc_types_books"].add(chunk["doc_type"])

    return docs


# ── Log chunks ─────────────────────────────────────────────────────────────────

def log_chunks(docs: list[Document], file_path: Path, namespace: str):
    log_file = LOG_DIR / f"{file_path.stem}_{namespace.split('-')[-1]}_chunks.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"FILE:         {file_path.name}\n")
        f.write(f"NAMESPACE:    {namespace}\n")
        f.write(f"INGESTED AT:  {datetime.now().isoformat()}\n")
        f.write(f"TOTAL CHUNKS: {len(docs)}\n")
        f.write("=" * 70 + "\n\n")
        for i, doc in enumerate(docs, 1):
            f.write(f"CHUNK {i}\n")
            f.write("-" * 40 + "\n")
            f.write("METADATA:\n")
            for k, v in doc.metadata.items():
                f.write(f"  {k:<20}: {v}\n")
            f.write("\nCONTENT PREVIEW:\n")
            f.write(doc.page_content[:400])
            f.write("\n\n" + "=" * 70 + "\n\n")
    print(f"  Log → {log_file}")


# ── Upsert ─────────────────────────────────────────────────────────────────────

def upsert(docs: list[Document], namespace: str):
    if not docs:
        print(f"  No chunks to upsert for {namespace}")
        return
    uuids = [str(uuid4()) for _ in docs]
    vector_store.add_documents(documents=docs, ids=uuids, namespace=namespace)
    print(f"  Upserted {len(docs)} chunks → Pinecone [{namespace}]")


# ── PASS 3: Generate taxonomy ──────────────────────────────────────────────────

TAXONOMY_PROMPT = """You are building a metadata taxonomy for a legal RAG retrieval system.

Below is everything collected from processing all documents.
Generate a clean controlled vocabulary that the retrieval LLM will use
to generate metadata filters at query time.

Make sure:
- All values are clean, properly cased, deduplicated
- legal_concepts are specific enough to be useful as filters
- Group related concepts together in comments

Return ONLY valid Python code defining these lists. No explanation.

Raw collected data:
{data}
"""

def generate_taxonomy():
    print("\n\n" + "=" * 60)
    print("[PASS 3] Generating taxonomy from collected metadata...")

    data = {
        "act_names":       sorted(_seen["act_names"]),
        "legal_concepts":  sorted(_seen["legal_concepts"]),
        "doc_types_legal": sorted(_seen["doc_types_legal"]),
        "doc_types_books": sorted(_seen["doc_types_books"]),
        "chapters":        sorted(_seen["chapters"])[:30],
        "book_titles":     sorted(_seen["book_titles"]),
        "topics":          sorted(_seen["topics"]),
        "explains_acts":   sorted(_seen["explains_acts"]),
        "reasoning_types": sorted(_seen["reasoning_types"]),
        "related_acts":    sorted(_seen["related_acts"]),
    }

    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": TAXONOMY_PROMPT.format(
            data=json.dumps(data, indent=2)
        )}],
        temperature=0.1,
        max_tokens=4000,
    )

    taxonomy_code = resp.choices[0].message.content.strip()
    if taxonomy_code.startswith("```"):
        taxonomy_code = taxonomy_code.split("```")[1]
        if taxonomy_code.startswith("python"):
            taxonomy_code = taxonomy_code[6:]
    taxonomy_code = taxonomy_code.strip()

    header = f'''"""
taxonomy.py — Vidhijna metadata controlled vocabulary
Auto-generated on {datetime.now().isoformat()} from actual ingested documents.

Used by the retrieval agent to generate metadata filters at query time.
The LLM picks from these lists — it never invents values.

Namespaces:
  vidhijna-legal  → acts, sections, constitution
  vidhijna-books  → commentary, study notes, reasoning
"""

'''
    TAXONOMY_FILE.parent.mkdir(parents=True, exist_ok=True)
    TAXONOMY_FILE.write_text(header + taxonomy_code, encoding="utf-8")
    print(f"[PASS 3] Taxonomy saved → {TAXONOMY_FILE}")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def process_legal_file(file_path: Path):
    print(f"\n{'='*60}")
    print(f"[LEGAL] {file_path.name}")
    print(f"{'='*60}")

    pages  = extract_pages(file_path)
    print(f"  Pages: {len(pages)}")

    ranges = pass1_find_ranges(pages, file_path.name)
    if not ranges:
        print("  No useful ranges — skipping")
        return

    all_docs = []
    for range_info in ranges:
        range_text = get_page_range_text(
            pages, range_info["start_page"], range_info["end_page"]
        )
        raw_chunks = pass2_chunk_legal(range_info, range_text, file_path.name)
        docs       = build_legal_docs(raw_chunks, file_path, range_info)
        all_docs.extend(docs)
        time.sleep(1.5)

    print(f"\n  Total chunks: {len(all_docs)}")
    log_chunks(all_docs, file_path, NS_LEGAL)
    upsert(all_docs, NS_LEGAL)


def process_book_file(file_path: Path):
    print(f"\n{'='*60}")
    print(f"[BOOKS] {file_path.name}")
    print(f"{'='*60}")

    pages  = extract_pages(file_path)
    print(f"  Pages: {len(pages)}")

    ranges = pass1_find_ranges(pages, file_path.name)
    if not ranges:
        print("  No useful ranges — skipping")
        return

    all_docs = []
    for range_info in ranges:
        range_text = get_page_range_text(
            pages, range_info["start_page"], range_info["end_page"]
        )
        raw_chunks = pass2_chunk_books(range_info, range_text, file_path.name)
        docs       = build_book_docs(raw_chunks, file_path, range_info)
        all_docs.extend(docs)
        time.sleep(1.5)

    print(f"\n  Total chunks: {len(all_docs)}")
    log_chunks(all_docs, file_path, NS_BOOKS)
    upsert(all_docs, NS_BOOKS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",  help="Process a single file")
    parser.add_argument("--books", action="store_true",
                        help="Process as book (default: auto-detect from folder)")
    args = parser.parse_args()

    if args.file:
        fp = Path(args.file)
        if args.books or any(d in str(fp) for d in BOOK_DIRS):
            process_book_file(fp)
        else:
            process_legal_file(fp)
        generate_taxonomy()
        return

    # Process all files
    legal_files = []
    for d in LEGAL_DIRS:
        p = Path(d)
        if p.exists():
            legal_files.extend(sorted(p.glob("*.pdf")))
            legal_files.extend(sorted(p.glob("*.txt")))

    book_files = []
    for d in BOOK_DIRS:
        p = Path(d)
        if p.exists():
            book_files.extend(sorted(p.glob("*.pdf")))
            book_files.extend(sorted(p.glob("*.txt")))

    print(f"Legal files: {len(legal_files)}")
    print(f"Book files:  {len(book_files)}")

    for f in legal_files:
        process_legal_file(f)

    for f in book_files:
        process_book_file(f)

    # Pass 3 — generate taxonomy from everything seen
    generate_taxonomy()

    print("\n" + "=" * 60)
    print("All done!")
    print(f"  vidhijna-legal  → {NS_LEGAL}")
    print(f"  vidhijna-books  → {NS_BOOKS}")
    print(f"  Taxonomy        → {TAXONOMY_FILE}")
    print(f"  Logs            → {LOG_DIR}/")
    print("=" * 60)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clean_json(raw: str) -> str:
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def _split_batches(text: str, batch_size: int = 4000) -> list[str]:
    if len(text) <= batch_size:
        return [text]
    batches = []
    while text:
        if len(text) <= batch_size:
            batches.append(text)
            break
        split_at = text.rfind("\n\n", 0, batch_size)
        if split_at == -1:
            split_at = batch_size
        batches.append(text[:split_at])
        text = text[split_at:].strip()
    return batches


def _filter_chunks(chunks: list[dict], min_words: int = 20) -> list[dict]:
    valid = []
    for c in chunks:
        text = c.get("chunk_text", c.get("rephrased_text", "")).strip()
        if not text:
            continue
        if len(text.split()) < min_words:
            continue
        # Skip table of contents style (many short lines)
        if text.count("\n") > 15 and len(text) < 300:
            continue
        valid.append(c)
    return valid


if __name__ == "__main__":
    main()