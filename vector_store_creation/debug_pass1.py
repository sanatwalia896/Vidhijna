"""
Quick debug script — shows exactly what the LLM returns for Pass 1
Run: python vector_store_creation/debug_pass1.py --file data/legal_docs/indian_contract_act_1872.pdf
"""

import os
import json
import argparse
from pathlib import Path

import pdfplumber
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "llama-3.1-8b-instant"
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def extract_pages(file_path: Path) -> list[dict]:
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    file_path = Path(args.file)
    pages     = extract_pages(file_path)
    preview   = pages_to_preview(pages)

    print(f"Pages: {len(pages)}")
    print(f"Preview length: {len(preview)} chars\n")

    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": PASS1_PROMPT.format(preview=preview)}],
        temperature=0.1,
        max_tokens=2000,
    )

    raw = resp.choices[0].message.content.strip()

    print("=" * 60)
    print("RAW LLM RESPONSE:")
    print("=" * 60)
    print(raw)
    print("=" * 60)

    # Try to parse
    cleaned = raw
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        print(f"\nPARSED OK — {len(parsed)} ranges found")
        for r in parsed:
            print(f"  Pages {r.get('start_page')}-{r.get('end_page')}: {r.get('area_title')} [{r.get('importance')}]")
    except json.JSONDecodeError as e:
        print(f"\nPARSE FAILED: {e}")
        print("\nFirst 200 chars of cleaned:")
        print(repr(cleaned[:200]))


if __name__ == "__main__":
    main()