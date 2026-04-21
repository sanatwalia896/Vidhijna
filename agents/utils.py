# utils.py — Shared utilities for Vidhijna v2
# Only contains functions actually used by the new stack.
# FAISS, Ollama, Perplexity, DuckDuckGo have been removed.

import re
from typing import Optional


# ── Text cleaning ──────────────────────────────────────────────────────────────

def clean_thinking_tags(text: str) -> str:
    """
    Strip reasoning blocks and extract content from output wrappers.
    Handles openai/gpt-oss-20b (and similar reasoning models) which may use:
      - <think>...</think> or <thinking>...</thinking> for reasoning
      - <output>...</output> or <answer>...</answer> for the final response
    """
    # Strip all reasoning blocks (discard — these are internal CoT)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)

    # If the model wrapped the answer in output/answer tags, extract that content
    for tag in ('output', 'answer', 'response'):
        m = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
        if m:
            return m.group(1).strip()

    return text.strip()


def clean_repeated_tokens(text: str) -> str:
    """
    Remove hallucination artifacts — repeated digits, letters, phrases.
    Useful when LLM output has degenerated repetition.
    """
    text = re.sub(r'(\d{2,})\1+', r'\1', text)       # repeated digits
    text = re.sub(r'(.)\1{4,}', r'\1', text)           # repeated chars
    text = re.sub(r'(\b\w+\b)( \1\b)+', r'\1', text)  # repeated words
    return text.strip()


def sanitize_legal_sections(text: str) -> str:
    """
    Remove obviously suspicious legal section dump patterns from answers.

    This is a lightweight guardrail, not a substitute for legal validation.
    """
    if not text:
        return text

    # Remove long runs of sequential Companies Act sections that are usually hallucinated.
    text = re.sub(
        r"(Section\s+24[0-9A-Z]{1,2}\s*[-–]\s*Section\s+2[5-6][0-9A-Z]{1,2}.*?)(?=\n\n|\Z)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove repeated section lists separated by commas that run far beyond the relevant issue.
    text = re.sub(
        r"(Section\s+2[4-9][0-9A-Z]{0,2}(?:\s*,\s*Section\s+2[4-9][0-9A-Z]{0,2}){5,})",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


# ── Source deduplication ───────────────────────────────────────────────────────

def deduplicate_sources(sources: list[dict]) -> list[dict]:
    """
    Deduplicate a list of sources by URL.
    Keeps the first occurrence of each URL.

    Works with both:
      - Tavily results: {"url": ..., "title": ..., "content": ...}
      - Pinecone matches: {"id": ..., "score": ..., "metadata": {...}}
    """
    seen_urls = set()
    unique = []

    for source in sources:
        # Handle Tavily-style results
        if isinstance(source, dict) and "url" in source:
            url = source["url"]
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique.append(source)

        # Handle Pinecone-style matches
        elif isinstance(source, dict) and "metadata" in source:
            url = source["metadata"].get("url", source.get("id", ""))
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique.append(source)
            elif not url:
                unique.append(source)  # no URL — keep it

        else:
            unique.append(source)

    return unique


def deduplicate_and_format_sources(
    sources: list[dict],
    max_tokens_per_source: int = 2000,
    include_raw_content: bool = False,
) -> str:
    """
    Deduplicate sources and format them into a readable string for the LLM.

    Supports:
      - Tavily results (dict with url, title, content, raw_content)
      - Pinecone matches (dict with metadata containing url, act_name etc.)

    Args:
        sources:               List of source dicts
        max_tokens_per_source: Approx token limit per source (chars = tokens * 4)
        include_raw_content:   Whether to include full raw content

    Returns:
        Formatted string ready to pass to LLM as context
    """
    deduped = deduplicate_sources(sources)
    if not deduped:
        return "No sources found."

    parts = []
    char_limit = max_tokens_per_source * 4

    for i, source in enumerate(deduped, 1):
        # ── Tavily-style source ──
        if "url" in source:
            title   = source.get("title", "Untitled")
            url     = source.get("url", "")
            content = source.get("content", "")[:char_limit]
            raw     = source.get("raw_content", "") or ""

            block = f"Source {i}: {title}\nURL: {url}\nContent: {content}"
            if include_raw_content and raw:
                if len(raw) > char_limit:
                    raw = raw[:char_limit] + "... [truncated]"
                block += f"\nFull content: {raw}"

        # ── Pinecone-style source ──
        elif "metadata" in source:
            meta    = source["metadata"]
            act     = meta.get("act_name", "")
            sec     = meta.get("section_number", "")
            title   = meta.get("section_title", meta.get("title", "Legal provision"))
            url     = meta.get("url", "")
            content = meta.get("text", meta.get("summary", ""))[:char_limit]
            ref     = f"{act} — Section {sec}" if act and sec else act or title

            block = f"Source {i}: {ref}\n"
            if url:
                block += f"URL: {url}\n"
            block += f"Content: {content}"

        else:
            block = f"Source {i}: {str(source)[:char_limit]}"

        parts.append(block)

    return "\n\n===\n\n".join(parts)


def format_sources_as_bullets(sources: list[dict]) -> str:
    """
    Format sources as a simple bullet list for citations.
    Used in response_formatter to append to final answers.
    """
    deduped = deduplicate_sources(sources)
    lines = []

    for source in deduped:
        if "url" in source:
            title = source.get("title", "Untitled")
            url   = source.get("url", "")
            lines.append(f"• {title} — {url}" if url else f"• {title}")

        elif "metadata" in source:
            meta  = source["metadata"]
            act   = meta.get("act_name", "")
            sec   = meta.get("section_number", "")
            ref   = f"{act} — Section {sec}" if act and sec else act or "Legal provision"
            lines.append(f"• {ref}")

    return "\n".join(lines) if lines else ""


# ── Misc helpers ───────────────────────────────────────────────────────────────

def truncate_text(text: str, max_chars: int = 6000) -> str:
    """Truncate text to max_chars, breaking at word boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    return (truncated[:last_space] if last_space > 0 else truncated) + "..."


def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Try to extract JSON from LLM output that may have markdown fences.
    Returns parsed dict or None if parsing fails.
    """
    import json
    text = clean_thinking_tags(text).strip()

    # Strip markdown fences
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else parts[0]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object within the text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None
