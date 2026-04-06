# tools/search.py — Tavily web search tool

import os
from tavily import TavilyClient
from agents.utils import deduplicate_sources, deduplicate_and_format_sources

_client = None

def _get_client():
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return None

    _client = TavilyClient(api_key=api_key)
    return _client


def tavily_search(
    query: str,
    fetch_type: str = "general",
    target_domains: list = None,
    max_results: int = 5,
    search_depth: str = "advanced",
) -> list[dict]:
    """
    Search via Tavily. fetch_type determines domain targeting.
    Returns list of {title, url, content, score}.
    """
    client = _get_client()
    if client is None:
        print("[Tavily] API key missing or empty, skipping web search.")
        return []

    # Legal prefix for better results
    search_query = f"Indian law legal: {query}"

    kwargs = {
        "query": search_query,
        "max_results": max_results,
        "search_depth": search_depth,
        "include_raw_content": False,
    }

    if target_domains:
        kwargs["include_domains"] = target_domains

    try:
        response = client.search(**kwargs)
        return deduplicate_sources(response.get("results", []))
    except Exception as e:
        print(f"[Tavily] Search failed: {e}")
        return []


def format_web_results(results: list[dict]) -> str:
    """Format Tavily results for LLM context.
    Kept tight (~400 tokens total) to stay within free-tier TPM limits."""
    if not results:
        return "No web results found."
    # max_tokens_per_source=150 → ~600 chars per source × 3 results ≈ 400 tokens
    return deduplicate_and_format_sources(results[:2], max_tokens_per_source=150)
