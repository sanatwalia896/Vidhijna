# tools/search.py — Tavily web search tool

import os
from tavily import TavilyClient
from agents.utils import deduplicate_sources, deduplicate_and_format_sources

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
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
    """Format Tavily results for LLM context."""
    if not results:
        return "No web results found."
    return deduplicate_and_format_sources(results, max_tokens_per_source=2000)