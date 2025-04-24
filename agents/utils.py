# utils.py
import os
import json
import requests
from typing import Dict, Any, List, Optional
from langsmith import traceable
from tavily import TavilyClient
from duckduckgo_search import DDGS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from agents.configuration import Configuration
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from agents.state import SummaryState


def load_faiss_retriever(path: str) -> FAISS:
    """
    Load a FAISS vector store from a local path.

    Args:
        path (str): Path to the FAISS index

    Returns:
        FAISS: The loaded FAISS vector store
    """
    embeddings = OllamaEmbeddings(
        model="all-minilm:33m"
    )  # Same model used for indexing
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def deduplicate_and_format_sources(
    search_response, max_tokens_per_source, include_raw_content=False
):
    """
    Formats a search response or list of vector store documents with deduplication.
    Supports both:
        - Dict with 'results' key from search APIs
        - List of vector store Documents

    Args:
        search_response (dict or list): Search results or Langchain Documents
        max_tokens_per_source (int): Approximate token limit for raw content
        include_raw_content (bool): Whether to include full raw content

    Returns:
        str: Formatted source output
    """
    sources_list = []

    # Normalize input
    if isinstance(search_response, dict) and "results" in search_response:
        sources_list = search_response["results"]
    elif isinstance(search_response, list):
        for item in search_response:
            if isinstance(item, dict) and "results" in item:
                sources_list.extend(item["results"])
            else:
                sources_list.append(item)  # Could be a Langchain Document
    else:
        raise ValueError(
            "Input must be dict with 'results' or list of results/Documents."
        )

    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        # Handle both raw dict and Langchain Document
        if hasattr(source, "metadata"):  # Langchain Document
            metadata = source.metadata
            url = metadata.get("url", "no-url")
            title = metadata.get("title", "Untitled")
            content = source.page_content
            raw_content = metadata.get("raw_content", "")
        else:  # Search API style dict
            url = source.get("url", "no-url")
            title = source.get("title", "Untitled")
            content = source.get("content", "")
            raw_content = source.get("raw_content", "")

        if url not in unique_sources:
            unique_sources[url] = {
                "url": url,
                "title": title,
                "content": content,
                "raw_content": raw_content,
            }

    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {i}. {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += (
            f"Most relevant content from source: {source['content']}\n===\n"
        )

        if include_raw_content:
            char_limit = max_tokens_per_source * 4
            raw_content = source.get("raw_content", "") or ""
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"

    return formatted_text.strip()


def format_sources(search_results):
    """Format search results or vector documents into a bullet-point list of sources.

    Args:
        search_results (dict or list): Either:
            - A dict with 'results' key (from Tavily/DuckDuckGo/etc.)
            - A list of Langchain Documents (from FAISS/Qdrant/etc.)

    Returns:
        str: Formatted string with sources and their URLs
    """
    formatted_lines = []

    if isinstance(search_results, dict) and "results" in search_results:
        for source in search_results["results"]:
            formatted_lines.append(
                f"* {source.get('title', 'Untitled')} : {source.get('url', 'No URL')}"
            )
    elif isinstance(search_results, list):
        for doc in search_results:
            if hasattr(doc, "metadata"):
                metadata = doc.metadata
                title = metadata.get("title", "Untitled")
                url = metadata.get("url", "No URL")
                formatted_lines.append(f"* {title} : {url}")
            elif isinstance(doc, dict):
                title = doc.get("title", "Untitled")
                url = doc.get("url", "No URL")
                formatted_lines.append(f"* {title} : {url}")
            else:
                formatted_lines.append(f"* Unknown format document")
    else:
        raise ValueError("Expected dict with 'results' or list of Documents.")

    return "\n".join(formatted_lines)


@traceable
def duckduckgo_search(
    query: str, max_results: int = 3, fetch_full_page: bool = False
) -> Dict[str, List[Dict[str, str]]]:
    """Search the web using DuckDuckGo.

    Args:
        query (str): The search query to execute
        max_results (int): Maximum number of results to return
        fetch_full_page (bool): Whether to fetch full page content

    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Full page content if fetch_full_page is True
    """
    try:
        with DDGS() as ddgs:
            results = []
            search_results = list(ddgs.text(query, max_results=max_results))

            for r in search_results:
                url = r.get("href")
                title = r.get("title")
                content = r.get("body")

                if not all([url, title, content]):
                    print(f"Warning: Incomplete result from DuckDuckGo: {r}")
                    continue

                raw_content = content
                if fetch_full_page:
                    try:
                        # Try to fetch the full page content
                        import urllib.request
                        from bs4 import BeautifulSoup

                        headers = {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                        }
                        req = urllib.request.Request(url, headers=headers)
                        response = urllib.request.urlopen(req, timeout=10)
                        html = response.read()
                        soup = BeautifulSoup(html, "html.parser")
                        # Remove scripts and styles to get cleaner text
                        for script in soup(["script", "style"]):
                            script.extract()
                        raw_content = soup.get_text(separator="\n", strip=True)

                    except Exception as e:
                        print(
                            f"Warning: Failed to fetch full page content for {url}: {str(e)}"
                        )

                # Add result to list
                result = {
                    "title": title,
                    "url": url,
                    "content": content,
                    "raw_content": raw_content,
                }
                results.append(result)

            return {"results": results}
    except Exception as e:
        print(f"Error in DuckDuckGo search: {str(e)}")
        print(f"Full error details: {type(e).__name__}")
        return {"results": []}


@traceable
def tavily_search(query, include_raw_content=True, max_results=3):
    """Search the web using the Tavily API.

    Args:
        query (str): The search query to execute
        include_raw_content (bool): Whether to include the raw_content from Tavily
        max_results (int): Maximum number of results to return

    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")

    try:
        tavily_client = TavilyClient(api_key=api_key)
        return tavily_client.search(
            query, max_results=max_results, include_raw_content=include_raw_content
        )
    except Exception as e:
        print(f"Error in Tavily search: {str(e)}")
        return {"results": []}


@traceable
def perplexity_search(query: str, perplexity_search_loop_count: int) -> Dict[str, Any]:
    """Search the web using the Perplexity API.

    Args:
        query (str): The search query to execute
        perplexity_search_loop_count (int): The loop step for perplexity search (starts at 0)

    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries
    """
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY environment variable is not set")

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "system",
                "content": "Search the web and provide factual information with sources.",
            },
            {"role": "user", "content": query},
        ],
    }

    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()  # Raise exception for bad status codes

        # Parse the response
        data = response.json()
        content = data["choices"][0]["message"]["content"]

        # Perplexity returns a list of citations for a single search result
        citations = data.get("citations", ["https://perplexity.ai"])

        # Return first citation with full content, others just as references
        results = [
            {
                "title": f"Perplexity Search {perplexity_search_loop_count + 1}, Source 1",
                "url": citations[0] if citations else "https://perplexity.ai",
                "content": content,
                "raw_content": content,
            }
        ]

        # Add additional citations without duplicating content
        for i, citation in enumerate(citations[1:], start=2):
            results.append(
                {
                    "title": f"Perplexity Search {perplexity_search_loop_count + 1}, Source {i}",
                    "url": citation,
                    "content": "See above for full content",
                    "raw_content": None,
                }
            )

        return {"results": results}
    except Exception as e:
        print(f"Error in Perplexity search: {str(e)}")
        return {"results": []}


def retrieve_from_laws_and_cases(
    query: str, config: Optional[Configuration] = None
) -> Dict[str, List[Document]]:
    """
    Perform retrieval from both laws and cases FAISS vector stores.

    Args:
        query (str): The user's query
        config (Configuration): The Configuration object to get paths from

    Returns:
        dict: {
            "laws": [List of Documents from laws vector store],
            "cases": [List of Documents from cases vector store]
        }
    """
    if config is None:
        config = Configuration()  # fallback to default config

    try:
        laws_retriever = load_faiss_retriever(config.laws_faiss_path)
        laws_docs = laws_retriever.similarity_search(query, k=1)
    except Exception as e:
        print(f"Error retrieving laws: {str(e)}")
        laws_docs = []

    try:
        cases_retriever = load_faiss_retriever(config.cases_faiss_path)
        cases_docs = cases_retriever.similarity_search(query, k=1)
    except Exception as e:
        print(f"Error retrieving cases: {str(e)}")
        cases_docs = []

    return {"laws": laws_docs, "cases": cases_docs}
