import os
import requests
from typing import Dict, Any, List, Optional
from langsmith import traceable
from tavily import TavilyClient
from duckduckgo_search import DDGS
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from agents.configuration import Configuration
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from agents.state import SummaryState

ollama_llm = OllamaLLM(model="gemma3:1b")


def load_faiss_retriever(path: str) -> FAISS:
    embeddings = OllamaEmbeddings(
        model="all-minilm:33m"
    )  # Same model you used for indexing
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
        formatted_text += f"Source {source['title']}:\n===\n"
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
            formatted_lines.append(f"* {source['title']} : {source['url']}")
    elif isinstance(search_results, list):
        for doc in search_results:
            metadata = getattr(doc, "metadata", {})
            title = metadata.get("title", "Untitled")
            url = metadata.get("url", "No URL")
            formatted_lines.append(f"* {title} : {url}")
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

    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Same as content since DDG doesn't provide full page content
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
                        # Try to fetch the full page content using curl
                        import urllib.request
                        from bs4 import BeautifulSoup

                        response = urllib.request.urlopen(url)
                        html = response.read()
                        soup = BeautifulSoup(html, "html.parser")
                        raw_content = soup.get_text()

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
        include_raw_content (bool): Whether to include the raw_content from Tavily in the formatted string
        max_results (int): Maximum number of results to return

    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Full content of the page if available"""

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")
    tavily_client = TavilyClient(api_key=api_key)
    return tavily_client.search(
        query, max_results=max_results, include_raw_content=include_raw_content
    )


@traceable
def perplexity_search(query: str, perplexity_search_loop_count: int) -> Dict[str, Any]:
    """Search the web using the Perplexity API.

    Args:
        query (str): The search query to execute
        perplexity_search_loop_count (int): The loop step for perplexity search (starts at 0)

    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Full content of the page if available
    """

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
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

    response = requests.post(
        "https://api.perplexity.ai/chat/completions", headers=headers, json=payload
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
            "url": citations[0],
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

    laws_retriever = load_faiss_retriever(config.laws_faiss_path)
    cases_retriever = load_faiss_retriever(config.cases_faiss_path)

    laws_docs = laws_retriever.similarity_search(query, k=3)
    cases_docs = cases_retriever.similarity_search(query, k=3)

    return {"laws": laws_docs, "cases": cases_docs}


def chunk_and_summarize(text, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(text)

    summaries = []
    for chunk in chunks:
        summary = ollama_llm.invoke(f"Summarize the following legal content:\n{chunk}")
        summaries.append(summary.strip())

    combined_summary_text = "\n".join(summaries)
    final_summary = ollama_llm.invoke(
        f"Summarize the following collection of summaries into one cohesive overview:\n{combined_summary_text}"
    )
    return final_summary.strip()


def summarize_vectors(state: SummaryState):
    combined = ""

    # Handle both possible types of data in the research results
    for doc in state.laws_research_results + state.cases_research_results:
        if hasattr(doc, "page_content"):
            # If it's a Document object
            combined += doc.page_content + "\n"
        else:
            # If it's a string (formatted source)
            combined += str(doc) + "\n"

    # If combined is still empty, check if we have any results at all
    if not combined.strip() and (
        state.laws_research_results or state.cases_research_results
    ):
        # Just concatenate whatever we have as strings
        combined = "\n".join(
            [
                str(doc)
                for doc in state.laws_research_results + state.cases_research_results
            ]
        )

    # If we have content to summarize
    if combined.strip():
        summary = chunk_and_summarize(combined)
    else:
        summary = "No relevant legal documents found in vector stores."

    return {"vector_summary": summary}


def combine_summaries(state: SummaryState):
    # Access the running_summary instead of web_summary if that's what's available
    web_summary = (
        state.running_summary
        if hasattr(state, "running_summary")
        else "No web summary available."
    )

    # Access vector_summary, with a fallback
    vector_summary = (
        state.vector_summary
        if hasattr(state, "vector_summary")
        else "No legal summary available."
    )

    full_text = f"Web Summary:\n{web_summary}\n\nLegal Summary:\n{vector_summary}"

    return {"combined_summary": full_text}
