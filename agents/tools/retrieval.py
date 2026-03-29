import os
from typing import Optional
from pinecone import Pinecone
# NEW: Use the partner package class
from langchain_huggingface import HuggingFaceEndpointEmbeddings 
from agents.utils import deduplicate_sources, truncate_text

_pc = None
_index = None
_embeddings = None

def _get_clients():
    global _pc, _index, _embeddings
    if _index is None:
        # Initialize Pinecone
        _pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        _index = _pc.Index(os.environ.get("PINECONE_INDEX_NAME", "vidhijana-indexes"))
        
        # FIX: Use HuggingFaceEndpointEmbeddings
        # This class correctly handles the serverless API response format
        _embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
            huggingfacehub_api_token=os.environ["HUGGINGFACE_TOKEN"],
        )
    return _index, _embeddings




def _match_to_dict(m) -> dict:
    """Convert a Pinecone ScoredVector to a plain dict for serialization."""
    # Logic remains the same, but kept for completeness
    return {
        "id":       m.get("id", "") if isinstance(m, dict) else getattr(m, "id", ""),
        "score":    m.get("score", 0.0) if isinstance(m, dict) else getattr(m, "score", 0.0),
        "metadata": dict(m.get("metadata", {})) if isinstance(m, dict) else dict(getattr(m, "metadata", {})),
    }

def _apply_authority_weights(matches: list[dict], weights: dict) -> list[dict]:
    # ... logic remains unchanged ...
    doc_type_weights   = weights.get("doc_type", {})
    importance_weights = weights.get("importance", {})
    book_type_weights  = weights.get("book_type", {})

    for m in matches:
        meta  = m.get("metadata", {})
        score = m.get("score", 0.0)
        score *= doc_type_weights.get(meta.get("doc_type", ""), 1.0)
        score *= importance_weights.get(meta.get("importance", ""), 1.0)
        score *= book_type_weights.get(meta.get("book_type", ""), 1.0)
        m["weighted_score"] = score

    return sorted(matches, key=lambda x: x["weighted_score"], reverse=True)

def retrieve_legal(
    query: str,
    top_k: int = 20,
    top_n: int = 6,
    filters: Optional[dict] = None,
    score_threshold: float = 0.4,
    authority_weights: dict = None,
) -> list[dict]:
    index, embeddings = _get_clients()
    # This now uses the correct API call
    vector = embeddings.embed_query(query)

    results = index.query(
        vector=vector,
        top_k=top_k,
        namespace="vidhijna-legal",
        include_metadata=True,
        filter=filters or None,
    )

    matches = [
        _match_to_dict(m) for m in results.get("matches", [])
        if (m.get("score", 0) if isinstance(m, dict) else getattr(m, "score", 0))
        >= score_threshold
    ]

    matches = deduplicate_sources(matches)
    if authority_weights:
        matches = _apply_authority_weights(matches, authority_weights)
    else:
        matches.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    return matches[:top_n]

# retrieve_books follows the same pattern as retrieve_legal
def retrieve_books(
    query: str,
    top_k: int = 10,
    top_n: int = 4,
    filters: Optional[dict] = None,
    score_threshold: float = 0.4,
    authority_weights: dict = None,
) -> list[dict]:
    index, embeddings = _get_clients()
    vector = embeddings.embed_query(query)

    results = index.query(
        vector=vector,
        top_k=top_k,
        namespace="vidhijna-books",
        include_metadata=True,
        filter=filters or None,
    )

    matches = [
        _match_to_dict(m) for m in results.get("matches", [])
        if (m.get("score", 0) if isinstance(m, dict) else getattr(m, "score", 0))
        >= score_threshold
    ]

    matches = deduplicate_sources(matches)
    if authority_weights:
        matches = _apply_authority_weights(matches, authority_weights)
    else:
        matches.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    return matches[:top_n]

def format_chunks(matches: list[dict]) -> str:
    if not matches:
        return "No relevant content found."
    parts = []
    for i, m in enumerate(matches, 1):
        meta = m.get("metadata", {})
        act  = meta.get("act_name", "")
        sec  = meta.get("section_number", "")
        ref  = f"{act} — Section {sec}" if act and sec else act or "Legal document"
        text = (
            meta.get("text")
            or meta.get("page_content")
            or m.get("page_content")
            or m.get("text")
            or ""
        )
        parts.append(f"[{i}] {ref}\n{text}")
    return truncate_text("\n\n".join(parts), 6000)