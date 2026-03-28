from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph

from agents.state import VidhijnaState
from agents.configuration import Configuration
from agents.prompts import DOCUMENT_ANALYSIS_PROMPT
from agents.tools.retrieval import retrieve_legal, format_chunks
from agents.tools.ocr import extract_text, detect_document_type
from agents.utils import clean_thinking_tags, extract_json_from_text


def _llm(model: str, temperature: float = 0.1, json_mode: bool = False):
    kwargs = dict(model=model, temperature=temperature)
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatGroq(**kwargs)


def validate_document(state: VidhijnaState, config: RunnableConfig) -> dict:
    """Extract text via OCR and PURGE raw bytes to save RAM."""
    
    # Check if we need to run OCR
    if not state.uploaded_file_text and state.uploaded_file_bytes:
        filename = state.uploaded_file_name or "document.pdf"
        print(f"[GRAPH] Running OCR on {filename}...")
        
        text, file_type = extract_text(state.uploaded_file_bytes, filename)
        
        if not text:
            return {
                "error": "Could not extract text from uploaded file",
                "final_response": f"Failed to extract text from '{filename}'.",
                "uploaded_file_bytes": None # Drop bytes even on failure
            }
            
        doc_type = detect_document_type(text, filename)
        print(f"[GRAPH] OCR Success. Detected type: {doc_type}. Purging bytes from RAM.")
        
        # KEY CHANGE: We return the text but set bytes to None to free memory
        return {
            "uploaded_file_text": text,
            "uploaded_file_type": file_type,
            "uploaded_file_bytes": None, # <--- THE RAM SAVER
            "document_analysis": {"detected_doc_type": doc_type},
        }

    if not state.uploaded_file_text:
        return {
            "error": "No document uploaded",
            "final_response": "Please upload a document to analyse.",
        }

    return {"uploaded_file_type": state.uploaded_file_type or "unknown"}


def analyse_document(state: VidhijnaState, config: RunnableConfig) -> dict:
    """Analyse document content using optimized text chunks."""
    if state.error: return {}

    cfg = Configuration.from_runnable_config(config)
    llm = _llm(cfg.groq_model)

    # We only take the first 8000 characters to ensure we stay under token/RAM limits
    context_text = state.uploaded_file_text[:8000]
    
    result = llm.invoke([SystemMessage(content=DOCUMENT_ANALYSIS_PROMPT.format(
        document_text=context_text,
        query=state.query or "Provide a full analysis of this document.",
    ))])

    cleaned = clean_thinking_tags(result.content)
    return {
        "final_response": cleaned,
        "document_analysis": {"raw_analysis": cleaned},
    }


def retrieve_relevant_law(state: VidhijnaState, config: RunnableConfig) -> dict:
    """Find legal context from Pinecone."""
    if state.error: return {}

    # Use a small snippet for retrieval to keep embedding calls light
    query_snippet = state.uploaded_file_text[:500]
    matches = retrieve_legal(
        query=query_snippet,
        top_k=5,
    )
    return {"legal_chunks": matches}


def flag_risks(state: VidhijnaState, config: RunnableConfig) -> dict:
    """Extract JSON risks with memory-safe formatting."""
    if state.error or not state.legal_chunks: return {}

    cfg = Configuration.from_runnable_config(config)
    llm = _llm(cfg.groq_model, json_mode=True)

    relevant_law = format_chunks(state.legal_chunks)
    doc_summary = state.document_analysis.get("raw_analysis", "")[:2000]

    prompt = f"""Identify legal risks based on the analysis.
Relevant law: {relevant_law[:1500]}
Doc Summary: {doc_summary}
Return JSON: {{"risk_flags": [], "missing_clauses": [], "non_compliant": []}}"""

    result = llm.invoke([SystemMessage(content=prompt)])
    data = extract_json_from_text(result.content)
    
    if data:
        return {
            "risk_flags": data.get("risk_flags", []) + data.get("non_compliant", []),
            "extracted_clauses": data.get("missing_clauses", []),
        }
    return {}


def build_document_graph():
    b = StateGraph(VidhijnaState)
    b.add_node("validate",      validate_document)
    b.add_node("analyse",       analyse_document)
    b.add_node("retrieve_law",  retrieve_relevant_law)
    b.add_node("flag_risks",    flag_risks)

    b.add_edge(START, "validate")
    b.add_edge("validate", "analyse")
    b.add_edge("analyse", "retrieve_law")
    b.add_edge("retrieve_law", "flag_risks")
    b.add_edge("flag_risks", END)
    return b.compile()

document_graph = build_document_graph()