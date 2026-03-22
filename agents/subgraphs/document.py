"""
subgraphs/document.py — Document analysis subgraph

Nodes:
  extract_text → analyse_document → retrieve_relevant_law → flag_risks
"""

import json

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph

from agents.state import VidhijnaState
from agents.configuration import Configuration
from agents.prompts import DOCUMENT_ANALYSIS_PROMPT
from agents.tools.retrieval import retrieve_legal, format_chunks


def _llm(model: str, temperature: float = 0.1, json_mode: bool = False):
    kwargs = dict(model=model, temperature=temperature)
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatGroq(**kwargs)


def validate_document(state: VidhijnaState, config: RunnableConfig) -> dict:
    """Check document is present and extract basic info."""
    if not state.uploaded_file_text:
        return {
            "error":          "No document uploaded",
            "final_response": "Please upload a document to analyse.",
        }
    file_type = state.uploaded_file_type or "unknown"
    return {"uploaded_file_type": file_type}


def analyse_document(state: VidhijnaState, config: RunnableConfig) -> dict:
    """Full document analysis — clauses, obligations, risks."""
    if state.error:
        return {}

    cfg = Configuration.from_runnable_config(config)
    llm = _llm(cfg.groq_model)

    result = llm.invoke([SystemMessage(content=DOCUMENT_ANALYSIS_PROMPT.format(
        document_text=state.uploaded_file_text[:8000],
        query=state.query or "Provide a full analysis of this document.",
    ))])

    return {
        "final_response":    result.content,
        "document_analysis": {"raw_analysis": result.content},
    }


def retrieve_relevant_law(state: VidhijnaState, config: RunnableConfig) -> dict:
    """Find relevant legal sections from Pinecone based on document content."""
    if state.error:
        return {}

    cfg = Configuration.from_runnable_config(config)
    matches = retrieve_legal(
        query=state.uploaded_file_text[:500],
        top_k=5,
        score_threshold=cfg.retrieval_score_threshold,
    )
    return {"legal_chunks": matches}


def flag_risks(state: VidhijnaState, config: RunnableConfig) -> dict:
    """Extract specific risk flags and missing clauses."""
    if state.error or not state.legal_chunks:
        return {}

    cfg = Configuration.from_runnable_config(config)
    llm = _llm(cfg.groq_model, json_mode=True)

    relevant_law = format_chunks(state.legal_chunks)

    prompt = f"""Based on the document analysis and relevant law, identify:
1. Risk flags — unfair or unusual clauses
2. Missing clauses — standard clauses absent from this document
3. Non-compliant clauses — clauses that may violate applicable law

Relevant law found:
{relevant_law[:2000]}

Document analysis:
{state.document_analysis.get("raw_analysis", "")[:2000]}

Return JSON: {{"risk_flags": [], "missing_clauses": [], "non_compliant": []}}"""

    try:
        result = llm.invoke([SystemMessage(content=prompt)])
        data = json.loads(result.content)
        flags = data.get("risk_flags", []) + data.get("non_compliant", [])
        return {
            "risk_flags":       flags,
            "extracted_clauses": data.get("missing_clauses", []),
        }
    except Exception:
        return {}


def build_document_graph():
    b = StateGraph(VidhijnaState)

    b.add_node("validate",      validate_document)
    b.add_node("analyse",       analyse_document)
    b.add_node("retrieve_law",  retrieve_relevant_law)
    b.add_node("flag_risks",    flag_risks)

    b.add_edge(START,        "validate")
    b.add_edge("validate",   "analyse")
    b.add_edge("analyse",    "retrieve_law")
    b.add_edge("retrieve_law", "flag_risks")
    b.add_edge("flag_risks", END)

    return b.compile()


document_graph = build_document_graph()