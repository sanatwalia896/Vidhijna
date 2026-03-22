"""
subgraphs/research.py — Deep legal research subgraph

Full pipeline:
  generate_query
    → parallel: retrieve_legal + retrieve_books + web_search
    → summarize_vectors + summarize_web
    → combine_summaries
    → extract_legal_entities
    → reflect
    → loop or finalize
"""

import json
from datetime import datetime
from typing import Literal

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph

from agents.state import VidhijnaState, TavilyFetchSignal
from agents.configuration import Configuration
from agents.prompts import (
    LEGAL_RETRIEVAL_SUMMARY_PROMPT,
    BOOKS_RETRIEVAL_SUMMARY_PROMPT,
    WEB_RESEARCH_SUMMARY_PROMPT,
    REFLECTION_PROMPT,
    FINAL_RESEARCH_PROMPT,
)
from agents.tools.retrieval import retrieve_legal, retrieve_books, format_chunks
from agents.tools.search import tavily_search, format_web_results


def _llm(model: str, temperature: float = 0.1, json_mode: bool = False):
    kwargs = dict(model=model, temperature=temperature)
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatGroq(**kwargs)


def _clean(text: str) -> str:
    while "<think>" in text and "</think>" in text:
        s = text.find("<think>")
        e = text.find("</think>") + len("</think>")
        text = text[:s] + text[e:]
    return text.strip()


# ── Nodes ──────────────────────────────────────────────────────────────────────

def generate_query(state: VidhijnaState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    llm = _llm(cfg.research_model, temperature=0.3, json_mode=True)

    prompt = f"""Rewrite for Indian commercial law vector store retrieval.
Include relevant legal terms and section numbers if known.
Query: {state.query}
Return JSON: {{"query": "rewritten query"}}"""

    result = llm.invoke([SystemMessage(content=prompt)])
    try:
        data = json.loads(_clean(result.content))
        return {"rewritten_query": data.get("query", state.query)}
    except json.JSONDecodeError:
        return {"rewritten_query": state.query}


def retrieve_legal_node(state: VidhijnaState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    matches = retrieve_legal(
        query=state.rewritten_query or state.query,
        top_k=cfg.retrieval_top_k_legal,
        filters=state.retrieval_filters or None,
        score_threshold=cfg.retrieval_score_threshold,
    )
    return {"legal_chunks": matches}


def retrieve_books_node(state: VidhijnaState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    matches = retrieve_books(
        query=state.rewritten_query or state.query,
        top_k=cfg.retrieval_top_k_books,
        score_threshold=cfg.retrieval_score_threshold,
    )
    return {"book_chunks": matches, "vector_loop_count": state.vector_loop_count + 1}


def web_search_node(state: VidhijnaState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    results = []

    if state.tavily_signals:
        for signal in state.tavily_signals:
            domains = signal.target_domains or cfg.get_domains_for_fetch_type(
                signal.fetch_type
            )
            fetched = tavily_search(
                query=signal.query,
                fetch_type=signal.fetch_type,
                target_domains=domains,
                max_results=cfg.tavily_max_results,
            )
            results.extend(fetched)
    else:
        results = tavily_search(
            query=state.rewritten_query or state.query,
            fetch_type="general",
            max_results=3,
        )

    return {
        "web_results":           results,
        "sources_gathered":      [r.get("url", "") for r in results if r.get("url")],
        "web_search_loop_count": state.web_search_loop_count + 1,
        "tavily_results_log":    [{"timestamp": datetime.utcnow().isoformat(),
                                   "count": len(results)}],
    }


def summarize_vectors_node(state: VidhijnaState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    llm = _llm(cfg.research_model)
    query = state.rewritten_query or state.query

    legal_summary, books_summary = "", ""

    if state.legal_chunks:
        r = llm.invoke([SystemMessage(content=LEGAL_RETRIEVAL_SUMMARY_PROMPT.format(
            query=query, chunks=format_chunks(state.legal_chunks)
        ))])
        legal_summary = _clean(r.content)

    if state.book_chunks:
        r = llm.invoke([SystemMessage(content=BOOKS_RETRIEVAL_SUMMARY_PROMPT.format(
            query=query, chunks=format_chunks(state.book_chunks)
        ))])
        books_summary = _clean(r.content)

    return {"legal_summary": legal_summary, "books_summary": books_summary}


def summarize_web_node(state: VidhijnaState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    llm = _llm(cfg.research_model)

    web_summary = ""
    if state.web_results:
        r = llm.invoke([SystemMessage(content=WEB_RESEARCH_SUMMARY_PROMPT.format(
            query=state.rewritten_query or state.query,
            results=format_web_results(state.web_results),
        ))])
        web_summary = _clean(r.content)

    return {"web_summary": web_summary}


def combine_summaries(state: VidhijnaState, config: RunnableConfig) -> dict:
    running = f"""# Legal Research Summary

## Applicable Law
{state.legal_summary or "No relevant law found."}

## Legal Commentary & Reasoning
{state.books_summary or "No commentary found."}

## Web Research (Cases & Regulations)
{state.web_summary or "No web results."}
"""
    return {"running_summary": running}


def extract_entities(state: VidhijnaState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    llm = _llm(cfg.research_model, temperature=0, json_mode=True)

    if not state.running_summary:
        return {"legal_entities": {
            "statutes": [], "cases": [], "principles": [],
            "jurisdictions": [], "dates": [], "parties": [],
        }}

    prompt = f"""Extract legal entities from this research. Return JSON with keys:
statutes, cases, principles, jurisdictions, dates, parties (each a list of strings).

Text: {state.running_summary[:5000]}"""

    try:
        r = llm.invoke([SystemMessage(content=prompt)])
        entities = json.loads(_clean(r.content))
        for k in ["statutes", "cases", "principles", "jurisdictions", "dates", "parties"]:
            entities.setdefault(k, [])
        return {"legal_entities": entities}
    except Exception:
        return {"legal_entities": {
            "statutes": [], "cases": [], "principles": [],
            "jurisdictions": [], "dates": [], "parties": [],
        }}


def reflect(state: VidhijnaState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    llm = _llm(cfg.research_model, temperature=0.3, json_mode=True)

    r = llm.invoke([SystemMessage(content=REFLECTION_PROMPT.format(
        query=state.query,
        legal_summary=state.legal_summary  or "None",
        books_summary=state.books_summary  or "None",
        web_summary=state.web_summary      or "None",
    ))])

    try:
        data = json.loads(_clean(r.content))
    except json.JSONDecodeError:
        data = {"has_gaps": False, "gaps": [], "followup_queries": []}

    new_signals = []
    if data.get("tavily_needed") and data.get("tavily_query"):
        new_signals.append(TavilyFetchSignal(
            fetch_type=data.get("tavily_fetch_type", "general"),
            query=data["tavily_query"],
            target_domains=cfg.get_domains_for_fetch_type(
                data.get("tavily_fetch_type", "general")
            ),
            reason="Gap found during reflection",
            priority="high",
        ))

    followups = data.get("followup_queries", [])
    return {
        "knowledge_gaps":        data.get("gaps", []),
        "followup_queries":      followups,
        "reflection_loop_count": state.reflection_loop_count + 1,
        "tavily_signals":        new_signals,
        "needs_web_search":      bool(new_signals),
        "rewritten_query":       followups[0] if followups else state.rewritten_query,
    }


def should_loop(
    state: VidhijnaState, config: RunnableConfig
) -> Literal["retrieve_legal", "finalize"]:
    cfg = Configuration.from_runnable_config(config)
    if state.knowledge_gaps and state.reflection_loop_count < cfg.max_reflection_loops:
        return "retrieve_legal"
    return "finalize"


def finalize(state: VidhijnaState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    llm = _llm(cfg.research_model)

    r = llm.invoke([SystemMessage(content=FINAL_RESEARCH_PROMPT.format(
        query=state.query,
        legal_summary=state.legal_summary  or "Not found.",
        books_summary=state.books_summary  or "Not found.",
        web_summary=state.web_summary      or "No web results.",
    ))])

    final_text = _clean(r.content)

    citations = []
    for chunk in state.legal_chunks[:5]:
        meta = chunk.get("metadata", {})
        if meta.get("act_name") and meta.get("section_number"):
            citations.append(f"{meta['act_name']} — Section {meta['section_number']}")
    for r in state.web_results[:3]:
        if r.get("url"):
            citations.append(r["url"])

    return {
        "running_summary": final_text,
        "final_response":  final_text,
        "citations":       list(dict.fromkeys(citations)),
    }


# ── Build subgraph ─────────────────────────────────────────────────────────────

def build_research_graph():
    b = StateGraph(VidhijnaState)

    b.add_node("generate_query",    generate_query)
    b.add_node("retrieve_legal",    retrieve_legal_node)
    b.add_node("retrieve_books",    retrieve_books_node)
    b.add_node("web_search",        web_search_node)
    b.add_node("summarize_vectors", summarize_vectors_node)
    b.add_node("summarize_web",     summarize_web_node)
    b.add_node("combine",           combine_summaries)
    b.add_node("extract_entities",  extract_entities)
    b.add_node("reflect",           reflect)
    b.add_node("finalize",          finalize)

    b.add_edge(START, "generate_query")

    # Parallel retrieval
    b.add_edge("generate_query", "retrieve_legal")
    b.add_edge("generate_query", "retrieve_books")
    b.add_edge("generate_query", "web_search")

    # Summarize independently
    b.add_edge("retrieve_legal",  "summarize_vectors")
    b.add_edge("retrieve_books",  "summarize_vectors")
    b.add_edge("web_search",      "summarize_web")

    # Combine → entity extraction → reflection
    b.add_edge("summarize_vectors", "combine")
    b.add_edge("summarize_web",     "combine")
    b.add_edge("combine",           "extract_entities")
    b.add_edge("extract_entities",  "reflect")

    b.add_conditional_edges("reflect", should_loop, {
        "retrieve_legal": "retrieve_legal",
        "finalize":       "finalize",
    })

    b.add_edge("finalize", END)
    return b.compile()


research_graph = build_research_graph()