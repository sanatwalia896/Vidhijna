"""
graph.py — Vidhijna v2 multi-agent system

Preserves the best of the original graph:
  - query rewriting → parallel retrieval + web research
  - summarize → reflect → loop pattern
  - legal entity extraction

Upgrades:
  - ChatOllama → ChatGroq
  - FAISS → Pinecone (vidhijna-legal + vidhijna-books namespaces)
  - SummaryState → VidhijnaState
  - Supervisor routes to 4 specialist agents
  - MemorySaver for conversational chat mode
  - Tavily with domain targeting per fetch type
"""

import json
import re
from typing import Literal
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

from agents.state import (
    VidhijnaState, VidhijnaInput, VidhijnaOutput,
    TavilyFetchSignal, VECTOR_STORE_GAPS,
)
from agents.configuration import Configuration
from agents.prompts import (
    SUPERVISOR_PROMPT,
    LEGAL_RETRIEVAL_SUMMARY_PROMPT,
    BOOKS_RETRIEVAL_SUMMARY_PROMPT,
    WEB_RESEARCH_SUMMARY_PROMPT,
    REFLECTION_PROMPT,
    FINAL_RESEARCH_PROMPT,
    CHAT_PROMPT,
    DOCUMENT_ANALYSIS_PROMPT,
    DRAFT_PROMPT,
)
from agents.tools.retrieval import retrieve_legal, retrieve_books, format_chunks
from agents.tools.search import tavily_search, format_web_results

load_dotenv()


# ── LLM factory ───────────────────────────────────────────────────────────────

def get_llm(model: str, temperature: float = 0.1, json_mode: bool = False):
    kwargs = dict(model=model, temperature=temperature)
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatGroq(**kwargs)


def clean_thinking_tags(text: str) -> str:
    """Remove <think>...</think> tags from model output."""
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end   = text.find("</think>") + len("</think>")
        text  = text[:start] + text[end:]
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# SUPERVISOR
# ══════════════════════════════════════════════════════════════════════════════

def supervisor(state: VidhijnaState, config: RunnableConfig) -> dict:
    """
    Classifies intent, rewrites query, generates retrieval filters,
    and decides whether Tavily is needed.
    """
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(cfg.supervisor_model, temperature=0.1, json_mode=True)

    recent = state.messages[-6:] if state.messages else []
    history_summary = " | ".join(
        f"{m.type}: {m.content[:80]}" for m in recent
    ) or "None"

    prompt = SUPERVISOR_PROMPT.format(
        query=state.query,
        history_summary=history_summary,
        has_file="Yes" if state.uploaded_file_text else "No",
    )

    result = llm.invoke([SystemMessage(content=prompt)])

    try:
        data = json.loads(clean_thinking_tags(result.content))
    except json.JSONDecodeError:
        data = {
            "intent": "research",
            "rewritten_query": state.query,
            "retrieval_filters": {},
            "target_namespaces": ["vidhijna-legal", "vidhijna-books"],
            "tavily_signals": [],
            "needs_web_search": False,
        }

    signals = data.get("tavily_signals", [])

    # Auto-add Tavily signal for known vector store gaps
    if cfg.is_vector_store_gap(state.query) and not signals:
        signals.append({
            "fetch_type": "regulation",
            "query": state.query,
            "target_domains": cfg.get_domains_for_fetch_type("regulation"),
            "reason": "Topic likely missing from vector store",
            "priority": "high",
        })

    signal_objects = [
        TavilyFetchSignal(
            fetch_type=s.get("fetch_type", "general"),
            query=s.get("query", state.query),
            target_domains=s.get("target_domains", []),
            reason=s.get("reason", ""),
            priority=s.get("priority", "medium"),
        )
        for s in signals
    ]

    return {
        "intent":            data.get("intent", "research"),
        "rewritten_query":   data.get("rewritten_query", state.query),
        "retrieval_filters": data.get("retrieval_filters", {}),
        "target_namespaces": data.get("target_namespaces", ["vidhijna-legal"]),
        "tavily_signals":    signal_objects,
        "needs_web_search":  bool(signal_objects),
        "messages":          [HumanMessage(content=state.query)],
    }


def route_intent(
    state: VidhijnaState,
) -> Literal["generate_query", "chat_agent", "document_agent", "draft_agent"]:
    intent = state.intent
    if intent == "chat":     return "chat_agent"
    if intent == "document": return "document_agent"
    if intent == "draft":    return "draft_agent"
    return "generate_query"   # research — enters deep research pipeline


# ══════════════════════════════════════════════════════════════════════════════
# DEEP RESEARCH PIPELINE
# (preserves original graph structure, upgraded to Groq + Pinecone)
# ══════════════════════════════════════════════════════════════════════════════

def generate_query(state: VidhijnaState, config: RunnableConfig) -> dict:
    """
    Rewrite the query for optimal legal retrieval.
    Mirrors original generate_query but uses Groq + structured output.
    """
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(cfg.research_model, temperature=0.3, json_mode=True)

    prompt = f"""You are a legal query optimiser for Indian commercial law.
Rewrite the following query to maximise vector store retrieval quality.
Make it specific, include relevant legal terms, section numbers if known.

Original query: {state.query}

Return JSON: {{"query": "rewritten query here"}}"""

    result = llm.invoke([SystemMessage(content=prompt)])

    try:
        data = json.loads(clean_thinking_tags(result.content))
        rewritten = data.get("query", state.query)
    except json.JSONDecodeError:
        rewritten = state.query

    return {"rewritten_query": rewritten}


def retrieve_from_vector_stores(state: VidhijnaState, config: RunnableConfig) -> dict:
    """
    Retrieve from both Pinecone namespaces in parallel.
    Replaces original FAISS retrieval from laws + cases.
    """
    cfg = Configuration.from_runnable_config(config)
    query = state.rewritten_query or state.query

    legal_matches = retrieve_legal(
        query=query,
        top_k=cfg.retrieval_top_k_legal,
        filters=state.retrieval_filters or None,
        score_threshold=cfg.retrieval_score_threshold,
    )
    book_matches = retrieve_books(
        query=query,
        top_k=cfg.retrieval_top_k_books,
        score_threshold=cfg.retrieval_score_threshold,
    )

    return {
        "legal_chunks":   legal_matches,
        "book_chunks":    book_matches,
        "vector_loop_count": state.vector_loop_count + 1,
    }


def web_research(state: VidhijnaState, config: RunnableConfig) -> dict:
    """
    Tavily search with domain targeting based on Tavily signals.
    Replaces original web_research with domain-aware fetching.
    """
    cfg = Configuration.from_runnable_config(config)

    if not state.needs_web_search or not state.tavily_signals:
        # Still do a general search if no signals but web search enabled
        query = state.rewritten_query or state.query
        results = tavily_search(
            query=query,
            fetch_type="general",
            target_domains=[],
            max_results=3,
        )
    else:
        results = []
        for signal in state.tavily_signals:
            domains = signal.target_domains or cfg.get_domains_for_fetch_type(
                signal.fetch_type
            )
            fetched = tavily_search(
                query=signal.query,
                fetch_type=signal.fetch_type,
                target_domains=domains,
                max_results=cfg.tavily_max_results,
                search_depth=cfg.tavily_search_depth,
            )
            results.extend(fetched)

    urls = [r.get("url", "") for r in results if r.get("url")]

    return {
        "web_results":          results,
        "sources_gathered":     urls,
        "web_search_loop_count": state.web_search_loop_count + 1,
        "tavily_results_log":   [{
            "timestamp":     datetime.utcnow().isoformat(),
            "results_count": len(results),
        }],
    }


def summarize_vectors(state: VidhijnaState, config: RunnableConfig) -> dict:
    """
    Summarize Pinecone retrieval results.
    Mirrors original summarize_vectors.
    """
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(cfg.research_model, temperature=0.1)
    query = state.rewritten_query or state.query

    legal_text = format_chunks(state.legal_chunks)
    books_text  = format_chunks(state.book_chunks)

    legal_summary = ""
    if state.legal_chunks:
        result = llm.invoke([SystemMessage(content=LEGAL_RETRIEVAL_SUMMARY_PROMPT.format(
            query=query, chunks=legal_text
        ))])
        legal_summary = clean_thinking_tags(result.content)

    books_summary = ""
    if state.book_chunks:
        result = llm.invoke([SystemMessage(content=BOOKS_RETRIEVAL_SUMMARY_PROMPT.format(
            query=query, chunks=books_text
        ))])
        books_summary = clean_thinking_tags(result.content)

    return {
        "legal_summary": legal_summary,
        "books_summary": books_summary,
    }


def summarize_web_sources(state: VidhijnaState, config: RunnableConfig) -> dict:
    """
    Summarize Tavily web results.
    Mirrors original summarize_legal_web_sources.
    """
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(cfg.research_model, temperature=0.1)
    query = state.rewritten_query or state.query

    web_summary = ""
    if state.web_results:
        web_text = format_web_results(state.web_results)
        result = llm.invoke([SystemMessage(content=WEB_RESEARCH_SUMMARY_PROMPT.format(
            query=query, results=web_text
        ))])
        web_summary = clean_thinking_tags(result.content)

    return {"web_summary": web_summary}


def combine_summaries(state: VidhijnaState, config: RunnableConfig) -> dict:
    """
    Combine all summaries into running_summary.
    Mirrors original combine_summaries.
    """
    legal  = state.legal_summary  or "No relevant law sections found."
    books  = state.books_summary  or "No commentary found."
    web    = state.web_summary    or "No web results."

    running = f"""# Legal Research Summary

## Applicable Law
{legal}

## Legal Commentary & Reasoning
{books}

## Web Research (Cases & Regulations)
{web}
"""
    return {"running_summary": running}


def extract_legal_entities(state: VidhijnaState, config: RunnableConfig) -> dict:
    """
    Extract statutes, cases, principles, parties from research results.
    Mirrors original extract_legal_entities — upgraded to Groq.
    """
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(cfg.research_model, temperature=0, json_mode=True)

    combined_text = state.running_summary or ""
    if not combined_text.strip():
        return {"legal_entities": {
            "statutes": [], "cases": [], "principles": [],
            "jurisdictions": [], "dates": [], "parties": [],
        }}

    prompt = f"""Extract key legal entities from the following research.
Return JSON with keys: statutes, cases, principles, jurisdictions, dates, parties.
Each value is a list of strings.

Text: {combined_text[:6000]}"""

    try:
        result = llm.invoke([SystemMessage(content=prompt)])
        entities = json.loads(clean_thinking_tags(result.content))
        for key in ["statutes", "cases", "principles", "jurisdictions", "dates", "parties"]:
            if key not in entities:
                entities[key] = []
        return {"legal_entities": entities}
    except Exception:
        return {"legal_entities": {
            "statutes": [], "cases": [], "principles": [],
            "jurisdictions": [], "dates": [], "parties": [],
        }}


def reflect_on_research(state: VidhijnaState, config: RunnableConfig) -> dict:
    """
    Identify gaps and generate follow-up queries.
    Mirrors original reflect_on_legal_research — upgraded to Groq.
    """
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(cfg.research_model, temperature=0.3, json_mode=True)

    result = llm.invoke([SystemMessage(content=REFLECTION_PROMPT.format(
        query=state.query,
        legal_summary=state.legal_summary   or "None",
        books_summary=state.books_summary   or "None",
        web_summary=state.web_summary       or "None",
    ))])

    try:
        data = json.loads(clean_thinking_tags(result.content))
    except json.JSONDecodeError:
        data = {"has_gaps": False, "gaps": [], "followup_queries": []}

    # Generate new Tavily signals from reflection gaps
    new_signals = []
    if data.get("tavily_needed") and data.get("tavily_query"):
        new_signals.append(TavilyFetchSignal(
            fetch_type=data.get("tavily_fetch_type", "general"),
            query=data["tavily_query"],
            target_domains=cfg.get_domains_for_fetch_type(
                data.get("tavily_fetch_type", "general")
            ),
            reason="Gap identified during reflection",
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


def route_research(
    state: VidhijnaState, config: RunnableConfig
) -> Literal["retrieve_from_vector_stores", "finalize_research"]:
    """
    Loop back for another retrieval pass or finalize.
    Mirrors original route_research.
    """
    cfg = Configuration.from_runnable_config(config)
    if (
        state.knowledge_gaps
        and state.reflection_loop_count < cfg.max_reflection_loops
    ):
        return "retrieve_from_vector_stores"
    return "finalize_research"


def finalize_research(state: VidhijnaState, config: RunnableConfig) -> dict:
    """
    Generate comprehensive final legal analysis.
    Mirrors original finalize_legal_summary — upgraded to Groq.
    """
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(cfg.research_model, temperature=0.1)

    result = llm.invoke([SystemMessage(content=FINAL_RESEARCH_PROMPT.format(
        query=state.query,
        legal_summary=state.legal_summary  or "Not found in legal database.",
        books_summary=state.books_summary  or "Not found in commentary.",
        web_summary=state.web_summary      or "No web results.",
    ))])

    final_text = clean_thinking_tags(result.content)

    # Build citations from retrieved chunks + web results
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


# ══════════════════════════════════════════════════════════════════════════════
# CHAT AGENT
# ══════════════════════════════════════════════════════════════════════════════

def chat_agent(state: VidhijnaState, config: RunnableConfig) -> dict:
    """
    Conversational agent with MemorySaver memory.
    Quick retrieval + direct answer, no deep reflection loop.
    """
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(cfg.chat_model, temperature=0.2)

    legal_matches = retrieve_legal(
        query=state.rewritten_query or state.query,
        top_k=4,
        score_threshold=cfg.retrieval_score_threshold,
    )
    legal_context = format_chunks(legal_matches) if legal_matches else "No specific sections found."

    messages = list(state.messages[-cfg.max_memory_messages:])
    messages.append(SystemMessage(content=CHAT_PROMPT.format(
        query=state.query,
        legal_context=legal_context,
    )))

    result = llm.invoke(messages)
    response = clean_thinking_tags(result.content)

    return {
        "final_response": response,
        "legal_chunks":   legal_matches,
        "messages":       [AIMessage(content=response)],
    }


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT AGENT
# ══════════════════════════════════════════════════════════════════════════════

def document_agent(state: VidhijnaState, config: RunnableConfig) -> dict:
    """OCR + contract analysis agent."""
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(cfg.groq_model, temperature=0.1)

    if not state.uploaded_file_text:
        return {
            "final_response": "No document found. Please upload a PDF or image.",
            "error": "No document uploaded",
        }

    result = llm.invoke([SystemMessage(content=DOCUMENT_ANALYSIS_PROMPT.format(
        document_text=state.uploaded_file_text[:8000],
        query=state.query or "Provide a full analysis of this document.",
    ))])

    legal_matches = retrieve_legal(
        query=state.uploaded_file_text[:400],
        top_k=4,
        score_threshold=cfg.retrieval_score_threshold,
    )

    return {
        "final_response":    clean_thinking_tags(result.content),
        "legal_chunks":      legal_matches,
        "document_analysis": {"raw_analysis": result.content},
    }


# ══════════════════════════════════════════════════════════════════════════════
# DRAFT AGENT
# ══════════════════════════════════════════════════════════════════════════════

def draft_agent(state: VidhijnaState, config: RunnableConfig) -> dict:
    """Legal document drafting agent."""
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(cfg.groq_model, temperature=0.2)

    draft_type = state.draft_type or "legal document"

    if draft_type not in cfg.supported_draft_types:
        return {
            "final_response": (
                f"Draft type '{draft_type}' not supported. "
                f"Supported: {', '.join(cfg.supported_draft_types)}"
            ),
            "error": f"Unsupported draft type: {draft_type}",
        }

    legal_matches = retrieve_legal(
        query=f"{draft_type} contract requirements India",
        top_k=4,
        score_threshold=cfg.retrieval_score_threshold,
    )

    inputs_text = "\n".join(
        f"- {k}: {v}" for k, v in (state.draft_inputs or {}).items()
    ) or "Use standard template."

    result = llm.invoke([SystemMessage(content=DRAFT_PROMPT.format(
        draft_type=draft_type,
        draft_inputs=inputs_text,
        jurisdiction=cfg.default_jurisdiction,
    ))])

    draft_text = clean_thinking_tags(result.content)

    return {
        "draft_output":   draft_text,
        "final_response": draft_text,
        "legal_chunks":   legal_matches,
        "draft_history":  [{"version": 1, "content": draft_text, "type": draft_type}],
    }


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE FORMATTER
# ══════════════════════════════════════════════════════════════════════════════

def response_formatter(state: VidhijnaState, config: RunnableConfig) -> dict:
    """Adds citations and legal disclaimer. Final node before END."""
    cfg = Configuration.from_runnable_config(config)

    response = state.final_response or state.running_summary or "No response generated."

    if cfg.include_citations and state.citations:
        cites = "\n".join(f"• {c}" for c in state.citations[:8])
        response = f"{response}\n\n**Sources:**\n{cites}"

    if not state.disclaimer_added:
        response = f"{response}\n\n---\n{cfg.legal_disclaimer}"

    return {
        "final_response":   response,
        "disclaimer_added": True,
        "messages":         [AIMessage(content=response)],
    }


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def build_graph():
    builder = StateGraph(
        VidhijnaState,
        input=VidhijnaInput,
        output=VidhijnaOutput,
        config_schema=Configuration,
    )

    # ── Nodes ──────────────────────────────────────────────────────────────────
    builder.add_node("supervisor",                supervisor)

    # Deep research pipeline (mirrors original graph flow)
    builder.add_node("generate_query",            generate_query)
    builder.add_node("retrieve_from_vector_stores", retrieve_from_vector_stores)
    builder.add_node("web_research",              web_research)
    builder.add_node("summarize_vectors",         summarize_vectors)
    builder.add_node("summarize_web_sources",     summarize_web_sources)
    builder.add_node("combine_summaries",         combine_summaries)
    builder.add_node("extract_legal_entities",    extract_legal_entities)
    builder.add_node("reflect_on_research",       reflect_on_research)
    builder.add_node("finalize_research",         finalize_research)

    # Specialist agents
    builder.add_node("chat_agent",     chat_agent)
    builder.add_node("document_agent", document_agent)
    builder.add_node("draft_agent",    draft_agent)

    # Shared formatter
    builder.add_node("response_formatter", response_formatter)

    # ── Edges ──────────────────────────────────────────────────────────────────

    builder.add_edge(START, "supervisor")

    # Supervisor routes by intent
    builder.add_conditional_edges(
        "supervisor",
        route_intent,
        {
            "generate_query": "generate_query",
            "chat_agent":     "chat_agent",
            "document_agent": "document_agent",
            "draft_agent":    "draft_agent",
        },
    )

    # ── Deep research pipeline (original graph structure preserved) ────────────
    builder.add_edge("generate_query", "retrieve_from_vector_stores")
    builder.add_edge("generate_query", "web_research")

    # Parallel: vector store + web → both summarize independently
    builder.add_edge("retrieve_from_vector_stores", "summarize_vectors")
    builder.add_edge("web_research",                "summarize_web_sources")

    # Both summaries must complete before combining
    builder.add_edge("summarize_vectors",     "combine_summaries")
    builder.add_edge("summarize_web_sources", "combine_summaries")

    # Combined → entity extraction → reflection
    builder.add_edge("combine_summaries",      "extract_legal_entities")
    builder.add_edge("extract_legal_entities", "reflect_on_research")

    # Reflection → loop or finalize
    builder.add_conditional_edges(
        "reflect_on_research",
        route_research,
        {
            "retrieve_from_vector_stores": "retrieve_from_vector_stores",
            "finalize_research":           "finalize_research",
        },
    )

    builder.add_edge("finalize_research", "response_formatter")

    # ── Specialist agents → formatter ──────────────────────────────────────────
    builder.add_edge("chat_agent",     "response_formatter")
    builder.add_edge("document_agent", "response_formatter")
    builder.add_edge("draft_agent",    "response_formatter")

    builder.add_edge("response_formatter", END)

    # ── Memory (per thread_id) ────────────────────────────────────────────────
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


graph = build_graph()