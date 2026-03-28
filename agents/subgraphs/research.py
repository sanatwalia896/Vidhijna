"""
subgraphs/research.py — Deep legal research subgraph

Full pipeline:
  generate_query
    → propose_plan (HITL breakpoint)
    → parallel: retrieve_legal + retrieve_books + web_search
    → summarize_vectors + summarize_web
    → combine_summaries
    → extract_legal_entities
    → reflect
    → loop or finalize

Each node emits rich status_log entries so the frontend can render
live flash-cards, entity highlights, and activity stream updates.
"""

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
from agents.utils import clean_thinking_tags, deduplicate_and_format_sources, extract_json_from_text


def _llm(model: str, temperature: float = 0.1, json_mode: bool = False):
    kwargs = dict(model=model, temperature=temperature, max_retries=2)
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatGroq(**kwargs)


# ── Nodes ──────────────────────────────────────────────────────────────────────

def generate_query(state: VidhijnaState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    llm = _llm(cfg.research_model, temperature=0.3, json_mode=True)

    prompt = f"""Rewrite for Indian commercial law vector store retrieval.
Include relevant legal terms and section numbers if known.
Query: {state.query}
Return JSON: {{"query": "rewritten query"}}"""

    result = llm.invoke([SystemMessage(content=prompt)])
    data = extract_json_from_text(result.content)
    
    rewritten = data.get("query", state.query) if data else state.query
    return {
        "rewritten_query": rewritten,
        "status_log":      [
            "🔍 Optimizing search query for Indian commercial law...",
            f"📝 Rewritten query: \"{rewritten[:120]}\"" if rewritten != state.query else "",
        ]
    }


def propose_plan(state: VidhijnaState, config: RunnableConfig) -> dict:
    """Point 1: HITL - Propose a plan before deep retrieval."""
    cfg = Configuration.from_runnable_config(config)
    llm = _llm(cfg.research_model, temperature=0.1, json_mode=True)
    
    from agents.prompts import RESEARCH_PLAN_PROMPT
    
    signals_text = ", ".join([s.query for s in state.tavily_signals]) or "None"
    
    r = llm.invoke([SystemMessage(content=RESEARCH_PLAN_PROMPT.format(
        query=state.query,
        rewritten_query=state.rewritten_query,
        search_signals=signals_text
    ))])
    
    plan = extract_json_from_text(r.content) or {"plan_description": "General legal research."}
    
    plan_desc = plan.get('plan_description', 'General legal research')
    acts = plan.get('target_acts', [])
    domains = plan.get('domains', [])
    complexity = plan.get('complexity', 'moderate')
    
    status_msgs = [
        f"📋 Research Plan: {plan_desc}",
    ]
    
    if acts:
        status_msgs.append(f"⚖️ Target Acts: {', '.join(acts[:5])}")
    if domains:
        status_msgs.append(f"📂 Legal Domains: {', '.join(domains[:5])}")
    status_msgs.append(f"📊 Complexity: {complexity}")
    status_msgs.append("🚀 Starting parallel retrieval from 3 sources...")
    
    return {
        "status_log": status_msgs,
        "running_summary": f"### Research Plan\n{plan_desc}\n\n"
    }


def retrieve_legal_node(state: VidhijnaState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    
    matches = retrieve_legal(
        query=state.rewritten_query or state.query,
        top_k=cfg.retrieval_top_k_legal,
        filters=state.retrieval_filters or None,
        score_threshold=cfg.retrieval_score_threshold,
    )
    
    # Build rich status messages showing what we found
    status_msgs = [
        f"⚖️ Searched vidhijna-legal namespace — Found {len(matches)} relevant provisions",
    ]
    
    # Extract chunk previews for flash cards
    for i, chunk in enumerate(matches[:4]):
        meta = chunk.get("metadata", {})
        act = meta.get("act_name", "")
        section = meta.get("section_number", "")
        title = meta.get("title", "")
        text_preview = chunk.get("text", "")[:100] if chunk.get("text") else ""
        
        if act and section:
            status_msgs.append(f"📑 [{act} — Section {section}] {title or text_preview}")
        elif act:
            status_msgs.append(f"📑 [{act}] {title or text_preview}")
        elif text_preview:
            status_msgs.append(f"📑 Provision {i+1}: {text_preview}...")
    
    return {
        "legal_chunks": matches,
        "status_log":   status_msgs
    }


def retrieve_books_node(state: VidhijnaState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    
    matches = retrieve_books(
        query=state.rewritten_query or state.query,
        top_k=cfg.retrieval_top_k_books,
        score_threshold=cfg.retrieval_score_threshold,
    )
    
    status_msgs = [
        f"📚 Searched vidhijna-books namespace — Found {len(matches)} commentary excerpts",
    ]
    
    for i, chunk in enumerate(matches[:3]):
        meta = chunk.get("metadata", {})
        source = meta.get("source", meta.get("book_name", ""))
        text_preview = chunk.get("text", "")[:100] if chunk.get("text") else ""
        
        if source:
            status_msgs.append(f"📖 [{source}] {text_preview[:80]}...")
        elif text_preview:
            status_msgs.append(f"📖 Commentary {i+1}: {text_preview[:80]}...")
    
    return {
        "book_chunks":       matches, 
        "vector_loop_count": state.vector_loop_count + 1,
        "status_log":        status_msgs
    }


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

    status_msgs = [
        f"🌐 Web search complete — Found {len(results)} results from legal sources",
    ]
    
    for r in results[:4]:
        title = r.get("title", "")[:60]
        url = r.get("url", "")
        domain = url.split("/")[2] if url and len(url.split("/")) > 2 else ""
        if title:
            status_msgs.append(f"🔗 [{domain}] {title}")
        elif url:
            status_msgs.append(f"🔗 {url[:80]}")
    
    return {
        "web_results":           results,
        "sources_gathered":      [r.get("url", "") for r in results if r.get("url")],
        "web_search_loop_count": state.web_search_loop_count + 1,
        "tavily_results_log":    [{"timestamp": datetime.utcnow().isoformat(),
                                   "count": len(results)}],
        "status_log":            status_msgs
    }


def summarize_vectors_node(state: VidhijnaState, config: RunnableConfig) -> dict:
    """Point 7: Recursive Summarization - Merge new info with existing research."""
    cfg = Configuration.from_runnable_config(config)
    llm = _llm(cfg.research_model)
    query = state.rewritten_query or state.query

    legal_summary, books_summary = "", ""
    combined_chunks = list(state.legal_chunks or []) + list(state.book_chunks or [])
    formatted_context = deduplicate_and_format_sources(combined_chunks) if combined_chunks else ""

    status_msgs = ["📝 Summarizing statutory provisions and legal commentary..."]

    if state.legal_chunks:
        prompt = LEGAL_RETRIEVAL_SUMMARY_PROMPT.format(query=query, chunks=formatted_context)
        if state.legal_summary:
            prompt = f"Existing context:\n{state.legal_summary}\n\nUpdate this summary with new info:\n{prompt}"
            
        r = llm.invoke([SystemMessage(content=prompt)])
        legal_summary = clean_thinking_tags(r.content)
        status_msgs.append(f"⚖️ Statutory summary: {legal_summary[:120]}...")

    if state.book_chunks:
        prompt = BOOKS_RETRIEVAL_SUMMARY_PROMPT.format(query=query, chunks=formatted_context)
        if state.books_summary:
             prompt = f"Existing context:\n{state.books_summary}\n\nUpdate this summary with new info:\n{prompt}"
             
        r = llm.invoke([SystemMessage(content=prompt)])
        books_summary = clean_thinking_tags(r.content)
        status_msgs.append(f"📚 Commentary summary: {books_summary[:120]}...")

    return {
        "legal_summary": legal_summary, 
        "books_summary": books_summary,
        "status_log":    status_msgs
    }


def summarize_web_node(state: VidhijnaState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    llm = _llm(cfg.research_model)

    web_summary = ""
    status_msgs = ["📰 Analyzing web research results..."]
    
    if state.web_results:
        prompt = WEB_RESEARCH_SUMMARY_PROMPT.format(
            query=state.rewritten_query or state.query,
            results=format_web_results(state.web_results),
        )
        if state.web_summary:
            prompt = f"Existing web research:\n{state.web_summary}\n\nUpdate with new findings:\n{prompt}"
            
        r = llm.invoke([SystemMessage(content=prompt)])
        web_summary = clean_thinking_tags(r.content)
        status_msgs.append(f"🌐 Web analysis: {web_summary[:120]}...")

    return {
        "web_summary": web_summary,
        "status_log":  status_msgs
    }


def combine_summaries(state: VidhijnaState, config: RunnableConfig) -> dict:
    running = f"""# Legal Research Summary

## Applicable Law
{state.legal_summary or "No relevant law found."}

## Legal Commentary & Reasoning
{state.books_summary or "No commentary found."}

## Web Research (Cases & Regulations)
{state.web_summary or "No web results."}
"""
    
    # Count what we've gathered
    n_legal = len(state.legal_chunks or [])
    n_books = len(state.book_chunks or [])
    n_web = len(state.web_results or [])
    
    return {
        "running_summary": running,
        "status_log":      [
            "🔗 Consolidating all research findings...",
            f"📊 Sources: {n_legal} statutory provisions, {n_books} commentary excerpts, {n_web} web results",
        ]
    }


def extract_entities(state: VidhijnaState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    llm = _llm(cfg.research_model, temperature=0, json_mode=True)

    if not state.running_summary:
        return {"status_log": ["⚠️ No research data to extract entities from."]}

    prompt = f"""Extract legal entities from this research. Return JSON with keys:
statutes, cases, principles, jurisdictions, dates, parties (each a list of strings).

Text: {state.running_summary[:5000]}"""

    r = llm.invoke([SystemMessage(content=prompt)])
    entities = extract_json_from_text(r.content)
    
    if entities:
        status_msgs = ["🏛️ Extracted legal entities:"]
        
        for category, items in entities.items():
            if items and isinstance(items, list) and len(items) > 0:
                items_preview = ", ".join(str(i) for i in items[:3])
                icon = {"statutes": "⚖️", "cases": "🏛️", "principles": "📜", 
                        "jurisdictions": "🏢", "dates": "📅", "parties": "👥"}.get(category, "📌")
                status_msgs.append(f"{icon} {category.title()}: {items_preview}")
        
        return {
            "legal_entities": entities,
            "status_log":     status_msgs
        }
    return {"status_log": ["📌 No specific entities extracted."]}


def reflect(state: VidhijnaState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    llm = _llm(cfg.research_model, temperature=0.3, json_mode=True)

    r = llm.invoke([SystemMessage(content=REFLECTION_PROMPT.format(
        query=state.query,
        legal_summary=state.legal_summary  or "None",
        books_summary=state.books_summary  or "None",
        web_summary=state.web_summary      or "None",
    ))])

    data = extract_json_from_text(r.content)
    if not data:
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
    gaps = data.get("gaps", [])
    
    status_msgs = ["🤔 Evaluating research quality..."]
    if gaps:
        status_msgs.append(f"⚠️ Found {len(gaps)} knowledge gaps:")
        for gap in gaps[:3]:
            status_msgs.append(f"   • {gap}")
        status_msgs.append(f"🔄 Loop {state.reflection_loop_count + 1}: Re-searching to fill gaps...")
    else:
        status_msgs.append("✅ Research appears comprehensive. Proceeding to final report.")

    return {
        "knowledge_gaps":        gaps,
        "followup_queries":      followups,
        "reflection_loop_count": state.reflection_loop_count + 1,
        "tavily_signals":        new_signals,
        "needs_web_search":      bool(new_signals),
        "rewritten_query":       followups[0] if followups else state.rewritten_query,
        "status_log":            status_msgs
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

    final_text = clean_thinking_tags(r.content)

    citations = []
    for chunk in (state.legal_chunks or [])[:5]:
        meta = chunk.get("metadata", {})
        if meta.get("act_name") and meta.get("section_number"):
            citations.append(f"{meta['act_name']} — Section {meta['section_number']}")
        elif meta.get("act_name"):
            citations.append(meta['act_name'])
    for r in (state.web_results or [])[:3]:
        if r.get("url"):
            citations.append(r["url"])

    return {
        "running_summary": final_text,
        "final_response":  final_text,
        "citations":       list(dict.fromkeys(citations)),
        "status_log":      [
            "📊 Final comprehensive legal research report generated.",
            f"📎 {len(citations)} citations attached.",
        ]
    }


# ── Build subgraph ─────────────────────────────────────────────────────────────

def build_research_graph():
    b = StateGraph(VidhijnaState)

    b.add_node("generate_query",    generate_query)
    b.add_node("propose_plan",      propose_plan)
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
    b.add_edge("generate_query", "propose_plan")

    # Parallel retrieval starts after plan proposal
    b.add_edge("propose_plan", "retrieve_legal")
    b.add_edge("propose_plan", "retrieve_books")
    b.add_edge("propose_plan", "web_search")

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
