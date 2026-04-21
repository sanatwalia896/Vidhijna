"""
subgraphs/research.py — Deep legal research subgraph

Pipeline:
  propose_plan
    → parallel: retrieve_legal + retrieve_books + web_search
    → parallel: summarize_legal + summarize_books + summarize_web
    → combine_summaries
    → extract_legal_entities
    → reflect
    → loop (ALL 3 sources via propose_plan) or finalize

Key improvements over v1:
  - Reflection loops re-run ALL 3 sources, not just legal
  - Legal and books summarized separately (no chunk mixing)
  - Typed followup queries per namespace
  - No redundant double query rewrite (supervisor already rewrites)
"""

import time
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
from agents.utils import clean_thinking_tags, extract_json_from_text, sanitize_legal_sections


def _llm(model: str, temperature: float = 0.1, json_mode: bool = False,
         reasoning_effort: str = None):
    kwargs = dict(model=model, temperature=temperature, max_retries=2)
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    # reasoning_effort is a top-level param — only for reasoning models (20B).
    # Do NOT pass it for 8B models; ChatGroq rejects it.
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    return ChatGroq(**kwargs)


def _invoke_safe(llm, messages, retries: int = 3) -> str:
    """Invoke with 429-aware backoff. Groq's max_retries doesn't sleep on rate limits."""
    for attempt in range(retries):
        try:
            return llm.invoke(messages).content
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
                continue
            raise


def _invoke_with_json_fallback(model: str, temperature: float, prompt: str,
                                reasoning_effort: str = None) -> str:
    """Invoke with JSON mode; fall back to plain mode if Groq json_validate_failed."""
    msgs = [SystemMessage(content=prompt)]
    try:
        return _invoke_safe(_llm(model, temperature=temperature, json_mode=True,
                                 reasoning_effort=reasoning_effort), msgs)
    except Exception:
        return _invoke_safe(_llm(model, temperature=temperature, json_mode=False,
                                 reasoning_effort=reasoning_effort), msgs)


# ── Nodes ──────────────────────────────────────────────────────────────────────

def propose_plan(state: VidhijnaState, config: RunnableConfig) -> dict:
    """Research plan + fan-out point for parallel retrieval.
    On loop iterations, adjusts plan based on reflection gaps."""
    cfg = Configuration.from_runnable_config(config)

    from agents.prompts import RESEARCH_PLAN_PROMPT

    signals_text = ", ".join([s.query for s in state.tavily_signals]) or "None"

    content = _invoke_with_json_fallback(
        cfg.groq_model,
        temperature=0.1,
        prompt=RESEARCH_PLAN_PROMPT.format(
            query=state.query,
            rewritten_query=state.rewritten_query or state.query,
            search_signals=signals_text,
        ),
    )
    plan = extract_json_from_text(content) or {"plan_description": "General legal research."}

    plan_desc = plan.get('plan_description', 'General legal research')
    acts = plan.get('target_acts', [])
    complexity = plan.get('complexity', 'moderate')
    loop = state.reflection_loop_count

    status_msgs = []
    if loop > 0:
        status_msgs.append(f"🔄 Research loop {loop} — re-searching all sources to fill gaps...")
        for gap in (state.knowledge_gaps or [])[:2]:
            status_msgs.append(f"   ⚠️ {gap}")
    else:
        status_msgs.append(f"📋 Research Plan: {plan_desc}")
        if acts:
            status_msgs.append(f"⚖️ Target Acts: {', '.join(acts[:5])}")
        status_msgs.append(f"📊 Complexity: {complexity}")

    status_msgs.append("🚀 Starting parallel retrieval from 3 sources...")

    return {
        "status_log": status_msgs,
        "running_summary": f"### Research Plan\n{plan_desc}\n\n" if loop == 0 else state.running_summary,
    }


def retrieve_legal_node(state: VidhijnaState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)

    # Use targeted followup from reflection, else supervisor's rewrite
    query = state.legal_followup_query or state.rewritten_query or state.query

    matches = retrieve_legal(
        query=query,
        top_k=cfg.retrieval_top_k_legal,
        top_n=cfg.rerank_top_n_legal,
        filters=state.retrieval_filters or None,
        score_threshold=cfg.retrieval_score_threshold,
    )

    status_msgs = [
        f"⚖️ Searched vidhijna-legal — Found {len(matches)} provisions",
    ]
    for chunk in matches[:4]:
        meta = chunk.get("metadata", {})
        act = meta.get("act_name", "")
        section = meta.get("section_number", "")
        title = meta.get("title", "")
        text_preview = chunk.get("text", "")[:100] if chunk.get("text") else ""
        if act and section:
            status_msgs.append(f"📑 [{act} — Section {section}] {title or text_preview}")
        elif act:
            status_msgs.append(f"📑 [{act}] {title or text_preview}")

    return {
        "legal_chunks": matches,
        "legal_followup_query": "",  # Clear after use
        "status_log": status_msgs,
    }


def retrieve_books_node(state: VidhijnaState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)

    # Use targeted followup from reflection, else supervisor's rewrite
    query = state.books_followup_query or state.rewritten_query or state.query

    matches = retrieve_books(
        query=query,
        top_k=cfg.retrieval_top_k_books,
        top_n=cfg.rerank_top_n_books,
        score_threshold=cfg.retrieval_score_threshold,
    )

    status_msgs = [
        f"📚 Searched vidhijna-books — Found {len(matches)} commentary excerpts",
    ]
    for chunk in matches[:3]:
        meta = chunk.get("metadata", {})
        source = meta.get("source", meta.get("book_name", ""))
        text_preview = chunk.get("text", "")[:100] if chunk.get("text") else ""
        if source:
            status_msgs.append(f"📖 [{source}] {text_preview[:80]}...")

    return {
        "book_chunks": matches,
        "books_followup_query": "",  # Clear after use
        "vector_loop_count": state.vector_loop_count + 1,
        "status_log": status_msgs,
    }


def web_search_node(state: VidhijnaState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    results = []

    if state.tavily_signals:
        for signal in state.tavily_signals:
            domains = signal.target_domains or cfg.get_domains_for_fetch_type(signal.fetch_type)
            fetched = tavily_search(
                query=signal.query,
                fetch_type=signal.fetch_type,
                target_domains=domains,
                max_results=cfg.tavily_max_results,
            )
            results.extend(fetched)
    elif state.web_followup_query:
        # Reflection asked for a specific web search
        results = tavily_search(
            query=state.web_followup_query,
            fetch_type="general",
            max_results=2,
        )
    else:
        results = tavily_search(
            query=state.rewritten_query or state.query,
            fetch_type="general",
            max_results=2,
        )

    status_msgs = [
        f"🌐 Web search — Found {len(results)} results",
    ]
    for r in results[:4]:
        title = r.get("title", "")[:60]
        url = r.get("url", "")
        domain = url.split("/")[2] if url and len(url.split("/")) > 2 else ""
        if title:
            status_msgs.append(f"🔗 [{domain}] {title}")

    return {
        "web_results": results,
        "sources_gathered": [r.get("url", "") for r in results if r.get("url")],
        "web_search_loop_count": state.web_search_loop_count + 1,
        "web_followup_query": "",  # Clear after use
        "tavily_results_log": [{"timestamp": datetime.utcnow().isoformat(), "count": len(results)}],
        "status_log": status_msgs,
    }


# ── Separate summarizers (no more chunk mixing) ──────────────────────────────

def summarize_legal_node(state: VidhijnaState, config: RunnableConfig) -> dict:
    """Summarize ONLY legal provisions — statutes, sections, penalties."""
    if not state.legal_chunks:
        return {"status_log": ["⚖️ No legal provisions to summarize."]}

    cfg = Configuration.from_runnable_config(config)
    llm = _llm(cfg.groq_model)  # 8B — summarization doesn't need 70B reasoning
    query = state.rewritten_query or state.query

    formatted = format_chunks(state.legal_chunks)
    prompt = LEGAL_RETRIEVAL_SUMMARY_PROMPT.format(query=query, chunks=formatted)
    if state.legal_summary:
        prompt = f"Existing summary:\n{state.legal_summary}\n\nNew provisions found — integrate into the summary:\n{prompt}"

    legal_summary = clean_thinking_tags(_invoke_safe(llm, [SystemMessage(content=prompt)]))

    return {
        "legal_summary": legal_summary,
        "status_log": [
            "📝 Summarizing statutory provisions...",
            f"⚖️ {legal_summary[:120]}...",
        ],
    }


def summarize_books_node(state: VidhijnaState, config: RunnableConfig) -> dict:
    """Summarize ONLY legal commentary — reasoning, interpretation, case analysis."""
    if not state.book_chunks:
        return {"status_log": ["📚 No commentary to summarize."]}

    cfg = Configuration.from_runnable_config(config)
    llm = _llm(cfg.groq_model)  # 8B — summarization doesn't need 70B reasoning
    query = state.rewritten_query or state.query

    formatted = format_chunks(state.book_chunks)
    prompt = BOOKS_RETRIEVAL_SUMMARY_PROMPT.format(query=query, chunks=formatted)
    if state.books_summary:
        prompt = f"Existing summary:\n{state.books_summary}\n\nNew commentary found — integrate into the summary:\n{prompt}"

    books_summary = clean_thinking_tags(_invoke_safe(llm, [SystemMessage(content=prompt)]))

    return {
        "books_summary": books_summary,
        "status_log": [
            "📚 Summarizing legal commentary...",
            f"📖 {books_summary[:120]}...",
        ],
    }


def summarize_web_node(state: VidhijnaState, config: RunnableConfig) -> dict:
    """Summarize web results — case law, circulars, recent amendments."""
    if not state.web_results:
        return {"web_summary": "", "status_log": ["🌐 No web results to summarize."]}

    cfg = Configuration.from_runnable_config(config)
    llm = _llm(cfg.groq_model)  # 8B — summarization doesn't need 70B reasoning

    prompt = WEB_RESEARCH_SUMMARY_PROMPT.format(
        query=state.rewritten_query or state.query,
        results=format_web_results(state.web_results),
    )
    if state.web_summary:
        prompt = f"Existing web research:\n{state.web_summary}\n\nNew findings — integrate:\n{prompt}"

    web_summary = clean_thinking_tags(_invoke_safe(llm, [SystemMessage(content=prompt)]))

    return {
        "web_summary": web_summary,
        "status_log": [
            "📰 Analyzing web results...",
            f"🌐 {web_summary[:120]}...",
        ],
    }


# ── Combine, Extract, Reflect ─────────────────────────────────────────────────

def combine_summaries(state: VidhijnaState, config: RunnableConfig) -> dict:
    running = f"""# Legal Research Summary

## Applicable Law
{state.legal_summary or "No relevant law found."}

## Legal Commentary & Reasoning
{state.books_summary or "No commentary found."}

## Web Research (Cases & Regulations)
{state.web_summary or "No web results."}
"""
    n_legal = len(state.legal_chunks or [])
    n_books = len(state.book_chunks or [])
    n_web = len(state.web_results or [])

    return {
        "running_summary": running,
        "status_log": [
            "🔗 Consolidating all research...",
            f"📊 Sources: {n_legal} provisions, {n_books} commentary, {n_web} web results",
        ],
    }


def extract_entities(state: VidhijnaState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)

    if not state.running_summary:
        return {"status_log": ["⚠️ No data for entity extraction."]}

    prompt = f"""Extract legal entities from this research. Return JSON with keys:
statutes, cases, principles, jurisdictions, dates, parties (each a list of strings).

Text: {state.running_summary[:5000]}"""

    content = _invoke_with_json_fallback(cfg.research_model, temperature=0, prompt=prompt,
                                          reasoning_effort="low")
    entities = extract_json_from_text(content)

    if entities:
        status_msgs = ["🏛️ Extracted legal entities:"]
        for category, items in entities.items():
            if items and isinstance(items, list) and len(items) > 0:
                items_preview = ", ".join(str(i) for i in items[:3])
                icon = {"statutes": "⚖️", "cases": "🏛️", "principles": "📜",
                        "jurisdictions": "🏢", "dates": "📅", "parties": "👥"}.get(category, "📌")
                status_msgs.append(f"{icon} {category.title()}: {items_preview}")
        return {"legal_entities": entities, "status_log": status_msgs}
    return {"status_log": ["📌 No entities extracted."]}


def reflect(state: VidhijnaState, config: RunnableConfig) -> dict:
    """Quality check — identifies gaps and generates TYPED followup queries
    so each source (legal, books, web) gets a targeted re-query."""
    cfg = Configuration.from_runnable_config(config)

    prompt = REFLECTION_PROMPT.format(
        query=state.query,
        legal_summary=state.legal_summary or "None",
        books_summary=state.books_summary or "None",
        web_summary=state.web_summary or "None",
    )
    content = _invoke_with_json_fallback(cfg.research_model, temperature=0.3, prompt=prompt,
                                          reasoning_effort="low")
    data = extract_json_from_text(content)
    if not data:
        data = {"has_gaps": False, "gaps": []}

    # Build Tavily signals from reflection
    new_signals = []
    if data.get("tavily_needed") and data.get("tavily_query"):
        new_signals.append(TavilyFetchSignal(
            fetch_type=data.get("tavily_fetch_type", "general"),
            query=data["tavily_query"],
            target_domains=cfg.get_domains_for_fetch_type(data.get("tavily_fetch_type", "general")),
            reason="Gap found during reflection",
            priority="high",
        ))

    gaps = data.get("gaps", [])

    status_msgs = ["🤔 Evaluating research quality..."]
    if gaps:
        status_msgs.append(f"⚠️ Found {len(gaps)} knowledge gaps:")
        for gap in gaps[:3]:
            status_msgs.append(f"   • {gap}")
        status_msgs.append(f"🔄 Loop {state.reflection_loop_count + 1}: Re-searching all 3 sources...")
    else:
        status_msgs.append("✅ Research comprehensive. Generating final report.")

    return {
        "knowledge_gaps": gaps,
        "reflection_loop_count": state.reflection_loop_count + 1,
        # Typed followup queries — each retrieval node reads its own
        "legal_followup_query": data.get("legal_followup_query", ""),
        "books_followup_query": data.get("books_followup_query", ""),
        "web_followup_query": data.get("web_followup_query", ""),
        # Fallback: update rewritten_query for nodes that don't have a typed followup
        "rewritten_query": data.get("legal_followup_query") or state.rewritten_query,
        "tavily_signals": new_signals,
        "needs_web_search": bool(new_signals),
        "status_log": status_msgs,
    }


def should_loop(
    state: VidhijnaState, config: RunnableConfig,
) -> Literal["propose_plan", "finalize"]:
    """Loop back to propose_plan which fans out to ALL 3 retrieval nodes."""
    cfg = Configuration.from_runnable_config(config)
    if state.knowledge_gaps and state.reflection_loop_count < cfg.max_reflection_loops:
        return "propose_plan"
    return "finalize"


def finalize(state: VidhijnaState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    llm = _llm(cfg.research_model, reasoning_effort="low")

    r_content = _invoke_safe(llm, [SystemMessage(content=FINAL_RESEARCH_PROMPT.format(
        query=state.query,
        legal_summary=state.legal_summary or "Not found.",
        books_summary=state.books_summary or "Not found.",
        web_summary=state.web_summary or "No web results.",
    ))])

    final_text = sanitize_legal_sections(clean_thinking_tags(r_content))

    citations = []
    for chunk in (state.legal_chunks or [])[:5]:
        meta = chunk.get("metadata", {})
        if meta.get("act_name") and meta.get("section_number"):
            citations.append(f"{meta['act_name']} — Section {meta['section_number']}")
        elif meta.get("act_name"):
            citations.append(meta['act_name'])
    for res in (state.web_results or [])[:3]:
        if res.get("url"):
            citations.append(res["url"])

    return {
        "running_summary": final_text,
        "final_response": final_text,
        "citations": list(dict.fromkeys(citations)),
        "status_log": [
            "📊 Final research report generated.",
            f"📎 {len(citations)} citations attached.",
        ],
    }


# ── Build subgraph ─────────────────────────────────────────────────────────────

def build_research_graph():
    b = StateGraph(VidhijnaState)

    b.add_node("propose_plan",    propose_plan)
    b.add_node("retrieve_legal",  retrieve_legal_node)
    b.add_node("retrieve_books",  retrieve_books_node)
    b.add_node("web_search",      web_search_node)
    b.add_node("summarize_legal", summarize_legal_node)
    b.add_node("summarize_books", summarize_books_node)
    b.add_node("summarize_web",   summarize_web_node)
    b.add_node("combine",         combine_summaries)
    b.add_node("extract_entities", extract_entities)
    b.add_node("reflect",         reflect)
    b.add_node("finalize",        finalize)

    # Entry → plan (supervisor already rewrote query, no need for generate_query)
    b.add_edge(START, "propose_plan")

    # Plan fans out to ALL 3 retrieval sources in parallel
    b.add_edge("propose_plan", "retrieve_legal")
    b.add_edge("propose_plan", "retrieve_books")
    b.add_edge("propose_plan", "web_search")

    # Each retrieval → its OWN summarizer (no chunk mixing)
    b.add_edge("retrieve_legal", "summarize_legal")
    b.add_edge("retrieve_books", "summarize_books")
    b.add_edge("web_search",     "summarize_web")

    # Legal + books summaries run parallel, then web summary feeds into combine
    # This staggers LLM calls to avoid hitting Groq TPM limits
    b.add_edge("summarize_legal", "combine")
    b.add_edge("summarize_books", "combine")
    b.add_edge("summarize_web",   "combine")
    b.add_edge("combine",         "extract_entities")
    b.add_edge("extract_entities", "reflect")

    # Reflect loops back to propose_plan (re-runs ALL 3 sources)
    b.add_conditional_edges("reflect", should_loop, {
        "propose_plan": "propose_plan",
        "finalize":     "finalize",
    })

    b.add_edge("finalize", END)
    return b.compile()


research_graph = build_research_graph()
