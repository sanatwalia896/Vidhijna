"""
graph.py — Vidhijna supervisor graph

This is the MAIN entry point for the entire system.
It does ONE thing — classifies the user's intent and routes to the right specialist.

Four specialists (each a compiled subgraph in agents/subgraphs/):
  research_graph  → deep legal research with reflection loop
  chat_graph      → conversational Q&A with memory
  document_graph  → OCR + contract analysis + risk flags
  drafting_graph  → contract/notice/petition drafting

Flow:
  START → supervisor → [research | chat | document | draft] → response_formatter → END

Memory: MemorySaver persists conversation per thread_id across turns.
"""

from typing import Literal

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph
from dotenv import load_dotenv

from agents.state import (
    VidhijnaState,
    VidhijnaInput,
    VidhijnaOutput,
    TavilyFetchSignal,
)
from agents.configuration import Configuration
from agents.prompts import SUPERVISOR_PROMPT
from agents.utils import clean_thinking_tags, extract_json_from_text

# Import compiled subgraphs — each is its own StateGraph
from agents.subgraphs.research import research_graph
from agents.subgraphs.chat     import chat_graph
from agents.subgraphs.document import document_graph
from agents.subgraphs.drafting import drafting_graph

load_dotenv()


# ── Node 1: Supervisor ────────────────────────────────────────────────────────

def supervisor(state: VidhijnaState, config: RunnableConfig) -> dict:
    """
    The brain of the system. Runs on every single user message.

    Does 4 things:
    1. Reads the query + conversation history
    2. Classifies intent: chat | research | document | draft
    3. Rewrites the query for better vector store retrieval
    4. Generates Tavily signals if web search is needed
       (e.g. GST query → regulation fetch from cbic-gst.gov.in)
    """
    cfg = Configuration.from_runnable_config(config)

    # Use the fast 8B model — classification doesn't need heavy reasoning
    llm = ChatGroq(
        model=cfg.supervisor_model,
        temperature=0.1,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    # Build a short summary of recent conversation for context
    recent   = state.messages[-6:] if state.messages else []
    history  = " | ".join(f"{m.type}: {m.content[:80]}" for m in recent) or "None"
    has_file = "Yes" if (state.uploaded_file_text or state.uploaded_file_bytes) else "No"
    has_prior = "Yes" if (state.legal_summary or state.running_summary) else "No"

    # If user explicitly set mode to a valid intent, respect it
    user_mode = state.mode or ""
    valid_intents = {"chat", "research", "document", "draft"}

    result = llm.invoke([SystemMessage(content=SUPERVISOR_PROMPT.format(
        query=state.query,
        user_mode=user_mode if user_mode in valid_intents else "auto",
        history_summary=history,
        has_file=has_file,
        has_prior_research=has_prior,
    ))])

    # Parse supervisor output
    data = extract_json_from_text(result.content)
    if not data:
        data = {
            "intent":            "research",
            "rewritten_query":   state.query,
            "retrieval_filters": {},
            "target_namespaces": ["vidhijna-legal", "vidhijna-books"],
            "tavily_signals":    [],
            "needs_web_search":  False,
        }

    # Hard override: if user explicitly set mode, trust it over LLM classification
    if user_mode in valid_intents:
        data["intent"] = user_mode

    # Build Tavily signals from supervisor output
    raw_signals = data.get("tavily_signals", [])

    # Auto-add a signal if the query topic is known to be missing
    # from the vector store (e.g. GST, RBI circulars, patents)
    if cfg.is_vector_store_gap(state.query) and not raw_signals:
        raw_signals.append({
            "fetch_type":     "regulation",
            "query":          state.query,
            "target_domains": cfg.get_domains_for_fetch_type("regulation"),
            "reason":         "Topic likely missing from vector store",
            "priority":       "high",
        })

    # Convert raw dicts to typed TavilyFetchSignal objects
    signals = [
        TavilyFetchSignal(
            fetch_type=s.get("fetch_type", "general"),
            query=s.get("query", state.query),
            target_domains=s.get("target_domains", []),
            reason=s.get("reason", ""),
            priority=s.get("priority", "medium"),
        )
        for s in raw_signals
    ]

    return {
        "intent":            data.get("intent", "research"),
        "rewritten_query":   data.get("rewritten_query", state.query),
        "retrieval_filters": data.get("retrieval_filters", {}),
        "target_namespaces": data.get("target_namespaces", ["vidhijna-legal"]),
        "tavily_signals":    signals,
        "needs_web_search":  bool(signals),
        "messages":          [HumanMessage(content=state.query)],
    }


# ── Routing function ──────────────────────────────────────────────────────────

def route_intent(
    state: VidhijnaState,
) -> Literal["research_agent", "chat_agent", "document_agent", "draft_agent"]:
    """
    Reads state.intent set by the supervisor and returns
    the name of the next node to execute.
    """
    return {
        "chat":     "chat_agent",
        "document": "document_agent",
        "draft":    "draft_agent",
    }.get(state.intent, "research_agent")  # default to research


# ── Node 2: Response formatter ────────────────────────────────────────────────

def response_formatter(state: VidhijnaState, config: RunnableConfig) -> dict:
    """
    Last node before END. Runs after every specialist subgraph.

    Does 3 things:
    1. Picks up final_response from whichever subgraph ran
    2. Appends citations (act name + section number, web URLs)
    3. Appends the legal disclaimer
    """
    cfg      = Configuration.from_runnable_config(config)
    response = state.final_response or state.running_summary or "No response generated."

    # Append citations
    if cfg.include_citations and state.citations:
        cites    = "\n".join(f"• {c}" for c in state.citations[:8])
        response = f"{response}\n\n**Sources:**\n{cites}"

    # Append disclaimer (only once)
    if not state.disclaimer_added:
        response = f"{response}\n\n---\n{cfg.legal_disclaimer}"

    return {
        "final_response":   response,
        "disclaimer_added": True,
        "messages":         [AIMessage(content=response)],
    }


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph():
    """
    Assembles the full supervisor graph.

    The four specialist subgraphs are added as single nodes —
    LangGraph runs each subgraph's internal nodes automatically.
    """
    builder = StateGraph(
        VidhijnaState,
        input=VidhijnaInput,
        output=VidhijnaOutput,
        config_schema=Configuration,
    )

    # ── Nodes ─────────────────────────────────────────────────────────────────

    # Supervisor — always runs first
    builder.add_node("supervisor", supervisor)

    # Specialist subgraphs — each is a compiled StateGraph
    # LangGraph treats them as black boxes that read/write VidhijnaState
    builder.add_node("research_agent",  research_graph)   # subgraphs/research.py
    builder.add_node("chat_agent",      chat_graph)        # subgraphs/chat.py
    builder.add_node("document_agent",  document_graph)    # subgraphs/document.py
    builder.add_node("draft_agent",     drafting_graph)    # subgraphs/drafting.py

    # Shared formatter — always runs last
    builder.add_node("response_formatter", response_formatter)

    # ── Edges ─────────────────────────────────────────────────────────────────

    # Entry point always hits supervisor first
    builder.add_edge(START, "supervisor")

    # Supervisor decides which specialist to invoke
    builder.add_conditional_edges(
        "supervisor",
        route_intent,
        {
            "research_agent": "research_agent",
            "chat_agent":     "chat_agent",
            "document_agent": "document_agent",
            "draft_agent":    "draft_agent",
        },
    )

    # Every specialist goes to formatter when done
    builder.add_edge("research_agent",  "response_formatter")
    builder.add_edge("chat_agent",      "response_formatter")
    builder.add_edge("document_agent",  "response_formatter")
    builder.add_edge("draft_agent",     "response_formatter")

    # Formatter is the last stop
    builder.add_edge("response_formatter", END)

    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# Compile on import — graph is ready to use
graph = build_graph()