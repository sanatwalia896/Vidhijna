# state.py
import operator
from dataclasses import dataclass, field
from typing import Annotated, Optional
from langchain_core.messages import BaseMessage


# ── Tavily fetch intent ────────────────────────────────────────────────────────

@dataclass(kw_only=True)
class TavilyFetchSignal:
    """
    Tells the web search node exactly what to fetch from Tavily and why.
    Supervisor generates these based on what is missing from the vector store
    or what the user explicitly asked for.
    """
    fetch_type: str = field(default="")
    # "case"        — specific case or similar judgments
    # "regulation"  — regulation/circular not in vector store (GST, RBI etc.)
    # "recent"      — recent judgments or amendments on a topic
    # "notification"— SEBI/MCA/RBI notification or circular
    # "general"     — general legal article or commentary

    query: str = field(default="")
    # Actual Tavily search query

    target_domains: list = field(default_factory=list)
    # Domains to restrict Tavily search to
    # Cases:       ["main.sci.gov.in", "nclt.gov.in", "delhihighcourt.nic.in"]
    # GST:         ["cbic-gst.gov.in", "gstcouncil.gov.in"]
    # SEBI:        ["sebi.gov.in"]
    # RBI:         ["rbi.org.in"]

    reason: str = field(default="")
    # Why this fetch is needed — for logging and reflection
    # e.g. "GST Act not in vector store"
    # e.g. "User asked for recent SC judgment on topic"
    # e.g. "Reflection loop found gap in IBC liquidation"

    priority: str = field(default="medium")
    # "high" | "medium" | "low"
    # high   = must fetch before answering
    # medium = fetch if time allows
    # low    = nice to have


# ── Domain maps for Tavily ────────────────────────────────────────────────────

CASE_DOMAINS = [
    "main.sci.gov.in",
    "delhihighcourt.nic.in",
    "bombayhighcourt.nic.in",
    "nclt.gov.in",
    "nclat.nic.in",
    "cci.gov.in",
]

REGULATION_DOMAINS = {
    "gst":         ["cbic-gst.gov.in", "gstcouncil.gov.in"],
    "sebi":        ["sebi.gov.in"],
    "rbi":         ["rbi.org.in"],
    "ibbi":        ["ibbi.gov.in"],
    "mca":         ["mca.gov.in"],
    "nclt":        ["nclt.gov.in", "nclat.nic.in"],
    "ip":          ["ipindia.gov.in", "copyright.gov.in"],
    "tax":         ["incometax.gov.in", "cbdt.gov.in"],
    "consumer":    ["consumeraffairs.nic.in", "ncdrc.nic.in"],
    "competition": ["cci.gov.in"],
    "rera":        ["rera.maharashtra.gov.in", "hrera.org.in"],
    "banking":     ["rbi.org.in", "drt.gov.in"],
}

# Topics not well covered in vector store — supervisor auto-triggers Tavily
VECTOR_STORE_GAPS = [
    "gst", "goods and services tax", "input tax credit",
    "rbi circular", "rbi master direction", "rbi notification",
    "income tax", "transfer pricing", "tds",
    "sebi notification", "sebi circular", "sebi order",
    "mca circular", "mca notification",
    "ibbi regulation", "cirp regulation", "liquidation regulation",
    "rera", "real estate", "rera registration",
    "patent", "patent filing", "patent infringement",
    "copyright registration", "trademark registration",
    "nclt rule", "nclat judgment",
]


# ── Main agent state ───────────────────────────────────────────────────────────

@dataclass(kw_only=True)
class VidhijnaState:
    """Complete state for the Vidhijna multi-agent system."""

    # ── Conversation ──────────────────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], operator.add] = field(
        default_factory=list
    )
    query: str = field(default="")
    thread_id: str = field(default="default")

    # ── Intent & routing ──────────────────────────────────────────────────────
    intent: str = field(default="")   # "chat"|"research"|"document"|"draft"
    mode: str = field(default="research") 
    complexity_score: str = field(default="medium") 

# "chat"|"research"

    # ── Query processing ──────────────────────────────────────────────────────
 
    rewritten_query: str = field(default="")

    # Single general filter (backwards compat) — supervisor writes this
    retrieval_filters: dict = field(default_factory=dict)

    # Track-specific filters — reflection + query nodes write these
    legal_filters: dict = field(default_factory=dict)
    # e.g. {"act_name": "Indian Contract Act, 1872", "doc_type": "act", "legal_domain": "contract"}

    books_filters: dict = field(default_factory=dict)
    # e.g. {"reasoning_focus": "application", "book_type": "commentary"}

    target_namespaces: list = field(default_factory=list)
    # ── Tavily fetch signals ───────────────────────────────────────────────────
    # Supervisor populates based on:
    #   1. Query topics matching VECTOR_STORE_GAPS
    #   2. User asking for cases / recent judgments
    #   3. Gaps found during reflection loop
    tavily_signals: list = field(default_factory=list)  # list[TavilyFetchSignal]

    # Derived flag — True if any signals exist
    needs_web_search: bool = field(default=False)

    # Log of what Tavily actually fetched each loop
    tavily_results_log: Annotated[list, operator.add] = field(
        default_factory=list
    )

    # ── Retrieval results ─────────────────────────────────────────────────────
    legal_chunks: Annotated[list, operator.add] = field(default_factory=list)
    book_chunks: Annotated[list, operator.add] = field(default_factory=list)
    web_results: Annotated[list, operator.add] = field(default_factory=list)
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list)
    legal_rerank_scores: dict = field(default_factory=dict)
    books_rerank_scores: dict = field(default_factory=dict)



    # ── Summaries ─────────────────────────────────────────────────────────────
    legal_summary: str = field(default="")
    books_summary: str = field(default="")
    web_summary: str = field(default="")
    running_summary: str = field(default="")
# ── Reflection & loop control ─────────────────────────────────────────────────
    knowledge_gaps: list = field(default_factory=list)

# Old — single mixed list (keep for backwards compat)
    followup_queries: Annotated[list, operator.add] = field(default_factory=list)

# New — three typed query tracks (reflection writes, retrieval nodes read)
    legal_followup_query: str = field(default="")
# What specific provision / section is still missing from legal namespace

    books_followup_query: str = field(default="")
# What interpretation / principle / application is still unclear from books

    web_followup_query: str = field(default="")
# What recent judgment / amendment / circular needs web fetch

    reflection_loop_count: int = field(default=0)
    web_search_loop_count: int = field(default=0)
    vector_loop_count: int = field(default=0)

    # ── Document handling ─────────────────────────────────────────────────────
    uploaded_file_bytes: bytes = field(default=b"")   # raw file upload
    uploaded_file_text: str = field(default="")        # extracted text (OCR or pre-extracted)
    uploaded_file_type: str = field(default="")
    uploaded_file_name: str = field(default="")
    document_analysis: dict = field(default_factory=dict)
    extracted_clauses: list = field(default_factory=list)
    risk_flags: list = field(default_factory=list)

    # ── Drafting ─────────────────────────────────────────────────────────────
    draft_type: str = field(default="")
    draft_inputs: dict = field(default_factory=dict)
    draft_output: str = field(default="")
    draft_history: Annotated[list, operator.add] = field(default_factory=list)

    # ── Legal entities ────────────────────────────────────────────────────────
    legal_entities: dict = field(default_factory=lambda: {
        "statutes": [], "cases": [], "principles": [],
        "parties": [], "courts": [], "dates": [],
    })

    # ── Final output ──────────────────────────────────────────────────────────
    final_response: str = field(default="")
    citations: list = field(default_factory=list)
    disclaimer_added: bool = field(default=False)

    # ── Error handling ────────────────────────────────────────────────────────
    error: str = field(default="")
    error_node: str = field(default="")

    # ── UX & Logging ──────────────────────────────────────────────────────────
    status_log: Annotated[list[str], operator.add] = field(default_factory=list)


# ── Input / Output ────────────────────────────────────────────────────────────

@dataclass(kw_only=True)
class VidhijnaInput:
    query: str = field(default="")
    thread_id: str = field(default="default")
    mode: str = field(default="research")
    uploaded_file_bytes: bytes = field(default=b"")
    uploaded_file_text: str = field(default="")
    uploaded_file_name: str = field(default="")
    uploaded_file_type: str = field(default="")
    draft_type: str = field(default="")
    draft_inputs: dict = field(default_factory=dict)


@dataclass(kw_only=True)
class VidhijnaOutput:
    final_response: str = field(default="")
    citations: list = field(default_factory=list)
    legal_entities: dict = field(default_factory=dict)
    sources_gathered: list = field(default_factory=list)
    draft_output: str = field(default="")
    error: str = field(default="")


# ── Backwards compatibility ───────────────────────────────────────────────────

@dataclass(kw_only=True)
class SummaryState(VidhijnaState):
    research_topic: str = field(default="")
    search_query: str = field(default="")
    laws_research_results: Annotated[list, operator.add] = field(default_factory=list)
    cases_research_results: Annotated[list, operator.add] = field(default_factory=list)
    complete_research_results: Annotated[list, operator.add] = field(default_factory=list)
    websearch_loop_count: int = field(default=0)
    vectorstore_loop_count: int = field(default=0)
    vector_summary: str = field(default="")
    websearch_summary: str = field(default="")


@dataclass(kw_only=True)
class SummaryStateInput:
    research_topic: str = field(default="")


@dataclass(kw_only=True)
class SummaryStateOutput:
    running_summary: str = field(default="")
    vector_summary: str = field(default="")
    websearch_summary: str = field(default="")