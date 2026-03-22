# configuration.py

import os
from dataclasses import dataclass, field, fields
from typing import Optional, Any, Dict
from enum import Enum

from langchain_core.runnables import RunnableConfig


class SearchAPI(str, Enum):
    TAVILY     = "tavily"
    DUCKDUCKGO = "duckduckgo"


class GroqModel(str, Enum):
    LLAMA_8B  = "llama-3.1-8b-instant"
    LLAMA_70B = "llama-3.3-70b-versatile"
    LLAMA_3B  = "llama-3.2-3b-preview"


class ResearchMode(str, Enum):
    CHAT     = "chat"
    RESEARCH = "research"


class Intent(str, Enum):
    CHAT     = "chat"
    RESEARCH = "research"
    DOCUMENT = "document"
    DRAFT    = "draft"


class TavilyFetchType(str, Enum):
    CASE         = "case"
    REGULATION   = "regulation"
    RECENT       = "recent"
    NOTIFICATION = "notification"
    GENERAL      = "general"
    NONE         = "none"


@dataclass(kw_only=True)
class Configuration:
    """
    Central configuration for Vidhijna multi-agent system.
    Supervisor + Chat + Research + Document + Drafting agents.
    All values overridable via env vars or LangGraph RunnableConfig.
    """

    # ── LLM — one model per agent ─────────────────────────────────────────────
    groq_model: str = os.environ.get("GROQ_MODEL", GroqModel.LLAMA_8B.value)
    research_model: str = os.environ.get("RESEARCH_MODEL", GroqModel.LLAMA_70B.value)
    chat_model: str = os.environ.get("CHAT_MODEL", GroqModel.LLAMA_8B.value)
    supervisor_model: str = os.environ.get("SUPERVISOR_MODEL", GroqModel.LLAMA_8B.value)
    groq_api_key: str = os.environ.get("GROQ_API_KEY", "")

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = os.environ.get(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_dimension: int = int(os.environ.get("EMBEDDING_DIMENSION", "384"))
    huggingface_token: str = os.environ.get("HUGGINGFACE_TOKEN", "")

    # ── Pinecone ──────────────────────────────────────────────────────────────
    pinecone_api_key: str = os.environ.get("PINECONE_API_KEY", "")
    pinecone_index: str = os.environ.get("PINECONE_INDEX_NAME", "vidhijana-indexes")
    pinecone_region: str = os.environ.get("PINECONE_REGION", "us-east-1")
    ns_legal: str = "vidhijna-legal"
    ns_books: str = "vidhijna-books"
    retrieval_top_k_legal: int = int(os.environ.get("RETRIEVAL_TOP_K_LEGAL", "6"))
    retrieval_top_k_books: int = int(os.environ.get("RETRIEVAL_TOP_K_BOOKS", "4"))
    retrieval_score_threshold: float = float(
        os.environ.get("RETRIEVAL_SCORE_THRESHOLD", "0.5")
    )

    # ── Tavily search ─────────────────────────────────────────────────────────
    search_api: SearchAPI = SearchAPI(
        os.environ.get("SEARCH_API", SearchAPI.TAVILY.value)
    )
    tavily_api_key: str = os.environ.get("TAVILY_API_KEY", "")
    fetch_full_page: bool = os.environ.get("FETCH_FULL_PAGE", "false").lower() in (
        "true", "1", "t"
    )
    tavily_max_results: int = int(os.environ.get("TAVILY_MAX_RESULTS", "5"))
    tavily_search_depth: str = os.environ.get("TAVILY_SEARCH_DEPTH", "advanced")

    # Domains per fetch type
    case_search_domains: list = field(default_factory=lambda: [
        "main.sci.gov.in", "delhihighcourt.nic.in", "bombayhighcourt.nic.in",
        "nclt.gov.in", "nclat.nic.in", "cci.gov.in", "sebi.gov.in", "ibbi.gov.in",
    ])
    regulation_search_domains: list = field(default_factory=lambda: [
        "sebi.gov.in", "ibbi.gov.in", "cbic-gst.gov.in",
        "rbi.org.in", "mca.gov.in", "incometax.gov.in",
    ])
    news_search_domains: list = field(default_factory=lambda: [
        "livelaw.in", "barandbench.com", "scconline.com", "taxmann.com",
    ])

    # Topics not in vector store — auto-triggers Tavily
    vector_store_gaps: list = field(default_factory=lambda: [
        "gst", "goods and services tax", "input tax credit",
        "rbi circular", "rbi master direction",
        "income tax", "transfer pricing", "tds",
        "sebi circular", "sebi notification", "sebi order",
        "mca circular", "ibbi regulation", "cirp regulation",
        "rera", "patent", "trademark registration", "copyright registration",
    ])

    # ── Supervisor ────────────────────────────────────────────────────────────
    intent_confidence_threshold: float = float(
        os.environ.get("INTENT_CONFIDENCE_THRESHOLD", "0.7")
    )

    # ── Research agent ────────────────────────────────────────────────────────
    max_reflection_loops: int = int(os.environ.get("MAX_REFLECTION_LOOPS", "3"))
    max_web_queries: int = int(os.environ.get("MAX_WEB_QUERIES", "3"))

    # ── Chat agent ────────────────────────────────────────────────────────────
    enable_memory: bool = os.environ.get("ENABLE_MEMORY", "true").lower() in (
        "true", "1", "t"
    )
    max_memory_messages: int = int(os.environ.get("MAX_MEMORY_MESSAGES", "20"))

    # ── Document agent ────────────────────────────────────────────────────────
    max_file_size_mb: int = int(os.environ.get("MAX_FILE_SIZE_MB", "20"))
    supported_file_types: list = field(default_factory=lambda: [
        "pdf", "png", "jpg", "jpeg", "tiff", "docx"
    ])
    ocr_engine: str = os.environ.get("OCR_ENGINE", "pdfplumber")

    # ── Drafting agent ────────────────────────────────────────────────────────
    supported_draft_types: list = field(default_factory=lambda: [
        "nda", "service_agreement", "employment", "sale_deed", "lease",
        "legal_notice", "cease_desist", "reply_notice",
        "nclt_petition", "consumer_complaint", "arbitration_notice",
    ])
    default_jurisdiction: str = os.environ.get("DEFAULT_JURISDICTION", "India")

    # ── Response formatting ───────────────────────────────────────────────────
    legal_disclaimer: str = (
        "⚠️ This is AI-generated legal information for research purposes only. "
        "It does not constitute legal advice. Please consult a qualified lawyer "
        "before taking any legal action."
    )
    include_citations: bool = os.environ.get("INCLUDE_CITATIONS", "true").lower() in (
        "true", "1", "t"
    )

    # ── Taxonomy ──────────────────────────────────────────────────────────────
    taxonomy_path: str = os.environ.get(
        "TAXONOMY_PATH", "vector_store_creation/taxonomy.py"
    )

    # ── API server ────────────────────────────────────────────────────────────
    api_host: str = os.environ.get("API_HOST", "0.0.0.0")
    api_port: int = int(os.environ.get("API_PORT", "8000"))
    api_key: str = os.environ.get("VIDHIJNA_API_KEY", "")

    # ── Dev toggles ───────────────────────────────────────────────────────────
    dev_mode: bool = os.environ.get("DEV_MODE", "true").lower() in ("true", "1", "t")
    debug_mode: bool = os.environ.get("DEBUG_MODE", "false").lower() in (
        "true", "1", "t"
    )
    log_dir: str = os.environ.get("LOG_DIR", "logs")

    # ── classmethod ───────────────────────────────────────────────────────────

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        configurable = config.get("configurable", {}) if config else {}
        values: Dict[str, Any] = {}
        for f in fields(cls):
            env_val = os.environ.get(f.name.upper())
            value   = configurable.get(f.name, env_val or getattr(cls, f.name, None))
            if f.type == "bool":
                value = str(value).lower() in ("true", "1", "t")
            elif f.type == "int":
                try: value = int(value)
                except (TypeError, ValueError): value = getattr(cls, f.name, 0)
            elif f.type == "float":
                try: value = float(value)
                except (TypeError, ValueError): value = getattr(cls, f.name, 0.0)
            elif f.type == "SearchAPI": value = SearchAPI(value)
            elif f.type == "ResearchMode": value = ResearchMode(value)
            values[f.name] = value
        return cls(**values)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def get_model_for_agent(self, agent: str) -> str:
        return {
            "supervisor": self.supervisor_model,
            "chat":       self.chat_model,
            "research":   self.research_model,
            "document":   self.groq_model,
            "draft":      self.groq_model,
        }.get(agent, self.groq_model)

    def get_domains_for_fetch_type(self, fetch_type: str) -> list:
        return {
            "case":         self.case_search_domains,
            "regulation":   self.regulation_search_domains,
            "recent":       self.case_search_domains + self.news_search_domains,
            "notification": self.regulation_search_domains,
            "general":      self.case_search_domains + self.regulation_search_domains,
        }.get(fetch_type, [])

    def is_vector_store_gap(self, query: str) -> bool:
        query_lower = query.lower()
        return any(gap in query_lower for gap in self.vector_store_gaps)

    def validate(self) -> list[str]:
        required = {
            "groq_api_key":      self.groq_api_key,
            "huggingface_token": self.huggingface_token,
            "pinecone_api_key":  self.pinecone_api_key,
        }
        if self.search_api == SearchAPI.TAVILY:
            required["tavily_api_key"] = self.tavily_api_key
        return [k for k, v in required.items() if not v]

    def __post_init__(self):
        missing = self.validate()
        if missing and not self.dev_mode:
            raise ValueError(f"Missing required config: {missing}. Set in .env file.")
        if missing and self.dev_mode:
            import warnings
            warnings.warn(f"Missing config (dev_mode): {missing}", stacklevel=2)