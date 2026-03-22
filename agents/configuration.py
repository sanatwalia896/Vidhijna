# configuration.py

import os
from dataclasses import dataclass, field, fields
from typing import Optional, Any, Dict
from enum import Enum

from langchain_core.runnables import RunnableConfig


# ── Enums ──────────────────────────────────────────────────────────────────────

class SearchAPI(str, Enum):
    TAVILY     = "tavily"
    DUCKDUCKGO = "duckduckgo"


class GroqModel(str, Enum):
    LLAMA_8B      = "llama-3.1-8b-instant"       # fast, free
    LLAMA_70B     = "llama-3.3-70b-versatile"     # best quality
    LLAMA_3B      = "llama-3.2-3b-preview"        # ultra fast


class ResearchMode(str, Enum):
    CHAT     = "chat"      # conversational, uses memory
    RESEARCH = "research"  # deep one-shot research report


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass(kw_only=True)
class Configuration:
    """
    Central configuration for Vidhijna.
    All values can be overridden via environment variables or RunnableConfig.
    """

    # ── LLM ───────────────────────────────────────────────────────────────────
    groq_model: str = os.environ.get("GROQ_MODEL", GroqModel.LLAMA_8B.value)
    groq_api_key: str = os.environ.get("GROQ_API_KEY", "")

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = os.environ.get(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    huggingface_token: str = os.environ.get("HUGGINGFACE_TOKEN", "")

    # ── Pinecone ──────────────────────────────────────────────────────────────
    pinecone_api_key: str = os.environ.get("PINECONE_API_KEY", "")
    pinecone_index: str = os.environ.get("PINECONE_INDEX_NAME", "vidhijana-indexes")
    pinecone_region: str = os.environ.get("PINECONE_REGION", "us-east-1")

    # Namespaces
    ns_legal: str = "vidhijna-legal"   # acts, sections, constitution
    ns_books: str = "vidhijna-books"   # commentary, study notes, reasoning

    # ── Search ────────────────────────────────────────────────────────────────
    search_api: SearchAPI = SearchAPI(
        os.environ.get("SEARCH_API", SearchAPI.TAVILY.value)
    )
    tavily_api_key: str = os.environ.get("TAVILY_API_KEY", "")
    fetch_full_page: bool = os.environ.get("FETCH_FULL_PAGE", "false").lower() in (
        "true", "1", "t"
    )

    # ── Research loop controls ────────────────────────────────────────────────
    # How many times the agent re-queries if gaps found
    max_reflection_loops: int = int(os.environ.get("MAX_REFLECTION_LOOPS", "3"))

    # How many web search queries per research loop
    max_web_queries: int = int(os.environ.get("MAX_WEB_QUERIES", "3"))

    # Top-k chunks from each Pinecone namespace
    retrieval_top_k_legal: int = int(os.environ.get("RETRIEVAL_TOP_K_LEGAL", "6"))
    retrieval_top_k_books: int = int(os.environ.get("RETRIEVAL_TOP_K_BOOKS", "4"))

    # Minimum similarity score to include a chunk
    retrieval_score_threshold: float = float(
        os.environ.get("RETRIEVAL_SCORE_THRESHOLD", "0.5")
    )

    # ── Research mode ─────────────────────────────────────────────────────────
    default_mode: ResearchMode = ResearchMode(
        os.environ.get("DEFAULT_MODE", ResearchMode.RESEARCH.value)
    )

    # ── Taxonomy ──────────────────────────────────────────────────────────────
    # Path to auto-generated taxonomy file from ingestion pipeline
    taxonomy_path: str = os.environ.get(
        "TAXONOMY_PATH", "vector_store_creation/taxonomy.py"
    )

    # ── Developer toggles ─────────────────────────────────────────────────────
    dev_mode: bool = os.environ.get("DEV_MODE", "true").lower() in ("true", "1", "t")
    debug_mode: bool = os.environ.get("DEBUG_MODE", "false").lower() in ("true", "1", "t")
    log_dir: str = os.environ.get("LOG_DIR", "logs")

    # ── Memory (for chat mode) ────────────────────────────────────────────────
    enable_memory: bool = os.environ.get("ENABLE_MEMORY", "true").lower() in (
        "true", "1", "t"
    )
    max_memory_messages: int = int(os.environ.get("MAX_MEMORY_MESSAGES", "20"))

    # ── API server ────────────────────────────────────────────────────────────
    api_host: str = os.environ.get("API_HOST", "0.0.0.0")
    api_port: int = int(os.environ.get("API_PORT", "8000"))
    api_key: str = os.environ.get("VIDHIJNA_API_KEY", "")   # for securing the FastAPI

    # ── classmethod: build from RunnableConfig ────────────────────────────────

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Build Configuration from RunnableConfig + environment variables."""
        configurable = config.get("configurable", {}) if config else {}
        values: Dict[str, Any] = {}

        for f in fields(cls):
            env_val = os.environ.get(f.name.upper())
            value   = configurable.get(f.name, env_val or getattr(cls, f.name, None))

            # Type casting
            if f.type == "bool":
                value = str(value).lower() in ("true", "1", "t")
            elif f.type == "int":
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    value = getattr(cls, f.name, 0)
            elif f.type == "float":
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = getattr(cls, f.name, 0.0)
            elif f.type == "SearchAPI":
                value = SearchAPI(value)
            elif f.type == "ResearchMode":
                value = ResearchMode(value)
            elif f.type == "GroqModel":
                value = GroqModel(value)

            values[f.name] = value

        return cls(**values)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def validate(self) -> list[str]:
        """Check required fields are set. Returns list of missing fields."""
        required = {
            "groq_api_key":       self.groq_api_key,
            "huggingface_token":  self.huggingface_token,
            "pinecone_api_key":   self.pinecone_api_key,
        }
        if self.search_api == SearchAPI.TAVILY:
            required["tavily_api_key"] = self.tavily_api_key

        return [k for k, v in required.items() if not v]

    def __post_init__(self):
        missing = self.validate()
        if missing and not self.dev_mode:
            raise ValueError(
                f"Missing required config fields: {missing}. "
                f"Set them in .env or as environment variables."
            )
        if missing and self.dev_mode:
            import warnings
            warnings.warn(
                f"Missing config fields (running in dev_mode): {missing}",
                stacklevel=2,
            )