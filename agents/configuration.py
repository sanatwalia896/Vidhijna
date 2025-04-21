# configuration.py

import os
from dataclasses import dataclass, fields
from typing import Optional, Any, Dict
from enum import Enum

from langchain_core.runnables import RunnableConfig


class SearchAPI(str, Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"


@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the research assistant."""

    # Research loop controls
    max_web_research_loops: int = int(os.environ.get("MAX_WEB_RESEARCH_LOOPS", "3"))
    max_vector_store_research_loops: int = int(
        os.environ.get("MAX_VECTOR_RESEARCH_LOOPS", "1")
    )

    # Rerouting and scoring
    search_depth: int = int(
        os.environ.get("SEARCH_DEPTH", "3")
    )  # Number of web queries per loop

    # LLM config
    local_llm: str = os.environ.get("OLLAMA_MODEL", "gemma3:1b")
    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/")

    # Search provider
    search_api: SearchAPI = SearchAPI(
        os.environ.get("SEARCH_API", SearchAPI.DUCKDUCKGO.value)
    )
    fetch_full_page: bool = os.environ.get("FETCH_FULL_PAGE", "False").lower() in (
        "true",
        "1",
        "t",
    )

    # Vector store paths
    laws_faiss_path: str = os.environ.get(
        "LAWS_FAISS_PATH",
        "/Users/sanatwalia/Desktop/Assignments_applications/Vidhijna/commercial_laws_index",
    )
    cases_faiss_path: str = os.environ.get(
        "CASES_FAISS_PATH",
        "/Users/sanatwalia/Desktop/Assignments_applications/Vidhijna/cases_index",
    )

    # Developer toggles
    dev_mode: bool = os.environ.get("DEV_MODE", "True").lower() in ("true", "1", "t")
    debug_mode: bool = os.environ.get("DEBUG_MODE", "True").lower() in (
        "true",
        "1",
        "t",
    )
    log_dir: str = os.environ.get("LOG_DIR", "logs")

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig and environment variables."""
        configurable = config.get("configurable", {}) if config else {}
        values: Dict[str, Any] = {}

        for f in fields(cls):
            env_key = f.name.upper()
            value = configurable.get(
                f.name, os.environ.get(env_key, getattr(cls, f.name, None))
            )
            # Ensure proper casting
            if isinstance(f.type, type) and f.type is bool:
                value = str(value).lower() in ("true", "1", "t")
            elif isinstance(f.type, type) and f.type is int:
                value = int(value)
            elif isinstance(f.type, type) and f.type is float:
                value = float(value)
            elif f.type == SearchAPI:
                value = SearchAPI(value)
            values[f.name] = value

        return cls(**values)

    def to_dict(self) -> dict:
        """Return the configuration as a dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
