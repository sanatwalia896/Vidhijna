from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union


class SearchAPI(str, Enum):
    TAVILY = "tavily"
    PERPLEXITY = "perplexity"
    DUCKDUCKGO = "duckduckgo"


class ConfigurationModel(BaseModel):
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Base URL for Ollama API"
    )
    local_llm: str = Field(default="llama3", description="Local LLM model to use")
    search_api: SearchAPI = Field(
        default=SearchAPI.DUCKDUCKGO, description="Search API to use"
    )
    max_web_research_loops: int = Field(
        default=3, description="Maximum number of web research loops"
    )
    fetch_full_page: bool = Field(
        default=True, description="Whether to fetch full page content"
    )


class ResearchRequest(BaseModel):
    research_topic: str = Field(..., description="Legal research topic or question")
    configuration: Optional[ConfigurationModel] = Field(
        default=None, description="Optional configuration"
    )


class QueryResponse(BaseModel):
    search_query: str = Field(..., description="Generated search query")


class VectorStoreResponse(BaseModel):
    formatted_laws: str = Field(
        ..., description="Formatted laws retrieved from vector store"
    )
    formatted_cases: str = Field(
        ..., description="Formatted cases retrieved from vector store"
    )
    formatted_combined: str = Field(..., description="Combined formatted results")
    vectorstore_loop_count: int = Field(
        ..., description="Number of vectorstore search loops"
    )


class WebResearchResponse(BaseModel):
    sources_gathered: List[str] = Field(
        ..., description="Sources gathered from web research"
    )
    web_research_results: List[str] = Field(..., description="Web research results")
    websearch_loop_count: int = Field(..., description="Number of web search loops")


class SummaryResponse(BaseModel):
    summary: str = Field(..., description="Generated summary")


class LegalEntitiesResponse(BaseModel):
    statutes: List[str] = Field(..., description="Statutes extracted")
    cases: List[str] = Field(..., description="Cases extracted")
    principles: List[str] = Field(..., description="Legal principles extracted")
    jurisdictions: List[str] = Field(..., description="Jurisdictions extracted")
    dates: List[str] = Field(..., description="Relevant dates extracted")
    parties: List[str] = Field(..., description="Parties extracted")


class ReflectionResponse(BaseModel):
    search_query: str = Field(..., description="Follow-up search query")


class FinalSummaryResponse(BaseModel):
    running_summary: str = Field(
        ..., description="Final legal analysis with recommendations"
    )


class FullResearchResponse(BaseModel):
    research_topic: str = Field(..., description="Original research topic")
    final_summary: str = Field(..., description="Final legal analysis")
    sources: List[str] = Field(..., description="All sources used in research")
    legal_entities: Optional[Dict[str, List[str]]] = Field(
        None, description="Extracted legal entities"
    )
