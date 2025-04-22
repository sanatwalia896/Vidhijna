from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import Dict, Any, List, Optional
import asyncio
from uuid import uuid4
import json

from api.models import (
    ResearchRequest,
    QueryResponse,
    VectorStoreResponse,
    WebResearchResponse,
    SummaryResponse,
    LegalEntitiesResponse,
    ReflectionResponse,
    FinalSummaryResponse,
    FullResearchResponse,
    ConfigurationModel,
)

from agents.state import SummaryState
from agents.configuration import Configuration, SearchAPI
from agents.graph import (
    generate_query,
    retrieve_from_vector_stores,
    web_research,
    summarize_vectors,
    summarize_legal_web_sources,
    combine_summaries,
    reflect_on_legal_research,
    finalize_legal_summary,
    graph,
)

router = APIRouter()

# Store ongoing research tasks and their states
research_tasks = {}


def get_runnable_config(config: Optional[ConfigurationModel] = None) -> Dict[str, Any]:
    """Convert config model to runnable config dictionary"""
    if config is None:
        return {
            "configurable": {
                "ollama_base_url": "http://localhost:11434",
                "local_llm": "llama3",
                "search_api": SearchAPI.DUCKDUCKGO,
                "max_web_research_loops": 3,
                "fetch_full_page": True,
            }
        }

    return {
        "configurable": {
            "ollama_base_url": config.ollama_base_url,
            "local_llm": config.local_llm,
            "search_api": config.search_api,
            "max_web_research_loops": config.max_web_research_loops,
            "fetch_full_page": config.fetch_full_page,
        }
    }


@router.post("/start", response_model=Dict[str, str])
async def start_research(request: ResearchRequest):
    """Start a new legal research task"""
    task_id = str(uuid4())

    # Initialize state
    state = SummaryState(
        research_topic=request.research_topic,
        search_query="",
        websearch_loop_count=0,
        vectorstore_loop_count=0,
        laws_research_results=[],
        cases_research_results=[],
        complete_research_results=[],
        web_research_results=[],
        sources_gathered=[],
        websearch_summary="",
        vector_summary="",
        running_summary="",
        legal_entities={},
    )

    # Store the state for later access
    research_tasks[task_id] = {
        "state": state,
        "config": get_runnable_config(request.configuration),
        "status": "initialized",
    }

    return {"task_id": task_id, "status": "initialized"}


@router.post("/run_full", response_model=Dict[str, str])
async def run_full_research(
    request: ResearchRequest, background_tasks: BackgroundTasks
):
    """Run the full legal research pipeline in the background"""
    task_id = str(uuid4())

    # Define background task
    async def run_graph():
        try:
            # Set task status to running
            research_tasks[task_id]["status"] = "running"

            # Initialize the input for the graph
            input_data = {"research_topic": request.research_topic}

            # Run the full graph
            config = get_runnable_config(request.configuration)
            result = await graph.ainvoke(input_data, config)

            # Store the result
            research_tasks[task_id]["result"] = result
            research_tasks[task_id]["status"] = "completed"
        except Exception as e:
            research_tasks[task_id]["status"] = "failed"
            research_tasks[task_id]["error"] = str(e)

    # Set up the task
    research_tasks[task_id] = {"request": request.dict(), "status": "initialized"}

    # Start the background task
    background_tasks.add_task(run_graph)

    return {"task_id": task_id, "status": "started"}


@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a research task"""
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return {"task_id": task_id, "status": research_tasks[task_id]["status"]}


@router.get("/result/{task_id}", response_model=Optional[FullResearchResponse])
async def get_task_result(task_id: str):
    """Get the full result of a completed research task"""
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = research_tasks[task_id]

    if task["status"] != "completed":
        return {"task_id": task_id, "status": task["status"], "result": None}

    if "result" not in task:
        raise HTTPException(status_code=404, detail="Result not found")

    # Return the full result
    return {
        "research_topic": task["request"]["research_topic"],
        "final_summary": task["result"]["running_summary"],
        "sources": task["result"].get("sources_gathered", []),
        "legal_entities": task["result"].get("legal_entities", {}),
    }


# Individual node endpoints for step-by-step processing


@router.post("/{task_id}/generate_query", response_model=QueryResponse)
async def api_generate_query(task_id: str):
    """Generate a legal-focused query for search"""
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = research_tasks[task_id]
    result = generate_query(task["state"], task["config"])

    # Update state
    task["state"].search_query = result["search_query"]

    return result


@router.post(
    "/{task_id}/retrieve_from_vector_stores", response_model=VectorStoreResponse
)
async def api_retrieve_from_vector_stores(task_id: str):
    """Retrieve content from vector stores"""
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = research_tasks[task_id]

    if not task["state"].search_query:
        raise HTTPException(status_code=400, detail="Generate a query first")

    result = retrieve_from_vector_stores(task["state"], task["config"])

    # Update state
    task["state"].laws_research_results = result["laws_research_results"]
    task["state"].cases_research_results = result["cases_research_results"]
    task["state"].complete_research_results = result["complete_research_results"]
    task["state"].vectorstore_loop_count = result["vectorstore_loop_count"]

    return result


@router.post("/{task_id}/web_research", response_model=WebResearchResponse)
async def api_web_research(task_id: str):
    """Gather information from the web with a legal focus"""
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = research_tasks[task_id]

    if not task["state"].search_query:
        raise HTTPException(status_code=400, detail="Generate a query first")

    result = web_research(task["state"], task["config"])

    # Update state
    task["state"].sources_gathered.extend(result["sources_gathered"])
    task["state"].web_research_results.extend(result["web_research_results"])
    task["state"].websearch_loop_count = result["websearch_loop_count"]

    return result


@router.post("/{task_id}/summarize_vectors", response_model=SummaryResponse)
async def api_summarize_vectors(task_id: str):
    """Summarize vector search results"""
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = research_tasks[task_id]

    if not task["state"].complete_research_results:
        raise HTTPException(status_code=400, detail="Retrieve from vector stores first")

    result = summarize_vectors(task["state"], task["config"])

    # Update state
    task["state"].vector_summary = result["vector_summary"]

    return {"summary": result["vector_summary"]}


@router.post("/{task_id}/summarize_legal_web_sources", response_model=SummaryResponse)
async def api_summarize_legal_web_sources(task_id: str):
    """Summarize legal web sources"""
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = research_tasks[task_id]

    if not task["state"].web_research_results:
        raise HTTPException(status_code=400, detail="Perform web research first")

    result = summarize_legal_web_sources(task["state"], task["config"])

    # Update state
    task["state"].websearch_summary = result["websearch_summary"]

    return {"summary": result["websearch_summary"]}


@router.post("/{task_id}/combine_summaries", response_model=SummaryResponse)
async def api_combine_summaries(task_id: str):
    """Combine web and vector summaries"""
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = research_tasks[task_id]

    if not task["state"].vector_summary and not task["state"].websearch_summary:
        raise HTTPException(status_code=400, detail="Generate summaries first")

    result = combine_summaries(task["state"], task["config"])

    # Update state
    task["state"].running_summary = result["running_summary"]

    return {"summary": result["running_summary"]}


@router.post("/{task_id}/reflect_on_legal_research", response_model=ReflectionResponse)
async def api_reflect_on_legal_research(task_id: str):
    """Reflect on the legal research and generate follow-up queries"""
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = research_tasks[task_id]

    if not task["state"].running_summary:
        raise HTTPException(status_code=400, detail="Combine summaries first")

    result = reflect_on_legal_research(task["state"], task["config"])

    # Update state
    task["state"].search_query = result["search_query"]

    return result


@router.post("/{task_id}/finalize_legal_summary", response_model=FinalSummaryResponse)
async def api_finalize_legal_summary(task_id: str):
    """Generate a final legal analysis with recommendations"""
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = research_tasks[task_id]

    if not task["state"].running_summary:
        raise HTTPException(status_code=400, detail="Complete previous steps first")

    result = finalize_legal_summary(task["state"], task["config"])

    # Update state
    task["state"].running_summary = result["running_summary"]

    return result


@router.delete("/{task_id}")
async def delete_task(task_id: str):
    """Delete a research task and clean up resources"""
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    del research_tasks[task_id]
    return {"status": "deleted"}
