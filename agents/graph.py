import json

from typing_extensions import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph

from agents.configuration import Configuration, SearchAPI
from agents.utils import (
    deduplicate_and_format_sources,
    tavily_search,
    format_sources,
    perplexity_search,
    duckduckgo_search,
    load_faiss_retriever,
    retrieve_from_laws_and_cases,
    summarize_vectors,
    combine_summaries,
    chunk_and_summarize,
)
from agents.state import SummaryState, SummaryStateInput, SummaryStateOutput
from agents.prompts import (
    query_writer_instructions,
    summarizer_instructions,
    reflection_instructions,
)


# Nodes
def generate_query(state: SummaryState, config: RunnableConfig):
    """Generate a query for web search"""

    # Format the prompt
    query_writer_instructions_formatted = query_writer_instructions.format(
        research_topic=state.research_topic
    )

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = ChatOllama(
        base_url=configurable.ollama_base_url,
        model=configurable.local_llm,
        temperature=0,
        format="json",
    )
    result = llm_json_mode.invoke(
        [
            SystemMessage(content=query_writer_instructions_formatted),
            HumanMessage(content=f"Generate a query for web search:"),
        ]
    )
    query = json.loads(result.content)

    return {"search_query": query["query"]}


# Node to Retrieve Content from Vector Stores (laws and cases)
def retrieve_from_vector_stores(state: SummaryState, config: RunnableConfig):
    """Retrieve content from both the 'vidhijan_laws' and 'vidhijan_cases' FAISS vector stores."""

    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Retrieve documents from laws and cases vector stores
    retrieval_results = retrieve_from_laws_and_cases(
        query=state.search_query, config=configurable
    )

    # Extract results
    laws_docs = retrieval_results["laws"]
    cases_docs = retrieval_results["cases"]

    # Combine both sets of documents
    combined_results = laws_docs + cases_docs

    # Update the state with the retrieved documents
    state.laws_research_results.extend(laws_docs)
    state.cases_research_results.extend(cases_docs)
    state.complete_research_results.extend(combined_results)

    return {
        "laws_research_results": [format_sources(laws_docs)],
        "cases_research_results": [format_sources(cases_docs)],
        "complete_research_results": [format_sources(combined_results)],
    }


def web_research(state: SummaryState, config: RunnableConfig):
    """Gather information from the web"""

    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Handle both cases for search_api:
    # 1. When selected in Studio UI -> returns a string (e.g. "tavily")
    # 2. When using default -> returns an Enum (e.g. SearchAPI.TAVILY)
    if isinstance(configurable.search_api, str):
        search_api = configurable.search_api
    else:
        search_api = configurable.search_api.value

    # Search the web
    if search_api == "tavily":
        search_results = tavily_search(
            state.search_query, include_raw_content=True, max_results=1
        )
        search_str = deduplicate_and_format_sources(
            search_results, max_tokens_per_source=2000, include_raw_content=True
        )
    elif search_api == "perplexity":
        search_results = perplexity_search(
            state.search_query, state.research_loop_count
        )
        search_str = deduplicate_and_format_sources(
            search_results, max_tokens_per_source=2000, include_raw_content=False
        )
    elif search_api == "duckduckgo":
        search_results = duckduckgo_search(
            state.search_query,
            max_results=3,
            fetch_full_page=configurable.fetch_full_page,
        )
        search_str = deduplicate_and_format_sources(
            search_results, max_tokens_per_source=2000, include_raw_content=True
        )
    else:
        raise ValueError(f"Unsupported search API: {configurable.search_api}")

    return {
        "sources_gathered": [format_sources(search_results)],
        "research_loop_count": state.research_loop_count + 1,
        "web_research_results": [search_str],
    }


def summarize_sources(state: SummaryState, config: RunnableConfig):
    """Summarize the gathered sources"""

    # Existing summary
    existing_summary = state.running_summary

    # Most recent web research
    most_recent_web_research = state.web_research_results[-1]

    # Build the human message
    if existing_summary:
        human_message_content = (
            f"<User Input> \n {state.research_topic} \n <User Input>\n\n"
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
            f"<New Search Results> \n {most_recent_web_research} \n <New Search Results>"
        )
    else:
        human_message_content = (
            f"<User Input> \n {state.research_topic} \n <User Input>\n\n"
            f"<Search Results> \n {most_recent_web_research} \n <Search Results>"
        )

    # Run the LLM
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOllama(
        base_url=configurable.ollama_base_url,
        model=configurable.local_llm,
        temperature=0,
    )
    result = llm.invoke(
        [
            SystemMessage(content=summarizer_instructions),
            HumanMessage(content=human_message_content),
        ]
    )

    running_summary = result.content

    # TODO: This is a hack to remove the <think> tags w/ Deepseek models
    # It appears very challenging to prompt them out of the responses
    while "<think>" in running_summary and "</think>" in running_summary:
        start = running_summary.find("<think>")
        end = running_summary.find("</think>") + len("</think>")
        running_summary = running_summary[:start] + running_summary[end:]

    return {"running_summary": running_summary}


def reflect_on_summary(state: SummaryState, config: RunnableConfig):
    """Reflect on the summary and generate a follow-up query"""

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = ChatOllama(
        base_url=configurable.ollama_base_url,
        model=configurable.local_llm,
        temperature=0,
        format="json",
    )
    result = llm_json_mode.invoke(
        [
            SystemMessage(
                content=reflection_instructions.format(
                    research_topic=state.research_topic
                )
            ),
            HumanMessage(
                content=f"Identify a knowledge gap and generate a follow-up web search query based on our existing knowledge: {state.running_summary}"
            ),
        ]
    )
    follow_up_query = json.loads(result.content)

    # Get the follow-up query
    query = follow_up_query.get("follow_up_query")

    # JSON mode can fail in some cases
    if not query:

        # Fallback to a placeholder query
        return {"search_query": f"Tell me more about {state.research_topic}"}

    # Update search query with follow-up query
    return {"search_query": follow_up_query["follow_up_query"]}


def finalize_summary(state: SummaryState):
    """Finalize the summary"""

    # Format all accumulated sources into a single bulleted list
    all_sources = "\n".join(source for source in state.sources_gathered)
    state.running_summary = (
        f"## Summary\n\n{state.running_summary}\n\n ### Sources:\n{all_sources}"
    )
    return {"running_summary": state.running_summary}


def route_research(
    state: SummaryState, config: RunnableConfig
) -> Literal["finalize_summary", "web_research", "retrieve_from_vector_stores"]:
    """Route the research based on the follow-up query"""

    configurable = Configuration.from_runnable_config(config)
    if state.research_loop_count <= int(configurable.max_web_research_loops):
        return "web_research"
    elif state.research_loop_count <= int(configurable.max_vector_store_research_loops):
        return "retrieve_from_vector_stores"  # New node for vector store retrieval
    else:
        return "finalize_summary"


# Define the state graph
builder = StateGraph(
    SummaryState,
    input=SummaryStateInput,
    output=SummaryStateOutput,
    config_schema=Configuration,
)

# Nodes
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("summarize_sources", summarize_sources)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("retrieve_from_vector_stores", retrieve_from_vector_stores)
builder.add_node(
    "summarize_vectors", summarize_vectors
)  # NEW: summarizing legal vector results
builder.add_node(
    "combine_summaries", combine_summaries
)  # NEW: merging legal and web summaries
builder.add_node("finalize_summary", finalize_summary)

# Edges
builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "web_research")
builder.add_edge("generate_query", "retrieve_from_vector_stores")
builder.add_edge("web_research", "summarize_sources")
builder.add_edge("retrieve_from_vector_stores", "summarize_vectors")
builder.add_edge("summarize_sources", "combine_summaries")
builder.add_edge("summarize_vectors", "combine_summaries")
builder.add_edge("combine_summaries", "reflect_on_summary")


# # Conditional routing after reflection
builder.add_conditional_edges("reflect_on_summary", route_research)


builder.add_edge("combine_summaries", "finalize_summary")

# Standard flow from summarization to finalization
builder.add_edge("finalize_summary", END)

# Compile the graph
graph = builder.compile()
