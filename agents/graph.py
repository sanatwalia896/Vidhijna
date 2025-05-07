# graph.py
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
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from agents.state import SummaryState, SummaryStateInput, SummaryStateOutput
from agents.prompts import (
    legal_query_rewriter_instructions,
    legal_summarizer_instructions,
    legal_reflection_instructions,
)


# Nodes
def generate_query(state: SummaryState, config: RunnableConfig):
    """Generate a legal-focused query for search"""

    # Format the prompt with legal-specific query writing instructions
    legal_query_instructions = legal_query_rewriter_instructions.format(
        research_topic=state.research_topic
    )

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = ChatOllama(
        base_url=configurable.ollama_base_url,
        model=configurable.local_llm,
        temperature=0.3,
        format="json",
    )
    result = llm_json_mode.invoke(
        [
            SystemMessage(content=legal_query_instructions),
            HumanMessage(content=f"Generate a legally-focused query for research:"),
        ]
    )
    query = json.loads(result.content)

    return {"search_query": query["query"]}


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

    # Return formatted strings for display and the actual Document objects for processing
    return {
        "laws_research_results": laws_docs,
        "cases_research_results": cases_docs,
        "complete_research_results": combined_results,
        # For logging/display purposes
        "formatted_laws": format_sources(laws_docs),
        "formatted_cases": format_sources(cases_docs),
        "formatted_combined": format_sources(combined_results),
        "vectorstore_loop_count": state.vectorstore_loop_count + 1,
    }


def web_research(state: SummaryState, config: RunnableConfig):
    """Gather information from the web with a legal focus"""

    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Handle both cases for search_api
    if isinstance(configurable.search_api, str):
        search_api = configurable.search_api
    else:
        search_api = configurable.search_api.value

    # Add legal context to the search query
    legal_query = f"legal perspective: {state.search_query}"

    # Search the web
    if search_api == "tavily":
        search_results = tavily_search(
            legal_query, include_raw_content=True, max_results=3
        )
        search_str = deduplicate_and_format_sources(
            search_results, max_tokens_per_source=2000, include_raw_content=True
        )
    elif search_api == "perplexity":
        search_results = perplexity_search(legal_query, state.websearch_loop_count)
        search_str = deduplicate_and_format_sources(
            search_results, max_tokens_per_source=2000, include_raw_content=False
        )
    elif search_api == "duckduckgo":
        search_results = duckduckgo_search(
            legal_query,
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
        "web_research_results": [search_str],
        "websearch_loop_count": state.websearch_loop_count + 1,
    }


def summarize_legal_web_sources(state: SummaryState, config: RunnableConfig):
    """Summarize legal sources with specialized legal context"""

    # Existing summary
    existing_summary = state.websearch_summary

    # Most recent web research
    web_research = state.web_research_results[-1] if state.web_research_results else ""

    # Get extracted legal entities if available
    legal_entities = getattr(state, "legal_entities", {})
    entities_context = (
        f"<Legal Entities>\n{json.dumps(legal_entities, indent=2)}\n</Legal Entities>"
        if legal_entities
        else ""
    )

    # Build the human message with legal context
    if existing_summary:
        human_message_content = (
            f"<Legal Query>\n{state.research_topic}\n</Legal Query>\n\n"
            f"{entities_context}\n\n"
            f"<Existing Summary>\n{existing_summary}\n</Existing Summary>\n\n"
            f"<New Research Results>\n{web_research}\n</New Research Results>"
        )
    else:
        human_message_content = (
            f"<Legal Query>\n{state.research_topic}\n</Legal Query>\n\n"
            f"{entities_context}\n\n"
            f"<Research Results>\n{web_research}\n</Research Results>"
        )

    # Run the LLM with legal-specific instructions
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOllama(
        base_url=configurable.ollama_base_url,
        model=configurable.local_llm,
        temperature=0.3,
    )
    result = llm.invoke(
        [
            SystemMessage(content=legal_summarizer_instructions),
            HumanMessage(content=human_message_content),
        ]
    )

    websearch_summary = result.content

    # Clean up any thinking tags
    while "<think>" in websearch_summary and "</think>" in websearch_summary:
        start = websearch_summary.find("<think>")
        end = websearch_summary.find("</think>") + len("</think>")
        websearch_summary = websearch_summary[:start] + websearch_summary[end:]

    return {"websearch_summary": websearch_summary}


def reflect_on_legal_research(state: SummaryState, config: RunnableConfig):
    """Reflect on the legal research and generate follow-up queries"""

    # Generate a follow-up query with legal focus
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = ChatOllama(
        base_url=configurable.ollama_base_url,
        model=configurable.local_llm,
        temperature=0.3,
        format="json",
    )

    result = llm_json_mode.invoke(
        [
            SystemMessage(
                content=legal_reflection_instructions.format(
                    research_topic=state.research_topic
                )
            ),
            HumanMessage(
                content=(
                    f"Identify gaps in our legal research and generate a follow-up query.\n"
                    f"WebResearch summary: {state.websearch_summary}\n"
                    f"Vector Summary: {state.vector_summary}\n"
                )
            ),
        ]
    )

    try:
        follow_up_data = json.loads(result.content)

        # Get the follow-up query
        query = follow_up_data.get("follow_up_query")

        if not query:
            # Fallback with legal focus
            return {"search_query": f"legal analysis of {state.research_topic}"}

        # Update search query with legal follow-up query
        return {"search_query": follow_up_data["follow_up_query"]}
    except json.JSONDecodeError:
        # Handle JSON parsing failures gracefully
        return {"search_query": f"legal precedents related to {state.research_topic}"}


def analyze_legal_entities(state: SummaryState, config: RunnableConfig):
    """Extract and analyze key legal entities from research results"""

    # Configure
    configurable = Configuration.from_runnable_config(config)
    # Extract the text from combined research results
    existing_summary = state.vector_summary

    combined_text = ""
    for doc in state.complete_research_results:
        if hasattr(doc, "page_content"):
            combined_text += doc.page_content + "\n\n"
        else:
            combined_text += str(doc) + "\n\n"

    # Add web research if available
    if state.web_research_results:
        combined_text += "\n\n".join(state.web_research_results)

    # Extract entities
    entities = extract_legal_entities(combined_text, state=state, config=config)

    return {"legal_entities": entities}


def finalize_legal_summary(state: SummaryState, config: RunnableConfig):
    """Generate a final legal analysis with recommendations"""

    # Analyze the accumulated research
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOllama(
        base_url=configurable.ollama_base_url,
        model=configurable.local_llm,
        temperature=0.1,
    )

    # Format all accumulated sources
    all_sources = "\n".join(source for source in state.sources_gathered)

    # Generate final legal analysis
    result = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a legal research assistant creating a final analysis. "
                    "Provide a comprehensive legal analysis with citations to relevant laws and cases. "
                    "Include applicable legal principles, precedents, and potential considerations. "
                    "Format your response with clear sections and professional legal language."
                )
            ),
            HumanMessage(
                content=(
                    f"Research Topic: {state.research_topic}\n\n"
                    f"Current Analysis: {state.running_summary}\n\n"
                    f"Create a final legal analysis with practical recommendations."
                )
            ),
        ]
    )

    final_analysis = result.content

    # Add sources to the final report
    complete_report = f"{final_analysis}\n\n## Sources\n{all_sources}"

    return {"running_summary": complete_report}


def route_research(
    state: SummaryState, config: RunnableConfig
) -> Literal["finalize_legal_summary", "web_research"]:
    """Route the research based on the current state and configuration"""

    configurable = Configuration.from_runnable_config(config)

    # Check if we've reached the maximum number of web research loops
    if state.websearch_loop_count >= int(configurable.max_web_research_loops):
        return "finalize_legal_summary"
    else:
        return "web_research"


def extract_legal_entities(
    text,
    state: SummaryState,
    config: RunnableConfig,
):
    """
    Extract key legal entities from text such as statutes, case names,
    jurisdictions, dates, and legal principles.

    Args:
        text (str): The input text from which to extract entities

    Returns:
        dict: Dictionary containing extracted entities by category
    """
    if not text or not str(text).strip():
        return {
            "statutes": [],
            "cases": [],
            "principles": [],
            "jurisdictions": [],
            "dates": [],
            "parties": [],
        }

    # Use ollama_llm to extract entities
    prompt = f"""Extract the key legal entities from the following text.
    Format your response as JSON with these keys:
    - statutes: List of mentioned statutes, acts, or regulations
    - cases: List of case names
    - principles: List of legal principles or doctrines
    - jurisdictions: List of jurisdictions mentioned
    - dates: List of relevant dates mentioned
    - parties: List of parties mentioned
    
    Text: {text[:8000]}  # Truncate to avoid token limits
    
    JSON Response:
    """
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOllama(
        base_url=configurable.ollama_base_url,
        model=configurable.local_llm,
        temperature=0,
    )

    try:
        result = llm.invoke(prompt)
        # Parse the result - finding JSON part if needed
        import re

        json_match = re.search(r"```json\n(.*?)\n```", result, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = result

        # Clean the JSON string
        json_str = re.sub(r"```.*?```", "", json_str, flags=re.DOTALL)

        # Try to parse as JSON
        try:
            entities = json.loads(json_str)
            # Ensure all expected keys exist
            for key in [
                "statutes",
                "cases",
                "principles",
                "jurisdictions",
                "dates",
                "parties",
            ]:
                if key not in entities:
                    entities[key] = []
            return entities
        except json.JSONDecodeError:
            # If JSON parsing fails, extract what we can
            print(f"Failed to parse JSON: {json_str}")
            return {
                "statutes": [],
                "cases": [],
                "principles": [],
                "jurisdictions": [],
                "dates": [],
                "parties": [],
            }
    except Exception as e:
        print(f"Error extracting legal entities: {str(e)}")
        return {
            "statutes": [],
            "cases": [],
            "principles": [],
            "jurisdictions": [],
            "dates": [],
            "parties": [],
        }


def chunk_and_summarize(
    state: SummaryState,
    config: RunnableConfig,
    text,
    chunk_size=8000,
    chunk_overlap=500,
):
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOllama(
        base_url=configurable.ollama_base_url,
        model=configurable.local_llm,
        temperature=0.1,
    )
    if not text or not str(text).strip():
        return "No content provided to summarize."

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(text)

    if not chunks:
        return "No content to summarize after splitting."

    if len(chunks) == 1 and len(text) < chunk_size:
        response = llm.invoke(f"Summarize the following legal content:\n{text}")
        summary = response.content if hasattr(response, "content") else str(response)
        return summary.strip()

    summaries = []
    for i, chunk in enumerate(chunks):
        try:
            response = llm.invoke(
                f"Summarize the following legal content (part {i+1}/{len(chunks)}):\n{chunk}"
            )
            summary = (
                response.content if hasattr(response, "content") else str(response)
            )
            summaries.append(summary.strip())
        except Exception as e:
            print(f"Error summarizing chunk {i+1}: {str(e)}")
            summaries.append(f"[Error summarizing this section: {str(e)}]")

    if len(summaries) > 1:
        combined_summary_text = "\n\n".join(summaries)
        try:
            response = llm.invoke(
                f"Create a cohesive summary of these {len(summaries)} section summaries:\n{combined_summary_text}"
            )
            final_summary = (
                response.content if hasattr(response, "content") else str(response)
            )
            return final_summary.strip()
        except Exception as e:
            print(f"Error creating final summary: {str(e)}")
            return (
                "Error creating final summary. Individual section summaries:\n\n"
                + combined_summary_text
            )
    elif summaries:
        return summaries[0]
    else:
        return "No summaries could be generated."


def summarize_vectors(state: SummaryState, config: RunnableConfig):
    """
    Summarize vector search results from the state.

    Args:
        state (SummaryState): The state containing research results

    Returns:
        dict: Dictionary with vector_summary key
    """
    combined = ""

    # Get document content from both laws and cases research results
    docs = []

    # Handle laws research results
    if hasattr(state, "laws_research_results") and state.laws_research_results:
        docs.extend(state.laws_research_results)

    # Handle cases research results
    if hasattr(state, "cases_research_results") and state.cases_research_results:
        docs.extend(state.cases_research_results)

    # Process all documents
    for doc in docs:
        if hasattr(doc, "page_content"):
            # If it's a Document object
            combined += str(doc.page_content) + "\n\n"
        elif isinstance(doc, dict) and "content" in doc:
            # If it's a dictionary with content
            combined += doc["content"] + "\n\n"
        else:
            # If it's a string or another format
            combined += str(doc) + "\n\n"

    # If we have content to summarize
    if combined.strip():
        summary = chunk_and_summarize(text=combined, config=config, state=state)
    else:
        summary = "No relevant legal documents found in vector stores."

    return {"vector_summary": summary}


def combine_summaries(state: SummaryState, config: RunnableConfig):
    """
    Combine web and vector summaries into a comprehensive summary.

    Args:
        state (SummaryState): The state containing summaries

    Returns:
        dict: Dictionary with combined_summary key
    """
    # Get web summary, with fallbacks
    web_summary = None
    for attr in ["vector_summary", "websearch_summary"]:
        if hasattr(state, attr) and getattr(state, attr):
            web_summary = getattr(state, attr)
            break

    if not web_summary:
        web_summary = "No web summary available."

    # Get vector summary, with fallback
    vector_summary = getattr(state, "vector_summary", "No legal summary available.")

    # Combine summaries with clear section headings
    full_text = f"""# Combined Legal Research Summary

## Web Research Summary
{web_summary}

## Legal Document Summary
{vector_summary}
"""

    return {"running_summary": full_text}


# Define the state graph
builder = StateGraph(
    SummaryState,
    input=SummaryStateInput,
    output=SummaryStateOutput,
    config_schema=Configuration,
)

# Nodes
builder.add_node("generate_query", generate_query)
builder.add_node("retrieve_from_vector_stores", retrieve_from_vector_stores)
builder.add_node("web_research", web_research)
builder.add_node("summarize_vectors", summarize_vectors)
builder.add_node("summarize_legal_sources", summarize_legal_web_sources)
builder.add_node("combine_summaries", combine_summaries)
builder.add_node("reflect_on_legal_research", reflect_on_legal_research)
builder.add_node("finalize_legal_summary", finalize_legal_summary)

# Edges - Legal-focused workflow
builder.add_edge(START, "generate_query")

# Initial query routes to vector store retrieval and web research
builder.add_edge("generate_query", "retrieve_from_vector_stores")
builder.add_edge("generate_query", "web_research")

# Vector store search flows
builder.add_edge("retrieve_from_vector_stores", "summarize_vectors")
builder.add_edge("summarize_vectors", "combine_summaries")

# Web research flows
builder.add_edge("web_research", "summarize_legal_sources")
builder.add_edge("summarize_legal_sources", "combine_summaries")

# Combined summaries route to reflection
builder.add_edge("combine_summaries", "reflect_on_legal_research")

# Conditional routing based on reflection
builder.add_conditional_edges("reflect_on_legal_research", route_research)

# Final node
builder.add_edge("finalize_legal_summary", END)

# Compile the graph
graph = builder.compile()
