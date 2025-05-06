# import streamlit as st
# import json
# import os
# import logging
# import traceback
# from agents.graph import graph, Configuration
# from agents.state import SummaryState, SummaryStateInput, SummaryStateOutput
# from dotenv import load_dotenv
# from dataclasses import fields
# from langchain_core.runnables import RunnableConfig

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[logging.FileHandler("vidhijan.log"), logging.StreamHandler()],
# )
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()

# # Streamlit app configuration
# st.set_page_config(
#     page_title="Vidhijan - Legal Research Assistant",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # Title and description
# st.title("Vidhijan")
# st.markdown(
#     """
# **Your AI-powered legal research assistant for India.**
# Enter a legal topic to generate a comprehensive analysis based on laws, cases, and web research. Configure settings in the sidebar for optimal results.
# """
# )

# # Sidebar for configuration
# with st.sidebar:
#     st.header("Research Settings")

#     ollama_base_url = st.text_input(
#         "Ollama Base URL",
#         value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
#         help="URL of the Ollama server (ensure it's running).",
#     )

#     local_llm = st.selectbox(
#         "Local LLM Model",
#         ["llama3.2:1b", "qwen2.5:0.5b", "gemma3:1b"],
#         index=2,
#         help="Select the LLM model for processing.",
#     )

#     search_api = st.selectbox(
#         "Search API",
#         ["tavily", "perplexity", "duckduckgo"],
#         index=2,
#         help="Choose the web search provider. Ensure API keys are set for Tavily or Perplexity.",
#     )

#     max_web_research_loops = st.slider(
#         "Max Web Research Loops", 1, 5, 3, help="Number of web research iterations."
#     )

#     fetch_full_page = st.checkbox(
#         "Fetch Full Page for DuckDuckGo",
#         value=False,
#         help="Fetch full page content for DuckDuckGo results (may increase processing time).",
#     )

#     # Validate API keys for Tavily and Perplexity
#     if search_api == "tavily" and not os.getenv("TAVILY_API_KEY"):
#         st.warning("TAVILY_API_KEY is not set. Please add it to your .env file.")
#     if search_api == "perplexity" and not os.getenv("PERPLEXITY_API_KEY"):
#         st.warning("PERPLEXITY_API_KEY is not set. Please add it to your .env file.")

#     # Validate FAISS paths
#     laws_faiss_path = os.getenv("LAWS_FAISS_PATH", "commercial_laws_index")
#     cases_faiss_path = os.getenv("CASES_FAISS_PATH", "cases_index")
#     if not os.path.exists(laws_faiss_path):
#         st.warning(
#             f"LAWS_FAISS_PATH ({laws_faiss_path}) does not exist. Vector store search may fail."
#         )
#     if not os.path.exists(cases_faiss_path):
#         st.warning(
#             f"CASES_FAISS_PATH ({cases_faiss_path}) does not exist. Vector store search may fail."
#         )

# # Create configuration object
# try:
#     config = Configuration(
#         ollama_base_url=ollama_base_url,
#         local_llm=local_llm,
#         search_api=search_api,
#         max_web_research_loops=max_web_research_loops,
#         fetch_full_page=fetch_full_page,
#         laws_faiss_path=laws_faiss_path,
#         cases_faiss_path=cases_faiss_path,
#     )
#     # Debug: Log the configuration
#     logger.info(f"Configuration created: {config.__dict__}")
# except Exception as e:
#     st.error(f"Error initializing configuration: {str(e)}")
#     logger.error(f"Configuration error: {str(e)}")
#     st.stop()

# # Input form
# with st.form("research_form"):
#     research_topic = st.text_input(
#         "Enter Legal Research Topic",
#         placeholder="e.g., Copyright infringement in digital media in India",
#         help="Specify a legal topic relevant to India.",
#     )
#     submitted = st.form_submit_button("Run Research")

# # Placeholder for results
# results_container = st.empty()
# progress_bar = st.progress(0)
# status_text = st.empty()


# # Progress simulation
# def update_progress(step, total_steps):
#     progress = min(step / total_steps, 1.0)
#     progress_bar.progress(progress)
#     status_text.text(f"Processing step {step}/{total_steps}...")


# # Validate state schema
# expected_state_fields = {f.name for f in fields(SummaryState)}
# required_state_fields = {
#     "research_topic",
#     "search_query",
#     "laws_research_results",
#     "cases_research_results",
#     "complete_research_results",
#     "web_research_results",
#     "sources_gathered",
#     "websearch_loop_count",
#     "vectorstore_loop_count",
#     "running_summary",
#     "vector_summary",
#     "websearch_summary",
# }
# if not required_state_fields.issubset(expected_state_fields):
#     missing_fields = required_state_fields - expected_state_fields
#     st.error(f"State schema mismatch: Missing fields in SummaryState: {missing_fields}")
#     logger.error(f"State schema mismatch: Missing fields: {missing_fields}")
#     st.stop()

# if submitted and research_topic:
#     try:
#         # Initialize state
#         initial_state = SummaryState(
#             research_topic=research_topic,
#             search_query="",
#             laws_research_results=[],
#             cases_research_results=[],
#             complete_research_results=[],
#             web_research_results=[],
#             sources_gathered=[],
#             running_summary="",
#             websearch_summary="",
#             vector_summary="",
#             websearch_loop_count=0,
#             vectorstore_loop_count=0,
#         )

#         # Convert Configuration to RunnableConfig
#         runnable_config = {"configurable": config.to_dict()}
#         logger.info(f"Runnable config: {runnable_config}")  # Debug

#         # Run the graph with progress simulation
#         with st.spinner("Running legal research..."):
#             total_steps = 8  # Number of nodes in graph
#             for step, node in enumerate(
#                 [
#                     "generate_query",
#                     "retrieve_from_vector_stores",
#                     "web_research",
#                     "summarize_vectors",
#                     "summarize_legal_sources",
#                     "combine_summaries",
#                     "reflect_on_legal_research",
#                     "finalize_legal_summary",
#                 ],
#                 1,
#             ):
#                 update_progress(step, total_steps)
#                 logger.info(f"Simulating progress for node: {node}")

#             logger.info(f"Starting graph execution for topic: {research_topic}")
#             result = graph.invoke(
#                 input=SummaryStateInput(research_topic=research_topic),
#                 config=runnable_config,
#             )
#             logger.info("Graph execution completed")

#         # Clear progress indicators
#         progress_bar.progress(1.0)
#         status_text.text("Research complete!")

#         # Display results
#         with results_container.container():
#             st.header("Research Results")

#             # Final Legal Analysis
#             st.subheader("Final Legal Analysis")
#             if result.get("running_summary"):
#                 st.markdown(result["running_summary"])
#             else:
#                 st.info("No final summary generated.")

#             # Expandable section for web research summary
#             if result.get("websearch_summary"):
#                 with st.expander("Web Research Summary", expanded=False):
#                     st.markdown(result["websearch_summary"])

#             # Expandable section for vector store summary
#             if result.get("vector_summary"):
#                 with st.expander("Legal Document Summary", expanded=False):
#                     st.markdown(result["vector_summary"])

#             # Expandable section for sources
#             if result.get("sources_gathered"):
#                 with st.expander("Sources Gathered", expanded=False):
#                     st.markdown("\n".join(result["sources_gathered"]))

#     except Exception as e:
#         st.error(f"An error occurred during research: {str(e)}")
#         st.markdown(
#             """
#         **Possible causes:**
#         - Ollama server not running at the specified URL.
#         - Missing or invalid API keys for Tavily or Perplexity.
#         - FAISS vector stores not found at specified paths.
#         - Network issues or rate limits.
#         - Incompatible configuration or node function.
#         Please check your configuration, logs (vidhijan.log), and try again.
#         """
#         )
#         status_text.text("Research failed.")
#         logger.error(f"Research error: {str(e)}\n{traceback.format_exc()}")

# # Footer
# st.markdown("---")
# st.markdown("**Vidhijan** | Built for legal research in India")

import streamlit as st
import logging
from agents.graph import graph, Configuration
from agents.state import SummaryStateInput

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler("vidhijan.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Vidhijan Test", layout="wide")
st.title("Vidhijan Test")

config = Configuration(
    ollama_base_url="http://localhost:11434",
    local_llm="gemma3:1b",
    search_api="duckduckgo",
    max_web_research_loops=1,
    fetch_full_page=False,
)

import requests


def check_ollama_server(url):
    try:
        response = requests.get(f"{url}/api/tags")
        return response.status_code == 200
    except requests.RequestException as e:
        logger.error(f"Ollama server check failed: {str(e)}")
        return False


if not check_ollama_server(config.ollama_base_url):
    st.error(
        f"Ollama server at {config.ollama_base_url} is not running. Start it and try again."
    )
    st.stop()

with st.form("research_form"):
    research_topic = st.text_input(
        "Enter Legal Research Topic", value="Indian contract law"
    )
    submitted = st.form_submit_button("Run Research")

if submitted and research_topic:
    try:
        runnable_config = {
            "configurable": config.to_dict(),
            "timeout": 60,
            "max_concurrency": 1,
        }
        logger.info(f"Running graph with topic: {research_topic}")
        result = graph.invoke(
            input=SummaryStateInput(research_topic=research_topic),
            config=runnable_config,
        )
        st.header("Results")
        st.markdown(result.get("running_summary", "No summary generated"))
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.error(f"Error: {str(e)}", exc_info=True)

st.markdown("---")
st.markdown("Vidhijan Test")
