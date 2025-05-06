import streamlit as st
import logging
import os
import traceback
from agents.graph import graph, Configuration
from agents.state import SummaryStateInput
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("vidhijan.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Streamlit app configuration
st.set_page_config(
    page_title="Vidhijan - Legal Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("⚖️ Vidhijan")
st.markdown(
    """
**Your AI-powered legal research assistant for India.**  
Enter a legal topic to generate a comprehensive analysis based on laws, cases, and web research. Configure settings in the sidebar for optimal results.  
*Recommended*: Use `gemma3:1b`, `duckduckgo`, and 1 web research loop for best performance.
"""
)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Research Settings")

    ollama_base_url = st.text_input(
        "Ollama Base URL",
        value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="URL of the Ollama server (e.g., http://localhost:11434). Ensure it's running.",
    )

    local_llm = st.selectbox(
        "Local LLM Model",
        ["llama3.2:1b", "qwen2.5:0.5b", "gemma3:1b"],
        index=2,
        help="Select the LLM model. `gemma3:1b` is recommended for stability.",
    )

    search_api = st.selectbox(
        "Search API",
        ["tavily", "perplexity", "duckduckgo"],
        index=2,
        help="Choose the web search provider. `duckduckgo` is recommended. Ensure API keys for Tavily or Perplexity.",
    )

    max_web_research_loops = st.slider(
        "Max Web Research Loops",
        1,
        5,
        1,  # Default to 1 for stability
        help="Number of web research iterations. 1 is recommended to avoid server issues.",
    )

    fetch_full_page = st.checkbox(
        "Fetch Full Page for DuckDuckGo",
        value=False,
        help="Fetch full page content for DuckDuckGo results (may increase processing time).",
    )

    # Warn about API keys
    if search_api == "tavily" and not os.getenv("TAVILY_API_KEY"):
        st.warning("⚠️ TAVILY_API_KEY is not set. Add it to your .env file.")
    if search_api == "perplexity" and not os.getenv("PERPLEXITY_API_KEY"):
        st.warning("⚠️ PERPLEXITY_API_KEY is not set. Add it to your .env file.")

    # Warn about FAISS paths
    laws_faiss_path = os.getenv("LAWS_FAISS_PATH", "commercial_laws_index")
    cases_faiss_path = os.getenv("CASES_FAISS_PATH", "cases_index")
    if not os.path.exists(laws_faiss_path):
        st.warning(f"⚠️ LAWS_FAISS_PATH ({laws_faiss_path}) does not exist.")
    if not os.path.exists(cases_faiss_path):
        st.warning(f"⚠️ CASES_FAISS_PATH ({cases_faiss_path}) does not exist.")

# Create configuration object
try:
    config = Configuration(
        ollama_base_url=ollama_base_url,
        local_llm=local_llm,
        search_api=search_api,
        max_web_research_loops=max_web_research_loops,
        fetch_full_page=fetch_full_page,
        laws_faiss_path=laws_faiss_path,
        cases_faiss_path=cases_faiss_path,
    )
    logger.info(f"Configuration created: {config.__dict__}")
except Exception as e:
    st.error(f"❌ Error initializing configuration: {str(e)}")
    logger.error(f"Configuration error: {str(e)}")
    st.stop()

# Input form
with st.form("research_form"):
    research_topic = st.text_input(
        "📝 Enter Legal Research Topic",
        placeholder="e.g., Copyright infringement in digital media in India",
        help="Specify a legal topic relevant to India.",
    )
    submitted = st.form_submit_button("🚀 Run Research")

# Placeholder for results
results_container = st.empty()
progress_bar = st.progress(0)
status_text = st.empty()


# Progress simulation
def update_progress(step, total_steps):
    progress = min(step / total_steps, 1.0)
    progress_bar.progress(progress)
    status_text.text(f"Processing step {step}/{total_steps}...")


if submitted and research_topic:
    try:
        # Run the graph with progress simulation
        with st.spinner("🔍 Running legal research..."):
            total_steps = 8  # Number of nodes in graph
            for step, node in enumerate(
                [
                    "generate_query",
                    "retrieve_from_vector_stores",
                    "web_research",
                    "summarize_vectors",
                    "summarize_legal_sources",
                    "combine_summaries",
                    "reflect_on_legal_research",
                    "finalize_legal_summary",
                ],
                1,
            ):
                update_progress(step, total_steps)
                logger.info(f"Simulating progress for node: {node}")

            logger.info(f"Running graph with topic: {research_topic}")
            runnable_config = {
                "configurable": config.to_dict(),
                "timeout": 60,  # Prevent hangs
                "max_concurrency": 1,  # Sequential execution
            }
            result = graph.invoke(
                input=SummaryStateInput(research_topic=research_topic),
                config=runnable_config,
            )
            logger.info(f"Graph execution completed: {result}")

        # Clear progress indicators
        progress_bar.progress(1.0)
        status_text.text("✅ Research complete!")

        # Display results
        with results_container.container():
            st.header("📊 Research Results")

            # Final Legal Analysis
            st.subheader("Final Legal Analysis")
            if result.get("running_summary"):
                st.markdown(result["running_summary"])
            else:
                st.info("No final summary generated. Check logs for details.")

            # Expandable section for web research summary
            if result.get("websearch_summary"):
                with st.expander("🌐 Web Research Summary", expanded=False):
                    st.markdown(result["websearch_summary"])

            # Expandable section for vector store summary
            if result.get("vector_summary"):
                with st.expander("📚 Legal Document Summary", expanded=False):
                    st.markdown(result["vector_summary"])

            # Expandable section for sources
            if result.get("sources_gathered"):
                with st.expander("🔗 Sources Gathered", expanded=False):
                    st.markdown("\n".join(result["sources_gathered"]))

    except Exception as e:
        st.error(f"❌ An error occurred during research: {str(e)}")
        st.markdown(
            """
            **Possible causes:**
            - Ollama server not running at the specified URL.
            - Missing or invalid API keys for Tavily or Perplexity.
            - FAISS vector stores not found at specified paths.
            - Large inputs or server resource issues.
            Please check your configuration, logs (vidhijan.log), and try again.
            """
        )
        status_text.text("❌ Research failed.")
        logger.error(f"Research error: {str(e)}\n{traceback.format_exc()}")

# Footer
st.markdown("---")
st.markdown("**Vidhijan**  | Built for legal research in India 🇮🇳")
