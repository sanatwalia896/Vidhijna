import streamlit as st
import requests
import time
import json
from enum import Enum

# Define API URL
API_URL = "http://localhost:8000/api/research"


class SearchAPI(str, Enum):
    TAVILY = "tavily"
    PERPLEXITY = "perplexity"
    DUCKDUCKGO = "duckduckgo"


# Set page config
st.set_page_config(page_title="Legal Research Assistant", page_icon="⚖️", layout="wide")

# App title
st.title("⚖️ Legal Research Assistant")
st.markdown(
    """
This application helps you conduct legal research by using vector stores and web search.
Enter your legal question or topic below to get started.
"""
)

# Sidebar for configuration
st.sidebar.title("Configuration")

ollama_base_url = st.sidebar.text_input(
    "Ollama Base URL", value="http://localhost:11434"
)
local_llm = st.sidebar.text_input("Local LLM Model", value="llama3")
search_api = st.sidebar.selectbox(
    "Search API",
    options=[api.value for api in SearchAPI],
    index=2,  # Default to DuckDuckGo
)
max_web_research_loops = st.sidebar.slider(
    "Max Web Research Loops", min_value=1, max_value=5, value=3
)
fetch_full_page = st.sidebar.checkbox("Fetch Full Pages", value=True)


# Function to start research
def start_research(research_topic):
    config = {
        "ollama_base_url": ollama_base_url,
        "local_llm": local_llm,
        "search_api": search_api,
        "max_web_research_loops": max_web_research_loops,
        "fetch_full_page": fetch_full_page,
    }

    payload = {"research_topic": research_topic, "configuration": config}

    response = requests.post(f"{API_URL}/run_full", json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error starting research: {response.text}")
        return None


# Function to check status
def check_status(task_id):
    response = requests.get(f"{API_URL}/status/{task_id}")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error checking status: {response.text}")
        return None


# Function to get results
def get_results(task_id):
    response = requests.get(f"{API_URL}/result/{task_id}")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error getting results: {response.text}")
        return None


# Main form
with st.form("research_form"):
    research_topic = st.text_area(
        "Enter your legal research topic or question", height=100
    )
    submit_button = st.form_submit_button("Start Research")

    if submit_button and research_topic:
        with st.spinner("Starting research..."):
            result = start_research(research_topic)

            if result:
                task_id = result.get("task_id")

                # Save task_id to session state
                st.session_state["task_id"] = task_id
                st.session_state["research_topic"] = research_topic
                st.success(f"Research started with task ID: {task_id}")

# Check if there's a task in progress
if "task_id" in st.session_state:
    task_id = st.session_state["task_id"]
    research_topic = st.session_state.get("research_topic", "Unknown topic")

    st.subheader(f"Research on: {research_topic}")

    # Status checking
    status_col, refresh_col = st.columns([5, 1])

    with status_col:
        status = check_status(task_id)
        if status:
            st.info(f"Status: {status.get('status')}")

    with refresh_col:
        if st.button("Refresh"):
            st.experimental_rerun()

    # Show results if completed
    if status and status.get("status") == "completed":
        results = get_results(task_id)

        if results:
            # Display final summary
            st.subheader("Final Legal Analysis")
            st.markdown(results.get("final_summary", "No summary available"))

            # Display legal entities if available
            if "legal_entities" in results and results["legal_entities"]:
                st.subheader("Key Legal Entities")

                entities = results["legal_entities"]

                col1, col2 = st.columns(2)

                with col1:
                    if "statutes" in entities and entities["statutes"]:
                        st.markdown("### Statutes")
                        for statute in entities["statutes"]:
                            st.markdown(f"- {statute}")

                    if "cases" in entities and entities["cases"]:
                        st.markdown("### Cases")
                        for case in entities["cases"]:
                            st.markdown(f"- {case}")

                    if "principles" in entities and entities["principles"]:
                        st.markdown("### Legal Principles")
                        for principle in entities["principles"]:
                            st.markdown(f"- {principle}")

                with col2:
                    if "jurisdictions" in entities and entities["jurisdictions"]:
                        st.markdown("### Jurisdictions")
                        for jurisdiction in entities["jurisdictions"]:
                            st.markdown(f"- {jurisdiction}")

                    if "dates" in entities and entities["dates"]:
                        st.markdown("### Key Dates")
                        for date in entities["dates"]:
                            st.markdown(f"- {date}")

                    if "parties" in entities and entities["parties"]:
                        st.markdown("### Parties")
                        for party in entities["parties"]:
                            st.markdown(f"- {party}")

            # Display sources
            if "sources" in results and results["sources"]:
                st.subheader("Sources")
                with st.expander("View Sources"):
                    for source in results["sources"]:
                        st.markdown(source)
                        st.markdown("---")

            # Button to start new research
            if st.button("Start New Research"):
                # Clear session state
                del st.session_state["task_id"]
                if "research_topic" in st.session_state:
                    del st.session_state["research_topic"]
                st.experimental_rerun()

    # Show progress and intermediate results if not completed
    elif status and status.get("status") == "running":
        st.markdown("### Research in Progress")

        # Create progress bar
        progress_bar = st.progress(0)

        # Try to display intermediate results if available
        try:
            # We can try to fetch intermediate results using individual node endpoints
            # This is just a placeholder - in a real app, you'd track progress more precisely
            progress_bar.progress(50)
            st.info(
                "Gathering and analyzing legal information. This may take a few minutes..."
            )
        except Exception as e:
            st.warning(f"Couldn't fetch intermediate results: {e}")

# Footer
st.markdown("---")
st.markdown("© 2025 Legal Research Assistant | Powered by FastAPI & Streamlit")
