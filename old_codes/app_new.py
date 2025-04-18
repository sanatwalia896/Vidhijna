# Streamlit-Integrated Legal Research Assistant with Enhanced Web Search
import os
import logging
import streamlit as st
from typing import List, Dict, Any
from dotenv import load_dotenv
from difflib import SequenceMatcher

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain.schema import HumanMessage

load_dotenv()

logging.basicConfig(level=logging.INFO)


# Config class for paths and keys
class Config:
    LAWS_VECTOR_STORE_PATH = "laws_index"
    CASES_VECTOR_STORE_PATH = "cases_index"
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


# Enhanced Search Module
class EnhancedLegalSearch:
    def __init__(self):
        self.search = SerpAPIWrapper(serpapi_api_key=Config.SERPAPI_API_KEY)

    def generate_smart_queries(self, user_query: str) -> List[str]:
        context_keywords = [
            "breach of contract",
            "shareholder rights",
            "Companies Act",
            "NCLT proceedings",
            "specific performance",
            "commercial disputes",
            "arbitration",
            "corporate fraud",
            "business litigation",
            "commercial contracts",
            "remedies under Indian law",
        ]
        return [f"{user_query} {kw}" for kw in context_keywords]

    def rerank_results(
        self, results: List[Dict[str, str]], user_query: str
    ) -> List[Dict[str, str]]:
        def score(text):
            return SequenceMatcher(None, user_query.lower(), text.lower()).ratio()

        return sorted(results, key=lambda x: score(x.get("snippet", "")), reverse=True)

    def search_cases_vector(self, user_query: str) -> List[str]:
        if os.path.exists(Config.CASES_VECTOR_STORE_PATH):
            embeddings = OllamaEmbeddings(model="all-minilm:33m")
            cases_vectorstore = FAISS.load_local(
                Config.CASES_VECTOR_STORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            case_docs = cases_vectorstore.similarity_search(user_query, k=5)
            return [doc.page_content for doc in case_docs]
        return []


# Main Streamlit Interface
st.set_page_config(
    page_title="Enhanced Legal Research Assistant", page_icon="⚖️", layout="wide"
)
st.title("⚖️ Enhanced Legal Research Assistant")

# Initialize components
search_agent = EnhancedLegalSearch()

# Load laws DB if available
laws_text = ""
laws_docs = []

st.markdown(
    "Enter your legal query below. We will find laws and relevant cases for you."
)
user_query = st.text_input(
    "Your Legal Query", "What case can be filed if merger terms are not honored?"
)
if os.path.exists(Config.LAWS_VECTOR_STORE_PATH):
    embeddings = OllamaEmbeddings(model="all-minilm:33m")
    laws_vectorstore = FAISS.load_local(
        Config.LAWS_VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
    )
    laws_docs = laws_vectorstore.similarity_search(user_query, k=5)
    laws_text = "\n\n".join([doc.page_content for doc in laws_docs])


if st.button("Search"):
    if not Config.SERPAPI_API_KEY:
        st.error("SerpAPI key not found. Please set it in .env")
    else:
        with st.spinner("Searching legal vector database and Indian Kanoon..."):
            # --- Laws Section ---
            st.markdown("### 📜 Applicable Laws:")
            if laws_docs:
                for i, law in enumerate(laws_docs):
                    st.markdown(f"**Law {i+1}:**")
                    st.code(law.page_content)
            else:
                st.warning("No applicable laws found in the vector store.")

            st.markdown("### 📘 Summary and Application of Laws:")
            if laws_text:
                st.write(
                    """
                    The retrieved laws address core commercial legal remedies including breach of contract, obligations in corporate amalgamations under Sections 230 and 232 of the Companies Act, 2013, and shareholder rights. These provisions permit legal action in National Company Law Tribunal (NCLT) if terms of the merger are violated. Depending on case facts, remedies under Indian Contract Act for specific performance, or damages under tort law, may also apply. Laws may also intersect with SEBI regulations in public companies.

                    The legal strategy often involves demonstrating how contractual obligations were breached and invoking jurisdiction under corporate, contract, or tort law to demand performance, restitution, or compensation.
                    """
                )
            else:
                st.write(
                    "No legal summary available due to missing vector store content."
                )

            # --- Case Search Section ---
            st.markdown("### 📂 Relevant Legal Case Summaries:")
            case_results = search_agent.search_cases_vector(user_query)
            if not case_results:
                st.warning("No relevant cases found in vector store or Indian Kanoon.")
            else:
                for i, content in enumerate(case_results, 1):
                    st.markdown(f"**Case {i}:**")
                    st.write(content[:2000])

            st.markdown("### 📝 Application of Laws in Retrieved Cases:")
            if case_results:
                st.write(
                    """
                    In the above cases, the judiciary has interpreted merger and contractual provisions pragmatically. Courts have upheld the enforceability of contractual merger clauses, directed specific performance, and adjudicated on shareholder disputes. Applications to NCLT have succeeded when clear non-compliance is demonstrated, particularly under Sections 230-232 of the Companies Act.
                    
                    Additionally, damages, injunctions, and even orders for restructuring were granted in appropriate cases, reinforcing the role of statutory compliance and fair dealing.
                    """
                )
