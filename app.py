# legal_research_app.py

import os
import logging
from typing import Annotated, Dict, Any, List
import streamlit as st
import requests
import PyPDF2
from io import BytesIO
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Load environment variables and setup logging
load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Config:
    GROQ_MODEL = "llama-3.1-8b-instant"
    OLLAMA_MODEL = "all-minilm:33m"
    LAWS_VECTOR_STORE_PATH = "laws_index"
    CASES_VECTOR_STORE_PATH = "cases_index"
    CHROME_DRIVER_PATH = "path/to/chromedriver"  # Update with your path


class CaseCrawler:
    """Custom crawler for Indian legal cases"""

    def __init__(self):
        self.options = webdriver.ChromeOptions()
        self.options.add_argument("--headless")
        self.driver = webdriver.Chrome(options=self.options)

    def search_cases(self, query: str, num_results: int = 10) -> List[Dict]:
        """
        Search for legal cases using custom crawler
        Returns: List of dicts with case details
        """
        try:
            # Example using Indian Kanoon
            base_url = f"https://indiankanoon.org/search/?formInput={query}"
            self.driver.get(base_url)

            # Wait for results to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "result_title"))
            )

            cases = []
            results = self.driver.find_elements(By.CLASS_NAME, "result")[:num_results]

            for result in results:
                title = result.find_element(By.CLASS_NAME, "result_title").text
                snippet = result.find_element(By.CLASS_NAME, "snippet").text
                link = result.find_element(By.TAG_NAME, "a").get_attribute("href")

                cases.append({"title": title, "snippet": snippet, "url": link})

            return cases

        except Exception as e:
            logging.error(f"Error in case search: {str(e)}")
            return []

    def get_case_content(self, url: str) -> str:
        """Extract full case content from URL"""
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "doc"))
            )
            content = self.driver.find_element(By.ID, "doc").text
            return content
        except Exception as e:
            logging.error(f"Error extracting case content: {str(e)}")
            return ""

    def __del__(self):
        self.driver.quit()


class LegalResearchBot:
    def __init__(self):
        self.setup_components()
        self.setup_graph()

    def setup_components(self):
        self.llm = ChatGroq(model=Config.GROQ_MODEL)
        self.embeddings = OllamaEmbeddings(model=Config.OLLAMA_MODEL)
        self.crawler = CaseCrawler()

        # Setup vector stores
        self.laws_vectorstore = FAISS.load_local(
            Config.LAWS_VECTOR_STORE_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        if os.path.exists(Config.CASES_VECTOR_STORE_PATH):
            self.cases_vectorstore = FAISS.load_local(
                Config.CASES_VECTOR_STORE_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            self.cases_vectorstore = FAISS.from_texts(
                ["Initial empty store"], self.embeddings
            )

        self.text_splitter = RecursiveCharacterTextSplitter()

    def laws_retriever_agent(self, state: Dict) -> Dict:
        query = state["messages"][-1].content
        results = self.laws_vectorstore.similarity_search(query, k=5)
        laws_text = "\n".join([doc.page_content for doc in results])
        return {"laws_output": laws_text}

    def query_formation_agent(self, state: Dict) -> Dict:
        prompt = f"""
        Based on these laws:
        {state['laws_output']}
        
        Generate a specific search query for finding relevant Indian legal cases.
        Focus on key legal principles and precedents.
        """
        response = self.llm.predict(prompt)
        return {"case_query": response}

    def case_search_agent(self, state: Dict) -> Dict:
        cases = self.crawler.search_cases(state["case_query"])
        return {"found_cases": cases}

    def case_processing_agent(self, state: Dict) -> Dict:
        processed_cases = []
        for case in state["found_cases"]:
            content = self.crawler.get_case_content(case["url"])
            if content:
                # Update vector store
                texts = self.text_splitter.split_text(content)
                self.cases_vectorstore.add_texts(texts)
                processed_cases.append({"title": case["title"], "content": content})

        self.cases_vectorstore.save_local(Config.CASES_VECTOR_STORE_PATH)
        return {"processed_cases": processed_cases}

    def summary_agent(self, state: Dict) -> Dict:
        cases_text = "\n\n".join(
            [
                f"Case: {case['title']}\n{case['content'][:1000]}..."
                for case in state["processed_cases"]
            ]
        )

        prompt = f"""
        Provide a comprehensive legal analysis:
        
        Laws:
        {state['laws_output']}
        
        Relevant Cases:
        {cases_text}
        
        Please include:
        1. Summary of applicable laws
        2. Key principles from cases
        3. How these cases interpret/apply the laws
        4. Important precedents established
        """

        analysis = self.llm.predict(prompt)
        return {"final_summary": analysis}

    def setup_graph(self):
        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node("laws_retriever", self.laws_retriever_agent)
        workflow.add_node("query_formation", self.query_formation_agent)
        workflow.add_node("case_search", self.case_search_agent)
        workflow.add_node("case_processing", self.case_processing_agent)
        workflow.add_node("summary", self.summary_agent)

        # Define edges
        workflow.add_edge("laws_retriever", "query_formation")
        workflow.add_edge("query_formation", "case_search")
        workflow.add_edge("case_search", "case_processing")
        workflow.add_edge("case_processing", "summary")
        workflow.add_edge("summary", END)

        workflow.set_entry_point("laws_retriever")

        self.graph = workflow.compile()

    def process_query(self, query: str) -> Dict[str, Any]:
        try:
            state = {
                "messages": [HumanMessage(content=query)],
                "laws_output": "",
                "case_query": "",
                "found_cases": [],
                "processed_cases": [],
                "final_summary": "",
            }

            result = self.graph.invoke(state)
            return {
                "laws": result.get("laws_output", ""),
                "cases": result.get("processed_cases", []),
                "summary": result.get("final_summary", ""),
            }

        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return {"laws": "", "cases": [], "summary": f"Error occurred: {str(e)}"}


# Streamlit UI
def main():
    st.set_page_config(page_title="Legal Research Assistant", layout="wide")

    st.title("Legal Research Assistant")
    st.write("Enter your legal query to search relevant laws and cases")

    # Initialize bot in session state
    if "legal_bot" not in st.session_state:
        st.session_state.legal_bot = LegalResearchBot()

    # Query input
    query = st.text_area("Enter your legal query:", height=100)

    if st.button("Research"):
        with st.spinner("Researching..."):
            results = st.session_state.legal_bot.process_query(query)

            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(
                ["Relevant Laws", "Case Precedents", "Legal Analysis"]
            )

            with tab1:
                st.header("Applicable Laws")
                st.write(results["laws"])

            with tab2:
                st.header("Relevant Cases")
                for case in results["cases"]:
                    with st.expander(case["title"]):
                        st.write(case["content"])

            with tab3:
                st.header("Legal Analysis")
                st.write(results["summary"])

    # Sidebar with additional information
    with st.sidebar:
        st.header("About")
        st.write(
            """
        This tool helps legal professionals research:
        - Relevant laws and regulations
        - Case precedents
        - Legal interpretations
        
        Note: This is an assistant tool and should not be considered as legal advice.
        """
        )

        st.header("Usage Tips")
        st.write(
            """
        - Be specific in your queries
        - Include relevant legal contexts
        - Mention specific areas of law
        - Include jurisdictional information
        """
        )


if __name__ == "__main__":
    main()
