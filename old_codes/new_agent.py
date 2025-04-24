import os
import logging
from typing import Annotated, Dict, Any, List
import requests
import PyPDF2
from io import BytesIO

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain.agents import AgentExecutor, Tool
from langchain.schema import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Config:
    GROQ_MODEL = "llama-3.1-8b-instant"
    OLLAMA_MODEL = "all-minilm:33m"
    LAWS_VECTOR_STORE_PATH = "/Users/sanatwalia/Desktop/Assignments_applications/Vidhijna/commercial_laws_index"
    CASES_VECTOR_STORE_PATH = (
        "/Users/sanatwalia/Desktop/Assignments_applications/Vidhijna/cases_index"
    )
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
    PDF_DOWNLOAD_PATH = "downloaded_cases"


class State(Dict[str, Any]):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    laws_output: str
    case_query: str
    cases_from_db: List[str]
    cases_from_search: List[str]
    downloaded_cases: List[str]
    final_summary: str


class LegalResearchChatbot:
    def __init__(self):
        self.console = Console()
        self.setup_components()
        self.setup_graph()

    def setup_components(self):
        self.llm = ChatGroq(model=Config.GROQ_MODEL)
        self.embeddings = OllamaEmbeddings(model=Config.OLLAMA_MODEL)

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

        self.search = SerpAPIWrapper(serpapi_api_key=Config.SERPAPI_API_KEY)
        self.text_splitter = RecursiveCharacterTextSplitter()

    def laws_retriever_agent(self, state: State) -> Dict:
        query = state["messages"][-1].content
        results = self.laws_vectorstore.similarity_search(query)
        laws_text = "\n".join([doc.page_content for doc in results])
        return {"laws_output": laws_text}

    def query_formation_agent(self, state: State) -> Dict:
        prompt = f"Based on these laws:\n{state['laws_output']}\n\nGenerate a search query for finding relevant Indian legal cases."
        response = self.llm.predict(prompt)
        return {"case_query": response}

    def cases_db_agent(self, state: State) -> Dict:
        results = self.cases_vectorstore.similarity_search(state["case_query"], k=5)
        cases = [doc.page_content for doc in results]
        return {"cases_from_db": cases}

    def search_agent(self, state: State) -> Dict:
        if len(state["cases_from_db"]) >= 5:
            return {"cases_from_search": []}

        search_query = f"site:indiankanoon.org {state['case_query']}"
        results = self.search.run(search_query)
        return {"cases_from_search": results[:10]}

    def pdf_download_agent(self, state: State) -> Dict:
        if not state["cases_from_search"]:
            return {"downloaded_cases": []}

        downloaded_texts = []
        for case_url in state["cases_from_search"]:
            try:
                response = requests.get(case_url)
                pdf = PyPDF2.PdfReader(BytesIO(response.content))
                text = "".join([page.extract_text() for page in pdf.pages])
                downloaded_texts.append(text)

                texts = self.text_splitter.split_text(text)
                self.cases_vectorstore.add_texts(texts)

            except Exception as e:
                logging.error(f"Error downloading case: {str(e)}")
                continue

        self.cases_vectorstore.save_local(Config.CASES_VECTOR_STORE_PATH)
        return {"downloaded_cases": downloaded_texts}

    def summary_agent(self, state: State) -> Dict:
        all_cases = state["cases_from_db"] + state["downloaded_cases"]
        joined_cases = (
            "\n\n".join(all_cases[:3]) if all_cases else "No relevant cases found"
        )

        prompt = f"""
        Given the following legal information, generate a detailed summary and analysis:

        ## Relevant Laws:
        {state['laws_output']}

        ## Relevant Cases:
        {joined_cases}

        Provide a consolidated and well-structured analysis based on the above content.
        """

        summary = self.llm.predict(prompt)
        return {"final_summary": summary}

    def setup_graph(self):
        workflow = StateGraph(State)
        workflow.add_node("laws_retriever", self.laws_retriever_agent)
        workflow.add_node("query_formation", self.query_formation_agent)
        workflow.add_node("cases_db", self.cases_db_agent)
        workflow.add_node("search", self.search_agent)
        workflow.add_node("pdf_download", self.pdf_download_agent)
        workflow.add_node("summary", self.summary_agent)

        workflow.add_edge("laws_retriever", "query_formation")
        workflow.add_edge("query_formation", "cases_db")
        workflow.add_edge("cases_db", "search")
        workflow.add_edge("search", "pdf_download")
        workflow.add_edge("pdf_download", "summary")
        workflow.add_edge("summary", END)

        workflow.set_entry_point("laws_retriever")
        self.graph = workflow.compile()

    def process_query(self, query: str) -> Dict[str, Any]:
        state = {
            "messages": [HumanMessage(content=query)],
            "laws_output": "",
            "case_query": "",
            "cases_from_db": [],
            "cases_from_search": [],
            "downloaded_cases": [],
            "final_summary": "",
        }
        result = self.graph.invoke(state)
        return {
            "laws": result.get("laws_output", ""),
            "cases": result.get("cases_from_db", [])
            + result.get("downloaded_cases", []),
            "summary": result.get("final_summary", ""),
        }

    def run_interactive(self):
        self.console.print(
            Markdown("# Legal Research Chatbot\nType 'quit', 'exit', or 'q' to end.")
        )

        while True:
            user_input = input("\nUser Query: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                self.console.print(Markdown("**Goodbye!**"))
                break

            results = self.process_query(user_input)
            self.console.print(Markdown(f"## Retrieved Laws\n{results['laws']}"))
            self.console.print(Markdown(f"## Relevant Cases\n{results['cases']}"))
            self.console.print(Markdown(f"## Final Summary\n{results['summary']}"))


def main():
    chatbot = LegalResearchChatbot()
    chatbot.run_interactive()


if __name__ == "__main__":
    main()
