import os
import logging
from typing import Annotated, Dict, Any, TypedDict, List

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Configuration
class Config:
    GROQ_MODEL = "llama-3.1-8b-instant"
    OLLAMA_MODEL = "all-minilm:33m"
    VECTOR_STORE_PATH = "commercial_laws_index"
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


# Define the state structure
class State(TypedDict):
    messages: Annotated[List[Dict[str, str]], add_messages]
    retriever_output: str
    search_output: str


class LangGraphChatbot:
    def __init__(self):
        self.setup_models()
        self.setup_vector_store()
        self.setup_search_tool()
        self.setup_graph()

    def setup_models(self):
        """Initialize LLM and embedding models"""
        try:
            self.llm = ChatGroq(model=Config.GROQ_MODEL)
            self.embeddings = OllamaEmbeddings(model=Config.OLLAMA_MODEL)
            logging.info("LLM and embeddings initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize models: {str(e)}")
            raise

    def setup_vector_store(self):
        """Initialize FAISS vector store"""
        try:
            self.vectorstore_db = FAISS.load_local(
                Config.VECTOR_STORE_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            self.retriever = self.vectorstore_db.as_retriever()
            logging.info("Vector store initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to load vector store: {str(e)}")
            raise

    def setup_search_tool(self):
        """Initialize SerpAPI wrapper"""
        if not Config.SERPAPI_API_KEY:
            raise ValueError("SERPAPI_API_KEY not found in environment variables")
        try:
            self.search = SerpAPIWrapper(serpapi_api_key=Config.SERPAPI_API_KEY)
            logging.info("SerpAPI search tool initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize SerpAPI: {str(e)}")
            raise

    def retriever_agent(self, state: State) -> Dict:
        """Retrieve documents from FAISS vector store"""
        try:
            query = state["messages"][-1]["content"]
            results = self.retriever.get_relevant_documents(query)
            output = "\n".join([doc.page_content for doc in results])
            return {"retriever_output": output}
        except Exception as e:
            logging.error(f"Retriever agent error: {str(e)}")
            return {"retriever_output": ""}

    def search_agent(self, state: State) -> Dict:
        """Perform a web search using SerpAPI"""
        try:
            query = state["messages"][-1]["content"]
            result = self.search.run(query)
            return {"search_output": result}
        except Exception as e:
            logging.error(f"Search agent error: {str(e)}")
            return {"search_output": ""}

    def final_processor(self, state: State) -> Dict:
        """Process results and generate response"""
        response = ""
        if state.get("retriever_output"):
            response += f"Based on our legal database: {state['retriever_output']}\n\n"
        if state.get("search_output"):
            response += (
                f"Additional information from web search: {state['search_output']}"
            )

        if not response:
            response = (
                "I couldn't find relevant information. Please consult a legal expert."
            )

        return {"messages": [{"role": "assistant", "content": response}]}

    def setup_graph(self):
        """Define the retrieval and search workflow using LangGraph"""
        workflow = StateGraph(State)
        workflow.add_node("retriever", self.retriever_agent)
        workflow.add_node("search", self.search_agent)
        workflow.add_node("final", self.final_processor)

        workflow.add_edge("retriever", "search")
        workflow.add_edge("search", "final")
        workflow.add_edge("final", END)

        workflow.set_entry_point("retriever")
        self.graph = workflow.compile()
        logging.info("Workflow graph initialized successfully.")

    def process_query(self, query: str) -> str:
        """Process user query through the workflow graph"""
        try:
            state = {
                "messages": [{"role": "user", "content": query}],
                "retriever_output": "",
                "search_output": "",
            }
            result = self.graph.invoke(state)
            return (
                result["messages"][-1]["content"]
                if "messages" in result
                else "Processing error."
            )
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return "An error occurred. Please try again."

    def run_interactive(self):
        """Run chatbot in interactive mode"""
        print("Interactive mode. Type 'quit' to exit.")
        while True:
            try:
                user_input = input("\nUser: ").strip()
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                print("\nAssistant:", self.process_query(user_input))
            except KeyboardInterrupt:
                print("\nChat terminated.")
                break
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    try:
        chatbot = LangGraphChatbot()
        chatbot.run_interactive()
    except Exception as e:
        logging.error(f"Failed to start chatbot: {str(e)}")
