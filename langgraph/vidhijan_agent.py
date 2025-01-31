# Standard library imports
import os
import logging
from typing import Annotated, Dict, Any, List

# Third-party imports
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain.agents import AgentExecutor, Tool
from langchain.schema import AIMessage, HumanMessage  # FIXED: Proper message handling
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Load environment variables
load_dotenv()

# Setup logging
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
class State(Dict[str, Any]):
    """State definition for the workflow"""

    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    retriever_output: str
    search_output: str


class LangGraphChatbot:
    def __init__(self):
        """Initialize the chatbot with necessary components"""
        self.setup_llm()
        self.setup_embeddings()
        self.setup_vector_store()
        self.setup_serpapi()
        self.setup_graph()

    def setup_llm(self):
        """Initialize the language model"""
        try:
            self.llm = ChatGroq(model=Config.GROQ_MODEL)
            logging.info("LLM initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize LLM: {str(e)}")
            raise

    def setup_embeddings(self):
        """Initialize embeddings model"""
        try:
            self.embeddings = OllamaEmbeddings(model=Config.OLLAMA_MODEL)
            logging.info("Embeddings model initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize embeddings: {str(e)}")
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

    def setup_serpapi(self):
        """Initialize SerpAPI wrapper"""
        if not Config.SERPAPI_API_KEY:
            raise ValueError("SERPAPI_API_KEY not found in environment variables")
        try:
            self.search = SerpAPIWrapper(serpapi_api_key=Config.SERPAPI_API_KEY)
            logging.info("SerpAPI initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize SerpAPI: {str(e)}")
            raise

    def retriever_agent(self, state: State) -> Dict:
        """Retrieve documents from FAISS vector store"""
        try:
            query = state["messages"][-1].content  # FIXED: Extract content correctly
            results = self.retriever.get_relevant_documents(query)
            output = "\n".join([doc.page_content for doc in results])
            return {"retriever_output": output}
        except Exception as e:
            logging.error(f"Retriever agent error: {str(e)}")
            return {"retriever_output": ""}

    def search_agent(self, state: State) -> Dict:
        """Perform a web search using SerpAPI"""
        try:
            query = state["messages"][-1].content  # FIXED: Extract content correctly
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

        return {"messages": [AIMessage(content=response)]}  # FIXED: Use AIMessage

    def setup_graph(self):
        """Set up the workflow graph"""
        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node("retriever", self.retriever_agent)
        workflow.add_node("search", self.search_agent)
        workflow.add_node("final", self.final_processor)

        # Define edges
        workflow.add_edge("retriever", "search")
        workflow.add_edge("search", "final")
        workflow.add_edge("final", END)

        # Set entry point
        workflow.set_entry_point("retriever")

        # Compile graph
        self.graph = workflow.compile()
        logging.info("Graph compiled successfully.")

    def process_query(self, query: str) -> str:
        """
        Process a single query through the workflow

        Args:
            query: User's input query

        Returns:
            str: Processed response
        """
        try:
            state = {
                "messages": [HumanMessage(content=query)],
                "retriever_output": "",
                "search_output": "",
            }

            result = self.graph.invoke(state)

            if result and "messages" in result and result["messages"]:
                return result["messages"][-1].content
            return "I apologize, but I couldn't process your query properly. Please try again."

        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return f"An error occurred: {str(e)}"

    def run_interactive(self):
        """Run the chatbot in interactive mode"""
        print("Starting interactive chat (type 'quit', 'exit', or 'q' to end)")
        print(
            "Note: This chatbot provides general information and should not be considered legal advice."
        )

        while True:
            try:
                user_input = input("\nUser: ").strip()
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                response = self.process_query(user_input)
                print("\nAssistant:", response)

            except KeyboardInterrupt:
                print("\nChat session terminated by user.")
                break
            except Exception as e:
                logging.error(f"Error: {str(e)}")
                print("Please try again.")


def main():
    """Main entry point of the application"""
    try:
        chatbot = LangGraphChatbot()
        chatbot.run_interactive()
    except Exception as e:
        logging.error(f"Failed to initialize chatbot: {str(e)}")


if __name__ == "__main__":
    main()
