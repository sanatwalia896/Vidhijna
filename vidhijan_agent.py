# Standard library imports
import os
from typing import Annotated, Dict, Any, TypedDict, List
from typing_extensions import TypedDict

# Third-party imports
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain.agents import AgentExecutor, Tool
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

# Load environment variables
load_dotenv()


# Configuration
class Config:
    GROQ_MODEL = "llama-3.1-8b-instant"
    OLLAMA_MODEL = "all-minilm:33m"
    VECTOR_STORE_PATH = "commercial_laws_index"
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


# Define the state structure
class State(TypedDict):
    """State definition for the workflow"""

    messages: Annotated[List[Dict[str, str]], add_messages]
    current_agent: str
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
        except Exception as e:
            raise Exception(f"Failed to initialize LLM: {str(e)}")

    def setup_embeddings(self):
        """Initialize embeddings model"""
        try:
            self.embeddings = OllamaEmbeddings(model=Config.OLLAMA_MODEL)
        except Exception as e:
            raise Exception(f"Failed to initialize embeddings: {str(e)}")

    def setup_vector_store(self):
        """Initialize FAISS vector store"""
        try:
            self.vectorstore_db = FAISS.load_local(
                Config.VECTOR_STORE_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            self.retriever = self.vectorstore_db.as_retriever()
        except Exception as e:
            raise Exception(f"Failed to load vector store: {str(e)}")

    def setup_serpapi(self):
        """Initialize SerpAPI wrapper"""
        if not Config.SERPAPI_API_KEY:
            raise ValueError("SERPAPI_API_KEY not found in environment variables")
        try:
            self.search = SerpAPIWrapper(serpapi_api_key=Config.SERPAPI_API_KEY)
        except Exception as e:
            raise Exception(f"Failed to initialize SerpAPI: {str(e)}")

    def get_query_from_state(self, state: State) -> str:
        """Extract query from state messages"""
        if state["messages"]:
            last_message = state["messages"][-1]
            if isinstance(last_message, dict):
                return last_message.get("content", "")
            return str(last_message)  # Convert message object to string if needed
        return ""

    def retriever_agent(self, state: State) -> Dict:
        """Handle retrieval from vector store"""
        try:
            query = self.get_query_from_state(state)
            if not query:
                return {"current_agent": "retriever", "retriever_output": ""}

            results = self.retriever.get_relevant_documents(query)
            output = "\n".join([doc.page_content for doc in results])
            return {"current_agent": "retriever", "retriever_output": output}
        except Exception as e:
            print(f"Retriever agent error: {str(e)}")
            return {"current_agent": "retriever", "retriever_output": ""}

    def search_agent(self, state: State) -> Dict:
        """Handle web search"""
        try:
            query = self.get_query_from_state(state)
            if not query:
                return {"current_agent": "search", "search_output": ""}

            result = self.search.run(query)
            return {"current_agent": "search", "search_output": result}
        except Exception as e:
            print(f"Search agent error: {str(e)}")
            return {"current_agent": "search", "search_output": ""}

    def final_processor(self, state: State) -> Dict:
        """Process and combine results"""
        try:
            response_parts = []

            if state.get("retriever_output"):
                response_parts.append(
                    f"Based on our legal database:\n{state['retriever_output']}"
                )

            if state.get("search_output"):
                response_parts.append(
                    f"Additional information from general sources:\n{state['search_output']}"
                )

            if not response_parts:
                response = (
                    "I apologize, but I couldn't find specific information about your query. "
                    "For merger-related legal matters, I recommend:\n"
                    "1. Consulting with a qualified legal professional\n"
                    "2. Reviewing your merger agreement documentation\n"
                    "3. Contacting your company's legal department\n"
                    "4. Consider reaching out to the Competition Commission of India (CCI) for guidance"
                )
            else:
                response = "\n\n".join(response_parts)
                response += "\n\nPlease note: This information is for general guidance only. For specific legal advice, please consult with a qualified legal professional."

            return {
                "messages": [{"role": "assistant", "content": response}],
                "current_agent": "final",
            }
        except Exception as e:
            print(f"Final processor error: {str(e)}")
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "I apologize, but I encountered an error processing your query. Please try again or consult with a legal professional for specific advice.",
                    }
                ],
                "current_agent": "final",
            }

    def setup_graph(self):
        """Set up the workflow graph"""
        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node("retriever", self.retriever_agent)
        workflow.add_node("search", self.search_agent)
        workflow.add_node("final", self.final_processor)

        # Add edges
        workflow.add_edge("retriever", "search")
        workflow.add_edge("search", "final")
        workflow.add_edge("final", END)

        # Set entry point
        workflow.set_entry_point("retriever")

        # Compile the graph
        self.graph = workflow.compile()

    def process_query(self, query: str) -> str:
        """Process a single query through the workflow"""
        try:
            # Initialize state
            state = {
                "messages": [{"role": "user", "content": query}],
                "current_agent": "",
                "retriever_output": "",
                "search_output": "",
            }

            # Run the graph
            result = self.graph.invoke(state)

            # Return the last message
            if result and "messages" in result and result["messages"]:
                return result["messages"][-1]["content"]
            return "I apologize, but I couldn't process your query properly. Please try again."

        except Exception as e:
            return f"Error processing query: {str(e)}"

    def run_interactive(self):
        """Run the chatbot in interactive mode"""
        print("\nLegal Information Assistant - Interactive Mode")
        print("============================================")
        print(
            "Note: This chatbot provides general information only and should not be considered legal advice."
        )
        print("Type 'quit', 'exit', or 'q' to end the session.\n")

        while True:
            try:
                user_input = input("\nUser: ").strip()
                if user_input.lower() in ["quit", "exit", "q"]:
                    print(
                        "\nGoodbye! Remember to consult with legal professionals for specific advice."
                    )
                    break

                response = self.process_query(user_input)
                print("\nAssistant:", response)

            except KeyboardInterrupt:
                print("\nChat session terminated by user.")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try again.")


def main():
    """Main entry point of the application"""
    try:
        chatbot = LangGraphChatbot()
        chatbot.run_interactive()
    except Exception as e:
        print(f"Failed to initialize chatbot: {str(e)}")
        return


if __name__ == "__main__":
    main()
