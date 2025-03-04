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
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
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
        self.setup_prompt_template()
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

    def setup_prompt_template(self):
        """Setup the prompt template for the final response generation"""
        self.prompt_template = ChatPromptTemplate.from_template(
            """You are a helpful legal assistant VIDHIJAN  providing information about commercial laws and regulations.
            
            User query: {query}
            
            Information from legal database:
            {retriever_output}
            
            Information from web search:
            {search_output}
            
            Please provide a comprehensive and accurate response to the user's query based on the information provided above.
            Include relevant citations where appropriate.
            If information is insufficient, acknowledge limitations and suggest seeking professional legal advice.
            Your response should be well-structured, clear, and to the point.
            """
        )
        logging.info("Prompt template initialized successfully.")

    def retriever_agent(self, state: State) -> Dict:
        """Retrieve documents from FAISS vector store"""
        try:
            query = state["messages"][-1].content
            results = self.retriever.get_relevant_documents(query)
            output = "\n".join([doc.page_content for doc in results])
            logging.info(f"Retrieved {len(results)} documents from vector store")
            return {"retriever_output": output}
        except Exception as e:
            logging.error(f"Retriever agent error: {str(e)}")
            return {
                "retriever_output": "No relevant information found in the legal database."
            }

    def search_agent(self, state: State) -> Dict:
        """Perform a web search using SerpAPI"""
        try:
            query = state["messages"][-1].content
            site_restricted_query = f"{query} site:https://indiankanoon.org/"
            result = self.search.run(site_restricted_query)
            logging.info("Web search completed successfully")
            return {"search_output": result}
        except Exception as e:
            logging.error(f"Search agent error: {str(e)}")
            return {"search_output": "No additional information found from web search."}

    def final_processor(self, state: State) -> Dict:
        """Process results and generate response using LLM"""
        try:
            # Extract user query
            user_query = state["messages"][-1].content

            # Get retriever and search outputs
            retriever_output = state.get(
                "retriever_output", "No information found in legal database."
            )
            search_output = state.get(
                "search_output", "No information found from web search."
            )

            # Format the prompt with context
            prompt = self.prompt_template.format(
                query=user_query,
                retriever_output=retriever_output,
                search_output=search_output,
            )

            # Generate response using the LLM
            llm_response = self.llm.invoke(prompt)
            response_content = llm_response.content

            logging.info("LLM generated response successfully")
            return {"messages": [AIMessage(content=response_content)]}

        except Exception as e:
            logging.error(f"Final processor error: {str(e)}")
            error_response = "I apologize, but I encountered an error while processing your query. Please try again or rephrase your question."
            return {"messages": [AIMessage(content=error_response)]}

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

            logging.info(f"Processing query: {query}")
            result = self.graph.invoke(state)

            if result and "messages" in result and result["messages"]:
                return result["messages"][-1].content
            return "I apologize, but I couldn't process your query properly. Please try again."

        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}"

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
