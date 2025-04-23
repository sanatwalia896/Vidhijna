# import os
# import logging
# import streamlit as st
# from typing import Annotated, Dict, Any, List
# import requests
# import PyPDF2
# from io import BytesIO

# from dotenv import load_dotenv
# from langchain_community.vectorstores import FAISS
# from langchain_groq import ChatGroq
# from langchain_ollama import OllamaEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.utilities.serpapi import SerpAPIWrapper
# from langchain.schema import AIMessage, HumanMessage
# from langgraph.graph import StateGraph, END
# from langgraph.graph.message import add_messages

# # Load environment variables
# load_dotenv()

# # Logging configuration
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )


# class Config:
#     GROQ_MODEL = "llama-3.1-8b-instant"
#     OLLAMA_MODEL = "all-minilm:33m"
#     LAWS_VECTOR_STORE_PATH = "laws_index"
#     CASES_VECTOR_STORE_PATH = "cases_index"
#     SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
#     PDF_DOWNLOAD_PATH = "downloaded_cases"


# class State(Dict[str, Any]):
#     """State definition for legal workflow"""

#     messages: Annotated[List[HumanMessage | AIMessage], add_messages]
#     laws_output: str
#     case_query: str
#     cases_from_db: List[str]
#     cases_from_search: List[str]
#     downloaded_cases: List[str]
#     final_summary: str


# class LegalResearchChatbot:
#     def __init__(self):
#         self.setup_components()
#         self.setup_graph()

#     def setup_components(self):
#         """Initialize all components"""
#         self.llm = ChatGroq(model=Config.GROQ_MODEL)
#         self.embeddings = OllamaEmbeddings(model=Config.OLLAMA_MODEL)

#         # Load or create vector stores
#         if os.path.exists(Config.LAWS_VECTOR_STORE_PATH):
#             self.laws_vectorstore = FAISS.load_local(
#                 Config.LAWS_VECTOR_STORE_PATH,
#                 self.embeddings,
#                 allow_dangerous_deserialization=True,
#             )
#         else:
#             st.error(
#                 "Laws vector store not found. Please initialize the laws database first."
#             )
#             self.laws_vectorstore = FAISS.from_texts(
#                 ["Initial empty store"], self.embeddings
#             )

#         if os.path.exists(Config.CASES_VECTOR_STORE_PATH):
#             self.cases_vectorstore = FAISS.load_local(
#                 Config.CASES_VECTOR_STORE_PATH,
#                 self.embeddings,
#                 allow_dangerous_deserialization=True,
#             )
#         else:
#             self.cases_vectorstore = FAISS.from_texts(
#                 ["Initial empty store"], self.embeddings
#             )

#         self.search = SerpAPIWrapper(serpapi_api_key=Config.SERPAPI_API_KEY)
#         self.text_splitter = RecursiveCharacterTextSplitter()

#     def laws_retriever_agent(self, state: State) -> Dict:
#         """Retrieve relevant laws from the laws vector store"""
#         query = state["messages"][-1].content
#         results = self.laws_vectorstore.similarity_search(query)
#         laws_text = "\n".join([doc.page_content for doc in results])
#         return {"laws_output": laws_text}

#     def query_formation_agent(self, state: State) -> Dict:
#         """Formulate a query for case search based on retrieved laws"""
#         prompt = f"Based on these laws:\n{state['laws_output']}\n\nGenerate a search query for finding relevant Indian legal cases."
#         response = self.llm.predict(prompt)
#         return {"case_query": response}

#     def cases_db_agent(self, state: State) -> Dict:
#         """Check if relevant cases exist in the cases vector store"""
#         results = self.cases_vectorstore.similarity_search(state["case_query"], k=5)
#         cases = [doc.page_content for doc in results]
#         return {"cases_from_db": cases}

#     def search_agent(self, state: State) -> Dict:
#         """Search for cases if not found in the database"""
#         if len(state["cases_from_db"]) >= 5:
#             return {"cases_from_search": []}

#         search_query = f"site:indiankanoon.org {state['case_query']}"
#         results = self.search.run(search_query)
#         return {"cases_from_search": results[:10]}

#     def pdf_download_agent(self, state: State) -> Dict:
#         """Download and process new case PDFs"""
#         if not state["cases_from_search"]:
#             return {"downloaded_cases": []}

#         downloaded_texts = []
#         for case_url in state["cases_from_search"]:
#             try:
#                 response = requests.get(case_url)
#                 pdf = PyPDF2.PdfReader(BytesIO(response.content))
#                 text = "".join([page.extract_text() for page in pdf.pages])
#                 downloaded_texts.append(text)

#                 texts = self.text_splitter.split_text(text)
#                 self.cases_vectorstore.add_texts(texts)

#             except Exception as e:
#                 logging.error(f"Error downloading case: {str(e)}")
#                 continue

#         self.cases_vectorstore.save_local(Config.CASES_VECTOR_STORE_PATH)
#         return {"downloaded_cases": downloaded_texts}

#     def summary_agent(self, state: State) -> Dict:
#         """Generate a final summary combining laws and cases"""
#         all_cases = state["cases_from_db"] + state["downloaded_cases"]
#         most_relevant_case = all_cases[0] if all_cases else "No relevant cases found"

#         prompt = f"""
#         Summarize the legal information:

#         **Laws:**
#         {state['laws_output']}

#         **Most Relevant Case:**
#         {most_relevant_case}
#         """

#         summary = self.llm.predict(prompt)
#         return {"final_summary": summary}

#     def setup_graph(self):
#         """Set up workflow graph with all agents"""
#         workflow = StateGraph(State)
#         workflow.add_node("laws_retriever", self.laws_retriever_agent)
#         workflow.add_node("query_formation", self.query_formation_agent)
#         workflow.add_node("cases_db", self.cases_db_agent)
#         workflow.add_node("search", self.search_agent)
#         workflow.add_node("pdf_download", self.pdf_download_agent)
#         workflow.add_node("summary", self.summary_agent)

#         workflow.add_edge("laws_retriever", "query_formation")
#         workflow.add_edge("query_formation", "cases_db")
#         workflow.add_edge("cases_db", "search")
#         workflow.add_edge("search", "pdf_download")
#         workflow.add_edge("pdf_download", "summary")
#         workflow.add_edge("summary", END)

#         workflow.set_entry_point("laws_retriever")
#         self.graph = workflow.compile()

#     def process_query(self, query: str) -> Dict[str, Any]:
#         """Process a query and return results"""
#         state = {
#             "messages": [HumanMessage(content=query)],
#             "laws_output": "",
#             "case_query": "",
#             "cases_from_db": [],
#             "cases_from_search": [],
#             "downloaded_cases": [],
#             "final_summary": "",
#         }
#         result = self.graph.invoke(state)
#         return {
#             "laws": result.get("laws_output", ""),
#             "cases": result.get("cases_from_db", [])
#             + result.get("downloaded_cases", []),
#             "summary": result.get("final_summary", ""),
#         }


# # Initialize session state to store chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# if "chatbot" not in st.session_state:
#     st.session_state.chatbot = None


# def main():
#     st.set_page_config(
#         page_title="Legal Research Assistant",
#         page_icon="⚖️",
#         layout="wide",
#     )

#     st.title("⚖️ Legal Research Assistant")
#     st.markdown(
#         """
#         A legal research chatbot that retrieves relevant laws and cases for your legal queries.
#         """
#     )

#     # Sidebar for status and configuration
#     with st.sidebar:
#         st.header("Status & Configuration")

#         if st.button("Initialize Chatbot"):
#             with st.spinner("Loading components..."):
#                 try:
#                     st.session_state.chatbot = LegalResearchChatbot()
#                     st.success("Chatbot successfully initialized!")
#                 except Exception as e:
#                     st.error(f"Error initializing chatbot: {str(e)}")

#         st.markdown("---")
#         st.markdown("### Database Status")

#         # Check if databases exist
#         laws_db_exists = os.path.exists(Config.LAWS_VECTOR_STORE_PATH)
#         cases_db_exists = os.path.exists(Config.CASES_VECTOR_STORE_PATH)

#         st.write(f"Laws Database: {'✅ Loaded' if laws_db_exists else '❌ Not Found'}")
#         st.write(
#             f"Cases Database: {'✅ Loaded' if cases_db_exists else '❌ Empty or Not Found'}"
#         )

#         st.markdown("---")
#         st.markdown("### About")
#         st.markdown(
#             """
#             This legal research assistant uses:
#             - LangGraph for workflow orchestration
#             - FAISS vector stores for legal data
#             - Groq LLM for language processing
#             - Ollama for embeddings
#             - SerpAPI for web search
#             """
#         )

#     # Main chat interface
#     chat_container = st.container()

#     with chat_container:
#         # Display chat history
#         for i, (query, result) in enumerate(st.session_state.chat_history):
#             with st.chat_message("user"):
#                 st.write(query)

#             with st.chat_message("assistant"):
#                 st.markdown("#### Summary")
#                 st.write(result["summary"])

#                 with st.expander("View Retrieved Laws"):
#                     st.markdown(result["laws"])

#                 with st.expander("View Retrieved Cases"):
#                     if result["cases"]:
#                         for j, case in enumerate(result["cases"]):
#                             st.markdown(f"**Case {j+1}**")
#                             st.write(case[:500] + "..." if len(case) > 500 else case)
#                             st.markdown("---")
#                     else:
#                         st.write("No relevant cases found.")

#     # Query input
#     query = st.chat_input("Enter your legal query here")

#     if query:
#         if st.session_state.chatbot is None:
#             with st.chat_message("assistant"):
#                 st.error(
#                     "Please initialize the chatbot first using the button in the sidebar."
#                 )
#         else:
#             # Display user message
#             with st.chat_message("user"):
#                 st.write(query)

#             # Process the query
#             with st.chat_message("assistant"):
#                 with st.spinner("Researching legal information..."):
#                     try:
#                         result = st.session_state.chatbot.process_query(query)

#                         # Display results
#                         st.markdown("#### Summary")
#                         st.write(result["summary"])

#                         with st.expander("View Retrieved Laws"):
#                             st.markdown(result["laws"])

#                         with st.expander("View Retrieved Cases"):
#                             if result["cases"]:
#                                 for j, case in enumerate(result["cases"]):
#                                     st.markdown(f"**Case {j+1}**")
#                                     st.write(
#                                         case[:500] + "..." if len(case) > 500 else case
#                                     )
#                                     st.markdown("---")
#                             else:
#                                 st.write("No relevant cases found.")

#                         # Add to chat history
#                         st.session_state.chat_history.append((query, result))

#                     except Exception as e:
#                         st.error(f"Error processing query: {str(e)}")
#                         st.warning(
#                             "If this is the first query, ensure that the laws database is properly initialized."
#                         )


# if __name__ == "__main__":
#     main()
