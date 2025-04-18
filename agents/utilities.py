import os
from langchain_groq import ChatGroq  # for inferenceimport json

from typing_extensions import Literal
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings  #
from langgraph.graph import START, END, StateGraph
from agents.configuration import Configuration, SearchAPI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()


class Config:
    def __init__(self):
        self.commercial_laws_path = "/Users/sanatwalia/Desktop/Assignments_applications/Vidhijna/commercial_laws_index/"
        self.commercial_cases_path = (
            "/Users/sanatwalia/Desktop/Assignments_applications/Vidhijna/cases_index"
        )
        self.serpapi: str = os.get_env()
