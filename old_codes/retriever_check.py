from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# Load the saved vector store
embeddings = OllamaEmbeddings(model="all-minilm:33m")
vectorstore_db = FAISS.load_local(
    "commercial_laws_index", embeddings, allow_dangerous_deserialization=True
)

# Use the vector store in a chatbot or other application
retriever = vectorstore_db.as_retriever()
results = retriever.invoke("How many  commercial laws are there in the INDIAN LAW ")
for result in results:
    print(result.page_content)
