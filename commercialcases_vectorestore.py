from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# Path to the PDF
commercial_laws_path = "/Users/sanatwalia/Desktop/Assignments_applications/Vidhijna/commercial_cases/merged_output_cases.pdf"

# Load the PDF
loader = PDFPlumberLoader(commercial_laws_path)
print("LOADER INITIALISED")

docs = loader.load()

# Check if documents were loaded
if not docs:
    print("Error: No pages loaded from the PDF. Please check the file path and format.")
    exit()

print(f"Total pages loaded: {len(docs)}")

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

# Check if splitting worked
if not documents:
    print("Error: Document splitting failed. No text chunks were generated.")
    exit()

print(f"Total text chunks created: {len(documents)}")

# Initialize Ollama embeddings
embeddings = OllamaEmbeddings(model="all-minilm:33m")

# Batch-wise embedding generation
batch_size = 100
vectorstore_db = None

print("Generating embeddings batch-wise...")
for i in range(0, len(documents), batch_size):
    batch = documents[i : i + batch_size]

    if not batch:
        continue  # Skip empty batches (unlikely but safe)

    if vectorstore_db is None:
        # Initialize the FAISS vector store with the first batch
        vectorstore_db = FAISS.from_documents(batch, embeddings)
    else:
        # Add subsequent batches to the existing vector store
        vectorstore_db.add_documents(batch)

    print(
        f"Processed batch {i // batch_size + 1} of {len(documents) // batch_size + 1}"
    )

# Ensure FAISS was created before saving
if vectorstore_db:
    vectorstore_db.save_local("cases_index")
    print("Vector store saved to 'cases_index' directory.")
else:
    print("Error: No documents were processed. FAISS vector store was not created.")
