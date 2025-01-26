from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# Path to the PDF
commercial_laws_path = "/Users/sanatwalia/Desktop/Assignments_applications/Vidhijna/commercial_laws/merged_output_laws.pdf"

# Load the PDF
loader = PDFPlumberLoader(commercial_laws_path)
print("LOADER INITIALISED")
docs = loader.load()


# Custom Chapter and Section Splitter
def chapter_section_splitter(docs):
    """
    Splits the document into chapters and sections based on headings.
    Assumes chapters and sections are marked by specific patterns (e.g., "Chapter X", "Section Y").
    """
    chapters = []
    current_chapter = []
    for doc in docs:
        text = doc.page_content
        # Split by chapter (assuming chapters start with "Chapter X")
        if "Chapter" in text:
            if current_chapter:
                chapters.append("\n".join(current_chapter))
                current_chapter = []
        current_chapter.append(text)
    if current_chapter:
        chapters.append("\n".join(current_chapter))
    return chapters


# Split the document into chapters and sections
print("Splitting into chapters and sections...")
chapters = chapter_section_splitter(docs)


# Hierarchical Splitting: Split chapters into sections and subsections
def hierarchical_splitter(chapters):
    """
    Splits chapters into smaller sections and subsections.
    """
    split_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for chapter in chapters:
        # Split each chapter into sections
        sections = text_splitter.split_text(chapter)
        for section in sections:
            split_documents.append(section)
    return split_documents


# Perform hierarchical splitting
print("Performing hierarchical splitting...")
split_documents = hierarchical_splitter(chapters)
print("Hiearchical Splitting Done ")

# Convert split documents into LangChain Document format
from langchain_core.documents import Document

documents = [Document(page_content=text) for text in split_documents]
print(f"Total documents after splitting: {len(documents)}")

# Initialize Ollama embeddings
embeddings = OllamaEmbeddings(model="all-minilm:33m")

# Batch-wise embedding generation
batch_size = 100
vectorstore_db = None

print("Generating embeddings batch-wise...")
for i in range(0, len(documents), batch_size):
    batch = documents[i : i + batch_size]
    if vectorstore_db is None:
        # Initialize the FAISS vector store with the first batch
        vectorstore_db = FAISS.from_documents(batch, embeddings)
    else:
        # Add subsequent batches to the existing vector store
        vectorstore_db.add_documents(batch)
    print(
        f"Processed batch {i // batch_size + 1} of {len(documents) // batch_size + 1}"
    )

print("FAISS vector store created successfully!")

# Save the vector store for later use
vectorstore_db.save_local("commercial_laws_index")
print("Vector store saved to 'faiss_index' directory.")
