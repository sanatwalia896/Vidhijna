# Vidhijna

<b>To view previous version: https://github.com/tanmay-vig/ai-driven-research-engine</b>

# Screenshots
![WhatsApp Image 2025-02-25 at 00 04 02_11edc480](https://github.com/user-attachments/assets/31417c66-9833-44dc-b632-77ec29847cde) <br>
![WhatsApp Image 2025-02-25 at 00 04 03_db82d3b8](https://github.com/user-attachments/assets/817651a1-0eb2-4d07-b6f7-e03c8c578f7d)


## 🚀 Overview

The Legal Assistant Chatbot is an intelligent system designed to provide information about commercial laws and regulations. Built using the LangGraph framework, it combines document retrieval from a vector database with web search capabilities to provide comprehensive responses to legal queries.


## ✨ Features

- *Interactive Chat Interface*: Streamlit-based UI for easy interaction
- *Vector Store Retrieval*: Uses FAISS to retrieve relevant legal information from a pre-populated database
- *Web Search Integration*: Supplements answers with up-to-date information from the web via SerpAPI
- *Multi-Step Processing*: Powered by LangGraph for flexible, multi-stage query processing
- *Chat History*: Maintains conversation context across interactions
- *Configurable Models*: Choose different language and embedding models based on your needs

## 📋 Requirements

- Python 3.8+
- Required Python packages (see requirements.txt)
- SerpAPI API key (for web search functionality)
- Groq API key (for LLM functionality)
- Ollama installed locally (for embeddings)

## 💾 Installation

1. Clone the repository:

   bash
   git clone https://github.com/yourusername/legal-assistant-chatbot.git
   cd legal-assistant-chatbot
   

2. Install dependencies:

   bash
   pip install -r requirements.txt
   

3. Create a .env file in the root directory with your API keys:

   
   SERPAPI_API_KEY=your_serpapi_key_here
   GROQ_API_KEY=your_groq_api_key_here
   

4. Ensure you have Ollama installed and running with the required models:
   bash
   ollama pull all-minilm:33m
   

## 🚀 Usage

### Running the Streamlit Web App

bash
streamlit run app.py


This will launch the web interface at http://localhost:8501.

### Using the Command Line Interface

For a simpler interface, you can run the chatbot in CLI mode:

bash
python main.py


## 📚 Vector Store Setup

The chatbot relies on a pre-populated FAISS vector store containing commercial law information. To create or update the vector store:

1. Prepare your document collection in a directory
2. Run the indexing script:
   bash
   python scripts/create_index.py --input_dir your_documents/ --output_dir commercial_laws_index
   

## 🧩 Technical Architecture

### Components

- *LangGraphChatbot*: Main class orchestrating the workflow
- *State Graph*:
  - *Retriever Node*: Queries the vector store for relevant documents
  - *Search Node*: Performs web searches for additional context
  - *Final Processor*: Combines information and generates responses
- *Streamlit Interface*: Provides a user-friendly web interface

### Data Flow


User Query → State Initialization → Document Retrieval → Web Search → Response Generation → UI Display


## 🔧 Configuration

You can modify the following settings in Config class:

- GROQ_MODEL: The language model to use (default: "llama-3.1-8b-instant")
- OLLAMA_MODEL: The embedding model for vector search (default: "all-minilm:33m")
- VECTOR_STORE_PATH: Path to the FAISS vector store (default: "commercial_laws_index")

## ⚠ Disclaimer

This chatbot provides general information about commercial laws and regulations. It should not be considered legal advice. Always consult with a qualified legal professional for specific legal matters.

## 📞 Support

If you encounter any issues or have questions, please open an issue on the GitHub repository. 

# Contributers:
<ol>
   <li>
      <a href="https://github.com/tanmay-vig">Tanmay Vig</a>
   </li>
   <li>
      <a href="https://github.com/sanatwalia896">Sanat Walia</a>
   </li>
</ol>
