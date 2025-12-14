# PDF-Rag-Chatbot-with-Web-fallback
It allows users to upload a PDF document and ask complex questions, maintaining conversation history (memory) while dynamically deciding whether to use the document's content, a web search, or a simple social response.

It leverages the speed of Groq's Llama 3.1 LLM for low-latency routing and generation, the efficiency of FAISS for vector retrieval, and the structure of LlamaIndex for managing chat memory.

📋 Prerequisites
Python 3.9+API Keys:Groq API Key: Required for all LLM calls (Intent Classification, Query Rewriting, and Answer Generation).
Google Custom Search API Key (GOOGLE_API_KEY): Required for the web search fallback.
Google Custom Search Engine ID (CUSTOM_SEARCH_ENGINE_ID): Required for the web search fallback.

🛠️ Installation
1. Clone the repository

git clone <repository_url>
cd <repository_name>
2. Create and activate the virtual environment

python -m venv venv
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate   # On Windows
3. Install dependencies
Install all required Python packages using the provided requirements.txt file:

pip install -r requirements.txt

Configure Secrets
Create a hidden directory and a secrets.toml file to store your API keys securely.

mkdir .streamlit
touch .streamlit/secrets.toml
Populate .streamlit/secrets.toml with your credentials:

Ini, TOML

# .streamlit/secrets.toml

# Sentence Transformer Model Name (Cached Resource)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Groq API Key
GROQ_API_KEY = "gsk_YOUR_GROQ_API_KEY_HERE"

# Google Custom Search API Credentials (for Web Fallback)
CUSTOM_SEARCH_ENGINE_ID = "YOUR_CUSTOM_SEARCH_ENGINE_ID" 
GOOGLE_API_KEY = "AIzaSy_YOUR_GOOGLE_API_KEY"

How to Run the App
Execute the Streamlit command from the root directory of the project:

streamlit run app.py


Architecture Overview:
The RAG pipeline operates in the following sequence:

PDF Ingestion: Uploaded PDF is loaded, chunked, and embedded.
Indexing: Embeddings are stored in a FAISS IndexFlatL2 for fast similarity search.
Chat Input: User asks a question.
Memory Sync: Streamlit history is synchronized with the LlamaIndex ChatMemoryBuffer.
Intent Routing (Groq): Classifies the input to determine the necessary path (SOCIAL or FACT).
Query Rewriting (Groq): If FACT, the LLM uses the ChatMemoryBuffer to rewrite the current query and history into a standalone search query.
Retrieval (FAISS): The rewritten query is used to fetch the top relevant text chunks from the document.
Answer Generation (Groq): The full conversation history and the retrieved chunks are passed to the Groq LLM, which streams the answer back.
Web Fallback: If the document's content is insufficient, a Google Custom Search is performed, and the LLM synthesizes the final answer from web snippets.
Memory Update: The final user/assistant exchange is saved to the ChatMemoryBuffer for the next turn.


