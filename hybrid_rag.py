import streamlit as st
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
# --- LlamaIndex Imports ---
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
# Corrected imports for ChatMessage and ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole # MessageRole is typically under core.llms
from llama_index.core.memory import ChatMemoryBuffer 
# --------------------------

from groq import Groq
import tempfile
from googleapiclient.discovery import build

# ---------------------------------------
# CONFIG
# ---------------------------------------

EMBED_MODEL_NAME = st.secrets["EMBED_MODEL_NAME"]

# Load Groq
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
groq_client = Groq(api_key=GROQ_API_KEY)

# Google Search Configuration
CUSTOM_SEARCH_ENGINE_ID = st.secrets["CUSTOM_SEARCH_ENGINE_ID"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]


# ---------------------------------------
# Functions (with Caching)
# ---------------------------------------

@st.cache_resource
def load_embedding_model():
    """Loads and caches the SentenceTransformer model."""
    return SentenceTransformer(EMBED_MODEL_NAME)

embed_model = load_embedding_model()


def load_single_pdf(path):
    """Loads a single PDF document."""
    reader = SimpleDirectoryReader(input_files=[path])
    docs = reader.load_data()
    return docs


@st.cache_data(show_spinner="Splitting document into chunks...")
def chunk_documents(docs):
    """Splits documents into smaller text chunks and caches the result."""
    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=30)
    nodes = splitter.get_nodes_from_documents(docs)
    return [n.get_content() for n in nodes]


@st.cache_resource(show_spinner="Building FAISS index...")
def build_faiss_index(texts):
    """Builds and caches the FAISS index."""
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32)) 
    return index


def retrieve(query, index, texts, k=4):
    """Retrieves top-k relevant text chunks using FAISS similarity search."""
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb.astype(np.float32), k)
    return [texts[i] for i in indices[0]]


# ---------------------------------------
# NEW FUNCTION: Memory Sync (LlamaIndex)
# ---------------------------------------

def sync_memory(history, li_memory: ChatMemoryBuffer):
    """
    Syncs the Streamlit history to the LlamaIndex memory object.
    Memory is cleared and rebuilt with the last 4 messages (excluding the current one).
    """
    li_memory.reset() 

    # history[:-1] ensures we only process past turns, excluding the current user query
    # We use a window of 4 messages for short-term memory (2 turns)
    recent_messages = history[:-1][-4:] 
    
    for message in recent_messages:
        content = message["content"]
        if message["role"] == "user":
            li_memory.put(ChatMessage(role=MessageRole.USER, content=content))
        elif message["role"] == "assistant":
            li_memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=content))


# ---------------------------------------
# MODIFIED FUNCTION: Query Rewriting (Uses LlamaIndex Memory)
# ---------------------------------------

def rewrite_query_with_history(current_query):
    """
    Uses Groq to condense the conversation history (from LlamaIndex memory) 
    and the latest message into a standalone query.
    """
    li_memory: ChatMemoryBuffer = st.session_state.li_memory
    
    # Get the history messages list from the LlamaIndex object
    memory_messages: list[ChatMessage] = li_memory.get_all()
    
    # Format LlamaIndex messages for the Groq prompt
    history_string = "\n".join([f"{m.role.value.capitalize()}: {m.content}" for m in memory_messages])
    
    if not history_string:
         return current_query
         
    rewrite_prompt = f"""
    You are a helpful chat assistant. Your task is to analyze the conversation history and the current question.

    ### INSTRUCTIONS:
    1.  Rewrite the current question into a single, comprehensive, standalone search query for document retrieval.
    2.  If the question is already standalone, return it exactly as is.

    ### CONVERSATION HISTORY (Last few turns):
    {history_string}
    
    ### CURRENT QUESTION:
    {current_query}
    
    ### STANDALONE QUERY:
    """
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Query Rewriting Error: {e}")
        return current_query
    

def check_question_type(query):
    """
    Performs a final check on a 'QUESTION' intent to distinguish between
    Factual (needs RAG) and Conversational/Social (needs a quick response).
    Returns 'FACT' or 'SOCIAL'.
    """
    router_prompt = f"""
    Analyze the user's input: '{query}'.
    
    Is this input a factual question requiring information retrieval from a document (e.g., 'What is X?', 'Summarize Y', 'Tell me about Z')?
    
    OR
    
    Is it a social greeting, a request for assistance, or a conversational opener (e.g., 'Hello', 'Can you help me', 'How are you')?

    Output ONLY one of the following exact words: FACT or SOCIAL.
    """
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": router_prompt}],
            temperature=0.0, 
            max_tokens=10,
        )
        return response.choices[0].message.content.strip().upper()
    except Exception as e:
        st.error(f"Router Error: {e}")
        return "SOCIAL" # Fail safe to SOCIAL to avoid RAG crash
    

# ---------------------------------------
# NEW FUNCTION: Input Classification (Unchanged)
# ---------------------------------------

def classify_input(query):
    # (Function body remains the same)
    """
    Uses Groq to classify the user's input to determine the required action.
    """

    classification_prompt = f"""
    You are an expert intent classification model.
    Your task is to analyze the user's CURRENT_PROMPT and determine the most appropriate action category.

    ### ACTION_TAGS:
    - QUESTION: The prompt requires a factual answer, summary, or an explanation related to the document or external knowledge (e.g., "What is the capital cost?", "Summarize the findings in Section 3.", "What is the purpose of the report?").
    - AFFIRMATION: The prompt is a simple statement of agreement, thanks, or a conversational closure (e.g., "Thanks.", "Got it.", "Nice.", "Okay, I understand.").
    - COMMAND: The prompt is a direct command unrelated to a specific question, like a greeting or a request to reset (e.g., "Hello," "Start over.").

    ### INSTRUCTIONS:
    1.  Based on the analysis of the CURRENT_PROMPT, select the single best ACTION_TAG.
    2.  Output ONLY the selected ACTION_TAG, followed by a colon, and then your confidence level (0.0 to 1.0).
    3.  Example Output: QUESTION:0.95 or AFFIRMATION:1.0

    ### CURRENT_PROMPT:
    {query}

    ### CLASSIFICATION:
    """
    
    try:
        # NOTE: Keeping temperature at 0.0 is best practice for classification tasks.
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": classification_prompt}],
            temperature=0.0, 
            max_tokens=20,
        )
        result = response.choices[0].message.content.strip().split(':')[0].upper()
        return result
    except Exception as e:
        st.error(f"Classification Error: {e}")
        return "QUESTION"
    
# ---------------------------------------
# NEW FUNCTION: Social Response Handler (Unchanged)
# ---------------------------------------

def generate_social_response(query, intent_tag):
    # (Function body remains the same)
    """Generates a conversational and empathetic response for comments/affirmations."""

    social_prompt = f"""
    You are a friendly and helpful AI assistant. The user's input was classified as '{intent_tag}'.

    ### INSTRUCTIONS:
    1.  **Acknowledge and Validate:** Acknowledge the user's input naturally and briefly, maintaining a helpful and positive tone.
    2.  **Contextual Continuation:** If the input was a simple conversational comment, offer a suggestion to continue the core task (document analysis/Q&A). Ask if they have any other questions about the document or their previous query.
    3.  **Handle Gratitude:** If the input was a 'thanks', respond politely and conclude your response, still leaving the door open for another question.
    4.  **Brevity:** Keep your entire response short and social.

    ### USER INPUT:
    {query}

    ### ASSISTANT RESPONSE:
    """
    
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": social_prompt}],
        temperature=0.7, 
        max_tokens=100,
        stream=True
    )
    return response

# ---------------------------------------
# Existing Functions (Web Search) (Unchanged)
# ---------------------------------------

def web_search(query):
    # (Function body remains the same, returns stream generator)
    st.info(f"🌐 Performing Web Search for: **{query}**")
    
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = service.cse().list(
            q=query,
            cx=CUSTOM_SEARCH_ENGINE_ID,
            num=3
        ).execute()
    except Exception as e:
        st.error(f"Google Custom Search API Error: {e}. Check your API Key and CX ID.")
        return iter(["Web search failed due to an API error."])

    search_results = res.get('items', [])
    if not search_results:
        return iter(["The web search returned no relevant results."])

    web_context = ""
    for i, item in enumerate(search_results):
        web_context += f"Source {i+1}: {item.get('title')}\n"
        web_context += f"URL: {item.get('link')}\n"
        web_context += f"Snippet: {item.get('snippet')}\n\n"
    
    web_prompt = f"""
    You are a highly efficient **Fact Synthesis Agent**.

    Your sole task is to generate a single, concise, and direct answer to the user's question, using **ONLY** the provided web snippets.

    ### INSTRUCTIONS:
    1.  **Strict Source Attribution (NEW!):** Start your final response with the exact phrase: **"The answer was not found in the document, but according to web search, "**
    2.  **Strictly Answer the Question:** Focus exclusively on answering the core question. Do elaborate, provide background, or discuss the source content's general topic.
    3.  **Brevity:** The answer must be as brief and direct as possible.
    4.  **Source Citation:** Cite the relevant sources by number and provide the url of the content. immediately following the answer.
    5.  **Handle Insufficiency:** If the snippets do not contain a direct answer, state *only* "The answer is not available in the provided sources." (This overrides instruction 1 if no facts are found.)

    ### WEB SNIPPETS:
    {web_context}

    ### QUESTION:
    {query}

    ### ANSWER:
    """
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": web_prompt}],
        temperature=0.2,
        max_tokens=500,
        stream=True
    )
    return response


# ---------------------------------------
# Content Verification (Unchanged)
# ---------------------------------------

def verify_content_availability(query, chunks):
    # ... (Function body remains the same) ...
    context = "\n\n".join(chunks)
    
    verification_prompt = f"""
    You are a meticulous content verifier. Your only task is to determine if the EXACT, FACTUAL ANSWER to the QUESTION is contained within the provided CONTEXT.
    ### INSTRUCTIONS:
    1.  **Analyze Context:** Read the CONTEXT carefully to locate a specific, unambiguous answer (e.g., a specific date, name, or number) that directly addresses the QUESTION.
    2.  **Strict Output (Mandatory):** - If the context contains the specific, unambiguous answer, output the exact word **YES**.
        - If the context does NOT contain the specific answer, or if the answer is vague/incomplete, output the exact word **NO**.
        - **DO NOT** provide any other text, reasoning, quotes, or punctuation.

    ### CONTEXT:
    {context}

    ### QUESTION:
    {query}

    ### VERIFICATION:
    """
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": verification_prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        return response.choices[0].message.content.strip().upper()
    except Exception as e:
        st.error(f"Verification Error: {e}")
        return "NO"

# ---------------------------------------
# MODIFIED generate_answer (Uses LlamaIndex Memory)
# ---------------------------------------

def generate_answer(query, chunks):
    """
    Generates an answer from document chunks, or falls back to web search.
    Handles all streaming internally and returns the final consolidated string.
    """
    context = "\n\n".join(chunks)

    # --- LLAMAINDEX MEMORY INTEGRATION ---
    li_memory: ChatMemoryBuffer = st.session_state.li_memory
    memory_messages: list[ChatMessage] = li_memory.get_all()
    
    # Format messages for the prompt
    history_string = "\n".join([f"{m.role.value.capitalize()}: {m.content}" for m in memory_messages])
    # ------------------------------------

    prompt = f"""

    You are an expert document analysis assistant. Your role is to process the user's request based on the provided context.
    
    ### CONVERSATION HISTORY (from LlamaIndex short-term memory):
    {history_string}

    ### CURRENT QUESTION:
    {query}

    ### MODE INSTRUCTIONS:

    **A. SUMMARIZATION & GENERAL COMMANDS:**
    If the USER REQUEST is a high-level command, use ALL of the retrieved CONTEXT chunks to provide a comprehensive and cohesive overview.

    ### FINAL OUTPUT INSTRUCTION:
    Based on the MODE INSTRUCTIONS above, provide the final answer immediately under the ANSWER tag. DO NOT repeat, summarize, or explain the rules used (A or B). Provide only the factual result.
    ### CONTEXT:
    {context}

    ### QUESTION:
    {query}

    ### ANSWER:
    """

    # --- 1. RAG Attempt and Streaming ---
    response_generator = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500,
        stream=True
    )

    full_response = ""
    answer_placeholder = st.empty()
    
    # Manually stream RAG response and consolidate the text
    for chunk in response_generator:
        content = chunk.choices[0].delta.content
        if content:
            full_response += content
            answer_placeholder.markdown(full_response)
    
    final_answer_text = full_response.strip()

    # --- 2. Fallback Check and Web Search ---
    if final_answer_text.startswith("NOT FOUND IN DOCUMENT"):
        st.warning("Could not find a relevant answer in the document. Initiating web search...")
        answer_placeholder.empty()
        
        web_result_generator = web_search(query) 
        
        final_web_answer = ""
        web_answer_placeholder = st.empty()

        for chunk in web_result_generator:
            content = chunk.choices[0].delta.content
            if content:
                final_web_answer += content
                web_answer_placeholder.markdown(final_web_answer)

        return final_web_answer
    
    else:
        return final_answer_text


# ---------------------------------------
# Streamlit App Flow
# ---------------------------------------

st.set_page_config(page_title="PDF RAG with Groq + Web Fallback + LlamaIndex Memory", page_icon="📘", layout="wide")
st.title("📘 PDF Question-Answering (RAG) with Groq + FAISS + LlamaIndex Memory")
st.write("Upload a PDF and ask any question. The system uses LlamaIndex memory and intent classification.")

# --- 1. SESSION STATE AND LLAMAINDEX MEMORY INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "li_memory" not in st.session_state:
    # ChatMemoryBuffer stores messages and handles the window logic internally
    st.session_state.li_memory = ChatMemoryBuffer.from_defaults(token_limit=3000) # Set a high limit to only use window size for now


uploaded_file = st.file_uploader("📤 Upload PDF", type=["pdf"])

if uploaded_file:
    st.success("PDF uploaded! Processing...")

    pdf_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            pdf_path = temp_file.name

        docs = load_single_pdf(pdf_path)
        texts = chunk_documents(docs)
        index = build_faiss_index(texts)

        st.success("✅ PDF processed! You can now ask questions.")

        # --- 2. DISPLAY HISTORY ---
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Question input logic
        if query := st.chat_input("🔍 Ask a question about the PDF..."):
            
            # 3. MEMORY SYNC & SAVE USER MESSAGE
            # Sync existing Streamlit history to LlamaIndex memory before adding the new query
            sync_memory(st.session_state.messages, st.session_state.li_memory)
            
            # Save the new user message to Streamlit state
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})
            
            # 4. CORE LOGIC GATE (Intent Classification)
            with st.spinner("Analyzing intent..."):
                intent = classify_input(query)
                st.info(f"Intent classified as: **{intent}**")
            
            with st.chat_message("assistant"):
                
                final_answer_text = ""
                
                if intent in ["AFFIRMATION", "COMMAND"]:
                    # --- PATH A: SOCIAL RESPONSE ---
                    st.info(f"Generating Social Response for: **{intent}**")
                    response_generator = generate_social_response(query, intent)

                    # final_answer_text = st.write_stream(response_generator)
                    
                    answer_placeholder = st.empty()
                    full_response = ""
                    for chunk in response_generator:
                        content = chunk.choices[0].delta.content
                        if content:
                            full_response += content
                            answer_placeholder.markdown(full_response)
                            
                    final_answer_text = full_response

                elif intent == "QUESTION":
                    # Default to QUESTION intent (PATH B: RAG/WEB SEARCH)
                    with st.spinner("Refining intent..."):
                        # Safety check for inputs like "Can you help me?" that are classified as QUESTION
                        q_type = check_question_type(query)
                        st.info(f"Question Type refined to: **{q_type}**")
                    
                    if q_type == "SOCIAL":
                        # Route conversational questions to the social handler
                        response_generator = generate_social_response(query, "SOCIAL_QUESTION")
                        # final_answer_text = st.write_stream(response_generator)

                        answer_placeholder = st.empty()
                        full_response = ""
                        for chunk in response_generator:
                            content = chunk.choices[0].delta.content
                            if content:
                                full_response += content
                                answer_placeholder.markdown(full_response)
                                
                        final_answer_text = full_response
                        
                    elif q_type == "FACT":
                    # --- RAG Core Logic Starts Here ---
                    
                    # 1. Rewrite Query (uses LlamaIndex memory internally)
                        with st.spinner("Thinking & Rewriting Query..."):
                            rewritten_query = rewrite_query_with_history(query)
                            st.info(f"Rewritten Query (for retrieval): **{rewritten_query}**")
                        
                        # 2. Retrieve Chunks
                        chunks = retrieve(rewritten_query, index, texts)
                        
                        # 3. Generate Answer (uses LlamaIndex memory internally and streams)
                        # Returns the final string
                        final_answer_text = generate_answer(query, chunks) 
                        
                        # --- RAG/Web Answer Display ---
                        st.subheader("🧠 Answer") 
                        
                        with st.expander("📄 Retrieved Chunks (Used for Document Check)"):
                            if "NOT FOUND IN DOCUMENT" not in final_answer_text:
                                for i, c in enumerate(chunks):
                                    st.markdown(f"*Chunk {i+1}:*\n{c}\n---")
                            else:
                                st.markdown("Web search was performed because the document chunks were deemed insufficient by the LLM.")
                                
                    # --- 5. SAVE ASSISTANT MESSAGE & UPDATE LLAMAINDEX MEMORY ---
                    
                    # Update Streamlit State
                if final_answer_text:
                    st.session_state.messages.append({"role": "assistant", "content": final_answer_text})
                    
                    # Update LlamaIndex Memory (Crucial for next turn's context)
                    st.session_state.li_memory.put(ChatMessage(role=MessageRole.USER, content=query))
                    st.session_state.li_memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=final_answer_text))

    finally:
            if pdf_path and os.path.exists(pdf_path):
                os.unlink(pdf_path)