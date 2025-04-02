import os
import streamlit as st
import streamlit.components.v1 as components
import PyPDF2
import openai
from openai import OpenAI # Explicit import for clarity
import chromadb
import pysqlite3
import sys
sys.modules['sqlite3'] = pysqlite3
import traceback # For detailed error logging


# -------------------------------
# PDF Parsing & Chunking Functions
# (Keep @st.cache_data - essential for performance after first load)
# -------------------------------

@st.cache_data(show_spinner=False)
def parse_pdf(file_path):
    """Extracts text from a PDF file using PyPDF2."""
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
            if num_pages == 0:
                st.warning(f"Warning: PDF '{file_path}' has 0 pages.")
                return "" # Return empty string if no pages

            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n" # Add newline between pages for clarity

        if not text:
            st.warning(f"Warning: No text extracted from PDF: {file_path}. It might be image-based or empty.")
        return text
    except FileNotFoundError:
        st.error(f"Error: PDF file not found at {file_path}")
        print(f"ERROR in parse_pdf: FileNotFoundError for {file_path}")
        return None # Return None on failure
    except Exception as e:
        st.error(f"Error parsing PDF '{file_path}': {e}")
        print(f"ERROR in parse_pdf: {traceback.format_exc()}")
        return None # Return None on failure

@st.cache_data(show_spinner=False)
def chunk_text(text, chunk_size=1200, overlap=200):
    """
    Splits text into chunks of a specified size with an overlap.
    """
    chunks = []
    if text is None or not isinstance(text, str) or len(text) == 0:
         st.error("Cannot chunk text: Input text is invalid or empty (check PDF parsing).")
         print("ERROR: Chunking function received invalid input text.")
         return [] # Return empty list

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # Move start forward, ensuring overlap; prevent infinite loop if chunk_size <= overlap
        next_start = start + chunk_size - overlap
        if next_start <= start:
             print(f"WARN: Adjusting chunking step; chunk_size ({chunk_size}) <= overlap ({overlap}). Moving by chunk_size.")
             next_start = start + chunk_size # Move forward by full chunk size to avoid loop
        start = next_start

    return chunks

# -------------------------------
# Embedding & LLM Functions (Using OpenAI v1.x SDK)
# -------------------------------

def get_embedding(text_to_embed, client, model="text-embedding-3-small"):
    """Fetches an embedding for the provided text using OpenAI's embeddings API."""
    if not text_to_embed or not isinstance(text_to_embed, str):
        print("ERROR: get_embedding called with invalid text.")
        raise ValueError("Cannot get embedding for empty or invalid text.")
    try:
        processed_text = text_to_embed.replace("\n", " ").strip()
        if not processed_text:
             print("WARN: Text became empty after replacing newlines, cannot embed.")
             raise ValueError("Cannot get embedding for whitespace-only text.")

        response = client.embeddings.create(input=[processed_text], model=model)
        embedding = response.data[0].embedding
        return embedding
    except openai.APIError as e:
        st.error(f"OpenAI API Error getting embedding: {e}")
        print(f"ERROR: OpenAI API Error in get_embedding: {traceback.format_exc()}")
        raise
    except Exception as e:
        st.error(f"Unexpected error getting embedding: {e}")
        print(f"ERROR: Unexpected error in get_embedding: {traceback.format_exc()}")
        raise

# <<< MODIFICATION 1: Refine the AI Prompt >>>
def generate_answer(query, context, client, temperature=0.2, model="gpt-4-turbo"):
    """Generates an answer using GPT based on the provided query and context."""
    if not query:
        print("WARN: generate_answer called with empty query.")
        return "Please provide a question."
    if not context:
        print("WARN: generate_answer called with empty context. Answering based on query alone might be misleading.")
        context = "No relevant context found." # Slightly better default

    # --- REVISED PROMPT ---
    prompt_instructions = f"""
You are a knowledgeable golf instructor answering questions based ONLY on the provided golf instruction materials below.

**Instructions:**
*   Answer the user's question accurately using ONLY the information found in the 'Provided Context'.
*   Directly address the question.
*   Do NOT mention the context itself (e.g., avoid phrases like "Based on the context...", "According to the text..."). Just provide the answer as if you know it from the material.
*   If the answer isn't found in the context, state clearly: "I couldn't find information about that specific topic in the provided golf guide."
*   Be concise.

Provided Context:
---
{context}
---

User's Question: {query}

Answer:
"""
    # --- END REVISED PROMPT ---

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                # System message can be simpler now
                {"role": "system", "content": "You are a helpful golf instructor answering questions based on provided text snippets."},
                {"role": "user", "content": prompt_instructions} # Use the revised prompt
            ],
            temperature=temperature,
            max_tokens=500
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except openai.APIError as e:
        st.error(f"OpenAI API Error generating answer: {e}")
        print(f"ERROR: OpenAI API Error in generate_answer: {traceback.format_exc()}")
        return f"Sorry, I encountered an API error while trying to generate an answer. ({e.type})"
    except Exception as e:
        st.error(f"Unexpected error generating answer: {e}")
        print(f"ERROR: Unexpected error in generate_answer: {traceback.format_exc()}")
        return "Sorry, I encountered an unexpected error while trying to generate an answer."
# <<< END MODIFICATION 1 >>>

# -------------------------------
# Chroma DB Functions with Persistence - CORRECTED LOGIC
# (No changes needed in this function)
# -------------------------------

def get_or_create_chroma_collection(chunks, client, persist_dir="./chroma_db_store"):
    """
    Initializes Chroma DB client, loads collection if exists, otherwise creates and embeds.
    Handles potential race conditions or errors during initial load vs create.
    """
    collection_name = "golf_guide_embeddings_v3"
    collection = None # Initialize collection variable

    # --- Ensure persist directory exists ---
    try:
        os.makedirs(persist_dir, exist_ok=True)
    except OSError as e:
        st.error(f"Fatal Error: Could not create persistence directory '{persist_dir}': {e}. Check permissions.")
        print(f"ERROR: Could not create directory {persist_dir}: {traceback.format_exc()}")
        return None

    # --- Initialize Chroma client ---
    try:
        chroma_client = chromadb.PersistentClient(path=persist_dir)
    except Exception as e:
        st.error(f"Fatal Error: Failed to initialize Chroma client for path '{persist_dir}': {e}")
        print(f"ERROR: Failed to initialize Chroma client: {traceback.format_exc()}")
        return None

    # --- Attempt to GET the collection first ---
    try:
        collection = chroma_client.get_collection(collection_name)
        print(f"Loaded existing collection '{collection_name}'.") # Use info or success
        count = collection.count()
        print(f"Successfully loaded collection '{collection_name}'. Contains {count} items.")
        if count == 0 and chunks:
             st.warning(f"Warning: Existing collection '{collection_name}' is empty. Consider deleting the '{persist_dir}' folder and restarting if you expect it to be populated.")
        return collection
    except Exception as get_err:
        # This exception could be due to collection not existing OR other DB errors.
        # We proceed cautiously to the CREATE step.
        st.warning(f"Could not get collection '{collection_name}': {get_err}. Attempting to create...")
        print(f"WARN: Failed to get collection '{collection_name}' initially: {traceback.format_exc()}. Will try to create.")

    # --- If GET failed, attempt to CREATE the collection ---
    # This block is only reached if the initial get_collection failed.
    try:
        collection = chroma_client.create_collection(name=collection_name)
        st.info(f"Created new collection: '{collection_name}'. Now embedding data...")
        print(f"Successfully created new collection '{collection_name}'.")

        # --- Embed and add chunks (only if creating successfully) ---
        if not chunks:
             st.error("Cannot populate collection: No text chunks provided (check PDF parsing and chunking steps).")
             print("ERROR: No chunks available to add to the new collection.")
             # Consider deleting the empty collection if creation succeeded but no chunks
             try:
                 chroma_client.delete_collection(collection_name)
                 st.info(f"Deleted empty collection '{collection_name}' as no chunks were available.")
             except Exception as del_e:
                 print(f"ERROR: Failed to delete empty collection '{collection_name}': {del_e}")
             return None # Return None as collection cannot be populated

        # Embedding process (progress bar, batching, etc.)
        progress_bar = st.progress(0)
        status_text = st.empty()
        batch_size = 100
        num_batches = (len(chunks) + batch_size - 1) // batch_size

        for i in range(0, len(chunks), batch_size):
            current_batch_num = (i // batch_size) + 1
            status_text.text(f"Embedding batch {current_batch_num}/{num_batches}...")

            batch_chunks = chunks[i:i+batch_size]
            batch_ids = [f"chunk_{j}" for j in range(i, i + len(batch_chunks))]
            valid_chunks = []
            valid_ids = []
            embeddings = []

            for chunk_idx, chunk_text_item in enumerate(batch_chunks):
                if chunk_text_item and chunk_text_item.strip():
                    try:
                        embedding = get_embedding(chunk_text_item, client=client)
                        valid_chunks.append(chunk_text_item)
                        valid_ids.append(batch_ids[chunk_idx])
                        embeddings.append(embedding)
                    except Exception as embed_err:
                        st.error(f"Error embedding chunk {batch_ids[chunk_idx]}: {embed_err}. Skipping this chunk.")
                        print(f"ERROR: Failed to embed chunk {batch_ids[chunk_idx]}: {traceback.format_exc()}")
                else:
                    print(f"WARN: Skipping empty or whitespace-only chunk at index {i + chunk_idx}.")

            if not valid_chunks:
                print(f"WARN: Batch {current_batch_num} had no valid chunks to add.")
                continue

            try:
                collection.add(
                    documents=valid_chunks,
                    embeddings=embeddings,
                    ids=valid_ids
                )
            except Exception as add_err:
                 st.error(f"Error adding batch {current_batch_num} to Chroma: {add_err}. Some chunks may be missing.")
                 print(f"ERROR: Failed to add batch {current_batch_num} to Chroma: {traceback.format_exc()}")

            progress = min(float(i + batch_size) / len(chunks), 1.0)
            progress_bar.progress(progress)

        status_text.text(f"Embedding complete. Collection '{collection_name}' populated.")
        progress_bar.empty()
        return collection # Return the newly created and populated collection

    except Exception as create_err:
        # This catches errors during create_collection OR during the embedding process
        # Check if the error message indicates the collection already exists
        # (The exact error type/message might vary slightly between Chroma versions)
        err_str = str(create_err).lower()
        if "already exists" in err_str or "duplicate" in err_str:
            st.warning(f"Collection '{collection_name}' already exists, but initial load failed. Retrying load...")
            print(f"WARN: create_collection failed because collection exists ('{create_err}'). Retrying get_collection.")
            # If creation failed because it already exists, try GETTING it again.
            try:
                collection = chroma_client.get_collection(collection_name)
                print(f"Successfully loaded existing collection '{collection_name}' on retry.")
                print(f"Successfully loaded collection '{collection_name}' on retry.")
                return collection
            except Exception as retry_get_err:
                st.error(f"Fatal Error: Failed to load existing collection '{collection_name}' even on retry: {retry_get_err}")
                print(f"ERROR: Failed to get collection '{collection_name}' on retry: {traceback.format_exc()}")
                return None # Failed to get it even after knowing it exists
        else:
            # Different error during creation or embedding
            st.error(f"Fatal Error during collection creation or population: {create_err}")
            print(f"ERROR: Failed during create/populate for {collection_name}: {traceback.format_exc()}")
            return None # Return None for other creation/population errors

# -------------------------------
# Streamlit App Main Function
# -------------------------------

def main():
    st.set_page_config(layout="wide", page_title="GolfGPT: a Q&A Chatbot")
    st.title("🏌️‍♂️ GolfGPT: a Q&A Chatbot ⛳")
    
    # <<< REVISED CSS BLOCK >>>
    st.markdown("""
    <style>
        /* Target the main container for each chat message */
        div[data-testid="stChatMessage"] {
            border: 2px solid #007bff !important; /* Blue border - added !important */
            border-radius: 10px !important; /* Ensure radius applies */
            padding: 10px 15px !important; /* Ensure padding applies */
            margin-bottom: 10px !important; /* Ensure margin applies */
            background-color: transparent !important; /* Override any potential background */
        }

        /* Target the markdown container *within* the chat message */
        div[data-testid="stChatMessage"] > div[data-testid="stMarkdownContainer"] {
             /* This targets the direct child markdown container */
        }

        /* Target paragraphs within the markdown container */
        div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] p {
            color: #0056b3 !important; /* Darker blue text color */
            line-height: 1.5; /* Optional: Adjust line spacing if needed */
        }

        /* Target list items within the markdown container */
        div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] li {
            color: #0056b3 !important; /* Darker blue text color for lists */
        }

        /* Ensure code blocks inside messages don't get the blue text if needed */
        div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] code {
             color: inherit !important; /* Or set a specific code color */
        }
        div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] pre code {
             color: inherit !important; /* Or set a specific code block color */
        }

    </style>
    """, unsafe_allow_html=True)
    # <<< END REVISED CSS BLOCK >>>

    # Wrap ALL initialization in a single spinner for better UX
    with st.spinner("Initializing the GolfGPT Chatbot... Please wait."):
        # --- OpenAI Client Initialization ---
        if "openai_client" not in st.session_state:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                st.error("🔴 FATAL ERROR: OPENAI_API_KEY environment variable not set!")
                print("ERROR: OPENAI_API_KEY not found in environment variables.")
                st.info("Please set the OPENAI_API_KEY environment variable and restart.")
                st.stop()
            else:
                try:
                    st.session_state.openai_client = OpenAI(api_key=openai_api_key)
                except Exception as e:
                    st.error(f"🔴 FATAL ERROR: Failed to initialize OpenAI client: {e}")
                    print(f"ERROR: OpenAI client initialization failed: {traceback.format_exc()}")
                    st.stop()

        if "openai_client" not in st.session_state or st.session_state.openai_client is None:
             st.error("OpenAI client is not available. Cannot proceed.")
             st.stop()
        client = st.session_state.openai_client

        # --- PDF Processing and Chunking ---
        pdf_path = "GolfForDummies.3rd.2006.pdf"
        if "chunks" not in st.session_state:
            if not os.path.exists(pdf_path):
                st.error(f"🔴 FATAL ERROR: PDF file not found at path: {pdf_path}")
                print(f"ERROR: PDF file does not exist at {pdf_path}")
                st.info(f"Please ensure '{os.path.basename(pdf_path)}' is in the same directory as the script.")
                st.stop()

            pdf_text = parse_pdf(pdf_path)
            if pdf_text is not None and pdf_text.strip():
                st.session_state.chunks = chunk_text(pdf_text)
                if not st.session_state.chunks:
                    st.error("PDF was parsed, but chunking resulted in zero chunks.")
                    print("ERROR: Chunking returned empty list even though PDF text was present.")
                    st.session_state.chunks = []
            else:
                st.error("🔴 FATAL ERROR: PDF parsing failed or returned no text.")
                print("ERROR: PDF parsing returned None or empty text.")
                st.session_state.chunks = []
                st.stop()

        # --- ChromaDB Vector Store Initialization ---
        if "collection" not in st.session_state:
            if "chunks" not in st.session_state or not st.session_state.chunks:
                 st.error("🔴 Cannot initialize vector store: No text chunks available.")
                 print("ERROR: Attempted to initialize ChromaDB without valid chunks.")
                 st.stop()

            collection = get_or_create_chroma_collection(
                st.session_state.chunks,
                client=client
            )
            if collection is not None:
                st.session_state.collection = collection
            else:
                st.error("🔴 FATAL ERROR: Failed to get or create ChromaDB collection. Check logs for details.")
                print("ERROR: get_or_create_chroma_collection returned None.")
                st.stop()

        if "collection" not in st.session_state or st.session_state.collection is None:
             st.error("Vector Store collection is not available. Cannot proceed.")
             st.stop()

        # --- Initialize chat history and suggestion flag ---
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "initial_prompts_shown" not in st.session_state:
            st.session_state.initial_prompts_shown = False

    # --- Initialization Complete ---
    st.success("GolfGPT Chatbot is ready!")
    st.write("Welcome! Learn to Play Golf by Asking Questions.")

    # --- Sidebar Guidance (No changes) ---
    st.sidebar.title("Example Golf Topics")
    st.sidebar.markdown("""
    You can ask about topics like:
    *   **Golf Swing** (Stance, Grip, Driver, Irons)
    *   **Teeing** the ball up
    *   **Putting** techniques
    *   Common **Faults & Easy Fixes** (e.g., slicing, hooking)
    *   **Rules**, **Etiquette**, and **Scoring**
    *   Golf’s Ten Basic **Commandments**
    """)

    # --- Chat Interface Logic ---

    # --- Initial Suggestion Buttons ---
    suggested_prompts = [
        "How should I grip my golf clubs?",
        "Explain the basic golf stance.",
        "Why do I slice the ball?",
        "What are the rules for putting?",
        "Tell me about golf etiquette.",
        "What are Golf's Ten Commandments?"
    ]

    # <<< MODIFICATION 3a: Initialize clicked_prompt_text >>>
    clicked_prompt_text = None # Initialize here to store button text if clicked

    if not st.session_state.chat_history and not st.session_state.get('initial_prompts_shown', False):
        st.markdown("---")
        st.markdown("**Need ideas? Try asking one of these:**")
        cols = st.columns(3)
        button_clicked = False # Flag to know if *any* button was clicked

        for i, prompt_text in enumerate(suggested_prompts):
            col_index = i % 3
            with cols[col_index]:
                # Check if *this specific* button is clicked
                if st.button(prompt_text, key=f"suggestion_{i}"):
                    button_clicked = True
                    clicked_prompt_text = prompt_text # Store the text of the clicked button
                    st.session_state.initial_prompts_shown = True # Mark that suggestions were used

        # <<< MODIFICATION 3b: Remove the immediate st.rerun() >>>
        # We will now process the clicked_prompt_text directly below

    st.markdown("---") # Separator

    # --- Display chat history ---
    # This loop runs on every interaction, drawing messages from history
    for speaker, message in st.session_state.chat_history:
         display_name = "AI" if speaker.lower() == "chatbot" else speaker
         user_avatar_to_display = "👨" if speaker.lower() == "user" else None # Use None for AI for CSS targeting
         # Display message using role and avatar
         with st.chat_message(display_name.lower(), avatar=user_avatar_to_display):
                st.markdown(message)

    # <<< MODIFICATION 3c: Handle Input: Either from clicked button OR chat input >>>
    user_input = None # Initialize user_input for this run
    if clicked_prompt_text:
        # If a button was clicked in *this run*, use its text as input
        user_input = clicked_prompt_text
    else:
        # Otherwise (no button clicked *this run*), show the chat input box
        # This will appear on the initial load (if no buttons shown)
        # AND on subsequent runs after processing is complete and st.rerun() is called
        user_input = st.chat_input("Ask your question to learn how to play golf:")
    # <<< END MODIFICATION 3c >>>

    # --- Process input if it exists (from button or typing) ---
    if user_input:
        # 1. Add user message to history
        st.session_state.chat_history.append(("User", user_input))

        # 2. Display user message immediately for better UX
        with st.chat_message("user", avatar="👨"):
             st.markdown(user_input)

        # 3. Process query and get AI response
        with st.spinner("Searching guide and thinking..."):
            try:
                query_embedding = get_embedding(user_input, client=client)
                results = st.session_state.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=6
                )
                retrieved_docs = results['documents'][0] if results and results['documents'] and results['documents'][0] else []

                if not retrieved_docs:
                    # Use the refined message if context is empty
                    answer = "I couldn't find information about that specific topic in the provided golf guide."
                    print("WARN: No relevant documents found in ChromaDB for the query.")
                else:
                    context = "\n\n---\n\n".join(retrieved_docs)
                    # Call the MODIFIED generate_answer function
                    answer = generate_answer(user_input, context, client=client, temperature=0.2)

                # 4. Add AI response to history
                st.session_state.chat_history.append(("Chatbot", answer))

                # 5. Display AI response immediately
                with st.chat_message("ai"): # Use "ai" or "assistant" consistently
                    st.markdown(answer)

                # <<< MODIFICATION 3d: Trigger rerun AFTER processing >>>
                st.rerun() # Rerun the script to update history display and show input box again

            except openai.APIError as api_err:
                 st.error(f"An OpenAI API error occurred: {api_err}")
                 print(f"ERROR: OpenAI API Error during query processing: {traceback.format_exc()}")
                 error_message = f"Sorry, I encountered an OpenAI API error ({api_err.type}). Please try again later."
                 st.session_state.chat_history.append(("Chatbot", error_message))
                 with st.chat_message("ai"):
                     st.markdown(error_message)
                 st.rerun() # Rerun even on error
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                print(f"ERROR: Unexpected error during query processing: {traceback.format_exc()}")
                error_message = "Sorry, I encountered an unexpected error processing your request."
                st.session_state.chat_history.append(("Chatbot", error_message))
                with st.chat_message("ai"):
                    st.markdown(error_message)
                st.rerun() # Rerun even on error
            # <<< END MODIFICATION 3d >>>

    # Final separator (optional)
    # st.markdown("---")

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
         print("WARNING: OPENAI_API_KEY environment variable not set before running main(). The app might fail if it's not set elsewhere.")
    main()

