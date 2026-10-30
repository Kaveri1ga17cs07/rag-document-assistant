import sys

import streamlit as st
import os
# Tell Python to include the parent root directory when searching for modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil
# Bypassing FastAPI network layer to communicate directly with your engine
from backend.rag_engine import process_pdf, answer_query

st.set_page_config(page_title="PDF RAG Conversational Assistant", layout="wide")
st.title("📚 Interactive RAG PDF Assistant")

# Define a secure localized path inside the cloud runtime container
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- SIDEBAR: Document ingestion panel ---
with st.sidebar:
    st.header("Document Operations Manager")
    uploaded_file = st.file_uploader("Upload target document (PDF)", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Ingest Document Into RAG Engine"):
            with st.spinner("Processing PDF fragments and creating vector indices..."):
                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                
                # Write file directly to local server workspace disk space
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(uploaded_file, buffer)
                
                try:
                    # Invoke your core engine logic function directly
                    chunks_created = process_pdf(file_path)
                    st.sidebar.success(f"Ingested {uploaded_file.name} into {chunks_created} shards.")
                except Exception as e:
                    st.sidebar.error(f"Ingestion Error: {str(e)}")

# --- MAIN PAGE: Conversational interface UI loop ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Document uploaded? Ask me questions below!"}]

# Output previous chat timeline array structures
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user inputs
if user_query := st.chat_input("Enter your message based on your uploaded document text..."):
    # Append user question to history UI
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Fire the contextual question payload over to the engine directly
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Analyzing document context..."):
            try:
                # Direct Python invocation bypassing API network requests entirely
                ai_answer = answer_query(user_query)
                message_placeholder.markdown(ai_answer)
                st.session_state.messages.append({"role": "assistant", "content": ai_answer})
            except Exception as e:
                message_placeholder.error(f"Query Inference Error: {str(e)}")
