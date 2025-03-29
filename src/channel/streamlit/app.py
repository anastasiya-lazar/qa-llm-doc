import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import List

import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the page
st.set_page_config(page_title="Document QA System", page_icon="📚", layout="wide")

# Constants
API_URL = os.getenv("API_URL", "http://localhost:8000")
UPLOAD_DIR = "uploads"

# Initialize session state for conversation management
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = str(uuid.uuid4())
    st.session_state.conversations[st.session_state.current_conversation_id] = (
        "New Conversation"
    )


def upload_document(file) -> bool:
    """Upload document to the API"""
    try:
        files = {"file": file}
        response = requests.post(f"{API_URL}/documents/upload", files=files)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error uploading document: {str(e)}")
        return False


def get_documents() -> List[dict]:
    """Get list of uploaded documents"""
    try:
        response = requests.get(f"{API_URL}/documents")
        if response.status_code == 200:
            data = response.json()
            return data.get("documents", [])
        return []
    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")
        return []


def ask_question(
    question: str, mode: str, uploaded_file=None, selected_documents: List[str] = None
) -> dict:
    """Send question to the API based on RAG mode"""
    try:
        # Set conversation name based on first question if it's still "New Conversation"
        if (
            st.session_state.conversations[st.session_state.current_conversation_id]
            == "New Conversation"
        ):
            # Truncate question to 50 characters for the conversation name
            conversation_name = (
                question[:50] + "..." if len(question) > 50 else question
            )
            st.session_state.conversations[st.session_state.current_conversation_id] = (
                conversation_name
            )

        if mode == "in_memory":
            if not uploaded_file:
                st.error("Please upload a document for in-memory RAG")
                return {}

            files = {"file": uploaded_file}
            data = {
                "question": question,
                "conversation_id": st.session_state.current_conversation_id,
            }
            response = requests.post(f"{API_URL}/qa/in-memory", files=files, data=data)
        else:  # source-based
            # If no documents selected, use all available documents
            if not selected_documents:
                documents = get_documents()
                selected_documents = [doc["document_id"] for doc in documents]
                if not selected_documents:
                    st.error("No documents available for searching")
                    return {}

            data = {
                "question": question,
                "document_ids": selected_documents,
                "max_documents": 5,
                "conversation_id": st.session_state.current_conversation_id,
            }
            response = requests.post(f"{API_URL}/questions/ask", json=data)

        return response.json() if response.status_code == 200 else {}
    except Exception as e:
        st.error(f"Error asking question: {str(e)}")
        return {}


def create_new_conversation():
    """Create a new conversation"""
    new_id = str(uuid.uuid4())
    st.session_state.conversations[new_id] = "New Conversation"
    st.session_state.current_conversation_id = new_id


def main():
    st.title("📚 Document QA System")
    st.write("Upload documents and ask questions about them!")

    # Sidebar for document upload and selection
    with st.sidebar:
        st.header("Conversations")

        # New conversation button
        if st.button("New Conversation"):
            create_new_conversation()

        # Conversation selector
        selected_conversation = st.selectbox(
            "Select Conversation",
            options=list(st.session_state.conversations.keys()),
            format_func=lambda x: st.session_state.conversations[x],
            key="conversation_selector",
        )

        if selected_conversation != st.session_state.current_conversation_id:
            st.session_state.current_conversation_id = selected_conversation

        st.divider()

        st.header("Upload Documents")
        uploaded_file = st.file_uploader(
            "Choose a document", type=["pdf", "txt", "doc", "docx"], key="file_uploader"
        )

        if uploaded_file:
            if st.button("Upload"):
                with st.spinner("Uploading..."):
                    if upload_document(uploaded_file):
                        st.success("Document uploaded successfully!")
                    else:
                        st.error("Failed to upload document")

        st.header("Uploaded Documents")
        documents = get_documents()
        for doc in documents:
            st.write(f"- {doc['filename']}")

    # Main content area
    st.header("Ask Questions")

    # Mode selection
    mode = st.radio(
        "Select Mode",
        ["in_memory", "source_based"],
        help="In-memory: Process and query a single document. Source-based: Query across multiple uploaded documents.",
    )

    # Document selection for source-based
    selected_documents = []
    if mode == "source_based" and documents:
        st.subheader("Select Documents to Query")
        selected_documents = st.multiselect(
            "Choose documents to search in",
            options=[doc["document_id"] for doc in documents],
            format_func=lambda x: next(
                (doc["filename"] for doc in documents if doc["document_id"] == x), x
            ),
        )

    # Question input
    question = st.text_input("Enter your question:", key="question_input")

    if st.button("Ask Question"):
        if not question:
            st.warning("Please enter a question")
        else:
            with st.spinner("Processing your question..."):
                response = ask_question(
                    question=question,
                    mode=mode,
                    uploaded_file=uploaded_file if mode == "in_memory" else None,
                    selected_documents=(
                        selected_documents if mode == "source_based" else None
                    ),
                )
                if response:
                    st.subheader("Answer:")
                    st.write(response.get("answer", "No answer available"))

                    # Display sources if available
                    if "source_documents" in response:
                        st.subheader("Sources:")
                        for source in response["source_documents"]:
                            st.write(
                                f"- {source.get('content', 'No content available')}"
                            )
                else:
                    st.error("Failed to get answer")


if __name__ == "__main__":
    main()
