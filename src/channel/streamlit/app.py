import streamlit as st
import os
from pathlib import Path
import tempfile
from typing import List
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the page
st.set_page_config(
    page_title="Document QA System",
    page_icon="📚",
    layout="wide"
)

# Constants
API_URL = os.getenv("API_URL", "http://localhost:8000")
UPLOAD_DIR = "uploads"

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

def ask_question(question: str, rag_mode: str, uploaded_file=None, selected_documents: List[str] = None) -> dict:
    """Send question to the API based on RAG mode"""
    try:
        if rag_mode == "in_memory":
            if not uploaded_file:
                st.error("Please upload a document for in-memory RAG")
                return {}
            
            files = {"file": uploaded_file}
            data = {"question": question}
            response = requests.post(
                f"{API_URL}/qa/in-memory",
                files=files,
                data=data
            )
        else:  # source-based
            if not selected_documents:
                st.error("Please select at least one document for source-based RAG")
                return {}
            
            data = {
                "query": question,
                "document_ids": selected_documents,
                "max_documents": 5
            }
            response = requests.post(
                f"{API_URL}/rag/retrieve",
                json=data
            )
        
        return response.json() if response.status_code == 200 else {}
    except Exception as e:
        st.error(f"Error asking question: {str(e)}")
        return {}

def main():
    st.title("📚 Document QA System")
    st.write("Upload documents and ask questions about them!")

    # Sidebar for document upload and selection
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=["pdf", "txt", "doc", "docx"],
            key="file_uploader"
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
    
    # RAG mode selection
    rag_mode = st.radio(
        "Select RAG Mode",
        ["in_memory", "source_based"],
        help="In-memory: Process and query a single document. Source-based: Query across multiple uploaded documents."
    )
    
    # Document selection for source-based RAG
    selected_documents = []
    if rag_mode == "source_based" and documents:
        st.subheader("Select Documents to Query")
        selected_documents = st.multiselect(
            "Choose documents to search in",
            options=[doc["document_id"] for doc in documents],
            format_func=lambda x: next((doc["filename"] for doc in documents if doc["document_id"] == x), x)
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
                    rag_mode=rag_mode,
                    uploaded_file=uploaded_file if rag_mode == "in_memory" else None,
                    selected_documents=selected_documents if rag_mode == "source_based" else None
                )
                if response:
                    st.subheader("Answer:")
                    st.write(response.get("answer", "No answer available"))
                    
                    # Display sources if available
                    if "source_documents" in response:
                        st.subheader("Sources:")
                        for source in response["source_documents"]:
                            st.write(f"- {source.get('content', 'No content available')}")
                else:
                    st.error("Failed to get answer")

if __name__ == "__main__":
    main() 