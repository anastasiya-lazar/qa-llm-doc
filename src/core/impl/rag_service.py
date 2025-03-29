from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.impl.document_processor import DocumentProcessor
from src.core.impl.in_memory_qa import InMemoryQA
from src.core.impl.storage import Storage
from src.core.impl.vector_store import VectorStore


class RAGService:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = None
        self.qa = None
        self.storage = Storage()

    async def initialize(self):
        """Initialize the RAG service components."""
        self.vector_store = await VectorStore.create(storage=self.storage)
        self.qa = await InMemoryQA.create(vector_store=self.vector_store)

    async def process_and_retrieve(
        self, file_content: bytes, filename: str, query: str, top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Process a document in memory and retrieve relevant chunks for a query.

        Args:
            file_content: The document content as bytes
            filename: Name of the file
            query: The query to find relevant chunks for
            top_k: Number of most relevant chunks to return

        Returns:
            Dictionary containing:
            - document_metadata: Metadata about the processed document
            - relevant_chunks: List of relevant document chunks
            - processing_time: Time taken to process and retrieve
        """
        start_time = datetime.now()

        # Process document in memory
        document_metadata, text = (
            await self.document_processor.process_document_in_memory(
                file_content=file_content, filename=filename
            )
        )

        # Chunk the document
        chunks = self.document_processor.chunk_document(text)

        # Update document IDs in chunks
        for chunk in chunks:
            chunk.document_id = document_metadata.document_id

        # Get relevant chunks using vector store
        relevant_chunks = await self.vector_store.get_relevant_chunks(
            query=query, chunks=chunks, top_k=top_k
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        return {
            "document_metadata": document_metadata,
            "relevant_chunks": relevant_chunks,
            "processing_time": processing_time,
        }

    async def answer_question(
        self, question: str, document_ids: Optional[List[str]] = None
    ) -> str:
        """Answer a question using the RAG system."""
        if not self.qa:
            await self.initialize()
        return await self.qa.answer_question(question, document_ids)
