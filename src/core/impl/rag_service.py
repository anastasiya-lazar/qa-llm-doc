from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.impl.document_processor import DocumentProcessor
from src.core.impl.in_memory_qa import InMemoryQA
from src.core.impl.logging_config import setup_logger
from src.core.impl.storage import Storage
from src.core.impl.vector_store import VectorStore

logger = setup_logger("rag_service")


class RAGService:
    def __init__(self):
        logger.info("Initializing RAGService")
        self.document_processor = DocumentProcessor()
        self.vector_store = None
        self.qa = None
        self.storage = Storage()
        logger.info("RAGService initialized successfully")

    async def initialize(self):
        """Initialize the RAG service components."""
        logger.info("Initializing RAG service components")
        self.vector_store = await VectorStore.create(storage=self.storage)
        logger.info("Vector store initialized")
        self.qa = await InMemoryQA.create(vector_store=self.vector_store)
        logger.info("QA system initialized")

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
        logger.info(f"Processing and retrieving information for file: {filename}")
        logger.debug(f"Query: {query}, top_k: {top_k}")
        start_time = datetime.now()

        # Process document in memory
        logger.debug("Processing document in memory")
        document_metadata, text = (
            await self.document_processor.process_document_in_memory(
                file_content=file_content, filename=filename
            )
        )
        logger.info(f"Document processed successfully: {document_metadata.document_id}")

        # Chunk the document
        logger.debug("Chunking document")
        chunks = self.document_processor.chunk_document(text)
        logger.info(f"Created {len(chunks)} chunks from document")

        # Update document IDs in chunks
        logger.debug("Updating document IDs in chunks")
        for chunk in chunks:
            chunk.document_id = document_metadata.document_id

        # Get relevant chunks using vector store
        logger.debug("Retrieving relevant chunks using vector store")
        relevant_chunks = await self.vector_store.get_relevant_chunks(
            query=query, chunks=chunks, top_k=top_k
        )
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")

        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Processing and retrieval completed in {processing_time:.2f} seconds"
        )

        return {
            "document_metadata": document_metadata,
            "relevant_chunks": relevant_chunks,
            "processing_time": processing_time,
        }

    async def answer_question(
        self, question: str, document_ids: Optional[List[str]] = None
    ) -> str:
        """Answer a question using the RAG system."""
        logger.info("Processing question")
        logger.debug(f"Question: {question}")
        if document_ids:
            logger.debug(f"Document IDs: {document_ids}")

        if not self.qa:
            logger.info("QA system not initialized, initializing now")
            await self.initialize()

        logger.debug("Getting answer from QA system")
        answer = await self.qa.answer_question(question, document_ids)
        logger.info("Successfully generated answer")
        return answer
