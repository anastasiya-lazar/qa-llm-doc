from fastapi import UploadFile

from src.core.api.dtos import QuestionResponse
from src.core.impl.document_processor import DocumentProcessor
from src.core.impl.logging_config import setup_logger
from src.core.impl.qa_engine import QAEngine
from src.core.impl.vector_store import VectorStore

logger = setup_logger("in_memory_qa")


class InMemoryQA:
    def __init__(self):
        logger.info("Initializing InMemoryQA")
        self.document_processor = DocumentProcessor()
        self.qa_engine = QAEngine()
        self.vector_store = None  # Will be initialized asynchronously
        logger.info("InMemoryQA initialized successfully")

    @classmethod
    async def create(cls) -> "InMemoryQA":
        """Create and initialize a new InMemoryQA instance."""
        logger.info("Creating new InMemoryQA instance")
        instance = cls()
        instance.vector_store = await VectorStore.create()
        logger.info("Vector store initialized for new InMemoryQA instance")
        return instance

    async def ensure_initialized(self):
        """Ensure the vector store is initialized."""
        if self.vector_store is None:
            logger.info("Vector store not initialized, initializing now")
            self.vector_store = await VectorStore.create()
            logger.info("Vector store initialized successfully")

    async def process_and_answer(
        self,
        file: UploadFile,
        question: str,
        conversation_id: str = None,
        max_documents: int = 5,
    ) -> QuestionResponse:
        """
        Process a file in memory and answer a question about it.

        Args:
            file: The uploaded file
            question: The question to answer
            conversation_id: Optional conversation ID for tracking
            max_documents: Maximum number of relevant documents to retrieve

        Returns:
            QuestionResponse containing the answer and source documents
        """
        logger.info(f"Processing file: {file.filename}")
        logger.debug(f"Question: {question}")
        if conversation_id:
            logger.debug(f"Conversation ID: {conversation_id}")
        logger.debug(f"Max documents: {max_documents}")

        await self.ensure_initialized()

        # Read file content
        logger.debug("Reading file content")
        file_content = await file.read()

        # Process document in memory
        logger.debug("Processing document in memory")
        document_metadata, text = (
            await self.document_processor.process_document_in_memory(
                file_content=file_content, filename=file.filename
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

        # Add new chunks to vector store
        logger.debug("Adding chunks to vector store")
        await self.vector_store.add_chunks(chunks)
        logger.info(f"Added {len(chunks)} chunks to vector store")

        # Search for relevant chunks using vector store
        logger.debug("Searching for relevant chunks")
        relevant_chunks = await self.vector_store.search(question, k=max_documents)
        logger.info(f"Found {len(relevant_chunks)} relevant chunks")

        # Get answer from QA engine
        logger.debug("Getting answer from QA engine")
        response = await self.qa_engine.answer_question(
            question=question,
            relevant_chunks=[chunk for chunk, _ in relevant_chunks],
            conversation_id=conversation_id,
        )
        logger.info("Successfully generated answer")

        return response
