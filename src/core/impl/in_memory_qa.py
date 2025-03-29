from fastapi import UploadFile
from src.core.impl.document_processor import DocumentProcessor
from src.core.impl.qa_engine import QAEngine
from src.core.impl.vector_store import VectorStore
from src.core.api.dtos import QuestionResponse


class InMemoryQA:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.qa_engine = QAEngine()
        self.vector_store = None  # Will be initialized asynchronously

    @classmethod
    async def create(cls) -> "InMemoryQA":
        """Create and initialize a new InMemoryQA instance."""
        instance = cls()
        instance.vector_store = await VectorStore.create()
        return instance

    async def ensure_initialized(self):
        """Ensure the vector store is initialized."""
        if self.vector_store is None:
            self.vector_store = await VectorStore.create()

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
        await self.ensure_initialized()

        # Read file content
        file_content = await file.read()

        # Process document in memory
        document_metadata, text = (
            await self.document_processor.process_document_in_memory(
                file_content=file_content, filename=file.filename
            )
        )

        # Chunk the document
        chunks = self.document_processor.chunk_document(text)

        # Update document IDs in chunks
        for chunk in chunks:
            chunk.document_id = document_metadata.document_id

        # Add new chunks to vector store
        await self.vector_store.add_chunks(chunks)

        # Search for relevant chunks using vector store
        relevant_chunks = await self.vector_store.search(question, k=max_documents)

        # Get answer from QA engine
        response = await self.qa_engine.answer_question(
            question=question,
            relevant_chunks=[chunk for chunk, _ in relevant_chunks],
            conversation_id=conversation_id,
        )

        return response
