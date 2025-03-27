from typing import List
from fastapi import UploadFile
from src.core.document_processor import DocumentProcessor
from src.core.qa_engine import QAEngine
from src.schemas.main_schemas import DocumentChunk, QuestionResponse, DocumentMetadata

class InMemoryQA:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.qa_engine = QAEngine()

    async def process_and_answer(
        self,
        file: UploadFile,
        question: str,
        conversation_id: str = None
    ) -> QuestionResponse:
        """
        Process a file in memory and answer a question about it.
        
        Args:
            file: The uploaded file
            question: The question to answer
            conversation_id: Optional conversation ID for tracking
            
        Returns:
            QuestionResponse containing the answer and source documents
        """
        # Read file content
        file_content = await file.read()
        
        # Process document in memory
        document_metadata, text = await self.document_processor.process_document_in_memory(
            file_content=file_content,
            filename=file.filename
        )
        
        # Chunk the document
        chunks = self.document_processor.chunk_document(text)
        
        # Update document IDs in chunks
        for chunk in chunks:
            chunk.document_id = document_metadata.document_id
        
        # Get answer from QA engine
        response = await self.qa_engine.answer_question(
            question=question,
            relevant_chunks=chunks,
            conversation_id=conversation_id
        )
        
        return response 