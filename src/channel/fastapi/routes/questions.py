from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from src.channel.fastapi.config import get_settings
from src.core.api.dtos import (ComplexQueryRequest, ComplexQueryResponse,
                               QuestionRequest, QuestionResponse)
from src.core.impl.agent_system import AgentSystem
from src.core.impl.in_memory_qa import InMemoryQA
from src.core.impl.qa_engine import QAEngine
from src.core.impl.storage import Storage
from src.core.impl.vector_store import VectorStore

from .base import limiter, logger, validate_file

settings = get_settings()
router = APIRouter(prefix="/questions", tags=["questions"])

# Initialize non-async components
storage = Storage(storage_dir=settings.STORAGE_DIR)
qa_engine = QAEngine()
agent_system = AgentSystem()


@router.post("/ask", response_model=QuestionResponse)
@limiter.limit("20/minute")
async def ask_question(
    request: Request,
    question_request: QuestionRequest,
):
    """Ask a question about uploaded documents"""
    try:
        logger.info(f"Processing question: {question_request.question}")
        # Initialize vector store
        vector_store = await VectorStore.create(storage=storage)

        # Search for relevant chunks
        search_results = await vector_store.search(
            question_request.question, k=question_request.max_documents
        )
        relevant_chunks = [chunk for chunk, _ in search_results]

        # Generate answer
        response = await qa_engine.answer_question(
            question=question_request.question,
            relevant_chunks=relevant_chunks,
            conversation_id=question_request.conversation_id,
        )

        logger.info(f"Question answered successfully")
        return response
    except Exception as e:
        logger.error(f"Error answering question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/complex-query", response_model=ComplexQueryResponse)
@limiter.limit("10/minute")
async def complex_query(
    request: Request,
    query_request: ComplexQueryRequest,
):
    """Process a complex query using multiple agents"""
    try:
        logger.info(f"Processing complex query: {query_request.query}")
        # Initialize vector store
        vector_store = await VectorStore.create(storage=storage)

        # Search for relevant chunks
        search_results = await vector_store.search(query_request.query, k=5)
        relevant_chunks = [chunk for chunk, _ in search_results]

        # Process with agents
        response = await agent_system.process_complex_query(
            request=query_request, relevant_chunks=relevant_chunks
        )

        logger.info(f"Complex query processed successfully")
        return response
    except Exception as e:
        logger.error(f"Error processing complex query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/in-memory", response_model=QuestionResponse)
@limiter.limit("20/minute")
async def process_and_answer_in_memory(
    request: Request,
    file: UploadFile = File(...),
    question: str = Form(...),
    conversation_id: Optional[str] = Form(None),
):
    """
    Process a file in memory and answer a question about it.
    """
    try:
        logger.info(f"Processing in-memory question: {question}")

        # Validate file
        await validate_file(file)

        # Initialize vector store and in-memory QA
        in_memory_qa = await InMemoryQA.create()

        # Process and answer
        response = await in_memory_qa.process_and_answer(
            file=file, question=question, conversation_id=conversation_id
        )

        logger.info(f"In-memory question answered successfully")
        return response
    except Exception as e:
        logger.error(f"Error processing in-memory question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
