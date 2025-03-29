from fastapi import APIRouter, File, UploadFile, Request, HTTPException, Form
from typing import Optional

from src.schemas.main_schemas import QuestionRequest, QuestionResponse, ComplexQueryRequest, ComplexQueryResponse
from src.core.vector_store import VectorStore
from src.core.qa_engine import QAEngine
from src.core.agent_system import AgentSystem
from src.core.storage import Storage
from src.core.in_memory_qa import InMemoryQA
from src.channel.fastapi.config import get_settings
from .base import limiter, validate_file, logger

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
        search_results = await vector_store.search(question_request.question, k=question_request.max_documents)
        relevant_chunks = [chunk for chunk, _ in search_results]
        
        # Generate answer
        response = await qa_engine.answer_question(
            question=question_request.question,
            relevant_chunks=relevant_chunks,
            conversation_id=question_request.conversation_id
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
            request=query_request,
            relevant_chunks=relevant_chunks
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
    conversation_id: Optional[str] = Form(None)
):
    """
    Process a file in memory and answer a question about it.
    """
    try:
        logger.info(f"Processing in-memory question: {question}")
        
        # Validate file
        await validate_file(file)
        
        # Initialize vector store and in-memory QA
        vector_store = await VectorStore.create(storage=storage)
        in_memory_qa = await InMemoryQA.create(vector_store=vector_store)
        
        # Process and answer
        response = await in_memory_qa.process_and_answer(
            file=file,
            question=question,
            conversation_id=conversation_id
        )
        
        logger.info(f"In-memory question answered successfully")
        return response
    except Exception as e:
        logger.error(f"Error processing in-memory question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 