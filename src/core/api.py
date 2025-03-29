from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body
from typing import Optional, List
from src.core.in_memory_qa import InMemoryQA
from src.core.rag_service import RAGService
from src.schemas.main_schemas import QuestionResponse, DocumentChunk, DocumentMetadata, QuestionRequest
from src.core.vector_store import VectorStore
from src.core.storage import Storage
from datetime import datetime

router = APIRouter()
in_memory_qa = None  # Will be initialized asynchronously
rag_service = RAGService()
storage = Storage()
vector_store = None  # Will be initialized asynchronously

@router.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global in_memory_qa, vector_store
    in_memory_qa = await InMemoryQA.create()
    vector_store = await VectorStore.create(storage=storage)

@router.post("/qa/in-memory", response_model=QuestionResponse)
async def process_and_answer(
    file: UploadFile = File(...),
    question: str = Form(...),
    conversation_id: Optional[str] = Form(None)
) -> QuestionResponse:
    """
    Process a file in memory and answer a question about it.
    
    Args:
        file: The uploaded file (PDF or TXT)
        question: The question to answer
        conversation_id: Optional conversation ID for tracking
        
    Returns:
        QuestionResponse containing the answer and source documents
    """
    try:
        # Validate file
        if not await validate_file(file):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type or size"
            )
            
        return await in_memory_qa.process_and_answer(
            file=file,
            question=question,
            conversation_id=conversation_id
        )
    except Exception as e:
        logger.error(f"Error processing in-memory QA request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error processing request"
        )


@router.get("/documents", response_model=List[DocumentMetadata])
async def list_documents():
    """Get a list of all available documents."""
    try:
        documents = await storage.list_documents()
        return documents
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error listing documents"
        )

@router.post("/questions/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest
) -> QuestionResponse:
    """
    Ask a question about previously uploaded documents.
    
    Args:
        request: QuestionRequest containing the question and optional parameters
        
    Returns:
        QuestionResponse containing the answer and source documents
    """
    try:
        # Ensure services are initialized
        await in_memory_qa.ensure_initialized()
        await vector_store.ensure_initialized()
        
        # Search for relevant chunks using vector store
        relevant_chunks = await vector_store.search(
            request.question,
            k=request.max_documents
        )
        
        # Get answer from QA engine
        response = await in_memory_qa.qa_engine.answer_question(
            question=request.question,
            relevant_chunks=[chunk for chunk, _ in relevant_chunks],
            conversation_id=request.conversation_id
        )
        
        return response
    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error processing question"
        ) 