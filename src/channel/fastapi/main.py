from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
import uvicorn
from datetime import datetime
import logging
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis
import magic
import aiofiles
from pathlib import Path
from pydantic import BaseModel

from src.schemas.main_schemas import (
    QuestionRequest,
    QuestionResponse,
    ComplexQueryRequest,
    ComplexQueryResponse,
    DocumentMetadata
)
from src.core.document_processor import DocumentProcessor
from src.core.vector_store import VectorStore
from src.core.qa_engine import QAEngine
from src.core.agent_system import AgentSystem
from src.core.storage import Storage
from src.channel.fastapi.config import get_settings
from src.core.in_memory_qa import InMemoryQA

# Load environment variables
load_dotenv()
settings = get_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Redis for rate limiting
redis_client = redis.from_url(settings.REDIS_URL)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address, storage_uri=settings.REDIS_URL)
limiter.exceeded_handler = _rate_limit_exceeded_handler

# Initialize components
document_processor = DocumentProcessor(upload_dir=settings.UPLOAD_DIR)
storage = Storage(storage_dir=settings.STORAGE_DIR)
vector_store = VectorStore(storage=storage)
qa_engine = QAEngine()
agent_system = AgentSystem()
in_memory_qa = InMemoryQA()

# Request models
class RAGRetrieveRequest(BaseModel):
    query: str
    document_ids: List[str]
    max_documents: int = 5

# Initialize FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI-powered document question answering system",
    version="1.0.0"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.state.limiter = limiter

# Error handling
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests"}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# File validation
async def validate_file(file: UploadFile) -> bool:
    """Validate file type and size"""
    # Check file size
    file_size = 0
    content = await file.read()
    file_size = len(content)
    await file.seek(0)  # Reset file pointer
    
    if file_size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE} bytes"
        )
    
    # Check file type
    mime = magic.Magic(mime=True)
    file_type = mime.from_buffer(content)
    file_extension = file.filename.split('.')[-1].lower()
    
    if file_extension not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"File type not allowed. Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    
    return True

# API Endpoints
@app.get("/")
@limiter.limit("5/minute")
async def root(request: Request):
    """Health check endpoint"""
    return {"status": "ok", "message": settings.PROJECT_NAME}

@app.post("/documents/upload")
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
):
    """Upload and process a document file (PDF, TXT, etc.)"""
    try:
        logger.info(f"Processing document upload: {file.filename}")
        
        # Validate file
        await validate_file(file)
        
        # Read file content
        content = await file.read()
        
        # Process document
        metadata, text = await document_processor.process_document(content, file.filename)
        
        # Chunk document
        chunks = document_processor.chunk_document(text)
        
        # Update chunk document IDs
        for chunk in chunks:
            chunk.document_id = metadata.document_id
        
        # Save to storage
        await storage.save_document(metadata)
        await storage.save_chunks(metadata.document_id, chunks)
        
        # Add to vector store
        await vector_store.add_chunks(chunks)
        
        logger.info(f"Document processed successfully: {metadata.document_id}")
        return {
            "status": "success",
            "message": "Document processed successfully",
            "document_id": metadata.document_id,
            "chunk_count": len(chunks)
        }
    except Exception as e:
        logger.error(f"Error processing document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
@limiter.limit("30/minute")
async def get_documents(request: Request):
    """Get all uploaded documents"""
    try:
        documents = await storage.list_documents()
        return {
            "status": "success",
            "documents": [doc.dict() for doc in documents]
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/questions/ask", response_model=QuestionResponse)
@limiter.limit("20/minute")
async def ask_question(
    request: Request,
    question_request: QuestionRequest,
):
    """Ask a question about uploaded documents"""
    try:
        logger.info(f"Processing question: {question_request.question}")
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

@app.post("/agents/complex-query", response_model=ComplexQueryResponse)
@limiter.limit("10/minute")
async def complex_query(
    request: Request,
    query_request: ComplexQueryRequest,
):
    """Process a complex query using multiple agents"""
    try:
        logger.info(f"Processing complex query: {query_request.query}")
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

@app.post("/qa/in-memory", response_model=QuestionResponse)
@limiter.limit("20/minute")
async def process_and_answer_in_memory(
    request: Request,
    file: UploadFile = File(...),
    question: str = Form(...),
    conversation_id: str = Form(None)
):
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
    except ValueError as e:
        logger.error(f"Document processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error processing in-memory QA request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error processing request"
        )

@app.post("/rag/retrieve")
@limiter.limit("20/minute")
async def retrieve_from_source(
    request: Request,
    rag_request: RAGRetrieveRequest
):
    """
    Retrieve relevant chunks from existing documents and generate an answer.
    
    Args:
        rag_request: RAGRetrieveRequest containing:
            - query: The query to find relevant chunks for
            - document_ids: List of document IDs to search in
            - max_documents: Maximum number of relevant chunks to return
        
    Returns:
        Dictionary containing:
        - answer: The generated answer
        - relevant_chunks: List of relevant document chunks
        - processing_time: Time taken to retrieve and answer
    """
    try:
        start_time = datetime.now()
        
        # Get chunks from storage
        chunks = await storage.get_chunks_by_document_ids(rag_request.document_ids)
        
        # Get relevant chunks using vector store
        relevant_chunks = await vector_store.get_relevant_chunks(
            query=rag_request.query,
            chunks=chunks,
            top_k=rag_request.max_documents
        )
        
        # Generate answer using QA engine
        answer_response = await qa_engine.answer_question(
            question=rag_request.query,
            relevant_chunks=relevant_chunks
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "answer": answer_response.answer,
            "relevant_chunks": relevant_chunks,
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Error retrieving from source: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error retrieving from source"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )