from fastapi import APIRouter, File, UploadFile, Request, HTTPException
from typing import List

from src.schemas.main_schemas import DocumentMetadata
from src.core.document_processor import DocumentProcessor
from src.core.vector_store import VectorStore
from src.core.storage import Storage
from src.channel.fastapi.config import get_settings
from .base import limiter, validate_file, logger

settings = get_settings()
router = APIRouter(prefix="/documents", tags=["documents"])

# Initialize non-async components
document_processor = DocumentProcessor(upload_dir=settings.UPLOAD_DIR)
storage = Storage(storage_dir=settings.STORAGE_DIR)

@router.post("/upload")
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
        
        # Initialize vector store
        vector_store = await VectorStore.create(storage=storage)
        
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

@router.get("", response_model=List[DocumentMetadata])
@limiter.limit("30/minute")
async def get_documents(request: Request):
    """Get all uploaded documents"""
    try:
        documents = await storage.list_documents()
        return documents
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its chunks."""
    try:
        # Initialize vector store
        vector_store = await VectorStore.create(storage=storage)
        
        # Delete from storage and vector store
        await storage.delete_document(document_id)
        await vector_store.delete_document(document_id)
        
        return {"status": "success", "message": f"Document {document_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 