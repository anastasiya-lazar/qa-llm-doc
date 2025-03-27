from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    OTHER = "other"


class DocumentMetadata(BaseModel):
    """Metadata for an uploaded document"""
    document_id: str
    filename: str
    document_type: DocumentType
    upload_timestamp: datetime = Field(default_factory=datetime.now)
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    chunk_count: int = 0
    user_id: Optional[str] = None


class DocumentChunk(BaseModel):
    """A chunk of text from a document with its metadata"""
    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]
    embedding_id: Optional[int] = None


class QuestionRequest(BaseModel):
    """Request model for asking a question"""
    question: str
    document_ids: Optional[List[str]] = None
    conversation_id: Optional[str] = None
    max_documents: int = Field(default=5, ge=1, le=20)


class DocumentReference(BaseModel):
    """Reference to a source document"""
    document_id: str
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: Optional[float] = None


class QuestionResponse(BaseModel):
    """Response model for a question"""
    answer: str
    source_documents: List[DocumentReference]
    processing_time: float
    conversation_id: Optional[str] = None


class ComplexQueryRequest(BaseModel):
    """Request model for a complex query requiring multiple agents"""
    query: str
    document_ids: Optional[List[str]] = None
    conversation_id: Optional[str] = None
    agent_types: List[str] = ["researcher", "writer"]
    max_iterations: int = Field(default=3, ge=1, le=10)


class AgentAction(BaseModel):
    """Record of an action taken by an agent"""
    agent_name: str
    action: str
    input: Any
    output: Any
    timestamp: datetime = Field(default_factory=datetime.now)


class ComplexQueryResponse(BaseModel):
    """Response model for a complex query"""
    answer: str
    source_documents: List[DocumentReference]
    processing_time: float
    conversation_id: Optional[str] = None
    agent_actions: List[AgentAction]