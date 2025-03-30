import pytest
import os
from unittest.mock import MagicMock, AsyncMock, patch
from src.core.impl.llm.openai_connector import OpenAIConnector
from src.core.impl.vector_store import VectorStore
from src.core.impl.document_processor import DocumentProcessor
from src.core.impl.rag_service import RAGService
from src.core.impl.qa_engine import QAEngine
from src.core.impl.agent_system import AgentSystem
import redis.asyncio as redis


@pytest.fixture(autouse=True)
def mock_openai_api_key():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        yield


@pytest.fixture
def mock_openai():
    mock = AsyncMock(spec=OpenAIConnector)
    mock.generate_response = AsyncMock(return_value="Test response")
    mock.count_tokens = MagicMock(return_value=5)
    mock.truncate_text = MagicMock(return_value="Test text")
    return mock


@pytest.fixture
def mock_vector_store():
    mock = AsyncMock(spec=VectorStore)
    mock.search = AsyncMock(
        return_value=[{"content": "Test content", "metadata": {"source": "test.pdf"}}]
    )
    mock.get_embedding = AsyncMock(return_value=[0.1] * 1536)
    mock.add_documents = MagicMock()
    mock.get_relevant_chunks = AsyncMock(
        return_value=[{"content": "Test content", "metadata": {"source": "test.pdf"}}]
    )
    return mock


@pytest.fixture
def mock_document_processor():
    mock = AsyncMock(spec=DocumentProcessor)
    mock.process_document_in_memory = AsyncMock(
        return_value=(
            {
                "document_id": "test-doc",
                "filename": "test.pdf",
                "file_type": "pdf",
                "created_at": "2024-01-01T00:00:00",
            },
            "Test document content",
        )
    )
    mock.chunk_document = MagicMock(
        return_value=[
            {"content": "Test chunk 1", "metadata": {"page": 1}},
            {"content": "Test chunk 2", "metadata": {"page": 2}},
        ]
    )
    return mock


@pytest.fixture
def mock_rag_service(mock_vector_store, mock_document_processor):
    mock = AsyncMock(spec=RAGService)
    mock.vector_store = mock_vector_store
    mock.document_processor = mock_document_processor
    mock.qa = AsyncMock()
    mock.qa.answer_question = AsyncMock(return_value="Test answer")
    return mock


@pytest.fixture
def mock_qa_engine():
    mock = AsyncMock(spec=QAEngine)
    mock.qa_chain = AsyncMock()
    mock.qa_chain.arun = AsyncMock(return_value="Test answer")
    mock.llm = AsyncMock()
    mock.llm.arun = AsyncMock(return_value="Follow-up 1?\nFollow-up 2?\nFollow-up 3?")
    mock.conversation_history = {}
    mock.conversation_documents = {}
    return mock


@pytest.fixture
def mock_agent_system():
    mock = AsyncMock(spec=AgentSystem)
    mock.process_complex_query = AsyncMock(
        return_value={
            "answer": "Test response",
            "source_documents": [],
            "processing_time": 0.1,
            "conversation_id": "test-conv-1",
            "agent_actions": [],
        }
    )
    return mock


@pytest.fixture
def mock_redis():
    mock = AsyncMock(spec=redis.Redis)
    mock.get = AsyncMock(return_value=b"test_value")
    mock.set = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=1)
    mock.exists = AsyncMock(return_value=1)
    mock.expire = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def test_env():
    os.environ["OPENAI_API_KEY"] = "test-key"
    os.environ["REDIS_URL"] = "redis://localhost:6379/0"
    return os.environ
