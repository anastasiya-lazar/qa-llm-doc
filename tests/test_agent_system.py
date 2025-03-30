import pytest
from unittest.mock import patch, MagicMock
from src.core.impl.agent_system import AgentSystem
from src.core.api.dtos import ComplexQueryRequest, DocumentChunk
from src.core.impl.llm.factory import LLMProvider


class TestAgentSystem:
    @pytest.fixture
    def agent_system(self):
        with patch("src.core.impl.agent_system.LLMFactory") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm_factory.create_llm.return_value = mock_llm
            return AgentSystem(LLMProvider.OPENAI)

    @pytest.fixture
    def sample_chunks(self):
        return [
            DocumentChunk(
                chunk_id="chunk1",
                document_id="doc1",
                content="This is a test chunk",
                metadata={"page": 1},
            ),
            DocumentChunk(
                chunk_id="chunk2",
                document_id="doc1",
                content="This is another test chunk",
                metadata={"page": 2},
            ),
        ]

    @pytest.fixture
    def sample_request(self):
        return ComplexQueryRequest(
            query="What is the capital of France?",
            agent_types=["researcher", "writer"],
            conversation_id="test-conv-1",
        )

    def test_convert_chunks_to_references(self, agent_system, sample_chunks):
        references = agent_system._convert_chunks_to_references(sample_chunks)

        assert len(references) == len(sample_chunks)
        for ref, chunk in zip(references, sample_chunks):
            assert ref.document_id == chunk.document_id
            assert ref.chunk_id == chunk.chunk_id
            assert ref.content == chunk.content
            assert ref.metadata == chunk.metadata
            assert ref.similarity_score == 1.0
