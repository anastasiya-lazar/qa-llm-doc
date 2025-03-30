import pytest
from unittest.mock import patch
from src.core.impl.document_processor import DocumentProcessor
from src.core.api.dtos import DocumentType, DocumentMetadata


class TestDocumentProcessor:
    @pytest.fixture
    def document_processor(self):
        return DocumentProcessor()

    @pytest.mark.parametrize(
        "filename,expected_type",
        [
            ("test.txt", DocumentType.TXT),
            ("test.unknown", ValueError),
            ("", ValueError),
        ],
    )
    def test_get_document_type(self, document_processor, filename, expected_type):
        if isinstance(expected_type, type) and issubclass(expected_type, Exception):
            with pytest.raises(expected_type):
                document_processor._get_document_type(filename)
        else:
            doc_type = document_processor._get_document_type(filename)
            assert doc_type == expected_type

    @pytest.mark.parametrize(
        "file_content,filename,expected_metadata",
        [
            (
                b"Test TXT content",
                "test.txt",
                DocumentMetadata(
                    document_id="test_id",
                    filename="test.txt",
                    document_type=DocumentType.TXT,
                    page_count=1,
                    word_count=3,
                    chunk_count=0,
                ),
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_process_document_in_memory(
        self, document_processor, file_content, filename, expected_metadata
    ):
        with patch("uuid.uuid4", return_value="test_id"):
            metadata, text = await document_processor.process_document_in_memory(
                file_content, filename
            )
            assert metadata.document_id == expected_metadata.document_id
            assert metadata.filename == expected_metadata.filename
            assert metadata.document_type == expected_metadata.document_type
            assert metadata.page_count == expected_metadata.page_count
            assert metadata.word_count == expected_metadata.word_count
            assert metadata.chunk_count == expected_metadata.chunk_count
            assert text == file_content.decode()

    @pytest.mark.parametrize(
        "text,chunk_size,overlap,expected_chunks",
        [
            ("Short text", 20, 0, ["Short text"]),
        ],
    )
    def test_chunk_document(
        self, document_processor, text, chunk_size, overlap, expected_chunks
    ):
        with patch("uuid.uuid4", return_value="test-chunk-id"):
            chunks = document_processor.chunk_document(text, chunk_size, overlap)
            assert len(chunks) == len(expected_chunks)
            for chunk, expected in zip(chunks, expected_chunks):
                assert chunk.content == expected
                assert chunk.chunk_id == "test-chunk-id"
                assert chunk.document_id == ""
                assert chunk.metadata == {"word_count": len(expected.split())}
