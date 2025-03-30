import pytest
from datetime import datetime
from src.core.impl.storage import Storage
from src.core.api.dtos import DocumentMetadata, DocumentChunk


@pytest.fixture
def storage(tmp_path):
    storage_dir = tmp_path / "test_storage"
    storage_dir.mkdir()
    return Storage(str(storage_dir))


@pytest.fixture
def sample_metadata():
    return DocumentMetadata(
        document_id="test_doc_1",
        filename="test.pdf",
        file_type="pdf",
        document_type="pdf",
        upload_timestamp=datetime.now(),
        chunk_count=2,
    )


@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            chunk_id="chunk_1",
            document_id="test_doc_1",
            content="This is the first chunk",
            metadata={"page": 1},
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            document_id="test_doc_1",
            content="This is the second chunk",
            metadata={"page": 2},
        ),
    ]
