from typing import List, Dict, Any, Optional
import json
from pathlib import Path
from datetime import datetime

from src.core.api.dtos import DocumentMetadata, DocumentChunk


class Storage:
    def __init__(self, storage_dir: str = "storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_dir / "metadata.json"
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load document metadata from storage."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def _save_metadata(self) -> None:
        """Save document metadata to storage."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

    async def save_document(self, metadata: DocumentMetadata) -> None:
        """Save document metadata to storage."""
        # Convert to dict and ensure datetime is properly serialized
        metadata_dict = metadata.dict()
        if isinstance(metadata_dict.get("upload_timestamp"), datetime):
            metadata_dict["upload_timestamp"] = metadata_dict[
                "upload_timestamp"
            ].isoformat()
        self.metadata[metadata.document_id] = metadata_dict
        self._save_metadata()

    async def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """Retrieve document metadata from storage."""
        if document_id in self.metadata:
            doc_data = self.metadata[document_id].copy()
            # Convert ISO format string back to datetime
            if "upload_timestamp" in doc_data and isinstance(
                doc_data["upload_timestamp"], str
            ):
                doc_data["upload_timestamp"] = datetime.fromisoformat(
                    doc_data["upload_timestamp"]
                )
            return DocumentMetadata(**doc_data)
        return None

    async def list_documents(self) -> List[DocumentMetadata]:
        """List all documents in storage."""
        documents = []
        for doc in self.metadata.values():
            doc_data = doc.copy()
            # Convert ISO format string back to datetime
            if "upload_timestamp" in doc_data and isinstance(
                doc_data["upload_timestamp"], str
            ):
                doc_data["upload_timestamp"] = datetime.fromisoformat(
                    doc_data["upload_timestamp"]
                )
            documents.append(DocumentMetadata(**doc_data))
        return documents

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from storage."""
        if document_id in self.metadata:
            del self.metadata[document_id]
            self._save_metadata()
            return True
        return False

    async def save_chunks(self, document_id: str, chunks: List[DocumentChunk]) -> None:
        """Save document chunks to storage."""
        chunks_file = self.storage_dir / f"{document_id}_chunks.json"
        chunks_data = [chunk.dict() for chunk in chunks]

        with open(chunks_file, "w") as f:
            json.dump(chunks_data, f, indent=2, default=str)

        # Update document metadata with chunk count
        if document_id in self.metadata:
            self.metadata[document_id]["chunk_count"] = len(chunks)
            self._save_metadata()

    async def get_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Retrieve document chunks from storage."""
        chunks_file = self.storage_dir / f"{document_id}_chunks.json"
        if chunks_file.exists():
            with open(chunks_file, "r") as f:
                chunks_data = json.load(f)
                return [DocumentChunk(**chunk) for chunk in chunks_data]
        return []

    async def get_chunks_by_document_ids(
        self, document_ids: List[str]
    ) -> List[DocumentChunk]:
        """Retrieve chunks from multiple documents."""
        all_chunks = []
        for doc_id in document_ids:
            chunks = await self.get_chunks(doc_id)
            all_chunks.extend(chunks)
        return all_chunks

    async def update_document_metadata(
        self, document_id: str, updates: Dict[str, Any]
    ) -> Optional[DocumentMetadata]:
        """Update document metadata."""
        if document_id in self.metadata:
            self.metadata[document_id].update(updates)
            self._save_metadata()
            return DocumentMetadata(**self.metadata[document_id])
        return None
