import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.api.dtos import DocumentChunk, DocumentMetadata
from src.core.impl.logging_config import setup_logger

logger = setup_logger("storage")


class Storage:
    def __init__(self, storage_dir: str = "storage"):
        logger.info(f"Initializing Storage with directory: {storage_dir}")
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_dir / "metadata.json"
        self._load_metadata()
        logger.info("Storage initialized successfully")

    def _load_metadata(self) -> None:
        """Load document metadata from storage."""
        if self.metadata_file.exists():
            logger.info("Loading existing metadata file")
            with open(self.metadata_file, "r") as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded metadata for {len(self.metadata)} documents")
        else:
            logger.info("No metadata file found, initializing empty metadata")
            self.metadata = {}

    def _save_metadata(self) -> None:
        """Save document metadata to storage."""
        logger.debug("Saving metadata to file")
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)
        logger.debug("Metadata saved successfully")

    async def save_document(self, metadata: DocumentMetadata) -> None:
        """Save document metadata to storage."""
        logger.info(f"Saving document metadata for document_id: {metadata.document_id}")
        # Convert to dict and ensure datetime is properly serialized
        metadata_dict = metadata.dict()
        if isinstance(metadata_dict.get("upload_timestamp"), datetime):
            metadata_dict["upload_timestamp"] = metadata_dict[
                "upload_timestamp"
            ].isoformat()
        self.metadata[metadata.document_id] = metadata_dict
        self._save_metadata()
        logger.info(
            f"Successfully saved metadata for document_id: {metadata.document_id}"
        )

    async def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """Retrieve document metadata from storage."""
        logger.debug(f"Retrieving document metadata for document_id: {document_id}")
        if document_id in self.metadata:
            doc_data = self.metadata[document_id].copy()
            # Convert ISO format string back to datetime
            if "upload_timestamp" in doc_data and isinstance(
                doc_data["upload_timestamp"], str
            ):
                doc_data["upload_timestamp"] = datetime.fromisoformat(
                    doc_data["upload_timestamp"]
                )
            logger.debug(f"Found document metadata for document_id: {document_id}")
            return DocumentMetadata(**doc_data)
        logger.warning(f"Document metadata not found for document_id: {document_id}")
        return None

    async def list_documents(self) -> List[DocumentMetadata]:
        """List all documents in storage."""
        logger.info("Listing all documents")
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
        logger.info(f"Found {len(documents)} documents")
        return documents

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from storage."""
        logger.info(f"Attempting to delete document: {document_id}")
        if document_id in self.metadata:
            del self.metadata[document_id]
            self._save_metadata()
            logger.info(f"Successfully deleted document: {document_id}")
            return True
        logger.warning(f"Document not found for deletion: {document_id}")
        return False

    async def save_chunks(self, document_id: str, chunks: List[DocumentChunk]) -> None:
        """Save document chunks to storage."""
        logger.info(f"Saving {len(chunks)} chunks for document: {document_id}")
        chunks_file = self.storage_dir / f"{document_id}_chunks.json"
        chunks_data = [chunk.dict() for chunk in chunks]

        with open(chunks_file, "w") as f:
            json.dump(chunks_data, f, indent=2, default=str)

        # Update document metadata with chunk count
        if document_id in self.metadata:
            self.metadata[document_id]["chunk_count"] = len(chunks)
            self._save_metadata()
            logger.info(f"Updated chunk count for document: {document_id}")
        else:
            logger.warning(
                f"Document metadata not found while saving chunks: {document_id}"
            )

    async def get_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Retrieve document chunks from storage."""
        logger.debug(f"Retrieving chunks for document: {document_id}")
        chunks_file = self.storage_dir / f"{document_id}_chunks.json"
        if chunks_file.exists():
            with open(chunks_file, "r") as f:
                chunks_data = json.load(f)
                chunks = [DocumentChunk(**chunk) for chunk in chunks_data]
                logger.debug(f"Found {len(chunks)} chunks for document: {document_id}")
                return chunks
        logger.warning(f"No chunks found for document: {document_id}")
        return []

    async def get_chunks_by_document_ids(
        self, document_ids: List[str]
    ) -> List[DocumentChunk]:
        """Retrieve chunks from multiple documents."""
        logger.info(f"Retrieving chunks for {len(document_ids)} documents")
        all_chunks = []
        for doc_id in document_ids:
            chunks = await self.get_chunks(doc_id)
            all_chunks.extend(chunks)
        logger.info(f"Retrieved total of {len(all_chunks)} chunks")
        return all_chunks

    async def update_document_metadata(
        self, document_id: str, updates: Dict[str, Any]
    ) -> Optional[DocumentMetadata]:
        """Update document metadata."""
        logger.info(f"Updating metadata for document: {document_id}")
        if document_id in self.metadata:
            self.metadata[document_id].update(updates)
            self._save_metadata()
            logger.info(f"Successfully updated metadata for document: {document_id}")
            return DocumentMetadata(**self.metadata[document_id])
        logger.warning(f"Document not found for metadata update: {document_id}")
        return None
