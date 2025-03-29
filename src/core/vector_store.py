from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
import os
from pathlib import Path
import pickle
import json

from src.schemas.main_schemas import DocumentChunk
from src.core.storage import Storage

class VectorStore:
    def __init__(self, dimension: int = 1536, storage: Storage = None):  # OpenAI embeddings dimension
        self.dimension = dimension
        self.embeddings_model = OpenAIEmbeddings()
        self.chunks: List[DocumentChunk] = []
        self.storage = storage or Storage()
        self.index_dir = Path(self.storage.storage_dir) / "indices"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store = None
        self._load_chunks_from_storage()

    @classmethod
    def load_local(
        cls,
        path: str,
        embeddings_model: Optional[OpenAIEmbeddings] = None,
        storage: Optional[Storage] = None,
        allow_dangerous_deserialization: bool = False
    ) -> 'VectorStore':
        """
        Load a VectorStore from a local directory.
        
        Args:
            path: Path to the directory containing the vector store files
            embeddings_model: Optional embeddings model to use
            storage: Optional storage instance to use
            allow_dangerous_deserialization: Whether to allow deserialization of potentially dangerous data
            
        Returns:
            A new VectorStore instance
        """
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Directory {path} does not exist")
            
        # Create instance
        instance = cls(
            dimension=1536,  # OpenAI embeddings dimension
            storage=storage
        )
        
        # Load FAISS index using LangChain's implementation
        instance.vector_store = FAISS.load_local(
            path,
            embeddings_model or instance.embeddings_model,
            allow_dangerous_deserialization=allow_dangerous_deserialization
        )
        
        return instance

    def save_local(self, path: str) -> None:
        """
        Save the VectorStore to a local directory.
        
        Args:
            path: Path to save the vector store files
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save")
        self.vector_store.save_local(path)

    async def _load_chunks_from_storage(self):
        """Load all chunks from storage and rebuild the vector index."""
        try:
            # Try to load existing index
            if (self.index_dir / "index.faiss").exists():
                self.vector_store = FAISS.load_local(
                    str(self.index_dir),
                    self.embeddings_model
                )
                print("Loaded existing FAISS index")
                return

            # If no index exists, load from storage and build new index
            documents = await self.storage.list_documents()
            all_chunks = []
            for doc in documents:
                chunks = await self.storage.get_chunks(doc.document_id)
                all_chunks.extend(chunks)
            
            if all_chunks:
                await self.add_chunks(all_chunks)
                print(f"Built new index with {len(all_chunks)} chunks")
        except Exception as e:
            print(f"Error loading chunks from storage: {e}")

    async def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to the vector store."""
        if not chunks:
            return

        # Convert chunks to LangChain documents
        documents = [
            Document(
                page_content=chunk.content,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    **chunk.metadata
                }
            )
            for chunk in chunks
        ]

        # Create or update FAISS index
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings_model)
        else:
            self.vector_store.add_documents(documents)

        # Update chunks list
        self.chunks.extend(chunks)

        # Save the updated index
        self.save_local(str(self.index_dir))

    async def search(self, query: str, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks using a query string."""
        if not self.chunks:
            # Try to load chunks if none exist
            await self._load_chunks_from_storage()
            if not self.chunks:
                return []

        # Adjust k if it's larger than the number of chunks
        k = min(k, len(self.chunks))
        
        # Search using LangChain's implementation
        docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Convert to our format
        results = []
        for doc, score in docs_and_scores:
            chunk_id = doc.metadata.get("chunk_id")
            chunk = next((c for c in self.chunks if c.chunk_id == chunk_id), None)
            if chunk:
                results.append((chunk, 1 / (1 + score)))  # Convert distance to similarity score

        return results

    def get_chunk_by_id(self, chunk_id: str) -> DocumentChunk:
        """Retrieve a chunk by its ID."""
        return next((chunk for chunk in self.chunks if chunk.chunk_id == chunk_id), None)

    def get_chunks_by_document_id(self, document_id: str) -> List[DocumentChunk]:
        """Retrieve all chunks belonging to a specific document."""
        return [chunk for chunk in self.chunks if chunk.document_id == document_id]

    async def delete_document(self, document_id: str) -> None:
        """Delete all chunks belonging to a specific document."""
        # Get chunks to remove
        chunks_to_remove = [chunk for chunk in self.chunks if chunk.document_id == document_id]
        if not chunks_to_remove:
            return

        # Remove chunks from our list
        self.chunks = [chunk for chunk in self.chunks if chunk.document_id != document_id]

        # Rebuild the vector store with remaining chunks
        if self.chunks:
            documents = [
                Document(
                    page_content=chunk.content,
                    metadata={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        **chunk.metadata
                    }
                )
                for chunk in self.chunks
            ]
            self.vector_store = FAISS.from_documents(documents, self.embeddings_model)
        else:
            self.vector_store = None

        # Save the updated index
        if self.vector_store:
            self.save_local(str(self.index_dir))

    async def get_relevant_chunks(
        self,
        query: str,
        chunks: List[DocumentChunk],
        top_k: int = 5
    ) -> List[DocumentChunk]:
        """
        Get the most relevant chunks from a list of chunks for a given query.
        
        Args:
            query: The search query
            chunks: List of chunks to search in
            top_k: Number of most relevant chunks to return
            
        Returns:
            List of most relevant chunks
        """
        if not chunks:
            return []

        # Create a temporary vector store for these chunks
        documents = [
            Document(
                page_content=chunk.content,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    **chunk.metadata
                }
            )
            for chunk in chunks
        ]
        
        temp_store = FAISS.from_documents(documents, self.embeddings_model)
        
        # Search in the temporary store
        docs = temp_store.similarity_search(query, k=min(top_k, len(chunks)))
        
        # Convert back to chunks
        relevant_chunks = []
        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id")
            chunk = next((c for c in chunks if c.chunk_id == chunk_id), None)
            if chunk:
                relevant_chunks.append(chunk)

        return relevant_chunks 