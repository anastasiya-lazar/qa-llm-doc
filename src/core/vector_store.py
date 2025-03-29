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
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
        return cls._instance

    def __init__(self, dimension: int = 1536, storage: Storage = None):  # OpenAI embeddings dimension
        if not self._initialized:
            self.dimension = dimension
            self.embeddings_model = OpenAIEmbeddings()
            self.chunks: List[DocumentChunk] = []
            self.storage = storage or Storage()
            self.index_dir = Path(self.storage.storage_dir) / "indices"
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self.vector_store = None
            self._initialized = True

    @classmethod
    async def create(cls, dimension: int = 1536, storage: Storage = None) -> 'VectorStore':
        """Create and initialize a new VectorStore instance."""
        instance = cls(dimension=dimension, storage=storage)
        await instance._initialize()
        return instance

    async def _initialize(self):
        """Initialize the vector store by loading chunks and building/loading the index."""
        if hasattr(self, '_store_initialized') and self._store_initialized:
            return

        try:
            # Try to load existing index
            if (self.index_dir / "index.faiss").exists():
                self.vector_store = FAISS.load_local(
                    str(self.index_dir),
                    self.embeddings_model,
                    allow_dangerous_deserialization=True  # Allow loading our own index
                )
                print("Loaded existing FAISS index")
                
                # Load chunks from storage to maintain the chunks list
                documents = await self.storage.list_documents()
                all_chunks = []
                for doc in documents:
                    chunks = await self.storage.get_chunks(doc.document_id)
                    all_chunks.extend(chunks)
                
                if all_chunks:
                    self.chunks = all_chunks
                    print(f"Loaded {len(all_chunks)} chunks from storage")
                
                self._store_initialized = True
                return

            # If no index exists, load from storage and build new index
            documents = await self.storage.list_documents()
            all_chunks = []
            for doc in documents:
                chunks = await self.storage.get_chunks(doc.document_id)
                all_chunks.extend(chunks)
            
            if all_chunks:
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
                    for chunk in all_chunks
                ]

                # Create new FAISS index
                self.vector_store = FAISS.from_documents(documents, self.embeddings_model)
                self.chunks = all_chunks
                print(f"Built new index with {len(all_chunks)} chunks")
                
                # Save the index
                self.save_local(str(self.index_dir))
            else:
                print("No chunks found in storage")
                # Initialize empty vector store
                self.vector_store = FAISS.from_texts(
                    [""], self.embeddings_model
                )
            
            self._store_initialized = True
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def save_local(self, path: str) -> None:
        """Save the vector store to a local directory."""
        if self.vector_store is None:
            raise ValueError("No vector store to save")
        self.vector_store.save_local(path)

    async def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to the vector store."""
        if not chunks:
            return

        # Ensure initialized
        await self._initialize()

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

        # Add to existing index or create new one
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
        # Ensure initialized
        await self._initialize()

        if not self.chunks:
            print("No chunks available for search")
            return []

        # Adjust k if it's larger than the number of chunks
        k = min(k, len(self.chunks))
        
        print(f"Searching for {k} most relevant chunks among {len(self.chunks)} total chunks")
        
        # Search using LangChain's implementation
        docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Convert to our format
        results = []
        for doc, score in docs_and_scores:
            chunk_id = doc.metadata.get("chunk_id")
            chunk = next((c for c in self.chunks if c.chunk_id == chunk_id), None)
            if chunk:
                results.append((chunk, 1 / (1 + score)))  # Convert distance to similarity score
                print(f"Found chunk {chunk_id} with similarity score {1 / (1 + score)}")
            else:
                print(f"Warning: Could not find chunk {chunk_id} in chunks list")

        print(f"Found {len(results)} relevant chunks")
        return results

    def get_chunk_by_id(self, chunk_id: str) -> DocumentChunk:
        """Retrieve a chunk by its ID."""
        return next((chunk for chunk in self.chunks if chunk.chunk_id == chunk_id), None)

    def get_chunks_by_document_id(self, document_id: str) -> List[DocumentChunk]:
        """Retrieve all chunks belonging to a specific document."""
        return [chunk for chunk in self.chunks if chunk.document_id == document_id]

    async def delete_document(self, document_id: str) -> None:
        """Delete all chunks belonging to a specific document."""
        # Ensure initialized
        await self._initialize()

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
            # Initialize empty vector store
            self.vector_store = FAISS.from_texts([""], self.embeddings_model)

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