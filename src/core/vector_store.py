from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from src.schemas.main_schemas import DocumentChunk
from src.core.storage import Storage

class VectorStore:
    def __init__(self, dimension: int = 1536, storage: Storage = None):  # OpenAI embeddings dimension
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.embeddings_model = OpenAIEmbeddings()
        self.chunks: List[DocumentChunk] = []
        self.chunk_to_id: Dict[str, int] = {}  # Maps chunk_id to FAISS index
        self.storage = storage or Storage()
        self._load_chunks_from_storage()

    async def _load_chunks_from_storage(self):
        """Load all chunks from storage and rebuild the vector index."""
        try:
            # Get all documents
            documents = await self.storage.list_documents()
            
            # Load chunks for each document
            all_chunks = []
            for doc in documents:
                chunks = await self.storage.get_chunks(doc.document_id)
                all_chunks.extend(chunks)
            
            if all_chunks:
                # Add chunks to vector store
                await self.add_chunks(all_chunks)
                print(f"Loaded {len(all_chunks)} chunks from storage")
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

        # Generate embeddings
        embeddings = await self.embeddings_model.aembed_documents(
            [doc.page_content for doc in documents]
        )

        # Add to FAISS index
        start_idx = len(self.chunks)
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Update chunk storage and mapping
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.chunk_to_id[chunk.chunk_id] = start_idx + i

    async def search(self, query: str, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks using a query string."""
        if not self.chunks:
            # Try to load chunks if none exist
            await self._load_chunks_from_storage()
            if not self.chunks:
                return []

        # Adjust k if it's larger than the number of chunks
        k = min(k, len(self.chunks))
        
        # Generate query embedding
        query_embedding = await self.embeddings_model.aembed_query(query)
        
        # Search in FAISS index
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), k
        )

        # Return chunks with their similarity scores
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):  # Ensure valid index
                chunk = self.chunks[idx]
                similarity = 1 / (1 + distance)  # Convert distance to similarity score
                results.append((chunk, similarity))

        return results

    def get_chunk_by_id(self, chunk_id: str) -> DocumentChunk:
        """Retrieve a chunk by its ID."""
        if chunk_id not in self.chunk_to_id:
            raise KeyError(f"Chunk ID {chunk_id} not found")
        return self.chunks[self.chunk_to_id[chunk_id]]

    def get_chunks_by_document_id(self, document_id: str) -> List[DocumentChunk]:
        """Retrieve all chunks belonging to a specific document."""
        return [chunk for chunk in self.chunks if chunk.document_id == document_id]

    async def delete_document(self, document_id: str) -> None:
        """Delete all chunks belonging to a specific document."""
        # Find indices to remove
        indices_to_remove = [
            self.chunk_to_id[chunk.chunk_id]
            for chunk in self.chunks
            if chunk.document_id == document_id
        ]
        
        if not indices_to_remove:
            return

        # Remove chunks and update mappings
        new_chunks = []
        new_mapping = {}
        for i, chunk in enumerate(self.chunks):
            if chunk.document_id != document_id:
                new_chunks.append(chunk)
                new_mapping[chunk.chunk_id] = i

        # Update storage
        self.chunks = new_chunks
        self.chunk_to_id = new_mapping

        # Rebuild FAISS index
        if new_chunks:
            documents = [
                Document(
                    page_content=chunk.content,
                    metadata={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        **chunk.metadata
                    }
                )
                for chunk in new_chunks
            ]
            embeddings = self.embeddings_model.embed_documents(
                [doc.page_content for doc in documents]
            )
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(np.array(embeddings).astype('float32'))
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

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

        # Generate embeddings for chunks
        chunk_embeddings = await self.embeddings_model.aembed_documents(
            [doc.page_content for doc in documents]
        )

        # Generate query embedding
        query_embedding = await self.embeddings_model.aembed_query(query)

        # Create a temporary FAISS index for these chunks
        temp_index = faiss.IndexFlatL2(self.dimension)
        temp_index.add(np.array(chunk_embeddings).astype('float32'))

        # Search in the temporary index
        k = min(top_k, len(chunks))
        distances, indices = temp_index.search(
            np.array([query_embedding]).astype('float32'), k
        )

        # Return the most relevant chunks
        relevant_chunks = []
        for idx in indices[0]:
            if idx < len(chunks):
                relevant_chunks.append(chunks[idx])

        return relevant_chunks 