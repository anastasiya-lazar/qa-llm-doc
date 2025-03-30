from pathlib import Path
from typing import List, Tuple

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from src.core.api.dtos import DocumentChunk
from src.core.impl.logging_config import setup_logger
from src.core.impl.storage import Storage

logger = setup_logger("vector_store")


class VectorStore:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
        return cls._instance

    def __init__(
        self, dimension: int = 1536, storage: Storage = None
    ):  # OpenAI embeddings dimension
        if not self._initialized:
            logger.info(f"Initializing VectorStore with dimension {dimension}")
            self.dimension = dimension
            self.embeddings_model = OpenAIEmbeddings()
            self.chunks: List[DocumentChunk] = []
            self.storage = storage or Storage()
            self.index_dir = Path(self.storage.storage_dir) / "indices"
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self.vector_store = None
            self._initialized = True
            logger.info("VectorStore initialized successfully")

    @classmethod
    async def create(
        cls, dimension: int = 1536, storage: Storage = None
    ) -> "VectorStore":
        """Create and initialize a new VectorStore instance."""
        logger.info(f"Creating new VectorStore instance with dimension {dimension}")
        instance = cls(dimension=dimension, storage=storage)
        await instance._initialize()
        return instance

    async def _initialize(self):
        """Initialize the vector store by loading chunks and building/loading the index."""
        if hasattr(self, "_store_initialized") and self._store_initialized:
            logger.debug("Vector store already initialized")
            return

        try:
            # Try to load existing index
            if (self.index_dir / "index.faiss").exists():
                logger.info("Loading existing FAISS index")
                self.vector_store = FAISS.load_local(
                    str(self.index_dir),
                    self.embeddings_model,
                    allow_dangerous_deserialization=True,  # Allow loading our own index
                )
                logger.info("Successfully loaded existing FAISS index")

                # Load chunks from storage to maintain the chunks list
                documents = await self.storage.list_documents()
                all_chunks = []
                for doc in documents:
                    chunks = await self.storage.get_chunks(doc.document_id)
                    all_chunks.extend(chunks)

                if all_chunks:
                    self.chunks = all_chunks
                    logger.info(f"Loaded {len(all_chunks)} chunks from storage")

                self._store_initialized = True
                return

            # If no index exists, load from storage and build new index
            logger.info("No existing index found, building new index")
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
                            **chunk.metadata,
                        },
                    )
                    for chunk in all_chunks
                ]

                # Create new FAISS index
                logger.info(f"Building new index with {len(all_chunks)} chunks")
                self.vector_store = FAISS.from_documents(
                    documents, self.embeddings_model
                )
                self.chunks = all_chunks
                logger.info("Successfully built new index")

                # Save the index
                self.save_local(str(self.index_dir))
            else:
                logger.warning("No chunks found in storage")
                # Initialize empty vector store
                self.vector_store = FAISS.from_texts([""], self.embeddings_model)
                logger.info("Initialized empty vector store")

            self._store_initialized = True
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise

    def save_local(self, path: str) -> None:
        """Save the vector store to a local directory."""
        if self.vector_store is None:
            logger.error("Attempted to save uninitialized vector store")
            raise ValueError("No vector store to save")
        logger.info(f"Saving vector store to {path}")
        self.vector_store.save_local(path)
        logger.info("Successfully saved vector store")

    async def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to the vector store."""
        if not chunks:
            logger.warning("Attempted to add empty chunks list")
            return

        logger.info(f"Adding {len(chunks)} new chunks to vector store")
        # Ensure initialized
        await self._initialize()

        # Convert chunks to LangChain documents
        documents = [
            Document(
                page_content=chunk.content,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    **chunk.metadata,
                },
            )
            for chunk in chunks
        ]

        # Add to existing index or create new one
        if self.vector_store is None:
            logger.info("Creating new vector store with chunks")
            self.vector_store = FAISS.from_documents(documents, self.embeddings_model)
        else:
            logger.info("Adding chunks to existing vector store")
            self.vector_store.add_documents(documents)

        # Update chunks list
        self.chunks.extend(chunks)
        logger.info(f"Updated chunks list, total chunks: {len(self.chunks)}")

        # Save the updated index
        self.save_local(str(self.index_dir))

    async def search(self, query: str, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks using a query string."""
        # Ensure initialized
        await self._initialize()

        if not self.chunks:
            logger.warning("No chunks available for search")
            return []

        # Adjust k if it's larger than the number of chunks
        k = min(k, len(self.chunks))
        logger.info(
            f"Searching for {k} most relevant chunks among {len(self.chunks)} total chunks"
        )

        # Search using LangChain's implementation
        docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)

        # Convert to our format
        results = []
        for doc, score in docs_and_scores:
            chunk_id = doc.metadata.get("chunk_id")
            chunk = next((c for c in self.chunks if c.chunk_id == chunk_id), None)
            if chunk:
                similarity_score = 1 / (1 + score)
                results.append((chunk, similarity_score))
                logger.debug(
                    f"Found chunk {chunk_id} with similarity score {similarity_score}"
                )
            else:
                logger.warning(f"Could not find chunk {chunk_id} in chunks list")

        logger.info(f"Found {len(results)} relevant chunks")
        return results

    def get_chunk_by_id(self, chunk_id: str) -> DocumentChunk:
        """Retrieve a chunk by its ID."""
        chunk = next(
            (chunk for chunk in self.chunks if chunk.chunk_id == chunk_id), None
        )
        if chunk is None:
            logger.warning(f"Chunk with ID {chunk_id} not found")
        return chunk

    def get_chunks_by_document_id(self, document_id: str) -> List[DocumentChunk]:
        """Retrieve all chunks belonging to a specific document."""
        chunks = [chunk for chunk in self.chunks if chunk.document_id == document_id]
        logger.info(f"Found {len(chunks)} chunks for document {document_id}")
        return chunks

    async def delete_document(self, document_id: str) -> None:
        """Delete all chunks belonging to a specific document."""
        # Ensure initialized
        await self._initialize()

        # Get chunks to remove
        chunks_to_remove = [
            chunk for chunk in self.chunks if chunk.document_id == document_id
        ]
        if not chunks_to_remove:
            logger.warning(f"No chunks found for document {document_id}")
            return

        logger.info(
            f"Deleting {len(chunks_to_remove)} chunks for document {document_id}"
        )
        # Remove chunks from our list
        self.chunks = [
            chunk for chunk in self.chunks if chunk.document_id != document_id
        ]

        # Rebuild the vector store with remaining chunks
        if self.chunks:
            documents = [
                Document(
                    page_content=chunk.content,
                    metadata={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        **chunk.metadata,
                    },
                )
                for chunk in self.chunks
            ]
            logger.info("Rebuilding vector store with remaining chunks")
            self.vector_store = FAISS.from_documents(documents, self.embeddings_model)
        else:
            logger.warning("No chunks remaining, initializing empty vector store")
            # Initialize empty vector store
            self.vector_store = FAISS.from_texts([""], self.embeddings_model)

        # Save the updated index
        if self.vector_store:
            self.save_local(str(self.index_dir))
            logger.info(
                "Successfully updated and saved vector store after document deletion"
            )

    async def get_relevant_chunks(
        self, query: str, chunks: List[DocumentChunk], top_k: int = 5
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
            logger.warning("Empty chunks list provided for relevance search")
            return []

        # Create a temporary vector store for these chunks
        documents = [
            Document(
                page_content=chunk.content,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    **chunk.metadata,
                },
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
