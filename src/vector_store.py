"""
Vector Store Module
===================
Provides ChromaDB-based vector storage with simulation fallback.

Architecture:
- ChromaDB for production vector similarity search
- In-memory simulation for testing and development
- Cosine similarity metric for semantic search

Design Patterns:
- Adapter: Abstracts ChromaDB vs simulation
- Strategy: Pluggable embedding models
"""

import logging
import random
from typing import List, Optional

import numpy as np

from .llm import REAL_LLM_AVAILABLE, get_embeddings

logger = logging.getLogger(__name__)

# Try to import ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not installed. Vector store will use simulation.")


class SimpleVectorStore:
    """
    Vector store using ChromaDB with simulation fallback.

    Similarity Metric: Cosine similarity
    Formula: sim(A, B) = (A · B) / (||A|| × ||B||)

    ChromaDB Configuration:
    - In-memory storage (no persistence by default)
    - HNSW index for approximate nearest neighbor
    - Cosine distance metric

    Usage:
        store = SimpleVectorStore()
        store.add(["chunk1", "chunk2"], embeddings)
        results = store.similarity_search("query", k=3)
    """

    def __init__(
        self,
        use_real: bool = None,
        collection_name: str = "context_lab",
        reset: bool = True
    ):
        """
        Initialize vector store.

        Args:
            use_real: Force ChromaDB (True) or simulation (False)
            collection_name: ChromaDB collection name
            reset: Reset collection to avoid dimension conflicts
        """
        if use_real is None:
            use_real = REAL_LLM_AVAILABLE and CHROMADB_AVAILABLE

        self.use_real = use_real
        self.chunks: List[str] = []
        self.embeddings_list: List[np.ndarray] = []
        self.collection = None
        self.embedding_dimension: Optional[int] = None

        if use_real and CHROMADB_AVAILABLE:
            try:
                # Initialize ChromaDB in-memory client
                self.client = chromadb.Client(Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                ))

                # Reset collection if requested
                if reset:
                    try:
                        self.client.delete_collection(name=collection_name)
                        logger.debug(f"Deleted existing collection: {collection_name}")
                    except Exception:
                        pass

                # Create fresh collection
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )

                logger.info("ChromaDB vector store initialized")

            except Exception as e:
                logger.warning(f"ChromaDB initialization failed: {e}")
                self.use_real = False
        else:
            logger.info("Using simulated vector store")

    def add(
        self,
        chunks: List[str],
        embeddings: Optional[List[np.ndarray]] = None
    ) -> None:
        """
        Add chunks and their embeddings to the store.

        If embeddings are not provided, they will be generated using:
        1. Ollama embeddings (if available)
        2. Sentence Transformers (fallback)
        3. Random vectors (simulation)

        Args:
            chunks: Text chunks to store
            embeddings: Pre-computed embedding vectors (optional)
        """
        self.chunks.extend(chunks)

        if self.use_real and self.collection is not None:
            try:
                # Generate embeddings if not provided
                if embeddings is None:
                    embeddings = self._generate_embeddings(chunks)

                # Generate unique IDs
                start_id = len(self.chunks) - len(chunks)
                ids = [f"doc_{i}" for i in range(start_id, start_id + len(chunks))]

                # Add to ChromaDB
                self.collection.add(
                    embeddings=embeddings,
                    documents=chunks,
                    ids=ids
                )

                logger.debug(f"Added {len(chunks)} chunks to ChromaDB")

            except Exception as e:
                logger.warning(f"Failed to add to ChromaDB: {e}")
                # Fallback to simulation
                if embeddings is None:
                    embeddings = [np.random.randn(384).tolist() for _ in chunks]
                self.embeddings_list.extend(embeddings)
        else:
            # Simulation mode
            if embeddings is None:
                embeddings = [np.random.randn(384).tolist() for _ in chunks]
            self.embeddings_list.extend(embeddings)

    def _generate_embeddings(self, chunks: List[str]) -> List:
        """Generate embeddings for chunks using available models."""
        embeddings_model = get_embeddings()

        if embeddings_model:
            try:
                return embeddings_model.embed_documents(chunks)
            except Exception as e:
                logger.warning(f"Ollama embeddings failed: {e}")

        # Try sentence transformers
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model.encode(chunks).tolist()
        except Exception as e:
            logger.warning(f"Sentence transformers failed: {e}")

        # Fallback to random
        return [np.random.randn(384).tolist() for _ in chunks]

    def similarity_search(self, query: str, k: int = 3) -> List[str]:
        """
        Return top-k most similar chunks to query.

        Algorithm:
        1. Embed query using same model as storage
        2. Compute cosine similarity with all stored embeddings
        3. Return k highest scoring chunks

        Args:
            query: Search query text
            k: Number of results to return

        Returns:
            List of most similar chunk texts
        """
        if not self.chunks:
            return []

        if self.use_real and self.collection is not None:
            try:
                # Detect Hebrew for consistent embedding
                is_hebrew = any('\u0590' <= c <= '\u05FF' for c in query)

                if is_hebrew:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                    query_embedding = model.encode(query, show_progress_bar=False).tolist()
                else:
                    embeddings_model = get_embeddings()
                    if embeddings_model:
                        query_embedding = embeddings_model.embed_query(query)
                    else:
                        from sentence_transformers import SentenceTransformer
                        model = SentenceTransformer('all-MiniLM-L6-v2')
                        query_embedding = model.encode(query, show_progress_bar=False).tolist()

                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(k, len(self.chunks))
                )

                if results['documents']:
                    logger.debug(f"Retrieved {len(results['documents'][0])} chunks")
                    return results['documents'][0]

                return []

            except Exception as e:
                logger.warning(f"ChromaDB query failed: {e}")

        # Simulation: random sampling
        logger.debug(f"Simulation: returning {k} random chunks")
        return random.sample(self.chunks, min(k, len(self.chunks)))

    def __len__(self) -> int:
        """Return number of stored chunks."""
        return len(self.chunks)

    def clear(self) -> None:
        """Clear all stored data."""
        self.chunks = []
        self.embeddings_list = []

        if self.collection is not None:
            try:
                # Delete and recreate collection
                name = self.collection.name
                self.client.delete_collection(name)
                self.collection = self.client.create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("Vector store cleared")
            except Exception as e:
                logger.warning(f"Failed to clear ChromaDB: {e}")
