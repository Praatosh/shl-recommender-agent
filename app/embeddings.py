"""
SHL Assessment Recommender - Embedding & Vector Store
=====================================================
Handles embedding generation and FAISS index management.

Architecture:
- Uses sentence-transformers/all-MiniLM-L6-v2 (384-dim, fast, good quality)
- FAISS IndexFlatIP (inner product = cosine similarity on normalized vectors)
- Stores catalog metadata alongside FAISS index for retrieval

Design choices:
1. all-MiniLM-L6-v2: Best tradeoff between quality and speed for this use case.
   It's 5x faster than larger models and still captures semantic meaning well.
2. FAISS IndexFlatIP: For <1000 items, flat search is fast enough and exact.
   No need for IVF or HNSW at this catalog size.
3. Normalized embeddings + inner product = cosine similarity.
   This gives us a clean 0-1 similarity score.

How to improve Recall@10:
- Use a larger embedding model (e5-large-v2) at the cost of latency
- Hybrid search: combine dense retrieval with BM25 keyword matching
- Query expansion: generate multiple query variants
- Use assessment metadata (keys, job_levels) as hard filters before ranking
"""

import json
import os
import pickle
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.logger import get_logger
from app.schemas import CatalogItem

logger = get_logger("embeddings")


class VectorStore:
    """
    FAISS-based vector store for SHL assessment catalog.
    
    Responsibilities:
    - Generate embeddings for catalog items
    - Build and persist FAISS index
    - Perform semantic similarity search
    - Return catalog items with similarity scores
    """

    def __init__(self):
        self.settings = get_settings()
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.catalog_items: List[Dict[str, Any]] = []
        self._initialized = False

    def _load_model(self) -> None:
        """Lazy-load the embedding model to avoid startup delay if not needed."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.settings.embedding_model}")
            self.model = SentenceTransformer(self.settings.embedding_model)
            logger.info("Embedding model loaded successfully")

    def _create_search_text(self, item: Dict[str, Any]) -> str:
        """
        Create rich text representation of a catalog item for embedding.
        
        Why combine multiple fields?
        A user might search by:
        - Technology name ("Java", "Python")
        - Job role ("senior engineer", "entry-level")  
        - Assessment type ("personality test", "coding simulation")
        - Skill area ("leadership", "data science")
        
        By combining all these fields, a single embedding captures
        all these search dimensions.
        """
        parts = [
            f"Assessment: {item.get('name', '')}",
            f"Description: {item.get('description', '')}",
            f"Categories: {', '.join(item.get('keys', []))}",
            f"Job Levels: {', '.join(item.get('job_levels', []))}",
            f"Duration: {item.get('duration', '')}",
        ]
        return " | ".join(parts)

    def build_index(self, catalog: List[Dict[str, Any]]) -> None:
        """
        Build FAISS index from catalog items.
        
        Process:
        1. Generate search text for each item
        2. Encode all texts into embeddings
        3. Normalize embeddings (for cosine similarity via inner product)
        4. Build FAISS IndexFlatIP
        5. Store catalog metadata for retrieval
        """
        self._load_model()
        self.catalog_items = catalog

        if not catalog:
            logger.warning("Empty catalog - cannot build index")
            return

        # Generate search texts
        texts = [self._create_search_text(item) for item in catalog]
        logger.info(f"Generating embeddings for {len(texts)} catalog items...")

        # Encode to embeddings
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
            batch_size=64
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # Build FAISS index (inner product on normalized vectors = cosine similarity)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        self._initialized = True
        logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={dimension}")

    def save_index(self, path: Optional[str] = None) -> None:
        """Persist FAISS index and catalog metadata to disk."""
        path = path or self.settings.faiss_index_path
        os.makedirs(path, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        # Save catalog metadata
        with open(os.path.join(path, "catalog_items.pkl"), "wb") as f:
            pickle.dump(self.catalog_items, f)

        logger.info(f"Index saved to {path}")

    def load_index(self, path: Optional[str] = None) -> bool:
        """Load FAISS index and catalog metadata from disk."""
        path = path or self.settings.faiss_index_path
        index_path = os.path.join(path, "index.faiss")
        meta_path = os.path.join(path, "catalog_items.pkl")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            logger.warning(f"Index files not found at {path}")
            return False

        self._load_model()
        self.index = faiss.read_index(index_path)

        with open(meta_path, "rb") as f:
            self.catalog_items = pickle.load(f)

        self._initialized = True
        logger.info(f"Index loaded: {self.index.ntotal} vectors")
        return True

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_keys: Optional[List[str]] = None,
        filter_job_levels: Optional[List[str]] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Semantic search over the catalog.
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            filter_keys: Optional filter by assessment category (e.g., "Knowledge & Skills")
            filter_job_levels: Optional filter by job level (e.g., "Mid-Professional")
        
        Returns:
            List of (catalog_item, similarity_score) tuples, sorted by score descending.
        
        How it works:
        1. Encode the query into an embedding
        2. Search FAISS for nearest neighbors
        3. Apply optional metadata filters
        4. Return results with scores
        """
        if not self._initialized:
            logger.error("Vector store not initialized. Call build_index() or load_index() first.")
            return []

        top_k = top_k or self.settings.top_k

        # Encode query
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype(np.float32)

        # Search FAISS (retrieve more than top_k to allow for filtering)
        search_k = min(top_k * 3, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, search_k)

        # Collect results with optional filtering
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            item = self.catalog_items[idx]

            # Apply category filter
            if filter_keys:
                item_keys = set(item.get("keys", []))
                if not item_keys.intersection(set(filter_keys)):
                    continue

            # Apply job level filter
            if filter_job_levels:
                item_levels = set(item.get("job_levels", []))
                if not item_levels.intersection(set(filter_job_levels)):
                    continue

            results.append((item, float(score)))

            if len(results) >= top_k:
                break

        return results

    def multi_query_search(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search with multiple queries and merge results.
        
        Why multi-query?
        A user's request often has multiple dimensions:
        "Java developer with leadership skills"
        -> Query 1: "Java programming assessment"
        -> Query 2: "leadership personality assessment"
        
        This improves recall by capturing different aspects.
        Results are deduplicated and the max score is kept.
        """
        if not self._initialized:
            return []

        top_k = top_k or self.settings.top_k
        seen: Dict[str, Tuple[Dict[str, Any], float]] = {}

        for query in queries:
            results = self.search(query, top_k=top_k)
            for item, score in results:
                item_id = item.get("entity_id", item.get("name"))
                if item_id not in seen or score > seen[item_id][1]:
                    seen[item_id] = (item, score)

        # Sort by score descending
        merged = sorted(seen.values(), key=lambda x: x[1], reverse=True)
        return merged[:top_k]


# Singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create the singleton VectorStore instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
