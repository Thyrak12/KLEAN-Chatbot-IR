"""Dense Vector Search retriever for KLEAN Chatbot.

Encodes user queries via EmbeddingModel and performs nearest-neighbor
cosine similarity search in FAISS VectorStore.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.embeddings.embedding_model import EmbeddingModel
from src.embeddings.vector_store import VectorStore

logger = logging.getLogger(__name__)


class VectorSearcher:
    """Retrieves ranked documents using dense query embeddings and FAISS."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        """Initialize VectorSearcher.

        Args:
            vector_store: Populated VectorStore instance.
            embedding_model: EmbeddingModel instance for encoding queries.
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model or EmbeddingModel()

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search the FAISS vector index for documents matching the query.

        Args:
            query: Natural language query string.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of ranked result dicts: [{"doc_id": id, "score": cosine_sim, "rank": r}, ...]
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        # 1. Encode user query into normalized 1D vector (multilingual-e5 query: prefix applied)
        query_vector = self.embedding_model.encode_query(
            clean_query, normalize_embeddings=True
        )

        # 2. Search FAISS index via inner product (Cosine Similarity)
        faiss_results = self.vector_store.search(query_vector, top_k=top_k)

        # 3. Format ranked results
        results: List[Dict[str, Any]] = []
        for rank, (doc_id, score) in enumerate(faiss_results, start=1):
            results.append({
                "doc_id": doc_id,
                "score": float(score),
                "rank": rank,
            })

        return results
