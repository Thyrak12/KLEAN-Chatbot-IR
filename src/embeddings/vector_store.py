"""FAISS Vector Store wrapper for KLEAN Chatbot.

Manages FAISS inner-product index for normalized cosine similarity search,
vector insertions, top-k retrievals, and ID mappings.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = Path(__file__).resolve().parent.parent.parent / "vector_db" / "faiss_index.bin"
DEFAULT_METADATA_PATH = Path(__file__).resolve().parent.parent.parent / "vector_db" / "index_metadata.json"


class VectorStore:
    """FAISS-based vector database supporting cosine similarity via inner-product."""

    def __init__(self, dimension: int = 384):
        """Initialize VectorStore.

        Args:
            dimension: Dimensionality of vectors (default 384 for multilingual-e5-small).
        """
        self.dimension = dimension
        # faiss.IndexFlatIP computes inner products. When vectors are L2-normalized,
        # inner product is mathematically identical to cosine similarity!
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        # Position in FAISS (0..N-1) -> Restaurant doc_id
        self.idx_to_doc_id: List[int] = []
        # Restaurant doc_id -> Position in FAISS
        self.doc_id_to_idx: Dict[int, int] = {}

    def add(self, vectors: np.ndarray, doc_ids: List[int]) -> None:
        """Insert normalized vectors and their corresponding restaurant doc_ids into FAISS.

        Args:
            vectors: np.ndarray of shape (N, dimension) of float32.
            doc_ids: List of integer restaurant IDs of length N.
        """
        if len(vectors) != len(doc_ids):
            raise ValueError(
                f"Mismatch: got {len(vectors)} vectors but {len(doc_ids)} doc_ids."
            )

        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        start_idx = self.index.ntotal
        self.index.add(vectors)

        for offset, doc_id in enumerate(doc_ids):
            pos = start_idx + offset
            self.idx_to_doc_id.append(doc_id)
            self.doc_id_to_idx[doc_id] = pos

        logger.info(
            "Added %d vectors to FAISS index. Total indexed: %d",
            len(vectors),
            self.index.ntotal,
        )

    def search(
        self, query_vector: np.ndarray, top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Search FAISS index for most similar vectors to query.

        Args:
            query_vector: 1D (dimension,) or 2D (1, dimension) float32 query vector.
            top_k: Number of nearest neighbors to return.

        Returns:
            List of (doc_id, cosine_similarity_score) sorted descending by score.
        """
        if self.index.ntotal == 0:
            return []

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)

        results: List[Tuple[int, float]] = []
        for pos, score in zip(indices[0], scores[0]):
            if pos != -1 and pos < len(self.idx_to_doc_id):
                doc_id = self.idx_to_doc_id[pos]
                results.append((doc_id, float(score)))

        return results

    @property
    def total_vectors(self) -> int:
        """Total vectors stored in index."""
        return self.index.ntotal

    def save(
        self,
        index_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Persist FAISS index binary and metadata mapping to disk."""
        idx_p = Path(index_path) if index_path else DEFAULT_INDEX_PATH
        meta_p = Path(metadata_path) if metadata_path else DEFAULT_METADATA_PATH

        idx_p.parent.mkdir(parents=True, exist_ok=True)
        meta_p.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(idx_p))

        metadata = {
            "dimension": self.dimension,
            "total_vectors": self.index.ntotal,
            "idx_to_doc_id": self.idx_to_doc_id,
        }
        with open(meta_p, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved FAISS index to %s and metadata to %s", idx_p, meta_p)

    @classmethod
    def load(
        cls,
        index_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
    ) -> "VectorStore":
        """Load a VectorStore from saved FAISS index and metadata files."""
        idx_p = Path(index_path) if index_path else DEFAULT_INDEX_PATH
        meta_p = Path(metadata_path) if metadata_path else DEFAULT_METADATA_PATH

        if not idx_p.exists():
            raise FileNotFoundError(f"FAISS index file not found at: {idx_p}")
        if not meta_p.exists():
            raise FileNotFoundError(f"Index metadata file not found at: {meta_p}")

        with open(meta_p, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        dimension = metadata.get("dimension", 384)
        store = cls(dimension=dimension)
        store.index = faiss.read_index(str(idx_p))
        store.idx_to_doc_id = metadata.get("idx_to_doc_id", [])
        store.doc_id_to_idx = {
            doc_id: pos for pos, doc_id in enumerate(store.idx_to_doc_id)
        }

        logger.info("Loaded VectorStore from %s with %d vectors", idx_p, store.index.ntotal)
        return store
