"""Embedding generation and caching manager for restaurant documents.

Generates dense document vectors, preserves ID mappings, and prevents
unnecessary regeneration using content hashing and disk caching.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from src.embeddings.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)

DEFAULT_VECTOR_DIR = Path(__file__).resolve().parent.parent.parent / "vector_db"


class EmbeddingGenerator:
    """Manages document vector generation and caching."""

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize EmbeddingGenerator.

        Args:
            embedding_model: EmbeddingModel instance. If None, initialized with default E5-small.
            vector_dir: Directory to save/load cached embeddings. Defaults to vector_db/.
        """
        self._model = embedding_model
        self.vector_dir = Path(vector_dir) if vector_dir else DEFAULT_VECTOR_DIR
        self.vector_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings_path = self.vector_dir / "embeddings.npy"
        self.metadata_path = self.vector_dir / "embedding_metadata.json"

    @property
    def model(self) -> EmbeddingModel:
        """Lazy load embedding model on first access."""
        if self._model is None:
            self._model = EmbeddingModel()
        return self._model

    def _compute_corpus_hash(self, corpus: Dict[int, str]) -> str:
        """Compute MD5 hash of corpus content and model name to detect changes."""
        hasher = hashlib.md5()
        hasher.update(self.model.model_name.encode("utf-8"))
        for doc_id in sorted(corpus.keys()):
            hasher.update(str(doc_id).encode("utf-8"))
            hasher.update(corpus[doc_id].encode("utf-8"))
        return hasher.hexdigest()

    def generate_or_load(
        self,
        corpus: Dict[int, str],
        force_regenerate: bool = False,
    ) -> Tuple[np.ndarray, List[int]]:
        """Generate embeddings or load from disk if cache matches corpus.

        Args:
            corpus: Mapping from doc_id to document text string.
            force_regenerate: If True, ignores disk cache and regenerates.

        Returns:
            Tuple of (embeddings_array, doc_ids_list).
        """
        current_hash = self._compute_corpus_hash(corpus)
        sorted_doc_ids = sorted(list(corpus.keys()))

        # Check if valid cache exists
        if not force_regenerate and self.embeddings_path.exists() and self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                if metadata.get("corpus_hash") == current_hash and metadata.get("doc_ids") == sorted_doc_ids:
                    logger.info("Loading cached embeddings from %s", self.embeddings_path)
                    embeddings = np.load(self.embeddings_path)
                    logger.info("Loaded %d embeddings of dimension %d from cache.", len(embeddings), embeddings.shape[1])
                    return embeddings, sorted_doc_ids
                else:
                    logger.info("Corpus content or model changed. Regenerating embeddings...")
            except Exception as e:
                logger.warning("Failed to load cached embeddings: %s. Regenerating...", e)

        # Generate fresh embeddings
        logger.info("Generating embeddings for %d documents...", len(corpus))
        passages = [corpus[doc_id] for doc_id in sorted_doc_ids]
        embeddings = self.model.encode_passages(
            passages,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=True,
        )

        # Save to disk
        self._save_cache(embeddings, sorted_doc_ids, current_hash)
        return embeddings, sorted_doc_ids

    def _save_cache(
        self,
        embeddings: np.ndarray,
        doc_ids: List[int],
        corpus_hash: str,
    ) -> None:
        """Persist embeddings array and metadata to disk."""
        np.save(self.embeddings_path, embeddings)
        metadata = {
            "model_name": self.model.model_name,
            "dimension": int(embeddings.shape[1]),
            "num_documents": len(doc_ids),
            "corpus_hash": corpus_hash,
            "doc_ids": doc_ids,
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved embeddings and metadata to %s", self.vector_dir)
