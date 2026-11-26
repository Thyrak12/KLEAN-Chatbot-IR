"""Embedding Model wrapper for KLEAN Chatbot.

Wraps Sentence Transformers for local multilingual dense representation generation
specifically supporting E5-style query: and passage: prefixes.
"""

import logging
from typing import List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"


class EmbeddingModel:
    """Loads and provides encoding methods for dense sentence embedding models."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        passage_prefix: str = "passage: ",
        query_prefix: str = "query: ",
    ):
        """Initialize EmbeddingModel.

        Args:
            model_name: HuggingFace model repo ID or local directory.
            device: 'cpu' or 'cuda'.
            passage_prefix: Prefix added to document texts for symmetric models like E5.
            query_prefix: Prefix added to user queries for symmetric models like E5.
        """
        self.model_name = model_name
        self.device = device
        self.passage_prefix = passage_prefix
        self.query_prefix = query_prefix

        logger.info("Loading embedding model '%s' on %s...", self.model_name, self.device)
        self.model = SentenceTransformer(self.model_name, device=self.device)
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info("Embedding model loaded successfully. Dimension: %d", self.dimension)

    def encode_passages(
        self,
        passages: List[str],
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode a list of document strings with the passage prefix.

        Args:
            passages: List of document text strings.
            normalize_embeddings: Whether to L2-normalize vectors.
            batch_size: Batch size for encoding.
            show_progress_bar: Whether to display a progress bar.

        Returns:
            np.ndarray of shape (len(passages), dimension) of float32.
        """
        if not passages:
            return np.empty((0, self.dimension), dtype=np.float32)

        # Prepend passage prefix if specified
        prefixed_passages = [
            f"{self.passage_prefix}{text}" if self.passage_prefix else text
            for text in passages
        ]

        embeddings = self.model.encode(
            prefixed_passages,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
        )

        return embeddings.astype(np.float32)

    def encode_queries(
        self,
        queries: List[str],
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode a list of user search queries with the query prefix.

        Args:
            queries: List of query strings.
            normalize_embeddings: Whether to L2-normalize vectors.
            batch_size: Batch size for encoding.

        Returns:
            np.ndarray of shape (len(queries), dimension) of float32.
        """
        if not queries:
            return np.empty((0, self.dimension), dtype=np.float32)

        prefixed_queries = [
            f"{self.query_prefix}{q}" if self.query_prefix else q
            for q in queries
        ]

        embeddings = self.model.encode(
            prefixed_queries,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
        )

        return embeddings.astype(np.float32)

    def encode_query(self, query: str, normalize_embeddings: bool = True) -> np.ndarray:
        """Encode a single query string. Returns 1D array of float32."""
        vectors = self.encode_queries([query], normalize_embeddings=normalize_embeddings)
        return vectors[0]

    def get_dimension(self) -> int:
        """Return embedding vector dimension."""
        return self.dimension
