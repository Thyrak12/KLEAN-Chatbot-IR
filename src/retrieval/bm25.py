"""Okapi BM25 Retrieval module for KLEAN Chatbot.

Implements Okapi BM25 ranking directly from mathematical first principles,
accounting for term frequency saturation, Robertson-Spärck Jones IDF,
and document length normalization.
"""

from collections import defaultdict
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from src.indexing.inverted_index import InvertedIndex
from src.preprocessing.text_preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)


class BM25Retriever:
    """Ranks documents using the Okapi BM25 retrieval function."""

    def __init__(
        self,
        inverted_index: InvertedIndex,
        preprocessor: Optional[TextPreprocessor] = None,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        """Initialize BM25Retriever.

        Args:
            inverted_index: Built InvertedIndex instance.
            preprocessor: Preprocessor instance. If None, default TextPreprocessor is used.
            k1: Term frequency saturation parameter (typically 1.2 to 2.0).
            b: Document length normalization parameter (0.0 = no normalization, 1.0 = full).
            epsilon: Floor value for IDF to prevent zero or negative weights for very frequent terms.
        """
        self.index = inverted_index
        self.preprocessor = preprocessor or TextPreprocessor()
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # Precalculated parameters
        self._total_docs: int = self.index.num_documents
        self._avg_doc_length: float = self.index.avg_doc_length
        self._idf_cache: Dict[str, float] = {}

        self._build_idf_cache()

    def _build_idf_cache(self) -> None:
        """Precalculate Robertson-Spärck Jones IDF for all terms in the vocabulary.

        Formula:
            IDF(t) = ln( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 )
        """
        self._total_docs = self.index.num_documents
        self._avg_doc_length = self.index.avg_doc_length
        self._idf_cache.clear()

        if self._total_docs == 0:
            return

        for term in self.index.vocabulary:
            df = self.index.get_document_frequency(term)
            # Lucene / Elasticsearch non-negative smoothed RSJ IDF
            idf = math.log(((self._total_docs - df + 0.5) / (df + 0.5)) + 1.0)
            self._idf_cache[term] = max(idf, self.epsilon)

        logger.info(
            "Precomputed BM25 IDF weights for %d terms (k1=%.2f, b=%.2f, avgdl=%.2f)",
            len(self._idf_cache),
            self.k1,
            self.b,
            self._avg_doc_length,
        )

    def get_idf(self, term: str) -> float:
        """Retrieve precomputed BM25 IDF for a term."""
        return self._idf_cache.get(term.lower(), 0.0)

    def score_document(self, query_terms: List[str], doc_id: int) -> float:
        """Calculate the BM25 score of a single document for a list of query terms."""
        doc_len = self.index.doc_lengths.get(doc_id, 0)
        if doc_len == 0 or self._avg_doc_length == 0:
            return 0.0

        # Length normalization factor
        len_norm = 1.0 - self.b + self.b * (doc_len / self._avg_doc_length)

        score = 0.0
        for term in query_terms:
            tf = self.index.get_term_frequency(term, doc_id)
            if tf <= 0:
                continue

            idf = self.get_idf(term)
            # BM25 TF component
            tf_component = (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)
            score += idf * tf_component

        return score

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Rank documents for a user query using Okapi BM25.

        Uses inverted index postings lists to only score candidate documents
        that contain at least one query term.

        Args:
            query: User search query.
            top_k: Number of ranked documents to return.

        Returns:
            List of ranked results: [{"doc_id": id, "score": bm25_score, "rank": r}, ...]
        """
        query_terms = self.preprocessor.preprocess(query)
        if not query_terms:
            return []

        # Find candidate documents containing at least one query term
        candidate_doc_ids = set()
        for term in query_terms:
            postings = self.index.get_postings(term)
            candidate_doc_ids.update(postings)

        if not candidate_doc_ids:
            return []

        # Calculate BM25 score for each candidate document
        scored_candidates: List[Tuple[int, float]] = []
        for doc_id in candidate_doc_ids:
            score = self.score_document(query_terms, doc_id)
            if score > 0.0:
                scored_candidates.append((doc_id, score))

        # Sort descending by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for rank, (doc_id, score) in enumerate(scored_candidates[:top_k], start=1):
            results.append({
                "doc_id": doc_id,
                "score": float(score),
                "rank": rank,
            })

        return results

    def explain_score(self, query: str, doc_id: int) -> Dict[str, Any]:
        """Provide a detailed mathematical breakdown of the BM25 score for a document.

        Returns:
            Dictionary explaining IDF, TF, length norm factor, and contribution of each term.
        """
        query_terms = self.preprocessor.preprocess(query)
        doc_len = self.index.doc_lengths.get(doc_id, 0)
        len_norm = (
            1.0 - self.b + self.b * (doc_len / self._avg_doc_length)
            if self._avg_doc_length > 0
            else 1.0
        )

        term_breakdowns = []
        total_score = 0.0

        for term in query_terms:
            tf = self.index.get_term_frequency(term, doc_id)
            idf = self.get_idf(term)
            tf_component = (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm) if tf > 0 else 0.0
            term_score = idf * tf_component
            total_score += term_score

            term_breakdowns.append({
                "term": term,
                "tf": tf,
                "idf": round(idf, 4),
                "tf_component": round(tf_component, 4),
                "term_score": round(term_score, 4),
            })

        return {
            "doc_id": doc_id,
            "doc_length": doc_len,
            "avg_doc_length": round(self._avg_doc_length, 2),
            "length_norm_factor": round(len_norm, 4),
            "k1": self.k1,
            "b": self.b,
            "total_bm25_score": round(total_score, 4),
            "terms": term_breakdowns,
        }
