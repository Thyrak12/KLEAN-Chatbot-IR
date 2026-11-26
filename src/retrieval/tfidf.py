"""TF-IDF Retrieval module for KLEAN Chatbot.

Implements TF-IDF vector space model and cosine similarity ranking from
first mathematical principles, with optional Scikit-learn comparison.
"""

from collections import defaultdict
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from src.indexing.inverted_index import InvertedIndex
from src.preprocessing.text_preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)


class TFIDFRetriever:
    """Calculates TF-IDF representations and cosine similarity ranking."""

    def __init__(
        self,
        inverted_index: InvertedIndex,
        preprocessor: Optional[TextPreprocessor] = None,
        sublinear_tf: bool = True,
        smooth_idf: bool = True,
    ):
        """Initialize TFIDFRetriever.

        Args:
            inverted_index: Built InvertedIndex instance.
            preprocessor: Preprocessor instance. If None, default TextPreprocessor is used.
            sublinear_tf: If True, uses 1 + ln(TF) for TF > 0 (prevents high term repetition bias).
            smooth_idf: If True, adds 1 to document frequencies to prevent division by zero.
        """
        self.index = inverted_index
        self.preprocessor = preprocessor or TextPreprocessor()
        self.sublinear_tf = sublinear_tf
        self.smooth_idf = smooth_idf

        # Precomputed IDF weights: term -> idf
        self._idf_cache: Dict[str, float] = {}
        # Precomputed document vector L2 norms: doc_id -> norm
        self._doc_vector_norms: Dict[int, float] = {}
        # Precomputed sparse doc vectors: doc_id -> {term: unnormalized_weight}
        self._doc_vectors: Dict[int, Dict[str, float]] = defaultdict(dict)

        self._build_model()

    def _calculate_tf(self, count: int) -> float:
        """Calculate Term Frequency weight."""
        if count <= 0:
            return 0.0
        if self.sublinear_tf:
            return 1.0 + math.log(count)
        return float(count)

    def _calculate_idf(self, df: int, total_docs: int) -> float:
        """Calculate Inverse Document Frequency weight.

        Formula (Smooth IDF):
            IDF(t) = ln((N + 1) / (DF(t) + 1)) + 1
        """
        if self.smooth_idf:
            return math.log((total_docs + 1) / (df + 1)) + 1.0
        if df == 0:
            return 0.0
        return math.log(total_docs / df)

    def _build_model(self) -> None:
        """Precompute IDF weights and document vector representations."""
        total_docs = self.index.num_documents
        if total_docs == 0:
            return

        # 1. Compute IDF for all vocabulary terms
        self._idf_cache.clear()
        for term in self.index.vocabulary:
            df = self.index.get_document_frequency(term)
            self._idf_cache[term] = self._calculate_idf(df, total_docs)

        # 2. Build sparse document vectors and calculate L2 norms
        self._doc_vectors.clear()
        self._doc_vector_norms.clear()

        for doc_id, doc_length in self.index.doc_lengths.items():
            terms = self.index.get_document_terms(doc_id)
            sum_squares = 0.0
            doc_weights: Dict[str, float] = {}

            for term, count in terms.items():
                tf = self._calculate_tf(count)
                idf = self._idf_cache.get(term, 0.0)
                weight = tf * idf
                doc_weights[term] = weight
                sum_squares += weight * weight

            self._doc_vectors[doc_id] = doc_weights
            self._doc_vector_norms[doc_id] = math.sqrt(sum_squares) if sum_squares > 0 else 1.0

        logger.info(
            "Built TFIDF model with %d terms across %d document vectors",
            len(self._idf_cache),
            len(self._doc_vectors),
        )

    def get_idf(self, term: str) -> float:
        """Retrieve precomputed IDF for a term."""
        return self._idf_cache.get(term.lower(), 0.0)

    def encode_query(self, query: str) -> Tuple[Dict[str, float], float]:
        """Convert query string into a sparse normalized TF-IDF vector.

        Returns:
            Tuple of ({term: weight}, l2_norm).
        """
        tokens = self.preprocessor.preprocess(query)
        if not tokens:
            return {}, 0.0

        term_counts = defaultdict(int)
        for t in tokens:
            term_counts[t] += 1

        query_weights: Dict[str, float] = {}
        sum_squares = 0.0

        for term, count in term_counts.items():
            if term in self._idf_cache:
                tf = self._calculate_tf(count)
                idf = self._idf_cache[term]
                weight = tf * idf
                query_weights[term] = weight
                sum_squares += weight * weight

        norm = math.sqrt(sum_squares) if sum_squares > 0 else 1.0
        return query_weights, norm

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Rank documents by Cosine Similarity to query TF-IDF vector.

        Uses inverted index postings accumulator for fast search.

        Args:
            query: User search query.
            top_k: Maximum number of ranked results to return.

        Returns:
            List of result dicts: [{"doc_id": id, "score": cosine_sim, "rank": r}, ...]
        """
        query_weights, query_norm = self.encode_query(query)
        if not query_weights or query_norm == 0.0:
            return []

        # Accumulator: doc_id -> dot_product(query, doc)
        dot_products = defaultdict(float)

        for term, q_weight in query_weights.items():
            postings = self.index.get_postings(term)
            for doc_id in postings:
                d_weight = self._doc_vectors.get(doc_id, {}).get(term, 0.0)
                dot_products[doc_id] += q_weight * d_weight

        # Compute cosine similarity = dot_product / (norm_q * norm_d)
        scores: List[Tuple[int, float]] = []
        for doc_id, dot_prod in dot_products.items():
            doc_norm = self._doc_vector_norms.get(doc_id, 1.0)
            cosine_sim = dot_prod / (query_norm * doc_norm)
            if cosine_sim > 0.0:
                scores.append((doc_id, cosine_sim))

        # Sort descending by score
        scores.sort(key=lambda x: x[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for rank, (doc_id, score) in enumerate(scores[:top_k], start=1):
            results.append({
                "doc_id": doc_id,
                "score": float(score),
                "rank": rank,
            })

        return results

    def compare_with_sklearn(
        self,
        corpus_dict: Dict[int, str],
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Compare custom TF-IDF rankings with Scikit-learn's TfidfVectorizer.

        Args:
            corpus_dict: Mapping {doc_id: raw_text}.
            query: Query string.
            top_k: Top-k results to compare.

        Returns:
            Comparison dictionary with both ranking lists and overlap ratio.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            raise ImportError("scikit-learn is required for compare_with_sklearn()")

        # 1. Custom results
        custom_results = self.search(query, top_k=top_k)
        custom_ids = [r["doc_id"] for r in custom_results]

        # 2. Scikit-learn results
        doc_ids = list(corpus_dict.keys())
        documents = [corpus_dict[doc_id] for doc_id in doc_ids]

        # Use token pattern matching Latin words and Khmer unicode sequences
        token_pattern = r"(?u)\b\w+\b|[\u1780-\u17FF\u19E0-\u19FF]+"
        vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=token_pattern,
            sublinear_tf=self.sublinear_tf,
            smooth_idf=self.smooth_idf,
        )

        tfidf_matrix = vectorizer.fit_transform(documents)
        query_vec = vectorizer.transform([query])
        sims = cosine_similarity(query_vec, tfidf_matrix).flatten()

        top_indices = sims.argsort()[::-1][:top_k]
        sklearn_results = [
            {"doc_id": doc_ids[idx], "score": float(sims[idx]), "rank": r + 1}
            for r, idx in enumerate(top_indices)
            if sims[idx] > 0
        ]
        sklearn_ids = [r["doc_id"] for r in sklearn_results]

        # Overlap in top-k
        overlap = set(custom_ids).intersection(set(sklearn_ids))
        overlap_pct = (len(overlap) / min(len(custom_ids), len(sklearn_ids)) * 100) if custom_ids and sklearn_ids else 0.0

        return {
            "query": query,
            "custom_rankings": custom_results,
            "sklearn_rankings": sklearn_results,
            "top_k_overlap_count": len(overlap),
            "top_k_overlap_percentage": round(overlap_pct, 1),
        }
