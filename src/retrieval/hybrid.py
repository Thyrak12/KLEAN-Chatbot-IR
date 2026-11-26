"""Hybrid Retrieval Engine for KLEAN Chatbot.

Combines lexical search (BM25) and dense semantic search (FAISS Vector Search)
using Reciprocal Rank Fusion (RRF) and Normalized Linear Score Fusion with
configurable weight tuning.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.vector_search import VectorSearcher

logger = logging.getLogger(__name__)

DEFAULT_RRF_K = 60
DEFAULT_BM25_WEIGHT = 0.5
DEFAULT_VECTOR_WEIGHT = 0.5


class HybridRetriever:
    """Hybrid retrieval engine combining BM25 lexical search and FAISS vector search."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_searcher: VectorSearcher,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        rrf_k: int = DEFAULT_RRF_K,
        default_fusion_method: str = "rrf",
    ):
        """Initialize HybridRetriever.

        Args:
            bm25_retriever: Pre-configured BM25Retriever instance.
            vector_searcher: Pre-configured VectorSearcher instance.
            bm25_weight: Default weight for lexical BM25 ranking (>= 0).
            vector_weight: Default weight for semantic vector ranking (>= 0).
            rrf_k: Smoothing constant for Reciprocal Rank Fusion (standard default 60).
            default_fusion_method: 'rrf' (rank-based) or 'linear' (score-based).
        """
        self.bm25_retriever = bm25_retriever
        self.vector_searcher = vector_searcher
        self.bm25_weight = float(bm25_weight)
        self.vector_weight = float(vector_weight)
        self.rrf_k = int(rrf_k)
        self.default_fusion_method = default_fusion_method.lower()

    def _normalize_weights(
        self, bm25_w: Optional[float], vector_w: Optional[float]
    ) -> Tuple[float, float]:
        """Normalize BM25 and Vector weights to sum to 1.0."""
        w_bm = float(bm25_w if bm25_w is not None else self.bm25_weight)
        w_vec = float(vector_w if vector_w is not None else self.vector_weight)

        total = w_bm + w_vec
        if total <= 0:
            return 0.5, 0.5
        return w_bm / total, w_vec / total

    def search(
        self,
        query: str,
        top_k: int = 10,
        fusion_method: Optional[str] = None,
        bm25_weight: Optional[float] = None,
        vector_weight: Optional[float] = None,
        rrf_k: Optional[int] = None,
        candidate_pool_size: int = 50,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining BM25 and Dense Vector Search.

        Args:
            query: User search query string.
            top_k: Number of ranked documents to return.
            fusion_method: 'rrf' (default) or 'linear'.
            bm25_weight: Optional per-query override for BM25 weight.
            vector_weight: Optional per-query override for Vector weight.
            rrf_k: Optional per-query override for RRF smoothing constant.
            candidate_pool_size: Number of top candidates to retrieve from each engine.

        Returns:
            List of result dicts: [
                {
                    "doc_id": int,
                    "score": float,
                    "rank": int,
                    "bm25_rank": Optional[int],
                    "bm25_score": Optional[float],
                    "vector_rank": Optional[int],
                    "vector_score": Optional[float],
                }, ...
            ]
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        method = (fusion_method or self.default_fusion_method).lower()
        w_bm, w_vec = self._normalize_weights(bm25_weight, vector_weight)
        k_val = int(rrf_k if rrf_k is not None else self.rrf_k)

        # 1. Fetch candidate pools from BM25 and Vector Search
        bm25_candidates = self.bm25_retriever.search(clean_query, top_k=candidate_pool_size)
        vector_candidates = self.vector_searcher.search(clean_query, top_k=candidate_pool_size)

        if not bm25_candidates and not vector_candidates:
            return []

        # 2. Build lookups for rank and score
        bm25_map: Dict[int, Dict[str, Any]] = {
            item["doc_id"]: item for item in bm25_candidates
        }
        vector_map: Dict[int, Dict[str, Any]] = {
            item["doc_id"]: item for item in vector_candidates
        }

        all_doc_ids = set(bm25_map.keys()) | set(vector_map.keys())

        # 3. Fuse scores
        merged_scores: List[Dict[str, Any]] = []

        if method == "linear":
            # Min-Max Normalize BM25 scores across retrieved candidates
            bm_scores = [item["score"] for item in bm25_candidates]
            min_bm = min(bm_scores) if bm_scores else 0.0
            max_bm = max(bm_scores) if bm_scores else 0.0
            range_bm = max_bm - min_bm

            # Min-Max Normalize Vector scores across retrieved candidates
            vec_scores = [item["score"] for item in vector_candidates]
            min_vec = min(vec_scores) if vec_scores else 0.0
            max_vec = max(vec_scores) if vec_scores else 0.0
            range_vec = max_vec - min_vec

            for doc_id in all_doc_ids:
                bm_info = bm25_map.get(doc_id)
                vec_info = vector_map.get(doc_id)

                # Normalized BM25 score in [0, 1]
                if bm_info is not None:
                    norm_bm = (bm_info["score"] - min_bm) / range_bm if range_bm > 0 else 1.0
                else:
                    norm_bm = 0.0

                # Normalized Vector score in [0, 1]
                if vec_info is not None:
                    norm_vec = (vec_info["score"] - min_vec) / range_vec if range_vec > 0 else 1.0
                else:
                    norm_vec = 0.0

                fused_score = (w_bm * norm_bm) + (w_vec * norm_vec)

                merged_scores.append({
                    "doc_id": doc_id,
                    "score": float(fused_score),
                    "bm25_rank": bm_info["rank"] if bm_info else None,
                    "bm25_score": bm_info["score"] if bm_info else None,
                    "vector_rank": vec_info["rank"] if vec_info else None,
                    "vector_score": vec_info["score"] if vec_info else None,
                })
        else:
            # Reciprocal Rank Fusion (RRF)
            for doc_id in all_doc_ids:
                bm_info = bm25_map.get(doc_id)
                vec_info = vector_map.get(doc_id)

                rrf_score = 0.0
                if bm_info is not None:
                    rrf_score += w_bm / (k_val + bm_info["rank"])
                if vec_info is not None:
                    rrf_score += w_vec / (k_val + vec_info["rank"])

                merged_scores.append({
                    "doc_id": doc_id,
                    "score": float(rrf_score),
                    "bm25_rank": bm_info["rank"] if bm_info else None,
                    "bm25_score": bm_info["score"] if bm_info else None,
                    "vector_rank": vec_info["rank"] if vec_info else None,
                    "vector_score": vec_info["score"] if vec_info else None,
                })

        # 4. Sort descending by fused score, breaking ties by doc_id
        merged_scores.sort(key=lambda x: (x["score"], -x["doc_id"]), reverse=True)

        # 5. Assign 1-based ranks and truncate to top_k
        results: List[Dict[str, Any]] = []
        for rank, item in enumerate(merged_scores[:top_k], start=1):
            item["rank"] = rank
            results.append(item)

        return results

    def explain_fusion(
        self,
        query: str,
        doc_id: int,
        fusion_method: Optional[str] = None,
        bm25_weight: Optional[float] = None,
        vector_weight: Optional[float] = None,
        rrf_k: Optional[int] = None,
        candidate_pool_size: int = 50,
    ) -> Dict[str, Any]:
        """Provide detailed mathematical explanation of hybrid score calculation for a document.

        Args:
            query: User search query string.
            doc_id: Restaurant document ID.
            fusion_method: 'rrf' or 'linear'.
            bm25_weight: Optional BM25 weight override.
            vector_weight: Optional Vector weight override.
            rrf_k: Optional RRF k constant.
            candidate_pool_size: Candidate retrieval pool size.

        Returns:
            Dictionary with breakdown details.
        """
        method = (fusion_method or self.default_fusion_method).lower()
        w_bm, w_vec = self._normalize_weights(bm25_weight, vector_weight)
        k_val = int(rrf_k if rrf_k is not None else self.rrf_k)

        bm25_candidates = self.bm25_retriever.search(query, top_k=candidate_pool_size)
        vector_candidates = self.vector_searcher.search(query, top_k=candidate_pool_size)

        bm_info = next((x for x in bm25_candidates if x["doc_id"] == doc_id), None)
        vec_info = next((x for x in vector_candidates if x["doc_id"] == doc_id), None)

        explanation: Dict[str, Any] = {
            "query": query,
            "doc_id": doc_id,
            "fusion_method": method,
            "bm25_weight": w_bm,
            "vector_weight": w_vec,
            "in_bm25_candidates": bm_info is not None,
            "in_vector_candidates": vec_info is not None,
            "bm25_rank": bm_info["rank"] if bm_info else None,
            "bm25_score": bm_info["score"] if bm_info else None,
            "vector_rank": vec_info["rank"] if vec_info else None,
            "vector_score": vec_info["score"] if vec_info else None,
        }

        if method == "linear":
            bm_scores = [item["score"] for item in bm25_candidates]
            min_bm = min(bm_scores) if bm_scores else 0.0
            max_bm = max(bm_scores) if bm_scores else 0.0
            range_bm = max_bm - min_bm

            vec_scores = [item["score"] for item in vector_candidates]
            min_vec = min(vec_scores) if vec_scores else 0.0
            max_vec = max(vec_scores) if vec_scores else 0.0
            range_vec = max_vec - min_vec

            norm_bm = (
                (bm_info["score"] - min_bm) / range_bm
                if (bm_info and range_bm > 0)
                else (1.0 if bm_info else 0.0)
            )
            norm_vec = (
                (vec_info["score"] - min_vec) / range_vec
                if (vec_info and range_vec > 0)
                else (1.0 if vec_info else 0.0)
            )

            final_score = (w_bm * norm_bm) + (w_vec * norm_vec)
            explanation.update({
                "norm_bm25_score": norm_bm,
                "norm_vector_score": norm_vec,
                "final_score": final_score,
                "formula": f"({w_bm:.2f} * {norm_bm:.4f}) + ({w_vec:.2f} * {norm_vec:.4f})",
            })
        else:
            bm_contrib = (w_bm / (k_val + bm_info["rank"])) if bm_info else 0.0
            vec_contrib = (w_vec / (k_val + vec_info["rank"])) if vec_info else 0.0
            final_score = bm_contrib + vec_contrib

            explanation.update({
                "rrf_k": k_val,
                "bm25_rrf_contribution": bm_contrib,
                "vector_rrf_contribution": vec_contrib,
                "final_score": final_score,
                "formula": f"({w_bm:.2f} / ({k_val} + {bm_info['rank'] if bm_info else 'inf'})) + ({w_vec:.2f} / ({k_val} + {vec_info['rank'] if vec_info else 'inf'}))",
            })

        return explanation
