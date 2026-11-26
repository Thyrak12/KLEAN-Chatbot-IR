"""Unit tests for Phase 5: Hybrid Retrieval Engine."""

import unittest
from unittest.mock import MagicMock

from src.retrieval.hybrid import HybridRetriever


class TestPhase5HybridRetriever(unittest.TestCase):
    """Test suite for HybridRetriever with RRF and Linear Fusion."""

    def setUp(self):
        # Create mock BM25 and Vector retrievers with controlled dummy return lists
        self.mock_bm25 = MagicMock()
        self.mock_vector = MagicMock()

        self.retriever = HybridRetriever(
            bm25_retriever=self.mock_bm25,
            vector_searcher=self.mock_vector,
            bm25_weight=0.5,
            vector_weight=0.5,
            rrf_k=60,
        )

    def test_empty_query(self):
        res = self.retriever.search("   ")
        self.assertEqual(res, [])
        self.mock_bm25.search.assert_not_called()
        self.mock_vector.search.assert_not_called()

    def test_rrf_scoring_calculation(self):
        """Verify RRF formula: w_bm / (k + r_bm) + w_vec / (k + r_vec)."""
        # Document 10: Rank 1 in BM25, Rank 2 in Vector
        # Document 20: Rank 2 in BM25, Rank 1 in Vector
        # Document 30: Rank 3 in BM25, not in Vector
        # Document 40: not in BM25, Rank 3 in Vector
        self.mock_bm25.search.return_value = [
            {"doc_id": 10, "score": 10.0, "rank": 1},
            {"doc_id": 20, "score": 8.0, "rank": 2},
            {"doc_id": 30, "score": 6.0, "rank": 3},
        ]
        self.mock_vector.search.return_value = [
            {"doc_id": 20, "score": 0.90, "rank": 1},
            {"doc_id": 10, "score": 0.85, "rank": 2},
            {"doc_id": 40, "score": 0.70, "rank": 3},
        ]

        results = self.retriever.search("test query", top_k=4, rrf_k=60)

        # Expected RRF scores with w_bm = 0.5, w_vec = 0.5:
        # doc 10: 0.5 / (60 + 1) + 0.5 / (60 + 2) = 0.5/61 + 0.5/62 = 0.0081967 + 0.0080645 = 0.016261
        # doc 20: 0.5 / (60 + 2) + 0.5 / (60 + 1) = 0.016261 (tied with doc 10, doc 20 has larger id)
        # doc 30: 0.5 / (60 + 3) = 0.5/63 = 0.0079365
        # doc 40: 0.5 / (60 + 3) = 0.5/63 = 0.0079365
        self.assertEqual(len(results), 4)

        top_ids = [r["doc_id"] for r in results[:2]]
        self.assertIn(10, top_ids)
        self.assertIn(20, top_ids)

        # Both top documents (found by both engines) should score higher than single-engine documents
        self.assertGreater(results[0]["score"], results[2]["score"])
        self.assertGreater(results[1]["score"], results[3]["score"])

        # Check precision of calculation for doc 10
        doc10_res = next(r for r in results if r["doc_id"] == 10)
        expected_score_10 = (0.5 / 61.0) + (0.5 / 62.0)
        self.assertAlmostEqual(doc10_res["score"], expected_score_10, places=5)
        self.assertEqual(doc10_res["bm25_rank"], 1)
        self.assertEqual(doc10_res["vector_rank"], 2)

    def test_weight_tuning_shifts_ranking(self):
        """Verify that shifting weights between BM25 and Vector re-ranks candidates."""
        # Doc 1: BM25 winner (Rank 1 in BM25, Rank 10 in Vector)
        # Doc 2: Vector winner (Rank 10 in BM25, Rank 1 in Vector)
        self.mock_bm25.search.return_value = [
            {"doc_id": 1, "score": 20.0, "rank": 1},
            {"doc_id": 2, "score": 2.0, "rank": 10},
        ]
        self.mock_vector.search.return_value = [
            {"doc_id": 2, "score": 0.95, "rank": 1},
            {"doc_id": 1, "score": 0.40, "rank": 10},
        ]

        # Heavy BM25 weight: 0.8 BM25, 0.2 Vector -> Doc 1 should win
        bm_heavy = self.retriever.search("query", top_k=2, bm25_weight=0.8, vector_weight=0.2)
        self.assertEqual(bm_heavy[0]["doc_id"], 1)

        # Heavy Vector weight: 0.2 BM25, 0.8 Vector -> Doc 2 should win
        vec_heavy = self.retriever.search("query", top_k=2, bm25_weight=0.2, vector_weight=0.8)
        self.assertEqual(vec_heavy[0]["doc_id"], 2)

    def test_linear_score_fusion(self):
        """Verify linear min-max score fusion."""
        self.mock_bm25.search.return_value = [
            {"doc_id": 100, "score": 10.0, "rank": 1},
            {"doc_id": 200, "score": 0.0, "rank": 2},
        ]
        self.mock_vector.search.return_value = [
            {"doc_id": 200, "score": 1.0, "rank": 1},
            {"doc_id": 100, "score": 0.0, "rank": 2},
        ]

        # With 0.5 / 0.5 weights and min-max:
        # doc 100: bm25_norm = 1.0, vec_norm = 0.0 -> score = 0.5
        # doc 200: bm25_norm = 0.0, vec_norm = 1.0 -> score = 0.5
        results = self.retriever.search("query", fusion_method="linear", bm25_weight=0.5, vector_weight=0.5)
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0]["score"], 0.5, places=4)
        self.assertAlmostEqual(results[1]["score"], 0.5, places=4)

        # With 0.7 BM25 / 0.3 Vector:
        # doc 100: 0.7 * 1.0 + 0.3 * 0.0 = 0.70
        # doc 200: 0.7 * 0.0 + 0.3 * 1.0 = 0.30
        results_weighted = self.retriever.search("query", fusion_method="linear", bm25_weight=0.7, vector_weight=0.3)
        self.assertEqual(results_weighted[0]["doc_id"], 100)
        self.assertAlmostEqual(results_weighted[0]["score"], 0.70, places=4)
        self.assertEqual(results_weighted[1]["doc_id"], 200)
        self.assertAlmostEqual(results_weighted[1]["score"], 0.30, places=4)

    def test_explain_fusion(self):
        """Verify explain_fusion returns complete breakdown."""
        self.mock_bm25.search.return_value = [
            {"doc_id": 55, "score": 15.0, "rank": 1}
        ]
        self.mock_vector.search.return_value = [
            {"doc_id": 55, "score": 0.88, "rank": 2}
        ]

        explanation = self.retriever.explain_fusion("pizza", doc_id=55, rrf_k=60)
        self.assertEqual(explanation["doc_id"], 55)
        self.assertTrue(explanation["in_bm25_candidates"])
        self.assertTrue(explanation["in_vector_candidates"])
        self.assertEqual(explanation["bm25_rank"], 1)
        self.assertEqual(explanation["vector_rank"], 2)
        self.assertIn("formula", explanation)
        expected_score = (0.5 / 61.0) + (0.5 / 62.0)
        self.assertAlmostEqual(explanation["final_score"], expected_score, places=5)


if __name__ == "__main__":
    unittest.main()
