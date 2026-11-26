"""Unit tests for Phase 6: IR Evaluation Metrics and Benchmark Evaluator."""

import math
import unittest
from unittest.mock import MagicMock

from src.evaluation.benchmark_dataset import (
    BenchmarkQuery,
    get_benchmark_queries,
    get_queries_by_category,
)
from src.evaluation.evaluator import RetrieverEvaluator
from src.evaluation.metrics import (
    average_precision,
    f1_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestPhase6EvaluationMetrics(unittest.TestCase):
    """Test suite for standard IR evaluation metrics."""

    def test_precision_at_k(self):
        retrieved = [1, 2, 3, 4, 5]
        relevant = {2, 4, 10}

        # P@1: doc 1 is not in relevant -> 0/1 = 0.0
        self.assertEqual(precision_at_k(retrieved, relevant, k=1), 0.0)
        # P@2: doc 2 is relevant -> 1/2 = 0.5
        self.assertEqual(precision_at_k(retrieved, relevant, k=2), 0.5)
        # P@4: docs 2 and 4 are relevant -> 2/4 = 0.5
        self.assertEqual(precision_at_k(retrieved, relevant, k=4), 0.5)
        # P@5: docs 2 and 4 are relevant -> 2/5 = 0.4
        self.assertEqual(precision_at_k(retrieved, relevant, k=5), 0.4)

    def test_recall_at_k(self):
        retrieved = [1, 2, 3, 4, 5]
        relevant = {2, 4, 10, 20}  # total 4 relevant docs

        # Recall@2: doc 2 is retrieved -> 1/4 = 0.25
        self.assertEqual(recall_at_k(retrieved, relevant, k=2), 0.25)
        # Recall@4: docs 2 and 4 retrieved -> 2/4 = 0.50
        self.assertEqual(recall_at_k(retrieved, relevant, k=4), 0.50)
        # Recall@5: docs 2 and 4 retrieved -> 2/4 = 0.50
        self.assertEqual(recall_at_k(retrieved, relevant, k=5), 0.50)

    def test_f1_at_k(self):
        retrieved = [1, 2, 3, 4]
        relevant = {2, 4}
        # P@4 = 2/4 = 0.5, R@4 = 2/2 = 1.0
        # F1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 1.0 / 1.5 = 2/3
        f1 = f1_at_k(retrieved, relevant, k=4)
        self.assertAlmostEqual(f1, 2.0 / 3.0, places=5)

    def test_reciprocal_rank(self):
        relevant = {10, 20}

        # First relevant doc is at rank 1 -> RR = 1/1 = 1.0
        self.assertEqual(reciprocal_rank([10, 2, 3], relevant), 1.0)
        # First relevant doc is at rank 3 -> RR = 1/3
        self.assertAlmostEqual(reciprocal_rank([1, 2, 10, 4], relevant), 1.0 / 3.0, places=5)
        # No relevant doc retrieved -> RR = 0.0
        self.assertEqual(reciprocal_rank([1, 2, 3, 4], relevant), 0.0)

    def test_average_precision(self):
        # Retrieved: [A, B, C, D, E] where relevant = {A, C, D} (total 3 relevant)
        # Rank 1: A (relevant) -> P@1 = 1/1 = 1.0
        # Rank 2: B (not relevant)
        # Rank 3: C (relevant) -> P@3 = 2/3
        # Rank 4: D (relevant) -> P@4 = 3/4
        # Rank 5: E (not relevant)
        # AP = (1.0 + 2/3 + 3/4) / 3 = (1.0 + 0.6667 + 0.75) / 3 = 2.416667 / 3 = 0.805556
        retrieved = [1, 2, 3, 4, 5]
        relevant = {1, 3, 4}
        expected_ap = (1.0 + (2.0 / 3.0) + (3.0 / 4.0)) / 3.0
        self.assertAlmostEqual(average_precision(retrieved, relevant), expected_ap, places=5)

    def test_ndcg_at_k(self):
        relevant = {1, 3}
        # Retrieved: [1, 2, 3] -> DCG = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5
        # Ideal: [1, 3, 2] -> IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.63092975 = 1.63092975
        # NDCG = 1.5 / 1.63092975 = 0.91972
        retrieved = [1, 2, 3]
        expected_ndcg = (1.0 + 0.5) / (1.0 + (1.0 / math.log2(3)))
        self.assertAlmostEqual(ndcg_at_k(retrieved, relevant, k=3), expected_ndcg, places=4)

        # Perfect ranking NDCG should be 1.0
        self.assertAlmostEqual(ndcg_at_k([1, 3, 2], relevant, k=3), 1.0, places=4)


class TestPhase6BenchmarkDatasetAndEvaluator(unittest.TestCase):
    """Test suite for benchmark dataset and RetrieverEvaluator."""

    def test_benchmark_dataset_integrity(self):
        queries = get_benchmark_queries()
        self.assertEqual(len(queries), 20)

        categories = {q.category for q in queries}
        expected_categories = {
            "exact_entity",
            "cuisine_dish",
            "ambiance_concept",
            "location_neighborhood",
            "khmer_script",
        }
        self.assertEqual(categories, expected_categories)

        for q in queries:
            self.assertTrue(len(q.query.strip()) > 0)
            self.assertTrue(len(q.relevant_doc_ids) > 0)

        # Test filtering by category
        khmer_queries = get_queries_by_category("khmer_script")
        self.assertEqual(len(khmer_queries), 4)

    def test_retriever_evaluator_pipeline(self):
        # Create a mock retriever that returns fixed answers
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"doc_id": 1, "score": 10.0, "rank": 1},
            {"doc_id": 2, "score": 8.0, "rank": 2},
        ]

        test_query = BenchmarkQuery(
            query_id="TEST01",
            query="test query",
            category="exact_entity",
            description="Unit test query",
            relevant_doc_ids={1},
        )

        evaluator = RetrieverEvaluator(queries=[test_query])
        result = evaluator.evaluate_retriever(mock_retriever, retriever_name="MockEngine")

        self.assertEqual(result["retriever"], "MockEngine")
        self.assertEqual(result["queries_evaluated"], 1)
        self.assertEqual(result["P@1"], 1.0)
        self.assertEqual(result["Recall@5"], 1.0)
        self.assertEqual(result["MRR"], 1.0)
        self.assertEqual(result["MAP"], 1.0)
        self.assertIn("details_df", result)


if __name__ == "__main__":
    unittest.main()
