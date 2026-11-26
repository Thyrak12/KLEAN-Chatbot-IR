"""Information retrieval evaluation metrics and benchmarking module."""

from .benchmark_dataset import BenchmarkQuery, get_benchmark_queries, get_queries_by_category
from .evaluator import RetrieverEvaluator
from .metrics import (
    average_precision,
    f1_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "reciprocal_rank",
    "average_precision",
    "ndcg_at_k",
    "BenchmarkQuery",
    "get_benchmark_queries",
    "get_queries_by_category",
    "RetrieverEvaluator",
]
