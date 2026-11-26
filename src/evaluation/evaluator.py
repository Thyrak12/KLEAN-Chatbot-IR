"""Retriever Evaluator for benchmarking Information Retrieval systems.

Evaluates any retriever conforming to the protocol:
  search(query: str, top_k: int) -> List[Dict[str, Any]]

Computes comprehensive IR metrics:
  - Precision @ K (P@1, P@3, P@5, P@10)
  - Recall @ K (Recall@5, Recall@10)
  - F1 @ K
  - Mean Reciprocal Rank (MRR)
  - Mean Average Precision (MAP)
  - NDCG @ 10
  - Mean Latency (ms)
"""

import logging
import time
from typing import Any, Dict, List, Optional, Protocol

import pandas as pd

from src.evaluation.benchmark_dataset import BenchmarkQuery, get_benchmark_queries
from src.evaluation.metrics import (
    average_precision,
    f1_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

logger = logging.getLogger(__name__)


class SearchRetrieverProtocol(Protocol):
    """Protocol for any search retriever object."""

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        ...


class RetrieverEvaluator:
    """Evaluates IR retrievers against ground-truth benchmark queries."""

    def __init__(self, queries: Optional[List[BenchmarkQuery]] = None):
        """Initialize evaluator with benchmark queries.

        Args:
            queries: List of BenchmarkQuery instances. Defaults to standard 20-query benchmark.
        """
        self.queries = queries or get_benchmark_queries()

    def evaluate_retriever(
        self,
        retriever: Any,
        retriever_name: str = "Retriever",
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Evaluate a single retriever across all benchmark queries.

        Args:
            retriever: Search retriever instance with .search() method.
            retriever_name: Display label for retriever.
            top_k: Maximum candidate cutoff rank for retrieval.

        Returns:
            Dict containing mean metrics and per-query evaluation rows.
        """
        query_records: List[Dict[str, Any]] = []

        total_latency_ms = 0.0

        for bq in self.queries:
            t0 = time.perf_counter()
            raw_results = retriever.search(bq.query, top_k=top_k)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            total_latency_ms += latency_ms

            retrieved_ids = [item["doc_id"] for item in raw_results]

            # Compute per-query metrics
            p1 = precision_at_k(retrieved_ids, bq.relevant_doc_ids, k=1)
            p3 = precision_at_k(retrieved_ids, bq.relevant_doc_ids, k=3)
            p5 = precision_at_k(retrieved_ids, bq.relevant_doc_ids, k=5)
            p10 = precision_at_k(retrieved_ids, bq.relevant_doc_ids, k=10)

            r5 = recall_at_k(retrieved_ids, bq.relevant_doc_ids, k=5)
            r10 = recall_at_k(retrieved_ids, bq.relevant_doc_ids, k=10)

            f1_5 = f1_at_k(retrieved_ids, bq.relevant_doc_ids, k=5)
            f1_10 = f1_at_k(retrieved_ids, bq.relevant_doc_ids, k=10)

            rr = reciprocal_rank(retrieved_ids, bq.relevant_doc_ids)
            ap = average_precision(retrieved_ids, bq.relevant_doc_ids, k=top_k)
            ndcg10 = ndcg_at_k(retrieved_ids, bq.relevant_doc_ids, k=10)

            query_records.append({
                "retriever": retriever_name,
                "query_id": bq.query_id,
                "query": bq.query,
                "category": bq.category,
                "num_relevant": len(bq.relevant_doc_ids),
                "num_retrieved": len(retrieved_ids),
                "P@1": p1,
                "P@3": p3,
                "P@5": p5,
                "P@10": p10,
                "Recall@5": r5,
                "Recall@10": r10,
                "F1@5": f1_5,
                "F1@10": f1_10,
                "RR": rr,
                "AP": ap,
                "NDCG@10": ndcg10,
                "latency_ms": latency_ms,
            })

        df = pd.DataFrame(query_records)
        num_q = len(self.queries)

        summary = {
            "retriever": retriever_name,
            "queries_evaluated": num_q,
            "P@1": float(df["P@1"].mean()),
            "P@3": float(df["P@3"].mean()),
            "P@5": float(df["P@5"].mean()),
            "P@10": float(df["P@10"].mean()),
            "Recall@5": float(df["Recall@5"].mean()),
            "Recall@10": float(df["Recall@10"].mean()),
            "F1@5": float(df["F1@5"].mean()),
            "F1@10": float(df["F1@10"].mean()),
            "MRR": float(df["RR"].mean()),
            "MAP": float(df["AP"].mean()),
            "NDCG@10": float(df["NDCG@10"].mean()),
            "mean_latency_ms": float(total_latency_ms / num_q) if num_q > 0 else 0.0,
            "details_df": df,
        }

        return summary

    def compare_retrievers(
        self,
        retrievers: Dict[str, Any],
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Evaluate and compare multiple retrievers side-by-side.

        Args:
            retrievers: Dict of {name: retriever_instance}
            top_k: Retrieval cutoff threshold.

        Returns:
            pd.DataFrame comparing all retrievers across standard metrics.
        """
        summaries: List[Dict[str, Any]] = []

        for name, retriever in retrievers.items():
            logger.info("Evaluating retriever: %s...", name)
            res = self.evaluate_retriever(retriever, retriever_name=name, top_k=top_k)
            # Exclude details dataframe from high-level comparison table
            row = {k: v for k, v in res.items() if k != "details_df"}
            summaries.append(row)

        comparison_df = pd.DataFrame(summaries)
        return comparison_df

    def evaluate_by_category(
        self,
        retrievers: Dict[str, Any],
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Compute MAP and MRR broken down by query category for each retriever.

        Args:
            retrievers: Dict of {name: retriever_instance}
            top_k: Retrieval cutoff.

        Returns:
            pd.DataFrame indexed by query category.
        """
        records: List[Dict[str, Any]] = []

        for name, retriever in retrievers.items():
            res = self.evaluate_retriever(retriever, retriever_name=name, top_k=top_k)
            df = res["details_df"]
            grouped = df.groupby("category").agg({
                "P@5": "mean",
                "Recall@5": "mean",
                "RR": "mean",
                "AP": "mean",
                "NDCG@10": "mean",
            }).reset_index()
            grouped.rename(columns={"RR": "MRR", "AP": "MAP"}, inplace=True)
            grouped["retriever"] = name
            records.extend(grouped.to_dict(orient="records"))

        return pd.DataFrame(records)
