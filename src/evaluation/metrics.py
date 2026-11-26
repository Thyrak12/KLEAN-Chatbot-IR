"""Evaluation metrics for Information Retrieval systems in KLEAN Chatbot.

Implements standard IR ranking evaluation metrics:
  - Precision@K (P@K)
  - Recall@K (R@K)
  - F1@K
  - Reciprocal Rank (RR) -> Mean Reciprocal Rank (MRR)
  - Average Precision (AP) -> Mean Average Precision (MAP)
  - Normalized Discounted Cumulative Gain (NDCG@K)
"""

import math
from typing import Dict, Iterable, List, Optional, Sequence, Set, Union


def precision_at_k(
    retrieved_ids: Sequence[int],
    relevant_ids: Set[int],
    k: int = 10,
) -> float:
    """Calculate Precision at rank cutoff K.

    P@K = (number of relevant documents in top K) / K

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: Set of ground-truth relevant document IDs.
        k: Rank cutoff threshold (k > 0).

    Returns:
        Precision score between 0.0 and 1.0.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not retrieved_ids or not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_retrieved / float(k)


def recall_at_k(
    retrieved_ids: Sequence[int],
    relevant_ids: Set[int],
    k: int = 10,
) -> float:
    """Calculate Recall at rank cutoff K.

    Recall@K = (number of relevant documents in top K) / (total number of relevant documents)

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: Set of ground-truth relevant document IDs.
        k: Rank cutoff threshold.

    Returns:
        Recall score between 0.0 and 1.0.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not relevant_ids:
        return 0.0
    if not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_retrieved / float(len(relevant_ids))


def f1_at_k(
    retrieved_ids: Sequence[int],
    relevant_ids: Set[int],
    k: int = 10,
) -> float:
    """Calculate F1 measure at rank cutoff K (harmonic mean of P@K and Recall@K)."""
    p = precision_at_k(retrieved_ids, relevant_ids, k=k)
    r = recall_at_k(retrieved_ids, relevant_ids, k=k)
    if (p + r) == 0.0:
        return 0.0
    return 2.0 * (p * r) / (p + r)


def reciprocal_rank(
    retrieved_ids: Sequence[int],
    relevant_ids: Set[int],
) -> float:
    """Calculate Reciprocal Rank (RR) for a single query.

    RR = 1 / rank_of_first_relevant_document
    If no relevant document is found in retrieved_ids, returns 0.0.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: Set of ground-truth relevant document IDs.

    Returns:
        Reciprocal rank between 0.0 and 1.0.
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / float(rank)
    return 0.0


def average_precision(
    retrieved_ids: Sequence[int],
    relevant_ids: Set[int],
    k: Optional[int] = None,
) -> float:
    """Calculate Average Precision (AP) for a single query.

    AP = (1 / |relevant|) * sum_{i=1}^K [ P@i * I(doc_i in relevant) ]

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: Set of ground-truth relevant document IDs.
        k: Optional cutoff threshold. If None, considers all retrieved items.

    Returns:
        Average Precision score between 0.0 and 1.0.
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0

    items = retrieved_ids[:k] if k is not None else retrieved_ids
    hits = 0
    sum_precisions = 0.0

    for rank, doc_id in enumerate(items, start=1):
        if doc_id in relevant_ids:
            hits += 1
            sum_precisions += hits / float(rank)

    return sum_precisions / float(len(relevant_ids))


def ndcg_at_k(
    retrieved_ids: Sequence[int],
    relevant_ids: Set[int],
    k: int = 10,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at rank cutoff K (binary relevance).

    DCG@K = sum_{i=1}^K [ rel_i / log2(i + 1) ]
    IDCG@K = sum_{i=1}^{min(K, |relevant|)} [ 1 / log2(i + 1) ]
    NDCG@K = DCG@K / IDCG@K

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: Set of ground-truth relevant document IDs.
        k: Rank cutoff threshold.

    Returns:
        NDCG@K score between 0.0 and 1.0.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not retrieved_ids or not relevant_ids:
        return 0.0

    items = retrieved_ids[:k]

    # Compute DCG@K
    dcg = 0.0
    for rank, doc_id in enumerate(items, start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    # Compute Ideal DCG@K (all relevant documents ranked at the top)
    ideal_count = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg
