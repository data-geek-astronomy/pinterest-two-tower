"""
Retrieval evaluation metrics: Recall@K, NDCG@K, MRR.
"""

import numpy as np


def recall_at_k(retrieved: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """
    Recall@K: fraction of relevant items found in top-K retrieved.

    Args:
        retrieved : (B, K) array of retrieved item IDs (ranked)
        relevant  : (B,)  array of ground-truth positive item IDs
        k         : cutoff
    """
    hits = 0
    for i in range(len(relevant)):
        if relevant[i] in retrieved[i, :k]:
            hits += 1
    return hits / len(relevant)


def ndcg_at_k(retrieved: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """
    NDCG@K: normalized discounted cumulative gain.

    Ideal DCG = 1 (single relevant item at rank 1).
    """
    ndcg = 0.0
    for i in range(len(relevant)):
        pos = np.where(retrieved[i, :k] == relevant[i])[0]
        if len(pos) > 0:
            rank = pos[0] + 1   # 1-indexed
            ndcg += 1.0 / np.log2(rank + 1)
    return ndcg / len(relevant)


def mrr(retrieved: np.ndarray, relevant: np.ndarray) -> float:
    """
    Mean Reciprocal Rank.
    """
    rr = 0.0
    for i in range(len(relevant)):
        pos = np.where(retrieved[i] == relevant[i])[0]
        if len(pos) > 0:
            rr += 1.0 / (pos[0] + 1)
    return rr / len(relevant)


def compute_all_metrics(
    retrieved: np.ndarray,
    relevant: np.ndarray,
    k_values: list[int],
) -> dict:
    """Compute Recall@K and NDCG@K for all K values + MRR."""
    metrics = {}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
    metrics["mrr"] = mrr(retrieved, relevant)
    return metrics
