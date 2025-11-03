"""
nDCG@k (Normalized Discounted Cumulative Gain) Metric for RAG Evaluation

nDCG@k measures the quality of ranking by considering both the relevance of documents
and their position in the ranking. It discounts the gain of relevant documents
that appear lower in the ranking.

Formula: nDCG@k = DCG@k / IDCG@k
where DCG@k is the Discounted Cumulative Gain for top-k results,
and IDCG@k is the Ideal DCG@k (best possible ranking).

Range: 0 to 1, where 1 indicates perfect ranking of all relevant documents at the top.
"""

import numpy as np
import math
from typing import List, Dict, Set, Union


def ndcg_at_k(retrieved_docs: List[List[str]],
             ground_truth_docs: List[Dict[str, float]],
             k: int) -> float:
    """
    Calculate nDCG@k for a set of queries with graded relevance.

    Args:
        retrieved_docs: List of lists, where each inner list contains document IDs
                       retrieved for a query (ordered by relevance)
        ground_truth_docs: List of dictionaries, where each dict maps document IDs
                          to their relevance scores (typically 0, 1, 2, 3, etc.)
        k: Number of top documents to consider

    Returns:
        Average nDCG@k across all queries (float between 0 and 1)
    """
    if len(retrieved_docs) != len(ground_truth_docs):
        raise ValueError("retrieved_docs and ground_truth_docs must have the same length")

    if k <= 0:
        raise ValueError("k must be a positive integer")

    ndcg_scores = []

    for retrieved, relevance_scores in zip(retrieved_docs, ground_truth_docs):
        # Calculate DCG@k for retrieved results
        dcg = _calculate_dcg(retrieved[:k], relevance_scores)

        # Calculate IDCG@k for ideal ranking
        ideal_retrieved = _get_ideal_ranking(relevance_scores, k)
        idcg = _calculate_dcg(ideal_retrieved, relevance_scores)

        # Calculate nDCG@k
        if idcg > 0:
            ndcg = dcg / idcg
        else:
            ndcg = 1.0  # Perfect score if no relevant documents

        ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def ndcg_at_k_binary(retrieved_docs: List[List[str]],
                    ground_truth_docs: List[Set[str]],
                    k: int) -> float:
    """
    Calculate nDCG@k for binary relevance (relevant/not relevant).

    Args:
        retrieved_docs: List of lists, where each inner list contains document IDs
                       retrieved for a query (ordered by relevance)
        ground_truth_docs: List of sets, where each set contains ground truth
                          relevant document IDs for the corresponding query
        k: Number of top documents to consider

    Returns:
        Average nDCG@k across all queries (float between 0 and 1)
    """
    # Convert binary relevance to graded relevance (1 for relevant, 0 for not relevant)
    graded_relevance = []
    for gt_set in ground_truth_docs:
        relevance_dict = {}
        for doc_id in gt_set:
            relevance_dict[doc_id] = 1.0  # Binary relevance
        graded_relevance.append(relevance_dict)

    return ndcg_at_k(retrieved_docs, graded_relevance, k)


def _calculate_dcg(retrieved_docs: List[str], relevance_scores: Dict[str, float]) -> float:
    """
    Calculate Discounted Cumulative Gain for a list of retrieved documents.

    Args:
        retrieved_docs: List of retrieved document IDs in ranking order
        relevance_scores: Dictionary mapping document IDs to relevance scores

    Returns:
        DCG score
    """
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_docs):
        rank = i + 1  # rank starts from 1
        relevance = relevance_scores.get(doc_id, 0.0)
        dcg += relevance / math.log2(rank + 1)  # log2(rank + 1)
    return dcg


def _get_ideal_ranking(relevance_scores: Dict[str, float], k: int) -> List[str]:
    """
    Get the ideal ranking of documents based on relevance scores.

    Args:
        relevance_scores: Dictionary mapping document IDs to relevance scores
        k: Number of top documents to return

    Returns:
        List of document IDs in ideal ranking order (highest relevance first)
    """
    # Sort documents by relevance score (descending)
    sorted_docs = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs[:k]]


def ndcg_at_k_single_query(retrieved_docs: List[str],
                          relevance_scores: Dict[str, float],
                          k: int) -> float:
    """
    Calculate nDCG@k for a single query with graded relevance.

    Args:
        retrieved_docs: List of retrieved document IDs (ordered by relevance)
        relevance_scores: Dictionary mapping document IDs to relevance scores
        k: Number of top documents to consider

    Returns:
        nDCG@k score for this query (float between 0 and 1)
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")

    # Calculate DCG@k
    dcg = _calculate_dcg(retrieved_docs[:k], relevance_scores)

    # Calculate IDCG@k
    ideal_retrieved = _get_ideal_ranking(relevance_scores, k)
    idcg = _calculate_dcg(ideal_retrieved, relevance_scores)

    # Calculate nDCG@k
    if idcg > 0:
        return dcg / idcg
    else:
        return 1.0  # Perfect score if no relevant documents


# Example usage
if __name__ == "__main__":
    # Example data with graded relevance
    retrieved_docs = [
        ["doc1", "doc2", "doc3", "doc4", "doc5"],  # Query 1
        ["doc6", "doc2", "doc3", "doc1", "doc7"],  # Query 2
        ["doc10", "doc11", "doc8", "doc9", "doc12"] # Query 3
    ]

    # Graded relevance scores (higher = more relevant)
    ground_truth_relevance = [
        {"doc1": 3.0, "doc3": 2.0, "doc5": 1.0},  # Query 1
        {"doc2": 3.0, "doc4": 2.0, "doc1": 1.0},  # Query 2
        {"doc8": 3.0, "doc9": 3.0, "doc12": 1.0}  # Query 3
    ]

    k = 3

    # Calculate average nDCG@3
    avg_ndcg = ndcg_at_k(retrieved_docs, ground_truth_relevance, k)
    print(".4f")

    # Calculate individual nDCG@3 scores
    for i, (retrieved, relevance) in enumerate(zip(retrieved_docs, ground_truth_relevance)):
        ndcg = ndcg_at_k_single_query(retrieved, relevance, k)
        print(".4f")

    print("\nBinary relevance example:")
    # Convert to binary relevance for comparison
    binary_relevance = [
        {"doc1", "doc3"},  # Query 1
        {"doc2", "doc4"},  # Query 2
        {"doc8", "doc9"}   # Query 3
    ]

    avg_ndcg_binary = ndcg_at_k_binary(retrieved_docs, binary_relevance, k)
    print(".4f")
