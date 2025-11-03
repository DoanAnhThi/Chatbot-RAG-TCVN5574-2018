"""
Mean Reciprocal Rank (MRR) Metric for RAG Evaluation

MRR measures the average of the reciprocal ranks of the first relevant document
found for each query. It favors systems that rank relevant documents higher.

Formula: MRR = (1/Q) * Σ(1/rank_i)
where rank_i is the position of the first relevant document for query i,
and Q is the total number of queries.

Range: 0 to 1, where 1 indicates the first relevant document is always ranked #1.
"""

import numpy as np
from typing import List, Set, Union


def mean_reciprocal_rank(retrieved_docs: List[List[str]],
                        ground_truth_docs: List[Set[str]]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) for a set of queries.

    Args:
        retrieved_docs: List of lists, where each inner list contains document IDs
                       retrieved for a query (ordered by relevance)
        ground_truth_docs: List of sets, where each set contains ground truth
                          relevant document IDs for the corresponding query

    Returns:
        MRR score across all queries (float between 0 and 1)
    """
    if len(retrieved_docs) != len(ground_truth_docs):
        raise ValueError("retrieved_docs and ground_truth_docs must have the same length")

    reciprocal_ranks = []

    for retrieved, ground_truth in zip(retrieved_docs, ground_truth_docs):
        # Find the rank of the first relevant document
        reciprocal_rank = 0.0

        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in ground_truth:
                reciprocal_rank = 1.0 / rank
                break

        reciprocal_ranks.append(reciprocal_rank)

    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def reciprocal_rank_single_query(retrieved_docs: List[str],
                               ground_truth_docs: Set[str]) -> float:
    """
    Calculate reciprocal rank for a single query.

    Args:
        retrieved_docs: List of retrieved document IDs (ordered by relevance)
        ground_truth_docs: Set of ground truth relevant document IDs

    Returns:
        Reciprocal rank score for this query (float between 0 and 1)
    """
    for rank, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in ground_truth_docs:
            return 1.0 / rank

    return 0.0  # No relevant document found


def mean_reciprocal_rank_at_k(retrieved_docs: List[List[str]],
                             ground_truth_docs: List[Set[str]],
                             k: int) -> float:
    """
    Calculate Mean Reciprocal Rank limited to top-k results.

    Args:
        retrieved_docs: List of lists, where each inner list contains document IDs
                       retrieved for a query (ordered by relevance)
        ground_truth_docs: List of sets, where each set contains ground truth
                          relevant document IDs for the corresponding query
        k: Maximum rank to consider (only look at top-k results)

    Returns:
        MRR@k score across all queries (float between 0 and 1)
    """
    if len(retrieved_docs) != len(ground_truth_docs):
        raise ValueError("retrieved_docs and ground_truth_docs must have the same length")

    if k <= 0:
        raise ValueError("k must be a positive integer")

    reciprocal_ranks = []

    for retrieved, ground_truth in zip(retrieved_docs, ground_truth_docs):
        # Only consider top-k results
        top_k = retrieved[:k]

        # Find the rank of the first relevant document within top-k
        reciprocal_rank = 0.0

        for rank, doc_id in enumerate(top_k, start=1):
            if doc_id in ground_truth:
                reciprocal_rank = 1.0 / rank
                break

        reciprocal_ranks.append(reciprocal_rank)

    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


# Example usage
if __name__ == "__main__":
    # Example data
    retrieved_docs = [
        ["doc1", "doc2", "doc3", "doc4", "doc5"],  # Query 1: relevant doc1 at rank 1
        ["doc6", "doc2", "doc3", "doc1", "doc7"],  # Query 2: relevant doc2 at rank 2
        ["doc8", "doc9", "doc1", "doc2", "doc3"]   # Query 3: relevant docs are doc8, doc9 - found at ranks 1,2
    ]

    ground_truth_docs = [
        {"doc1", "doc3"},  # Query 1: doc1 and doc3 are relevant
        {"doc2", "doc4"},  # Query 2: doc2 and doc4 are relevant
        {"doc8", "doc9"}   # Query 3: doc8 and doc9 are relevant
    ]

    # Calculate MRR
    mrr_score = mean_reciprocal_rank(retrieved_docs, ground_truth_docs)
    print(".4f")

    # Calculate MRR@3 (only consider top-3 results)
    mrr_at_3 = mean_reciprocal_rank_at_k(retrieved_docs, ground_truth_docs, 3)
    print(".4f")

    # Calculate individual reciprocal ranks
    for i, (retrieved, gt) in enumerate(zip(retrieved_docs, ground_truth_docs)):
        rr = reciprocal_rank_single_query(retrieved, gt)
        print(".4f")
