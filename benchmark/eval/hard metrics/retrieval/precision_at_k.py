"""
Precision@k Metric for RAG Evaluation

Precision@k measures the proportion of retrieved documents that are actually relevant
within the top-k results. It evaluates the accuracy of the retrieval system.

Formula: Precision@k = (Number of relevant documents in top-k) / k

Range: 0 to 1, where 1 indicates all retrieved documents in top-k are relevant.
"""

import numpy as np
from typing import List, Set, Union


def precision_at_k(retrieved_docs: List[List[str]],
                  ground_truth_docs: List[Set[str]],
                  k: int) -> float:
    """
    Calculate Precision@k for a set of queries.

    Args:
        retrieved_docs: List of lists, where each inner list contains document IDs
                       retrieved for a query (ordered by relevance)
        ground_truth_docs: List of sets, where each set contains ground truth
                          relevant document IDs for the corresponding query
        k: Number of top documents to consider

    Returns:
        Average Precision@k across all queries (float between 0 and 1)
    """
    if len(retrieved_docs) != len(ground_truth_docs):
        raise ValueError("retrieved_docs and ground_truth_docs must have the same length")

    if k <= 0:
        raise ValueError("k must be a positive integer")

    precisions = []

    for retrieved, ground_truth in zip(retrieved_docs, ground_truth_docs):
        # Get top-k retrieved documents
        top_k = retrieved[:k]

        # Count relevant documents in top-k
        relevant_count = sum(1 for doc in top_k if doc in ground_truth)

        # Calculate precision: relevant_count / k
        precision = relevant_count / k
        precisions.append(precision)

    return np.mean(precisions) if precisions else 0.0


def precision_at_k_single_query(retrieved_docs: List[str],
                              ground_truth_docs: Set[str],
                              k: int) -> float:
    """
    Calculate Precision@k for a single query.

    Args:
        retrieved_docs: List of retrieved document IDs (ordered by relevance)
        ground_truth_docs: Set of ground truth relevant document IDs
        k: Number of top documents to consider

    Returns:
        Precision@k score for this query (float between 0 and 1)
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")

    # Get top-k retrieved documents
    top_k = retrieved_docs[:k]

    # Count relevant documents in top-k
    relevant_count = sum(1 for doc in top_k if doc in ground_truth_docs)

    # Calculate precision
    return relevant_count / k


# Example usage
if __name__ == "__main__":
    # Example data
    retrieved_docs = [
        ["doc1", "doc2", "doc3", "doc4", "doc5"],  # Query 1
        ["doc2", "doc3", "doc1", "doc6", "doc7"],  # Query 2
        ["doc8", "doc9", "doc1", "doc2", "doc3"]   # Query 3
    ]

    ground_truth_docs = [
        {"doc1", "doc3"},  # Query 1: doc1 and doc3 are relevant
        {"doc2", "doc4"},  # Query 2: doc2 and doc4 are relevant
        {"doc8", "doc9"}   # Query 3: doc8 and doc9 are relevant
    ]

    k = 3

    # Calculate average Precision@3
    avg_precision = precision_at_k(retrieved_docs, ground_truth_docs, k)
    print(".3f")

    # Calculate individual Precision@3 scores
    for i, (retrieved, gt) in enumerate(zip(retrieved_docs, ground_truth_docs)):
        precision = precision_at_k_single_query(retrieved, gt, k)
        print(".3f")
