"""
Hit@k Metric for RAG Evaluation

Hit@k measures the proportion of queries where at least one relevant document
appears in the top-k retrieved results. It's a binary metric (hit or miss).

Formula: Hit@k = 1 if at least one relevant document is in top-k, else 0
Average Hit@k = (Number of queries with hits) / (Total number of queries)

Range: 0 to 1, where 1 indicates all queries have at least one relevant document in top-k.
"""

import numpy as np
from typing import List, Set, Union


def hit_at_k(retrieved_docs: List[List[str]],
            ground_truth_docs: List[Set[str]],
            k: int) -> float:
    """
    Calculate Hit@k for a set of queries.

    Args:
        retrieved_docs: List of lists, where each inner list contains document IDs
                       retrieved for a query (ordered by relevance)
        ground_truth_docs: List of sets, where each set contains ground truth
                          relevant document IDs for the corresponding query
        k: Number of top documents to consider

    Returns:
        Average Hit@k across all queries (float between 0 and 1)
    """
    if len(retrieved_docs) != len(ground_truth_docs):
        raise ValueError("retrieved_docs and ground_truth_docs must have the same length")

    if k <= 0:
        raise ValueError("k must be a positive integer")

    hits = 0

    for retrieved, ground_truth in zip(retrieved_docs, ground_truth_docs):
        # Get top-k retrieved documents
        top_k = set(retrieved[:k])

        # Check if there's at least one relevant document in top-k
        if top_k.intersection(ground_truth):
            hits += 1

    return hits / len(retrieved_docs) if retrieved_docs else 0.0


def hit_at_k_single_query(retrieved_docs: List[str],
                         ground_truth_docs: Set[str],
                         k: int) -> int:
    """
    Calculate Hit@k for a single query.

    Args:
        retrieved_docs: List of retrieved document IDs (ordered by relevance)
        ground_truth_docs: Set of ground truth relevant document IDs
        k: Number of top documents to consider

    Returns:
        Hit@k score for this query (1 if hit, 0 if miss)
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")

    # Get top-k retrieved documents
    top_k = set(retrieved_docs[:k])

    # Check if there's at least one relevant document
    return 1 if top_k.intersection(ground_truth_docs) else 0


def hits_at_multiple_k(retrieved_docs: List[List[str]],
                      ground_truth_docs: List[Set[str]],
                      k_values: List[int]) -> dict:
    """
    Calculate Hit@k for multiple k values.

    Args:
        retrieved_docs: List of lists, where each inner list contains document IDs
                       retrieved for a query (ordered by relevance)
        ground_truth_docs: List of sets, where each set contains ground truth
                          relevant document IDs for the corresponding query
        k_values: List of k values to evaluate

    Returns:
        Dictionary mapping k values to Hit@k scores
    """
    results = {}
    for k in k_values:
        results[k] = hit_at_k(retrieved_docs, ground_truth_docs, k)
    return results


# Example usage
if __name__ == "__main__":
    # Example data
    retrieved_docs = [
        ["doc1", "doc2", "doc3", "doc4", "doc5"],  # Query 1: doc1 is relevant (hit)
        ["doc6", "doc7", "doc2", "doc3", "doc1"],  # Query 2: doc2 is relevant but at rank 3
        ["doc10", "doc11", "doc12", "doc8", "doc9"] # Query 3: no relevant docs in top-5
    ]

    ground_truth_docs = [
        {"doc1", "doc3"},  # Query 1: doc1 and doc3 are relevant
        {"doc2", "doc4"},  # Query 2: doc2 and doc4 are relevant
        {"doc8", "doc9"}   # Query 3: doc8 and doc9 are relevant
    ]

    # Calculate Hit@1, Hit@3, Hit@5
    k_values = [1, 3, 5]
    hits = hits_at_multiple_k(retrieved_docs, ground_truth_docs, k_values)

    for k, hit_score in hits.items():
        print(".3f")

    print("\nIndividual Hit@3 scores:")
    for i, (retrieved, gt) in enumerate(zip(retrieved_docs, ground_truth_docs)):
        hit = hit_at_k_single_query(retrieved, gt, 3)
        print(f"Query {i+1}: {hit}")
