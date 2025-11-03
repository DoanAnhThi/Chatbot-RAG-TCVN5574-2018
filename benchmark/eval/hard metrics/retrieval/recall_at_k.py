"""
Recall@k Metric for RAG Evaluation

Recall@k measures the proportion of relevant documents that are successfully retrieved
in the top-k results. It evaluates how many of the ground truth relevant documents
are captured within the top-k retrieved documents.

Formula: Recall@k = (Number of relevant documents in top-k) / (Total number of relevant documents)

Range: 0 to 1, where 1 indicates perfect recall (all relevant documents retrieved).
"""

import numpy as np
from typing import List, Set, Union


def recall_at_k(retrieved_docs: List[List[str]],
                ground_truth_docs: List[Set[str]],
                k: int) -> float:
    """
    Calculate Recall@k for a set of queries.

    Args:
        retrieved_docs: List of lists, where each inner list contains document IDs
                       retrieved for a query (ordered by relevance)
        ground_truth_docs: List of sets, where each set contains ground truth
                          relevant document IDs for the corresponding query
        k: Number of top documents to consider

    Returns:
        Average Recall@k across all queries (float between 0 and 1)
    """
    if len(retrieved_docs) != len(ground_truth_docs):
        raise ValueError("retrieved_docs and ground_truth_docs must have the same length")

    if k <= 0:
        raise ValueError("k must be a positive integer")

    recalls = []

    for retrieved, ground_truth in zip(retrieved_docs, ground_truth_docs):
        # Get top-k retrieved documents
        top_k = set(retrieved[:k])

        # Calculate intersection with ground truth
        relevant_retrieved = len(top_k.intersection(ground_truth))

        # Calculate recall: relevant_retrieved / total_relevant
        if len(ground_truth) > 0:
            recall = relevant_retrieved / len(ground_truth)
        else:
            recall = 1.0  # If no ground truth documents, consider it perfect recall

        recalls.append(recall)

    return np.mean(recalls) if recalls else 0.0


def recall_at_k_single_query(retrieved_docs: List[str],
                           ground_truth_docs: Set[str],
                           k: int) -> float:
    """
    Calculate Recall@k for a single query.

    Args:
        retrieved_docs: List of retrieved document IDs (ordered by relevance)
        ground_truth_docs: Set of ground truth relevant document IDs
        k: Number of top documents to consider

    Returns:
        Recall@k score for this query (float between 0 and 1)
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")

    # Get top-k retrieved documents
    top_k = set(retrieved_docs[:k])

    # Calculate intersection with ground truth
    relevant_retrieved = len(top_k.intersection(ground_truth_docs))

    # Calculate recall
    if len(ground_truth_docs) > 0:
        return relevant_retrieved / len(ground_truth_docs)
    else:
        return 1.0  # If no ground truth documents, consider it perfect recall


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

    # Calculate average Recall@3
    avg_recall = recall_at_k(retrieved_docs, ground_truth_docs, k)
    print(".3f")

    # Calculate individual Recall@3 scores
    for i, (retrieved, gt) in enumerate(zip(retrieved_docs, ground_truth_docs)):
        recall = recall_at_k_single_query(retrieved, gt, k)
        print(".3f")
