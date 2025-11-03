"""
Cosine similarity utilities for vector operations.
Used for calculating similarity between vectors in retrieval.
"""

import numpy as np
from typing import List, Tuple, Optional


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0.0


def find_top_similar_vectors(
    query_vector: np.ndarray,
    document_vectors: List[np.ndarray],
    top_k: int = 5
) -> List[Tuple[int, float]]:
    """
    Find top-k most similar vectors using cosine similarity

    Args:
        query_vector: Query embedding vector
        document_vectors: List of document embedding vectors
        top_k: Number of top similar vectors to return

    Returns:
        List of (index, similarity_score) tuples sorted by similarity (descending)
    """
    similarities = []
    for i, doc_vec in enumerate(document_vectors):
        sim = cosine_similarity(query_vector, doc_vec)
        similarities.append((i, sim))

    # Sort by similarity score (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def batch_cosine_similarity(
    query_vectors: List[np.ndarray],
    document_vectors: List[np.ndarray]
) -> np.ndarray:
    """
    Calculate cosine similarity between multiple query vectors and document vectors

    Args:
        query_vectors: List of query embedding vectors
        document_vectors: List of document embedding vectors

    Returns:
        2D numpy array of similarity scores (queries x documents)
    """
    # Convert to numpy arrays
    query_matrix = np.array(query_vectors)
    doc_matrix = np.array(document_vectors)

    # Normalize vectors
    query_norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
    doc_norms = np.linalg.norm(doc_matrix, axis=1, keepdims=True)

    # Avoid division by zero
    query_norms = np.where(query_norms == 0, 1, query_norms)
    doc_norms = np.where(doc_norms == 0, 1, doc_norms)

    query_normalized = query_matrix / query_norms
    doc_normalized = doc_matrix / doc_norms

    # Calculate cosine similarity
    similarity_matrix = np.dot(query_normalized, doc_normalized.T)

    return similarity_matrix
