"""
Document ranking utilities for retrieval pipeline.
Includes reranking logic and relevance scoring.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from langchain_core.documents import Document


def rank_documents_by_similarity(
    docs: List[Document],
    query: str,
    similarity_scores: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """
    Rank documents by their similarity scores

    Args:
        docs: List of documents
        query: Original query string
        similarity_scores: Pre-computed similarity scores (optional)

    Returns:
        List of dicts with document, score, and rank info
    """
    if similarity_scores is None:
        # If no scores provided, assume equal relevance
        similarity_scores = [0.5] * len(docs)

    ranked_results = []
    for i, (doc, score) in enumerate(zip(docs, similarity_scores)):
        ranked_results.append({
            "document": doc,
            "similarity_score": score,
            "rank": i + 1,
            "query": query
        })

    # Sort by similarity score (descending)
    ranked_results.sort(key=lambda x: x["similarity_score"], reverse=True)

    # Update ranks after sorting
    for i, result in enumerate(ranked_results):
        result["rank"] = i + 1

    return ranked_results


def rerank_with_diversity(
    docs: List[Document],
    query: str,
    similarity_scores: List[float],
    diversity_factor: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Rerank documents considering both relevance and diversity

    Args:
        docs: List of documents
        query: Query string
        similarity_scores: Similarity scores for each document
        diversity_factor: Weight for diversity (0-1, higher = more diverse)

    Returns:
        Reranked documents with diversity consideration
    """
    if len(docs) <= 1:
        return rank_documents_by_similarity(docs, query, similarity_scores)

    # Calculate diversity scores (simplified - could use more sophisticated methods)
    diversity_scores = []
    for i, doc1 in enumerate(docs):
        diversity = 0
        for j, doc2 in enumerate(docs):
            if i != j:
                # Simple diversity based on content length difference
                content_diff = abs(len(doc1.page_content) - len(doc2.page_content))
                diversity += content_diff
        diversity_scores.append(diversity / len(docs))

    # Normalize diversity scores
    if diversity_scores:
        max_div = max(diversity_scores)
        diversity_scores = [d / max_div if max_div > 0 else 0 for d in diversity_scores]

    # Combine relevance and diversity
    combined_scores = []
    for sim_score, div_score in zip(similarity_scores, diversity_scores):
        combined_score = (1 - diversity_factor) * sim_score + diversity_factor * div_score
        combined_scores.append(combined_score)

    return rank_documents_by_similarity(docs, query, combined_scores)


def filter_documents_by_threshold(
    docs: List[Document],
    similarity_scores: List[float],
    threshold: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Filter documents below similarity threshold

    Args:
        docs: List of documents
        similarity_scores: Corresponding similarity scores
        threshold: Minimum similarity score to keep

    Returns:
        Filtered documents above threshold
    """
    filtered_docs = []
    filtered_scores = []

    for doc, score in zip(docs, similarity_scores):
        if score >= threshold:
            filtered_docs.append(doc)
            filtered_scores.append(score)

    return rank_documents_by_similarity(filtered_docs, "", filtered_scores)


def calculate_relevance_score(
    doc: Document,
    query: str,
    similarity_score: float
) -> Dict[str, Any]:
    """
    Calculate comprehensive relevance score for a document

    Args:
        doc: Document to score
        query: Query string
        similarity_score: Vector similarity score

    Returns:
        Dict with various relevance metrics
    """
    # Simple relevance calculation - can be extended
    content_length = len(doc.page_content)
    query_terms = set(query.lower().split())
    doc_terms = set(doc.page_content.lower().split())

    # Term overlap
    term_overlap = len(query_terms.intersection(doc_terms))
    term_overlap_ratio = term_overlap / len(query_terms) if query_terms else 0

    # Length appropriateness (prefer medium-length docs)
    optimal_length = 500
    length_score = 1 - abs(content_length - optimal_length) / optimal_length
    length_score = max(0, length_score)

    # Combine scores
    final_score = 0.7 * similarity_score + 0.2 * term_overlap_ratio + 0.1 * length_score

    return {
        "similarity_score": similarity_score,
        "term_overlap_ratio": term_overlap_ratio,
        "length_score": length_score,
        "final_score": final_score,
        "content_length": content_length,
        "query_terms_found": term_overlap
    }
