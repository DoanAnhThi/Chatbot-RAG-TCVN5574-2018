"""
Cosine Similarity Metric for RAG Generation Evaluation

Cosine Similarity measures the semantic similarity between generated answers
and reference answers using vector embeddings. It computes the cosine of the
angle between two vectors in high-dimensional space.

Formula: Cosine Similarity = (A • B) / (||A|| * ||B||)
where A and B are embedding vectors.

Range: -1 to 1, where 1 indicates identical semantic meaning,
0 indicates orthogonal (unrelated) vectors, and -1 indicates opposite meaning.
"""

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Warning: sentence-transformers and scikit-learn not found.")
    print("Install with: pip install sentence-transformers scikit-learn")
    SentenceTransformer = None

from typing import List, Union, Optional


class CosineSimilarityScorer:
    """Cosine similarity calculator using sentence embeddings."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize with specified embedding model.

        Args:
            model_name: Name of the sentence transformer model to use
        """
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers scikit-learn")

        self.model = SentenceTransformer(model_name)

    def score(self, predictions: Union[str, List[str]],
             references: Union[str, List[str]]) -> Union[float, List[float]]:
        """
        Calculate cosine similarity between predictions and references.

        Args:
            predictions: Generated answers (string or list of strings)
            references: Ground truth answers (string or list of strings)

        Returns:
            Cosine similarity score(s) (float between -1 and 1, or list of floats)
        """
        # Handle single string inputs
        if isinstance(predictions, str) and isinstance(references, str):
            pred_embedding = self.model.encode([predictions])[0]
            ref_embedding = self.model.encode([references])[0]
            return float(cosine_similarity([pred_embedding], [ref_embedding])[0][0])

        # Handle list inputs
        elif isinstance(predictions, list) and isinstance(references, list):
            if len(predictions) != len(references):
                raise ValueError("predictions and references must have the same length")

            pred_embeddings = self.model.encode(predictions)
            ref_embeddings = self.model.encode(references)

            similarities = cosine_similarity(pred_embeddings, ref_embeddings)
            # Get diagonal elements (pairwise similarities)
            return [float(similarities[i, i]) for i in range(len(predictions))]

        else:
            raise ValueError("predictions and references must both be strings or both be lists")

    def pairwise_score(self, predictions: List[str],
                      references: List[str]) -> np.ndarray:
        """
        Calculate pairwise cosine similarities between all predictions and references.

        Args:
            predictions: List of generated answers
            references: List of ground truth answers

        Returns:
            Similarity matrix (predictions x references)
        """
        pred_embeddings = self.model.encode(predictions)
        ref_embeddings = self.model.encode(references)

        return cosine_similarity(pred_embeddings, ref_embeddings)


def cosine_similarity_score(predictions: Union[str, List[str]],
                          references: Union[str, List[str]],
                          model_name: str = 'all-MiniLM-L6-v2') -> Union[float, List[float]]:
    """
    Calculate cosine similarity using sentence embeddings.

    Args:
        predictions: Generated answers
        references: Ground truth answers
        model_name: Embedding model to use

    Returns:
        Cosine similarity score(s)
    """
    scorer = CosineSimilarityScorer(model_name)
    return scorer.score(predictions, references)


def _simple_cosine_similarity(text1: str, text2: str) -> float:
    """
    Simple cosine similarity using basic word frequency vectors.
    Use this as fallback when sentence-transformers is not available.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Cosine similarity score
    """
    import re
    from collections import Counter
    from math import sqrt

    def tokenize(text):
        # Simple tokenization: lowercase, remove punctuation, split by space
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()

    def get_vector(text):
        tokens = tokenize(text)
        return Counter(tokens)

    vec1 = get_vector(text1)
    vec2 = get_vector(text2)

    # Get all unique words
    all_words = set(vec1.keys()) | set(vec2.keys())

    # Create frequency vectors
    v1 = [vec1.get(word, 0) for word in all_words]
    v2 = [vec2.get(word, 0) for word in all_words]

    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(v1, v2))

    # Calculate magnitudes
    mag1 = sqrt(sum(a * a for a in v1))
    mag2 = sqrt(sum(b * b for b in v2))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)


# Example usage
if __name__ == "__main__":
    predictions = [
        "The capital of France is Paris.",
        "Machine learning is a subset of artificial intelligence.",
        "The Earth orbits around the Sun."
    ]

    references = [
        "Paris is the capital of France.",
        "Machine learning is part of AI.",
        "The Earth revolves around the Sun."
    ]

    try:
        # Calculate cosine similarity
        similarities = cosine_similarity_score(predictions, references)
        print("Cosine similarities:", similarities)
        print(".4f")

        # Individual scores
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            score = cosine_similarity_score(pred, ref)
            print(f"Sample {i+1}: Cosine similarity = {score:.4f}")

    except ImportError as e:
        # Fallback to simple implementation
        print(f"Cannot use sentence-transformers: {e}")
        print("Using simple word-frequency cosine similarity:")

        for i, (pred, ref) in enumerate(zip(predictions, references)):
            score = _simple_cosine_similarity(pred, ref)
            print(f"Sample {i+1}: Simple cosine similarity = {score:.4f}")
