"""
METEOR Metric for RAG Generation Evaluation

METEOR (Metric for Evaluation of Translation with Explicit ORdering) measures
the quality of generated text by considering exact matches, stemmed matches,
synonym matches, and word order. It provides a more nuanced evaluation than
BLEU by considering linguistic features.

Formula: METEOR = F_mean * (1 - penalty)
where F_mean is the harmonic mean of precision and recall,
and penalty accounts for word order differences.

Range: 0 to 1, where 1 indicates identical texts.
"""

try:
    from nltk.translate.meteor_score import meteor_score
    from nltk import word_tokenize
except ImportError:
    print("Warning: nltk library not found. Install with: pip install nltk")
    meteor_score = None

from typing import List, Union
import re


def meteor_score_calc(predictions: Union[str, List[str]],
                     references: Union[str, List[str]],
                     alpha: float = 0.9,
                     beta: float = 3.0,
                     gamma: float = 0.5) -> Union[float, List[float]]:
    """
    Calculate METEOR score between predictions and references.

    Args:
        predictions: Generated answers (string or list of strings)
        references: Ground truth answers (string or list of strings)
        alpha: Weight for precision (default=0.9)
        beta: Weight for fragmentation penalty (default=3.0)
        gamma: Weight for length penalty (default=0.5)

    Returns:
        METEOR score(s) (float or list of floats)
    """
    if meteor_score is None:
        raise ImportError("nltk library is required. Install with: pip install nltk")

    # Handle single string inputs
    if isinstance(predictions, str) and isinstance(references, str):
        pred_tokens = word_tokenize(predictions.lower())
        ref_tokens = word_tokenize(references.lower())
        return meteor_score([ref_tokens], pred_tokens, alpha=alpha, beta=beta, gamma=gamma)

    # Handle list inputs
    elif isinstance(predictions, list) and isinstance(references, list):
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length")

        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens = word_tokenize(ref.lower())
            score = meteor_score([ref_tokens], pred_tokens, alpha=alpha, beta=beta, gamma=gamma)
            scores.append(score)

        return scores

    else:
        raise ValueError("predictions and references must both be strings or both be lists")


def _simple_meteor(prediction: str, reference: str) -> float:
    """
    Simple METEOR implementation without NLTK.
    This is a basic approximation focusing on unigram overlap.

    Args:
        prediction: Generated answer
        reference: Ground truth answer

    Returns:
        METEOR-like score
    """
    # Simple tokenization
    def tokenize(text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()

    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    # Calculate unigram precision and recall
    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)

    intersection = pred_set.intersection(ref_set)

    precision = len(intersection) / len(pred_set) if pred_set else 0.0
    recall = len(intersection) / len(ref_set) if ref_set else 0.0

    # F-mean (harmonic mean)
    if precision + recall == 0:
        f_mean = 0.0
    else:
        f_mean = (10 * precision * recall) / (9 * precision + recall)  # Standard METEOR weighting

    # Simple fragmentation penalty (approximation)
    # Count chunks of consecutive matches
    pred_list = pred_tokens
    ref_list = ref_tokens

    matches = []
    for word in pred_list:
        matches.append(1 if word in ref_set else 0)

    # Count transitions (simple fragmentation measure)
    if len(matches) > 1:
        transitions = sum(1 for i in range(1, len(matches)) if matches[i] != matches[i-1])
        penalty = transitions / len(matches) if matches else 0.0
    else:
        penalty = 0.0

    # Apply penalty
    return f_mean * (1 - penalty)


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
        # Calculate METEOR scores
        meteor_scores = meteor_score_calc(predictions, references)
        print("METEOR scores:", meteor_scores)
        print(".4f")

        # Individual scores
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            score = meteor_score_calc(pred, ref)
            print(f"Sample {i+1}: METEOR = {score:.4f}")

    except ImportError:
        # Fallback to simple implementation
        print("Using simple METEOR implementation:")
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            score = _simple_meteor(pred, ref)
            print(f"Sample {i+1}: METEOR-like = {score:.4f}")
