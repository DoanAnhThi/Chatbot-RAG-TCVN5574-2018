"""
BLEU Metric for RAG Generation Evaluation

BLEU (Bilingual Evaluation Understudy) measures n-gram overlap between
generated answers and reference answers. It evaluates precision of n-grams
and includes a brevity penalty for short generations.

Formula: BLEU = BP * exp(∑(1/n) * log(precision_n))
where BP is the brevity penalty, and precision_n is n-gram precision.

Range: 0 to 1, where 1 indicates identical texts.
"""

try:
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
    from nltk import word_tokenize
except ImportError:
    print("Warning: nltk library not found. Install with: pip install nltk")
    sentence_bleu = None
    corpus_bleu = None

from typing import List, Union
import math


def bleu_score(predictions: Union[str, List[str]],
              references: Union[str, List[str]],
              n_gram: int = 4) -> Union[float, List[float]]:
    """
    Calculate BLEU score between predictions and references.

    Args:
        predictions: Generated answers (string or list of strings)
        references: Ground truth answers (string or list of strings)
        n_gram: Maximum n-gram to consider (1-4, default=4)

    Returns:
        BLEU score(s) (float or list of floats)
    """
    if sentence_bleu is None:
        raise ImportError("nltk library is required. Install with: pip install nltk")

    if n_gram < 1 or n_gram > 4:
        raise ValueError("n_gram must be between 1 and 4")

    weights = [1.0/n_gram] * n_gram

    # Handle single string inputs
    if isinstance(predictions, str) and isinstance(references, str):
        pred_tokens = word_tokenize(predictions.lower())
        ref_tokens = word_tokenize(references.lower())
        return sentence_bleu([ref_tokens], pred_tokens, weights=weights)

    # Handle list inputs
    elif isinstance(predictions, list) and isinstance(references, list):
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length")

        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens = word_tokenize(ref.lower())
            score = sentence_bleu([ref_tokens], pred_tokens, weights=weights)
            scores.append(score)

        return scores

    else:
        raise ValueError("predictions and references must both be strings or both be lists")


def bleu_1gram(predictions: Union[str, List[str]],
              references: Union[str, List[str]]) -> Union[float, List[float]]:
    """Calculate BLEU-1 (unigram) score."""
    return bleu_score(predictions, references, n_gram=1)


def bleu_2gram(predictions: Union[str, List[str]],
              references: Union[str, List[str]]) -> Union[float, List[float]]:
    """Calculate BLEU-2 (bigram) score."""
    return bleu_score(predictions, references, n_gram=2)


def bleu_3gram(predictions: Union[str, List[str]],
              references: Union[str, List[str]]) -> Union[float, List[float]]:
    """Calculate BLEU-3 (trigram) score."""
    return bleu_score(predictions, references, n_gram=3)


def bleu_4gram(predictions: Union[str, List[str]],
              references: Union[str, List[str]]) -> Union[float, List[float]]:
    """Calculate BLEU-4 score."""
    return bleu_score(predictions, references, n_gram=4)


def _simple_bleu(prediction: str, reference: str, n_gram: int = 4) -> float:
    """
    Simple BLEU implementation without NLTK.
    Use this if NLTK is not available.

    Args:
        prediction: Generated answer
        reference: Ground truth answer
        n_gram: Maximum n-gram to consider

    Returns:
        BLEU score
    """
    # Simple tokenization (split by whitespace and remove punctuation)
    import re

    def tokenize(text):
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()

    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    # Calculate n-gram precisions
    precisions = []
    for n in range(1, min(n_gram + 1, len(pred_tokens) + 1)):
        pred_ngrams = _get_ngrams(pred_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)

        if not pred_ngrams:
            precisions.append(0.0)
            continue

        # Count matching n-grams
        pred_counts = {}
        ref_counts = {}

        for ngram in pred_ngrams:
            pred_counts[ngram] = pred_counts.get(ngram, 0) + 1

        for ngram in ref_ngrams:
            ref_counts[ngram] = ref_counts.get(ngram, 0) + 1

        # Calculate clipped counts
        clipped_counts = 0
        for ngram, count in pred_counts.items():
            clipped_counts += min(count, ref_counts.get(ngram, 0))

        precision = clipped_counts / len(pred_ngrams)
        precisions.append(precision)

    if not precisions:
        return 0.0

    # Geometric mean of precisions
    log_precision_sum = sum(math.log(p) for p in precisions if p > 0)
    if len(precisions) == 0:
        return 0.0

    geometric_mean = math.exp(log_precision_sum / len(precisions))

    # Brevity penalty
    pred_len = len(pred_tokens)
    ref_len = len(ref_tokens)

    if pred_len > ref_len:
        brevity_penalty = 1.0
    else:
        if pred_len == 0:
            brevity_penalty = 0.0
        else:
            brevity_penalty = math.exp(1 - ref_len / pred_len)

    return brevity_penalty * geometric_mean


def _get_ngrams(tokens: List[str], n: int) -> List[tuple]:
    """Get n-grams from token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


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
        # Calculate BLEU scores
        bleu1_scores = bleu_1gram(predictions, references)
        bleu2_scores = bleu_2gram(predictions, references)
        bleu4_scores = bleu_4gram(predictions, references)

        print("BLEU-1 scores:", bleu1_scores)
        print(".4f")
        print("BLEU-2 scores:", bleu2_scores)
        print(".4f")
        print("BLEU-4 scores:", bleu4_scores)
        print(".4f")

        # Individual BLEU-4 scores
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            score = bleu_score(pred, ref, n_gram=4)
            print(f"Sample {i+1}: BLEU-4 = {score:.4f}")

    except ImportError:
        # Fallback to simple implementation
        print("Using simple BLEU implementation:")
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            score = _simple_bleu(pred, ref, n_gram=4)
            print(f"Sample {i+1}: BLEU-4 = {score:.4f}")
