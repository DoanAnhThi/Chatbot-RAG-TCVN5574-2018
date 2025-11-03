"""
F1 Score (Token-level) Metric for RAG End-to-End Evaluation

F1 Score measures the balance between precision and recall at the token level.
It's commonly used for evaluating generated text against ground truth, especially
for tasks where exact word matching is important.

Formula: F1 = 2 * (Precision * Recall) / (Precision + Recall)
where Precision = (tokens in prediction ∩ tokens in ground truth) / |prediction tokens|
and Recall = (tokens in prediction ∩ tokens in ground truth) / |ground truth tokens|

Range: 0 to 1, where 1 indicates perfect token-level match.
"""

from typing import List, Union, Tuple
import re
from collections import Counter


def f1_score(predictions: Union[str, List[str]],
            ground_truths: Union[str, List[str]],
            tokenize: bool = True) -> Union[float, List[float]]:
    """
    Calculate F1 score at token level.

    Args:
        predictions: Generated answers (string or list of strings)
        ground_truths: Ground truth answers (string or list of strings)
        tokenize: Whether to tokenize text or treat as character-level

    Returns:
        F1 score(s) (0-1)
    """
    # Handle single string inputs
    if isinstance(predictions, str) and isinstance(ground_truths, str):
        return _calculate_f1_single(predictions, ground_truths, tokenize)

    # Handle list inputs
    elif isinstance(predictions, list) and isinstance(ground_truths, list):
        if len(predictions) != len(ground_truths):
            raise ValueError("predictions and ground_truths must have the same length")

        scores = []
        for pred, gt in zip(predictions, ground_truths):
            score = _calculate_f1_single(pred, gt, tokenize)
            scores.append(score)

        return scores

    else:
        raise ValueError("predictions and ground_truths must both be strings or both be lists")


def _calculate_f1_single(prediction: str, ground_truth: str, tokenize: bool = True) -> float:
    """
    Calculate F1 score for a single prediction-ground truth pair.

    Args:
        prediction: Generated answer
        ground_truth: Ground truth answer
        tokenize: Whether to use token-level or character-level comparison

    Returns:
        F1 score (0-1)
    """
    if tokenize:
        # Token-level F1
        pred_tokens = _tokenize(prediction)
        gt_tokens = _tokenize(ground_truth)

        if not gt_tokens and not pred_tokens:
            return 1.0
        if not gt_tokens or not pred_tokens:
            return 0.0

        # Count tokens
        pred_counter = Counter(pred_tokens)
        gt_counter = Counter(gt_tokens)

        # Calculate intersection
        intersection = 0
        for token in pred_counter:
            intersection += min(pred_counter[token], gt_counter.get(token, 0))

        # Calculate precision and recall
        precision = intersection / len(pred_tokens) if pred_tokens else 0.0
        recall = intersection / len(gt_tokens) if gt_tokens else 0.0

    else:
        # Character-level F1
        pred_chars = list(prediction)
        gt_chars = list(ground_truth)

        if not gt_chars and not pred_chars:
            return 1.0
        if not gt_chars or not pred_chars:
            return 0.0

        # Count characters
        pred_counter = Counter(pred_chars)
        gt_counter = Counter(gt_chars)

        # Calculate intersection
        intersection = 0
        for char in pred_counter:
            intersection += min(pred_counter[char], gt_counter.get(char, 0))

        # Calculate precision and recall
        precision = intersection / len(pred_chars)
        recall = intersection / len(gt_chars)

    # Calculate F1
    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def _tokenize(text: str) -> List[str]:
    """
    Simple tokenization function.

    Args:
        text: Input text

    Returns:
        List of tokens
    """
    # Convert to lowercase and split by whitespace and punctuation
    text = text.lower()
    # Split by whitespace and remove empty tokens
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def precision_score(predictions: Union[str, List[str]],
                   ground_truths: Union[str, List[str]],
                   tokenize: bool = True) -> Union[float, List[float]]:
    """
    Calculate precision at token level.

    Args:
        predictions: Generated answers
        ground_truths: Ground truth answers
        tokenize: Whether to tokenize text

    Returns:
        Precision score(s) (0-1)
    """
    # Handle single string inputs
    if isinstance(predictions, str) and isinstance(ground_truths, str):
        return _calculate_precision_single(predictions, ground_truths, tokenize)

    # Handle list inputs
    elif isinstance(predictions, list) and isinstance(ground_truths, list):
        if len(predictions) != len(ground_truths):
            raise ValueError("predictions and ground_truths must have the same length")

        scores = []
        for pred, gt in zip(predictions, ground_truths):
            score = _calculate_precision_single(pred, gt, tokenize)
            scores.append(score)

        return scores

    else:
        raise ValueError("predictions and ground_truths must both be strings or both be lists")


def _calculate_precision_single(prediction: str, ground_truth: str, tokenize: bool = True) -> float:
    """Calculate precision for a single pair."""
    pred_tokens = _tokenize(prediction) if tokenize else list(prediction)
    gt_tokens = _tokenize(ground_truth) if tokenize else list(ground_truth)

    if not pred_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)

    intersection = 0
    for token in pred_counter:
        intersection += min(pred_counter[token], gt_counter.get(token, 0))

    return intersection / len(pred_tokens)


def recall_score(predictions: Union[str, List[str]],
                ground_truths: Union[str, List[str]],
                tokenize: bool = True) -> Union[float, List[float]]:
    """
    Calculate recall at token level.

    Args:
        predictions: Generated answers
        ground_truths: Ground truth answers
        tokenize: Whether to tokenize text

    Returns:
        Recall score(s) (0-1)
    """
    # Handle single string inputs
    if isinstance(predictions, str) and isinstance(ground_truths, str):
        return _calculate_recall_single(predictions, ground_truths, tokenize)

    # Handle list inputs
    elif isinstance(predictions, list) and isinstance(ground_truths, list):
        if len(predictions) != len(ground_truths):
            raise ValueError("predictions and ground_truths must have the same length")

        scores = []
        for pred, gt in zip(predictions, ground_truths):
            score = _calculate_recall_single(pred, gt, tokenize)
            scores.append(score)

        return scores

    else:
        raise ValueError("predictions and ground_truths must both be strings or both be lists")


def _calculate_recall_single(prediction: str, ground_truth: str, tokenize: bool = True) -> float:
    """Calculate recall for a single pair."""
    pred_tokens = _tokenize(prediction) if tokenize else list(prediction)
    gt_tokens = _tokenize(ground_truth) if tokenize else list(ground_truth)

    if not gt_tokens:
        return 1.0 if not pred_tokens else 0.0

    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)

    intersection = 0
    for token in gt_counter:
        intersection += min(gt_counter[token], pred_counter.get(token, 0))

    return intersection / len(gt_tokens)


# Example usage
if __name__ == "__main__":
    predictions = [
        "The capital of France is Paris.",  # Good match
        "Paris is the capital city.",      # Partial match
        "London is a big city.",           # No match
        "The quick brown fox jumps.",      # Different text
    ]

    ground_truths = [
        "Paris is the capital of France.",
        "Paris is the capital of France.",
        "Paris is the capital of France.",
        "The quick brown fox jumps over the lazy dog."
    ]

    # Calculate F1 scores
    f1_scores = f1_score(predictions, ground_truths)
    print("F1 scores:", f1_scores)
    print(".4f")

    # Calculate precision and recall
    precision_scores = precision_score(predictions, ground_truths)
    recall_scores = recall_score(predictions, ground_truths)

    print("Precision scores:", precision_scores)
    print("Recall scores:", recall_scores)

    # Individual detailed analysis
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        f1 = f1_score(pred, gt)
        precision = precision_score(pred, gt)
        recall = recall_score(pred, gt)
        print(f"Sample {i+1}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
        print(f"  Pred: '{pred}'")
        print(f"  GT:   '{gt}'")
