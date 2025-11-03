"""
Exact Match (EM) Metric for RAG End-to-End Evaluation

Exact Match measures the percentage of predictions that exactly match the ground truth.
It's a strict, binary metric commonly used for question answering tasks where
the answer should be exact (like entity names, dates, etc.).

Formula: EM = (Number of exact matches) / (Total number of predictions)

Range: 0 to 1, where 1 indicates all predictions exactly match ground truth.
"""

from typing import List, Union
import re


def exact_match(predictions: Union[str, List[str]],
               ground_truths: Union[str, List[str]],
               normalize: bool = True) -> Union[float, List[int]]:
    """
    Calculate Exact Match score.

    Args:
        predictions: Generated answers (string or list of strings)
        ground_truths: Ground truth answers (string or list of strings)
        normalize: Whether to normalize text before comparison

    Returns:
        Exact Match score (0-1) for lists, or list of binary matches (0/1)
    """
    # Handle single string inputs
    if isinstance(predictions, str) and isinstance(ground_truths, str):
        return 1 if _normalize_text(predictions, normalize) == _normalize_text(ground_truths, normalize) else 0

    # Handle list inputs
    elif isinstance(predictions, list) and isinstance(ground_truths, list):
        if len(predictions) != len(ground_truths):
            raise ValueError("predictions and ground_truths must have the same length")

        matches = []
        for pred, gt in zip(predictions, ground_truths):
            match = 1 if _normalize_text(pred, normalize) == _normalize_text(gt, normalize) else 0
            matches.append(match)

        # Return average score for compatibility
        return sum(matches) / len(matches)

    else:
        raise ValueError("predictions and ground_truths must both be strings or both be lists")


def exact_match_binary(predictions: Union[str, List[str]],
                      ground_truths: Union[str, List[str]],
                      normalize: bool = True) -> Union[int, List[int]]:
    """
    Calculate Exact Match as binary values (returns 0 or 1 for each prediction).

    Args:
        predictions: Generated answers (string or list of strings)
        ground_truths: Ground truth answers (string or list of strings)
        normalize: Whether to normalize text before comparison

    Returns:
        Binary match results (0/1) - single int or list of ints
    """
    # Handle single string inputs
    if isinstance(predictions, str) and isinstance(ground_truths, str):
        return 1 if _normalize_text(predictions, normalize) == _normalize_text(ground_truths, normalize) else 0

    # Handle list inputs
    elif isinstance(predictions, list) and isinstance(ground_truths, list):
        if len(predictions) != len(ground_truths):
            raise ValueError("predictions and ground_truths must have the same length")

        matches = []
        for pred, gt in zip(predictions, ground_truths):
            match = 1 if _normalize_text(pred, normalize) == _normalize_text(gt, normalize) else 0
            matches.append(match)

        return matches

    else:
        raise ValueError("predictions and ground_truths must both be strings or both be lists")


def _normalize_text(text: str, normalize: bool = True) -> str:
    """
    Normalize text for comparison.

    Args:
        text: Input text
        normalize: Whether to apply normalization

    Returns:
        Normalized text
    """
    if not normalize:
        return text

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove punctuation (optional - can be adjusted based on needs)
    # text = re.sub(r'[^\w\s]', '', text)

    return text


def exact_match_with_multiple_ground_truths(predictions: List[str],
                                           ground_truths: List[List[str]],
                                           normalize: bool = True) -> float:
    """
    Calculate Exact Match when there are multiple possible correct answers.

    Args:
        predictions: List of generated answers
        ground_truths: List of lists, where each inner list contains possible correct answers
        normalize: Whether to normalize text before comparison

    Returns:
        Average Exact Match score (0-1)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must have the same length")

    matches = 0
    for pred, gt_list in zip(predictions, ground_truths):
        pred_norm = _normalize_text(pred, normalize)
        gt_norm_list = [_normalize_text(gt, normalize) for gt in gt_list]

        # Check if prediction matches any ground truth
        if pred_norm in gt_norm_list:
            matches += 1

    return matches / len(predictions)


# Example usage
if __name__ == "__main__":
    predictions = [
        "Paris",  # Exact match
        "paris",  # Case difference (will match with normalization)
        "London", # No match
        "The capital of France is Paris.",  # Partial match but not exact
    ]

    ground_truths = [
        "Paris",
        "Paris",
        "Paris",
        "Paris"
    ]

    # Calculate exact match
    em_score = exact_match(predictions, ground_truths)
    print(".4f")

    # Get binary results
    binary_results = exact_match_binary(predictions, ground_truths)
    print("Binary results:", binary_results)

    # Individual comparisons
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        match = exact_match(pred, gt)
        print(f"Sample {i+1}: '{pred}' vs '{gt}' -> {'Match' if match else 'No match'}")

    # Example with multiple ground truths
    predictions_multi = ["Paris", "London"]
    ground_truths_multi = [["Paris", "paris"], ["London", "london", "UK capital"]]

    em_multi = exact_match_with_multiple_ground_truths(predictions_multi, ground_truths_multi)
    print(".4f")
