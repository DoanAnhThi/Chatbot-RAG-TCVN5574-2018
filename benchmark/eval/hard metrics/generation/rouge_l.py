"""
ROUGE-L Metric for RAG Generation Evaluation

ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation - Longest Common Subsequence)
measures the longest common subsequence between the generated answer and ground truth.
It focuses on sentence-level structure similarity and is good for evaluating
factual overlap between texts.

Formula: ROUGE-L = LCS(X,Y) / length(Y)
where LCS is the length of the longest common subsequence,
X is the generated answer, Y is the ground truth.

Range: 0 to 1, where 1 indicates identical texts.
"""

try:
    from rouge import Rouge
except ImportError:
    print("Warning: rouge library not found. Install with: pip install rouge")
    Rouge = None

from typing import List, Union
import re


def rouge_l_score(predictions: Union[str, List[str]],
                 references: Union[str, List[str]]) -> Union[float, List[float]]:
    """
    Calculate ROUGE-L score between predictions and references.

    Args:
        predictions: Generated answers (string or list of strings)
        references: Ground truth answers (string or list of strings)

    Returns:
        ROUGE-L F1 score(s) (float or list of floats)
    """
    if Rouge is None:
        raise ImportError("rouge library is required. Install with: pip install rouge")

    rouge = Rouge()

    # Handle single string inputs
    if isinstance(predictions, str) and isinstance(references, str):
        scores = rouge.get_scores(predictions, references, avg=False)
        return scores[0]['rouge-l']['f']

    # Handle list inputs
    elif isinstance(predictions, list) and isinstance(references, list):
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length")

        scores = []
        for pred, ref in zip(predictions, references):
            score = rouge.get_scores(pred, ref, avg=False)
            scores.append(score[0]['rouge-l']['f'])

        return scores

    else:
        raise ValueError("predictions and references must both be strings or both be lists")


def rouge_l_precision(predictions: Union[str, List[str]],
                     references: Union[str, List[str]]) -> Union[float, List[float]]:
    """
    Calculate ROUGE-L precision.

    Args:
        predictions: Generated answers (string or list of strings)
        references: Ground truth answers (string or list of strings)

    Returns:
        ROUGE-L precision score(s) (float or list of floats)
    """
    if Rouge is None:
        raise ImportError("rouge library is required. Install with: pip install rouge")

    rouge = Rouge()

    # Handle single string inputs
    if isinstance(predictions, str) and isinstance(references, str):
        scores = rouge.get_scores(predictions, references, avg=False)
        return scores[0]['rouge-l']['p']

    # Handle list inputs
    elif isinstance(predictions, list) and isinstance(references, list):
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length")

        scores = []
        for pred, ref in zip(predictions, references):
            score = rouge.get_scores(pred, ref, avg=False)
            scores.append(score[0]['rouge-l']['p'])

        return scores

    else:
        raise ValueError("predictions and references must both be strings or both be lists")


def rouge_l_recall(predictions: Union[str, List[str]],
                  references: Union[str, List[str]]) -> Union[float, List[float]]:
    """
    Calculate ROUGE-L recall.

    Args:
        predictions: Generated answers (string or list of strings)
        references: Ground truth answers (string or list of strings)

    Returns:
        ROUGE-L recall score(s) (float or list of floats)
    """
    if Rouge is None:
        raise ImportError("rouge library is required. Install with: pip install rouge")

    rouge = Rouge()

    # Handle single string inputs
    if isinstance(predictions, str) and isinstance(references, str):
        scores = rouge.get_scores(predictions, references, avg=False)
        return scores[0]['rouge-l']['r']

    # Handle list inputs
    elif isinstance(predictions, list) and isinstance(references, list):
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length")

        scores = []
        for pred, ref in zip(predictions, references):
            score = rouge.get_scores(pred, ref, avg=False)
            scores.append(score[0]['rouge-l']['r'])

        return scores

    else:
        raise ValueError("predictions and references must both be strings or both be lists")


def _simple_rouge_l(prediction: str, reference: str) -> float:
    """
    Simple implementation of ROUGE-L using dynamic programming.
    Use this if the rouge library is not available.

    Args:
        prediction: Generated answer
        reference: Ground truth answer

    Returns:
        ROUGE-L F1 score
    """
    def lcs_length(x: str, y: str) -> int:
        """Calculate length of longest common subsequence."""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    # Tokenize (simple split by whitespace)
    pred_tokens = prediction.split()
    ref_tokens = reference.split()

    if not ref_tokens:
        return 1.0 if not pred_tokens else 0.0

    lcs_len = lcs_length(pred_tokens, ref_tokens)

    # ROUGE-L precision = LCS / len(prediction)
    precision = lcs_len / len(pred_tokens) if pred_tokens else 0.0

    # ROUGE-L recall = LCS / len(reference)
    recall = lcs_len / len(ref_tokens)

    # F1 score
    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


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
        # Using rouge library
        rouge_l_scores = rouge_l_score(predictions, references)
        print("ROUGE-L F1 scores:", rouge_l_scores)
        print(".4f")

        # Individual scores
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            score = rouge_l_score(pred, ref)
            precision = rouge_l_precision(pred, ref)
            recall = rouge_l_recall(pred, ref)
            print(f"Sample {i+1}: F1={score:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

    except ImportError:
        # Fallback to simple implementation
        print("Using simple ROUGE-L implementation:")
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            score = _simple_rouge_l(pred, ref)
            print(f"Sample {i+1}: ROUGE-L F1 = {score:.4f}")
