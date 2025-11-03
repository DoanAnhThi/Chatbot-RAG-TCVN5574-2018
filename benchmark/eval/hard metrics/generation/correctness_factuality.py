"""
Correctness/Factuality Metric for RAG Generation Evaluation

Correctness (also called Factuality) measures whether the generated answer
is factually accurate and contains correct information. It evaluates the
factual correctness of claims made in the answer.

This metric typically requires an LLM or external knowledge to verify
the factual accuracy of statements in the generated answer.

Range: 0 to 1, where 1 indicates the answer is completely factually correct.
"""

from typing import List, Union, Optional, Dict
import re


class CorrectnessScorer:
    """Correctness/Factuality scorer using LLM-as-a-judge or external knowledge."""

    def __init__(self, llm_client=None, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize correctness scorer.

        Args:
            llm_client: LLM client for evaluation (e.g., OpenAI client)
            model_name: Name of the LLM model to use for judging
        """
        self.llm_client = llm_client
        self.model_name = model_name

    def score(self, predictions: Union[str, List[str]],
             ground_truths: Union[str, List[str]]) -> Union[float, List[float]]:
        """
        Calculate correctness scores by comparing predictions with ground truths.

        Args:
            predictions: Generated answers
            ground_truths: Ground truth correct answers

        Returns:
            Correctness score(s) (0-1)
        """
        # Handle single inputs
        if isinstance(predictions, str) and isinstance(ground_truths, str):
            return self._calculate_correctness_single(predictions, ground_truths)

        # Handle list inputs
        elif isinstance(predictions, list) and isinstance(ground_truths, list):
            if len(predictions) != len(ground_truths):
                raise ValueError("predictions and ground_truths must have the same length")

            scores = []
            for pred, gt in zip(predictions, ground_truths):
                score = self._calculate_correctness_single(pred, gt)
                scores.append(score)

            return scores

        else:
            raise ValueError("predictions and ground_truths must both be strings or both be lists")

    def _calculate_correctness_single(self, prediction: str, ground_truth: str) -> float:
        """
        Calculate correctness for a single prediction-ground truth pair.

        Args:
            prediction: Generated answer
            ground_truth: Ground truth answer

        Returns:
            Correctness score (0-1)
        """
        if not self.llm_client:
            # Fallback: simple semantic similarity
            return self._simple_correctness(prediction, ground_truth)

        try:
            # Use LLM to judge correctness
            return self._llm_correctness_score(prediction, ground_truth)

        except Exception as e:
            print(f"Error in correctness calculation: {e}")
            # Fallback to simple method
            return self._simple_correctness(prediction, ground_truth)

    def _llm_correctness_score(self, prediction: str, ground_truth: str) -> float:
        """
        Calculate correctness using LLM-as-a-judge.

        Args:
            prediction: Generated answer
            ground_truth: Ground truth answer

        Returns:
            Correctness score (0-1)
        """
        prompt = f"""
        Given the following generated answer and ground truth answer, evaluate
        how factually correct the generated answer is.

        Generated Answer: {prediction}

        Ground Truth Answer: {ground_truth}

        Rate the factual correctness on a scale from 0 to 1, where:
        - 1.0 = Completely factually correct (all information matches ground truth)
        - 0.5 = Partially correct (some information is correct, some is wrong)
        - 0.0 = Completely factually incorrect (major factual errors)

        Consider:
        - Are the facts stated in the generated answer accurate?
        - Does it contradict known facts?
        - Is the information consistent with the ground truth?

        Provide only a number between 0 and 1.
        """

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0
            )

            score_text = response.choices[0].message.content.strip()
            # Extract number from response
            score_match = re.search(r'(\d*\.?\d+)', score_text)
            if score_match:
                score = float(score_match.group(1))
                return max(0.0, min(1.0, score))  # Clamp to [0, 1]
            else:
                return 0.5  # Default if parsing fails

        except Exception as e:
            print(f"LLM call failed: {e}")
            return self._simple_correctness(prediction, ground_truth)

    def _simple_correctness(self, prediction: str, ground_truth: str) -> float:
        """
        Simple correctness calculation using text overlap and similarity.

        Args:
            prediction: Generated answer
            ground_truth: Ground truth answer

        Returns:
            Correctness score (0-1)
        """
        # Normalize texts
        pred_norm = prediction.lower().strip()
        gt_norm = ground_truth.lower().strip()

        # Exact match
        if pred_norm == gt_norm:
            return 1.0

        # Token-level overlap
        pred_tokens = set(re.findall(r'\b\w+\b', pred_norm))
        gt_tokens = set(re.findall(r'\b\w+\b', gt_norm))

        if not gt_tokens:
            return 1.0 if not pred_tokens else 0.0

        # Jaccard similarity
        intersection = len(pred_tokens.intersection(gt_tokens))
        union = len(pred_tokens.union(gt_tokens))

        jaccard = intersection / union if union > 0 else 0.0

        # Length ratio penalty (penalize very different lengths)
        len_ratio = min(len(pred_tokens), len(gt_tokens)) / max(len(pred_tokens), len(gt_tokens))
        len_penalty = 0.8 + 0.2 * len_ratio  # Penalty between 0.8 and 1.0

        return jaccard * len_penalty


def correctness_score(predictions: Union[str, List[str]],
                     ground_truths: Union[str, List[str]],
                     llm_client=None) -> Union[float, List[float]]:
    """
    Calculate correctness/factuality score.

    Args:
        predictions: Generated answers
        ground_truths: Ground truth correct answers
        llm_client: LLM client for evaluation (optional)

    Returns:
        Correctness score(s)
    """
    scorer = CorrectnessScorer(llm_client)
    return scorer.score(predictions, ground_truths)


# Example usage
if __name__ == "__main__":
    predictions = [
        "Paris is the capital of France.",  # Correct
        "Machine learning is a type of artificial intelligence that learns from data.",  # Mostly correct
        "The Earth is flat and the moon is made of cheese.",  # Completely wrong
    ]

    ground_truths = [
        "Paris is the capital of France.",
        "Machine learning is a subset of artificial intelligence.",
        "The Earth is an oblate spheroid that orbits the Sun."
    ]

    # Calculate correctness (without LLM - using simple method)
    scores = correctness_score(predictions, ground_truths)
    print("Correctness scores:", scores)
    print(".4f")

    # Individual scores
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        score = correctness_score(pred, gt)
        print(f"Sample {i+1}: Correctness = {score:.4f}")

    print("\nNote: For accurate correctness evaluation, provide an LLM client.")
    print("Example with OpenAI:")
    print("import openai")
    print("client = openai.OpenAI()")
    print("scores = correctness_score(predictions, ground_truths, client)")
