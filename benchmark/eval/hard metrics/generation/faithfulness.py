"""
Faithfulness Metric for RAG Generation Evaluation

Faithfulness measures whether the generated answer is factually consistent
with the provided context. It evaluates if the answer contains information
that can be inferred from the context and doesn't contradict it.

This metric typically requires an LLM to judge whether claims in the answer
are supported by the context. It's part of the RAGAS framework.

Range: 0 to 1, where 1 indicates the answer is completely faithful to the context.
"""

from typing import List, Union, Optional, Dict
import re
import json


class FaithfulnessScorer:
    """Faithfulness scorer using LLM-as-a-judge."""

    def __init__(self, llm_client=None, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize faithfulness scorer.

        Args:
            llm_client: LLM client for evaluation (e.g., OpenAI client)
            model_name: Name of the LLM model to use for judging
        """
        self.llm_client = llm_client
        self.model_name = model_name

    def score(self, predictions: Union[str, List[str]],
             contexts: Union[str, List[str]]) -> Union[float, List[float]]:
        """
        Calculate faithfulness scores.

        Args:
            predictions: Generated answers
            contexts: Context documents used for generation

        Returns:
            Faithfulness score(s) (0-1)
        """
        # Handle single inputs
        if isinstance(predictions, str) and isinstance(contexts, str):
            return self._calculate_faithfulness_single(predictions, contexts)

        # Handle list inputs
        elif isinstance(predictions, list) and isinstance(contexts, list):
            if len(predictions) != len(contexts):
                raise ValueError("predictions and contexts must have the same length")

            scores = []
            for pred, ctx in zip(predictions, contexts):
                score = self._calculate_faithfulness_single(pred, ctx)
                scores.append(score)

            return scores

        else:
            raise ValueError("predictions and contexts must both be strings or both be lists")

    def _calculate_faithfulness_single(self, prediction: str, context: str) -> float:
        """
        Calculate faithfulness for a single prediction-context pair.

        Args:
            prediction: Generated answer
            context: Context document

        Returns:
            Faithfulness score (0-1)
        """
        if not self.llm_client:
            # Fallback: simple keyword overlap method
            return self._simple_faithfulness(prediction, context)

        try:
            # Extract claims from prediction
            claims = self._extract_claims(prediction)

            if not claims:
                return 1.0  # Empty prediction is considered faithful

            # Check each claim against context
            supported_claims = 0

            for claim in claims:
                if self._is_claim_supported(claim, context):
                    supported_claims += 1

            return supported_claims / len(claims)

        except Exception as e:
            print(f"Error in faithfulness calculation: {e}")
            # Fallback to simple method
            return self._simple_faithfulness(prediction, context)

    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract atomic claims from text.
        This is a simple implementation - in practice, you might use
        more sophisticated NLP techniques.

        Args:
            text: Input text

        Returns:
            List of atomic claims
        """
        # Simple sentence splitting as claims
        sentences = re.split(r'[.!?]+', text)
        claims = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        return claims

    def _is_claim_supported(self, claim: str, context: str) -> bool:
        """
        Check if a claim is supported by the context using LLM.

        Args:
            claim: Claim to verify
            context: Context document

        Returns:
            True if claim is supported
        """
        if not self.llm_client:
            return False

        prompt = f"""
        Given the following context and claim, determine if the claim is supported by the context.

        Context: {context}

        Claim: {claim}

        Answer with only "YES" or "NO".
        """

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0
            )

            answer = response.choices[0].message.content.strip().upper()
            return "YES" in answer

        except Exception as e:
            print(f"LLM call failed: {e}")
            return False

    def _simple_faithfulness(self, prediction: str, context: str) -> float:
        """
        Simple faithfulness calculation using keyword overlap.
        This is a fallback when LLM is not available.

        Args:
            prediction: Generated answer
            context: Context document

        Returns:
            Faithfulness score (0-1)
        """
        # Tokenize and get unique words
        pred_words = set(re.findall(r'\b\w+\b', prediction.lower()))
        context_words = set(re.findall(r'\b\w+\b', context.lower()))

        if not pred_words:
            return 1.0

        # Calculate overlap ratio
        overlap = len(pred_words.intersection(context_words))
        return overlap / len(pred_words)


def faithfulness_score(predictions: Union[str, List[str]],
                      contexts: Union[str, List[str]],
                      llm_client=None) -> Union[float, List[float]]:
    """
    Calculate faithfulness score.

    Args:
        predictions: Generated answers
        contexts: Context documents
        llm_client: LLM client for evaluation

    Returns:
        Faithfulness score(s)
    """
    scorer = FaithfulnessScorer(llm_client)
    return scorer.score(predictions, contexts)


# Example usage
if __name__ == "__main__":
    predictions = [
        "The capital of France is Paris, and it has a population of over 2 million people.",
        "Machine learning is part of artificial intelligence and uses algorithms to learn from data.",
        "The Earth orbits around the Sun once every 365 days."
    ]

    contexts = [
        "Paris is the capital and most populous city of France. With an official estimated population of 2,102,650 residents as of 1 January 2023 in an area of more than 105 km², Paris is the fourth-most populated city in Europe.",
        "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
        "Earth orbits the Sun at an average distance of 149.60 million km, completing one orbit every 365.256 days."
    ]

    # Calculate faithfulness (without LLM - using simple method)
    scores = faithfulness_score(predictions, contexts)
    print("Faithfulness scores:", scores)
    print(".4f")

    # Individual scores
    for i, (pred, ctx) in enumerate(zip(predictions, contexts)):
        score = faithfulness_score(pred, ctx)
        print(f"Sample {i+1}: Faithfulness = {score:.4f}")

    print("\nNote: For accurate faithfulness evaluation, provide an LLM client.")
    print("Example with OpenAI:")
    print("import openai")
    print("client = openai.OpenAI()")
    print("scores = faithfulness_score(predictions, contexts, client)")
