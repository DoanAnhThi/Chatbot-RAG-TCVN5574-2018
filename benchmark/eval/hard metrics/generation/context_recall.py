"""
Context Recall Metric for RAG Generation Evaluation

Context Recall measures how much of the ground truth information is present
in the retrieved context. It evaluates whether the context contains all the
necessary facts to answer the question correctly.

This metric is part of the RAGAS framework and typically requires an LLM
to judge whether facts from the ground truth are present in the context.

Range: 0 to 1, where 1 indicates the context contains all necessary information.
"""

from typing import List, Union, Optional, Dict
import re


class ContextRecallScorer:
    """Context Recall scorer using LLM-as-a-judge."""

    def __init__(self, llm_client=None, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize context recall scorer.

        Args:
            llm_client: LLM client for evaluation (e.g., OpenAI client)
            model_name: Name of the LLM model to use for judging
        """
        self.llm_client = llm_client
        self.model_name = model_name

    def score(self, ground_truths: Union[str, List[str]],
             contexts: Union[str, List[str]]) -> Union[float, List[float]]:
        """
        Calculate context recall scores.

        Args:
            ground_truths: Ground truth answers
            contexts: Retrieved context documents

        Returns:
            Context recall score(s) (0-1)
        """
        # Handle single inputs
        if isinstance(ground_truths, str) and isinstance(contexts, str):
            return self._calculate_context_recall_single(ground_truths, contexts)

        # Handle list inputs
        elif isinstance(ground_truths, list) and isinstance(contexts, list):
            if len(ground_truths) != len(contexts):
                raise ValueError("ground_truths and contexts must have the same length")

            scores = []
            for gt, ctx in zip(ground_truths, contexts):
                score = self._calculate_context_recall_single(gt, ctx)
                scores.append(score)

            return scores

        else:
            raise ValueError("ground_truths and contexts must both be strings or both be lists")

    def _calculate_context_recall_single(self, ground_truth: str, context: str) -> float:
        """
        Calculate context recall for a single ground truth-context pair.

        Args:
            ground_truth: Ground truth answer
            context: Retrieved context

        Returns:
            Context recall score (0-1)
        """
        if not self.llm_client:
            # Fallback: simple keyword overlap method
            return self._simple_context_recall(ground_truth, context)

        try:
            # Extract key facts from ground truth
            facts = self._extract_facts(ground_truth)

            if not facts:
                return 1.0  # No facts to check

            # Check if each fact is present in context
            supported_facts = 0

            for fact in facts:
                if self._is_fact_in_context(fact, context):
                    supported_facts += 1

            return supported_facts / len(facts)

        except Exception as e:
            print(f"Error in context recall calculation: {e}")
            # Fallback to simple method
            return self._simple_context_recall(ground_truth, context)

    def _extract_facts(self, text: str) -> List[str]:
        """
        Extract key facts from text.
        This is a simple implementation - in practice, you might use
        more sophisticated fact extraction techniques.

        Args:
            text: Input text

        Returns:
            List of key facts
        """
        # Simple sentence splitting as facts
        sentences = re.split(r'[.!?]+', text)
        facts = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        return facts

    def _is_fact_in_context(self, fact: str, context: str) -> bool:
        """
        Check if a fact from ground truth is present in the context using LLM.

        Args:
            fact: Fact to check
            context: Context document

        Returns:
            True if fact is present in context
        """
        if not self.llm_client:
            return False

        prompt = f"""
        Given the following context and fact, determine if the fact is supported
        by or present in the context.

        Context: {context}

        Fact: {fact}

        Answer with only "YES" or "NO". The fact should be considered present if
        the context contains the same information, even if phrased differently.
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

    def _simple_context_recall(self, ground_truth: str, context: str) -> float:
        """
        Simple context recall using keyword overlap.
        This is a fallback when LLM is not available.

        Args:
            ground_truth: Ground truth answer
            context: Retrieved context

        Returns:
            Context recall score (0-1)
        """
        # Extract keywords from ground truth
        gt_words = set(re.findall(r'\b\w+\b', ground_truth.lower()))
        context_words = set(re.findall(r'\b\w+\b', context.lower()))

        if not gt_words:
            return 1.0

        # Calculate what fraction of ground truth words appear in context
        overlap = len(gt_words.intersection(context_words))
        return overlap / len(gt_words)


def context_recall_score(ground_truths: Union[str, List[str]],
                        contexts: Union[str, List[str]],
                        llm_client=None) -> Union[float, List[float]]:
    """
    Calculate context recall score.

    Args:
        ground_truths: Ground truth answers
        contexts: Retrieved context documents
        llm_client: LLM client for evaluation

    Returns:
        Context recall score(s)
    """
    scorer = ContextRecallScorer(llm_client)
    return scorer.score(ground_truths, contexts)


# Example usage
if __name__ == "__main__":
    ground_truths = [
        "Paris is the capital of France with a population of over 2 million.",
        "Machine learning is a subset of artificial intelligence that uses algorithms.",
        "The Earth orbits the Sun at an average distance of 149.6 million kilometers."
    ]

    contexts = [
        "Paris is the capital of France. It has a population of about 2.1 million people.",
        "Artificial intelligence includes machine learning. Algorithms are used to process data.",
        "The color of the sky is blue. Grass is green. Water is wet."
    ]

    # Calculate context recall (without LLM - using simple method)
    scores = context_recall_score(ground_truths, contexts)
    print("Context recall scores:", scores)
    print(".4f")

    # Individual scores
    for i, (gt, ctx) in enumerate(zip(ground_truths, contexts)):
        score = context_recall_score(gt, ctx)
        print(f"Sample {i+1}: Context recall = {score:.4f}")

    print("\nNote: For accurate context recall evaluation, provide an LLM client.")
    print("Example with OpenAI:")
    print("import openai")
    print("client = openai.OpenAI()")
    print("scores = context_recall_score(ground_truths, contexts, client)")
