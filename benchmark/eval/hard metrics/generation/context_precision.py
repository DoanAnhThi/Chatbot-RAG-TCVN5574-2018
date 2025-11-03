"""
Context Precision Metric for RAG Generation Evaluation

Context Precision measures how relevant and useful the retrieved context is
for answering the given question. It evaluates whether the context contains
information necessary to answer the question accurately.

This metric is part of the RAGAS framework and typically requires an LLM
to judge the relevance of each context chunk to the question.

Range: 0 to 1, where 1 indicates all context is perfectly relevant to the question.
"""

from typing import List, Union, Optional, Dict
import re


class ContextPrecisionScorer:
    """Context Precision scorer using LLM-as-a-judge."""

    def __init__(self, llm_client=None, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize context precision scorer.

        Args:
            llm_client: LLM client for evaluation (e.g., OpenAI client)
            model_name: Name of the LLM model to use for judging
        """
        self.llm_client = llm_client
        self.model_name = model_name

    def score(self, questions: Union[str, List[str]],
             contexts: Union[List[str], List[List[str]]]) -> Union[float, List[float]]:
        """
        Calculate context precision scores.

        Args:
            questions: Input questions
            contexts: Retrieved context chunks (can be list of chunks per question)

        Returns:
            Context precision score(s) (0-1)
        """
        # Handle single question with multiple contexts
        if isinstance(questions, str) and isinstance(contexts, list):
            if contexts and isinstance(contexts[0], str):
                # Single question with list of context chunks
                return self._calculate_context_precision_single(questions, contexts)

        # Handle multiple questions with multiple contexts each
        elif isinstance(questions, list) and isinstance(contexts, list):
            if len(questions) != len(contexts):
                raise ValueError("questions and contexts must have the same length")

            scores = []
            for question, ctx_list in zip(questions, contexts):
                if isinstance(ctx_list, list):
                    score = self._calculate_context_precision_single(question, ctx_list)
                else:
                    # Single context string
                    score = self._calculate_context_precision_single(question, [ctx_list])
                scores.append(score)

            return scores

        else:
            raise ValueError("Invalid input format. See docstring for expected formats.")

    def _calculate_context_precision_single(self, question: str, contexts: List[str]) -> float:
        """
        Calculate context precision for a single question-context pair.

        Args:
            question: Input question
            contexts: List of context chunks

        Returns:
            Context precision score (0-1)
        """
        if not contexts:
            return 0.0

        if not self.llm_client:
            # Fallback: simple keyword overlap method
            return self._simple_context_precision(question, contexts)

        try:
            # Evaluate relevance of each context chunk
            relevant_chunks = 0
            total_chunks = len(contexts)

            for context in contexts:
                if self._is_context_relevant(question, context):
                    relevant_chunks += 1

            return relevant_chunks / total_chunks

        except Exception as e:
            print(f"Error in context precision calculation: {e}")
            # Fallback to simple method
            return self._simple_context_precision(question, contexts)

    def _is_context_relevant(self, question: str, context: str) -> bool:
        """
        Check if context is relevant to the question using LLM.

        Args:
            question: Input question
            context: Context chunk

        Returns:
            True if context is relevant
        """
        if not self.llm_client:
            return False

        prompt = f"""
        Given the following question and context, determine if the context contains
        information that is useful for answering the question.

        Question: {question}

        Context: {context}

        Answer with only "YES" or "NO". Consider the context relevant if it contains
        any information that could help answer the question, even if it's not complete.
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

    def _simple_context_precision(self, question: str, contexts: List[str]) -> float:
        """
        Simple context precision using keyword overlap.
        This is a fallback when LLM is not available.

        Args:
            question: Input question
            contexts: List of context chunks

        Returns:
            Context precision score (0-1)
        """
        if not contexts:
            return 0.0

        # Extract keywords from question
        question_words = set(re.findall(r'\b\w+\b', question.lower()))

        relevant_count = 0

        for context in contexts:
            context_words = set(re.findall(r'\b\w+\b', context.lower()))
            overlap = len(question_words.intersection(context_words))

            # Consider context relevant if it shares at least 20% of question words
            if overlap >= max(1, len(question_words) * 0.2):
                relevant_count += 1

        return relevant_count / len(contexts)


def context_precision_score(questions: Union[str, List[str]],
                          contexts: Union[List[str], List[List[str]]],
                          llm_client=None) -> Union[float, List[float]]:
    """
    Calculate context precision score.

    Args:
        questions: Input questions
        contexts: Retrieved context chunks
        llm_client: LLM client for evaluation

    Returns:
        Context precision score(s)
    """
    scorer = ContextPrecisionScorer(llm_client)
    return scorer.score(questions, contexts)


# Example usage
if __name__ == "__main__":
    questions = [
        "What is the capital of France?",
        "How does machine learning work?",
        "What is the distance from Earth to the Sun?"
    ]

    contexts = [
        # Question 1: relevant contexts
        ["Paris is the capital of France.", "France is a country in Europe.", "The weather in Paris is temperate."],
        # Question 2: mixed relevance
        ["Machine learning uses algorithms.", "Paris is in France.", "Data analysis is important."],
        # Question 3: irrelevant contexts
        ["Cats are mammals.", "The color blue is cool.", "Pizza is delicious."]
    ]

    # Calculate context precision (without LLM - using simple method)
    scores = context_precision_score(questions, contexts)
    print("Context precision scores:", scores)
    print(".4f")

    # Individual scores
    for i, (question, ctx_list) in enumerate(zip(questions, contexts)):
        score = context_precision_score(question, ctx_list)
        print(f"Question {i+1}: Context precision = {score:.4f}")

    print("\nNote: For accurate context precision evaluation, provide an LLM client.")
    print("Example with OpenAI:")
    print("import openai")
    print("client = openai.OpenAI()")
    print("scores = context_precision_score(questions, contexts, client)")
