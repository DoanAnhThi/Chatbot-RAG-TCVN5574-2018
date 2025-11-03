"""
Answer Relevance Metric for RAG Generation Evaluation

Answer Relevance measures how directly and completely the generated answer
addresses the input question. It evaluates whether the answer is focused on
the question and provides relevant information.

This metric can be calculated using semantic similarity (cosine similarity
of embeddings) or LLM-as-a-judge to evaluate the relevance.

Range: 0 to 1, where 1 indicates the answer is perfectly relevant to the question.
"""

from typing import List, Union, Optional
import re

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    SentenceTransformer = None


class AnswerRelevanceScorer:
    """Answer Relevance scorer using embeddings or LLM."""

    def __init__(self, llm_client=None, model_name: str = "gpt-3.5-turbo",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize answer relevance scorer.

        Args:
            llm_client: LLM client for evaluation (optional)
            model_name: Name of the LLM model to use for judging
            embedding_model: Name of the embedding model for semantic similarity
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.embedding_model_name = embedding_model

        if SentenceTransformer is not None:
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            self.embedding_model = None

    def score(self, questions: Union[str, List[str]],
             answers: Union[str, List[str]]) -> Union[float, List[float]]:
        """
        Calculate answer relevance scores.

        Args:
            questions: Input questions
            answers: Generated answers

        Returns:
            Answer relevance score(s) (0-1)
        """
        # Handle single inputs
        if isinstance(questions, str) and isinstance(answers, str):
            return self._calculate_answer_relevance_single(questions, answers)

        # Handle list inputs
        elif isinstance(questions, list) and isinstance(answers, list):
            if len(questions) != len(answers):
                raise ValueError("questions and answers must have the same length")

            scores = []
            for question, answer in zip(questions, answers):
                score = self._calculate_answer_relevance_single(question, answer)
                scores.append(score)

            return scores

        else:
            raise ValueError("questions and answers must both be strings or both be lists")

    def _calculate_answer_relevance_single(self, question: str, answer: str) -> float:
        """
        Calculate answer relevance for a single question-answer pair.

        Args:
            question: Input question
            answer: Generated answer

        Returns:
            Answer relevance score (0-1)
        """
        if self.llm_client:
            # Use LLM-as-a-judge
            return self._llm_relevance_score(question, answer)
        elif self.embedding_model:
            # Use semantic similarity
            return self._semantic_relevance_score(question, answer)
        else:
            # Fallback to simple keyword overlap
            return self._simple_relevance_score(question, answer)

    def _llm_relevance_score(self, question: str, answer: str) -> float:
        """
        Calculate relevance using LLM-as-a-judge.

        Args:
            question: Input question
            answer: Generated answer

        Returns:
            Relevance score (0-1)
        """
        prompt = f"""
        Given the following question and answer, rate how relevant the answer is
        to the question on a scale from 0 to 1.

        Question: {question}

        Answer: {answer}

        Consider:
        - Does the answer directly address the question?
        - Does the answer stay on topic?
        - Is the answer focused and not rambling?

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
            # Fallback to semantic similarity
            if self.embedding_model:
                return self._semantic_relevance_score(question, answer)
            else:
                return self._simple_relevance_score(question, answer)

    def _semantic_relevance_score(self, question: str, answer: str) -> float:
        """
        Calculate relevance using semantic similarity.

        Args:
            question: Input question
            answer: Generated answer

        Returns:
            Relevance score (0-1)
        """
        if not self.embedding_model:
            return self._simple_relevance_score(question, answer)

        try:
            # Get embeddings
            question_emb = self.embedding_model.encode([question])[0]
            answer_emb = self.embedding_model.encode([answer])[0]

            # Calculate cosine similarity and normalize to [0, 1]
            similarity = cosine_similarity([question_emb], [answer_emb])[0][0]

            # Cosine similarity is in [-1, 1], convert to [0, 1]
            return (similarity + 1) / 2

        except Exception as e:
            print(f"Embedding calculation failed: {e}")
            return self._simple_relevance_score(question, answer)

    def _simple_relevance_score(self, question: str, answer: str) -> float:
        """
        Simple relevance calculation using keyword overlap.

        Args:
            question: Input question
            answer: Generated answer

        Returns:
            Relevance score (0-1)
        """
        # Extract keywords from question (excluding stop words)
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

        question_words = set(re.findall(r'\b\w+\b', question.lower())) - stop_words
        answer_words = set(re.findall(r'\b\w+\b', answer.lower())) - stop_words

        if not question_words:
            return 1.0 if answer_words else 0.0

        # Calculate overlap ratio
        overlap = len(question_words.intersection(answer_words))
        return overlap / len(question_words)


def answer_relevance_score(questions: Union[str, List[str]],
                          answers: Union[str, List[str]],
                          llm_client=None) -> Union[float, List[float]]:
    """
    Calculate answer relevance score.

    Args:
        questions: Input questions
        answers: Generated answers
        llm_client: LLM client for evaluation (optional)

    Returns:
        Answer relevance score(s)
    """
    scorer = AnswerRelevanceScorer(llm_client)
    return scorer.score(questions, answers)


# Example usage
if __name__ == "__main__":
    questions = [
        "What is the capital of France?",
        "How does machine learning work?",
        "What is the distance from Earth to the Sun?"
    ]

    answers = [
        "Paris is the capital of France.",  # Perfectly relevant
        "Machine learning uses algorithms to learn from data and make predictions.",  # Relevant
        "The sky is blue because of light scattering.",  # Irrelevant to the question
    ]

    # Calculate answer relevance (without LLM - using semantic similarity if available)
    try:
        scores = answer_relevance_score(questions, answers)
        print("Answer relevance scores:", scores)
        print(".4f")

        # Individual scores
        for i, (question, answer) in enumerate(zip(questions, answers)):
            score = answer_relevance_score(question, answer)
            print(f"Question {i+1}: Answer relevance = {score:.4f}")

    except ImportError:
        print("Sentence transformers not available, using simple keyword overlap:")

        scorer = AnswerRelevanceScorer()
        scores = scorer.score(questions, answers)
        print("Simple answer relevance scores:", scores)

    print("\nNote: For more accurate evaluation, provide an LLM client or install sentence-transformers.")
    print("Example with OpenAI:")
    print("import openai")
    print("client = openai.OpenAI()")
    print("scores = answer_relevance_score(questions, answers, client)")
