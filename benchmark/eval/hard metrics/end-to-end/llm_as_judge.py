"""
LLM-as-a-Judge Metric for RAG End-to-End Evaluation

LLM-as-a-Judge uses a powerful language model to evaluate the quality of
generated answers based on multiple criteria. This approach leverages the
reasoning capabilities of LLMs to provide human-like judgments.

The metric evaluates answers on criteria like:
- Correctness/Factuality
- Relevance
- Completeness
- Clarity/Coherence
- Naturalness

Range: Typically 0-5 or 0-100, depending on the scale used.
"""

from typing import List, Union, Dict, Optional
import re
import statistics


class LLMJudgeScorer:
    """LLM-as-a-Judge scorer for comprehensive answer evaluation."""

    def __init__(self, llm_client=None, model_name: str = "gpt-4",
                 criteria: Optional[List[str]] = None, scale: str = "0-5"):
        """
        Initialize LLM judge scorer.

        Args:
            llm_client: LLM client for evaluation (e.g., OpenAI client)
            model_name: Name of the LLM model to use for judging
            criteria: List of criteria to evaluate (optional)
            scale: Scoring scale ("0-5" or "0-100")
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.scale = scale

        # Default criteria if none provided
        if criteria is None:
            self.criteria = [
                "correctness",  # Factual accuracy
                "relevance",    # Answer addresses the question
                "completeness", # Answer is comprehensive
                "clarity",      # Answer is clear and coherent
                "naturalness"   # Answer sounds natural
            ]
        else:
            self.criteria = criteria

    def score(self, questions: Union[str, List[str]],
             answers: Union[str, List[str]],
             contexts: Optional[Union[str, List[str]]] = None) -> Union[float, List[float]]:
        """
        Calculate LLM-as-a-judge scores.

        Args:
            questions: Input questions
            answers: Generated answers
            contexts: Optional context documents used for generation

        Returns:
            Average judge score(s)
        """
        # Handle single inputs
        if isinstance(questions, str) and isinstance(answers, str):
            return self._judge_single(questions, answers, contexts[0] if contexts else None)

        # Handle list inputs
        elif isinstance(questions, list) and isinstance(answers, list):
            if len(questions) != len(answers):
                raise ValueError("questions and answers must have the same length")

            if contexts and len(contexts) != len(questions):
                raise ValueError("contexts must have the same length as questions")

            scores = []
            for i, (question, answer) in enumerate(zip(questions, answers)):
                ctx = contexts[i] if contexts else None
                score = self._judge_single(question, answer, ctx)
                scores.append(score)

            return scores

        else:
            raise ValueError("questions and answers must both be strings or both be lists")

    def _judge_single(self, question: str, answer: str, context: Optional[str] = None) -> float:
        """
        Judge a single question-answer pair.

        Args:
            question: Input question
            answer: Generated answer
            context: Optional context document

        Returns:
            Judge score
        """
        if not self.llm_client:
            raise ValueError("LLM client is required for LLM-as-a-Judge evaluation")

        prompt = self._build_judge_prompt(question, answer, context)

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0
            )

            response_text = response.choices[0].message.content.strip()
            score = self._parse_judge_response(response_text)

            return score

        except Exception as e:
            print(f"LLM judgment failed: {e}")
            return 0.0  # Default low score on failure

    def _build_judge_prompt(self, question: str, answer: str, context: Optional[str] = None) -> str:
        """
        Build the judgment prompt.

        Args:
            question: Input question
            answer: Generated answer
            context: Optional context

        Returns:
            Formatted prompt for LLM
        """
        scale_description = {
            "0-5": "0-5 scale where 5 is perfect and 0 is terrible",
            "0-100": "0-100 scale where 100 is perfect and 0 is terrible"
        }.get(self.scale, "0-5 scale where 5 is perfect and 0 is terrible")

        prompt = f"""
You are an expert evaluator of question-answering systems. Evaluate the following answer based on multiple criteria.

Question: {question}

Generated Answer: {answer}
"""

        if context:
            prompt += f"\nContext Provided: {context}\n"

        prompt += f"""
Evaluate the answer on these criteria using a {scale_description}:

1. **Correctness**: How factually accurate is the answer?
2. **Relevance**: How well does the answer address the question?
3. **Completeness**: How complete and comprehensive is the answer?
4. **Clarity**: How clear and coherent is the answer?
5. **Naturalness**: How natural and human-like is the answer?

Provide scores for each criterion and an overall score.

Format your response as:
Correctness: [score]
Relevance: [score]
Completeness: [score]
Clarity: [score]
Naturalness: [score]
Overall: [score]

Briefly explain your reasoning.
"""

        return prompt

    def _parse_judge_response(self, response: str) -> float:
        """
        Parse the LLM response to extract the overall score.

        Args:
            response: LLM response text

        Returns:
            Overall score (normalized to 0-1)
        """
        # Look for "Overall:" pattern
        overall_match = re.search(r'Overall:\s*([0-9]*\.?[0-9]+)', response, re.IGNORECASE)

        if overall_match:
            score = float(overall_match.group(1))

            # Normalize to 0-1 based on scale
            if self.scale == "0-100":
                return score / 100.0
            else:  # 0-5 scale
                return score / 5.0

        # Fallback: try to find any number in the response
        numbers = re.findall(r'(\d*\.?\d+)', response)
        if numbers:
            # Take the last number as overall score
            score = float(numbers[-1])

            # Heuristic: if score > 10, assume it's 0-100 scale
            if score > 10:
                return score / 100.0
            else:
                return score / 5.0

        # Default fallback
        return 0.5

    def detailed_score(self, question: str, answer: str, context: Optional[str] = None) -> Dict:
        """
        Get detailed scores for all criteria.

        Args:
            question: Input question
            answer: Generated answer
            context: Optional context

        Returns:
            Dictionary with scores for each criterion
        """
        if not self.llm_client:
            raise ValueError("LLM client is required for detailed evaluation")

        prompt = self._build_judge_prompt(question, answer, context)

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0
            )

            response_text = response.choices[0].message.content.strip()
            return self._parse_detailed_response(response_text)

        except Exception as e:
            print(f"Detailed LLM judgment failed: {e}")
            return {criterion: 0.0 for criterion in self.criteria + ["overall"]}

    def _parse_detailed_response(self, response: str) -> Dict:
        """
        Parse detailed scores from LLM response.

        Args:
            response: LLM response text

        Returns:
            Dictionary of normalized scores
        """
        scores = {}

        # Extract scores for each criterion
        for criterion in self.criteria + ["overall"]:
            match = re.search(f'{criterion}:\s*([0-9]*\.?[0-9]+)', response, re.IGNORECASE)
            if match:
                score = float(match.group(1))

                # Normalize based on scale
                if self.scale == "0-100":
                    scores[criterion] = score / 100.0
                else:  # 0-5 scale
                    scores[criterion] = score / 5.0
            else:
                scores[criterion] = 0.5  # Default

        return scores


def llm_judge_score(questions: Union[str, List[str]],
                   answers: Union[str, List[str]],
                   llm_client=None,
                   contexts: Optional[Union[str, List[str]]] = None,
                   scale: str = "0-5") -> Union[float, List[float]]:
    """
    Calculate LLM-as-a-Judge score.

    Args:
        questions: Input questions
        answers: Generated answers
        llm_client: LLM client for evaluation
        contexts: Optional context documents
        scale: Scoring scale ("0-5" or "0-100")

    Returns:
        Judge score(s) (normalized to 0-1)
    """
    scorer = LLMJudgeScorer(llm_client, scale=scale)
    return scorer.score(questions, answers, contexts)


# Example usage
if __name__ == "__main__":
    questions = [
        "What is the capital of France?",
        "How does machine learning work?"
    ]

    answers = [
        "Paris is the capital of France. It's located in Western Europe and has a population of about 2.2 million people in the city proper.",
        "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention."
    ]

    contexts = [
        "Paris is the capital and most populous city of France. With an official estimated population of 2,102,650 residents, Paris is the fourth-most populated city in Europe.",
        "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data."
    ]

    # Example without LLM client (will raise error)
    try:
        scores = llm_judge_score(questions, answers, contexts=contexts)
        print("LLM Judge scores:", scores)
    except ValueError as e:
        print(f"Cannot run without LLM client: {e}")

    print("\nExample usage with OpenAI:")
    print("import openai")
    print("client = openai.OpenAI()")
    print("scores = llm_judge_score(questions, answers, client, contexts)")
    print("print('Average score:', sum(scores) / len(scores))")

    print("\nExample detailed evaluation:")
    print("scorer = LLMJudgeScorer(client)")
    print("detailed = scorer.detailed_score(questions[0], answers[0], contexts[0])")
    print("print('Detailed scores:', detailed)")
