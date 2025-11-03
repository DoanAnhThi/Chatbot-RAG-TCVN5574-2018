"""
Faithfulness Evaluation

Evaluates whether the generated answer is faithful to the provided context.
Faithfulness measures if the answer contains only information that can be
inferred or directly found in the retrieved context.
"""

from typing import List, Dict, Any
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from . import BaseEvaluator, EvaluationResult, RAGEvaluationInput


class FaithfulnessEvaluator(BaseEvaluator):
    """Evaluates faithfulness of answers to the context"""

    def __init__(self):
        super().__init__(
            name="faithfulness",
            description="Measures if the answer is consistent with and supported by the retrieved context"
        )
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.0,
            api_key=settings.openai_api_key
        )

    def evaluate(self, input_data: RAGEvaluationInput) -> EvaluationResult:
        """
        Evaluate faithfulness using LLM-based assessment

        Returns:
            EvaluationResult with score 0-1 where 1 means fully faithful
        """
        if not input_data.context.strip():
            return EvaluationResult(
                score=0.0,
                reasoning="No context provided to evaluate faithfulness against",
                metadata={"context_length": 0}
            )

        if not input_data.answer.strip():
            return EvaluationResult(
                score=0.0,
                reasoning="No answer provided to evaluate",
                metadata={"answer_length": 0}
            )

        # Use LLM to evaluate faithfulness
        evaluation = self._evaluate_with_llm(input_data)

        return EvaluationResult(
            score=evaluation["score"],
            reasoning=evaluation["reasoning"],
            metadata={
                "method": "llm_based",
                "context_length": len(input_data.context),
                "answer_length": len(input_data.answer),
                "faithfulness_score": evaluation["score"]
            }
        )

    def _evaluate_with_llm(self, input_data: RAGEvaluationInput) -> Dict[str, Any]:
        """Use LLM to evaluate faithfulness"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là chuyên gia đánh giá chất lượng câu trả lời.
Hãy đánh giá độ trung thực (faithfulness) của câu trả lời dựa trên ngữ cảnh được cung cấp.

Độ trung thực đo lường mức độ câu trả lời chỉ chứa thông tin có thể suy ra hoặc tìm thấy trực tiếp từ ngữ cảnh.

Tiêu chí đánh giá:
- 1.0: Câu trả lời hoàn toàn trung thực, mọi thông tin đều được hỗ trợ bởi ngữ cảnh
- 0.8: Hầu hết thông tin đều trung thực, chỉ có một vài điểm nhỏ không được hỗ trợ rõ ràng
- 0.6: Một số thông tin không được hỗ trợ bởi ngữ cảnh, nhưng không mâu thuẫn
- 0.4: Có thông tin không được hỗ trợ và có thể mâu thuẫn với ngữ cảnh
- 0.2: Nhiều thông tin không được hỗ trợ hoặc mâu thuẫn với ngữ cảnh
- 0.0: Câu trả lời không liên quan hoặc hoàn toàn không được hỗ trợ bởi ngữ cảnh

Hãy trả lời theo định dạng JSON:
{
    "score": <điểm số từ 0.0 đến 1.0>,
    "reasoning": "<giải thích chi tiết về đánh giá>"
}"""),
            ("human", """Ngữ cảnh:
{context}

Câu hỏi: {question}

Câu trả lời: {answer}

Hãy đánh giá độ trung thực của câu trả lời.""")
        ])

        try:
            messages = prompt.format_messages(
                context=input_data.context,
                question=input_data.question,
                answer=input_data.answer
            )
            response = self.llm.invoke(messages)

            # Parse JSON response
            import json
            result = json.loads(response.content.strip())

            return {
                "score": float(result.get("score", 0.5)),
                "reasoning": result.get("reasoning", "Unable to parse evaluation")
            }

        except Exception as e:
            return {
                "score": 0.5,  # Default to neutral score on error
                "reasoning": f"Error during evaluation: {str(e)}"
            }


class SemanticFaithfulnessEvaluator(BaseEvaluator):
    """Evaluates faithfulness using semantic similarity and fact extraction"""

    def __init__(self):
        super().__init__(
            name="semantic_faithfulness",
            description="Measures semantic consistency between answer and context using fact extraction"
        )

    def evaluate(self, input_data: RAGEvaluationInput) -> EvaluationResult:
        """
        Evaluate faithfulness by extracting facts from answer and checking
        if they can be found in the context
        """
        if not input_data.context.strip():
            return EvaluationResult(
                score=0.0,
                reasoning="No context provided",
                metadata={"context_length": 0}
            )

        if not input_data.answer.strip():
            return EvaluationResult(
                score=0.0,
                reasoning="No answer provided",
                metadata={"answer_length": 0}
            )

        # Extract facts from answer
        answer_facts = self._extract_facts(input_data.answer)

        if not answer_facts:
            return EvaluationResult(
                score=0.0,
                reasoning="No factual claims found in answer",
                metadata={"facts_extracted": 0}
            )

        # Check each fact against context
        supported_facts = 0
        fact_checks = []

        for fact in answer_facts:
            is_supported = self._check_fact_in_context(fact, input_data.context)
            fact_checks.append({
                "fact": fact,
                "supported": is_supported
            })
            if is_supported:
                supported_facts += 1

        score = supported_facts / len(answer_facts) if answer_facts else 0

        reasoning = f"Found {supported_facts}/{len(answer_facts)} facts supported by context. "
        reasoning += f"Facts checked: {fact_checks}"

        return EvaluationResult(
            score=score,
            reasoning=reasoning,
            metadata={
                "method": "fact_extraction",
                "total_facts": len(answer_facts),
                "supported_facts": supported_facts,
                "fact_checks": fact_checks
            }
        )

    def _extract_facts(self, text: str) -> List[str]:
        """Extract factual statements from text"""
        # Simple fact extraction - split by sentences and filter
        sentences = re.split(r'[.!?]+', text.strip())

        facts = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Ignore very short sentences
                # Check if sentence contains factual content
                if self._is_factual_statement(sentence):
                    facts.append(sentence)

        return facts

    def _is_factual_statement(self, sentence: str) -> bool:
        """Check if sentence appears to be a factual statement"""
        # Simple heuristics for factual statements
        sentence_lower = sentence.lower()

        # Contains numbers, dates, names, or specific terms
        factual_indicators = [
            r'\d+',  # numbers
            r'(là|sẽ|đã|có|không)',  # Vietnamese verbs indicating statements
            r'(theo|từ|trong|vào)',  # prepositions indicating specific info
        ]

        for pattern in factual_indicators:
            if re.search(pattern, sentence_lower):
                return True

        return False

    def _check_fact_in_context(self, fact: str, context: str) -> bool:
        """Check if a fact is supported by the context"""
        # Simple string matching approach
        fact_processed = self._preprocess_text(fact)
        context_processed = self._preprocess_text(context)

        # Check for exact substring match
        if fact_processed in context_processed:
            return True

        # Check for partial matches (key terms)
        fact_words = set(fact_processed.split())
        context_words = set(context_processed.split())

        # If 70% of fact words appear in context, consider it supported
        overlap = len(fact_words.intersection(context_words))
        if overlap / len(fact_words) >= 0.7:
            return True

        return False
