"""
Completeness Evaluation

Evaluates whether the generated answer fully addresses all aspects of the question.
Completeness measures if the answer provides complete information without gaps.
"""

from typing import List, Dict, Any
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.config import settings
from . import BaseEvaluator, EvaluationResult, RAGEvaluationInput


class CompletenessEvaluator(BaseEvaluator):
    """Evaluates completeness of answers to questions"""

    def __init__(self):
        super().__init__(
            name="completeness",
            description="Measures whether the answer fully addresses all aspects of the question"
        )
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.0,
            api_key=settings.openai_api_key
        )

    def evaluate(self, input_data: RAGEvaluationInput) -> EvaluationResult:
        """
        Evaluate completeness using LLM-based assessment

        Returns:
            EvaluationResult with score 0-1 where 1 means fully complete
        """
        if not input_data.question.strip():
            return EvaluationResult(
                score=0.0,
                reasoning="No question provided to evaluate completeness against",
                metadata={"question_length": 0}
            )

        if not input_data.answer.strip():
            return EvaluationResult(
                score=0.0,
                reasoning="No answer provided to evaluate",
                metadata={"answer_length": 0}
            )

        # Use LLM to evaluate completeness
        evaluation = self._evaluate_with_llm(input_data)

        return EvaluationResult(
            score=evaluation["score"],
            reasoning=evaluation["reasoning"],
            metadata={
                "method": "llm_based",
                "question_length": len(input_data.question),
                "answer_length": len(input_data.answer)
            }
        )

    def _evaluate_with_llm(self, input_data: RAGEvaluationInput) -> Dict[str, Any]:
        """Use LLM to evaluate completeness"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là chuyên gia đánh giá chất lượng câu trả lời.
Hãy đánh giá độ đầy đủ (completeness) của câu trả lời đối với câu hỏi.

Độ đầy đủ đo lường mức độ câu trả lời giải quyết toàn diện tất cả các khía cạnh của câu hỏi, không bỏ sót thông tin quan trọng.

Tiêu chí đánh giá:
- 1.0: Câu trả lời hoàn toàn đầy đủ, giải quyết tất cả khía cạnh của câu hỏi
- 0.8: Câu trả lời rất đầy đủ, chỉ thiếu một vài chi tiết nhỏ không quan trọng
- 0.6: Câu trả lời khá đầy đủ, nhưng thiếu một số khía cạnh quan trọng
- 0.4: Câu trả lời chưa đầy đủ, thiếu nhiều thông tin quan trọng
- 0.2: Câu trả lời rất không đầy đủ, chỉ giải quyết một phần nhỏ của câu hỏi
- 0.0: Câu trả lời hoàn toàn không đầy đủ hoặc không liên quan

Hãy trả lời theo định dạng JSON:
{
    "score": <điểm số từ 0.0 đến 1.0>,
    "reasoning": "<giải thích chi tiết về đánh giá, bao gồm các khía cạnh còn thiếu>"
}"""),
            ("human", """Câu hỏi: {question}

Câu trả lời: {answer}

Hãy đánh giá độ đầy đủ của câu trả lời đối với câu hỏi.""")
        ])

        try:
            messages = prompt.format_messages(
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


class AspectCompletenessEvaluator(BaseEvaluator):
    """Evaluates completeness by checking coverage of question aspects"""

    def __init__(self):
        super().__init__(
            name="aspect_completeness",
            description="Measures completeness by analyzing coverage of different question aspects"
        )

    def evaluate(self, input_data: RAGEvaluationInput) -> EvaluationResult:
        """
        Evaluate completeness by breaking down question and checking aspect coverage
        """
        if not input_data.question.strip():
            return EvaluationResult(
                score=0.0,
                reasoning="No question provided",
                metadata={"question_length": 0}
            )

        if not input_data.answer.strip():
            return EvaluationResult(
                score=0.0,
                reasoning="No answer provided",
                metadata={"answer_length": 0}
            )

        # Extract question aspects
        question_aspects = self._extract_question_aspects(input_data.question)

        if not question_aspects:
            return EvaluationResult(
                score=0.5,
                reasoning="Unable to identify question aspects",
                metadata={"aspects_identified": 0}
            )

        # Check which aspects are addressed in the answer
        aspect_coverage = []
        covered_count = 0

        for aspect in question_aspects:
            is_covered = self._check_aspect_in_answer(aspect, input_data.answer)
            aspect_coverage.append({
                "aspect": aspect,
                "covered": is_covered
            })
            if is_covered:
                covered_count += 1

        completeness_score = covered_count / len(question_aspects)

        reasoning_parts = [
            f"Question has {len(question_aspects)} aspects",
            f"{covered_count} aspects are addressed in the answer",
            f"Completeness: {completeness_score:.2f}"
        ]

        if covered_count < len(question_aspects):
            missing_aspects = [a["aspect"] for a in aspect_coverage if not a["covered"]]
            reasoning_parts.append(f"Missing aspects: {missing_aspects}")

        return EvaluationResult(
            score=float(completeness_score),
            reasoning=" | ".join(reasoning_parts),
            metadata={
                "method": "aspect_analysis",
                "total_aspects": len(question_aspects),
                "covered_aspects": covered_count,
                "aspect_coverage": aspect_coverage
            }
        )

    def _extract_question_aspects(self, question: str) -> List[str]:
        """Extract different aspects/components from the question"""
        question = self._preprocess_text(question)

        aspects = []

        # Handle compound questions with multiple parts
        if any(word in question for word in ['và', 'hay', 'hoặc']):
            # Split by conjunctions
            parts = re.split(r'\b(và|hay|hoặc)\b', question)
            for part in parts:
                part = part.strip()
                if len(part) > 3 and part not in ['và', 'hay', 'hoặc']:
                    aspects.append(part)
        else:
            # Single aspect question
            aspects = [question]

        # Handle questions with multiple question words
        question_words = ['cái gì', 'như thế nào', 'tại sao', 'khi nào', 'ở đâu', 'ai', 'bao nhiêu']
        found_questions = []

        for q_word in question_words:
            if q_word in question:
                found_questions.append(q_word)

        if len(found_questions) > 1:
            # Multiple question types in one question
            aspects = []
            for q_word in found_questions:
                start_idx = question.find(q_word)
                if start_idx >= 0:
                    aspect = question[start_idx:].strip()
                    if len(aspect) > 5:
                        aspects.append(aspect)

        return aspects if aspects else [question]

    def _check_aspect_in_answer(self, aspect: str, answer: str) -> bool:
        """Check if a specific aspect is addressed in the answer"""
        aspect_keywords = self._extract_keywords(aspect)
        answer_processed = self._preprocess_text(answer)

        if not aspect_keywords:
            return False

        # Check keyword overlap
        matched_keywords = 0
        for keyword in aspect_keywords:
            if keyword in answer_processed:
                matched_keywords += 1

        # Consider aspect covered if 50%+ keywords are found
        coverage = matched_keywords / len(aspect_keywords)
        return coverage >= 0.5


class InformationDepthEvaluator(BaseEvaluator):
    """Evaluates completeness based on information depth and detail level"""

    def __init__(self):
        super().__init__(
            name="information_depth",
            description="Measures completeness based on depth and detail of information provided"
        )

    def evaluate(self, input_data: RAGEvaluationInput) -> EvaluationResult:
        """
        Evaluate completeness based on information depth
        """
        if not input_data.answer.strip():
            return EvaluationResult(
                score=0.0,
                reasoning="No answer provided",
                metadata={"answer_length": 0}
            )

        # Analyze answer characteristics
        depth_score = self._calculate_depth_score(input_data.answer)

        # Analyze question complexity and expected depth
        expected_depth = self._analyze_question_complexity(input_data.question)

        # Compare provided depth with expected depth
        completeness_score = min(1.0, depth_score / expected_depth) if expected_depth > 0 else depth_score

        reasoning = f"Answer depth score: {depth_score:.2f}, Expected depth: {expected_depth:.2f}, "
        reasoning += f"Completeness: {completeness_score:.2f}"

        return EvaluationResult(
            score=float(completeness_score),
            reasoning=reasoning,
            metadata={
                "method": "depth_analysis",
                "depth_score": depth_score,
                "expected_depth": expected_depth,
                "answer_length": len(input_data.answer)
            }
        )

    def _calculate_depth_score(self, answer: str) -> float:
        """Calculate depth score based on answer characteristics"""
        score = 0.0
        answer_processed = self._preprocess_text(answer)

        # Length factor (longer answers tend to be more complete)
        word_count = len(answer.split())
        length_score = min(1.0, word_count / 50)  # Max score at 50 words
        score += 0.3 * length_score

        # Detail indicators
        detail_indicators = [
            'bởi vì', 'do', 'vì', 'cụ thể', 'chi tiết', 'đầu tiên', 'tiếp theo',
            'cuối cùng', 'thêm vào đó', 'hơn nữa', 'quan trọng', 'lưu ý'
        ]
        detail_count = sum(1 for indicator in detail_indicators if indicator in answer_processed)
        detail_score = min(1.0, detail_count / 3)  # Max score with 3+ detail indicators
        score += 0.3 * detail_score

        # Specific information indicators
        specific_indicators = [
            r'\d+',  # Numbers
            r'(năm|tháng|ngày|giờ|phút)',  # Time units
            r'(vnđ|\$|đồng)',  # Currency
            r'(@|\.com|\.vn)',  # Contact info
            r'(điện thoại|sđt|email)',  # Contact types
        ]
        specific_count = 0
        for pattern in specific_indicators:
            if re.search(pattern, answer):
                specific_count += 1
        specific_score = min(1.0, specific_count / 2)  # Max score with 2+ specific indicators
        score += 0.4 * specific_score

        return score

    def _analyze_question_complexity(self, question: str) -> float:
        """Analyze expected depth based on question complexity"""
        question_processed = self._preprocess_text(question)

        complexity_score = 1.0  # Base expectation

        # Increase expectation for complex questions
        complexity_indicators = [
            'như thế nào', 'làm thế nào', 'cách', 'quy trình', 'hướng dẫn',  # How questions
            'tại sao', 'lí do', 'nguyên nhân',  # Why questions
            'so sánh', 'khác nhau', 'giống nhau',  # Comparison questions
            'tất cả', 'toàn bộ', 'chi tiết',  # Comprehensive questions
        ]

        for indicator in complexity_indicators:
            if indicator in question_processed:
                complexity_score += 0.2

        # Increase for compound questions
        if any(word in question_processed for word in ['và', 'hay', 'hoặc', 'cũng']):
            complexity_score += 0.3

        return min(2.0, complexity_score)  # Cap at 2.0
