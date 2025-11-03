"""
Relevance Evaluation

Evaluates whether the generated answer is relevant to the question asked.
Relevance measures how well the answer addresses the specific query.
"""

from typing import List, Dict, Any
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from . import BaseEvaluator, EvaluationResult, RAGEvaluationInput


class RelevanceEvaluator(BaseEvaluator):
    """Evaluates relevance of answers to the question"""

    def __init__(self):
        super().__init__(
            name="relevance",
            description="Measures how well the answer addresses the specific question asked"
        )
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.0,
            api_key=settings.openai_api_key
        )

    def evaluate(self, input_data: RAGEvaluationInput) -> EvaluationResult:
        """
        Evaluate relevance using LLM-based assessment

        Returns:
            EvaluationResult with score 0-1 where 1 means fully relevant
        """
        if not input_data.question.strip():
            return EvaluationResult(
                score=0.0,
                reasoning="No question provided to evaluate relevance against",
                metadata={"question_length": 0}
            )

        if not input_data.answer.strip():
            return EvaluationResult(
                score=0.0,
                reasoning="No answer provided to evaluate",
                metadata={"answer_length": 0}
            )

        # Use LLM to evaluate relevance
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
        """Use LLM to evaluate relevance"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là chuyên gia đánh giá chất lượng câu trả lời.
Hãy đánh giá độ liên quan (relevance) của câu trả lời đối với câu hỏi được đặt ra.

Độ liên quan đo lường mức độ câu trả lời trực tiếp giải quyết và trả lời đúng câu hỏi.

Tiêu chí đánh giá:
- 1.0: Câu trả lời hoàn toàn liên quan, trực tiếp giải quyết câu hỏi
- 0.8: Câu trả lời chủ yếu liên quan, nhưng có thể thêm một số thông tin thừa
- 0.6: Câu trả lời có liên quan một phần, nhưng thiếu một số khía cạnh quan trọng
- 0.4: Câu trả lời ít liên quan, chỉ giải quyết một phần nhỏ của câu hỏi
- 0.2: Câu trả lời hầu như không liên quan đến câu hỏi
- 0.0: Câu trả lời hoàn toàn không liên quan hoặc đi lạc chủ đề

Hãy trả lời theo định dạng JSON:
{
    "score": <điểm số từ 0.0 đến 1.0>,
    "reasoning": "<giải thích chi tiết về đánh giá>"
}"""),
            ("human", """Câu hỏi: {question}

Câu trả lời: {answer}

Hãy đánh giá độ liên quan của câu trả lời đối với câu hỏi.""")
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


class SemanticRelevanceEvaluator(BaseEvaluator):
    """Evaluates relevance using semantic similarity between question and answer"""

    def __init__(self):
        super().__init__(
            name="semantic_relevance",
            description="Measures semantic similarity between question and answer"
        )
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        except ImportError:
            self.model = None

    def evaluate(self, input_data: RAGEvaluationInput) -> EvaluationResult:
        """
        Evaluate relevance using semantic similarity
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

        if self.model is None:
            return EvaluationResult(
                score=0.5,
                reasoning="Sentence transformers not available, using fallback method",
                metadata={"method": "fallback"}
            )

        # Calculate semantic similarity
        question_embedding = self.model.encode(input_data.question)
        answer_embedding = self.model.encode(input_data.answer)

        # Cosine similarity
        import numpy as np
        similarity = np.dot(question_embedding, answer_embedding) / (
            np.linalg.norm(question_embedding) * np.linalg.norm(answer_embedding)
        )

        # Normalize to 0-1 scale (cosine similarity is -1 to 1, but typically positive for text)
        score = (similarity + 1) / 2

        reasoning = f"Semantic similarity between question and answer: {similarity:.3f}"

        return EvaluationResult(
            score=float(score),
            reasoning=reasoning,
            metadata={
                "method": "semantic_similarity",
                "raw_similarity": float(similarity),
                "question_length": len(input_data.question),
                "answer_length": len(input_data.answer)
            }
        )


class KeywordRelevanceEvaluator(BaseEvaluator):
    """Evaluates relevance using keyword overlap and question analysis"""

    def __init__(self):
        super().__init__(
            name="keyword_relevance",
            description="Measures relevance based on keyword overlap and question type analysis"
        )

    def evaluate(self, input_data: RAGEvaluationInput) -> EvaluationResult:
        """
        Evaluate relevance using keyword matching and linguistic analysis
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

        # Extract keywords from question
        question_keywords = self._extract_keywords(input_data.question)
        answer_keywords = self._extract_keywords(input_data.answer)

        if not question_keywords:
            return EvaluationResult(
                score=0.5,
                reasoning="Unable to extract keywords from question",
                metadata={"question_keywords": 0}
            )

        # Calculate overlap
        overlap = len(set(question_keywords).intersection(set(answer_keywords)))
        total_question_keywords = len(set(question_keywords))

        # Base score from keyword overlap
        overlap_score = overlap / total_question_keywords if total_question_keywords > 0 else 0

        # Analyze question type and answer structure
        question_type_score = self._analyze_question_type_match(
            input_data.question, input_data.answer
        )

        # Combine scores (weighted average)
        final_score = 0.7 * overlap_score + 0.3 * question_type_score

        reasoning_parts = [
            f"Keyword overlap: {overlap}/{total_question_keywords}",
            f"Question type match score: {question_type_score:.2f}",
            f"Question keywords: {question_keywords[:10]}",  # Show first 10
            f"Answer keywords: {answer_keywords[:10]}"
        ]

        return EvaluationResult(
            score=float(final_score),
            reasoning=" | ".join(reasoning_parts),
            metadata={
                "method": "keyword_overlap",
                "keyword_overlap": overlap,
                "total_question_keywords": total_question_keywords,
                "overlap_score": overlap_score,
                "question_type_score": question_type_score,
                "question_keywords": question_keywords,
                "answer_keywords": answer_keywords
            }
        )

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        text = self._preprocess_text(text)

        # Remove stopwords (Vietnamese common words)
        vietnamese_stopwords = {
            'là', 'có', 'không', 'được', 'cái', 'của', 'và', 'với', 'cho', 'như',
            'từ', 'đến', 'trong', 'trên', 'dưới', 'sau', 'trước', 'bên', 'giữa',
            'thì', 'mà', 'nếu', 'khi', 'để', 'tại', 'về', 'hay', 'hoặc', 'cũng'
        }

        words = re.findall(r'\b\w+\b', text)
        keywords = []

        for word in words:
            if len(word) > 1 and word not in vietnamese_stopwords:
                keywords.append(word)

        return keywords

    def _analyze_question_type_match(self, question: str, answer: str) -> float:
        """Analyze if answer matches the expected question type"""
        question_lower = question.lower()
        answer_lower = answer.lower()

        # Question type indicators
        question_types = {
            'what': ['cái gì', 'gì', 'là gì'],
            'how': ['như thế nào', 'làm thế nào', 'cách', 'bằng cách nào'],
            'why': ['tại sao', 'vì sao', 'lí do'],
            'when': ['khi nào', 'thời gian', 'bao giờ'],
            'where': ['ở đâu', 'tại đâu', 'địa điểm'],
            'who': ['ai', 'người nào'],
            'which': ['loại nào', 'mẫu nào', 'đâu']
        }

        # Check question type
        detected_type = None
        for qtype, indicators in question_types.items():
            for indicator in indicators:
                if indicator in question_lower:
                    detected_type = qtype
                    break
            if detected_type:
                break

        if not detected_type:
            return 0.5  # Neutral if can't determine question type

        # Check if answer provides appropriate information for question type
        type_scores = {
            'what': self._score_what_answer(answer_lower),
            'how': self._score_how_answer(answer_lower),
            'why': self._score_why_answer(answer_lower),
            'when': self._score_when_answer(answer_lower),
            'where': self._score_where_answer(answer_lower),
            'who': self._score_who_answer(answer_lower),
            'which': self._score_which_answer(answer_lower)
        }

        return type_scores.get(detected_type, 0.5)

    def _score_what_answer(self, answer: str) -> float:
        """Score if answer is appropriate for 'what' questions"""
        # What questions expect definitions, descriptions, explanations
        if any(word in answer for word in ['là', 'được', 'có', 'bao gồm']):
            return 0.8
        return 0.6

    def _score_how_answer(self, answer: str) -> float:
        """Score if answer is appropriate for 'how' questions"""
        # How questions expect procedures, methods, steps
        if any(word in answer for word in ['bước', 'cách', 'thực hiện', 'làm']):
            return 0.8
        return 0.6

    def _score_why_answer(self, answer: str) -> float:
        """Score if answer is appropriate for 'why' questions"""
        # Why questions expect reasons, causes, explanations
        if any(word in answer for word in ['vì', 'do', 'bởi', 'lí do']):
            return 0.8
        return 0.6

    def _score_when_answer(self, answer: str) -> float:
        """Score if answer is appropriate for 'when' questions"""
        # When questions expect time-related information
        time_indicators = ['năm', 'tháng', 'ngày', 'giờ', 'lúc', 'khi']
        if any(word in answer for word in time_indicators):
            return 0.8
        return 0.6

    def _score_where_answer(self, answer: str) -> float:
        """Score if answer is appropriate for 'where' questions"""
        # Where questions expect location information
        location_indicators = ['tại', 'ở', 'địa chỉ', 'địa điểm', 'khu vực']
        if any(word in answer for word in location_indicators):
            return 0.8
        return 0.6

    def _score_who_answer(self, answer: str) -> float:
        """Score if answer is appropriate for 'who' questions"""
        # Who questions expect person/organization information
        if any(word in answer for word in ['người', 'công ty', 'tổ chức', 'anh', 'chị']):
            return 0.8
        return 0.6

    def _score_which_answer(self, answer: str) -> float:
        """Score if answer is appropriate for 'which' questions"""
        # Which questions expect selection/comparison information
        if any(word in answer for word in ['loại', 'mẫu', 'dòng', 'series']):
            return 0.8
        return 0.6
