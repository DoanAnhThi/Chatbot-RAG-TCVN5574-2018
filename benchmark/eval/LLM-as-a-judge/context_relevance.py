"""
Context Relevance Evaluation

Evaluates whether the retrieved documents/context are relevant to the question.
Context relevance measures how well the retrieved information addresses the query.
"""

from typing import List, Dict, Any
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.config import settings
from . import BaseEvaluator, EvaluationResult, RAGEvaluationInput


class ContextRelevanceEvaluator(BaseEvaluator):
    """Evaluates relevance of retrieved context to the question"""

    def __init__(self):
        super().__init__(
            name="context_relevance",
            description="Measures how well the retrieved documents address the question"
        )
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.0,
            api_key=settings.openai_api_key
        )

    def evaluate(self, input_data: RAGEvaluationInput) -> EvaluationResult:
        """
        Evaluate context relevance using LLM-based assessment

        Returns:
            EvaluationResult with score 0-1 where 1 means context is highly relevant
        """
        if not input_data.question.strip():
            return EvaluationResult(
                score=0.0,
                reasoning="No question provided to evaluate context relevance against",
                metadata={"question_length": 0}
            )

        if not input_data.context.strip():
            return EvaluationResult(
                score=0.0,
                reasoning="No context provided to evaluate",
                metadata={"context_length": 0}
            )

        # Use LLM to evaluate context relevance
        evaluation = self._evaluate_with_llm(input_data)

        return EvaluationResult(
            score=evaluation["score"],
            reasoning=evaluation["reasoning"],
            metadata={
                "method": "llm_based",
                "question_length": len(input_data.question),
                "context_length": len(input_data.context),
                "num_docs": len(input_data.retrieved_docs) if input_data.retrieved_docs else 0
            }
        )

    def _evaluate_with_llm(self, input_data: RAGEvaluationInput) -> Dict[str, Any]:
        """Use LLM to evaluate context relevance"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là chuyên gia đánh giá chất lượng thông tin.
Hãy đánh giá độ liên quan của ngữ cảnh được truy xuất đối với câu hỏi.

Độ liên quan của ngữ cảnh đo lường mức độ thông tin trong ngữ cảnh giúp trả lời câu hỏi.

Tiêu chí đánh giá:
- 1.0: Ngữ cảnh rất liên quan, chứa đầy đủ thông tin để trả lời câu hỏi
- 0.8: Ngữ cảnh khá liên quan, chứa hầu hết thông tin cần thiết
- 0.6: Ngữ cảnh có liên quan một phần, chứa một số thông tin hữu ích
- 0.4: Ngữ cảnh ít liên quan, chỉ chứa một ít thông tin liên quan
- 0.2: Ngữ cảnh hầu như không liên quan đến câu hỏi
- 0.0: Ngữ cảnh hoàn toàn không liên quan hoặc không chứa thông tin hữu ích

Hãy trả lời theo định dạng JSON:
{
    "score": <điểm số từ 0.0 đến 1.0>,
    "reasoning": "<giải thích chi tiết về đánh giá>"
}"""),
            ("human", """Câu hỏi: {question}

Ngữ cảnh được truy xuất:
{context}

Hãy đánh giá độ liên quan của ngữ cảnh đối với câu hỏi.""")
        ])

        try:
            messages = prompt.format_messages(
                question=input_data.question,
                context=input_data.context
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


class DocumentRelevanceEvaluator(BaseEvaluator):
    """Evaluates relevance of individual retrieved documents"""

    def __init__(self):
        super().__init__(
            name="document_relevance",
            description="Evaluates relevance of each retrieved document individually"
        )
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        except ImportError:
            self.model = None

    def evaluate(self, input_data: RAGEvaluationInput) -> EvaluationResult:
        """
        Evaluate relevance of each retrieved document
        """
        if not input_data.question.strip():
            return EvaluationResult(
                score=0.0,
                reasoning="No question provided",
                metadata={"question_length": 0}
            )

        if not input_data.retrieved_docs:
            return EvaluationResult(
                score=0.0,
                reasoning="No retrieved documents to evaluate",
                metadata={"num_docs": 0}
            )

        # Evaluate each document
        doc_scores = []
        total_score = 0

        for i, doc in enumerate(input_data.retrieved_docs):
            doc_content = doc.get('page_content', '') if isinstance(doc, dict) else str(doc)
            doc_score = self._evaluate_single_document(input_data.question, doc_content)
            doc_scores.append({
                "doc_index": i,
                "score": doc_score,
                "content_preview": doc_content[:200] + "..." if len(doc_content) > 200 else doc_content
            })
            total_score += doc_score

        # Average score across all documents
        avg_score = total_score / len(input_data.retrieved_docs)

        # Weight by document position (earlier documents more important)
        weighted_score = self._calculate_weighted_score(doc_scores)

        reasoning = f"Evaluated {len(doc_scores)} documents. "
        reasoning += f"Average relevance: {avg_score:.3f}, Weighted score: {weighted_score:.3f}"

        return EvaluationResult(
            score=float(weighted_score),
            reasoning=reasoning,
            metadata={
                "method": "per_document_evaluation",
                "num_docs": len(doc_scores),
                "avg_score": avg_score,
                "weighted_score": weighted_score,
                "doc_scores": doc_scores
            }
        )

    def _evaluate_single_document(self, question: str, doc_content: str) -> float:
        """Evaluate relevance of a single document"""
        if not doc_content.strip():
            return 0.0

        if self.model is None:
            # Fallback to keyword matching
            return self._keyword_relevance_score(question, doc_content)

        # Use semantic similarity
        try:
            question_embedding = self.model.encode(question)
            doc_embedding = self.model.encode(doc_content[:1000])  # Limit content length

            import numpy as np
            similarity = np.dot(question_embedding, doc_embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(doc_embedding)
            )

            # Convert to 0-1 scale
            return float((similarity + 1) / 2)
        except Exception:
            return self._keyword_relevance_score(question, doc_content)

    def _keyword_relevance_score(self, question: str, doc_content: str) -> float:
        """Calculate relevance score based on keyword overlap"""
        question_keywords = self._extract_keywords(question)
        doc_keywords = self._extract_keywords(doc_content)

        if not question_keywords:
            return 0.5

        overlap = len(set(question_keywords).intersection(set(doc_keywords)))
        return overlap / len(question_keywords)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        text = self._preprocess_text(text)

        # Vietnamese stopwords
        vietnamese_stopwords = {
            'là', 'có', 'không', 'được', 'cái', 'của', 'và', 'với', 'cho', 'như',
            'từ', 'đến', 'trong', 'trên', 'dưới', 'sau', 'trước', 'bên', 'giữa',
            'thì', 'mà', 'nếu', 'khi', 'để', 'tại', 'về', 'hay', 'hoặc', 'cũng'
        }

        words = re.findall(r'\b\w+\b', text)
        keywords = [word for word in words if len(word) > 1 and word not in vietnamese_stopwords]

        return keywords

    def _calculate_weighted_score(self, doc_scores: List[Dict[str, Any]]) -> float:
        """Calculate weighted average where earlier documents have higher weight"""
        if not doc_scores:
            return 0.0

        total_weight = 0
        weighted_sum = 0

        for i, doc_score in enumerate(doc_scores):
            # Weight decreases exponentially with position
            weight = 1.0 / (i + 1)  # 1, 0.5, 0.33, 0.25, ...
            weighted_sum += doc_score["score"] * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0


class ContextCoverageEvaluator(BaseEvaluator):
    """Evaluates how well the context covers different aspects of the question"""

    def __init__(self):
        super().__init__(
            name="context_coverage",
            description="Measures how comprehensively the context covers the question"
        )

    def evaluate(self, input_data: RAGEvaluationInput) -> EvaluationResult:
        """
        Evaluate how well the context covers different aspects of the question
        """
        if not input_data.question.strip():
            return EvaluationResult(
                score=0.0,
                reasoning="No question provided",
                metadata={"question_length": 0}
            )

        if not input_data.context.strip():
            return EvaluationResult(
                score=0.0,
                reasoning="No context provided",
                metadata={"context_length": 0}
            )

        # Break down question into aspects/components
        question_aspects = self._extract_question_aspects(input_data.question)

        if not question_aspects:
            return EvaluationResult(
                score=0.5,
                reasoning="Unable to break down question into aspects",
                metadata={"aspects_found": 0}
            )

        # Check coverage for each aspect
        covered_aspects = 0
        aspect_coverage = []

        for aspect in question_aspects:
            is_covered = self._check_aspect_coverage(aspect, input_data.context)
            aspect_coverage.append({
                "aspect": aspect,
                "covered": is_covered
            })
            if is_covered:
                covered_aspects += 1

        coverage_score = covered_aspects / len(question_aspects)

        reasoning = f"Question has {len(question_aspects)} aspects, "
        reasoning += f"{covered_aspects} are covered by context. "
        reasoning += f"Coverage: {coverage_score:.2f}"

        return EvaluationResult(
            score=float(coverage_score),
            reasoning=reasoning,
            metadata={
                "method": "aspect_coverage",
                "total_aspects": len(question_aspects),
                "covered_aspects": covered_aspects,
                "aspect_coverage": aspect_coverage
            }
        )

    def _extract_question_aspects(self, question: str) -> List[str]:
        """Break down question into key aspects/components"""
        question = self._preprocess_text(question)

        # Simple aspect extraction based on conjunctions and question structure
        aspects = []

        # Split by conjunctions that indicate multiple aspects
        conjunctions = ['và', 'hoặc', 'hay', 'cũng như', 'bao gồm']
        parts = [question]

        for conj in conjunctions:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(conj))
            parts = new_parts

        # Clean and filter aspects
        for part in parts:
            part = part.strip()
            if len(part) > 5:  # Only consider substantial aspects
                aspects.append(part)

        # If no aspects found from conjunctions, treat whole question as one aspect
        if not aspects:
            aspects = [question]

        return aspects

    def _check_aspect_coverage(self, aspect: str, context: str) -> bool:
        """Check if an aspect is covered in the context"""
        aspect_keywords = self._extract_keywords(aspect)
        context_processed = self._preprocess_text(context)

        # Check for keyword overlap
        overlap_count = 0
        for keyword in aspect_keywords:
            if keyword in context_processed:
                overlap_count += 1

        # Consider covered if 60% of keywords are found
        coverage_ratio = overlap_count / len(aspect_keywords) if aspect_keywords else 0
        return coverage_ratio >= 0.6
