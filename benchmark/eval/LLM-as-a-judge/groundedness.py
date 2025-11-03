"""
Groundedness Evaluation

Evaluates whether every claim in the generated answer is supported by the context.
Groundedness measures if the answer is fully grounded in the provided information.
"""

from typing import List, Dict, Any
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from . import BaseEvaluator, EvaluationResult, RAGEvaluationInput


class GroundednessEvaluator(BaseEvaluator):
    """Evaluates groundedness of answers in context"""

    def __init__(self):
        super().__init__(
            name="groundedness",
            description="Measures whether every claim in the answer is supported by the context"
        )
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.0,
            api_key=settings.openai_api_key
        )

    def evaluate(self, input_data: RAGEvaluationInput) -> EvaluationResult:
        """
        Evaluate groundedness using LLM-based claim verification

        Returns:
            EvaluationResult with score 0-1 where 1 means fully grounded
        """
        if not input_data.context.strip():
            return EvaluationResult(
                score=0.0,
                reasoning="No context provided to verify claims against",
                metadata={"context_length": 0}
            )

        if not input_data.answer.strip():
            return EvaluationResult(
                score=0.0,
                reasoning="No answer provided to evaluate",
                metadata={"answer_length": 0}
            )

        # Use LLM to evaluate groundedness
        evaluation = self._evaluate_with_llm(input_data)

        return EvaluationResult(
            score=evaluation["score"],
            reasoning=evaluation["reasoning"],
            metadata={
                "method": "llm_based",
                "context_length": len(input_data.context),
                "answer_length": len(input_data.answer)
            }
        )

    def _evaluate_with_llm(self, input_data: RAGEvaluationInput) -> Dict[str, Any]:
        """Use LLM to evaluate groundedness by checking each claim"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là chuyên gia kiểm chứng thông tin.
Hãy đánh giá độ dựa trên cơ sở (groundedness) của câu trả lời trong ngữ cảnh được cung cấp.

Độ dựa trên cơ sở đo lường mức độ mọi tuyên bố trong câu trả lời đều được hỗ trợ bởi thông tin trong ngữ cảnh.

Quy trình đánh giá:
1. Xác định từng tuyên bố riêng lẻ trong câu trả lời
2. Kiểm tra từng tuyên bố có được hỗ trợ bởi ngữ cảnh hay không
3. Tính tỷ lệ các tuyên bố được hỗ trợ

Tiêu chí đánh giá:
- 1.0: Tất cả tuyên bố đều được hỗ trợ đầy đủ bởi ngữ cảnh
- 0.8: Hầu hết tuyên bố đều được hỗ trợ, chỉ có 1-2 tuyên bố nhỏ không được hỗ trợ rõ ràng
- 0.6: Một số tuyên bố không được hỗ trợ, nhưng không mâu thuẫn với ngữ cảnh
- 0.4: Có nhiều tuyên bố không được hỗ trợ hoặc có thể mâu thuẫn với ngữ cảnh
- 0.2: Hầu hết tuyên bố không được hỗ trợ bởi ngữ cảnh
- 0.0: Không có tuyên bố nào được hỗ trợ hoặc câu trả lời hoàn toàn không dựa trên ngữ cảnh

Hãy trả lời theo định dạng JSON:
{
    "score": <điểm số từ 0.0 đến 1.0>,
    "reasoning": "<giải thích chi tiết về việc kiểm chứng từng tuyên bố>",
    "claims_analysis": [
        {
            "claim": "tuyên bố cụ thể",
            "supported": true/false,
            "evidence": "bằng chứng từ ngữ cảnh hoặc lý do không được hỗ trợ"
        }
    ]
}"""),
            ("human", """Ngữ cảnh:
{context}

Câu hỏi: {question}

Câu trả lời: {answer}

Hãy đánh giá độ dựa trên cơ sở của câu trả lời.""")
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
                "reasoning": result.get("reasoning", "Unable to parse evaluation"),
                "claims_analysis": result.get("claims_analysis", [])
            }

        except Exception as e:
            return {
                "score": 0.5,  # Default to neutral score on error
                "reasoning": f"Error during evaluation: {str(e)}",
                "claims_analysis": []
            }


class ClaimVerificationEvaluator(BaseEvaluator):
    """Evaluates groundedness by extracting and verifying individual claims"""

    def __init__(self):
        super().__init__(
            name="claim_verification",
            description="Extracts individual claims from answer and verifies each against context"
        )

    def evaluate(self, input_data: RAGEvaluationInput) -> EvaluationResult:
        """
        Evaluate groundedness by extracting and verifying claims
        """
        if not input_data.context.strip():
            return EvaluationResult(
                score=0.0,
                reasoning="No context provided for claim verification",
                metadata={"context_length": 0}
            )

        if not input_data.answer.strip():
            return EvaluationResult(
                score=0.0,
                reasoning="No answer provided to extract claims from",
                metadata={"answer_length": 0}
            )

        # Extract claims from answer
        claims = self._extract_claims(input_data.answer)

        if not claims:
            return EvaluationResult(
                score=0.5,
                reasoning="No verifiable claims found in answer",
                metadata={"claims_extracted": 0}
            )

        # Verify each claim against context
        verified_claims = 0
        claim_verifications = []

        for claim in claims:
            is_verified = self._verify_claim(claim, input_data.context)
            claim_verifications.append({
                "claim": claim,
                "verified": is_verified
            })
            if is_verified:
                verified_claims += 1

        groundedness_score = verified_claims / len(claims)

        reasoning = f"Extracted {len(claims)} claims from answer, "
        reasoning += f"{verified_claims} claims verified against context. "
        reasoning += f"Groundedness: {groundedness_score:.2f}"

        return EvaluationResult(
            score=float(groundedness_score),
            reasoning=reasoning,
            metadata={
                "method": "claim_extraction",
                "total_claims": len(claims),
                "verified_claims": verified_claims,
                "claim_verifications": claim_verifications
            }
        )

    def _extract_claims(self, answer: str) -> List[str]:
        """Extract individual claims from the answer"""
        # Split answer into sentences
        sentences = re.split(r'[.!?]+', answer.strip())

        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Only consider substantial sentences
                # Check if sentence contains a verifiable claim
                if self._is_verifiable_claim(sentence):
                    claims.append(sentence)

        return claims

    def _is_verifiable_claim(self, sentence: str) -> bool:
        """Check if a sentence contains a verifiable claim"""
        sentence_lower = sentence.lower()

        # Claims typically contain factual statements
        claim_indicators = [
            r'\b(là|sẽ|đã|có|không)\b',  # Vietnamese verbs
            r'\d+',  # Numbers
            r'(theo|từ|trong|vào)',  # Prepositions indicating specific info
            r'(hơn|ít|nhiều|ít)',  # Comparative words
            r'(tất cả|toàn bộ|chỉ|mỗi)',  # Quantifiers
        ]

        for pattern in claim_indicators:
            if re.search(pattern, sentence_lower):
                return True

        return False

    def _verify_claim(self, claim: str, context: str) -> bool:
        """Verify if a claim is supported by the context"""
        claim_processed = self._preprocess_text(claim)
        context_processed = self._preprocess_text(context)

        # Direct substring match
        if claim_processed in context_processed:
            return True

        # Key phrase matching
        claim_words = set(claim_processed.split())
        context_words = set(context_processed.split())

        # If most key words appear in context, consider verified
        # But be more strict than simple keyword matching
        key_words = [word for word in claim_words if len(word) > 2]
        if not key_words:
            return False

        matched_words = 0
        for word in key_words:
            if word in context_processed:
                matched_words += 1

        match_ratio = matched_words / len(key_words)
        return match_ratio >= 0.8  # Require 80% of key words to match


class FactualConsistencyEvaluator(BaseEvaluator):
    """Evaluates factual consistency between answer and context"""

    def __init__(self):
        super().__init__(
            name="factual_consistency",
            description="Checks for factual contradictions between answer and context"
        )

    def evaluate(self, input_data: RAGEvaluationInput) -> EvaluationResult:
        """
        Evaluate factual consistency by checking for contradictions
        """
        if not input_data.context.strip():
            return EvaluationResult(
                score=0.0,
                reasoning="No context provided for consistency check",
                metadata={"context_length": 0}
            )

        if not input_data.answer.strip():
            return EvaluationResult(
                score=0.0,
                reasoning="No answer provided to check consistency",
                metadata={"answer_length": 0}
            )

        # Extract facts from both answer and context
        answer_facts = self._extract_facts(input_data.answer)
        context_facts = self._extract_facts(input_data.context)

        if not answer_facts:
            return EvaluationResult(
                score=0.5,
                reasoning="No verifiable facts found in answer",
                metadata={"answer_facts": 0}
            )

        # Check for contradictions
        contradictions = []
        consistent_facts = 0

        for answer_fact in answer_facts:
            is_consistent = True
            contradiction_reason = None

            # Check against each context fact for contradictions
            for context_fact in context_facts:
                contradiction = self._check_contradiction(answer_fact, context_fact)
                if contradiction:
                    is_consistent = False
                    contradiction_reason = f"Contradicts context fact: '{context_fact}'"
                    break

            if is_consistent:
                consistent_facts += 1
            else:
                contradictions.append({
                    "answer_fact": answer_fact,
                    "reason": contradiction_reason
                })

        # Score based on consistency (1.0 if no contradictions, 0.0 if contradictions found)
        consistency_score = 1.0 if not contradictions else (consistent_facts / len(answer_facts))

        reasoning_parts = [
            f"Found {len(answer_facts)} facts in answer",
            f"{consistent_facts} facts are consistent with context"
        ]

        if contradictions:
            reasoning_parts.append(f"Found {len(contradictions)} contradictions")

        return EvaluationResult(
            score=float(consistency_score),
            reasoning=" | ".join(reasoning_parts),
            metadata={
                "method": "factual_consistency",
                "total_answer_facts": len(answer_facts),
                "consistent_facts": consistent_facts,
                "contradictions": contradictions
            }
        )

    def _extract_facts(self, text: str) -> List[str]:
        """Extract factual statements from text"""
        sentences = re.split(r'[.!?]+', text.strip())

        facts = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and self._is_factual(sentence):
                facts.append(sentence)

        return facts

    def _is_factual(self, sentence: str) -> bool:
        """Check if sentence appears to be factual"""
        # Look for factual indicators
        factual_patterns = [
            r'\b(là|được|có|không)\b',
            r'\d+',
            r'(năm|tháng|ngày)',
        ]

        return any(re.search(pattern, sentence.lower()) for pattern in factual_patterns)

    def _check_contradiction(self, fact1: str, fact2: str) -> bool:
        """Check if two facts contradict each other"""
        # Simple contradiction detection based on antonyms and negations
        fact1_lower = fact1.lower()
        fact2_lower = fact2.lower()

        # Check for direct negations
        negation_pairs = [
            ('không', 'có'),
            ('chưa', 'đã'),
            ('ít', 'nhiều'),
            ('nhỏ', 'lớn'),
            ('sai', 'đúng'),
        ]

        for neg, pos in negation_pairs:
            if ((neg in fact1_lower and pos in fact2_lower) or
                (neg in fact2_lower and pos in fact1_lower)):
                return True

        # Check for contradictory quantities
        # This is a simplified approach - real contradiction detection is complex
        return False
