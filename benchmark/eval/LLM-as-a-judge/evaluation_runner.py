"""
RAG Evaluation Runner

A comprehensive evaluation system for RAG (Retrieval-Augmented Generation) models.
Combines multiple evaluation methods to provide thorough assessment of RAG performance.
"""

from typing import Dict, List, Any, Optional
import json
import time
from datetime import datetime

from . import RAGEvaluationInput, create_evaluation_input
from .faithfulness import FaithfulnessEvaluator, SemanticFaithfulnessEvaluator
from .relevance import RelevanceEvaluator, SemanticRelevanceEvaluator, KeywordRelevanceEvaluator
from .context_relevance import ContextRelevanceEvaluator, DocumentRelevanceEvaluator, ContextCoverageEvaluator
from .completeness import CompletenessEvaluator, AspectCompletenessEvaluator, InformationDepthEvaluator
from .groundedness import GroundednessEvaluator, ClaimVerificationEvaluator, FactualConsistencyEvaluator


class RAGEvaluator:
    """Comprehensive RAG evaluation system"""

    def __init__(self):
        self.evaluators = {
            # Faithfulness evaluators
            "faithfulness_llm": FaithfulnessEvaluator(),
            "faithfulness_semantic": SemanticFaithfulnessEvaluator(),

            # Relevance evaluators
            "relevance_llm": RelevanceEvaluator(),
            "relevance_semantic": SemanticRelevanceEvaluator(),
            "relevance_keyword": KeywordRelevanceEvaluator(),

            # Context relevance evaluators
            "context_relevance_llm": ContextRelevanceEvaluator(),
            "document_relevance": DocumentRelevanceEvaluator(),
            "context_coverage": ContextCoverageEvaluator(),

            # Completeness evaluators
            "completeness_llm": CompletenessEvaluator(),
            "aspect_completeness": AspectCompletenessEvaluator(),
            "information_depth": InformationDepthEvaluator(),

            # Groundedness evaluators
            "groundedness_llm": GroundednessEvaluator(),
            "claim_verification": ClaimVerificationEvaluator(),
            "factual_consistency": FactualConsistencyEvaluator(),
        }

    def evaluate(self, input_data: RAGEvaluationInput, methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on RAG output

        Args:
            input_data: RAGEvaluationInput containing question, answer, context, etc.
            methods: List of evaluation methods to run. If None, runs all methods.

        Returns:
            Dict containing evaluation results and summary
        """
        if methods is None:
            methods = list(self.evaluators.keys())

        results = {}
        start_time = time.time()

        for method in methods:
            if method in self.evaluators:
                try:
                    result = self.evaluators[method].evaluate(input_data)
                    results[method] = {
                        "score": result.score,
                        "reasoning": result.reasoning,
                        "metadata": result.metadata,
                        "success": True
                    }
                except Exception as e:
                    results[method] = {
                        "score": 0.0,
                        "reasoning": f"Evaluation failed: {str(e)}",
                        "metadata": {},
                        "success": False
                    }

        evaluation_time = time.time() - start_time

        # Generate summary
        summary = self._generate_summary(results)

        return {
            "input_data": {
                "question": input_data.question,
                "answer_length": len(input_data.answer),
                "context_length": len(input_data.context),
                "num_docs": len(input_data.retrieved_docs) if input_data.retrieved_docs else 0
            },
            "results": results,
            "summary": summary,
            "metadata": {
                "evaluation_time": evaluation_time,
                "methods_run": len([r for r in results.values() if r["success"]]),
                "methods_failed": len([r for r in results.values() if not r["success"]]),
                "timestamp": datetime.now().isoformat()
            }
        }

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from evaluation results"""

        # Group results by category
        categories = {
            "faithfulness": ["faithfulness_llm", "faithfulness_semantic"],
            "relevance": ["relevance_llm", "relevance_semantic", "relevance_keyword"],
            "context_relevance": ["context_relevance_llm", "document_relevance", "context_coverage"],
            "completeness": ["completeness_llm", "aspect_completeness", "information_depth"],
            "groundedness": ["groundedness_llm", "claim_verification", "factual_consistency"]
        }

        category_scores = {}
        successful_evaluations = [r for r in results.values() if r["success"]]

        if not successful_evaluations:
            return {
                "overall_score": 0.0,
                "category_scores": {},
                "total_methods": len(results),
                "successful_methods": 0
            }

        # Calculate category scores
        for category, methods in categories.items():
            category_results = [results[method] for method in methods if method in results and results[method]["success"]]
            if category_results:
                avg_score = sum(r["score"] for r in category_results) / len(category_results)
                category_scores[category] = {
                    "score": avg_score,
                    "methods_used": len(category_results),
                    "methods_available": len(methods)
                }

        # Calculate overall score (weighted average)
        weights = {
            "faithfulness": 0.25,
            "relevance": 0.20,
            "context_relevance": 0.20,
            "completeness": 0.20,
            "groundedness": 0.15
        }

        overall_score = 0.0
        total_weight = 0.0

        for category, weight in weights.items():
            if category in category_scores:
                overall_score += category_scores[category]["score"] * weight
                total_weight += weight

        if total_weight > 0:
            overall_score /= total_weight

        return {
            "overall_score": overall_score,
            "category_scores": category_scores,
            "total_methods": len(results),
            "successful_methods": len(successful_evaluations),
            "weights_used": weights
        }


class BatchEvaluator:
    """Run evaluations on multiple RAG outputs"""

    def __init__(self):
        self.evaluator = RAGEvaluator()

    def evaluate_batch(
        self,
        evaluation_inputs: List[RAGEvaluationInput],
        methods: Optional[List[str]] = None,
        save_results: bool = False,
        output_file: str = "evaluation_results.json"
    ) -> Dict[str, Any]:
        """
        Evaluate multiple RAG outputs

        Args:
            evaluation_inputs: List of RAGEvaluationInput objects
            methods: Evaluation methods to use
            save_results: Whether to save results to file
            output_file: Output filename if saving results

        Returns:
            Dict containing batch evaluation results
        """
        batch_results = []
        start_time = time.time()

        for i, input_data in enumerate(evaluation_inputs):
            print(f"Evaluating sample {i+1}/{len(evaluation_inputs)}...")
            result = self.evaluator.evaluate(input_data, methods)
            batch_results.append(result)

        batch_time = time.time() - start_time

        # Aggregate results
        summary = self._aggregate_batch_results(batch_results)

        batch_output = {
            "batch_summary": summary,
            "individual_results": batch_results,
            "metadata": {
                "total_samples": len(evaluation_inputs),
                "evaluation_time": batch_time,
                "avg_time_per_sample": batch_time / len(evaluation_inputs),
                "timestamp": datetime.now().isoformat()
            }
        }

        if save_results:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(batch_output, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {output_file}")

        return batch_output

    def _aggregate_batch_results(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from batch evaluation"""

        if not batch_results:
            return {}

        # Collect all category scores
        category_scores = {}
        overall_scores = []

        for result in batch_results:
            summary = result.get("summary", {})
            overall_scores.append(summary.get("overall_score", 0.0))

            for category, cat_data in summary.get("category_scores", {}).items():
                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].append(cat_data["score"])

        # Calculate averages
        avg_category_scores = {}
        for category, scores in category_scores.items():
            avg_category_scores[category] = {
                "average_score": sum(scores) / len(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "std_dev": self._calculate_std_dev(scores)
            }

        avg_overall_score = sum(overall_scores) / len(overall_scores)

        return {
            "average_overall_score": avg_overall_score,
            "category_averages": avg_category_scores,
            "overall_score_distribution": {
                "min": min(overall_scores),
                "max": max(overall_scores),
                "average": avg_overall_score,
                "std_dev": self._calculate_std_dev(overall_scores)
            },
            "total_samples": len(batch_results)
        }

    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) <= 1:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5


# Convenience functions for easy evaluation
def evaluate_rag_output(
    question: str,
    answer: str,
    context: str,
    retrieved_docs: List[Dict[str, Any]] = None,
    ground_truth: str = None,
    methods: List[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a single RAG output

    Args:
        question: The input question
        answer: The generated answer
        context: The retrieved context
        retrieved_docs: List of retrieved documents
        ground_truth: Ground truth answer (optional)
        methods: Evaluation methods to use (optional)

    Returns:
        Evaluation results
    """
    evaluator = RAGEvaluator()
    input_data = create_evaluation_input(question, answer, context, retrieved_docs, ground_truth)
    return evaluator.evaluate(input_data, methods)


def evaluate_from_chat_response(
    chat_request: Dict[str, Any],
    chat_response: Dict[str, Any],
    retrieved_docs: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluate using ChatRequest/ChatResponse format (matching the API)

    Args:
        chat_request: Dict with 'question' key
        chat_response: Dict with 'answer' key
        retrieved_docs: Retrieved documents used to generate the answer

    Returns:
        Evaluation results
    """
    return evaluate_rag_output(
        question=chat_request.get("question", ""),
        answer=chat_response.get("answer", ""),
        context="",  # Context would need to be passed separately or reconstructed
        retrieved_docs=retrieved_docs
    )


if __name__ == "__main__":
    # Example usage
    sample_input = create_evaluation_input(
        question="Làm thế nào để đăng ký tài khoản trên hệ thống?",
        answer="Để đăng ký tài khoản, bạn cần truy cập trang đăng ký và điền đầy đủ thông tin cá nhân bao gồm tên, email và mật khẩu.",
        context="Hướng dẫn đăng ký: Truy cập trang web và nhấp vào nút 'Đăng ký'. Điền thông tin: họ tên, địa chỉ email, mật khẩu. Xác nhận email để hoàn tất.",
        retrieved_docs=[]
    )

    evaluator = RAGEvaluator()
    results = evaluator.evaluate(sample_input)

    print("Evaluation Results:")
    print(f"Overall Score: {results['summary']['overall_score']:.3f}")
    print("\nCategory Scores:")
    for category, data in results['summary']['category_scores'].items():
        print(f"  {category}: {data['score']:.3f}")
