"""
Diversity Metric for RAG System Evaluation

Diversity measures the variety and richness of generated answers, including:
- Lexical diversity (unique words, vocabulary richness)
- Semantic diversity (different meanings/concepts)
- Structural diversity (sentence variety, length variation)
- Answer variety across similar queries

High diversity indicates the system can generate varied responses and
avoid repetitive or generic answers.
"""

from typing import List, Union, Dict, Set, Optional, Any
import re
import math
import statistics
from collections import Counter

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    SentenceTransformer = None


class DiversityAnalyzer:
    """Analyzes diversity in generated answers."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize diversity analyzer.

        Args:
            embedding_model: Name of the sentence transformer model for semantic diversity
        """
        self.embedding_model_name = embedding_model
        if SentenceTransformer is not None:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
            except Exception:
                print(f"Warning: Could not load embedding model '{embedding_model}'")
                self.embedding_model = None
        else:
            self.embedding_model = None

    def analyze_lexical_diversity(self, answers: List[str]) -> Dict[str, float]:
        """
        Analyze lexical diversity (vocabulary richness) in answers.

        Args:
            answers: List of generated answers

        Returns:
            Lexical diversity metrics
        """
        if not answers:
            return {}

        # Combine all answers for overall analysis
        combined_text = ' '.join(answers)

        # Tokenize
        tokens = self._tokenize_text(combined_text)
        unique_tokens = set(tokens)

        # Basic metrics
        metrics = {
            'total_tokens': len(tokens),
            'unique_tokens': len(unique_tokens),
            'type_token_ratio': len(unique_tokens) / len(tokens) if tokens else 0,
            'vocabulary_richness': len(unique_tokens) / len(tokens) if tokens else 0,
        }

        # Advanced metrics
        if len(tokens) > 0:
            # Hapax legomena (words appearing only once)
            token_counts = Counter(tokens)
            hapax_count = sum(1 for count in token_counts.values() if count == 1)
            metrics['hapax_ratio'] = hapax_count / len(unique_tokens) if unique_tokens else 0

            # Yule's K measure (vocabulary diversity)
            metrics['yules_k'] = self._calculate_yules_k(token_counts)

            # Simpson's diversity index
            metrics['simpsons_diversity'] = self._calculate_simpsons_diversity(token_counts)

        # Per-answer diversity
        answer_diversities = []
        for answer in answers:
            answer_tokens = self._tokenize_text(answer)
            if answer_tokens:
                unique_in_answer = len(set(answer_tokens))
                diversity = unique_in_answer / len(answer_tokens)
                answer_diversities.append(diversity)

        if answer_diversities:
            metrics['avg_answer_diversity'] = statistics.mean(answer_diversities)
            metrics['answer_diversity_std'] = statistics.stdev(answer_diversities) if len(answer_diversities) > 1 else 0

        return metrics

    def analyze_structural_diversity(self, answers: List[str]) -> Dict[str, Any]:
        """
        Analyze structural diversity (sentence variety, length variation).

        Args:
            answers: List of generated answers

        Returns:
            Structural diversity metrics
        """
        if not answers:
            return {}

        # Sentence analysis
        sentence_lengths = []
        sentence_counts = []

        for answer in answers:
            sentences = re.split(r'[.!?]+', answer)
            sentences = [s.strip() for s in sentences if s.strip()]

            sentence_counts.append(len(sentences))

            for sentence in sentences:
                words = sentence.split()
                if words:
                    sentence_lengths.append(len(words))

        # Length statistics
        metrics = {}

        if sentence_lengths:
            metrics.update({
                'avg_sentence_length': statistics.mean(sentence_lengths),
                'sentence_length_std': statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0,
                'min_sentence_length': min(sentence_lengths),
                'max_sentence_length': max(sentence_lengths),
            })

        if sentence_counts:
            metrics.update({
                'avg_sentences_per_answer': statistics.mean(sentence_counts),
                'sentence_count_std': statistics.stdev(sentence_counts) if len(sentence_counts) > 1 else 0,
                'total_sentences': sum(sentence_counts),
            })

        # Structural variety (coefficient of variation)
        if sentence_lengths and len(sentence_lengths) > 1:
            metrics['sentence_length_variability'] = statistics.stdev(sentence_lengths) / statistics.mean(sentence_lengths)

        if sentence_counts and len(sentence_counts) > 1:
            metrics['sentence_count_variability'] = statistics.stdev(sentence_counts) / statistics.mean(sentence_counts)

        return metrics

    def analyze_semantic_diversity(self, answers: List[str]) -> Dict[str, Any]:
        """
        Analyze semantic diversity using embeddings.

        Args:
            answers: List of generated answers

        Returns:
            Semantic diversity metrics
        """
        if not answers or not self.embedding_model:
            return {'semantic_diversity_available': False}

        try:
            # Get embeddings
            embeddings = self.embedding_model.encode(answers)

            # Calculate pairwise similarities
            similarity_matrix = cosine_similarity(embeddings)

            # Average similarity (lower = more diverse)
            # Exclude diagonal (self-similarity)
            n = len(answers)
            total_similarity = 0
            count = 0

            for i in range(n):
                for j in range(i + 1, n):
                    total_similarity += similarity_matrix[i, j]
                    count += 1

            avg_similarity = total_similarity / count if count > 0 else 0

            # Diversity is inverse of average similarity
            semantic_diversity = 1 - avg_similarity

            # Additional metrics
            similarities = []
            for i in range(n):
                for j in range(i + 1, n):
                    similarities.append(similarity_matrix[i, j])

            metrics = {
                'semantic_diversity_available': True,
                'semantic_diversity_score': semantic_diversity,
                'average_pairwise_similarity': avg_similarity,
                'min_pairwise_similarity': min(similarities) if similarities else 0,
                'max_pairwise_similarity': max(similarities) if similarities else 0,
            }

            if len(similarities) > 1:
                metrics['similarity_std'] = statistics.stdev(similarities)

            return metrics

        except Exception as e:
            print(f"Error in semantic diversity analysis: {e}")
            return {'semantic_diversity_available': False, 'error': str(e)}

    def analyze_answer_variety(self, answers: List[str],
                              questions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze overall answer variety and uniqueness.

        Args:
            answers: List of generated answers
            questions: Optional corresponding questions

        Returns:
            Answer variety metrics
        """
        if not answers:
            return {}

        metrics = {
            'total_answers': len(answers),
            'unique_answers': len(set(answers)),
            'answer_uniqueness_ratio': len(set(answers)) / len(answers),
        }

        # Length distribution
        answer_lengths = [len(answer.split()) for answer in answers]
        if answer_lengths:
            metrics.update({
                'avg_answer_length': statistics.mean(answer_lengths),
                'answer_length_std': statistics.stdev(answer_lengths) if len(answer_lengths) > 1 else 0,
                'length_variability': statistics.stdev(answer_lengths) / statistics.mean(answer_lengths) if len(answer_lengths) > 1 else 0,
            })

        # Question-answer alignment analysis (if questions provided)
        if questions and len(questions) == len(answers):
            # Check for generic answers
            generic_phrases = [
                "i don't know", "i'm not sure", "it depends", "that's a good question",
                "based on the context", "according to the information"
            ]

            generic_count = 0
            for answer in answers:
                answer_lower = answer.lower()
                if any(phrase in answer_lower for phrase in generic_phrases):
                    generic_count += 1

            metrics['generic_answer_ratio'] = generic_count / len(answers)

            # Question-answer length correlation
            question_lengths = [len(q.split()) for q in questions]
            if len(question_lengths) > 1 and len(answer_lengths) > 1:
                # Simple correlation coefficient
                q_mean = statistics.mean(question_lengths)
                a_mean = statistics.mean(answer_lengths)
                q_std = statistics.stdev(question_lengths)
                a_std = statistics.stdev(answer_lengths)

                if q_std > 0 and a_std > 0:
                    correlation = sum((q - q_mean) * (a - a_mean) for q, a in zip(question_lengths, answer_lengths))
                    correlation /= (len(question_lengths) - 1) * q_std * a_std
                    metrics['question_answer_length_correlation'] = correlation

        return metrics

    def get_comprehensive_diversity_score(self, answers: List[str],
                                        questions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get comprehensive diversity analysis.

        Args:
            answers: List of generated answers
            questions: Optional corresponding questions

        Returns:
            Comprehensive diversity metrics
        """
        lexical = self.analyze_lexical_diversity(answers)
        structural = self.analyze_structural_diversity(answers)
        semantic = self.analyze_semantic_diversity(answers)
        variety = self.analyze_answer_variety(answers, questions)

        # Overall diversity score (weighted combination)
        scores = []

        if 'type_token_ratio' in lexical:
            scores.append(lexical['type_token_ratio'])

        if 'sentence_length_variability' in structural:
            scores.append(min(structural['sentence_length_variability'], 1.0))  # Cap at 1.0

        if semantic.get('semantic_diversity_available', False):
            scores.append(semantic['semantic_diversity_score'])

        if 'answer_uniqueness_ratio' in variety:
            scores.append(variety['answer_uniqueness_ratio'])

        overall_diversity = statistics.mean(scores) if scores else 0.0

        return {
            'overall_diversity_score': overall_diversity,
            'lexical_diversity': lexical,
            'structural_diversity': structural,
            'semantic_diversity': semantic,
            'answer_variety': variety,
            'diversity_components': len(scores)
        }

    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()

    def _calculate_yules_k(self, token_counts: Counter) -> float:
        """Calculate Yule's K measure of vocabulary diversity."""
        if not token_counts:
            return 0.0

        m1 = len(token_counts)  # Number of unique words
        m2 = sum(count ** 2 for count in token_counts.values())  # Sum of squared frequencies

        if m1 == 0:
            return 0.0

        return 10000 * (m2 - m1) / (m1 ** 2)

    def _calculate_simpsons_diversity(self, token_counts: Counter) -> float:
        """Calculate Simpson's diversity index."""
        if not token_counts:
            return 0.0

        total_tokens = sum(token_counts.values())
        if total_tokens <= 1:
            return 0.0

        # Simpson's index: sum(pi^2) where pi is proportion of token i
        simpson_index = sum((count / total_tokens) ** 2 for count in token_counts.values())

        # Simpson's diversity = 1 - Simpson's index
        return 1 - simpson_index


# Convenience functions
def calculate_diversity_score(answers: List[str],
                            questions: Optional[List[str]] = None) -> float:
    """
    Calculate overall diversity score for answers.

    Args:
        answers: List of generated answers
        questions: Optional corresponding questions

    Returns:
        Overall diversity score (0-1)
    """
    analyzer = DiversityAnalyzer()
    results = analyzer.get_comprehensive_diversity_score(answers, questions)
    return results['overall_diversity_score']


# Example usage
if __name__ == "__main__":
    # Example answers with varying diversity
    answers = [
        "Paris is the capital of France. It is located in Western Europe.",
        "The capital city of France is Paris, situated in Western Europe.",
        "France's capital, Paris, lies in the western part of Europe.",
        "Paris serves as the capital of France in Western Europe.",
        "I don't know the answer to that question.",  # Less diverse/generic
    ]

    questions = [
        "What is the capital of France?",
        "Where is the capital of France located?",
        "What city is the capital of France?",
        "Can you tell me about France's capital?",
        "What is the meaning of life?"
    ]

    # Create analyzer
    analyzer = DiversityAnalyzer()

    # Lexical diversity
    print("Lexical Diversity Analysis:")
    lexical = analyzer.analyze_lexical_diversity(answers)
    print(".4f")
    print(".4f")
    print(".4f")

    # Structural diversity
    print("\nStructural Diversity Analysis:")
    structural = analyzer.analyze_structural_diversity(answers)
    print(".2f")
    print(".2f")

    # Semantic diversity
    print("\nSemantic Diversity Analysis:")
    semantic = analyzer.analyze_semantic_diversity(answers)
    if semantic.get('semantic_diversity_available', False):
        print(".4f")
        print(".4f")
    else:
        print("Semantic diversity analysis not available")

    # Answer variety
    print("\nAnswer Variety Analysis:")
    variety = analyzer.analyze_answer_variety(answers, questions)
    print(".4f")
    print(".4f")

    # Comprehensive diversity score
    print("\nComprehensive Diversity Analysis:")
    comprehensive = analyzer.get_comprehensive_diversity_score(answers, questions)
    print(".4f")

    # Quick diversity score
    quick_score = calculate_diversity_score(answers, questions)
    print(".4f")
