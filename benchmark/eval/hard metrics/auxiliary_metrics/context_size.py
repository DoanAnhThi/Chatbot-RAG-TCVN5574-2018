"""
Context Size Metric for RAG System Evaluation

Context Size measures the amount of contextual information provided to the
language model for answer generation. This includes:
- Token count of retrieved documents
- Character count
- Number of documents retrieved
- Context length distribution

Monitoring context size helps understand:
- Retrieval efficiency (are we retrieving too much/little context?)
- Token usage and cost implications
- Potential for context overflow issues
"""

from typing import List, Union, Dict, Optional, Any
import re
import statistics

try:
    import tiktoken
except ImportError:
    tiktoken = None


class ContextSizeAnalyzer:
    """Analyzes context size metrics for RAG systems."""

    def __init__(self, tokenizer: Optional[str] = "cl100k_base"):
        """
        Initialize context size analyzer.

        Args:
            tokenizer: Tokenizer to use ("cl100k_base" for GPT, "p50k_base" for older GPT, or None for character count)
        """
        self.tokenizer_name = tokenizer
        if tokenizer and tiktoken:
            try:
                self.tokenizer = tiktoken.get_encoding(tokenizer)
            except Exception:
                print(f"Warning: Could not load tokenizer '{tokenizer}', falling back to character count")
                self.tokenizer = None
        else:
            self.tokenizer = None

    def analyze_context(self, contexts: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Analyze context size for a single query or batch of queries.

        Args:
            contexts: Context string or list of context strings/documents

        Returns:
            Dictionary with context size metrics
        """
        if isinstance(contexts, str):
            return self._analyze_single_context(contexts)
        elif isinstance(contexts, list):
            return self._analyze_multiple_contexts(contexts)
        else:
            raise ValueError("contexts must be a string or list of strings")

    def _analyze_single_context(self, context: str) -> Dict[str, Any]:
        """
        Analyze a single context string.

        Args:
            context: Context string

        Returns:
            Context size metrics
        """
        metrics = {
            'total_characters': len(context),
            'total_words': len(context.split()) if context else 0,
            'total_sentences': len(re.split(r'[.!?]+', context)) - 1 if context else 0,
        }

        # Token count
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(context)
                metrics['total_tokens'] = len(tokens)
            except Exception:
                metrics['total_tokens'] = None
        else:
            # Rough estimate: 1 token ≈ 4 characters for English text
            metrics['total_tokens'] = len(context) // 4

        return metrics

    def _analyze_multiple_contexts(self, contexts: List[str]) -> Dict[str, Any]:
        """
        Analyze multiple context strings (e.g., multiple retrieved documents).

        Args:
            contexts: List of context strings

        Returns:
            Aggregated context size metrics
        """
        if not contexts:
            return {
                'num_documents': 0,
                'total_characters': 0,
                'total_words': 0,
                'total_sentences': 0,
                'total_tokens': 0,
                'avg_document_length_chars': 0,
                'avg_document_length_tokens': 0,
                'document_length_distribution': []
            }

        # Analyze each document
        individual_metrics = [self._analyze_single_context(ctx) for ctx in contexts]

        # Aggregate metrics
        metrics = {
            'num_documents': len(contexts),
            'total_characters': sum(m['total_characters'] for m in individual_metrics),
            'total_words': sum(m['total_words'] for m in individual_metrics),
            'total_sentences': sum(m['total_sentences'] for m in individual_metrics),
            'total_tokens': sum(m['total_tokens'] for m in individual_metrics if m['total_tokens'] is not None),
            'avg_document_length_chars': statistics.mean(m['total_characters'] for m in individual_metrics),
            'document_length_distribution_chars': [m['total_characters'] for m in individual_metrics],
        }

        # Token-based metrics
        token_lengths = [m['total_tokens'] for m in individual_metrics if m['total_tokens'] is not None]
        if token_lengths:
            metrics['avg_document_length_tokens'] = statistics.mean(token_lengths)
            metrics['document_length_distribution_tokens'] = token_lengths
            metrics['max_document_length_tokens'] = max(token_lengths)
            metrics['min_document_length_tokens'] = min(token_lengths)
        else:
            metrics['avg_document_length_tokens'] = None
            metrics['document_length_distribution_tokens'] = []
            metrics['max_document_length_tokens'] = None
            metrics['min_document_length_tokens'] = None

        # Character-based distribution stats
        char_lengths = [m['total_characters'] for m in individual_metrics]
        metrics.update({
            'max_document_length_chars': max(char_lengths),
            'min_document_length_chars': min(char_lengths),
            'std_document_length_chars': statistics.stdev(char_lengths) if len(char_lengths) > 1 else 0,
        })

        return metrics

    def analyze_context_efficiency(self, contexts: List[str],
                                 question: str,
                                 answer: str) -> Dict[str, Any]:
        """
        Analyze context efficiency - how much context is actually used.

        Args:
            contexts: List of retrieved context documents
            question: Original question
            answer: Generated answer

        Returns:
            Context efficiency metrics
        """
        base_metrics = self._analyze_multiple_contexts(contexts)

        # Calculate question-answer overlap
        question_tokens = set(self._tokenize_text(question))
        answer_tokens = set(self._tokenize_text(answer))

        # Calculate context utilization
        total_context_tokens = set()
        for context in contexts:
            total_context_tokens.update(self._tokenize_text(context))

        used_tokens = answer_tokens.intersection(total_context_tokens)
        question_tokens_in_context = question_tokens.intersection(total_context_tokens)

        efficiency_metrics = {
            'context_utilization_ratio': len(used_tokens) / len(total_context_tokens) if total_context_tokens else 0,
            'question_coverage_ratio': len(question_tokens_in_context) / len(question_tokens) if question_tokens else 0,
            'answer_context_overlap_ratio': len(used_tokens) / len(answer_tokens) if answer_tokens else 0,
            'unused_context_ratio': 1 - (len(used_tokens) / len(total_context_tokens)) if total_context_tokens else 1,
        }

        # Combine with base metrics
        base_metrics.update(efficiency_metrics)
        return base_metrics

    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization for efficiency analysis."""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()

    def get_context_size_distribution(self, all_contexts: List[List[str]]) -> Dict[str, Any]:
        """
        Analyze context size distribution across multiple queries.

        Args:
            all_contexts: List of context lists (one per query)

        Returns:
            Distribution statistics
        """
        if not all_contexts:
            return {}

        # Collect metrics for each query
        query_metrics = []
        for contexts in all_contexts:
            metrics = self._analyze_multiple_contexts(contexts)
            query_metrics.append(metrics)

        # Aggregate across queries
        distribution = {
            'num_queries': len(all_contexts),
            'avg_contexts_per_query': statistics.mean(m['num_documents'] for m in query_metrics),
            'avg_total_tokens_per_query': statistics.mean(m['total_tokens'] for m in query_metrics if m['total_tokens']),
            'avg_total_chars_per_query': statistics.mean(m['total_characters'] for m in query_metrics),
            'context_size_variance_tokens': statistics.stdev([m['total_tokens'] for m in query_metrics if m['total_tokens']]) if len(query_metrics) > 1 else 0,
            'context_size_variance_chars': statistics.stdev([m['total_characters'] for m in query_metrics]) if len(query_metrics) > 1 else 0,
        }

        # Percentiles for token counts
        token_counts = [m['total_tokens'] for m in query_metrics if m['total_tokens']]
        if token_counts:
            token_counts.sort()
            distribution.update({
                'p25_tokens_per_query': self._percentile(token_counts, 25),
                'p50_tokens_per_query': self._percentile(token_counts, 50),
                'p75_tokens_per_query': self._percentile(token_counts, 75),
                'p95_tokens_per_query': self._percentile(token_counts, 95),
            })

        return distribution

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile from sorted data."""
        if not data:
            return 0.0
        data_sorted = sorted(data)
        k = (len(data_sorted) - 1) * (percentile / 100.0)
        f = int(k)
        c = k - f
        if f + 1 < len(data_sorted):
            return data_sorted[f] * (1 - c) + data_sorted[f + 1] * c
        else:
            return data_sorted[f]

    def check_context_limits(self, contexts: List[str],
                           max_tokens: Optional[int] = None,
                           max_chars: Optional[int] = None) -> Dict[str, Any]:
        """
        Check if context exceeds specified limits.

        Args:
            contexts: List of context documents
            max_tokens: Maximum allowed tokens
            max_chars: Maximum allowed characters

        Returns:
            Limit check results
        """
        metrics = self._analyze_multiple_contexts(contexts)

        results = {
            'within_token_limit': True,
            'within_char_limit': True,
            'token_usage_percent': 0.0,
            'char_usage_percent': 0.0,
        }

        if max_tokens and metrics['total_tokens']:
            results['token_usage_percent'] = (metrics['total_tokens'] / max_tokens) * 100
            results['within_token_limit'] = metrics['total_tokens'] <= max_tokens

        if max_chars:
            results['char_usage_percent'] = (metrics['total_characters'] / max_chars) * 100
            results['within_char_limit'] = metrics['total_characters'] <= max_chars

        return results


# Convenience functions
def get_context_size_summary(contexts: Union[str, List[str]],
                           tokenizer: Optional[str] = "cl100k_base") -> Dict[str, Any]:
    """
    Get a summary of context size metrics.

    Args:
        contexts: Context string(s) to analyze
        tokenizer: Tokenizer to use

    Returns:
        Context size summary
    """
    analyzer = ContextSizeAnalyzer(tokenizer)
    return analyzer.analyze_context(contexts)


def check_context_overflow(contexts: List[str],
                         max_context_length: int = 4096) -> bool:
    """
    Check if context exceeds typical model limits.

    Args:
        contexts: List of context documents
        max_context_length: Maximum context length in tokens

    Returns:
        True if context exceeds limit
    """
    analyzer = ContextSizeAnalyzer()
    metrics = analyzer.analyze_context(contexts)
    return metrics.get('total_tokens', 0) > max_context_length


# Example usage
if __name__ == "__main__":
    # Example contexts
    contexts_single = [
        "Paris is the capital of France. It is located in Western Europe and has a population of over 2 million people.",
        "France is a unitary semi-presidential republic with its capital in Paris.",
        "The city of Paris is known for landmarks like the Eiffel Tower and Louvre Museum."
    ]

    contexts_multiple_queries = [
        contexts_single,  # Query 1
        ["Machine learning is a subset of artificial intelligence."],  # Query 2
        ["Photosynthesis is the process by which plants make food using sunlight.", "Plants convert CO2 and water into glucose and oxygen."]  # Query 3
    ]

    # Analyze single context set
    print("Single context analysis:")
    analyzer = ContextSizeAnalyzer()
    metrics = analyzer.analyze_context(contexts_single)
    print(f"Number of documents: {metrics['num_documents']}")
    print(f"Total tokens: {metrics['total_tokens']}")
    print(f"Total characters: {metrics['total_characters']}")
    print(".1f")

    # Analyze multiple queries
    print("\nMultiple queries analysis:")
    dist_metrics = analyzer.get_context_size_distribution(contexts_multiple_queries)
    print(f"Number of queries: {dist_metrics['num_queries']}")
    print(".1f")
    print(".1f")

    # Context efficiency analysis
    print("\nContext efficiency analysis:")
    question = "What is the capital of France?"
    answer = "The capital of France is Paris."
    efficiency = analyzer.analyze_context_efficiency(contexts_single, question, answer)
    print(".3f")
    print(".3f")

    # Check limits
    print("\nLimit checking:")
    limits = analyzer.check_context_limits(contexts_single, max_tokens=1000, max_chars=5000)
    print(f"Within token limit: {limits['within_token_limit']}")
    print(".1f")
