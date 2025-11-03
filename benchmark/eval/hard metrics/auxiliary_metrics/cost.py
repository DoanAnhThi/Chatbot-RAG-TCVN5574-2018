"""
Cost Metric for RAG System Evaluation

Cost analysis measures the economic efficiency of RAG systems, including:
- API costs (OpenAI, Anthropic, etc.)
- Computational costs (GPU/CPU usage)
- Token-based pricing
- Cost per query
- Cost-benefit analysis

This helps monitor expenses and optimize for cost-effectiveness.
"""

from typing import Dict, List, Union, Optional, Any
import time
from dataclasses import dataclass
from datetime import datetime


@dataclass
class APICostConfig:
    """Configuration for API pricing."""
    model_name: str
    input_token_cost: float  # Cost per 1K input tokens
    output_token_cost: float  # Cost per 1K output tokens
    currency: str = "USD"


class CostTracker:
    """Tracks costs for RAG system operations."""

    def __init__(self):
        # Common API pricing (as of 2024, subject to change)
        self.api_configs = {
            "gpt-4": APICostConfig("gpt-4", 0.03, 0.06, "USD"),
            "gpt-4-turbo": APICostConfig("gpt-4-turbo", 0.01, 0.03, "USD"),
            "gpt-3.5-turbo": APICostConfig("gpt-3.5-turbo", 0.0015, 0.002, "USD"),
            "claude-3-opus": APICostConfig("claude-3-opus", 0.015, 0.075, "USD"),
            "claude-3-sonnet": APICostConfig("claude-3-sonnet", 0.003, 0.015, "USD"),
            "claude-3-haiku": APICostConfig("claude-3-haiku", 0.00025, 0.00125, "USD"),
        }

        self.usage_records: List[Dict[str, Any]] = []
        self.computational_costs: Dict[str, float] = {}

    def record_api_usage(self, model_name: str,
                        input_tokens: int,
                        output_tokens: int,
                        operation_type: str = "generation",
                        timestamp: Optional[datetime] = None) -> float:
        """
        Record API usage and calculate cost.

        Args:
            model_name: Name of the model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            operation_type: Type of operation (generation, embedding, etc.)
            timestamp: Timestamp of usage

        Returns:
            Cost in USD for this operation
        """
        if timestamp is None:
            timestamp = datetime.now()

        config = self.api_configs.get(model_name)
        if not config:
            print(f"Warning: No pricing config for model '{model_name}', using default")
            config = APICostConfig(model_name, 0.01, 0.02, "USD")  # Default pricing

        # Calculate cost
        input_cost = (input_tokens / 1000) * config.input_token_cost
        output_cost = (output_tokens / 1000) * config.output_token_cost
        total_cost = input_cost + output_cost

        # Record usage
        record = {
            'timestamp': timestamp,
            'model_name': model_name,
            'operation_type': operation_type,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost,
            'currency': config.currency
        }

        self.usage_records.append(record)
        return total_cost

    def add_custom_model_pricing(self, model_name: str,
                               input_token_cost: float,
                               output_token_cost: float,
                               currency: str = "USD") -> None:
        """
        Add custom pricing for a model.

        Args:
            model_name: Name of the model
            input_token_cost: Cost per 1K input tokens
            output_token_cost: Cost per 1K output tokens
            currency: Currency for pricing
        """
        self.api_configs[model_name] = APICostConfig(
            model_name, input_token_cost, output_token_cost, currency
        )

    def record_computational_cost(self, resource_type: str,
                                usage_amount: float,
                                cost_per_unit: float,
                                operation: str = "inference") -> float:
        """
        Record computational resource costs.

        Args:
            resource_type: Type of resource (GPU-hours, CPU-hours, etc.)
            usage_amount: Amount of resource used
            cost_per_unit: Cost per unit of resource
            operation: Type of operation

        Returns:
            Cost for this computational usage
        """
        cost = usage_amount * cost_per_unit

        key = f"{operation}_{resource_type}"
        if key not in self.computational_costs:
            self.computational_costs[key] = 0.0
        self.computational_costs[key] += cost

        return cost

    def get_cost_summary(self, start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get comprehensive cost summary.

        Args:
            start_date: Start date for filtering
            end_date: End date for filtering

        Returns:
            Cost summary statistics
        """
        # Filter records by date
        filtered_records = self.usage_records
        if start_date or end_date:
            filtered_records = [
                record for record in self.usage_records
                if (not start_date or record['timestamp'] >= start_date) and
                   (not end_date or record['timestamp'] <= end_date)
            ]

        if not filtered_records:
            return {
                'total_api_cost': 0.0,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_operations': 0,
                'computational_costs': self.computational_costs.copy(),
                'cost_per_operation': 0.0,
                'cost_per_token': 0.0
            }

        # Calculate API costs
        total_api_cost = sum(record['total_cost'] for record in filtered_records)
        total_input_tokens = sum(record['input_tokens'] for record in filtered_records)
        total_output_tokens = sum(record['output_tokens'] for record in filtered_records)
        total_tokens = total_input_tokens + total_output_tokens
        total_operations = len(filtered_records)

        # Cost per operation/token
        cost_per_operation = total_api_cost / total_operations if total_operations > 0 else 0.0
        cost_per_token = total_api_cost / total_tokens if total_tokens > 0 else 0.0

        # Model breakdown
        model_costs = {}
        for record in filtered_records:
            model = record['model_name']
            if model not in model_costs:
                model_costs[model] = {'cost': 0.0, 'operations': 0, 'tokens': 0}
            model_costs[model]['cost'] += record['total_cost']
            model_costs[model]['operations'] += 1
            model_costs[model]['tokens'] += record['input_tokens'] + record['output_tokens']

        summary = {
            'total_api_cost': total_api_cost,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tokens': total_tokens,
            'total_operations': total_operations,
            'computational_costs': self.computational_costs.copy(),
            'cost_per_operation': cost_per_operation,
            'cost_per_token': cost_per_token,
            'model_breakdown': model_costs,
            'currency': 'USD'  # Assuming USD for simplicity
        }

        return summary

    def estimate_cost_for_operation(self, model_name: str,
                                  estimated_input_tokens: int,
                                  estimated_output_tokens: int) -> float:
        """
        Estimate cost for a hypothetical operation.

        Args:
            model_name: Name of the model
            estimated_input_tokens: Expected input tokens
            estimated_output_tokens: Expected output tokens

        Returns:
            Estimated cost
        """
        config = self.api_configs.get(model_name)
        if not config:
            print(f"Warning: No pricing for model '{model_name}'")
            return 0.0

        input_cost = (estimated_input_tokens / 1000) * config.input_token_cost
        output_cost = (estimated_output_tokens / 1000) * config.output_token_cost

        return input_cost + output_cost

    def get_cost_efficiency_metrics(self, quality_scores: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Calculate cost-efficiency metrics.

        Args:
            quality_scores: Optional quality scores for cost-benefit analysis

        Returns:
            Cost-efficiency metrics
        """
        summary = self.get_cost_summary()

        metrics = {
            'cost_per_token': summary['cost_per_token'],
            'cost_per_operation': summary['cost_per_operation'],
            'total_cost': summary['total_api_cost'] + sum(self.computational_costs.values()),
        }

        # Cost-benefit analysis if quality scores provided
        if quality_scores and len(quality_scores) == summary['total_operations']:
            avg_quality = sum(quality_scores) / len(quality_scores)
            metrics['cost_per_quality_point'] = summary['total_api_cost'] / (avg_quality * len(quality_scores))
            metrics['quality_adjusted_efficiency'] = avg_quality / summary['cost_per_operation']

        return metrics

    def reset(self) -> None:
        """Reset all cost tracking data."""
        self.usage_records.clear()
        self.computational_costs.clear()


def calculate_cost_savings(current_cost: float,
                          baseline_cost: float) -> Dict[str, float]:
    """
    Calculate cost savings metrics.

    Args:
        current_cost: Current cost
        baseline_cost: Baseline cost to compare against

    Returns:
        Cost savings metrics
    """
    if baseline_cost == 0:
        return {'savings_absolute': 0.0, 'savings_percentage': 0.0}

    savings_absolute = baseline_cost - current_cost
    savings_percentage = (savings_absolute / baseline_cost) * 100

    return {
        'savings_absolute': savings_absolute,
        'savings_percentage': savings_percentage,
        'current_cost': current_cost,
        'baseline_cost': baseline_cost
    }


# Example usage
if __name__ == "__main__":
    # Create cost tracker
    tracker = CostTracker()

    # Record some API usage
    print("Recording API usage examples:")

    # GPT-4 generation
    cost1 = tracker.record_api_usage(
        model_name="gpt-4",
        input_tokens=500,
        output_tokens=200,
        operation_type="generation"
    )
    print(".4f")

    # GPT-3.5-turbo generation
    cost2 = tracker.record_api_usage(
        model_name="gpt-3.5-turbo",
        input_tokens=300,
        output_tokens=150,
        operation_type="generation"
    )
    print(".4f")

    # Add custom model pricing
    tracker.add_custom_model_pricing("custom-model", 0.005, 0.015)

    cost3 = tracker.record_api_usage(
        model_name="custom-model",
        input_tokens=400,
        output_tokens=100,
        operation_type="generation"
    )
    print(".4f")

    # Record computational costs
    gpu_cost = tracker.record_computational_cost(
        resource_type="GPU-hours",
        usage_amount=2.5,
        cost_per_unit=1.5,  # $1.50 per GPU-hour
        operation="inference"
    )
    print(".2f")

    # Get cost summary
    print("\nCost Summary:")
    summary = tracker.get_cost_summary()
    print(".4f")
    print(f"Total tokens used: {summary['total_tokens']}")
    print(f"Total operations: {summary['total_operations']}")
    print(".6f")
    print(".6f")

    print("\nModel breakdown:")
    for model, stats in summary['model_breakdown'].items():
        print(f"  {model}: ${stats['cost']:.4f} ({stats['operations']} ops, {stats['tokens']} tokens)")

    # Cost estimation
    print("\nCost estimation for hypothetical operation:")
    estimated_cost = tracker.estimate_cost_for_operation("gpt-4", 600, 250)
    print(".4f")

    # Cost savings calculation
    print("\nCost savings analysis:")
    current_total = summary['total_api_cost']
    baseline_total = current_total * 1.5  # Assume 50% more expensive baseline
    savings = calculate_cost_savings(current_total, baseline_total)
    print(".4f")
    print(".1f")

    # Cost efficiency with quality scores
    quality_scores = [0.8, 0.9, 0.7]  # Hypothetical quality scores
    efficiency = tracker.get_cost_efficiency_metrics(quality_scores)
    print("\nCost-efficiency metrics:")
    print(".4f")
    print(".4f")
