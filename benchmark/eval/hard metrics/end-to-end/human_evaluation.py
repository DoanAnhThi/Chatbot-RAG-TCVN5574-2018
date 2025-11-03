"""
Human Evaluation Framework for RAG End-to-End Evaluation

Human Evaluation provides the gold standard for assessing RAG system quality.
This framework provides tools and templates for collecting and analyzing
human judgments on various quality dimensions.

Key criteria typically evaluated:
- Correctness/Factuality
- Relevance
- Completeness
- Clarity/Coherence
- Readability
- Informativeness

This module provides tools for:
- Creating evaluation templates
- Collecting human ratings
- Analyzing inter-rater agreement
- Aggregating results
"""

from typing import List, Dict, Union, Optional, Any
import json
import csv
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime
import os


@dataclass
class HumanEvaluationSample:
    """Data structure for a single evaluation sample."""
    sample_id: str
    question: str
    answer: str
    context: Optional[str] = None
    ground_truth: Optional[str] = None
    ratings: Dict[str, Any] = None

    def __post_init__(self):
        if self.ratings is None:
            self.ratings = {}


@dataclass
class EvaluationCriteria:
    """Definition of evaluation criteria."""
    name: str
    description: str
    scale_type: str  # "likert", "binary", "continuous"
    scale_range: tuple  # (min, max)
    labels: Optional[Dict[int, str]] = None  # For likert scales


class HumanEvaluationFramework:
    """Framework for managing human evaluations of RAG systems."""

    def __init__(self):
        self.samples: List[HumanEvaluationSample] = []
        self.criteria: Dict[str, EvaluationCriteria] = self._get_default_criteria()

    def _get_default_criteria(self) -> Dict[str, EvaluationCriteria]:
        """Get default evaluation criteria."""
        return {
            "correctness": EvaluationCriteria(
                name="correctness",
                description="How factually accurate is the answer?",
                scale_type="likert",
                scale_range=(1, 5),
                labels={1: "Completely incorrect", 2: "Mostly incorrect", 3: "Partially correct",
                       4: "Mostly correct", 5: "Completely correct"}
            ),
            "relevance": EvaluationCriteria(
                name="relevance",
                description="How relevant is the answer to the question?",
                scale_type="likert",
                scale_range=(1, 5),
                labels={1: "Not relevant", 2: "Slightly relevant", 3: "Moderately relevant",
                       4: "Very relevant", 5: "Perfectly relevant"}
            ),
            "completeness": EvaluationCriteria(
                name="completeness",
                description="How complete is the answer?",
                scale_type="likert",
                scale_range=(1, 5),
                labels={1: "Very incomplete", 2: "Incomplete", 3: "Somewhat complete",
                       4: "Mostly complete", 5: "Completely comprehensive"}
            ),
            "clarity": EvaluationCriteria(
                name="clarity",
                description="How clear and coherent is the answer?",
                scale_type="likert",
                scale_range=(1, 5),
                labels={1: "Very confusing", 2: "Somewhat confusing", 3: "Neutral",
                       4: "Clear", 5: "Very clear and coherent"}
            ),
            "naturalness": EvaluationCriteria(
                name="naturalness",
                description="How natural and human-like is the answer?",
                scale_type="likert",
                scale_range=(1, 5),
                labels={1: "Very unnatural", 2: "Somewhat unnatural", 3: "Neutral",
                       4: "Natural", 5: "Very natural and fluent"}
            )
        }

    def add_sample(self, question: str, answer: str,
                  context: Optional[str] = None,
                  ground_truth: Optional[str] = None,
                  sample_id: Optional[str] = None) -> str:
        """
        Add a sample for evaluation.

        Args:
            question: The input question
            answer: The generated answer
            context: Optional context used for generation
            ground_truth: Optional ground truth answer
            sample_id: Optional unique identifier

        Returns:
            Sample ID
        """
        if sample_id is None:
            sample_id = f"sample_{len(self.samples) + 1}"

        sample = HumanEvaluationSample(
            sample_id=sample_id,
            question=question,
            answer=answer,
            context=context,
            ground_truth=ground_truth
        )

        self.samples.append(sample)
        return sample_id

    def add_rating(self, sample_id: str, rater_id: str,
                  ratings: Dict[str, Union[int, float]],
                  comments: Optional[str] = None) -> None:
        """
        Add a rating for a sample.

        Args:
            sample_id: ID of the sample to rate
            rater_id: ID of the human rater
            ratings: Dictionary of criterion -> score
            comments: Optional comments from the rater
        """
        sample = self._get_sample_by_id(sample_id)
        if sample is None:
            raise ValueError(f"Sample with ID {sample_id} not found")

        if rater_id not in sample.ratings:
            sample.ratings[rater_id] = {}

        sample.ratings[rater_id].update(ratings)

        if comments:
            sample.ratings[rater_id]["comments"] = comments

    def _get_sample_by_id(self, sample_id: str) -> Optional[HumanEvaluationSample]:
        """Get sample by ID."""
        for sample in self.samples:
            if sample.sample_id == sample_id:
                return sample
        return None

    def get_evaluation_template(self, sample_id: str) -> str:
        """
        Generate an evaluation template for a sample.

        Args:
            sample_id: ID of the sample

        Returns:
            Formatted evaluation template
        """
        sample = self._get_sample_by_id(sample_id)
        if sample is None:
            raise ValueError(f"Sample with ID {sample_id} not found")

        template = f"""
EVALUATION TEMPLATE
===================

Sample ID: {sample.sample_id}

QUESTION:
{sample.question}

GENERATED ANSWER:
{sample.answer}

"""

        if sample.context:
            template += f"""
CONTEXT PROVIDED:
{sample.context}

"""

        if sample.ground_truth:
            template += f"""
GROUND TRUTH ANSWER:
{sample.ground_truth}

"""

        template += """
EVALUATION CRITERIA:
"""

        for criterion_name, criterion in self.criteria.items():
            template += f"""
{criterion_name.upper()} ({criterion.scale_range[0]}-{criterion.scale_range[1]})
{criterion.description}
"""
            if criterion.labels:
                for score, label in criterion.labels.items():
                    template += f"  {score}: {label}\n"

            template += "Your rating: ____\n"

        template += """
COMMENTS (optional):
____________________________________________________________
____________________________________________________________
____________________________________________________________

RATER ID: ____________________
DATE: ____________________
"""

        return template

    def calculate_average_scores(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate average scores across all samples and raters.

        Returns:
            Dictionary of sample_id -> {criterion: average_score}
        """
        results = {}

        for sample in self.samples:
            sample_results = {}

            # Collect all ratings for this sample
            all_ratings = {}
            for rater_ratings in sample.ratings.values():
                for criterion, score in rater_ratings.items():
                    if criterion != "comments" and isinstance(score, (int, float)):
                        if criterion not in all_ratings:
                            all_ratings[criterion] = []
                        all_ratings[criterion].append(score)

            # Calculate averages
            for criterion, scores in all_ratings.items():
                if scores:
                    sample_results[criterion] = statistics.mean(scores)

            results[sample.sample_id] = sample_results

        return results

    def calculate_inter_rater_agreement(self, criterion: str) -> Dict[str, float]:
        """
        Calculate inter-rater agreement statistics for a criterion.

        Args:
            criterion: Name of the criterion to analyze

        Returns:
            Dictionary with agreement statistics
        """
        agreements = {}

        for sample in self.samples:
            ratings = []
            for rater_ratings in sample.ratings.values():
                if criterion in rater_ratings:
                    rating = rater_ratings[criterion]
                    if isinstance(rating, (int, float)):
                        ratings.append(rating)

            if len(ratings) >= 2:
                # Calculate standard deviation as measure of agreement
                if len(ratings) > 1:
                    std_dev = statistics.stdev(ratings)
                    # Lower std_dev = higher agreement
                    agreement_score = 1.0 / (1.0 + std_dev)  # Normalize to 0-1
                    agreements[sample.sample_id] = agreement_score

        if agreements:
            avg_agreement = statistics.mean(agreements.values())
            return {
                "average_agreement": avg_agreement,
                "per_sample_agreement": agreements
            }

        return {"average_agreement": 0.0, "per_sample_agreement": {}}

    def export_to_csv(self, filename: str) -> None:
        """
        Export evaluation results to CSV.

        Args:
            filename: Output filename
        """
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['sample_id', 'question', 'answer', 'context', 'ground_truth']
            fieldnames.extend(self.criteria.keys())
            fieldnames.append('rater_id')
            fieldnames.append('comments')

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for sample in self.samples:
                for rater_id, ratings in sample.ratings.items():
                    row = {
                        'sample_id': sample.sample_id,
                        'question': sample.question,
                        'answer': sample.answer,
                        'context': sample.context or '',
                        'ground_truth': sample.ground_truth or '',
                        'rater_id': rater_id,
                        'comments': ratings.get('comments', '')
                    }

                    # Add criterion ratings
                    for criterion in self.criteria.keys():
                        row[criterion] = ratings.get(criterion, '')

                    writer.writerow(row)

    def save_evaluation_data(self, filename: str) -> None:
        """
        Save the complete evaluation data to JSON.

        Args:
            filename: Output filename
        """
        data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'num_samples': len(self.samples),
                'criteria': {name: asdict(criterion) for name, criterion in self.criteria.items()}
            },
            'samples': [asdict(sample) for sample in self.samples]
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_evaluation_data(self, filename: str) -> None:
        """
        Load evaluation data from JSON.

        Args:
            filename: Input filename
        """
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Restore criteria
        self.criteria = {}
        for name, criterion_data in data['metadata']['criteria'].items():
            self.criteria[name] = EvaluationCriteria(**criterion_data)

        # Restore samples
        self.samples = []
        for sample_data in data['samples']:
            sample = HumanEvaluationSample(**sample_data)
            self.samples.append(sample)


# Standalone functions for convenience
def create_evaluation_template(framework: HumanEvaluationFramework,
                             sample_ids: List[str],
                             output_file: Optional[str] = None) -> str:
    """
    Create evaluation templates for multiple samples.

    Args:
        framework: HumanEvaluationFramework instance
        sample_ids: List of sample IDs to include
        output_file: Optional file to save templates

    Returns:
        Combined evaluation templates
    """
    templates = []

    for sample_id in sample_ids:
        template = framework.get_evaluation_template(sample_id)
        templates.append(template)

    combined = "\n\n" + "="*80 + "\n\n".join(templates)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(combined)

    return combined


# Example usage
if __name__ == "__main__":
    # Create evaluation framework
    framework = HumanEvaluationFramework()

    # Add samples
    sample1_id = framework.add_sample(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        context="France is a country in Europe with Paris as its capital.",
        ground_truth="Paris"
    )

    sample2_id = framework.add_sample(
        question="How does photosynthesis work?",
        answer="Plants use sunlight to convert carbon dioxide and water into glucose.",
        ground_truth="Photosynthesis is the process by which plants use sunlight, carbon dioxide, and water to produce glucose and oxygen."
    )

    # Simulate adding ratings from multiple raters
    framework.add_rating(sample1_id, "rater1",
                        {"correctness": 5, "relevance": 5, "completeness": 4, "clarity": 5, "naturalness": 5},
                        "Excellent answer")

    framework.add_rating(sample1_id, "rater2",
                        {"correctness": 4, "relevance": 5, "completeness": 4, "clarity": 4, "naturalness": 5})

    framework.add_rating(sample2_id, "rater1",
                        {"correctness": 4, "relevance": 4, "completeness": 3, "clarity": 4, "naturalness": 4})

    # Calculate average scores
    avg_scores = framework.calculate_average_scores()
    print("Average scores per sample:")
    for sample_id, scores in avg_scores.items():
        print(f"  {sample_id}: {scores}")

    # Calculate inter-rater agreement
    agreement = framework.calculate_inter_rater_agreement("correctness")
    print(".4f")

    # Generate evaluation template
    template = framework.get_evaluation_template(sample1_id)
    print("\nFirst 500 characters of evaluation template:")
    print(template[:500] + "...")

    # Export to CSV
    framework.export_to_csv("human_evaluation_results.csv")
    print("\nExported results to human_evaluation_results.csv")

    # Save complete data
    framework.save_evaluation_data("evaluation_data.json")
    print("Saved complete evaluation data to evaluation_data.json")
