"""
BERTScore Metric for RAG Generation Evaluation

BERTScore uses contextual embeddings from BERT models to measure semantic
similarity between generated answers and reference answers. It computes
cosine similarity between token embeddings and provides precision, recall,
and F1 scores.

Formula: BERTScore = cosine_similarity(embeddings_prediction, embeddings_reference)

Range: 0 to 1, where 1 indicates identical semantic meaning.
"""

try:
    import torch
    from transformers import BertTokenizer, BertModel
    import numpy as np
    from scipy.spatial.distance import cosine
except ImportError:
    print("Warning: transformers and torch libraries not found.")
    print("Install with: pip install transformers torch scipy")
    torch = None

from typing import List, Union, Optional


class BERTScore:
    """BERTScore calculator using pre-trained BERT model."""

    def __init__(self, model_name: str = 'bert-base-uncased'):
        """
        Initialize BERTScore with specified model.

        Args:
            model_name: Name of the BERT model to use
        """
        if torch is None:
            raise ImportError("transformers and torch are required. Install with: pip install transformers torch scipy")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def score(self, predictions: Union[str, List[str]],
             references: Union[str, List[str]],
             return_precision_recall: bool = False) -> Union[float, dict, List[float], List[dict]]:
        """
        Calculate BERTScore between predictions and references.

        Args:
            predictions: Generated answers (string or list of strings)
            references: Ground truth answers (string or list of strings)
            return_precision_recall: Whether to return precision/recall/F1 or just F1

        Returns:
            BERTScore F1 (or dict with P/R/F1 if return_precision_recall=True)
        """
        # Handle single string inputs
        if isinstance(predictions, str) and isinstance(references, str):
            result = self._calculate_bertscore_single(predictions, references)
            if return_precision_recall:
                return result
            else:
                return result['f1']

        # Handle list inputs
        elif isinstance(predictions, list) and isinstance(references, list):
            if len(predictions) != len(references):
                raise ValueError("predictions and references must have the same length")

            results = []
            for pred, ref in zip(predictions, references):
                result = self._calculate_bertscore_single(pred, ref)
                if return_precision_recall:
                    results.append(result)
                else:
                    results.append(result['f1'])

            return results

        else:
            raise ValueError("predictions and references must both be strings or both be lists")

    def _calculate_bertscore_single(self, prediction: str, reference: str) -> dict:
        """
        Calculate BERTScore for a single prediction-reference pair.

        Args:
            prediction: Generated answer
            reference: Ground truth answer

        Returns:
            Dictionary with precision, recall, and f1 scores
        """
        # Get embeddings
        pred_embeddings = self._get_embeddings(prediction)
        ref_embeddings = self._get_embeddings(reference)

        if pred_embeddings.size == 0 or ref_embeddings.size == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        # Calculate similarity matrix
        pred_embeddings = pred_embeddings.reshape(-1, 768)  # BERT base hidden size
        ref_embeddings = ref_embeddings.reshape(-1, 768)

        # Cosine similarity for each prediction token to all reference tokens
        pred_to_ref_sim = np.zeros((pred_embeddings.shape[0], ref_embeddings.shape[0]))
        for i, pred_emb in enumerate(pred_embeddings):
            for j, ref_emb in enumerate(ref_embeddings):
                pred_to_ref_sim[i, j] = 1 - cosine(pred_emb, ref_emb)

        # Cosine similarity for each reference token to all prediction tokens
        ref_to_pred_sim = pred_to_ref_sim.T

        # Calculate precision: average of max similarities for each pred token
        precision = np.mean(np.max(pred_to_ref_sim, axis=1))

        # Calculate recall: average of max similarities for each ref token
        recall = np.mean(np.max(ref_to_pred_sim, axis=1))

        # Calculate F1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }

    def _get_embeddings(self, text: str) -> np.ndarray:
        """
        Get BERT embeddings for text.

        Args:
            text: Input text

        Returns:
            Token embeddings as numpy array
        """
        # Tokenize
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use last hidden states (shape: batch_size, seq_len, hidden_size)
            embeddings = outputs.last_hidden_state.squeeze(0)  # Remove batch dimension

            # Exclude [CLS] and [SEP] tokens, keep only content tokens
            embeddings = embeddings[1:-1]  # Remove first ([CLS]) and last ([SEP]) tokens

        return embeddings.cpu().numpy()


def bertscore_f1(predictions: Union[str, List[str]],
                references: Union[str, List[str]],
                model_name: str = 'bert-base-uncased') -> Union[float, List[float]]:
    """
    Calculate BERTScore F1 score.

    Args:
        predictions: Generated answers
        references: Ground truth answers
        model_name: BERT model to use

    Returns:
        F1 score(s)
    """
    scorer = BERTScore(model_name)
    return scorer.score(predictions, references, return_precision_recall=False)


def bertscore_precision(predictions: Union[str, List[str]],
                       references: Union[str, List[str]],
                       model_name: str = 'bert-base-uncased') -> Union[float, List[float]]:
    """
    Calculate BERTScore precision.

    Args:
        predictions: Generated answers
        references: Ground truth answers
        model_name: BERT model to use

    Returns:
        Precision score(s)
    """
    scorer = BERTScore(model_name)
    results = scorer.score(predictions, references, return_precision_recall=True)
    if isinstance(results, dict):
        return results['precision']
    else:
        return [r['precision'] for r in results]


def bertscore_recall(predictions: Union[str, List[str]],
                    references: Union[str, List[str]],
                    model_name: str = 'bert-base-uncased') -> Union[float, List[float]]:
    """
    Calculate BERTScore recall.

    Args:
        predictions: Generated answers
        references: Ground truth answers
        model_name: BERT model to use

    Returns:
        Recall score(s)
    """
    scorer = BERTScore(model_name)
    results = scorer.score(predictions, references, return_precision_recall=True)
    if isinstance(results, dict):
        return results['recall']
    else:
        return [r['recall'] for r in results]


# Example usage
if __name__ == "__main__":
    predictions = [
        "The capital of France is Paris.",
        "Machine learning is a subset of artificial intelligence.",
        "The Earth orbits around the Sun."
    ]

    references = [
        "Paris is the capital of France.",
        "Machine learning is part of AI.",
        "The Earth revolves around the Sun."
    ]

    try:
        # Calculate BERTScore
        f1_scores = bertscore_f1(predictions, references)
        print("BERTScore F1:", f1_scores)
        print(".4f")

        # Get detailed scores for first example
        scorer = BERTScore()
        detailed = scorer.score(predictions[0], references[0], return_precision_recall=True)
        print(f"First example - Precision: {detailed['precision']:.4f}, "
              f"Recall: {detailed['recall']:.4f}, F1: {detailed['f1']:.4f}")

    except ImportError as e:
        print(f"Cannot run BERTScore: {e}")
        print("This metric requires heavy dependencies (transformers, torch).")
        print("Consider using lighter alternatives like cosine similarity with pre-computed embeddings.")
