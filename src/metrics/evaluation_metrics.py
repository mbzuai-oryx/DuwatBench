#!/usr/bin/env python3
"""
Evaluation metrics for DuwatBench
Implements all 5 metrics from paper Section 3.1:
- CER (Character Error Rate)
- WER (Word Error Rate)
- chrF (Character F-score)
- ExactMatch
- NLD (Normalized Levenshtein Distance)
"""

import numpy as np
from typing import List, Tuple
import Levenshtein


def calculate_cer(prediction: str, reference: str, cap_at_one: bool = True) -> float:
    """
    Calculate Character Error Rate (CER)

    CER = edit_distance(pred_chars, ref_chars) / len(ref_chars)

    From paper (lines 220-223):
    "CER captures the minimum number of character edits required to match
    the reference and is particularly important for Arabic calligraphy,
    where diacritics and ligatures can alter meaning."

    Args:
        prediction: Model prediction text
        reference: Ground truth text
        cap_at_one: If True, cap CER at 1.0 for easier interpretation
                   (CER > 1.0 means prediction has more errors than reference length)
    """
    if not reference:
        return 1.0 if prediction else 0.0

    distance = Levenshtein.distance(prediction, reference)
    cer = distance / len(reference)

    # Optionally cap at 1.0 for easier interpretation
    if cap_at_one:
        cer = min(cer, 1.0)

    return cer


def calculate_wer(prediction: str, reference: str, cap_at_one: bool = True) -> float:
    """
    Calculate Word Error Rate (WER)

    WER = word_edit_distance(pred_words, ref_words) / len(ref_words)

    From paper (lines 224-225):
    "WER evaluates accuracy at the word level, offering a more
    interpretable measure for end-users."

    Proper word-level edit distance:
    Computes minimum edits (insertions, deletions, substitutions) at word level.

    Args:
        prediction: Model prediction text
        reference: Ground truth text
        cap_at_one: If True, cap WER at 1.0 for easier interpretation
    """
    pred_words = prediction.split()
    ref_words = reference.split()

    if not ref_words:
        return 1.0 if pred_words else 0.0

    # Use dynamic programming to compute word-level edit distance
    # Create DP matrix
    m, n = len(pred_words), len(ref_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill DP matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_words[i-1] == ref_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )

    word_distance = dp[m][n]
    wer = word_distance / len(ref_words)

    # Optionally cap at 1.0 for easier interpretation
    if cap_at_one:
        wer = min(wer, 1.0)

    return wer


def calculate_chrf(prediction: str, reference: str, n: int = 6, beta: float = 2.0) -> float:
    """
    Calculate character F-score (chrF)

    From paper (lines 226-229):
    "chrF computes character n-gram F-scores, rewarding partial overlaps
    and providing robustness to tokenization errors and spelling variants
    common in stylized text."

    Args:
        prediction: Model output
        reference: Ground truth
        n: Maximum n-gram length (default 6)
        beta: F-score beta parameter (default 2.0 for chrF2)
    """
    if not reference and not prediction:
        return 100.0
    if not reference or not prediction:
        return 0.0

    def get_ngrams(text: str, n: int) -> dict:
        """Extract character n-grams"""
        ngrams = {}
        for i in range(1, n + 1):
            for j in range(len(text) - i + 1):
                ngram = text[j:j + i]
                ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams

    pred_ngrams = get_ngrams(prediction, n)
    ref_ngrams = get_ngrams(reference, n)

    # Calculate matches
    matches = 0
    for ngram, count in pred_ngrams.items():
        if ngram in ref_ngrams:
            matches += min(count, ref_ngrams[ngram])

    # Calculate precision and recall
    total_pred = sum(pred_ngrams.values())
    total_ref = sum(ref_ngrams.values())

    if total_pred == 0 or total_ref == 0:
        return 0.0

    precision = matches / total_pred if total_pred > 0 else 0.0
    recall = matches / total_ref if total_ref > 0 else 0.0

    # Calculate F-score
    if precision + recall == 0:
        return 0.0

    beta_squared = beta ** 2
    chrf = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)

    return chrf * 100.0  # Return as percentage


def calculate_exact_match(prediction: str, reference: str) -> float:
    """
    Calculate Exact Match accuracy

    From paper (lines 229-231):
    "ExactMatch is the strictest metric, counting only perfect matches
    between prediction and reference."

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    return 1.0 if prediction == reference else 0.0


def calculate_nld(prediction: str, reference: str) -> float:
    """
    Calculate Normalized Levenshtein Distance (NLD)

    From paper (lines 231-235):
    "NLD normalizes edit distance by sequence length, situating itself
    between CER and WER in granularity and offering a balanced perspective
    on recognition errors across variable word sizes."

    NLD = edit_distance(pred, ref) / max(len(pred), len(ref))
    """
    if not prediction and not reference:
        return 0.0

    max_len = max(len(prediction), len(reference))
    if max_len == 0:
        return 0.0

    distance = Levenshtein.distance(prediction, reference)
    nld = distance / max_len

    return nld


class MetricsCalculator:
    """
    Unified metrics calculator for DuwatBench evaluation
    """

    def __init__(self):
        self.metrics = {
            'CER': calculate_cer,
            'WER': calculate_wer,
            'chrF': calculate_chrf,
            'ExactMatch': calculate_exact_match,
            'NLD': calculate_nld
        }

    def calculate_all_metrics(self, prediction: str, reference: str) -> dict:
        """
        Calculate all metrics for a single prediction-reference pair

        Returns:
            Dictionary with all metric scores
        """
        results = {}
        for metric_name, metric_func in self.metrics.items():
            results[metric_name] = metric_func(prediction, reference)

        return results

    def calculate_batch_metrics(self, predictions: List[str],
                                references: List[str]) -> dict:
        """
        Calculate metrics for a batch of predictions

        Returns:
            Dictionary with mean scores for each metric
        """
        assert len(predictions) == len(references), \
            "Predictions and references must have the same length"

        batch_results = {metric: [] for metric in self.metrics.keys()}

        for pred, ref in zip(predictions, references):
            scores = self.calculate_all_metrics(pred, ref)
            for metric, score in scores.items():
                batch_results[metric].append(score)

        # Calculate means
        mean_results = {}
        for metric, scores in batch_results.items():
            mean_results[f"{metric}_mean"] = np.mean(scores)
            mean_results[f"{metric}_std"] = np.std(scores)
            mean_results[f"{metric}_min"] = np.min(scores)
            mean_results[f"{metric}_max"] = np.max(scores)

        return mean_results

    def calculate_metrics_by_style(self, predictions: List[str],
                                   references: List[str],
                                   styles: List[str]) -> dict:
        """
        Calculate metrics grouped by calligraphy style

        Args:
            predictions: List of model predictions
            references: List of ground truth texts
            styles: List of style labels for each sample

        Returns:
            Dictionary with metrics per style
        """
        style_results = {}
        unique_styles = set(styles)

        for style in unique_styles:
            # Filter predictions/references for this style
            style_preds = [p for p, s in zip(predictions, styles) if s == style]
            style_refs = [r for r, s in zip(references, styles) if s == style]

            if style_preds:
                style_results[style] = self.calculate_batch_metrics(
                    style_preds, style_refs
                )

        return style_results
