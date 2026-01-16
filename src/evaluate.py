#!/usr/bin/env python3
"""
Main evaluation script for DuwatBench with checkpointing and resume capability
Reproduces evaluation from paper Tables 2 & 3

Features:
- Checkpointing: Saves progress after each sample
- Resume: Skip already completed samples to save cost
- Retry logic for failed samples
- Save all model responses for verification

Usage:
    python evaluate.py --model gemini-2.5-flash --mode full_image
    python evaluate.py --model gemini-2.5-flash --mode with_bbox --limit 10
    python evaluate.py --model EasyOCR --mode both
    python evaluate.py --model gpt-4o-mini --mode full_image --resume  # Resume from checkpoint
"""

import argparse
import json
import os
import sys
from tqdm import tqdm
import time
import traceback
from typing import List, Dict, Optional, Set
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.eval_config import *
from utils.data_loader import load_dataset
from utils.arabic_normalization import ArabicNormalizer, normalize_prediction_and_reference
from metrics.evaluation_metrics import MetricsCalculator
from models.model_wrapper import create_model


class DuwatBenchEvaluator:
    """
    Enhanced evaluator with checkpointing, resume, and retry logic
    Implements evaluation methodology from paper Section 3
    """

    def __init__(self, model_name: str, eval_mode: str = "full_image",
                 max_retries: int = 3, resume: bool = True):
        """
        Args:
            model_name: Name of model to evaluate
            eval_mode: "full_image" or "with_bbox"
            max_retries: Maximum retry attempts for failed samples
            resume: If True, load existing checkpoint and skip completed samples
        """
        self.model_name = model_name
        self.eval_mode = eval_mode
        self.max_retries = max_retries
        self.resume = resume
        self.normalizer = ArabicNormalizer()
        self.metrics_calc = MetricsCalculator()

        # Checkpoint file path (consistent naming for resume)
        self.checkpoint_dir = os.path.join(RESULTS_DIR, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        safe_model_name = model_name.replace('/', '_')
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir,
            f"{safe_model_name}_{eval_mode}_checkpoint.json"
        )

        # Results directory for individual responses
        self.responses_dir = os.path.join(RESULTS_DIR, "responses", f"{safe_model_name}_{eval_mode}")
        os.makedirs(self.responses_dir, exist_ok=True)

        # Load existing checkpoint if resuming
        self.results = []
        self.failed_samples = []
        self.completed_indices: Set[int] = set()

        if resume:
            self._load_checkpoint()

        print(f"Initializing model: {model_name}")
        self.model = create_model(model_name)

        print(f"Loading dataset...")
        self.dataset = load_dataset(DATA_MANIFEST, IMAGES_DIR)

    def _load_checkpoint(self):
        """Load existing checkpoint if available"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)

                self.results = checkpoint.get('results', [])
                self.failed_samples = checkpoint.get('failed_samples', [])

                # Build set of completed indices
                self.completed_indices = set(r['sample_idx'] for r in self.results)
                failed_indices = set(f['sample_idx'] for f in self.failed_samples)
                self.completed_indices.update(failed_indices)

                print(f"✓ Loaded checkpoint: {len(self.results)} completed, {len(self.failed_samples)} failed")
                print(f"  Will skip {len(self.completed_indices)} already processed samples")

            except Exception as e:
                print(f"⚠ Could not load checkpoint: {e}")
                print("  Starting fresh evaluation")
        else:
            print(f"No checkpoint found. Starting fresh evaluation.")

    def _save_checkpoint(self):
        """Save current progress to checkpoint file"""
        checkpoint = {
            'model': self.model_name,
            'eval_mode': self.eval_mode,
            'timestamp': datetime.now().isoformat(),
            'n_completed': len(self.results),
            'n_failed': len(self.failed_samples),
            'results': self.results,
            'failed_samples': self.failed_samples
        }

        # Write to temp file first, then rename (atomic write)
        temp_file = self.checkpoint_file + '.tmp'
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

        os.replace(temp_file, self.checkpoint_file)

    def evaluate_sample(self, idx: int) -> Optional[Dict]:
        """
        Evaluate a single sample with retry logic

        Args:
            idx: Sample index

        Returns:
            Result dict or None if all retries failed
        """
        sample = self.dataset.get_sample(idx)
        reference_text = self.dataset.get_full_text(idx)

        for attempt in range(self.max_retries):
            try:
                # Get prediction
                if self.eval_mode == "full_image":
                    image = self.dataset.get_image(idx)
                    prediction = self.model.transcribe(image, prompt=OCR_PROMPT_TEMPLATE)

                elif self.eval_mode == "with_bbox":
                    cropped_regions = self.dataset.get_cropped_regions(idx)
                    if not cropped_regions:
                        # No bboxes, fallback to full image
                        image = self.dataset.get_image(idx)
                        prediction = self.model.transcribe(image, prompt=OCR_PROMPT_TEMPLATE)
                    else:
                        # Process each bbox independently and average metrics
                        # This avoids text duplication from overlapping bboxes
                        bbox_predictions = []
                        bbox_references = []
                        for cropped_img, bbox_text in cropped_regions:
                            pred = self.model.transcribe(cropped_img, prompt=OCR_PROMPT_WITH_BBOX)
                            bbox_predictions.append(pred)
                            bbox_references.append(bbox_text)

                        # Store for per-bbox metric calculation
                        # Concatenate for display/storage but metrics calculated per-bbox
                        prediction = ' '.join(bbox_predictions)
                        # Store bbox-level data for proper evaluation
                        sample['_bbox_predictions'] = bbox_predictions
                        sample['_bbox_references'] = bbox_references

                else:
                    raise ValueError(f"Invalid eval_mode: {self.eval_mode}")

                # Save raw response for verification
                self._save_response(idx, sample, prediction, reference_text)

                # Calculate metrics - for bbox mode, average metrics across individual bboxes
                if self.eval_mode == "with_bbox" and '_bbox_predictions' in sample:
                    # Per-bbox evaluation to handle overlapping regions correctly
                    bbox_metrics_list = []
                    norm_preds = []
                    norm_refs = []
                    for bp, br in zip(sample['_bbox_predictions'], sample['_bbox_references']):
                        np_i, nr_i = normalize_prediction_and_reference(bp, br, self.normalizer)
                        norm_preds.append(np_i)
                        norm_refs.append(nr_i)
                        m = self.metrics_calc.calculate_all_metrics(np_i, nr_i)
                        bbox_metrics_list.append(m)

                    # Average the metrics across all bboxes
                    metrics = {}
                    for key in bbox_metrics_list[0].keys():
                        metrics[key] = sum(m[key] for m in bbox_metrics_list) / len(bbox_metrics_list)

                    norm_pred = ' '.join(norm_preds)
                    norm_ref = ' '.join(norm_refs)

                    # Clean up temporary data
                    del sample['_bbox_predictions']
                    del sample['_bbox_references']
                else:
                    # Full image mode or single bbox - standard evaluation
                    norm_pred, norm_ref = normalize_prediction_and_reference(
                        prediction, reference_text, self.normalizer
                    )
                    metrics = self.metrics_calc.calculate_all_metrics(norm_pred, norm_ref)

                # Store result
                result = {
                    'sample_idx': idx,
                    'image_path': sample['image_path'],
                    'style': sample['style'],
                    'reference': reference_text,
                    'prediction': prediction,
                    'normalized_prediction': norm_pred,
                    'normalized_reference': norm_ref,
                    'attempt': attempt + 1,
                    'success': True,
                    'timestamp': datetime.now().isoformat(),
                    **metrics
                }

                return result

            except Exception as e:
                error_msg = f"Attempt {attempt + 1}/{self.max_retries} failed for sample {idx}: {str(e)}"
                print(f"\n{error_msg}")

                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    # All retries failed
                    print(f"❌ Sample {idx} failed after {self.max_retries} attempts")
                    failed_entry = {
                        'sample_idx': idx,
                        'image_path': sample['image_path'],
                        'error': str(e),
                        'traceback': traceback.format_exc(),
                        'timestamp': datetime.now().isoformat()
                    }
                    self.failed_samples.append(failed_entry)
                    return None

        return None

    def _save_response(self, idx: int, sample: Dict, prediction: str, reference: str):
        """Save model response for manual verification"""
        response_file = os.path.join(self.responses_dir, f"sample_{idx:04d}.json")

        response_data = {
            'sample_idx': idx,
            'image_path': sample['image_path'],
            'style': sample['style'],
            'reference_text': reference,
            'model_prediction': prediction,
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'eval_mode': self.eval_mode
        }

        with open(response_file, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, ensure_ascii=False, indent=2)

    def evaluate_all(self, limit: Optional[int] = None) -> Dict:
        """
        Evaluate all samples in dataset with checkpointing

        Args:
            limit: Optional limit on number of samples to evaluate

        Returns:
            Dictionary with aggregated results
        """
        n_samples = len(self.dataset) if limit is None else min(limit, len(self.dataset))

        # Calculate how many samples to process
        samples_to_process = [i for i in range(n_samples) if i not in self.completed_indices]
        n_to_process = len(samples_to_process)
        n_already_done = n_samples - n_to_process

        print(f"\n{'='*70}")
        print(f"Evaluating with {self.model_name}")
        print(f"Mode: {self.eval_mode}")
        print(f"{'='*70}")
        print(f"Total samples: {n_samples}")
        print(f"Already completed: {n_already_done} (will skip)")
        print(f"To process: {n_to_process}")
        print(f"Max retries per sample: {self.max_retries}")
        print(f"Checkpoint file: {self.checkpoint_file}")
        print("="*70)

        if n_to_process == 0:
            print("\n✓ All samples already completed! Use --no-resume to re-run.")
            return self._aggregate_results()

        # Process remaining samples
        for idx in tqdm(samples_to_process, desc="Evaluating"):
            try:
                result = self.evaluate_sample(idx)

                if result is not None:
                    self.results.append(result)
                    self.completed_indices.add(idx)

                # Save checkpoint after EVERY sample to preserve progress
                self._save_checkpoint()

                # Rate limiting for API calls
                if self.model_name in CLOSED_SOURCE_MODELS:
                    time.sleep(0.5)  # Reduced from 1s for efficiency

            except KeyboardInterrupt:
                print(f"\n\n⚠ Interrupted! Saving checkpoint...")
                self._save_checkpoint()
                print(f"✓ Progress saved. Run again with --resume to continue.")
                sys.exit(0)

            except Exception as e:
                print(f"\nUnexpected error on sample {idx}: {e}")
                traceback.print_exc()
                # Save checkpoint even on error
                self._save_checkpoint()
                continue

        # Final save
        self._save_checkpoint()

        # Print failed samples summary
        if self.failed_samples:
            print(f"\n⚠ {len(self.failed_samples)} samples failed after {self.max_retries} retries")
            self._save_failed_samples_report()

        # Aggregate results
        aggregated = self._aggregate_results()

        return aggregated

    def _aggregate_results(self) -> Dict:
        """Aggregate results across all samples"""
        if not self.results:
            return {
                'model': self.model_name,
                'eval_mode': self.eval_mode,
                'n_samples': 0,
                'n_failed': len(self.failed_samples),
                'overall_metrics': {},
                'per_style_metrics': {},
                'detailed_results': []
            }

        # Overall metrics
        all_preds = [r['normalized_prediction'] for r in self.results]
        all_refs = [r['normalized_reference'] for r in self.results]

        overall_metrics = self.metrics_calc.calculate_batch_metrics(all_preds, all_refs)

        # Per-style metrics (Table 4)
        styles = [r['style'] for r in self.results]
        style_metrics = self.metrics_calc.calculate_metrics_by_style(
            all_preds, all_refs, styles
        )

        aggregated = {
            'model': self.model_name,
            'eval_mode': self.eval_mode,
            'n_samples': len(self.results),
            'n_failed': len(self.failed_samples),
            'success_rate': len(self.results) / (len(self.results) + len(self.failed_samples)) if (len(self.results) + len(self.failed_samples)) > 0 else 0.0,
            'overall_metrics': overall_metrics,
            'per_style_metrics': style_metrics,
            'detailed_results': self.results
        }

        return aggregated

    def save_results(self, output_path: str):
        """Save final results to JSON file"""
        aggregated = self._aggregate_results()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(aggregated, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to: {output_path}")

    def _save_failed_samples_report(self):
        """Save failed samples report to a separate file"""
        if not self.failed_samples:
            return

        safe_model_name = self.model_name.replace('/', '_')
        failed_report_path = os.path.join(
            RESULTS_DIR,
            f"{safe_model_name}_{self.eval_mode}_failed_samples.json"
        )

        failed_report = {
            'model': self.model_name,
            'eval_mode': self.eval_mode,
            'timestamp': datetime.now().isoformat(),
            'total_failed': len(self.failed_samples),
            'max_retries': self.max_retries,
            'failed_sample_indices': [f['sample_idx'] for f in self.failed_samples],
            'failed_samples': self.failed_samples
        }

        with open(failed_report_path, 'w', encoding='utf-8') as f:
            json.dump(failed_report, f, ensure_ascii=False, indent=2)

        print(f"Failed samples report saved to: {failed_report_path}")

    def print_summary(self):
        """Print evaluation summary"""
        aggregated = self._aggregate_results()

        print("\n" + "="*70)
        print(f"EVALUATION SUMMARY: {self.model_name}")
        print(f"Mode: {self.eval_mode}")
        print("="*70)

        # Success/failure stats
        total_attempted = len(self.results) + len(self.failed_samples)
        success_rate = aggregated.get('success_rate', 0.0) * 100

        print(f"\nEvaluation Stats:")
        print(f"  Total samples attempted: {total_attempted}")
        print(f"  Successful samples:      {len(self.results)}")
        print(f"  Failed samples:          {len(self.failed_samples)}")
        print(f"  Success rate:            {success_rate:.2f}%")

        if not self.results:
            print("\n⚠ No successful evaluations to report metrics.")
            print("="*70)
            return

        metrics = aggregated['overall_metrics']
        print(f"\nOverall Metrics ({len(self.results)} samples):")
        print(f"  CER (mean): {metrics['CER_mean']:.4f}")
        print(f"  WER (mean): {metrics['WER_mean']:.4f}")
        print(f"  chrF:       {metrics['chrF_mean']:.4f}")
        print(f"  ExactMatch: {metrics['ExactMatch_mean']:.4f}")
        print(f"  NLD error:  {metrics['NLD_mean']:.4f}")

        if aggregated['per_style_metrics']:
            print(f"\nPer-Style WER (reproducing Table 4):")
            for style, style_metrics in aggregated['per_style_metrics'].items():
                print(f"  {style:12s}: {style_metrics['WER_mean']:.4f}")

        print("="*70)

    def retry_failed_samples(self) -> int:
        """
        Retry only the failed samples from previous run

        Returns:
            Number of samples successfully recovered
        """
        if not self.failed_samples:
            print("No failed samples to retry.")
            return 0

        failed_indices = [f['sample_idx'] for f in self.failed_samples]
        print(f"\nRetrying {len(failed_indices)} failed samples...")

        # Clear failed samples list (will be re-populated if they fail again)
        old_failed = self.failed_samples.copy()
        self.failed_samples = []

        recovered = 0
        for idx in tqdm(failed_indices, desc="Retrying failed"):
            result = self.evaluate_sample(idx)
            if result is not None:
                self.results.append(result)
                self.completed_indices.add(idx)
                recovered += 1

            self._save_checkpoint()

            if self.model_name in CLOSED_SOURCE_MODELS:
                time.sleep(0.5)

        print(f"✓ Recovered {recovered}/{len(failed_indices)} samples")
        return recovered


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on DuwatBench Arabic Calligraphy Benchmark"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to evaluate (e.g., gemini-2.5-flash, gpt-4o-mini, EasyOCR)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full_image",
        choices=["full_image", "with_bbox", "both"],
        help="Evaluation mode"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to evaluate (for testing)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSON"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for failed samples (default: 3)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint (default: True)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore existing checkpoint"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Only retry previously failed samples"
    )

    args = parser.parse_args()

    # Handle resume flag
    resume = args.resume and not args.no_resume

    modes = ["full_image", "with_bbox"] if args.mode == "both" else [args.mode]

    for mode in modes:
        evaluator = DuwatBenchEvaluator(
            model_name=args.model,
            eval_mode=mode,
            max_retries=args.max_retries,
            resume=resume
        )

        if args.retry_failed:
            # Only retry failed samples
            evaluator.retry_failed_samples()
        else:
            # Normal evaluation
            evaluator.evaluate_all(limit=args.limit)

        # Print summary
        evaluator.print_summary()

        # Save results
        if args.output:
            output_path = args.output
        else:
            output_path = os.path.join(
                RESULTS_DIR,
                f"{args.model.replace('/', '_')}_{mode}_results.json"
            )

        evaluator.save_results(output_path)


if __name__ == "__main__":
    main()
