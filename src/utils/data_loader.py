#!/usr/bin/env python3
"""
Data loading utilities for DuwatBench
Loads JSONL data with bounding boxes, styles, and text annotations
"""

import json
import os
from typing import List, Dict, Optional, Tuple
from PIL import Image
import numpy as np


class DuwatBenchDataset:
    """
    Dataset loader for DuwatBench Arabic calligraphy benchmark

    Format from paper (Figure 2):
    {
        "image_path": "images/1_4.jpg",
        "style": "Kufic",
        "texts": ["مَا شَاءَ ٱللَّهُ لَا قُوَّةَ إِلَّا بِٱللَّهِ"],
        "word_count": [20],
        "total_words": [20],
        "bboxes": [[405, 259, 390, 681]]
    }
    """

    def __init__(self, jsonl_path: str, images_dir: str):
        """
        Args:
            jsonl_path: Path to data_manifest_flat.jsonl
            images_dir: Path to images directory
        """
        self.jsonl_path = jsonl_path
        self.images_dir = images_dir
        self.samples = []
        self.load_data()

    def load_data(self):
        """Load JSONL data"""
        print(f"Loading data from {self.jsonl_path}...")

        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(f"JSONL file not found: {self.jsonl_path}")

        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    self.samples.append(sample)

        print(f"Loaded {len(self.samples)} samples")

    def get_sample(self, idx: int) -> Dict:
        """Get a single sample by index"""
        return self.samples[idx]

    def get_image(self, idx: int) -> Image.Image:
        """Load image for a sample and convert to RGB if needed"""
        sample = self.samples[idx]
        image_path = os.path.join(self.images_dir,
                                 sample['image_path'].replace('images/', ''))

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path)

        # Convert problematic image modes to RGB
        # P = palette mode, CMYK = print colors, LA/PA = with alpha
        if image.mode in ('P', 'CMYK', 'LA', 'PA', 'I', 'F'):
            image = image.convert('RGB')

        return image

    def get_cropped_regions(self, idx: int) -> List[Tuple[Image.Image, str]]:
        """
        Get cropped text regions using bounding boxes
        Note: JSONL is pre-sorted by y-coordinate (top to bottom) for proper Arabic reading order

        Returns:
            List of tuples (cropped_image, corresponding_text)
        """
        sample = self.samples[idx]
        image = self.get_image(idx)
        bboxes = sample['bboxes']
        texts = sample['texts']

        cropped_regions = []
        for bbox, text in zip(bboxes, texts):
            # Bbox format: [x, y, width, height] (COCO format)
            x, y, w, h = bbox
            cropped = image.crop((x, y, x + w, y + h))
            cropped_regions.append((cropped, text))

        return cropped_regions

    def get_full_text(self, idx: int) -> str:
        """Get concatenated full text from all text regions
        Note: JSONL is pre-sorted by y-coordinate (top to bottom) for proper Arabic reading order
        """
        sample = self.samples[idx]
        return ' '.join(sample['texts'])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_samples_by_style(self, style: str) -> List[int]:
        """Get indices of all samples with a specific style"""
        return [i for i, s in enumerate(self.samples) if s['style'] == style]

    def get_style_distribution(self) -> Dict[str, int]:
        """Get count of samples per style"""
        styles = {}
        for sample in self.samples:
            style = sample['style']
            styles[style] = styles.get(style, 0) + 1
        return styles

    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            'total_samples': len(self.samples),
            'styles': self.get_style_distribution(),
            'total_words': sum(s.get('total_words', [0])[0] for s in self.samples),
            'avg_words_per_sample': np.mean([s.get('total_words', [0])[0] for s in self.samples]),
        }

        # Count samples by word count ranges (from Figure 8)
        word_count_ranges = {
            '[01-10]': 0,
            '[11-20]': 0,
            '[21-30]': 0,
            '[31-40]': 0,
            '[41-50]': 0,
            '[51-60]': 0,
            '[61-70]': 0,
            '[71-80]': 0,
            '[81-90]': 0
        }

        for sample in self.samples:
            total = sum(sample.get('total_words', [0]))
            if 1 <= total <= 10:
                word_count_ranges['[01-10]'] += 1
            elif 11 <= total <= 20:
                word_count_ranges['[11-20]'] += 1
            elif 21 <= total <= 30:
                word_count_ranges['[21-30]'] += 1
            elif 31 <= total <= 40:
                word_count_ranges['[31-40]'] += 1
            elif 41 <= total <= 50:
                word_count_ranges['[41-50]'] += 1
            elif 51 <= total <= 60:
                word_count_ranges['[51-60]'] += 1
            elif 61 <= total <= 70:
                word_count_ranges['[61-70]'] += 1
            elif 71 <= total <= 80:
                word_count_ranges['[71-80]'] += 1
            elif 81 <= total <= 90:
                word_count_ranges['[81-90]'] += 1

        stats['word_count_distribution'] = word_count_ranges

        return stats


def load_dataset(jsonl_path: str, images_dir: str) -> DuwatBenchDataset:
    """
    Convenience function to load dataset

    Args:
        jsonl_path: Path to JSONL manifest
        images_dir: Path to images directory

    Returns:
        DuwatBenchDataset instance
    """
    return DuwatBenchDataset(jsonl_path, images_dir)
