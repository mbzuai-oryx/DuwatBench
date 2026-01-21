#!/usr/bin/env python3
"""
Data loading utilities for DuwatBench
Loads JSON data with styles and text annotations

Supports duwatbench.json format:
{
    "image_id": "dwn34.jpg",
    "Text": ["text1", "text2"],
    "word_count": [11, 4],
    "Style": "Diwani",
    "Category": "quranic",
    "total_words": 14,
    "bboxes": [[x, y, w, h], ...]
}
"""

import json
import os
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageDraw
import numpy as np


class DuwatBenchDataset:
    """
    Dataset loader for DuwatBench Arabic calligraphy benchmark
    """

    def __init__(self, json_path: str, images_dir: str):
        """
        Args:
            json_path: Path to duwatbench.json
            images_dir: Path to images directory
        """
        self.json_path = json_path
        self.images_dir = images_dir
        self.samples = []
        self.load_data()

    def load_data(self):
        """Load JSON data"""
        print(f"Loading data from {self.json_path}...")

        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")

        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert to internal format for compatibility
        for item in data:
            sample = {
                'image_id': item['image_id'],
                'image_path': item['image_id'],
                'texts': item['Text'],
                'word_count': item['word_count'],
                'style': item['Style'],
                'category': item['Category'],
                'total_words': item['total_words'],
                'bboxes': item.get('bboxes', []),
                'image_width': item.get('image_width'),
                'image_height': item.get('image_height')
            }
            self.samples.append(sample)

        print(f"Loaded {len(self.samples)} samples")

    def get_sample(self, idx: int) -> Dict:
        """Get a single sample by index"""
        return self.samples[idx]

    def get_image(self, idx: int) -> Image.Image:
        """Load image for a sample and convert to RGB if needed"""
        sample = self.samples[idx]
        image_path = os.path.join(self.images_dir, sample['image_path'])

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path)

        # Convert all non-RGB images to RGB for consistent processing
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image

    def get_cropped_regions(self, idx: int) -> List[Tuple[Image.Image, str]]:
        """
        Get cropped text regions using bounding boxes

        Returns:
            List of tuples (cropped_image, corresponding_text)
        """
        sample = self.samples[idx]
        image = self.get_image(idx)
        bboxes = sample.get('bboxes', [])
        texts = sample['texts']

        if not bboxes:
            return []

        cropped_regions = []
        for bbox, text in zip(bboxes, texts):
            # Bbox format: [x, y, width, height] (COCO format)
            x, y, w, h = bbox
            cropped = image.crop((x, y, x + w, y + h))
            cropped_regions.append((cropped, text))

        return cropped_regions

    def has_bboxes(self, idx: int) -> bool:
        """Check if sample has bounding boxes"""
        sample = self.samples[idx]
        return bool(sample.get('bboxes', []))

    def get_image_with_bbox_drawn(self, idx: int, bbox_idx: int,
                                   colors_rgb: List[Tuple[int, int, int]],
                                   color_names: List[str]) -> Tuple[Image.Image, str, str]:
        """
        Get image with specific bbox highlighted

        Args:
            idx: Sample index
            bbox_idx: Index of the bbox to draw (0-indexed)
            colors_rgb: List of RGB color tuples
            color_names: List of color names (English)

        Returns:
            Tuple of (image_with_bbox, corresponding_text, color_name)
        """
        sample = self.samples[idx]
        image = self.get_image(idx).copy()
        bboxes = sample.get('bboxes', [])
        texts = sample['texts']

        if not bboxes or bbox_idx >= len(bboxes):
            return image, texts[bbox_idx] if bbox_idx < len(texts) else "", ""

        bbox = bboxes[bbox_idx]
        text = texts[bbox_idx]

        # Use RED (first color) for single bbox
        color = colors_rgb[0]
        color_name = color_names[0]

        x, y, w, h = bbox

        # Draw thick rectangle outline around the bbox
        draw = ImageDraw.Draw(image)
        border_width = 10
        for i in range(border_width):
            draw.rectangle([x-i, y-i, x+w+i, y+h+i], outline=color)

        return image, text, color_name

    def get_full_text(self, idx: int) -> str:
        """Get concatenated full text from all text regions"""
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
            'total_words': sum(s.get('total_words', 0) for s in self.samples),
            'avg_words_per_sample': np.mean([s.get('total_words', 0) for s in self.samples]),
        }
        return stats


def load_dataset(json_path: str, images_dir: str) -> DuwatBenchDataset:
    """
    Convenience function to load dataset

    Args:
        json_path: Path to JSON file
        images_dir: Path to images directory

    Returns:
        DuwatBenchDataset instance
    """
    return DuwatBenchDataset(json_path, images_dir)
