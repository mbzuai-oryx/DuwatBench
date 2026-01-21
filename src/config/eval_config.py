#!/usr/bin/env python3
"""
Configuration file for DuwatBench evaluation
Based on the paper: "DuwatBench: Bridging Language and Visual Heritage"
"""

import os

# Get the root directory of the project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dataset paths (relative to project root)
DATA_DIR = os.path.join(ROOT_DIR, "data")
DATA_MANIFEST = os.path.join(DATA_DIR, "duwatbench.json")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

# Results directory
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Evaluation metrics (from paper Section 3.1)
METRICS = [
    "CER",          # Character Error Rate
    "WER",          # Word Error Rate
    "chrF",         # Character F-score
    "ExactMatch",   # Exact Match accuracy
    "NLD"           # Normalized Levenshtein Distance
]

# Calligraphy styles (6 styles from paper Figure 1)
STYLES = [
    "Thuluth",
    "Diwani",
    "Kufic",
    "Naskh",
    "Ruq'ah",
    "Nasta'liq"
]

# Textual categories (from Figure 6)
TEXT_CATEGORIES = [
    "Non-religious",
    "Quranic",
    "Devotional",
    "Names of Prophet/Companions",
    "Names of Allah",
    "Personal Names/Dedications",
    "Hadith"
]

# Arabic normalization settings (from paper Section 3.1)
ARABIC_NORMALIZATION = {
    "remove_tatweel": True,          # Remove ـ
    "normalize_alef": True,           # Unify أ إ آ → ا
    "normalize_alef_maqsurah": True,  # ى → ي
    "unicode_normalize": True,        # Unicode normalization
    "remove_diacritics": True,        # Remove harakat/tashkeel for fair comparison
}

# Models to evaluate (from Tables 2 & 3)
OPEN_SOURCE_MODELS = [
    "Llava-v1.6-mistral-7b-hf",
    "EasyOCR",
    "InternVL3-8B",
    "Qwen2.5-VL-7B",
    "Qwen2.5-VL-72B-Instruct",
    "gemma-3-27b-it",
    "trocr-base-arabic-handwritten",
    "MBZUAI/AIN"
]

CLOSED_SOURCE_MODELS = [
    "claude-sonnet-4.5",
    "gemini-1.5-flash",
    "gemini-2.5-flash",
    "gpt-4o-mini",
    "gpt-4o"
]

ALL_MODELS = OPEN_SOURCE_MODELS + CLOSED_SOURCE_MODELS

# API keys - loaded from api_keys.py file or environment variables
try:
    from .api_keys import (
        GEMINI_API_KEY,
        OPENAI_API_KEY,
        ANTHROPIC_API_KEY,
        HUGGINGFACE_TOKEN
    )
except ImportError:
    # Fallback to environment variables if api_keys.py doesn't exist
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# Evaluation modes
EVAL_MODES = {
    "full_image": "Evaluate on full calligraphy images without bbox cropping",
    "with_bbox": "Evaluate with bounding box annotations (Table 3)"
}

# Arabic OCR prompts (from paper methodology)
OCR_PROMPT_TEMPLATE = """اقرأ النص العربي في الصورة وانسخه بالضبط. أخرج النص العربي فقط."""

OCR_PROMPT_WITH_BBOX = """اقرأ النص العربي في الصورة وانسخه بالضبط. أخرج النص العربي فقط."""

# Prompt for bbox mode - draws colored box on image and asks model to read text inside
OCR_PROMPT_BBOX_COLORED = """انظر إلى المربع الملون ({color}) المرسوم على الصورة. اقرأ النص العربي الموجود داخل هذا المربع فقط وانسخه بالضبط. أخرج النص العربي فقط."""

# Color names in Arabic for bbox prompts
BBOX_COLORS_ARABIC = {
    "red": "أحمر",
    "green": "أخضر",
    "blue": "أزرق",
    "yellow": "أصفر",
    "magenta": "وردي",
    "cyan": "سماوي",
    "orange": "برتقالي",
    "purple": "بنفسجي",
}

# Colors for drawing bboxes (RGB)
BBOX_COLORS_RGB = [
    (255, 0, 0),      # red
    (0, 255, 0),      # green
    (0, 0, 255),      # blue
    (255, 255, 0),    # yellow
    (255, 0, 255),    # magenta
    (0, 255, 255),    # cyan
    (255, 165, 0),    # orange
    (128, 0, 128),    # purple
]

# Color names in order (English)
BBOX_COLOR_NAMES = ["red", "green", "blue", "yellow", "magenta", "cyan", "orange", "purple"]

# Evaluation settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_BATCH_SIZE = 1

# Statistical analysis settings
SIGNIFICANCE_LEVEL = 0.05
