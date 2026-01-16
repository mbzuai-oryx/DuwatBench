#!/usr/bin/env python3
"""
Configuration for DuwatBench evaluation
"""

import os

# Get the root directory of the project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dataset paths
DATA_DIR = os.path.join(ROOT_DIR, "data")
DATA_MANIFEST = os.path.join(DATA_DIR, "duwatbench.jsonl")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

# Results directory
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# API Keys - Set these via environment variables or api_keys.py
try:
    from .api_keys import GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, HUGGINGFACE_TOKEN
except ImportError:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# Model categories from paper
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

# Arabic OCR prompts (from paper methodology)
OCR_PROMPT_TEMPLATE = "اقرأ النص العربي في الصورة وانسخه بالضبط. أخرج النص العربي فقط."
OCR_PROMPT_WITH_BBOX = "اقرأ النص العربي في الصورة وانسخه بالضبط. أخرج النص العربي فقط."

# Calligraphy styles (from paper Figure 1)
CALLIGRAPHY_STYLES = [
    "Thuluth",
    "Diwani",
    "Kufic",
    "Naskh",
    "Ruq'ah",
    "Nasta'liq"
]

# Evaluation settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_BATCH_SIZE = 1
