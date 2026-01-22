<p align="center">
  <img src="figures/logo.png" alt="DuwatBench Logo" width="200"/>
</p>

<h1 align="center">DuwatBench</h1>
<h3 align="center">دواة - معيار الخط العربي</h3>

<p align="center">
  <b>Bridging Language and Visual Heritage through an Arabic Calligraphy Benchmark for Multimodal Understanding</b>
</p>

<p align="center">
  <a href="#">Shubham Patle</a><sup>1†</sup>,
  <a href="#">Sara Ghaboura</a><sup>1†</sup>,
  <a href="#">Hania Tariq</a><sup>2</sup>,
  <a href="#">Mohammad Usman Khan</a><sup>3</sup>,
  <a href="https://omkarthawakar.github.io/">Omkar Thawakar</a><sup>1</sup>,
  <a href="https://scholar.google.fi/citations?user=_KlvMVoAAAAJ&hl=en">Rao Muhammad Anwer</a><sup>1</sup>,
  <a href="https://salman-h-khan.github.io/">Salman Khan</a><sup>1,4</sup>
</p>

<p align="center">
  <sup>1</sup>Mohamed bin Zayed University of AI &nbsp;&nbsp;
  <sup>2</sup>NUCES &nbsp;&nbsp;
  <sup>3</sup>NUST &nbsp;&nbsp;
  <sup>4</sup>Australian National University
</p>

<p align="center">
  <sup>†</sup>Equal Contribution
</p>

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-Paper-red.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/MBZUAI/DuwatBench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow" alt="HuggingFace"></a>
  <a href="https://mbzuai-oryx.github.io/DuwatBench/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"></a>
</p>

---

## Overview

**DuwatBench** is a comprehensive benchmark for evaluating multimodal large language models (LMMs) on Arabic calligraphy recognition. Arabic calligraphy represents one of the richest visual traditions of the Arabic language, blending linguistic meaning with artistic form. DuwatBench addresses the gap in evaluating how well modern AI systems can process stylized Arabic text.

<p align="center">
  <img src="figures/teaser.png" alt="DuwatBench Teaser" width="800"/>
</p>

### Key Features

- **1,272 curated samples** spanning 6 classical and modern calligraphic styles
- **~1,475 unique words** across religious and cultural domains
- **Bounding box annotations** for detection-level evaluation
- **Full text transcriptions** with style and theme labels
- **Complex artistic backgrounds** preserving real-world visual complexity

### Calligraphic Styles

| Style | Arabic | Samples | Description |
|-------|--------|---------|-------------|
| **Thuluth** | الثلث | 706 (55%) | Ornate script used in mosque decorations |
| **Diwani** | الديواني | 230 (18%) | Flowing Ottoman court script |
| **Naskh** | النسخ | 110 (9%) | Standard readable script |
| **Kufic** | الكوفي | 83 (7%) | Geometric angular early Arabic script |
| **Ruq'ah** | الرقعة | 76 (6%) | Modern everyday handwriting |
| **Nasta'liq** | النستعليق | 67 (5%) | Persian-influenced flowing script |

### Thematic Categories

| Category | Percentage |
|----------|------------|
| Quranic | 44% |
| Devotional/Hadith | 28% |
| Non-religious/Dedication | 11% |
| Names of Allah | 8% |
| Names of Prophet/Companions | 6% |
| Personal/Place Names | 3% |

---

## Installation

### Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended for open-source models)

### Setup

```bash
# Clone the repository
git clone https://github.com/mbzuai-oryx/DuwatBench.git
cd DuwatBench

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### API Keys Configuration

For closed-source models, set your API keys:

```bash
# Option 1: Environment variables
export GEMINI_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# Option 2: Create config file
cp src/config/api_keys.example.py src/config/api_keys.py
# Edit api_keys.py with your keys
```

---

## Dataset

### Download

```bash
# Download from Hugging Face
huggingface-cli download MBZUAI/DuwatBench --local-dir ./data

# Or use Python
from datasets import load_dataset
dataset = load_dataset("MBZUAI/DuwatBench")
```

### Data Format

Each sample in the JSON manifest contains:

```json
{
  "image_id": "images/2_129.jpg",
  "Style": "Thuluth",
  "Text": ["صَدَقَ اللَّهُ الْعَظِيمُ"],
  "word_count": [3],
  "total_words": 3,
  "bboxes": [[34, 336, 900, 312]],
  "Category": "quranic"
}
```

### Dataset Statistics

| Category | Count |
|----------|-------|
| Total Samples | 1,272 |
| Unique Words | ~1,475 |
| Calligraphy Styles | 6 |
| Quranic | 44% |
| Devotional/Hadith | 28% |
| Non-religious/Dedication | 11% |
| Names of Allah | 8% |
| Names of Prophet/Companions | 6% |

---

## Evaluation

### Quick Start

```bash
# Evaluate a single model
python src/evaluate.py --model gemini-2.5-flash --mode full_image

# Evaluate with bounding boxes
python src/evaluate.py --model gpt-4o-mini --mode with_bbox

# Evaluate both modes
python src/evaluate.py --model EasyOCR --mode both

# Resume interrupted evaluation
python src/evaluate.py --model claude-sonnet-4.5 --mode full_image --resume
```

### Supported Models (13 Total)

#### Open-Source (8)
| Model | CER ↓ | WER ↓ | chrF ↑ | ExactMatch ↑ | NLD ↓ |
|-------|-------|-------|--------|--------------|-------|
| MBZUAI/AIN* | 0.5494 | 0.6912 | 42.67 | 0.1895 | 0.5134 |
| Gemma-3-27B-IT | 0.5556 | 0.6591 | 51.53 | 0.2398 | 0.4741 |
| Qwen2.5-VL-72B | 0.5709 | 0.7039 | 43.98 | 0.1761 | 0.5298 |
| Qwen2.5-VL-7B | 0.6453 | 0.7768 | 36.97 | 0.1211 | 0.5984 |
| InternVL3-8B | 0.7588 | 0.8822 | 21.75 | 0.0574 | 0.7132 |
| EasyOCR | 0.8538 | 0.9895 | 12.30 | 0.0031 | 0.8163 |
| TrOCR-Arabic* | 0.9728 | 0.9998 | 1.79 | 0.0000 | 0.9632 |
| LLaVA-v1.6-Mistral-7B | 0.9932 | 0.9998 | 9.16 | 0.0000 | 0.9114 |

#### Closed-Source (5)
| Model | CER ↓ | WER ↓ | chrF ↑ | ExactMatch ↑ | NLD ↓ |
|-------|-------|-------|--------|--------------|-------|
| **Gemini-2.5-flash** | **0.3700** | **0.4478** | **71.82** | **0.4167** | **0.3166** |
| Gemini-1.5-flash | 0.3933 | 0.5112 | 63.28 | 0.3522 | 0.3659 |
| GPT-4o | 0.4766 | 0.5692 | 56.85 | 0.3388 | 0.4245 |
| GPT-4o-mini | 0.6039 | 0.7077 | 42.67 | 0.2115 | 0.5351 |
| Claude-Sonnet-4.5 | 0.6494 | 0.7255 | 42.97 | 0.2225 | 0.5599 |

*\* Arabic-specific models*

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **CER** | Character Error Rate - edit distance at character level |
| **WER** | Word Error Rate - edit distance at word level |
| **chrF** | Character n-gram F-score - partial match robustness |
| **ExactMatch** | Strict full-sequence accuracy |
| **NLD** | Normalized Levenshtein Distance - balanced error measure |

---

## Results

### Key Findings

- **Gemini-2.5-flash** achieves the best overall performance with 41.67% exact match accuracy
- Models perform best on **Naskh** and **Ruq'ah** (standardized strokes)
- **Diwani** and **Thuluth** (ornate scripts with dense ligatures) remain challenging
- **Kufic** records the lowest scores due to geometric rigidity
- Bounding box localization improves performance across most models

### Per-Style WER Performance (Full Image)

| Model | Kufic | Thuluth | Diwani | Naskh | Ruq'ah | Nasta'liq |
|-------|-------|---------|--------|-------|--------|-----------|
| Gemini-2.5-flash | 0.7067 | 0.3527 | 0.5698 | 0.4765 | 0.5817 | 0.5222 |
| Gemini-1.5-flash | 0.7212 | 0.4741 | 0.5783 | 0.4444 | 0.5445 | 0.5023 |
| GPT-4o | 0.8041 | 0.5540 | 0.6370 | 0.4189 | 0.5507 | 0.4434 |
| Gemma-3-27B-IT | 0.7802 | 0.6315 | 0.7326 | 0.5138 | 0.7571 | 0.6637 |
| MBZUAI/AIN | 0.7916 | 0.7036 | 0.7130 | 0.5367 | 0.6111 | 0.6916 |

### Statistical Analysis

| Metric | Mean | Std | Min | Max | Range |
|--------|------|-----|-----|-----|-------|
| CER | 0.6456 | 0.1993 | 0.3700 | 0.9932 | 62.7% |
| WER | 0.7434 | 0.1819 | 0.4478 | 0.9998 | 55.2% |
| NLD | 0.5940 | 0.2017 | 0.3166 | 0.9632 | 67.1% |
| chrF | 38.29 | 21.39 | 1.79 | 71.82 | 70.02 |
| ExactMatch | 0.2175 | 0.1390 | 0.0000 | 0.4167 | 0.42 |

**t-test (Open vs Closed):** All metrics show statistically significant differences (p < 0.05), with closed-source models outperforming open-source across all metrics.

---

## Project Structure

```
DuwatBench/
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
├── CITATION.cff
├── data/
│   ├── images/                    # Calligraphy images
│   └── duwatbench.json           # Dataset manifest
├── src/
│   ├── evaluate.py               # Main evaluation script
│   ├── models/
│   │   └── model_wrapper.py      # Model implementations
│   ├── metrics/
│   │   └── evaluation_metrics.py # CER, WER, chrF, etc.
│   ├── utils/
│   │   ├── data_loader.py        # Dataset loading
│   │   └── arabic_normalization.py
│   └── config/
│       ├── eval_config.py
│       └── api_keys.example.py
├── scripts/
│   ├── download_data.sh
│   └── run_all_evaluations.sh
└── results/                       # Evaluation outputs
```

---

## Citation

If you use DuwatBench in your research, please cite our paper:

```bibtex
@article{duwatbench2025,
  title={DuwatBench: Bridging Language and Visual Heritage through an
         Arabic Calligraphy Benchmark for Multimodal Understanding},
  author={Patle, Shubham and Ghaboura, Sara and Tariq, Hania and
          Khan, Mohammad Usman and Thawakar, Omkar and
          Anwer, Rao Muhammad and Khan, Salman},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

The dataset images are sourced from public digital archives and community repositories under their respective licenses.

---

## Acknowledgments

- Digital archives: [Library of Congress](https://www.loc.gov/collections/), [NYPL Digital Collections](https://digitalcollections.nypl.org/)
- Community repositories: [Calligraphy Qalam](https://calligraphyqalam.com/), [Free Islamic Calligraphy](https://freeislamiccalligraphy.com/), [Pinterest](https://www.pinterest.com/)
- Annotation tool: [MakeSense.ai](https://www.makesense.ai/)
- Arabic NLP tools: [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools)

---

## Contact

For questions or issues, please:
- Open an issue on [GitHub](https://github.com/mbzuai-oryx/DuwatBench/issues)
- Contact the authors at: {shubham.patle, sara.ghaboura, omkar.thawakar}@mbzuai.ac.ae

---

<p align="center">
  <a href="https://mbzuai.ac.ae"><img src="figures/mbzuai_logo.png" height="50" alt="MBZUAI"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/mbzuai-oryx"><img src="figures/oryx_logo.png" height="50" alt="Oryx"></a>
</p>
