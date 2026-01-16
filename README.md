<p align="center">
  <img src="docs/assets/duwatbench_logo.png" alt="DuwatBench Logo" width="200"/>
</p>

<h1 align="center">DuwatBench</h1>

<p align="center">
  <b>Bridging Language and Visual Heritage through an Arabic Calligraphy Benchmark for Multimodal Understanding</b>
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
  <img src="docs/assets/taxonomy.png" alt="DuwatBench Taxonomy" width="800"/>
</p>

### Key Features

- **1,050+ curated samples** spanning 6 classical and modern calligraphic styles
- **~1,400 unique words** across religious and non-religious domains
- **Bounding box annotations** for detection-level evaluation
- **Full text transcriptions** with style and theme labels
- **Complex artistic backgrounds** preserving real-world visual complexity

### Calligraphic Styles

| Style | Samples | Description |
|-------|---------|-------------|
| **Thuluth** | 699 | Ornate script used in mosque decorations |
| **Diwani** | 258 | Flowing Ottoman court script |
| **Kufic** | 62 | Geometric angular early Arabic script |
| **Naskh** | 15 | Standard readable script |
| **Ruq'ah** | 10 | Modern everyday handwriting |
| **Nasta'liq** | 6 | Persian-influenced flowing script |

---

## Installation

### Requirements

- Python 3.8+
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

Each sample in the JSONL manifest contains:

```json
{
  "image_path": "images/2_129.jpg",
  "style": "Thuluth",
  "texts": ["صَدَقَ اللَّهُ الْعَظِيمُ"],
  "word_count": [3],
  "total_words": 3,
  "bboxes": [[34, 336, 900, 312]]
}
```

### Dataset Statistics

- **Total Samples**: 1,050
- **Total Words**: ~1,400
- **Styles**: 6 (Thuluth, Diwani, Kufic, Naskh, Ruq'ah, Nasta'liq)
- **Themes**: Non-religious (45%), Quranic (22%), Devotional (20%), Names (12%)

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
| Model | CER | WER | chrF | ExactMatch |
|-------|-----|-----|------|------------|
| gemma-3-27b-it | 0.637 | 0.768 | 38.83 | 0.324 |
| MBZUAI/AIN* | 0.669 | 0.819 | 22.08 | 0.227 |
| Qwen2.5-VL-72B-Instruct | 0.697 | 0.859 | 29.26 | 0.243 |
| Qwen2.5-VL-7B | 0.650 | 0.808 | 19.17 | 0.207 |
| InternVL3-8B | 0.746 | 0.878 | 10.33 | 0.119 |
| EasyOCR | 0.786 | 1.021 | 7.74 | 0.019 |
| trocr-base-arabic-handwritten* | 1.034 | 1.044 | 0.76 | 0.000 |
| Llava-v1.6-mistral-7b-hf | 1.096 | 1.787 | 0.48 | 0.006 |

#### Closed-Source (5)
| Model | CER | WER | chrF | ExactMatch |
|-------|-----|-----|------|------------|
| **gemini-2.5-flash** | **0.316** | **0.416** | **59.96** | **0.561** |
| gpt-4o-mini | 0.533 | 0.683 | 27.70 | 0.355 |
| gpt-4o | 0.830 | 0.980 | 17.12 | 0.186 |
| gemini-1.5-flash | 0.912 | 1.026 | 41.93 | 0.244 |
| claude-sonnet-4.5 | 1.181 | 1.080 | 27.63 | 0.338 |

*\* Arabic-specific models*

### Evaluation Metrics

Following standard OCR evaluation practices, we use five complementary metrics:

| Metric | Description |
|--------|-------------|
| **CER** | Character Error Rate - edit distance at character level |
| **WER** | Word Error Rate - edit distance at word level |
| **chrF** | Character n-gram F-score - partial match robustness |
| **ExactMatch** | Strict full-sequence accuracy |
| **NLD** | Normalized Levenshtein Distance - balanced error measure |

---

## Results

### Full Image Evaluation (Table 2)

<p align="center">
  <img src="docs/figures/table2_full_image.png" alt="Full Image Results" width="700"/>
</p>

### With Bounding Box Evaluation (Table 3)

<p align="center">
  <img src="docs/figures/table3_with_bbox.png" alt="Bounding Box Results" width="700"/>
</p>

### Per-Style Performance (Table 4)

Models perform best on **Naskh** and **Ruq'ah** (standardized strokes), while **Diwani** and **Thuluth** (ornate scripts with dense ligatures) remain challenging.

---

## Project Structure

```
DuwatBench/
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
├── data/
│   ├── images/                    # Calligraphy images
│   └── duwatbench.jsonl          # Dataset manifest
├── src/
│   ├── evaluate.py               # Main evaluation script
│   ├── models/
│   │   ├── __init__.py
│   │   └── model_wrapper.py      # Model implementations
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── evaluation_metrics.py # CER, WER, chrF, etc.
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py        # Dataset loading
│   │   └── arabic_normalization.py
│   └── config/
│       ├── __init__.py
│       ├── eval_config.py
│       └── api_keys.example.py
├── scripts/
│   ├── download_data.sh
│   └── run_all_evaluations.sh
├── results/                       # Evaluation outputs
└── docs/                          # GitHub Pages website
    ├── index.html
    └── assets/
```

---

## Adding New Models

To evaluate a new model, implement the `BaseModel` interface:

```python
from src.models.model_wrapper import BaseModel

class MyNewModel(BaseModel):
    def __init__(self):
        super().__init__("my-new-model")
        # Initialize your model

    def transcribe(self, image, prompt=None):
        # Return Arabic text transcription
        return "النص العربي"
```

Then register it in `model_wrapper.py`:

```python
model_map = {
    # ... existing models ...
    "my-new-model": lambda: MyNewModel(),
}
```

---

## Citation

If you use DuwatBench in your research, please cite our paper:

```bibtex
@article{duwatbench2025,
  title={DuwatBench: Bridging Language and Visual Heritage through an Arabic Calligraphy Benchmark for Multimodal Understanding},
  author={Anonymous},
  journal={ACL},
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
- Community repositories: [Calligraphy Qalam](https://calligraphyqalam.com/), [Free Islamic Calligraphy](https://freeislamiccalligraphy.com/)
- Arabic NLP tools: [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools)

---

## Contact

For questions or issues, please:
- Open an issue on [GitHub](https://github.com/mbzuai-oryx/DuwatBench/issues)
- Contact the authors at: duwatbench@mbzuai.ac.ae

---

<p align="center">
  Made with ❤️ at <a href="https://mbzuai.ac.ae">MBZUAI</a>
</p>
