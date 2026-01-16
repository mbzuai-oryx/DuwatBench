#!/usr/bin/env python3
"""
DuwatBench: Arabic Calligraphy Benchmark for Multimodal Understanding
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="duwatbench",
    version="1.0.0",
    author="MBZUAI",
    author_email="duwatbench@mbzuai.ac.ae",
    description="Arabic Calligraphy Benchmark for Multimodal Understanding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mbzuai-oryx/DuwatBench",
    project_urls={
        "Bug Tracker": "https://github.com/mbzuai-oryx/DuwatBench/issues",
        "Documentation": "https://mbzuai-oryx.github.io/DuwatBench/",
        "Source Code": "https://github.com/mbzuai-oryx/DuwatBench",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "open-source": [
            "transformers>=4.35.0",
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "accelerate>=0.24.0",
            "easyocr>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "duwatbench-eval=evaluate:main",
        ],
    },
    include_package_data=True,
    keywords=[
        "arabic",
        "calligraphy",
        "ocr",
        "benchmark",
        "multimodal",
        "vision-language",
        "nlp",
    ],
)
