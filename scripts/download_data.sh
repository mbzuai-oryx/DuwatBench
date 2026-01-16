#!/bin/bash
# Download DuwatBench dataset from Hugging Face

echo "==================================="
echo "DuwatBench Dataset Downloader"
echo "==================================="

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface-cli..."
    pip install huggingface_hub
fi

# Create data directory
mkdir -p data/images

# Download dataset
echo "Downloading DuwatBench dataset..."
huggingface-cli download MBZUAI/DuwatBench --local-dir ./data --repo-type dataset

echo ""
echo "Download complete!"
echo "Dataset location: ./data/"
echo ""
echo "Files:"
ls -la ./data/
