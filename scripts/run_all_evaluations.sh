#!/bin/bash
# Run evaluations for all models on DuwatBench

echo "==================================="
echo "DuwatBench Full Evaluation Suite"
echo "==================================="

# Create results directory
mkdir -p results

# Open-source models
OPEN_MODELS=(
    "EasyOCR"
    "trocr-base-arabic-handwritten"
    # Add more as needed - these require GPU:
    # "Llava-v1.6-mistral-7b-hf"
    # "InternVL3-8B"
    # "Qwen2.5-VL-7B"
    # "Qwen2.5-VL-72B-Instruct"
    # "gemma-3-27b-it"
    # "MBZUAI/AIN"
)

# Closed-source models (require API keys)
CLOSED_MODELS=(
    "gemini-2.5-flash"
    "gpt-4o-mini"
    "gpt-4o"
    "claude-sonnet-4.5"
    "gemini-1.5-flash"
)

# Evaluation modes
MODES=("full_image" "with_bbox")

echo ""
echo "Starting evaluations..."
echo ""

# Evaluate open-source models
echo "=== Open-Source Models ==="
for model in "${OPEN_MODELS[@]}"; do
    for mode in "${MODES[@]}"; do
        echo ""
        echo "Evaluating: $model ($mode)"
        echo "---"
        python src/evaluate.py --model "$model" --mode "$mode" --resume
    done
done

# Evaluate closed-source models (if API keys are set)
echo ""
echo "=== Closed-Source Models ==="
if [[ -n "$GEMINI_API_KEY" || -n "$OPENAI_API_KEY" || -n "$ANTHROPIC_API_KEY" ]]; then
    for model in "${CLOSED_MODELS[@]}"; do
        for mode in "${MODES[@]}"; do
            echo ""
            echo "Evaluating: $model ($mode)"
            echo "---"
            python src/evaluate.py --model "$model" --mode "$mode" --resume
        done
    done
else
    echo "Skipping closed-source models (no API keys set)"
    echo "Set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY to evaluate"
fi

echo ""
echo "==================================="
echo "Evaluation Complete!"
echo "Results saved to: results/"
echo "==================================="
