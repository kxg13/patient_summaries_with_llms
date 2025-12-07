#!/bin/bash
# Ablation 1: Training data size (25 vs 50 vs 100 examples)
# Ablation 2: Prompt engineering (default vs alternative prompt)

set -e

echo "=========================================="
echo "ABLATION STUDIES"
echo "=========================================="

# Setup
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
export PYTHONPATH="${REPO_ROOT}"
cd "${REPO_ROOT}"

DATA_DIR="${REPO_ROOT}/hallucination_data"
OUTPUT_DIR="${REPO_ROOT}/outputs"
model="meta-llama/Llama-2-7b-hf"
data_path="${DATA_DIR}/cleaned"

mkdir -p ${OUTPUT_DIR}/ablations

# ABLATION 1: Train with 25 and 50 examples
# (100 examples is already done in main replication)

echo ""
echo "=========================================="
echo "ABLATION 1a: Training with 25 examples"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python summarization/fine_tune_llama.py \
    --model_name_or_path ${model} \
    --data_path ${data_path} \
    --output_path ${OUTPUT_DIR}/ablations/data_size_25 \
    --device cuda \
    --max_steps 25 \
    --save_and_logging_steps 25 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --num_target_modules 2 \
    --learning_rate 2e-5 \
    --num_train_examples 25 \
    --num_val_examples 100 \
    --num_test_examples 100

echo ""
echo "=========================================="
echo "ABLATION 1b: Training with 50 examples"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python summarization/fine_tune_llama.py \
    --model_name_or_path ${model} \
    --data_path ${data_path} \
    --output_path ${OUTPUT_DIR}/ablations/data_size_50 \
    --device cuda \
    --max_steps 25 \
    --save_and_logging_steps 25 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --num_target_modules 2 \
    --learning_rate 2e-5 \
    --num_train_examples 50 \
    --num_val_examples 100 \
    --num_test_examples 100

echo ""
echo "=========================================="
echo "ABLATION 2: Alternative Prompt"
echo "=========================================="

# Create modified version with alternative prompt
cp summarization/fine_tune_llama.py summarization/fine_tune_llama_alt_prompt.py

# Modify the prompt (uncomment lines 222-224, comment lines 219-220)
sed -i '219,220s/^/# /' summarization/fine_tune_llama_alt_prompt.py
sed -i '222,224s/^# //' summarization/fine_tune_llama_alt_prompt.py

CUDA_VISIBLE_DEVICES=0 python summarization/fine_tune_llama_alt_prompt.py \
    --model_name_or_path ${model} \
    --data_path ${data_path} \
    --output_path ${OUTPUT_DIR}/ablations/alt_prompt \
    --device cuda \
    --max_steps 25 \
    --save_and_logging_steps 25 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --num_target_modules 2 \
    --learning_rate 2e-5 \
    --num_train_examples 100 \
    --num_val_examples 100 \
    --num_test_examples 100

# Clean up temporary file
rm summarization/fine_tune_llama_alt_prompt.py

echo ""
echo "=========================================="
echo "ALL ABLATIONS COMPLETE!"
echo "Results saved to: ${OUTPUT_DIR}/ablations/"
echo "=========================================="
