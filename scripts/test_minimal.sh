#!/bin/bash
# MINIMAL TEST
# Test that everything works before running full experiments

set -e

echo "=========================================="
echo "MINIMAL TEST (2 min)"
echo "=========================================="

# Setup
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
export PYTHONPATH="${REPO_ROOT}"
cd "${REPO_ROOT}"

# Data paths
PHYSIO_DATA="${REPO_ROOT}/physionet.org/files/ann-pt-summ/1.0.0/derived_datasets"
DATA_DIR="${REPO_ROOT}/hallucination_data"
OUTPUT_DIR="${REPO_ROOT}/outputs"

# Create data directories
echo "Setting up data directories..."
mkdir -p ${DATA_DIR}/original
mkdir -p ${OUTPUT_DIR}/test_minimal

# Create symlinks
cd ${DATA_DIR}/original
ln -sf ${PHYSIO_DATA}/hallucinations_mimic_di_original.json train_4000_600_chars_251-350_pt.json
ln -sf ${PHYSIO_DATA}/hallucinations_mimic_di_validation_original.json valid_4000_600_chars.json

cd "${REPO_ROOT}"

# Model and parameters
model="meta-llama/Llama-2-7b-hf"
max_steps="2"  # Original: 100
save_steps="2"  # Original: 100
batch_size="1"
gradient_accumulation_steps="16"
lora_rank="8"
lora_alpha="32"
lora_dropout="0.1"
num_target_modules="2"
learning_rate="2e-5"
num_train_examples="5"  # Original: 100,
num_val_examples="5"    # Original: 100,
num_test_examples="5"   # Original: 100

echo ""
echo "MINIMAL TEST: 2 steps, 5 examples"
echo ""

CUDA_VISIBLE_DEVICES=0 python summarization/fine_tune_llama.py \
    --model_name_or_path ${model} \
    --data_path ${DATA_DIR}/original \
    --output_path ${OUTPUT_DIR}/test_minimal \
    --device cuda \
    --max_steps ${max_steps} \
    --save_and_logging_steps ${save_steps} \
    --batch_size ${batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --num_target_modules ${num_target_modules} \
    --learning_rate ${learning_rate} \
    --num_train_examples ${num_train_examples} \
    --num_val_examples ${num_val_examples} \
    --num_test_examples ${num_test_examples}

echo ""
echo "=========================================="
echo "TEST COMPLETE!"
echo "Check: ${OUTPUT_DIR}/test_minimal/"
echo "=========================================="
