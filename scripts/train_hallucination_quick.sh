#!/bin/bash
# Trains on Original, Cleaned, and Cleaned & Improved datasets

set -e

echo "=========================================="
echo "HALLUCINATION EXPERIMENTS - QUICK VERSION"
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

# Create data directories with proper structure
echo "Setting up data directories..."
mkdir -p ${DATA_DIR}/{original,cleaned,cleaned_improved}
mkdir -p ${OUTPUT_DIR}/{original,cleaned,cleaned_improved}

# Create symlinks for original dataset
cd ${DATA_DIR}/original
ln -sf ${PHYSIO_DATA}/hallucinations_mimic_di_original.json train_4000_600_chars_251-350_pt.json
ln -sf ${PHYSIO_DATA}/hallucinations_mimic_di_validation_original.json valid_4000_600_chars.json

# Create symlinks for cleaned dataset
cd ${DATA_DIR}/cleaned
ln -sf ${PHYSIO_DATA}/hallucinations_mimic_di_cleaned.json train_4000_600_chars_251-350_pt.json
ln -sf ${PHYSIO_DATA}/hallucinations_mimic_di_validation_cleaned.json valid_4000_600_chars.json

# Create symlinks for cleaned & improved dataset
cd ${DATA_DIR}/cleaned_improved
ln -sf ${PHYSIO_DATA}/hallucinations_mimic_di_cleaned_improved.json train_4000_600_chars_251-350_pt.json
ln -sf ${PHYSIO_DATA}/hallucinations_mimic_di_validation_cleaned_improved.json valid_4000_600_chars.json

cd "${REPO_ROOT}"

# Model and training parameters
model="meta-llama/Llama-2-7b-hf"
max_steps="100"
save_steps="100"
batch_size="1"
gradient_accumulation_steps="16"
lora_rank="8"
lora_alpha="32"
lora_dropout="0.1"
num_target_modules="2"
learning_rate="2e-5"
num_train_examples="100"
num_val_examples="100"
num_test_examples="100"

# Train Model 1: Original
echo ""
echo "=========================================="
echo "Training 1/3: Original (hallucination-containing)"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python summarization/fine_tune_llama.py \
    --model_name_or_path ${model} \
    --data_path ${DATA_DIR}/original \
    --output_path ${OUTPUT_DIR}/original \
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

# Train Model 2: Cleaned
echo ""
echo "=========================================="
echo "Training 2/3: Cleaned (hallucinations removed)"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python summarization/fine_tune_llama.py \
    --model_name_or_path ${model} \
    --data_path ${DATA_DIR}/cleaned \
    --output_path ${OUTPUT_DIR}/cleaned \
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

# Train Model 3: Cleaned & Improved
echo ""
echo "=========================================="
echo "Training 3/3: Cleaned & Improved"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python summarization/fine_tune_llama.py \
    --model_name_or_path ${model} \
    --data_path ${DATA_DIR}/cleaned_improved \
    --output_path ${OUTPUT_DIR}/cleaned_improved \
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
echo "REPLICATION COMPLETE!"
echo "Results saved to: ${OUTPUT_DIR}/"
echo "=========================================="
