#!/bin/bash

# Generate responses for BeaverTails evaluation dataset with different LLMs

DATASET_NAME="PKU-Alignment/BeaverTails-Evaluation"
DATASET_SPLIT="test"

models=(
    # Gemma
    "google/gemma-2-27b-it"
    "google/gemma-2-9b-it"
    # Phi
    "microsoft/phi-4"
    "microsoft/Phi-3.5-MoE-instruct"
    "microsoft/Phi-3-medium-128k-instruct"
    # Qwen
    "Qwen/Qwen2.5-72B-Instruct"
    "Qwen/Qwen2.5-32B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    # Mistral
    "mistralai/Mistral-Small-24B-Instruct-2501"
    "mistralai/Mistral-Nemo-Instruct-2407"
)

for model in "${models[@]}"; do
    python run_response_generation.py \
        --model-name "$model" \
        --dataset-name $DATASET_NAME \
        --split $DATASET_SPLIT \
        --field "prompt" \
        --dtype "bfloat16" \
        --output-path "./outputs" \
        --log-file "./logs/responses/${model}.log"
done
