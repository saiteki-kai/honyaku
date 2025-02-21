#!/bin/bash

python generate.py \
    --model_name "google/gemma-27b-it" \
    --dataset_name "PKU-Alignment/BeaverTails-Evaluation" \
    --split "test" \
    --field "prompt" \
    --dtype "bfloat16" \
    --output_path "outputs" \
    --log_file "logs/responses/gemma-27b-it.log"

python generate.py \
    --model_name "microsoft/phi-4" \
    --dataset_name "PKU-Alignment/BeaverTails-Evaluation" \
    --split "test" \
    --field "prompt" \
    --dtype "bfloat16" \
    --output_path "outputs" \
    --log_file "logs/responses/phi-4.log"
