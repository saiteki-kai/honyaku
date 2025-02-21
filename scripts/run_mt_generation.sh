#!/bin/bash

# Translate BeaverTails dataset to Italian with different SOTA models

declare -A aya=(
    ["name"]="CohereForAI/aya-23-35B"
    ["ngpus"]=2
)
declare -A llamax=(
    ["name"]="LLaMAX/LLaMAX3-8B-Alpaca"
    ["ngpus"]=1
)
declare -A xalma=(
    ["name"]="haoranxu/X-ALMA-13B-Group2"
    ["ngpus"]=1
)
declare -A towerinstruct=(
    ["name"]="Unbabel/TowerInstruct-Mistral-7B-v0.2"
    ["ngpus"]=1
)
declare -A nllb=(
    ["name"]="facebook/nllb-moe-54b"
    ["ngpus"]=2
)

models=("aya" "llamax" "xalma" "towerinstruct" "nllb")

datasets=("PKU-Alignment/BeaverTails-Evaluation" "PKU-Alignment/BeaverTails" "PKU-Alignment/BeaverTails")
splits=("test" "330k_train" "330k_test")

fields=("prompt" "response")

for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    split="${splits[$i]}"

    for model_ref in "${models[@]}"; do
        declare -n model=$model_ref
        name=${model["name"]}
        ngpus=${model["ngpus"]}

        for field in "${fields[@]}"; do
            echo "Translating ${field} of \"${dataset}\" [${split}] with \"${name}\" on ${ngpus} GPUs"

            python translate.py \
                --model-name "$name" \
                --dataset-name $dataset \
                --split $split \
                --field $field \
                --dtype "bfloat16" \
                --ngpus $ngpus \
                --output-path "./outputs/" \
                --log-file "./logs/translation/${dataset}_${split}_${field}_${name//\//__}.log"
        done
    done
done
