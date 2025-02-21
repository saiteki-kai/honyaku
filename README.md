# Machine Translation

This repository contains code for automatically translating a dataset from English to Italian using different machine translation models and evaluating the quality of the translations using reference-free quality metrics.

## Models

Translation models:

- [X-ALMA-13B-Group2](https://huggingface.co/haoranxu/X-ALMA-13B-Group2)
- [TowerInstruct-Mistral-7B-v0.2](https://huggingface.co/Unbabel/TowerInstruct-Mistral-7B-v0.2)
- [aya-23-35B](https://huggingface.co/CohereForAI/aya-23-35B)
- [nllb-200-3.3B](https://huggingface.co/facebook/nllb-200-3.3B)
- [nllb-moe-54B](https://huggingface.co/facebook/nllb-moe-54B)
- [LLaMAX3-8B-Alpaca](https://huggingface.co/LLaMAX/LLaMAX3-8B-Alpaca)

Quality metrics:

- [MetricX-24-Hybrid-XXL](https://huggingface.co/google/metricx-24-hybrid-xxl-v2p6-bfloat16)
- [XCOMET-XXL](https://huggingface.co/Unbabel/XCOMET-XXL)
- [COMETKIWI-23-XXL](https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xxl)

## Requirements

- Python >= 3.12
- PyTorch >= 2.5.1
- vLLM >= 0.7.3

You can install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

or [`poetry`](https://python-poetry.org/):

```bash
poetry install
```

## Usage

In the `./scripts/` folder, there are predefined scripts to run multiple configurations in sequence for the same dataset.

If you want to run a single configuration, you can use the `run_translation.py`, `run_translation_quality.py` and `run_response_generation.py` scripts.

### Translation

The translation script can be used to translate a dataset from English to Italian using an Hugging Face model. The outputs are saved in a Parquet file named `./outputs/<dataset_name>/translations/<model_name>/<split>.parquet`.

```bash
python run_translation.py \
    --model-name "haoranxu/X-ALMA-13B-Group2" \
    --dataset-name "PKU-Alignment/BeaverTails" \
    --split "330k_train" \
    --field "prompt" \
    --dtype "bfloat16" \
    --ngpus 2 \
    --output-path "./outputs/" \
    --log-file "./logs/translation.log"
```

### Quality Evaluation

The configuration files for the quality evaluation can be found in the `configs/quality_evaluation` directory. Each configuration file specifies the list of translation models, the dataset, and the quality metric to be used for the evaluation.

```bash
python run_translation_quality.py \
    --config ./configs/quality_evaluation/beavertails_it_metricx-24.yaml \
    --log-file ./logs/evaluation_metricx-24.log
```

The outputs are saved in a Parquet file named `./outputs/<dataset_name>/scores/<metric>/<model_name>/<split>/scores.parquet`.

### Response Generation

The response generation script can be used to generate responses for a dataset using an Hugging Face model. The generated responses are saved in a Parquet file named `./outputs/<dataset_name>/responses/<model_name>/<split>.parquet`.

```bash
python run_response_generation.py \
    --model-name Qwen/Qwen2.5-72B-Instruct \
    --dataset-name PKU-Alignment/BeaverTails-Evaluation \
    --split test \
    --field "prompt" \
    --dtype "bfloat16" \
    --output-path "./outputs/" \
    --log-file "./logs/responses/${model}.log"
```
