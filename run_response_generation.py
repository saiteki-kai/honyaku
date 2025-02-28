import argparse
import logging
import logging.handlers
import sys
import time
import typing

from pathlib import Path

import datasets
import pandas as pd
import transformers

from datasets import Dataset
from transformers import GenerationConfig, set_seed

from src.data import hf_name_to_path, load_data
from src.inference import AnyTokenizer, apply_chat_template
from src.inference.transformers import generate, load_model
from src.logger import setup_logging


logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    logger.info(f"Arguments: {args}")

    set_seed(args.seed)

    logger.info(f"Loading model {args.model_name}")
    model, tokenizer = load_model(args.model_name, args.dtype)
    logger.info("Model loaded")

    out_filepath = (
        args.output_path
        / args.dataset_name.split("/")[-1]
        / "responses"
        / hf_name_to_path(args.model_name)
        / f"{args.split}.parquet"
    )

    if out_filepath.exists():
        dataset = load_data(out_filepath, config_name=args.config_name, split=args.split)
        logger.info(f"Dataset loaded from {out_filepath}")
    else:
        dataset = load_data(args.dataset_name, config_name=args.config_name, split=args.split)
        logger.info(f"Dataset loaded from huggingface ({args.dataset_name})")

    dataset = typing.cast(Dataset, dataset)

    if args.field not in dataset.features:
        logger.error(f"Field '{args.field}' does not exist")
        sys.exit(1)

    df = dataset.to_pandas()
    df = typing.cast(pd.DataFrame, df)

    config = GenerationConfig(do_sample=False, max_new_tokens=1024)

    # TODO: refactor to support file input (e.g. [Path | str] argument or system_prompt_path)
    system_prompt = args.system_prompt
    prompts = dataset[args.field]

    logger.info("Generation started")
    start_time = time.perf_counter()
    responses = generate(
        model,
        tokenizer,
        prompts,
        config=config,
        preprocess=pre_process(system_prompt=system_prompt),
        postprocess=post_process,
        batch_size=args.batch_size,
    )
    end_time = time.perf_counter()
    logger.info(f"Generation finished. Took {end_time - start_time:.2f} seconds")

    # TODO: define the output schema (prompt, response, category) or just the response
    # Save the dataset with the responses
    df["response"] = responses
    out_dataset = Dataset.from_pandas(df)
    out_dataset.to_parquet(str(out_filepath))

    logger.info(f"Dataset saved to {out_filepath}")


def pre_process(system_prompt: str | None = None) -> typing.Callable[[list[str] | str, AnyTokenizer], list[str]]:
    def _pre_process(text: list[str] | str, tokenizer: AnyTokenizer) -> list[str]:
        if isinstance(text, str):
            text = [text]

        return apply_chat_template(text, tokenizer, system_prompt=system_prompt)

    return _pre_process


def post_process(output: str) -> str:
    return output.strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses for a dataset using an Hugging Face model")
    parser.add_argument("--model-name", type=str, required=True, help="Model name or path")
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset name or path")
    parser.add_argument("--split", type=str, required=True, help="Split name of the dataset")
    parser.add_argument("--config-name", type=str, required=False, default=None, help="Config name of the dataset")
    parser.add_argument("--field", type=str, required=True, help="Input text field")
    parser.add_argument("--dtype", type=str, required=False, default="bfloat16", help="Dtype of the model")
    parser.add_argument("--system-prompt", type=str, default=None, help="System prompt for the model")
    parser.add_argument("--log-file", type=Path, default="logs/translation.log", help="Path to the log file")
    parser.add_argument("--output-path", type=Path, default="outputs", help="Path to the output directory")
    parser.add_argument("--seed", type=int, required=False, default=42, help="Seed for reproducibility")
    parser.add_argument("--batch-size", type=int, required=False, default=8, help="Batch size for generation")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    setup_logging(logger, args.log_file)
    transformers.utils.logging.set_verbosity_info()
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    main(args)
