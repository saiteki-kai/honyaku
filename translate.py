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
from vllm import CompletionOutput, SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer

from log import setup_logging
from utils import apply_chat_template, generate, load_data, load_model


logger = logging.getLogger(__name__)


def post_process(output: CompletionOutput) -> str:
    if output.finish_reason == "length":
        logger.warning("Finished due to length")
        logger.warning(output.text)

    return output.text.strip()


def translation_prompt(input_text: str, model_name: str, src_lang: str = "English", trg_lang: str = "Italian") -> str:
    if "X-ALMA" in model_name:
        return f"Translate this from {src_lang} to {trg_lang}:\n{src_lang}: {input_text}\n{trg_lang}:"
    elif "TowerInstruct" in model_name:
        return f"Translate the following text from {src_lang} into {trg_lang}.\n{src_lang}: {input_text}.\n{trg_lang}:"
    elif "LLaMAX3" in model_name:
        instruction = f"Translate the following sentences from {src_lang} to {trg_lang}."

        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n"
            f"### Instruction:\n{instruction}\n"
            f"### Input:\n{input_text}\n### Response:"
        )

    return f"Translate this from {src_lang} to {trg_lang}:\n{src_lang}: {input_text}\n{trg_lang}:"


def pre_process(
    model_name: str,
    src_lang: str = "English",
    trg_lang: str = "Italian",
) -> typing.Callable[[list[str] | str, AnyTokenizer], list[str]]:
    def _pre_process(text: list[str] | str, tokenizer: AnyTokenizer) -> list[str]:
        if isinstance(text, str):
            text = [text]

        prompts = [translation_prompt(t, model_name, src_lang, trg_lang) for t in text]

        if "LLaMAX3" in model_name:
            return prompts

        return apply_chat_template(prompts, tokenizer)

    return _pre_process


def main(args: argparse.Namespace) -> None:
    logger.info(f"Arguments: {args}")

    logger.info(f"Loading model {args.model_name}")
    model = load_model(args.model_name, args.dtype, args.ngpus)
    logger.info("Model loaded")

    dataset = load_data(args.dataset_name, args.split, config_name=args.config_name)
    dataset = typing.cast(Dataset, dataset)

    if args.field not in dataset.features:
        logger.error(f"Field '{args.field}' does not exist")
        sys.exit(1)

    df = dataset.to_pandas()
    df = typing.cast(pd.DataFrame, df)

    params = SamplingParams(temperature=0.0, max_tokens=1024)

    logger.info("Inference started")
    start_time = time.perf_counter()
    translations = generate(
        model,
        dataset[args.field],
        params=params,
        preprocess=pre_process(model_name=model_name, src_lang="English", trg_lang="Italian"),
        postprocess=post_process,
    )
    end_time = time.perf_counter()
    logger.info(f"Inference finished. Took {end_time - start_time:.2f} seconds")

    df[f"{args.field}_it"] = translations
    it_dataset = Dataset.from_pandas(df)

    model_dir = model_name.split("/")[-1]
    dataset_dir = dataset_name.split("/")[-1] + "-it"
    filename = split.split("_")[-1]
    it_dataset.to_parquet(args.output_path / dataset_dir / model_dir / split / f"{filename}.parquet")

    logger.info(f"Dataset saved to {args.output_path / dataset_dir / model_dir / split}")


def parse_args():
    parser = argparse.ArgumentParser(description="Translate a dataset with an Hugging Face model")
    parser.add_argument("--model-name", type=str, required=True, help="Model name or path")
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset name or path")
    parser.add_argument("--split", type=str, required=True, help="Split name of the dataset")
    parser.add_argument("--config-name", type=str, required=False, default=None, help="Config name of the dataset")
    parser.add_argument("--field", type=str, required=True, help="Field to translate")
    parser.add_argument("--dtype", type=str, required=False, default="bfloat16", help="Dtype of the model")
    parser.add_argument("--ngpus", type=int, required=False, default=1, help="Number of GPUs to use")
    parser.add_argument("--log-file", type=Path, default="logs/translation.log", help="Path to the log file")

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

    # TODO: arguments parsing

    model_name = "LLaMAX/LLaMAX3-8B-Alpaca"
    dtype = "float16"

    dataset_name = "PKU-Alignment/BeaverTails"
    split = "330k_train"

    main(args)
