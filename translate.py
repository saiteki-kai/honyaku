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

from honyaku.data import hf_name_to_path, load_data
from honyaku.inference import generate, load_model
from honyaku.logger import setup_logging
from honyaku.translation import pre_process


logger = logging.getLogger(__name__)


def post_process(output: CompletionOutput) -> str:
    if output.finish_reason == "length":
        logger.warning("Finished due to length")
        logger.warning(output.text)

    return output.text.strip()


def main(args: argparse.Namespace) -> None:
    logger.info(f"Arguments: {args}")

    logger.info(f"Loading model {args.model_name}")
    model = load_model(args.model_name, args.dtype, args.ngpus)
    logger.info("Model loaded")

    out_filepath = (
        args.output_path
        / args.dataset_name.split("/")[-1]
        / "translations"
        / hf_name_to_path(args.model_name)
        / f"{args.split}.parquet"
    )

    if out_filepath.exists():
        dataset = load_data(out_filepath, config_name=args.config_name, split=args.split)
    else:
        dataset = load_data(args.dataset_name, config_name=args.config_name, split=args.split)

    dataset = typing.cast(Dataset, dataset)

    if args.field not in dataset.features:
        logger.error(f"Field '{args.field}' does not exist")
        sys.exit(1)

    if args.field.endswith("_it"):
        logger.warning(f"Field '{args.field}' already exists. Skipping translation.")
        sys.exit(0)

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

    # Save the dataset with the translated field added
    df[f"{args.field}_it"] = translations
    it_dataset = Dataset.from_pandas(df)
    it_dataset.to_parquet(str(out_filepath))

    logger.info(f"Dataset saved to {out_filepath}")


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
    parser.add_argument("--output-path", type=Path, default="outputs", help="Path to the output directory")

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
