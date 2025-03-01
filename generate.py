import argparse
import logging
import logging.handlers
import os
import sys
import time
import typing

from pathlib import Path
from typing import Callable

import datasets
import torch
import transformers

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    set_seed,
)
from transformers.utils import is_flash_attn_2_available


type Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")

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

    model_config = GenerationConfig.from_pretrained(args.model_name)
    logger.info(f"BOS token ID: {model_config.bos_token_id}")
    logger.info(f"EOS token ID: {model_config.eos_token_id}")
    logger.info(f"PAD token ID: {model_config.pad_token_id}")

    config = GenerationConfig(
        do_sample=False,
        max_length=args.max_length,
        max_new_tokens=args.max_length,
        bos_token_id=model_config.bos_token_id,
        eos_token_id=model_config.eos_token_id,
        pad_token_id=model_config.pad_token_id,
        cache_implementation=model_config.cache_implementation,
        repetition_penalty=model_config.repetition_penalty,
    )

    logger.info("Generation started")
    start_time = time.perf_counter()

    responses = []
    for prompt in tqdm(dataset[args.field], desc="Generating responses"):
        response = generate(model, tokenizer, prompt, config=config)
        responses.append(response)

    end_time = time.perf_counter()
    logger.info(f"Generation finished. Took {end_time - start_time:.2f} seconds")

    dataset = dataset.add_column("response", responses)  # type: ignore  # noqa: PGH003
    dataset.to_parquet(str(out_filepath))

    logger.info(f"Dataset saved to {out_filepath}")


def load_model(model_name_or_path: str, dtype: torch.dtype = torch.bfloat16) -> tuple[PreTrainedModel, Tokenizer]:
    """Load model and tokenizer from Hugging Face Hub"""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa",
        device_map="auto",
        trust_remote_code=True,
    )

    return model, tokenizer


@torch.inference_mode()
def generate(  # noqa: PLR0913
    model: PreTrainedModel | Callable[[PreTrainedModel], PreTrainedModel],
    tokenizer: Tokenizer,
    prompt: str | list[str],
    config: GenerationConfig,
    include_prompt: bool = False,
    skip_special_tokens: bool = True,
) -> str | list[str]:
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    input_ids = typing.cast(torch.Tensor, input_ids)

    generated_ids = model.generate(input_ids.to(model.device), generation_config=config)

    if include_prompt:
        outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=skip_special_tokens)
    else:
        generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(input_ids, generated_ids)]
        outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=skip_special_tokens)

    return outputs[0]


def hf_name_to_path(model_name: str) -> str:
    """Converts huggingface name to a path-safe string."""
    return model_name.replace("/", "__")


def load_data(
    dataset_name_or_path: Path | str,
    split: str,
    config_name: str | None = None,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    """Load dataset from huggingface or local."""
    if isinstance(dataset_name_or_path, Path):
        ds = Dataset.from_parquet(str(dataset_name_or_path))
        return typing.cast(Dataset, ds)

    return load_dataset(dataset_name_or_path, name=config_name, split=split)


def setup_logging(logger: logging.Logger, log_file: Path) -> None:
    LOG_FORMAT = "[%(asctime)s] %(levelname)s: [%(process)s] %(name)s: %(message)s"

    if not log_file.exists():
        log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)

    def log_unhandled_exceptions(exc_type, exc_value, exc_traceback):
        logger.error("Unhandled exception occurred", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = log_unhandled_exceptions


def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses for a dataset using an Hugging Face model")
    parser.add_argument("--model-name", type=str, required=True, help="Model name or path")
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset name or path")
    parser.add_argument("--split", type=str, required=True, help="Split name of the dataset")
    parser.add_argument("--config-name", type=str, required=False, default=None, help="Config name of the dataset")
    parser.add_argument("--field", type=str, required=True, help="Input text field")
    parser.add_argument("--dtype", type=str, required=False, default="bfloat16", help="Dtype of the model")
    parser.add_argument("--log-file", type=Path, default="logs/translation.log", help="Path to the log file")
    parser.add_argument("--output-path", type=Path, default="outputs", help="Path to the output directory")
    parser.add_argument("--seed", type=int, required=False, default=42, help="Seed for reproducibility")
    parser.add_argument("--max-length", type=int, required=False, default=2048, help="Max number of tokens to generate")

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
