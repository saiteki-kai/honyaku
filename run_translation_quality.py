import argparse
import logging
import time
import typing

from pathlib import Path

import numpy as np
import yaml

from datasets import Dataset, load_dataset

from src.data import hf_name_to_path
from src.logger import setup_logging
from src.metrics.quality import load_quality_metric


logger = logging.getLogger(__name__)


def main(config: dict) -> None:
    metric = config["metric"]
    tokenizer_name = config.get("tokenizer")
    dataset_name = config["dataset"]["name"]
    translators = config["translators"]
    splits = config["dataset"]["splits"]
    fields = config["dataset"]["fields"]

    logger.info(f"Metric: {metric}")
    logger.info(f"Tokenizer: {tokenizer_name}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Translators: {translators}")
    logger.info(f"Splits: {splits}")
    logger.info(f"Fields: {fields}")

    translators = [hf_name_to_path(translator) for translator in translators]

    scores_path = Path("outputs") / hf_name_to_path(dataset_name) / "scores"
    metric_path = scores_path / hf_name_to_path(metric)

    logger.info(f"Loading metric: {metric}")
    metric_model = load_quality_metric(metric, tokenizer_name=tokenizer_name)
    logger.info(f"Loaded metric: {metric}")

    for translator in translators:
        for split in splits:
            split_path = metric_path / translator / split
            score_path = split_path / "scores.parquet"

            if score_path.exists():
                logger.info(f"Skipping {translator} for {metric} as {score_path} already exists")
                score_dataset = Dataset.from_parquet(str(score_path))
                continue
            score_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Loading split: {split} for translator: {translator}")
            dataset = load_split_dataset(dataset_name, translator, split)

            score_dict = {}
            for field in fields:
                score_field_path = split_path / f"{field['source']}_{hf_name_to_path(metric)}.npy"

                if score_field_path.exists():
                    logger.info(f"Skipping {field['source']} for {metric} as {score_field_path} already exists")
                    scores = np.load(score_field_path)
                    score_dict[f"{field['source']}_{metric}"] = scores
                    continue

                sources = dataset[field["source"]]
                hypothesis = dataset[field["hypothesis"]]

                logger.info(f"Scoring for field: {field['source']}")
                start_time = time.perf_counter()
                scores = metric_model.score(hypothesis, sources, batch_size=field["batch_size"])
                time_elapsed = time.perf_counter() - start_time
                logger.info(f"Scoring complete for field: {field['source']} in {time_elapsed:.2f} seconds")

                np.save(score_field_path, scores)
                score_dict[f"{field['source']}_{metric}"] = scores

            score_dataset = Dataset.from_dict(score_dict)

            logger.info(f"Saving split: {split} for translator: {translator}")
            score_dataset.to_parquet(score_path)


# TODO: allow for local parquet files
def load_split_dataset(dataset_name: str, config_name: str, split: str) -> Dataset:
    ds = load_dataset(dataset_name, config_name, split=split)

    return typing.cast(Dataset, ds)


def parse_args():
    parser = argparse.ArgumentParser(description="Run quality evaluation on a dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--log-file", type=Path, default="logs/evaluation.log", help="Path to the log file")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    return parser.parse_args()


def load_config(config_path: Path | str) -> dict:
    if not isinstance(config_path, Path):
        config_path = Path(config_path)

    with config_path.open("r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    args = parse_args()
    setup_logging(logger, args.log_file)
    config = load_config(args.config)
    main(config)
