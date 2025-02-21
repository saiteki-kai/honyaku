import typing

import numpy as np

from datasets import Dataset
from pandas import DataFrame
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def dataset_statistics(
    dataset: Dataset, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, fields: list[str]
) -> DataFrame:
    statistics = dataset.map(
        lambda example: {f"{field}_token_len": len(tokenizer.tokenize(example[field])) for field in fields},
    )
    statistics_df = typing.cast(DataFrame, statistics.to_pandas())

    for field in fields:
        field_lengths = statistics_df[f"{field}_token_len"]
        compute_stats(field, list(field_lengths))

    return statistics_df


def compute_stats(title: str, lengths: list) -> None:
    stats = {
        "min": np.min(lengths),
        "max": np.max(lengths),
        "mean": np.mean(lengths),
        "std": np.std(lengths),
    }

    print(title)
    print("-" * len(title))

    for key, value in stats.items():
        print(f"{key:<20}: {value:.2f}")

    print()
