import typing

from pathlib import Path

from datasets import Dataset, concatenate_datasets


def read_dataset_split(path: Path, split_name: str, include_scores: bool = False) -> Dataset:
    score_suffix = "_scores" if include_scores else ""

    dataset = Dataset.from_parquet(str(path / split_name / f"{split_name.split('_')[-1]}{score_suffix}.parquet"))

    return typing.cast(Dataset, dataset)


def read_dataset(path: Path, split: str | list[str], include_scores: bool = False) -> Dataset:
    if isinstance(split, list):
        dsets = [read_dataset_split(path, split_name, include_scores=include_scores) for split_name in split]
        dataset = concatenate_datasets(dsets)
    else:
        dataset = read_dataset_split(path, split, include_scores=include_scores)

    return dataset
