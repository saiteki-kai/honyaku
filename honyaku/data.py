from pathlib import Path
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset


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
        return load_dataset("parquet", data_dir=str(dataset_name_or_path.parent), name=config_name, split=split)

    return load_dataset(dataset_name_or_path, name=config_name, split=split)
