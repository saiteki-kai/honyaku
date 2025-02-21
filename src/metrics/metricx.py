import sys
import typing

from pathlib import Path

import torch

from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BatchEncoding, T5Tokenizer
from transformers.data.data_collator import DataCollatorWithPadding

from src.metrics.metric import QualityMetric


# add the git submodule to the path
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir / "metricx"))

try:
    from metricx24.models import MT5ForRegression
except ImportError as e:
    raise RuntimeError(f"Failed to import metricx24: {e}") from e


class MetricX24(QualityMetric):
    """MetricX24 metric for quality evaluation of translations."""

    _model: MT5ForRegression
    _tokenizer: T5Tokenizer

    def __init__(self, model_name_or_path: str, tokenizer_name: str) -> None:
        """Load the MetricX24 metric.

        This function loads the MetricX24 metric from a given name or path.

        Parameters
        ----------
        model_name_or_path : str
            the name or path to the model from huggingface
        tokenizer_name : str
            the name of the tokenizer from huggingface
        device : str, optional
            the device to use, by default "cuda"
        """
        self._model = MT5ForRegression.from_pretrained(model_name_or_path, torch_dtype="auto", device_map="auto")
        self._model.eval()

        self._tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)

    @torch.inference_mode()
    def score(
        self,
        hypotheses: list[str],
        contexts: list[str],
        references: list[str] | None = None,
        batch_size: int = 32,
        smart_batching: bool = True,
    ) -> float | list[float]:
        """Score a batch of hypotheses, contexts and optionally references.

        If references are not provided, the score is computed for each hypothesis-context pair.

        Parameters
        ----------
        hypotheses : list[str]
            list of translations
        contexts : list[str]
            list of source texts
        references : list[str] | None, optional
            list of reference translations, by default None
        batch_size : int, optional
            batch size, by default 32

        Returns
        -------
        list[float] | float
            list of scores or a single score
        """
        inputs = self._prepare_inputs(hypotheses, contexts, references)

        data = self._tokenize(inputs)
        data = typing.cast(dict[str, list[int]], data)
        data["original_index"] = list(range(len(list(data["input_ids"]))))

        dataset = Dataset.from_dict(dict(data))

        if smart_batching:
            dataset = dataset.map(lambda example: {"length": len(example["input_ids"])})  # add lengths
            dataset = dataset.sort("length", reverse=True)

        data_collator = DataCollatorWithPadding(tokenizer=self._tokenizer, padding=True, pad_to_multiple_of=8)

        dataloader = DataLoader(
            dataset,  # type: ignore  # noqa: PGH003
            batch_size=batch_size,
            collate_fn=data_collator,
            num_workers=2,
            pin_memory=True,
            shuffle=False,
        )

        device = next(self._model.parameters()).device
        scores = torch.zeros(len(list(data["original_index"])), device=device) if smart_batching else []

        for batch in tqdm(dataloader):
            indices = batch["original_index"]
            batch = {k: v.to(device) for (k, v) in batch.items() if k not in ["original_index", "length"]}

            output = self._model(**batch)
            results = output.predictions.detach()

            if smart_batching:
                scores[indices] = results
            else:
                scores = typing.cast(list[torch.Tensor], scores)
                scores.append(results)

        if not smart_batching:
            scores = typing.cast(list[torch.Tensor], scores)
            scores = torch.cat(scores)

        scores = typing.cast(torch.Tensor, scores)

        return scores.cpu().tolist()

    def _tokenize(self, text: str | list[str]) -> BatchEncoding:
        """Tokenizes the input text.

        Tokenizes the input text using the tokenizer and removes the EOS token from the end of each sequence.
        The maximum input length is set to 1536 which is the length used during training.

        Parameters
        ----------
        text : str | list[str]
            A string or list of strings to tokenize

        Returns
        -------
        BatchEncoding
            A BatchEncoding object containing the input ids and the attention mask
        """
        tokens = self._tokenizer(text, max_length=1536, truncation=True, padding=False)

        # Remove EOS token from the end of each sequence
        tokens["input_ids"] = [ids[:-1] for ids in tokens.input_ids]
        tokens["attention_mask"] = [mask[:-1] for mask in tokens.attention_mask]

        return tokens

    def _prepare_inputs(
        self,
        hypotheses: list[str],
        contexts: list[str],
        references: list[str] | None = None,
    ) -> list[str]:
        """Prepares the inputs for the model."""
        if references is None:
            inputs = [self._prepare_input(src, hyp) for src, hyp in zip(contexts, hypotheses)]
        else:
            inputs = [self._prepare_input(src, hyp, ref) for src, hyp, ref in zip(contexts, hypotheses, references)]

        return inputs

    def _prepare_input(self, source: str, hypothesis: str, reference: str | None = None) -> str:
        """Prepares the input for the model."""
        if reference is None:
            return f"source: {source} candidate: {hypothesis}"

        return f"source: {source} candidate: {hypothesis} reference: {reference}"


if __name__ == "__main__":
    import numpy as np

    model = MetricX24(model_name_or_path="google/metricx-24-hybrid-xxl-v2p6", tokenizer_name="google/mt5-xl")

    scores = model.score(
        ["this is a test", "sentence to translate", "random text"],
        ["questa è una prova", "frase tradurre", "testo casuale"],
        ["questo è un test", "frase da tradurre", "testo a caso"],
        batch_size=3,
    )

    print(f"Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
