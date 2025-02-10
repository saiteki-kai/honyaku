import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BatchEncoding, T5Tokenizer
from transformers.data.data_collator import DataCollatorWithPadding

from metrics.metric import QualityMetric


# add the git submodule to the path
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir / "metricx"))

try:
    from metricx24.models import MT5ForRegression
except ImportError as e:
    raise RuntimeError(f"Failed to import metricx24: {e}") from e


class MetricX24(QualityMetric):
    _model: MT5ForRegression
    _tokenizer: T5Tokenizer

    def __init__(self, model_name_or_path: str, tokenizer_name: str, device: str = "cuda") -> None:
        self._device = device

        self._model = MT5ForRegression.from_pretrained(model_name_or_path, torch_dtype="auto")
        self._model.to(device)
        self._model.eval()

        self._tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)

    def _prepare_input(self, source: str, hypothesis: str, reference: str | None = None) -> str:
        if reference is None:
            return f"source: {source} candidate: {hypothesis}"

        return f"source: {source} candidate: {hypothesis} reference: {reference}"

    def _tokenize(self, text: str | list[str]) -> BatchEncoding:
        # INFO: The model was trained with a maximum input length of 1536
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
        if references is None:
            inputs = [self._prepare_input(src, hyp) for src, hyp in zip(contexts, hypotheses)]
        else:
            inputs = [self._prepare_input(src, hyp, ref) for src, hyp, ref in zip(contexts, hypotheses, references)]

        return inputs

    @torch.no_grad()
    def score(
        self,
        hypotheses: list[str],
        contexts: list[str],
        references: list[str] | None = None,
        batch_size: int = 32,
    ) -> float | list[float]:
        inputs = self._prepare_inputs(hypotheses, contexts, references)
        data = self._tokenize(inputs)
        dataset = Dataset.from_dict(dict(data))

        data_collator = DataCollatorWithPadding(tokenizer=self._tokenizer, padding=True, pad_to_multiple_of=8)

        dataloader = DataLoader(
            dataset,  # type: ignore
            batch_size=batch_size,
            collate_fn=data_collator,
            num_workers=2,
            pin_memory=True,
            shuffle=False,
        )

        scores = []

        start_time = time.perf_counter()

        for batch in tqdm(dataloader):
            output = self._model(**batch.to(self._device))
            scores.extend(output.predictions.detach().cpu().numpy())

        print(f"Time: {time.perf_counter() - start_time:.2f} s")

        return scores


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
