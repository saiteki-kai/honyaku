import torch

from comet import download_model, load_from_checkpoint
from comet.models import CometModel

from metrics.metric import QualityMetric


class CometMetric(QualityMetric):
    _model: CometModel

    def __init__(self, model_name: str) -> None:
        """Load a Comet model.

        This function loads a Comet model from a given name or path.

        Parameters
        ----------
        model_name : str
            name or path to the model from huggingface
        """
        checkpoint_path = download_model(model_name)
        self._model = load_from_checkpoint(checkpoint_path)

        self._model.eval()
        self._model.set_embedding_cache()

    @torch.inference_mode()
    def score(
        self,
        hypotheses: list[str],
        contexts: list[str],
        references: list[str] | None = None,
        batch_size: int = 32,
    ) -> list[float]:
        """Score a batch of hypotheses, contexts and optionally references.

        If references are not provided, the score is computed for each hypothesis-context pair
        assuming a reference-free metric is used.

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
        data = self._prepare_inputs(hypotheses, contexts, references)
        outputs = self._model.predict(data, batch_size=batch_size)

        return outputs["scores"]

    def _prepare_inputs(
        self,
        hypotheses: list[str],
        contexts: list[str],
        references: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """Prepares the inputs for the model."""
        if references is None:
            data = [{"src": src, "mt": mt} for src, mt in zip(contexts, hypotheses, strict=True)]
        else:
            data = [
                {"src": src, "mt": mt, "ref": ref}
                for src, mt, ref in zip(contexts, hypotheses, references, strict=True)
            ]

        return data
