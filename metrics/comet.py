from comet import download_model, load_from_checkpoint
from comet.models import CometModel

from metrics.metric import QualityMetric


class CometMetric(QualityMetric):
    _model: CometModel

    def __init__(self, model_name: str) -> None:
        checkpoint_path = download_model(model_name)
        self._model = load_from_checkpoint(checkpoint_path)

        self._model.eval()
        self._model.set_embedding_cache()

    def score(
        self,
        hypotheses: list[str],
        contexts: list[str],
        references: list[str] | None = None,
        batch_size: int = 32,
    ) -> list[float]:
        if references is None:
            data = [{"src": src, "mt": mt} for src, mt in zip(contexts, hypotheses, strict=True)]
        else:
            data = [
                {"src": src, "mt": mt, "ref": ref}
                for src, mt, ref in zip(contexts, hypotheses, references, strict=True)
            ]

        outputs = self._model.predict(data, batch_size=batch_size)

        return outputs["scores"]
