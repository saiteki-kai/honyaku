from metrics.comet import CometMetric
from metrics.metric import QualityMetric
from metrics.metricx import MetricX24


def load_quality_metric(model_name_or_path: str, tokenizer_name: str | None = None) -> QualityMetric:
    """Load a quality metric model

    This function loads a quality metric model from a given name or path. The
    model can be a Comet model or a MetricX24 model.

    Parameters
    ----------
    model_name_or_path : str
        name or path to the quality metric model from huggingface.
    tokenizer_name : str | None, optional
        name of the tokenizer from huggingface, by default None

    Returns
    -------
    QualityMetric
        The loaded quality metric model

    Raises
    ------
    ValueError
        If the specified model is not available
    """
    if "comet" in model_name_or_path:
        return CometMetric(model_name_or_path)

    if "metricx-24" in model_name_or_path:
        if tokenizer_name is None:
            tokenizer_name = "google/mt5-xl"

        return MetricX24(model_name_or_path, tokenizer_name=tokenizer_name)

    raise ValueError(f"Unknown quality metric: {model_name_or_path}")
