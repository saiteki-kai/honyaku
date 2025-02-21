from abc import abstractmethod


class QualityMetric:
    """Abstract class for quality metrics that can be used to score translations."""

    @abstractmethod
    def score(
        self,
        hypotheses: list[str],
        contexts: list[str],
        references: list[str] | None = None,
        batch_size: int = 32,
    ) -> list[float] | float:
        """Abstract method to score a batch of hypotheses, contexts and optionally references."""
        raise NotImplementedError
