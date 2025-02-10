from abc import abstractmethod


class QualityMetric:
    @abstractmethod
    def score(
        self,
        hypotheses: list[str],
        contexts: list[str],
        references: list[str] | None = None,
        batch_size: int = 32,
    ) -> list[float] | float:
        raise NotImplementedError
