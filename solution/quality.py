from abc import ABC, abstractmethod


class SolutionQuality(ABC):
    def __lt__(self, other) -> bool:
        # Can only compare solution qualities of same types!
        assert isinstance(other, type(self))
        return float(self) < float(other)

    @abstractmethod
    def __float__(self):
        """
        Float representation of th solution quality.
        Comparable with float representations of solution quality objects of same types.
        """
        ...

    @abstractmethod
    def __str__(self):
        """
        String exploration of the solution quality
        """
        ...

# TODO Introduce SingleSolutionQualityMetric anc multi-metric versions
