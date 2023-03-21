from abc import ABC, abstractmethod


class ConvergenceCriterion(ABC):
    @abstractmethod
    def converged(self, **kwargs) -> bool:
        ...
