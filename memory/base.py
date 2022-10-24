from abc import ABC, abstractmethod

from numpy import ndarray


class MemoryCriterion(ABC):
    @abstractmethod
    def satisfied(self, x: ndarray) -> bool:
        ...
