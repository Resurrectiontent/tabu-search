from abc import ABC, abstractmethod
from typing import List

from numpy import ndarray


class Move(ABC):
    @abstractmethod
    def move(self) -> List[ndarray]:
        ...
