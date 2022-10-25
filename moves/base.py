from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from numpy import ndarray


@dataclass
class Move(ABC):
    position: ndarray
    quality: float
    name: str

    @abstractmethod
    def get_neighbours(self) -> List['Move']:
        ...
