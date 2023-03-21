from dataclasses import dataclass

from numpy import ndarray

from tabusearch.solution.id import SolutionId
from tabusearch.solution.quality.base import BaseSolutionQualityInfo


@dataclass
class Solution:
    id: SolutionId
    position: ndarray
    quality: BaseSolutionQualityInfo

    def __repr__(self):
        return f'{{{self.id}}}:{self.quality}: {self.position}'
