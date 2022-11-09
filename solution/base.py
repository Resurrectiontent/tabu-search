from dataclasses import dataclass

from numpy import ndarray

from solution.id import SolutionId


@dataclass
class Solution:
    id: SolutionId
    position: ndarray
    # TODO: introduce a separate class for solution quality
    quality: float
