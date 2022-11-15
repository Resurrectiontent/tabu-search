from dataclasses import dataclass

from numpy import ndarray

from solution.id import SolutionId
from solution.quality.base import BaseSolutionQualityInfo


@dataclass
class Solution:
    id: SolutionId
    position: ndarray
    quality: BaseSolutionQualityInfo
