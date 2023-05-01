from typing import Generic

from dataclasses import dataclass, field

from tabusearch.solution.id import SolutionId
from tabusearch.solution.quality.base import BaseSolutionQualityInfo
from tabusearch.typing_ import TData


@dataclass
class Solution(Generic[TData]):
    id: SolutionId
    position: TData
    quality: BaseSolutionQualityInfo

    def __repr__(self):
        return f'{{{self.id}}}:{self.quality}: {self.position}'
