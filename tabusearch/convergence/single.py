from collections import deque

from tabusearch.convergence.base import ConvergenceCriterion
from tabusearch.solution.quality.single import SolutionQualityInfo


class EnhancementConvergence(ConvergenceCriterion):
    def __init__(self, min_enhancement: float, window: int = 1):
        assert isinstance(window, int), f'Window size should be an integer value, was {type(window)}.'
        assert window > 0, f'Window size should be at least 1, was {window}.'

        self.min_enhancement = min_enhancement
        self.quality_history = deque(maxlen=window)

    def converged(self, new_result: SolutionQualityInfo, **kwargs):
        new = float(new_result)
        self.quality_history.append(new)
        res = max(self.quality_history) - min(self.quality_history) <= self.min_enhancement \
            if len(self.quality_history) == self.quality_history.maxlen \
            else False
        return res


class IterativeConvergence(ConvergenceCriterion):
    def __init__(self, max_iter: int):
        assert max_iter > 0, 'max_iter should be > 0'

        self.max_iter = max_iter
        self.iter = 0

    def converged(self, **kwargs):
        self.iter += 1
        return self.max_iter <= self.iter
