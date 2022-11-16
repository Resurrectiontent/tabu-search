from numpy import NAN

from convergence.base import ConvergenceCriterion
from solution.quality.single import SolutionQualityInfo


# TODO: introduce WindowedEnhancementConvergence to account several latest enhancements
#  or just add an optional argument `window` to EnhancementConvergence

class EnhancementConvergence(ConvergenceCriterion):
    def __init__(self, min_enhancement: float):
        self.min_enhancement = min_enhancement
        self.last_quality = NAN

    def converged(self, new_result: SolutionQualityInfo, **kwargs):
        last = self.last_quality
        self.last_quality = float(new_result)
        return 0 <= self.last_quality - last <= self.min_enhancement


class IterativeConvergence(ConvergenceCriterion):
    def __init__(self, max_iter: int):
        assert max_iter > 0, 'max_iter should be > 0'

        self.max_iter = max_iter
        self.iter = 0

    def converged(self, **kwargs):
        self.iter += 1
        return self.max_iter > self.iter
