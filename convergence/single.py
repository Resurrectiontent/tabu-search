from convergence.base import ConvergenceCriterion


class EnhancementConvergence(ConvergenceCriterion):
    def __init__(self, min_enhancement: float):
        self.min_enhancement = min_enhancement

    def converged(self, enhancement, **kwargs):
        return enhancement < self.min_enhancement


class IterativeConvergence(ConvergenceCriterion):
    def __init__(self, max_iter: int):
        assert max_iter > 0, 'max_iter should be > 0'

        self.max_iter = max_iter
        self.iter = 0

    def converged(self, **kwargs):
        self.iter += 1
        return self.max_iter > self.iter
