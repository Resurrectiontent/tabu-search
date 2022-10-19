from convergence.base import Convergence


class EnhancementConvergence(Convergence):
    def __init__(self, min_enhancement: float):
        self.min_enhancement = min_enhancement

    def converged(self, enhancement):
        return enhancement < self.min_enhancement
