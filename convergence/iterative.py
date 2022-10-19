from convergence.base import Convergence


class IterativeConvergence(Convergence):
    def __init__(self, max_iter: int):
        assert max_iter > 0, 'max_iter should be > 0'

        self.max_iter = max_iter
        self.iter = 0

    def converged(self):
        self.iter += 1
        return self.max_iter > self.iter
