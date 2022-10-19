from abc import ABC, abstractmethod


class Convergence(ABC):
    @abstractmethod
    def converged(self, *args):
        ...
