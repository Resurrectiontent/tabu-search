from abc import ABC

from convergence.base import ConvergenceCriterion


class AggregatedConvergence(ConvergenceCriterion, ABC):
    def __init__(self, *args):
        self.criteria = [c for c in args if issubclass(c, ConvergenceCriterion)]

    def _convergence_results(self, **kwargs):
        return [c.converged(**kwargs) for c in self.criteria]


class ConjunctiveConvergence(AggregatedConvergence):
    def __init__(self, *args):
        """
        Initializes conjunctive Convergence. Converges, when all memory from *args converge.
        :param args: Convergences to account.
        """
        super().__init__(*args)

    def converged(self, **kwargs):
        """
        Checks, whether all the memory converged.
        :param kwargs: arguments for memory convergence.
        :return: True, if all memory converged, otherwise, False.
        """
        return all(self._convergence_results(**kwargs))


class DisjunctiveConvergence(AggregatedConvergence):
    def __init__(self, *args):
        """
        Initializes disjunctive convergence. Converges, when any memory from *args converged.
        :param args: Convergences to account.
        """
        super().__init__(*args)

    def converged(self, **kwargs) -> bool:
        """
        Checks, whether any criterion converged.
        :param kwargs: arguments for memory convergence.
        :return: True, if at least one criterion converged, otherwise, False.
        """
        return any(self._convergence_results(**kwargs))
