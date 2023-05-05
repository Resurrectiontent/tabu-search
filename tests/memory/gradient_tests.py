from itertools import chain
from operator import attrgetter

from sortedcontainers import SortedList

from solution.quality.lib.aggregated import normalized_weighted_metrics_aggregation
from solution.quality.lib.complex import complex_metric
from tests.function_optimisation.functions import rosenbrock
from tests.function_optimisation.classical_optimisation_tests import setup_func_x0
from tabusearch.solution.factory import SolutionFactory
from tabusearch.memory import GradientAccelerator
from tabusearch.mutation.base import MutationBehaviour
from tabusearch.mutation.neighbourhood import NearestNeighboursMutation, FullAxisShiftMutation
from tabusearch.solution.quality.lib import custom_metric


def test_gradient_accelerator():

    ga = GradientAccelerator()

    function = rosenbrock
    class Request:
        param = function
    x0 = setup_func_x0(Request)['x0']
    mutations: list[MutationBehaviour] = [NearestNeighboursMutation(), FullAxisShiftMutation()],
    metric = custom_metric(function.__name__, function, minimized=True)
    solution_factory = SolutionFactory(metric)
    # noinspection PyTypeChecker
    solution_factory.quality_factory.add_evaluation_layer(ga, metrics_aggregation=complex_metric(
                    normalized_weighted_metrics_aggregation('Add eval agg', [0.75, 0.25])))

    ga.memorize(x0)
    x = x0
    for i in range(5):
        neighbourhood = [(behaviour.mutation_type, behaviour.mutate(x)) for behaviour in mutations]
        solutions = solution_factory(neighbourhood)
        sorted_solutions = SortedList(solutions, key=attrgetter('quality'))
        x = sorted_solutions[-1]
        ga.memorize(x)

