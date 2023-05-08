from operator import attrgetter

from matplotlib import pyplot as plt
from numpy import random as np_rnd
from sortedcontainers import SortedList

from tests.function_optimisation.functions import rosenbrock, styblinski_tang
from tabusearch.solution.quality.lib.aggregated import normalized_weighted_metrics_aggregation
from tabusearch.solution.quality.lib.complex import complex_metric
from tabusearch.solution.factory import SolutionFactory
from tabusearch.memory.evaluating.gradient import GradientAccelerator
from tabusearch.mutation.base import MutationBehaviour
from tabusearch.mutation.neighbourhood import NearestNeighboursMutation, FullAxisShiftMutation
from tabusearch.solution.quality.lib import custom_metric


def test_gradient_accelerator():

    ga = GradientAccelerator()

    function = rosenbrock
    x0 = np_rnd.randint(-10, 10, 10, int)
    mutations: list[MutationBehaviour] = [NearestNeighboursMutation(), FullAxisShiftMutation()]
    metric = custom_metric(function.__name__, function, minimized=True)
    solution_factory = SolutionFactory(metric)
    # noinspection PyTypeChecker
    solution_factory.quality_factory.add_evaluation_layer(ga, metrics_aggregation=complex_metric(
                    normalized_weighted_metrics_aggregation('Add eval agg', [0.8, 0.2])))

    history = []
    x = solution_factory.initial(x0)
    history.append(x)
    ga.memorize(x)
    for i in range(100):
        neighbourhood = [(behaviour.mutation_type, behaviour.mutate(x)) for behaviour in mutations]
        solutions = solution_factory(neighbourhood)
        sorted_solutions = SortedList(solutions, key=attrgetter('quality'))
        x = sorted_solutions[-1]
        history.append(x)
        ga.memorize(x)

    plt.plot(list(range(len(history))), [i.quality.value for i in history])
    plt.show()


def test_gradient_no_gradient_comparison():

    ga = GradientAccelerator()

    function = styblinski_tang
    x0 = np_rnd.randint(-10, 10, 10, int)
    mutations: list[MutationBehaviour] = [NearestNeighboursMutation(), FullAxisShiftMutation()]
    metric = custom_metric(function.__name__, function, minimized=True)
    solution_factory = SolutionFactory(metric)
    # noinspection PyTypeChecker
    solution_factory.quality_factory.add_evaluation_layer(ga, metrics_aggregation=complex_metric(
                    normalized_weighted_metrics_aggregation('Add eval agg', [0.8, 0.2])))
    solution_factory_no_grad = SolutionFactory(metric)

    def opt(factory, grad_accel=None):
        x = factory.initial(x0)
        best_x, best_i = x, -1
        history = [x]
        if grad_accel:
            grad_accel.memorize(x)
        for i in range(100):
            neighbourhood = [(behaviour.mutation_type, behaviour.mutate(x)) for behaviour in mutations]
            solutions = factory(neighbourhood)
            sorted_solutions = SortedList(solutions, key=attrgetter('quality'))
            x = sorted_solutions[-1]
            history.append(x)
            if grad_accel and x.quality.main > best_x.quality.main or x.quality > best_x.quality:
                best_x, best_i = x, i
            if grad_accel:
                grad_accel.memorize(x)

        return best_x, best_i, history

    grad_results = opt(solution_factory, ga)
    no_grad_results = opt(solution_factory_no_grad)

    print()
    print(f'no grad - [{no_grad_results[1]}]: {no_grad_results[0].quality.value}')
    print(f'with grad - [{grad_results[1]}]: {grad_results[0].quality.value}')

    fig, ax = plt.subplots()

    ax.plot(list(range(len(no_grad_results[-1]))), [i.quality.value for i in no_grad_results[-1]], 'b-')
    ax.plot(list(range(len(grad_results[-1]))), [i.quality.value for i in grad_results[-1]], 'g-')

    fig.show()
