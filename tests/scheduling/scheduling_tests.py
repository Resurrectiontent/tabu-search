from functools import partial

import numpy as np
import pytest
from matplotlib import pyplot as plt

from scheduling.order_neighbourhood import variable_partitioning_order_neighbourhood
from scheduling.resource_neighbourhood import variable_partitioning_resource_neighbourhood
from solution.quality.lib import custom_metric
from tabusearch.mutation import create_custom_mutation
from tabusearch import TabuSearch
from tabusearch.solution.base import Solution
from tests.scheduling.fixtures import setup_schedule, setup_toolbox, setup_wg, setup_contractors, setup_worker_pool


def test_order_optimisation(setup_schedule, setup_toolbox):
    init_schedule = setup_toolbox.schedule_to_chromosome(schedule=setup_schedule)

    optimiser = TabuSearch([create_custom_mutation('VP_ord', partial(variable_partitioning_order_neighbourhood,
                                                                     is_order_correct=setup_toolbox.is_order_correct)),
                            create_custom_mutation('Same', lambda x: [(x, '')])],
                           metric=custom_metric('time-res', setup_toolbox.evaluate2, minimized=True),
                           max_iter=50,
                           tabu_time=10,)
    s = optimiser.optimize(init_schedule)
    h: list[Solution] = optimiser._history
    print('\n', s.position)
    print(s.quality)
    plt.plot(np.linspace(0, 1, len(h)), np.array([float(x.quality) for x in h]))
    plt.show()


def test_res_optimisation(setup_schedule, setup_toolbox):
    init_schedule = setup_toolbox.schedule_to_chromosome(schedule=setup_schedule)

    optimiser = TabuSearch([create_custom_mutation('VP_res', partial(variable_partitioning_resource_neighbourhood,
                                                                     worker_reqs=setup_toolbox.get_worker_reqs())),
                            create_custom_mutation('Same', lambda x: [(x, '')])],
                           metric=custom_metric('time-res', setup_toolbox.evaluate2, minimized=True),
                           max_iter=500,
                           tabu_time=10,)
    s = optimiser.optimize(init_schedule)
    h: list[Solution] = optimiser._history
    print('\n', s.position)
    print(s.quality)
    plt.plot(np.linspace(0, 1, len(h)), np.array([float(x.quality) for x in h]))
    plt.show()