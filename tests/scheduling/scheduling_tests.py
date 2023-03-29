from operator import attrgetter

from functools import partial
from time import time

import numpy as np
import pytest
from matplotlib import pyplot as plt

from sampo.scheduler.genetic.converter import ChromosomeType
from sampo.scheduler.genetic.operators import copy_chromosome
from scheduling.order_neighbourhood import order_shuffle, variable_partitioning_order_neighbourhood, \
    variable_partitioning_order_shuffle
from scheduling.resource_neighbourhood import variable_partitioning_resource_neighbourhood
from solution.quality.lib import custom_metric
from tabusearch.mutation import create_custom_mutation
from tabusearch import TabuSearch
from tabusearch.solution.base import Solution
from tests.scheduling.fixtures import setup_schedule_heft, setup_schedule_genetic, setup_schedule, \
    setup_toolbox, setup_wg, setup_contractors, setup_worker_pool


def test_order_optimisation(setup_schedule, setup_toolbox):
    init_schedule = setup_toolbox.schedule_to_chromosome(schedule=setup_schedule)

    # noinspection PyTypeChecker
    optimiser = TabuSearch([create_custom_mutation('VP_ord', partial(variable_partitioning_order_neighbourhood,
                                                                     is_order_correct=setup_toolbox.is_order_correct)),
                            create_custom_mutation('VP_shfl', partial(variable_partitioning_order_shuffle,
                                                                      is_order_correct=setup_toolbox.is_order_correct)),
                            create_custom_mutation('shfl', partial(order_shuffle,
                                                                   is_order_correct=setup_toolbox.is_order_correct)),
                            create_custom_mutation('Same', lambda x: [(copy_chromosome(x), '')])
                            ],
                           metric=custom_metric('time', setup_toolbox.evaluate_time_res, minimized=True),
                           max_iter=100,
                           tabu_time=10, )
    s = optimiser.optimize(init_schedule)
    h: list[Solution] = optimiser._history
    print('\n', '\n'.join([str(i.id) for i in optimiser._history]))
    print(s.quality)
    plt.plot(np.arange(len(h)), np.array([float(x.quality) for x in h]))
    plt.show()


def test_res_optimisation(setup_schedule_heft, setup_toolbox):
    init_schedule = setup_toolbox.schedule_to_chromosome(schedule=setup_schedule_heft)

    optimiser = TabuSearch([create_custom_mutation('VP_res', partial(variable_partitioning_resource_neighbourhood,
                                                                     worker_reqs=setup_toolbox.get_worker_reqs())),
                            create_custom_mutation('Same', lambda x: [(x, '')])],
                           metric=custom_metric('time-res', setup_toolbox.evaluate_time_res, minimized=True),
                           max_iter=500,
                           tabu_time=10, )
    s = optimiser.optimize(init_schedule)
    h: list[Solution] = optimiser._history
    print('\n', s.position)
    print(s.quality)
    plt.plot(np.arange(len(h)), np.array([float(x.quality) for x in h]))
    plt.show()


def test_schedule_optimisation(setup_schedule_genetic, setup_toolbox):
    t0 = time()
    init_schedule: ChromosomeType = setup_toolbox.schedule_to_chromosome(schedule=setup_schedule_genetic)
    init_quality = setup_toolbox.evaluate_time_res(init_schedule)
    same_mutation = create_custom_mutation('Same', lambda x: [(x, '')])

    optimiser_ord = TabuSearch([create_custom_mutation('VP_ord', partial(variable_partitioning_order_neighbourhood,
                                                                     is_order_correct=setup_toolbox.is_order_correct)),
                                create_custom_mutation('VP_shfl', partial(variable_partitioning_order_shuffle,
                                                                          is_order_correct=setup_toolbox.is_order_correct)),
                                create_custom_mutation('shfl', partial(order_shuffle,
                                                                       is_order_correct=setup_toolbox.is_order_correct))],
                               metric=custom_metric('time-res', setup_toolbox.evaluate_time_res, minimized=True),
                               max_iter=50,
                               tabu_time=10, )

    optimiser_res = TabuSearch([create_custom_mutation('VP_res', partial(variable_partitioning_resource_neighbourhood,
                                                                         worker_reqs=setup_toolbox.get_worker_reqs())),
                                same_mutation],
                               metric=custom_metric('time-res', setup_toolbox.evaluate_time_res, minimized=True),
                               max_iter=500,
                               tabu_time=20, )
    t1 = time()
    s_ord = optimiser_ord.optimize(init_schedule)
    t2 = time()
    s_res = optimiser_res.optimize(s_ord.position)
    t3 = time()

    print(f'Initialization took {t1 - t0:.3f} sec'
          f'\nOrder optimisation took {t2 - t1:.3f} sec ({len(optimiser_ord._history)} moves,'
          f' {float(s_ord.quality) - init_quality} quality enhancement)'
          f'\nResource optimisation took {t3 - t2:.3f} sec ({len(optimiser_res._history)} moves,'
          f' {float(s_res.quality) - float(s_ord.quality)} quality enhancement)'
          f'\nTotal objective enhancement {float(s_ord.quality) - init_quality}')

    # TODO: remove "-", fix quality history
    history = lambda opt: [-float(s.quality) for s in opt._history]
    h_ord: list[float] = history(optimiser_ord)
    h_res: list[float] = history(optimiser_res)

    plt.plot([0], [init_quality], 'ro')
    plt.plot(np.arange(len(h_ord)), h_ord, 'b-')
    plt.plot(np.arange(len(h_ord), len(h_ord) + len(h_res)), h_res, 'g-')
    plt.show()
    pass
