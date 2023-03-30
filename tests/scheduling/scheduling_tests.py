from operator import attrgetter

import pickle
from uuid import uuid4
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
    setup_toolbox, setup_wg, setup_contractors, setup_worker_pool, setup_base_optimisers


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
                           convergence_criterion=100,
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
                           convergence_criterion=500,
                           tabu_time=10, )
    s = optimiser.optimize(init_schedule)
    h: list[Solution] = optimiser._history
    print('\n', s.position)
    print(s.quality)
    plt.plot(np.arange(len(h)), np.array([float(x.quality) for x in h]))
    plt.show()


def test_order_res_optimisation(setup_schedule_genetic, setup_toolbox, setup_base_optimisers, setup_wg):
    t0 = time()
    init_schedule: ChromosomeType = setup_toolbox.schedule_to_chromosome(schedule=setup_schedule_genetic)
    init_quality = setup_toolbox.evaluate_time_res(init_schedule)
    same_mutation = create_custom_mutation('Same', lambda x: [(x, '')])

    wg_name, *_ = setup_wg
    move_selection, optimiser_ord, optimiser_res = setup_base_optimisers

    optimiser_ord = optimiser_ord(
        mutation_behaviour=[create_custom_mutation('VP_ord', partial(variable_partitioning_order_neighbourhood,
                                                                     is_order_correct=setup_toolbox.is_order_correct)),
                            create_custom_mutation('VP_shfl', partial(variable_partitioning_order_shuffle,
                                                                      is_order_correct=setup_toolbox.is_order_correct)),
                            create_custom_mutation('shfl', partial(order_shuffle,
                                                                   is_order_correct=setup_toolbox.is_order_correct)),
                            same_mutation],
        metric=custom_metric('time-res', setup_toolbox.evaluate_time_res, minimized=True))

    optimiser_res = optimiser_res(
        mutation_behaviour=[create_custom_mutation('VP_res', partial(variable_partitioning_resource_neighbourhood,
                                                                     worker_reqs=setup_toolbox.get_worker_reqs())),
                            same_mutation],
        metric=custom_metric('time-res', setup_toolbox.evaluate_time_res, minimized=True))

    t1 = time()
    s_ord = optimiser_ord.optimize(init_schedule)
    t2 = time()
    s_res = optimiser_res.optimize(s_ord.position)
    t3 = time()

    history2 = lambda opt: [(s.quality.value, str(s.id)) for s in opt]
    data = {'works': init_schedule[0].size,
            'wg_name': wg_name,
            'move_selection': move_selection,
            'ord': [history2(optimiser_ord._history), history2(list(optimiser_ord.hall_of_fame))],
            'res': [history2(optimiser_res._history), history2(list(optimiser_res.hall_of_fame))],
            't': [t0, t1, t2, t3]}
    with open(f'results/{uuid4()}.pickle', 'wb') as pkl:
        pickle.dump(data, pkl)

    print(f'Initialization took {t1 - t0:.3f} sec'
          f'\nOrder optimisation took {t2 - t1:.3f} sec ({len(optimiser_ord._history)} moves,'
          f' {init_quality - s_ord.quality.value} quality enhancement)'
          f'\nResource optimisation took {t3 - t2:.3f} sec ({len(optimiser_res._history)} moves,'
          f' {s_ord.quality.value - s_res.quality.value} quality enhancement)'
          f'\nTotal objective enhancement {init_quality-s_res.quality.value}')

    # # TODO: remove "-", fix quality history
    history = lambda opt: [s.quality.value for s in opt._history]
    h_ord: list[float] = history(optimiser_ord)
    h_res: list[float] = history(optimiser_res)

    plt.plot([0], [init_quality], 'ro')
    plt.plot(np.arange(len(h_ord) + 1), [init_quality] + h_ord, 'b-')
    plt.plot(np.arange(len(h_ord), len(h_ord) + len(h_res)), h_res, 'g-')
    plt.title(f'WG - {wg_name}; Selection - {move_selection}')
    plt.show()
    # pass
