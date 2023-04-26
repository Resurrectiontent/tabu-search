from copy import deepcopy

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
from tabusearch.mutation import create_custom_mutation, same_mutation
from tabusearch import TabuSearch
from tabusearch.solution.base import Solution
from tests.scheduling.fixtures import setup_schedule_topo, setup_schedule_heft, setup_schedule_genetic, setup_schedule,\
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
    with open(f'scheduling/results/o-r-{wg_name}_{move_selection}.pickle', 'wb') as pkl:
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


def test_res_order_optimisation(setup_schedule_genetic, setup_toolbox, setup_base_optimisers, setup_wg):
    t0 = time()
    init_schedule: ChromosomeType = setup_toolbox.schedule_to_chromosome(schedule=setup_schedule_genetic)
    init_quality = setup_toolbox.evaluate_time_res(init_schedule)

    wg_name, *_ = setup_wg
    move_selection, optimiser_ord_base, optimiser_res_base = setup_base_optimisers

    optimiser_ord = init_order_optimiser(deepcopy(optimiser_ord_base), setup_toolbox)
    optimiser_res = init_resource_optimiser(deepcopy(optimiser_res_base), setup_toolbox)

    t1 = time()
    s_res = optimiser_res.optimize(init_schedule)
    t2 = time()
    s_ord = optimiser_ord.optimize(s_res.position)
    t3 = time()

    history2 = lambda opt: [(s and s.quality.value, str(s and s.id)) for s in opt]
    data = {'works': init_schedule[0].size,
            'wg_name': wg_name,
            'move_selection': move_selection,
            'ord': [history2(optimiser_ord._history), history2(list(optimiser_ord.hall_of_fame))],
            'res': [history2(optimiser_res._history), history2(list(optimiser_res.hall_of_fame))],
            't': [t0, t1, t2, t3]}
    f_name = f'scheduling/results/r_o-{wg_name}'
    with open(f'{f_name}.pickle', 'wb') as pkl:
        pickle.dump(data, pkl)

    print(f'Initialization took {t1 - t0:.3f} sec'
          f'\nResource optimisation took {t2 - t1:.3f} sec ({len(optimiser_res._history)} moves,'
          f' {init_quality - s_res.quality.value} quality enhancement)'
          f'\nOrder optimisation took {t3 - t2:.3f} sec ({len(optimiser_ord._history)} moves,'
          f' {s_res.quality.value - s_ord.quality.value} quality enhancement)'
          f'\nTotal objective enhancement {init_quality-s_ord.quality.value}')

    history = lambda opt: [s.quality.value for s in opt._history]
    h_ord: list[float] = history(optimiser_ord)
    h_res: list[float] = history(optimiser_res)

    dpi = 300
    fig, ax = plt.subplots(figsize=(1024 / dpi, 768 / dpi), dpi=dpi)
    ax.plot([0], [h_res[0]], 'ro')
    ax.plot(np.arange(len(h_res)), h_res, 'b-', linewidth=2)
    ax.plot(np.arange(len(h_res) - 1, len(h_res) + len(h_ord) - 1), h_ord, 'g-', linewidth=2)
    fig.suptitle(f'WG - {wg_name}; Selection - {move_selection}')
    fig.savefig(f'{f_name}.png')


def test_both_optimisation(setup_schedule_genetic, setup_toolbox, setup_base_optimisers, setup_wg):
    init_schedule: ChromosomeType = setup_toolbox.schedule_to_chromosome(schedule=setup_schedule_genetic)
    init_quality = setup_toolbox.evaluate_time_res(init_schedule)

    wg_name, *_ = setup_wg
    move_selection, optimiser_ord_, optimiser_res_ = setup_base_optimisers

    optimiser_ord = init_order_optimiser(deepcopy(optimiser_ord_), setup_toolbox)
    optimiser_res = init_resource_optimiser(deepcopy(optimiser_res_), setup_toolbox)

    optimiser_ord2 = init_order_optimiser(deepcopy(optimiser_ord_), setup_toolbox)
    optimiser_res2 = init_resource_optimiser(deepcopy(optimiser_res_), setup_toolbox)

    t1 = time()
    s_res = optimiser_res.optimize(init_schedule)
    t2 = time()
    s_ord = optimiser_ord.optimize(s_res.position)

    t3 = time()

    s_ord2 = optimiser_ord2.optimize(init_schedule)
    t4 = time()
    s_res2 = optimiser_res2.optimize(s_ord2.position)
    t5 = time()

    history2 = lambda opt: [(s.quality.value, str(s.id)) for s in opt]
    data = {'works': init_schedule[0].size,
            'wg_name': wg_name,
            'move_selection': move_selection,
            'res-ord': {'ord': [history2(optimiser_ord._history), history2(list(optimiser_ord.hall_of_fame))],
                        'res': [history2(optimiser_res._history), history2(list(optimiser_res.hall_of_fame))],
                        't': [t1, t2, t3]},
            'ord-res': {'ord': [history2(optimiser_ord2._history), history2(list(optimiser_ord2.hall_of_fame))],
                        'res': [history2(optimiser_res2._history), history2(list(optimiser_res2.hall_of_fame))],
                        't': [t3, t4, t5]}}

    f_name = f'scheduling/results/both-{wg_name}'
    with open(f'{f_name}.pickle', 'wb') as pkl:
        pickle.dump(data, pkl)

    print(f'Total res-ord objective enhancement {init_quality-s_ord.quality.value}'
          f'\nTotal ord-res objective enhancement {init_quality-s_res2.quality.value}')

    history = lambda opt: [s.quality.value for s in opt._history]
    h_ord: list[float] = history(optimiser_ord)
    h_res: list[float] = history(optimiser_res)

    h_ord2: list[float] = history(optimiser_ord2)
    h_res2: list[float] = history(optimiser_res2)

    dpi = 150
    fig, ax = plt.subplots(figsize=(1024 / dpi, 768 / dpi), dpi=dpi)
    ax.plot(np.arange(len(h_res)), h_res, 'b-', linewidth=2, label='Resource-first')
    ax.plot(np.arange(len(h_ord2)), h_ord2, 'g--', linewidth=2, label='Order-first')
    ax.plot(np.arange(len(h_res) - 1, len(h_res) + len(h_ord) - 1), h_ord, 'g-', linewidth=2, label='Order-second')
    ax.plot(np.arange(len(h_ord2) - 1, len(h_ord2) + len(h_res2) - 1), h_res2, 'b--', linewidth=2,
            label='Resource-second')
    ax.plot([0], [h_res[0]], 'ro', label='Genetic output')
    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Schedule cost')
    fig.suptitle(f'{wg_name.capitalize()} work graph')
    fig.savefig(f'{f_name}.png')
    # pass


def init_resource_optimiser(opt_preset, setup_toolbox):
    return opt_preset(mutation_behaviour=[
            create_custom_mutation('VP_res', partial(variable_partitioning_resource_neighbourhood,
                                                     worker_reqs=setup_toolbox.get_worker_reqs())),
            same_mutation],
        metric=custom_metric('time-res', setup_toolbox.evaluate_time_res, minimized=True))


def init_order_optimiser(opt_preset, setup_toolbox):
    return opt_preset(mutation_behaviour=[
            create_custom_mutation('VP_ord',
                                   partial(variable_partitioning_order_neighbourhood,
                                           # max_distance=8,
                                           # one_distance_trials=2,
                                           # max_one_distance_offset=4,
                                           is_order_correct=setup_toolbox.is_order_correct)),
            create_custom_mutation('VP_shfl', partial(variable_partitioning_order_shuffle,
                                                      # max_distance=8,
                                                      # one_distance_trials=2,
                                                      is_order_correct=setup_toolbox.is_order_correct)),
            create_custom_mutation('shfl', partial(order_shuffle,
                                                   # attempts=9,
                                                   is_order_correct=setup_toolbox.is_order_correct)),
            same_mutation],
        metric=custom_metric('time-res', setup_toolbox.evaluate_time_res, minimized=True))


def test_scheduler_comparison(setup_schedule_genetic, setup_schedule_heft,
                                setup_toolbox, setup_base_optimisers, setup_wg):
    t0 = time()

    topological_schedule: ChromosomeType = setup_toolbox.schedule_to_chromosome(schedule=setup_schedule_genetic)
    topological_quality = setup_toolbox.evaluate_time_res(topological_schedule)
    heft_schedule: ChromosomeType = setup_toolbox.schedule_to_chromosome(schedule=setup_schedule_genetic)
    heft_quality = setup_toolbox.evaluate_time_res(heft_schedule)
    genetic_schedule: ChromosomeType = setup_toolbox.schedule_to_chromosome(schedule=setup_schedule_genetic)
    genetic_quality = setup_toolbox.evaluate_time_res(genetic_schedule)

    wg_name, *_ = setup_wg
    move_selection, optimiser_ord, optimiser_res = setup_base_optimisers

    optimiser_ord = init_order_optimiser(optimiser_ord, setup_toolbox)
    optimiser_res = init_resource_optimiser(optimiser_res, setup_toolbox)

    t1 = time()
    s_res = optimiser_res.optimize(genetic_schedule)
    t2 = time()
    s_ord = optimiser_ord.optimize(s_res.position)
    t3 = time()

    data = {'works': genetic_schedule[0].size,
            'wg_name': wg_name,
            'move_selection': move_selection,
            'topo': topological_quality,
            'heft': heft_quality,
            'genetic': genetic_quality,
            'tabu': s_ord.quality.value,
            't': [t0, t1, t2, t3]}

    f_name = f'scheduling/results/cmp-{wg_name}'
    with open(f'{f_name}.pickle', 'wb') as pkl:
        pickle.dump(data, pkl)

    print(f'Initialization took {t1 - t0:.3f} sec'
          f'\nResource optimisation took {t2 - t1:.3f} sec ({len(optimiser_res._history)} moves,'
          f' {genetic_quality - s_res.quality.value} quality enhancement)'
          f'\nOrder optimisation took {t3 - t2:.3f} sec ({len(optimiser_ord._history)} moves,'
          f' {s_res.quality.value - s_ord.quality.value} quality enhancement)'
          f'\nTotal objective enhancement {genetic_quality-s_ord.quality.value}')

    history = lambda opt: [s.quality.value for s in opt._history]
    h_ord: list[float] = history(optimiser_ord)
    h_res: list[float] = history(optimiser_res)

    dpi = 150
    fig, ax = plt.subplots(figsize=(1024 / dpi, 768 / dpi), dpi=dpi)
    ax.plot([-2], [topological_quality], 'co')
    ax.plot([-1], [heft_quality], 'mo')
    ax.plot([0], [h_res[0]], 'ro')
    ax.plot(np.arange(len(h_res)), h_res, 'b-', linewidth=2)
    ax.plot(np.arange(len(h_res) - 1, len(h_res) + len(h_ord) - 1), h_ord, 'g-', linewidth=2)
    fig.suptitle(f'WG - {wg_name};')
    fig.savefig(f'{f_name}.png')


def test_repeated_res_order_optimisation(setup_schedule_genetic, setup_toolbox, setup_base_optimisers, setup_wg):
    t0 = time()
    init_schedule: ChromosomeType = setup_toolbox.schedule_to_chromosome(schedule=setup_schedule_genetic)
    init_quality = setup_toolbox.evaluate_time_res(init_schedule)

    wg_name, *_ = setup_wg
    move_selection, optimiser_ord_base, optimiser_res_base = setup_base_optimisers

    for attempt in range(10):
        optimiser_ord = init_order_optimiser(deepcopy(optimiser_ord_base), setup_toolbox)
        optimiser_res = init_resource_optimiser(deepcopy(optimiser_res_base), setup_toolbox)

        t1 = time()
        s_res = optimiser_res.optimize(init_schedule)
        t2 = time()
        s_ord = optimiser_ord.optimize(s_res.position)
        t3 = time()

        history2 = lambda opt: [(s and s.quality.value, str(s and s.id)) for s in opt]
        data = {'works': init_schedule[0].size,
                'wg_name': wg_name,
                'move_selection': move_selection,
                'ord': [history2(optimiser_ord._history), history2(list(optimiser_ord.hall_of_fame))],
                'res': [history2(optimiser_res._history), history2(list(optimiser_res.hall_of_fame))],
                't': [t0, t1, t2, t3],
                'attempt': attempt}
        f_name = f'scheduling/results/r_o-{wg_name}_{attempt}'
        with open(f'{f_name}.pickle', 'wb') as pkl:
            pickle.dump(data, pkl)

        print(f'ATTEMPT {attempt}:'
              f'\nInitialization took {t1 - t0:.3f} sec'
              f'\nResource optimisation took {t2 - t1:.3f} sec ({len(optimiser_res._history)} moves,'
              f' {init_quality - s_res.quality.value} quality enhancement)'
              f'\nOrder optimisation took {t3 - t2:.3f} sec ({len(optimiser_ord._history)} moves,'
              f' {s_res.quality.value - s_ord.quality.value} quality enhancement)'
              f'\nTotal objective enhancement {init_quality-s_ord.quality.value}')

        history = lambda opt: [s.quality.value for s in opt._history if s]
        h_ord: list[float] = history(optimiser_ord)
        h_res: list[float] = history(optimiser_res)

        dpi = 300
        fig, ax = plt.subplots(figsize=(1024 / dpi, 768 / dpi), dpi=dpi)
        ax.plot([0], [h_res[0]], 'ro')
        ax.plot(np.arange(len(h_res)), h_res, 'b-', linewidth=2)
        ax.plot(np.arange(len(h_res) - 1, len(h_res) + len(h_ord) - 1), h_ord, 'g-', linewidth=2)
        fig.suptitle(f'WG - {wg_name}; Selection - {move_selection}')
        fig.savefig(f'{f_name}.png')
