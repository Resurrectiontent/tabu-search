from datetime import datetime
import os
from copy import deepcopy

from operator import attrgetter

import pickle
from uuid import uuid4
from functools import partial
from time import time

import numpy as np
import pytest
from matplotlib import pyplot as plt
from scipy.stats import norm, expon

from convergence import DisjunctiveConvergence, IterativeConvergence, EnhancementConvergence
from memory.evaluating.gradient import GradientAccelerator
from memory.filtering.aspiration import AspirationBoundType
from sampo.scheduler.genetic.base import GeneticScheduler
from sampo.scheduler.genetic.converter import ChromosomeType
from sampo.scheduler.genetic.operators import copy_chromosome
from sampo.scheduler.genetic.schedule_builder import build_schedule
from sampo.schemas.contractor import get_worker_contractor_pool
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time
from scheduling.order_neighbourhood import order_shuffle, variable_partitioning_order_neighbourhood, \
    variable_partitioning_order_shuffle
from scheduling.resource_neighbourhood import variable_partitioning_resource_neighbourhood
from solution.quality.lib import custom_metric
from solution.quality.lib.single import custom_metric_parallel
from tabusearch.mutation import create_custom_mutation, same_mutation
from tabusearch import TabuSearch
from tabusearch.solution.base import Solution
from tests.scheduling.fixtures import setup_schedule_topo, setup_schedule_heft, setup_schedule_genetic, setup_schedule, \
    setup_toolbox, setup_wg, setup_contractors, setup_worker_pool, setup_base_optimisers

if os.path.exists('/tabu-trialing'):
    storage = '/results-out/'
else:
    storage = 'scheduling/results/'

logs = 'logs.txt'


def log(s):
    with open(os.path.join(storage, logs), 'a') as write_log:
        write_log.write(f'[{datetime.now().strftime("%H:%M:%S")}]: {s}\n')


def schedule_genetic(schedules, wg, contractors):
    dummy_scheduler = GeneticScheduler()
    size_selection, mutate_order, mutate_resources, size_of_population = dummy_scheduler.get_params(wg.vertex_count)
    agents = get_worker_contractor_pool(contractors)

    init_schedules: dict[str, tuple[Schedule, None]] = {i: (schedule, None) for i, schedule in
                                                        zip(('heft_between', 'heft_end'), schedules)}

    scheduled_works, schedule_start_time, timeline, order_nodes = build_schedule(wg,
                                                                                 contractors,
                                                                                 agents,
                                                                                 size_of_population,
                                                                                 dummy_scheduler.number_of_generation,
                                                                                 size_selection,
                                                                                 mutate_order,
                                                                                 mutate_resources,
                                                                                 init_schedules,
                                                                                 dummy_scheduler.rand,
                                                                                 ScheduleSpec(),
                                                                                 dummy_scheduler.fitness_constructor,
                                                                                 dummy_scheduler.work_estimator,
                                                                                 assigned_parent_time=Time(0))
    return Schedule.from_scheduled_works(scheduled_works.values(), wg)


def get_optimiser(setup_toolbox, stochastic_tabu=False, stochastic_selection=False, simple_id=False, use_ga=False,
                  aspiration_ge=False, use_vp=False, short_optimisers=False, **kwargs):
    if stochastic_tabu:
        tabu = lambda _: norm.rvs(10, 5, size=1).astype(int)[0]
    else:
        tabu = 2 if simple_id else 10

    if stochastic_selection:
        selection = lambda collection_len: min(expon.rvs(size=1).astype(int)[0], collection_len - 1)
    else:
        selection = None

    if use_ga:
        ga_res = [GradientAccelerator(data_converter=lambda x: [ch[1] for ch in x])]
        ga_ord = [GradientAccelerator(data_converter=lambda x: [ch[0] for ch in x])]
        ga_weight = (0.35,)
    else:
        ga_res, ga_ord = None, None
        ga_weight = None

    if aspiration_ge:
        aspiration_bound_type = AspirationBoundType.GreaterEquals
    else:
        aspiration_bound_type = AspirationBoundType.Greater

    metric = custom_metric_parallel('time-res', setup_toolbox.evaluate_time_res, minimized=True)

    mutation_behaviour_ord = [
        create_custom_mutation('VP_ord',
                               partial(variable_partitioning_order_neighbourhood,
                                       is_order_correct=setup_toolbox.is_order_correct)),
        create_custom_mutation('VP_shfl', partial(variable_partitioning_order_shuffle,
                                                  is_order_correct=setup_toolbox.is_order_correct)),
        create_custom_mutation('shfl', partial(order_shuffle,
                                               attempts=15,
                                               is_order_correct=setup_toolbox.is_order_correct)),
        same_mutation] if use_vp else [
        create_custom_mutation('shfl', partial(order_shuffle,
                                               attempts=50,
                                               is_order_correct=setup_toolbox.is_order_correct)),
        same_mutation]
    mutation_behaviour_res = [
        create_custom_mutation('VP_res', partial(variable_partitioning_resource_neighbourhood,
                                                 worker_reqs=setup_toolbox.get_worker_reqs())),
        same_mutation]

    convergence_ord = DisjunctiveConvergence(IterativeConvergence(15),
                                             EnhancementConvergence(1, 8)) \
        if short_optimisers \
        else DisjunctiveConvergence(IterativeConvergence(45),
                                    EnhancementConvergence(1, 10 if use_vp else 25))

    convergence_res = DisjunctiveConvergence(IterativeConvergence(10),
                                             EnhancementConvergence(1, 5)) \
        if short_optimisers \
        else DisjunctiveConvergence(IterativeConvergence(30),
                                    EnhancementConvergence(1, 15))

    optimiser_ord = TabuSearch(mutation_behaviour=mutation_behaviour_ord,
                               metric=metric,
                               convergence_criterion=convergence_ord,
                               tabu_time=tabu,
                               aspiration_bound_type=aspiration_bound_type,
                               selection=selection,
                               additional_evaluation=ga_ord,
                               additional_evaluation_weights=ga_weight,
                               use_simple_solution_ids=simple_id)

    optimiser_res = TabuSearch(mutation_behaviour=mutation_behaviour_res,
                               metric=metric,
                               convergence_criterion=convergence_res,
                               tabu_time=tabu,
                               aspiration_bound_type=aspiration_bound_type,
                               selection=selection,
                               additional_evaluation=ga_res,
                               additional_evaluation_weights=ga_weight,
                               use_simple_solution_ids=simple_id)

    return optimiser_ord, optimiser_res


def test_repeated_modification_optimisation(setup_schedule_genetic, setup_toolbox, setup_base_optimisers, setup_wg):
    t0 = time()
    init_schedule: ChromosomeType = setup_toolbox.schedule_to_chromosome(schedule=setup_schedule_genetic)
    init_quality = setup_toolbox.evaluate_time_res(init_schedule)

    wg_name, *_ = setup_wg
    log(f'test_repeated_modification_optimisation("{wg_name}")')
    move_selection, optimiser_ord_base, optimiser_res_base = setup_base_optimisers

    modifications = ['no', 'stochastic_tabu', 'stochastic_selection', 'simple_id', 'use_ga', 'aspiration_ge']
    # modifications = ['no', 'use_ga', 'aspiration_ge']
    # modifications = ['no', 'use_vp']

    optimisers = [(mod, get_optimiser(setup_toolbox, use_vp=True, **{mod: True})) for mod in modifications]

    for modification, (optimiser_ord_, optimiser_res_) in optimisers:
        log(modification)
        for attempt in range(4):
            log(attempt)
            optimiser_ord = deepcopy(optimiser_ord_)
            optimiser_res = deepcopy(optimiser_res_)

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
                    'modification': modification,
                    'attempt': attempt}
            f_name = os.path.join(storage, f'r_o-{wg_name}-{modification}_{attempt}')
            with open(f'{f_name}.pickle', 'wb') as pkl:
                pickle.dump(data, pkl)

            print(f'ATTEMPT {attempt}:'
                  f'\nInitialization took {t1 - t0:.3f} sec'
                  f'\nResource optimisation took {t2 - t1:.3f} sec ({len(optimiser_res._history)} moves,'
                  f' {init_quality - s_res.quality.value} quality enhancement)'
                  f'\nOrder optimisation took {t3 - t2:.3f} sec ({len(optimiser_ord._history)} moves,'
                  f' {s_res.quality.value - s_ord.quality.value} quality enhancement)'
                  f'\nTotal objective enhancement {init_quality - s_ord.quality.value}')

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


def test_repeated_combination_optimisation(setup_schedule_genetic, setup_toolbox, setup_wg, setup_contractors):
    history2 = lambda opt: [(s and s.quality.value, str(s and s.id)) for s in opt]

    init_schedule_: ChromosomeType = setup_toolbox.schedule_to_chromosome(schedule=setup_schedule_genetic)
    init_quality_ = setup_toolbox.evaluate_time_res(init_schedule_)

    wg_name, wg = setup_wg
    log(f'test_repeated_combination_optimisation({wg_name})')

    for attempt in range(3):
        log(attempt)
        hist = []
        init_schedule, init_quality = init_schedule_, init_quality_

        for i in range(3):
            optimiser_ord, optimiser_res = get_optimiser(setup_toolbox, short_optimisers=True)

            s_res = optimiser_res.optimize(init_schedule)
            s_ord = optimiser_ord.optimize(s_res.position)

            hist.append(((init_schedule, init_quality), optimiser_res, optimiser_ord))

            if i < 2:
                scheduled_works, _, _, _ = setup_toolbox.chromosome_to_schedule(chromosome=s_ord.position)
                tabu_schedule = Schedule.from_scheduled_works(scheduled_works.values(), wg)

                gen_schedule = schedule_genetic((tabu_schedule, setup_schedule_genetic), wg, setup_contractors)
                init_schedule = setup_toolbox.schedule_to_chromosome(schedule=gen_schedule)
                init_quality = setup_toolbox.evaluate_time_res(init_schedule)

        opt_process = [{'gen': gen_quality,
                        'ord': [history2(opt_ord._history), history2(list(opt_ord.hall_of_fame))],
                        'res': [history2(opt_res._history), history2(list(opt_res.hall_of_fame))]}
                       for (_, gen_quality), opt_res, opt_ord in hist]

        data = {'works': wg.vertex_count,
                'wg_name': wg_name,
                'opt': opt_process,
                'attempt': attempt}
        f_name = os.path.join(storage, f'iter-gen-tabu-{wg_name}_{attempt}')
        with open(f'{f_name}.pickle', 'wb') as pkl:
            pickle.dump(data, pkl)

# def test_best_modification_optimisation(setup_schedule_genetic, setup_toolbox, setup_base_optimisers, setup_wg):
#     t0 = time()
#     init_schedule: ChromosomeType = setup_toolbox.schedule_to_chromosome(schedule=setup_schedule_genetic)
#     init_quality = setup_toolbox.evaluate_time_res(init_schedule)
#
#     wg_name, *_ = setup_wg
#
#     mod_optimiser_ord_base, _ = get_optimiser(setup_toolbox, stochastic_tabu=True, stochastic_selection=True, simple_id=True)
#     _, mod_optimiser_res_base = get_optimiser(setup_toolbox, use_ga=True, aspiration_ge=True)
#     optimiser_ord_base, optimiser_res_base = get_optimiser(setup_toolbox)
#
#     for attempt in range(4):
#         optimiser_ord = deepcopy(optimiser_ord_base)
#         optimiser_res = deepcopy(optimiser_res_base)
#         mod_optimiser_ord = deepcopy(mod_optimiser_ord_base)
#         mod_optimiser_res = deepcopy(mod_optimiser_res_base)
#
#         s_res = optimiser_res.optimize(init_schedule)
#         s_ord = optimiser_ord.optimize(s_res.position)
#
#         s_mod_res = mod_optimiser_res.optimize(init_schedule)
#         s_mod_ord = mod_optimiser_ord.optimize(s_mod_res.position)
#
#         history2 = lambda opt: [(s and s.quality.value, str(s and s.id)) for s in opt]
#         data = {'works': init_schedule[0].size,
#                 'wg_name': wg_name,
#                 'ord': [history2(optimiser_ord._history), history2(list(optimiser_ord.hall_of_fame))],
#                 'res': [history2(optimiser_res._history), history2(list(optimiser_res.hall_of_fame))],
#                 'modification': False,
#                 'attempt': attempt}
#         f_name = f'scheduling/results/pure-{wg_name}_{attempt}'
#         with open(f'{f_name}.pickle', 'wb') as pkl:
#             pickle.dump(data, pkl)
#
#         data_mod = {'works': init_schedule[0].size,
#                     'wg_name': wg_name,
#                     'ord': [history2(mod_optimiser_ord._history), history2(list(mod_optimiser_ord.hall_of_fame))],
#                     'res': [history2(mod_optimiser_res._history), history2(list(mod_optimiser_res.hall_of_fame))],
#                     'modification': True,
#                     'attempt': attempt}
#         f_name = f'scheduling/results/mod-{wg_name}_{attempt}'
#         with open(f'{f_name}.pickle', 'wb') as pkl:
#             pickle.dump(data_mod, pkl)
