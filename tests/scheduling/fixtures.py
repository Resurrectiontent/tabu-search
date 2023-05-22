import os
from collections import defaultdict
from functools import partial

# from sampo.generator import ContractorGenerationMethod
from scipy.stats import expon, norm
from typing import Callable

import numpy as np
import pytest
from deap.base import Toolbox

from sampo.scheduler.topological.base import TopologicalScheduler, RandomizedTopologicalScheduler
from tabusearch.convergence import DisjunctiveConvergence, IterativeConvergence, EnhancementConvergence
from sampo import generator
from sampo.scheduler.base import Scheduler
from sampo.scheduler.genetic.base import GeneticScheduler
from sampo.scheduler.genetic.operators import init_toolbox, TimeAndResourcesFitness, is_chromosome_order_correct, \
    TimeFitness
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.schemas.contractor import Contractor, WorkerContractorPool
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.utilities.collections import reverse_dictionary
from tabusearch import TabuSearch

storage = 'scheduling/resources/'

if os.path.exists('/tabu-trialing'):
    storage = os.path.join('/tabu-trialing/tabu-search/tests/', storage)


@pytest.fixture(scope='session', params=[#('static', None, lambda x: lambda _: x),
                                         ('statistic',
                                          lambda collection_len: min(expon.rvs(size=1).astype(int)[0],
                                                                     collection_len - 1),
                                          lambda x: lambda _: norm.rvs(x, 5, size=1).astype(int)[0])
                                         ])
def setup_base_optimisers(request, setup_wg):
    name, selection, tabu = request.param

    match setup_wg[0]:
        case 'medium':
            tabu_time, conv_ord, conv_res = 10, 15, 20
        case 'large':
            tabu_time, conv_ord, conv_res = 15, 20, 25
        case 'small':
            tabu_time, conv_ord, conv_res = 3, 10, 15
        case _:
            mul = int(setup_wg[0].split('-')[-1])
            tabu_time, conv_ord, conv_res = mul // 20, 5, 5

    optimiser_ord = partial(TabuSearch,
                            convergence_criterion=DisjunctiveConvergence(IterativeConvergence(15 * conv_ord),
                                                                         EnhancementConvergence(1, conv_ord)),
                            tabu_time=tabu(tabu_time),
                            selection=selection)
    optimiser_res = partial(TabuSearch,
                            convergence_criterion=DisjunctiveConvergence(IterativeConvergence(8 * conv_res),
                                                                         EnhancementConvergence(1, conv_res)),
                            tabu_time=tabu(tabu_time),
                            selection=selection)

    return name, optimiser_ord, optimiser_res


@pytest.fixture(scope='session', params=[#('wg-50', 50, 25, 15),
                                         # ('wg-100', 100, 35, 17),
                                         # ('wg-150', 150, 40, 20),
                                         # ('wg-225', 225, 50, 30),
                                         # ('wg-300', 300, 60, 40),
                                         # ('wg-400', 400, 70, 50)
                                         #('small', 80, 30, 17),
                                         #('medium', 150, 40, 20),
                                         #('large', 250, 60, 40),
                                         #('real-0', 'wg_0'),
                                         #('real-1', 'wg_1'),
                                         ('real-1002', 'wg_1002'),
                                         # ('real-2037', 'wg_2037'),
                                         # ('real-5085', 'wg_5085'),
                                         # ('real-10081', 'wg_10081'),
                                        ])
def setup_wg(request):
    name, *params = request.param
    if 'real' in name:
        wg = WorkGraph.load(os.path.join(storage, params[0]), 'work_graph')
    else:
        wg = generator.SimpleSynthetic().advanced_work_graph(*params)
    return name, wg


@pytest.fixture(scope='session')
def setup_contractors(setup_wg):
    if 'real' in setup_wg[0]:
        contractors = [Contractor.load(os.path.join(storage, f'wg_{setup_wg[0].split("-")[-1]}'), f'contractor_{i}')
                       for i in range(2)]
    else:
        contractors = [generator.get_contractor_by_wg(setup_wg[1])]
    return contractors


@pytest.fixture(scope='session')
def setup_worker_pool(setup_contractors) -> WorkerContractorPool:
    worker_pool = defaultdict(dict)
    for contractor in setup_contractors:
        for worker in contractor.workers.values():
            worker_pool[worker.name][worker.contractor_id] = worker
    return worker_pool


@pytest.fixture(scope='session', params=[RandomizedTopologicalScheduler])
def setup_schedule_topo(request, setup_wg, setup_contractors) -> Schedule:
    scheduler: Callable[[], Scheduler] = request.param

    schedule = scheduler().schedule(setup_wg[1], setup_contractors)
    return schedule


@pytest.fixture(scope='session', params=[HEFTScheduler])
def setup_schedule_heft(request, setup_wg, setup_contractors) -> Schedule:
    scheduler: Callable[[], Scheduler] = request.param

    schedule = scheduler().schedule(setup_wg[1], setup_contractors)
    return schedule


@pytest.fixture(scope='session', params=[GeneticScheduler])
def setup_schedule_genetic(request, setup_wg, setup_contractors) -> Schedule:
    scheduler: Callable[[], Scheduler] = request.param
    schedule = scheduler().schedule(setup_wg[1], setup_contractors)
    return schedule


@pytest.fixture(scope='session', params=[HEFTScheduler, GeneticScheduler])
def setup_schedule(request, setup_wg, setup_contractors) -> Schedule:
    scheduler: Callable[[], Scheduler] = request.param

    schedule = scheduler().schedule(setup_wg[1], setup_contractors)
    return schedule


@pytest.fixture(scope='session')
def setup_toolbox(setup_wg, setup_contractors, setup_worker_pool) -> Toolbox:
    """
    returns a Toolbox that supports following methods:
      evaluate
      validate
      schedule_to_chromosome
      chromosome_to_schedule
    """
    return create_toolbox(setup_wg[1],
                          setup_contractors,
                          setup_worker_pool)


# TODO: delete unused data structures
# TODO: don't use sampo init_toolbox function
def create_toolbox(wg: WorkGraph,
                   contractors: list[Contractor],
                   worker_pool: WorkerContractorPool,
                   spec: ScheduleSpec = ScheduleSpec(),
                   work_estimator: WorkTimeEstimator = None) -> Toolbox:
    nodes = [node for node in wg.nodes if not node.is_inseparable_son()]

    index2node: dict[int, GraphNode] = {index: node for index, node in enumerate(nodes)}
    work_id2index: dict[str, int] = {node.id: index for index, node in index2node.items()}
    worker_name2index = {worker_name: index for index, worker_name in enumerate(worker_pool)}
    index2contractor = {ind: contractor.id for ind, contractor in enumerate(contractors)}
    index2contractor_obj = {ind: contractor for ind, contractor in enumerate(contractors)}
    contractor2index = reverse_dictionary(index2contractor)
    worker_pool_indices = {worker_name2index[worker_name]: {
        contractor2index[contractor_id]: worker for contractor_id, worker in workers_of_type.items()
    } for worker_name, workers_of_type in worker_pool.items()}
    node_indices = list(range(len(nodes)))

    contractors_capacity = np.zeros((len(contractors), len(worker_pool)))
    for w_ind, cont2worker in worker_pool_indices.items():
        for c_ind, worker in cont2worker.items():
            contractors_capacity[c_ind][w_ind] = worker.count

    resources_border = np.zeros((2, len(worker_pool), len(index2node)))
    resources_min_border = np.zeros((len(worker_pool)))
    for work_index, node in index2node.items():
        for req in node.work_unit.worker_reqs:
            worker_index = worker_name2index[req.kind]
            resources_border[0, worker_index, work_index] = req.min_count
            resources_border[1, worker_index, work_index] = req.max_count
            resources_min_border[worker_index] = max(resources_min_border[worker_index], req.min_count)

    contractor_borders = np.zeros((len(contractor2index), len(worker_name2index)), dtype=int)
    for ind, contractor in enumerate(contractors):
        for ind_worker, worker in enumerate(contractor.workers.values()):
            contractor_borders[ind, ind_worker] = worker.count

    # construct inseparable_child -> inseparable_parent mapping
    inseparable_parents = {}
    for node in nodes:
        for child in node.get_inseparable_chain_with_self():
            inseparable_parents[child] = node

    # here we aggregate information about relationships from the whole inseparable chain
    children = {work_id2index[node.id]: [work_id2index[inseparable_parents[child].id]
                                         for inseparable in node.get_inseparable_chain_with_self()
                                         for child in inseparable.children]
                for node in nodes}

    parents = {work_id2index[node.id]: [] for node in nodes}
    for node, node_children in children.items():
        for child in node_children:
            parents[child].append(node)

    toolbox = init_toolbox(wg=wg,
                           work_id2index=work_id2index,
                           worker_name2index=worker_name2index,
                           contractor2index=contractor2index,
                           contractor_borders=contractor_borders,
                           fitness_constructor=TimeAndResourcesFitness,
                           node_indices=node_indices,
                           parents=parents,
                           worker_pool=worker_pool,
                           index2node=index2node,
                           index2contractor_obj=index2contractor_obj,
                           worker_pool_indices=worker_pool_indices,
                           spec=spec,
                           work_estimator=work_estimator,
                           # don't fill parameters that won't be used during tabu search
                           contractors=None,
                           index2contractor=[],
                           index2node_list=None,
                           init_chromosomes=None,
                           mutate_order=None,
                           mutate_resources=None,
                           rand=None,
                           selection_size=None)

    toolbox.register('get_worker_reqs', lambda: resources_border)
    toolbox.register('is_order_correct', is_chromosome_order_correct, parents=parents)
    toolbox.register('evaluate_time_res', lambda chromosome, fitness: float(fitness.evaluate(chromosome).value),
                     fitness=TimeAndResourcesFitness(toolbox))
    toolbox.register('evaluate_time', lambda chromosome, fitness: float(fitness.evaluate(chromosome).value),
                     fitness=TimeFitness(toolbox))
    return toolbox
